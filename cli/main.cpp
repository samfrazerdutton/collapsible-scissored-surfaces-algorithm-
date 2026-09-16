// scissorc — command-line front end for the Collapsible Scissored Surfaces
// compression algorithm.
//
//   scissorc squeeze   <in> [out] [--quality 1-9] [--explain] [--force]
//   scissorc unsqueeze <in> [out] [--explain] [--force]
//
// squeeze/unsqueeze are the front door: point squeeze at any file and it
// sniffs whether it's a numeric 2D/3D point table, a 7-column 6-DOF pose
// stream, or just arbitrary bytes, picks a --scale that exactly preserves
// the file's own decimal precision (not a guessed default), and verifies
// the round-trip live before printing a number -- no need to already know
// this codec's specialized modes to get a real, honest result. Every
// mode below still exists, unchanged, for scripts/pipelines that already
// know their data's shape and want to name it (and the exact scale)
// themselves rather than have it detected:
//
//   scissorc compress   <in> <out> [--gpu] [--level fast|balanced|high]
//   scissorc decompress <in> <out>
//   scissorc compress-geo2d   <in.xy>  <out> [--scale N]
//   scissorc decompress-geo2d <in> <out.xy>
//   scissorc compress-geo3d   <in.xyz> <out> [--scale N]
//   scissorc decompress-geo3d <in> <out.xyz>
//   scissorc compress-pose    <in.pose> <out> [--scale N] [--qscale N]
//   scissorc decompress-pose  <in> <out.pose>
//   scissorc info <file>
//
// Geo-mode point files are plain text, one point per line: "x y" (geo2d)
// or "x y z" (geo3d), floating point. They are quantized to fixed-point
// integers (value * scale, rounded) so the transform stays exact-integer
// lossless; --scale controls the quantization step (default 1000, i.e.
// 3 decimal digits of precision).
//
// Pose-mode files are 6-DOF samples, one per line: "x y z qw qx qy qz"
// (position, then a unit quaternion) -- a VR/AR headset or controller
// track, a drone/robot odometry log, a SLAM camera path. Position uses
// --scale like geo2d/geo3d; orientation gets its own --qscale (default
// 1000000 -- quaternion components live in [-1, 1], so --scale's default
// would leave barely 3 meaningful digits). See quaternion_joint.hpp for
// the model.
#include "csa/codec.hpp"
#include "csa/lz_matcher.hpp"
#include "csa/pantograph_lift_cuda.hpp"
#include "csa/rans_coder.hpp"
#include "csa/simd.hpp"
#include "csa/system_info.hpp"
#include "miniz.h" // vendored (see CMakeLists.txt) -- real gzip-equivalent baseline for `scissorc benchmark`
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>
#include <thread>

using namespace csa;

namespace {

std::vector<u8> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open input file: " + path);
    f.seekg(0, std::ios::end);
    std::streamoff size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<u8> data((size_t)size);
    if (size > 0) f.read((char*)data.data(), size);
    return data;
}

void write_file(const std::string& path, const std::vector<u8>& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open output file: " + path);
    if (!data.empty()) f.write((const char*)data.data(), (std::streamsize)data.size());
}

constexpr char kGeoMagic[4] = {'C', 'S', 'A', 'G'};

std::vector<Point2i> read_points2d(const std::string& path, i64 scale) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open input file: " + path);
    std::vector<Point2i> pts;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        double x, y;
        if (!(ss >> x >> y)) continue;
        pts.push_back({(i32)llround(x * (double)scale), (i32)llround(y * (double)scale)});
    }
    return pts;
}

std::vector<Point3i> read_points3d(const std::string& path, i64 scale) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open input file: " + path);
    std::vector<Point3i> pts;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        double x, y, z;
        if (!(ss >> x >> y >> z)) continue;
        pts.push_back({(i32)llround(x * (double)scale), (i32)llround(y * (double)scale), (i32)llround(z * (double)scale)});
    }
    return pts;
}

void write_geo_header(std::vector<u8>& out, u8 dims, i64 scale) {
    for (char c : kGeoMagic) out.push_back((u8)c);
    out.push_back(dims);
    put_u64(out, (u64)scale);
}

// The same speed/ratio "level" every production LZ compressor exposes
// (gzip -1..-9, zstd -1..-22), applied to this codec's LZ match search.
// See lz_matcher.hpp / REAL_CORPUS_BENCHMARK.md for what these trade off.
void level_to_lz_params(const std::string& level, int& max_chain, size_t& nice_length) {
    if (level == "fast") { max_chain = 32; nice_length = 32; }
    else if (level == "high") { max_chain = 1024; nice_length = 4096; }
    else { max_chain = kLzDefaultMaxChain; nice_length = kLzDefaultNiceLength; } // "balanced" / default
}

int cmd_compress(const std::string& in, const std::string& out, bool gpu, const std::string& level) {
    auto data = read_file(in);
    int max_chain;
    size_t nice_length;
    level_to_lz_params(level, max_chain, nice_length);
    auto blob = compress(data, gpu, max_chain, nice_length);
    write_file(out, blob);
    std::cout << "compress: " << data.size() << " -> " << blob.size() << " bytes ("
              << (data.empty() ? 0.0 : 100.0 * (1.0 - (double)blob.size() / (double)data.size()))
              << "% smaller)" << (gpu ? " [gpu]" : "") << " [level=" << level << "]\n";
    return 0;
}

int cmd_decompress(const std::string& in, const std::string& out) {
    auto blob = read_file(in);
    auto data = decompress(blob);
    write_file(out, data);
    std::cout << "decompress: " << blob.size() << " -> " << data.size() << " bytes\n";
    return 0;
}

int cmd_compress_geo2d(const std::string& in, const std::string& out, i64 scale) {
    auto pts = read_points2d(in, scale);
    auto blob = compress_geo2d(pts);
    std::vector<u8> file;
    write_geo_header(file, 2, scale);
    file.insert(file.end(), blob.begin(), blob.end());
    write_file(out, file);
    size_t raw_estimate = pts.size() * 2 * sizeof(double);
    std::cout << "compress-geo2d: " << pts.size() << " points, raw~=" << raw_estimate
              << " bytes -> " << file.size() << " bytes\n";
    return 0;
}

int cmd_compress_geo2d_lossy(const std::string& in, const std::string& out, i64 scale,
                              u32 quant_step, u32 resync_interval) {
    auto pts = read_points2d(in, scale);
    auto blob = compress_geo2d_lossy(pts, quant_step, resync_interval);
    std::vector<u8> file;
    write_geo_header(file, 2, scale);
    file.insert(file.end(), blob.begin(), blob.end());
    write_file(out, file);

    // Report the real, measured max error (not just the theoretical
    // bound) by actually decoding and comparing -- honesty over
    // convenience, consistent with the rest of this project.
    auto decoded = decompress_geo2d(blob);
    // Same flat-array SIMD max-abs-diff kernel as cmd_optimize/cmd_benchmark
    // (csa::max_abs_diff_i32, include/csa/simd.hpp) -- valid because
    // Point2i is exactly two contiguous i32 members with no padding
    // (include/csa/rod_joint_transform.hpp), so an array of them is
    // layout-compatible with a flat i32 array of twice the length.
    size_t n2 = std::min(pts.size(), decoded.size());
    double max_err = n2 ? (double)max_abs_diff_i32(reinterpret_cast<const i32*>(pts.data()),
                                                    reinterpret_cast<const i32*>(decoded.data()), n2 * 2) / (double)scale : 0.0;

    size_t raw_estimate = pts.size() * 2 * sizeof(double);
    std::cout << "compress-geo2d-lossy: " << pts.size() << " points, raw~=" << raw_estimate
              << " bytes -> " << file.size() << " bytes, quant_step=" << quant_step
              << " resync_interval=" << resync_interval
              << ", measured max coordinate error=" << max_err << " (in original units)\n";
    return 0;
}

int cmd_decompress_geo2d(const std::string& in, const std::string& out) {
    auto file = read_file(in);
    if (file.size() < 13 || file[0] != (u8)kGeoMagic[0] || file[4] != 2)
        throw std::runtime_error("not a geo2d CSAG file");
    size_t pos = 5;
    i64 scale = (i64)get_u64(file.data(), file.size(), pos);
    std::vector<u8> blob(file.begin() + pos, file.end());
    auto pts = decompress_geo2d(blob);
    std::ofstream f(out);
    f << std::fixed << std::setprecision(6);
    for (auto& p : pts) f << (p.x / (double)scale) << " " << (p.y / (double)scale) << "\n";
    std::cout << "decompress-geo2d: " << pts.size() << " points\n";
    return 0;
}

int cmd_compress_geo3d(const std::string& in, const std::string& out, i64 scale) {
    auto pts = read_points3d(in, scale);
    auto blob = compress_geo3d(pts);
    std::vector<u8> file;
    write_geo_header(file, 3, scale);
    file.insert(file.end(), blob.begin(), blob.end());
    write_file(out, file);
    size_t raw_estimate = pts.size() * 3 * sizeof(double);
    std::cout << "compress-geo3d: " << pts.size() << " points, raw~=" << raw_estimate
              << " bytes -> " << file.size() << " bytes\n";
    return 0;
}

int cmd_compress_geo3d_lossy(const std::string& in, const std::string& out, i64 scale,
                              u32 quant_step, u32 resync_interval) {
    auto pts = read_points3d(in, scale);
    auto blob = compress_geo3d_lossy(pts, quant_step, resync_interval);
    std::vector<u8> file;
    write_geo_header(file, 3, scale);
    file.insert(file.end(), blob.begin(), blob.end());
    write_file(out, file);

    // Same honesty-over-convenience measured-error reporting as
    // cmd_compress_geo2d_lossy.
    auto decoded = decompress_geo3d(blob);
    // See cmd_compress_geo2d_lossy's identical comment -- Point3i is three
    // contiguous i32 members with no padding, so this is the same
    // flat-array SIMD kernel applied to 3-wide points instead of 2-wide.
    size_t n3 = std::min(pts.size(), decoded.size());
    double max_err = n3 ? (double)max_abs_diff_i32(reinterpret_cast<const i32*>(pts.data()),
                                                    reinterpret_cast<const i32*>(decoded.data()), n3 * 3) / (double)scale : 0.0;

    size_t raw_estimate = pts.size() * 3 * sizeof(double);
    std::cout << "compress-geo3d-lossy: " << pts.size() << " points, raw~=" << raw_estimate
              << " bytes -> " << file.size() << " bytes, quant_step=" << quant_step
              << " resync_interval=" << resync_interval
              << ", measured max coordinate error=" << max_err << " (in original units)\n";
    return 0;
}

int cmd_decompress_geo3d(const std::string& in, const std::string& out) {
    auto file = read_file(in);
    if (file.size() < 13 || file[0] != (u8)kGeoMagic[0] || file[4] != 3)
        throw std::runtime_error("not a geo3d CSAG file");
    size_t pos = 5;
    i64 scale = (i64)get_u64(file.data(), file.size(), pos);
    std::vector<u8> blob(file.begin() + pos, file.end());
    auto pts = decompress_geo3d(blob);
    std::ofstream f(out);
    f << std::fixed << std::setprecision(6);
    for (auto& p : pts) f << (p.x / (double)scale) << " " << (p.y / (double)scale) << " " << (p.z / (double)scale) << "\n";
    std::cout << "decompress-geo3d: " << pts.size() << " points\n";
    return 0;
}

// Pose files are plain text, one 6-DOF sample per line: "x y z qw qx qy
// qz" (position, then a unit quaternion). Position is quantized by
// --scale like geo2d/geo3d; orientation gets its own --qscale (default
// much finer -- quaternion components live in [-1, 1], so --scale's
// default of 1000 would leave barely 3 meaningful digits).
std::vector<Pose> read_poses(const std::string& path, i64 scale, i64 qscale) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open input file: " + path);
    std::vector<Pose> poses;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        double x, y, z, qw, qx, qy, qz;
        if (!(ss >> x >> y >> z >> qw >> qx >> qy >> qz)) continue;
        Pose p;
        p.position = {(i32)llround(x * (double)scale), (i32)llround(y * (double)scale), (i32)llround(z * (double)scale)};
        p.orientation = {(i32)llround(qw * (double)qscale), (i32)llround(qx * (double)qscale),
                          (i32)llround(qy * (double)qscale), (i32)llround(qz * (double)qscale)};
        poses.push_back(p);
    }
    return poses;
}

void write_pose_header(std::vector<u8>& out, i64 scale, i64 qscale) {
    for (char c : kGeoMagic) out.push_back((u8)c);
    out.push_back(7); // dims=7 marks a Pose (position+orientation) file
    put_u64(out, (u64)scale);
    put_u64(out, (u64)qscale);
}

int cmd_compress_pose(const std::string& in, const std::string& out, i64 scale, i64 qscale) {
    auto poses = read_poses(in, scale, qscale);
    auto blob = compress_pose(poses);
    std::vector<u8> file;
    write_pose_header(file, scale, qscale);
    file.insert(file.end(), blob.begin(), blob.end());
    write_file(out, file);
    size_t raw_estimate = poses.size() * 7 * sizeof(double);
    std::cout << "compress-pose: " << poses.size() << " poses, raw~=" << raw_estimate
              << " bytes -> " << file.size() << " bytes\n";
    return 0;
}

int cmd_compress_pose_lossy(const std::string& in, const std::string& out, i64 scale, i64 qscale,
                             u32 pos_quant_step, u32 pos_resync_interval, u32 quat_quant_step, u32 quat_resync_interval) {
    auto poses = read_poses(in, scale, qscale);
    auto blob = compress_pose_lossy(poses, pos_quant_step, pos_resync_interval, quat_quant_step, quat_resync_interval);
    std::vector<u8> file;
    write_pose_header(file, scale, qscale);
    file.insert(file.end(), blob.begin(), blob.end());
    write_file(out, file);

    // Same honesty-over-convenience measured-error reporting as
    // cmd_compress_geo2d_lossy/cmd_compress_geo3d_lossy.
    auto decoded = decompress_pose(blob);
    double max_pos_err = 0.0, max_quat_err = 0.0;
    for (size_t i = 0; i < poses.size() && i < decoded.size(); i++) {
        max_pos_err = std::max({max_pos_err,
            std::abs((double)(poses[i].position.x - decoded[i].position.x)) / (double)scale,
            std::abs((double)(poses[i].position.y - decoded[i].position.y)) / (double)scale,
            std::abs((double)(poses[i].position.z - decoded[i].position.z)) / (double)scale});
        max_quat_err = std::max({max_quat_err,
            std::abs((double)(poses[i].orientation.w - decoded[i].orientation.w)) / (double)qscale,
            std::abs((double)(poses[i].orientation.x - decoded[i].orientation.x)) / (double)qscale,
            std::abs((double)(poses[i].orientation.y - decoded[i].orientation.y)) / (double)qscale,
            std::abs((double)(poses[i].orientation.z - decoded[i].orientation.z)) / (double)qscale});
    }

    size_t raw_estimate = poses.size() * 7 * sizeof(double);
    std::cout << "compress-pose-lossy: " << poses.size() << " poses, raw~=" << raw_estimate
              << " bytes -> " << file.size() << " bytes, pos_quant=" << pos_quant_step << " pos_resync=" << pos_resync_interval
              << " quat_quant=" << quat_quant_step << " quat_resync=" << quat_resync_interval
              << ", measured max position error=" << max_pos_err
              << " max orientation-component error=" << max_quat_err << " (in original units)\n";
    return 0;
}

int cmd_decompress_pose(const std::string& in, const std::string& out) {
    auto file = read_file(in);
    if (file.size() < 21 || file[0] != (u8)kGeoMagic[0] || file[4] != 7)
        throw std::runtime_error("not a pose CSAG file");
    size_t pos = 5;
    i64 scale = (i64)get_u64(file.data(), file.size(), pos);
    i64 qscale = (i64)get_u64(file.data(), file.size(), pos);
    std::vector<u8> blob(file.begin() + pos, file.end());
    auto poses = decompress_pose(blob);
    std::ofstream f(out);
    f << std::fixed << std::setprecision(6);
    for (auto& p : poses)
        f << (p.position.x / (double)scale) << " " << (p.position.y / (double)scale) << " " << (p.position.z / (double)scale) << " "
          << (p.orientation.w / (double)qscale) << " " << (p.orientation.x / (double)qscale) << " "
          << (p.orientation.y / (double)qscale) << " " << (p.orientation.z / (double)qscale) << "\n";
    std::cout << "decompress-pose: " << poses.size() << " poses\n";
    return 0;
}

// ============================================================================
// squeeze / unsqueeze: the "point it at a file, it does the right thing"
// front door. Every specialized mode above (compress-geo2d/-geo3d/-pose)
// requires the caller to already know their data's shape and pick a
// --scale that preserves its precision -- fine for someone who already
// knows this codec, a real barrier for anyone else. squeeze sniffs the
// input instead (numeric text table -> column count -> geo2d/geo3d/pose;
// anything else -> general compress()) and picks a --scale that
// preserves exactly the decimal precision actually present in the file,
// not a guessed default -- so "lossless" printed below is a real,
// verified fact about this run, not an assumption carried over from the
// test suite. All of the explicit commands above still exist unchanged
// for anyone who wants to skip the detection and name the shape/scale
// themselves.
struct SniffResult {
    int columns = 0;         // 0 (not a clean numeric table -> general), 2, 3, or 7
    int decimals_a = 0;      // max decimal digits seen: all columns (2/3-col) or first 3 (7-col, position)
    int decimals_b = 0;      // 7-col only: last 4 columns (orientation)
    double max_abs_a = 0.0;
    double max_abs_b = 0.0;
    size_t rows = 0;         // ok_lines: non-blank lines matching the detected column count
    size_t total_lines = 0;  // all non-blank lines considered, whether or not they matched
};

// Digits after '.' in a token's own text (not the parsed double's binary
// value -- text is what tells us the file's real quoted precision).
// Returns -1 if the token uses exponent notation, where "digits after
// the dot" isn't a meaningful precision measure.
int count_decimals(const std::string& tok) {
    if (tok.find_first_of("eE") != std::string::npos) return -1;
    auto dot = tok.find('.');
    if (dot == std::string::npos) return 0;
    int n = 0;
    for (size_t i = dot + 1; i < tok.size() && std::isdigit((unsigned char)tok[i]); i++) n++;
    return n;
}

SniffResult sniff_table(const std::string& path) {
    SniffResult r;
    std::ifstream f(path);
    if (!f) return r;
    std::string line;
    int expected_columns = -1;
    size_t total_lines = 0, ok_lines = 0;
    bool any_exponent = false;

    while (std::getline(f, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        total_lines++;
        std::istringstream ss(line);
        std::vector<std::string> toks;
        std::string tok;
        while (ss >> tok) toks.push_back(tok);

        bool all_numeric = !toks.empty();
        std::vector<double> vals(toks.size());
        for (size_t i = 0; i < toks.size() && all_numeric; i++) {
            try {
                size_t consumed = 0;
                vals[i] = std::stod(toks[i], &consumed);
                if (consumed != toks[i].size()) all_numeric = false;
            } catch (...) { all_numeric = false; }
        }
        if (!all_numeric) continue;

        int cols = (int)toks.size();
        if (expected_columns == -1) expected_columns = cols;
        if (cols != expected_columns) continue;

        ok_lines++;
        for (int i = 0; i < cols; i++) {
            bool group_a = (cols != 7) || (i < 3);
            int dec = count_decimals(toks[i]);
            if (dec < 0) { any_exponent = true; dec = 0; }
            if (group_a) { r.decimals_a = std::max(r.decimals_a, dec); r.max_abs_a = std::max(r.max_abs_a, std::fabs(vals[i])); }
            else         { r.decimals_b = std::max(r.decimals_b, dec); r.max_abs_b = std::max(r.max_abs_b, std::fabs(vals[i])); }
        }
    }

    r.total_lines = total_lines;
    r.rows = ok_lines; // set even on the "not classified" paths below, so a caller (inspect) can still report a real match-rate diagnostic
    if (total_lines == 0 || expected_columns <= 0) return r;
    if ((double)ok_lines / (double)total_lines < 0.95) return r; // not a clean table -> general
    if (expected_columns != 2 && expected_columns != 3 && expected_columns != 7) return r;

    r.columns = expected_columns;
    if (any_exponent) { r.decimals_a = std::max(r.decimals_a, 6); r.decimals_b = std::max(r.decimals_b, 6); }
    r.decimals_a = std::min(r.decimals_a, 9);
    r.decimals_b = std::min(r.decimals_b, 9);
    return r;
}

i64 pow10i(int n) {
    i64 v = 1;
    for (int i = 0; i < n; i++) v *= 10;
    return v;
}

// scale = 10^decimals, but never so large that max_abs * scale would
// overflow the int32 the codec stores components as -- shrinking scale
// (losing some precision, but staying correct) rather than ever risking
// wraparound, and saying so honestly if it had to.
i64 safe_scale(int decimals, double max_abs, bool& capped) {
    i64 scale = pow10i(decimals);
    capped = false;
    while (scale > 1 && max_abs * (double)scale > 2.0e9) { scale /= 10; capped = true; }
    if (scale < 1) scale = 1;
    return scale;
}

u32 quality_to_quant_step(int quality) {
    quality = std::max(1, std::min(9, quality));
    return 1u << (10 - quality); // quality 9 -> 2, quality 1 -> 512
}

bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return (bool)f;
}

int cmd_squeeze(const std::string& in, std::string out, bool have_quality, int quality,
                bool explain, bool force, bool gpu, const std::string& level) {
    if (out.empty()) out = in + ".csa";
    if (!force && file_exists(out))
        throw std::runtime_error("output already exists: " + out + " (pass an output path, or --force to overwrite)");

    auto raw = read_file(in);
    SniffResult s = sniff_table(in);
    bool lossy = have_quality;
    u32 quant_step = lossy ? quality_to_quant_step(quality) : 1;
    const u32 kResync = 64;

    std::vector<u8> file_out;
    size_t raw_estimate = raw.size();
    std::string shape_name = "general (arbitrary bytes)";
    std::string extra; // --explain detail lines

    if (s.columns == 2 || s.columns == 3) {
        bool capped = false;
        i64 scale = safe_scale(s.decimals_a, s.max_abs_a, capped);
        shape_name = (s.columns == 2) ? "geo2d (2D points/trajectory)" : "geo3d (3D points/trajectory)";
        auto pts2 = s.columns == 2 ? read_points2d(in, scale) : std::vector<Point2i>{};
        auto pts3 = s.columns == 3 ? read_points3d(in, scale) : std::vector<Point3i>{};
        raw_estimate = (s.columns == 2 ? pts2.size() * 2 : pts3.size() * 3) * sizeof(double);

        std::vector<u8> blob = s.columns == 2
            ? (lossy ? compress_geo2d_lossy(pts2, quant_step, kResync) : compress_geo2d(pts2))
            : (lossy ? compress_geo3d_lossy(pts3, quant_step, kResync) : compress_geo3d(pts3));
        write_geo_header(file_out, (u8)s.columns, scale);
        file_out.insert(file_out.end(), blob.begin(), blob.end());

        // Always verify round-trip live, not just trust the test suite --
        // this is the entry point meant to be trusted without reading the
        // manual, so "lossless"/error numbers below are checked facts.
        double max_err = 0.0;
        if (s.columns == 2) {
            auto back = decompress_geo2d(blob);
            size_t n = std::min(pts2.size(), back.size());
            if (n) max_err = (double)max_abs_diff_i32(reinterpret_cast<const i32*>(pts2.data()),
                                                       reinterpret_cast<const i32*>(back.data()), n * 2) / (double)scale;
        } else {
            auto back = decompress_geo3d(blob);
            size_t n = std::min(pts3.size(), back.size());
            if (n) max_err = (double)max_abs_diff_i32(reinterpret_cast<const i32*>(pts3.data()),
                                                       reinterpret_cast<const i32*>(back.data()), n * 3) / (double)scale;
        }
        std::ostringstream e;
        e << "  detected: " << s.columns << " numeric columns (" << s.rows << " rows) -> " << shape_name << " mode\n";
        e << "  scale: " << scale << " (auto" << (capped ? ", capped to avoid overflow" : "") << ")\n";
        e << "  round-trip: verified, max coordinate error=" << max_err << " (in original file's units)";
        extra = e.str();
    } else if (s.columns == 7) {
        bool capped_a = false, capped_b = false;
        i64 scale = safe_scale(s.decimals_a, s.max_abs_a, capped_a);
        i64 qscale = safe_scale(s.decimals_b, s.max_abs_b, capped_b);
        shape_name = "pose (6-DOF position + orientation)";
        auto poses = read_poses(in, scale, qscale);
        raw_estimate = poses.size() * 7 * sizeof(double);

        std::vector<u8> blob = lossy
            ? compress_pose_lossy(poses, quant_step, kResync, quant_step, kResync)
            : compress_pose(poses);
        write_pose_header(file_out, scale, qscale);
        file_out.insert(file_out.end(), blob.begin(), blob.end());

        auto back = decompress_pose(blob);
        double max_pos_err = 0.0, max_quat_err = 0.0;
        for (size_t i = 0; i < poses.size() && i < back.size(); i++) {
            max_pos_err = std::max({max_pos_err,
                std::abs((double)(poses[i].position.x - back[i].position.x)) / (double)scale,
                std::abs((double)(poses[i].position.y - back[i].position.y)) / (double)scale,
                std::abs((double)(poses[i].position.z - back[i].position.z)) / (double)scale});
            max_quat_err = std::max({max_quat_err,
                std::abs((double)(poses[i].orientation.w - back[i].orientation.w)) / (double)qscale,
                std::abs((double)(poses[i].orientation.x - back[i].orientation.x)) / (double)qscale,
                std::abs((double)(poses[i].orientation.y - back[i].orientation.y)) / (double)qscale,
                std::abs((double)(poses[i].orientation.z - back[i].orientation.z)) / (double)qscale});
        }
        std::ostringstream e;
        e << "  detected: 7 numeric columns (" << s.rows << " rows) -> " << shape_name << " mode\n";
        e << "  scale: " << scale << (capped_a ? " (capped)" : "") << ", qscale: " << qscale << (capped_b ? " (capped)" : "") << " (both auto)\n";
        e << "  round-trip: verified, max position error=" << max_pos_err << ", max orientation-component error=" << max_quat_err << " (in original file's units)";
        extra = e.str();
    } else {
        if (lossy) {
            std::cout << "note: --quality only applies to detected geo2d/geo3d/pose data; "
                         "this file has no lossy mode, compressing losslessly instead.\n";
            lossy = false;
        }
        int max_chain; size_t nice_length;
        level_to_lz_params(level, max_chain, nice_length);
        file_out = compress(raw, gpu, max_chain, nice_length);

        auto back = decompress(file_out);
        if (back != raw) throw std::runtime_error("internal error: general compress() round-trip mismatch -- please report this");

        std::ostringstream e;
        e << "  detected: not a clean numeric table -> " << shape_name << " mode" << (gpu ? " [gpu]" : "") << " [level=" << level << "]\n";
        e << "  round-trip: verified, byte-identical";
        extra = e.str();
    }

    write_file(out, file_out);
    double pct = raw.empty() ? 0.0 : 100.0 * (1.0 - (double)file_out.size() / (double)raw.size());
    std::cout << "squeeze: " << in << " (" << raw.size() << " bytes) -> " << out << " (" << file_out.size()
               << " bytes), " << pct << "% smaller, " << (lossy ? ("lossy (quality " + std::to_string(std::max(1, std::min(9, quality))) + "/9)") : "lossless") << "\n";
    if (explain) std::cout << extra << "\n";
    return 0;
}

// Largest quant_step in [1, hi] whose real measured error (from eval_fn,
// which must actually compress+decompress+measure -- not estimate) stays
// <= budget. eval_fn is assumed monotonically non-decreasing in
// quant_step, which is true of this codec's quantization by construction
// (a coarser step can only add error, never remove it). Returns 0 if even
// quant_step=1 -- the finest granularity the *lossy* code path offers --
// already exceeds the budget; the caller should recommend true lossless
// in that case; rather than a fabricated "best effort" number.
template <typename EvalFn>
u32 search_max_quant_step(double budget, u32 hi, EvalFn eval_fn) {
    if (eval_fn(1u) > budget) return 0;
    u32 lo = 1, best = 1;
    while (lo <= hi) {
        u32 mid = lo + (hi - lo) / 2;
        double err = eval_fn(mid);
        if (err <= budget) {
            best = mid;
            if (mid == hi) break;
            lo = mid + 1;
        } else {
            if (mid == lo) break;
            hi = mid - 1;
        }
    }
    return best;
}

// Auto-Optimize: rather than making the caller guess a quant_step and
// check whether the resulting error happens to be acceptable, this
// searches the codec's real quant_step parameter space directly against
// a user-stated error budget, using the same real round-trip
// compress+decompress+measure cmd_squeeze already trusts -- not an
// estimate or a model of expected error, the actual measured error at
// each candidate step. For pose data with both a position and a
// rotation budget, position and rotation are searched independently
// (holding the other at quant_step=1, since the format encodes them as
// two genuinely separate sub-streams -- see FORMAT.md's Pose mode
// description) and the combined configuration is re-verified for real
// before being reported, rather than assumed additive.
int cmd_optimize(const std::string& in, std::string out, bool have_pos_budget, double pos_budget,
                  bool have_quat_budget, double quat_budget, bool explain, bool force) {
    if (out.empty()) out = in + ".csa";
    if (!force && file_exists(out))
        throw std::runtime_error("output already exists: " + out + " (pass an output path, or --force to overwrite)");
    if (!have_pos_budget && !have_quat_budget)
        throw std::runtime_error("optimize needs at least one of --max-pos-error / --max-quat-error");

    auto raw = read_file(in);
    SniffResult s = sniff_table(in);
    const u32 kResync = 64;
    const u32 kSearchHi = 1u << 20; // real upper bound tried; ~20 real compress+decompress rounds to bisect it

    if (s.columns != 2 && s.columns != 3 && s.columns != 7)
        throw std::runtime_error("optimize only applies to detected geo2d/geo3d/pose data -- this file has no lossy mode to search");

    std::ostringstream report;
    std::vector<u8> file_out;
    double raw_bytes = (double)raw.size();

    if (s.columns == 2 || s.columns == 3) {
        if (!have_pos_budget)
            throw std::runtime_error("this file is geo2d/geo3d (position only) -- pass --max-pos-error");
        bool capped = false;
        i64 scale = safe_scale(s.decimals_a, s.max_abs_a, capped);
        auto pts2 = s.columns == 2 ? read_points2d(in, scale) : std::vector<Point2i>{};
        auto pts3 = s.columns == 3 ? read_points3d(in, scale) : std::vector<Point3i>{};

        auto eval = [&](u32 step) -> double {
            std::vector<u8> blob = s.columns == 2 ? compress_geo2d_lossy(pts2, step, kResync)
                                                   : compress_geo3d_lossy(pts3, step, kResync);
            double max_err = 0.0;
            // This closure runs once per candidate quant_step in
            // search_max_quant_step's binary search (~20 real
            // compress+decompress+measure rounds per optimize() call) --
            // the hottest of this file's four max-abs-diff call sites, and
            // the one most worth the SIMD kernel below.
            if (s.columns == 2) {
                auto back = decompress_geo2d(blob);
                size_t n = std::min(pts2.size(), back.size());
                if (n) max_err = (double)max_abs_diff_i32(reinterpret_cast<const i32*>(pts2.data()),
                                                           reinterpret_cast<const i32*>(back.data()), n * 2) / (double)scale;
            } else {
                auto back = decompress_geo3d(blob);
                size_t n = std::min(pts3.size(), back.size());
                if (n) max_err = (double)max_abs_diff_i32(reinterpret_cast<const i32*>(pts3.data()),
                                                           reinterpret_cast<const i32*>(back.data()), n * 3) / (double)scale;
            }
            return max_err;
        };

        u32 step = search_max_quant_step(pos_budget, kSearchHi, eval);
        if (step == 0) {
            std::vector<u8> blob = s.columns == 2 ? compress_geo2d(pts2) : compress_geo3d(pts3);
            write_geo_header(file_out, (u8)s.columns, scale);
            file_out.insert(file_out.end(), blob.begin(), blob.end());
            report << "  even the finest lossy quantization step (1) measured "
                   << eval(1) << " error, already within or exceeding your " << pos_budget
                   << " budget either way -- using true lossless instead, since it's always at least as good and is exact.\n";
        } else {
            double achieved = eval(step);
            std::vector<u8> blob = s.columns == 2 ? compress_geo2d_lossy(pts2, step, kResync) : compress_geo3d_lossy(pts3, step, kResync);
            write_geo_header(file_out, (u8)s.columns, scale);
            file_out.insert(file_out.end(), blob.begin(), blob.end());
            report << "  searched quant_step in [1, " << kSearchHi << "]: chose " << step
                   << " (real measured max error=" << achieved << ", budget=" << pos_budget << ")\n";
        }
    } else {
        bool capped_a = false, capped_b = false;
        i64 scale = safe_scale(s.decimals_a, s.max_abs_a, capped_a);
        i64 qscale = safe_scale(s.decimals_b, s.max_abs_b, capped_b);
        auto poses = read_poses(in, scale, qscale);

        auto eval_pos = [&](u32 pos_step) -> double {
            auto blob = compress_pose_lossy(poses, pos_step, kResync, 1, kResync);
            auto back = decompress_pose(blob);
            double e = 0.0;
            for (size_t i = 0; i < poses.size() && i < back.size(); i++)
                e = std::max({e, std::abs((double)(poses[i].position.x - back[i].position.x)) / (double)scale,
                              std::abs((double)(poses[i].position.y - back[i].position.y)) / (double)scale,
                              std::abs((double)(poses[i].position.z - back[i].position.z)) / (double)scale});
            return e;
        };
        auto eval_quat = [&](u32 quat_step) -> double {
            auto blob = compress_pose_lossy(poses, 1, kResync, quat_step, kResync);
            auto back = decompress_pose(blob);
            double e = 0.0;
            for (size_t i = 0; i < poses.size() && i < back.size(); i++)
                e = std::max({e, std::abs((double)(poses[i].orientation.w - back[i].orientation.w)) / (double)qscale,
                              std::abs((double)(poses[i].orientation.x - back[i].orientation.x)) / (double)qscale,
                              std::abs((double)(poses[i].orientation.y - back[i].orientation.y)) / (double)qscale,
                              std::abs((double)(poses[i].orientation.z - back[i].orientation.z)) / (double)qscale});
            return e;
        };

        u32 pos_step = have_pos_budget ? search_max_quant_step(pos_budget, kSearchHi, eval_pos) : 1;
        u32 quat_step = have_quat_budget ? search_max_quant_step(quat_budget, kSearchHi, eval_quat) : 1;
        bool pos_fell_back_lossless = have_pos_budget && pos_step == 0;
        bool quat_fell_back_lossless = have_quat_budget && quat_step == 0;

        if (pos_fell_back_lossless && quat_fell_back_lossless) {
            auto blob = compress_pose(poses);
            write_pose_header(file_out, scale, qscale);
            file_out.insert(file_out.end(), blob.begin(), blob.end());
            report << "  neither budget is satisfiable by the lossy path's finest step -- using true lossless for both position and rotation.\n";
        } else {
            if (pos_step == 0) pos_step = 1;
            if (quat_step == 0) quat_step = 1;
            // Real combined verification -- position and rotation are
            // independent sub-streams by format design (see FORMAT.md),
            // so no interaction is expected, but this re-measures the
            // actual combined blob rather than assuming the two isolated
            // searches compose without checking.
            auto blob = compress_pose_lossy(poses, pos_step, kResync, quat_step, kResync);
            auto back = decompress_pose(blob);
            double combined_pos_err = 0.0, combined_quat_err = 0.0;
            for (size_t i = 0; i < poses.size() && i < back.size(); i++) {
                combined_pos_err = std::max({combined_pos_err,
                    std::abs((double)(poses[i].position.x - back[i].position.x)) / (double)scale,
                    std::abs((double)(poses[i].position.y - back[i].position.y)) / (double)scale,
                    std::abs((double)(poses[i].position.z - back[i].position.z)) / (double)scale});
                combined_quat_err = std::max({combined_quat_err,
                    std::abs((double)(poses[i].orientation.w - back[i].orientation.w)) / (double)qscale,
                    std::abs((double)(poses[i].orientation.x - back[i].orientation.x)) / (double)qscale,
                    std::abs((double)(poses[i].orientation.y - back[i].orientation.y)) / (double)qscale,
                    std::abs((double)(poses[i].orientation.z - back[i].orientation.z)) / (double)qscale});
            }
            bool pos_ok = !have_pos_budget || combined_pos_err <= pos_budget;
            bool quat_ok = !have_quat_budget || combined_quat_err <= quat_budget;
            if (!pos_ok || !quat_ok)
                throw std::runtime_error("internal error: combined configuration violated a budget that passed in isolation "
                                          "(pos_err=" + std::to_string(combined_pos_err) + ", quat_err=" + std::to_string(combined_quat_err) +
                                          ") -- please report this, it should not be possible given independent sub-streams");
            write_pose_header(file_out, scale, qscale);
            file_out.insert(file_out.end(), blob.begin(), blob.end());
            report << "  position: " << (pos_fell_back_lossless ? "lossless (finest lossy step still too coarse)" : "quant_step=" + std::to_string(pos_step))
                   << ", measured error=" << combined_pos_err << (have_pos_budget ? " (budget=" + std::to_string(pos_budget) + ")" : " (no budget given)") << "\n";
            report << "  rotation: " << (quat_fell_back_lossless ? "lossless (finest lossy step still too coarse)" : "quant_step=" + std::to_string(quat_step))
                   << ", measured error=" << combined_quat_err << (have_quat_budget ? " (budget=" + std::to_string(quat_budget) + ")" : " (no budget given)") << "\n";
        }
    }

    write_file(out, file_out);
    double pct = raw_bytes == 0 ? 0.0 : 100.0 * (1.0 - (double)file_out.size() / raw_bytes);
    std::cout << "optimize: " << in << " (" << raw.size() << " bytes) -> " << out << " (" << file_out.size()
               << " bytes), " << pct << "% smaller\n";
    if (explain) std::cout << report.str();
    return 0;
}

int cmd_unsqueeze(const std::string& in, std::string out, bool explain, bool force) {
    if (out.empty()) out = in + ".restored";
    if (!force && file_exists(out))
        throw std::runtime_error("output already exists: " + out + " (pass an output path, or --force to overwrite)");

    auto file = read_file(in);
    std::string shape_name = "general (arbitrary bytes)";
    std::vector<u8> restored;

    if (file.size() >= 5 && file[0] == (u8)kGeoMagic[0] && file[1] == (u8)kGeoMagic[1] &&
        file[2] == (u8)kGeoMagic[2] && file[3] == (u8)kGeoMagic[3]) {
        u8 dims = file[4];
        size_t pos = 5;
        if (dims == 2) {
            i64 scale = (i64)get_u64(file.data(), file.size(), pos);
            std::vector<u8> blob(file.begin() + pos, file.end());
            auto pts = decompress_geo2d(blob);
            std::ostringstream out_text;
            out_text << std::fixed << std::setprecision(9);
            for (auto& p : pts) out_text << (p.x / (double)scale) << " " << (p.y / (double)scale) << "\n";
            std::string s = out_text.str();
            restored.assign(s.begin(), s.end());
            shape_name = "geo2d (" + std::to_string(pts.size()) + " points)";
        } else if (dims == 3) {
            i64 scale = (i64)get_u64(file.data(), file.size(), pos);
            std::vector<u8> blob(file.begin() + pos, file.end());
            auto pts = decompress_geo3d(blob);
            std::ostringstream out_text;
            out_text << std::fixed << std::setprecision(9);
            for (auto& p : pts) out_text << (p.x / (double)scale) << " " << (p.y / (double)scale) << " " << (p.z / (double)scale) << "\n";
            std::string s = out_text.str();
            restored.assign(s.begin(), s.end());
            shape_name = "geo3d (" + std::to_string(pts.size()) + " points)";
        } else if (dims == 7) {
            i64 scale = (i64)get_u64(file.data(), file.size(), pos);
            i64 qscale = (i64)get_u64(file.data(), file.size(), pos);
            std::vector<u8> blob(file.begin() + pos, file.end());
            auto poses = decompress_pose(blob);
            std::ostringstream out_text;
            out_text << std::fixed << std::setprecision(9);
            for (auto& p : poses)
                out_text << (p.position.x / (double)scale) << " " << (p.position.y / (double)scale) << " " << (p.position.z / (double)scale) << " "
                         << (p.orientation.w / (double)qscale) << " " << (p.orientation.x / (double)qscale) << " "
                         << (p.orientation.y / (double)qscale) << " " << (p.orientation.z / (double)qscale) << "\n";
            std::string s = out_text.str();
            restored.assign(s.begin(), s.end());
            shape_name = "pose (" + std::to_string(poses.size()) + " poses)";
        } else {
            throw std::runtime_error("unrecognized CSAG file (unknown dims byte)");
        }
    } else {
        restored = decompress(file);
    }

    write_file(out, restored);
    std::cout << "unsqueeze: " << in << " (" << file.size() << " bytes) -> " << out << " (" << restored.size() << " bytes)\n";
    if (explain) std::cout << "  detected: " << shape_name << " mode\n";
    return 0;
}

int cmd_info(const std::string& path) {
    auto data = read_file(path);
    std::cout << path << ": " << data.size() << " bytes\n";
    std::cout << "cuda available: " << (cuda_is_available() ? "yes" : "no") << "\n";
    return 0;
}

const char* mode_name(u8 mode) {
    switch ((Mode)mode) {
        case Mode::Raw: return "Raw";
        case Mode::General: return "General";
        case Mode::Geo2D: return "Geo2D";
        case Mode::Geo3D: return "Geo3D";
        case Mode::GeneralLZ: return "GeneralLZ";
        case Mode::GeneralBWT: return "GeneralBWT";
        case Mode::Pose: return "Pose";
        default: return "unknown";
    }
}

// Reports exactly the real, documented fields (FORMAT.md's Layer 2 "CSAG"
// header and Layer 1 "CSA1" magic+mode byte) -- nothing past the mode byte
// is decoded here, since the mode-specific payload isn't byte-level
// specified in FORMAT.md either; inventing a deeper breakdown here would
// just be guessing. For a file that isn't a recognized .csa at all, falls
// back to the same sniff_table() squeeze itself uses, so "what would
// squeeze do with this" is answerable without actually compressing.
int cmd_inspect(const std::string& path) {
    auto file = read_file(path);
    std::cout << "inspect: " << path << " (" << file.size() << " bytes)\n";

    bool has_geo_header = file.size() >= 5 && file[0] == (u8)kGeoMagic[0] && file[1] == (u8)kGeoMagic[1] &&
                          file[2] == (u8)kGeoMagic[2] && file[3] == (u8)kGeoMagic[3];
    bool has_csa1_header = !has_geo_header && file.size() >= 5 &&
                           file[0] == 'C' && file[1] == 'S' && file[2] == 'A' && file[3] == '1';

    if (has_geo_header) {
        u8 dims = file[4];
        size_t pos = 5;
        const char* dims_name = dims == 2 ? "geo2d" : dims == 3 ? "geo3d" : dims == 7 ? "pose" : "unknown";
        std::cout << "layer 2 (CLI/browser convention): magic=\"CSAG\", dims=" << (int)dims << " (" << dims_name << ")";
        if (dims == 2 || dims == 3 || dims == 7) {
            if (file.size() < pos + 8) { std::cout << "\n  TRUNCATED: missing scale field\n"; return 1; }
            i64 scale = (i64)get_u64(file.data(), file.size(), pos);
            std::cout << ", scale=" << scale;
            if (dims == 7) {
                if (file.size() < pos + 8) { std::cout << "\n  TRUNCATED: missing qscale field\n"; return 1; }
                i64 qscale = (i64)get_u64(file.data(), file.size(), pos);
                std::cout << ", qscale=" << qscale;
            }
        } else {
            std::cout << "\n  unrecognized dims byte -- not a value this inspector's Layer 2 knows (2/3/7)\n";
            return 1;
        }
        std::cout << "\n";
        file.erase(file.begin(), file.begin() + (long)pos);
        has_csa1_header = file.size() >= 5 && file[0] == 'C' && file[1] == 'S' && file[2] == 'A' && file[3] == '1';
    }

    if (has_csa1_header) {
        u8 mode = file[4];
        std::cout << "layer 1 (library container): magic=\"CSA1\", mode=" << (int)mode << " (" << mode_name(mode) << ")\n";
        std::cout << "payload: " << (file.size() - 5) << " bytes (mode-specific serialized result; not further decoded by inspect)\n";
        return 0;
    }
    if (has_geo_header) {
        // had a CSAG header but the remainder wasn't a CSA1 blob -- a real, reportable inconsistency, not silently ignored.
        std::cout << "  WARNING: CSAG header present but no CSA1 magic follows it -- file may be truncated or corrupted\n";
        return 1;
    }

    std::cout << "not a recognized .csa file (no CSAG or CSA1 magic) -- treating as raw squeeze input:\n";
    SniffResult s = sniff_table(path);
    if (s.total_lines == 0) {
        std::cout << "  could not read any non-blank lines from this file\n";
        return 0;
    }
    double confidence = s.total_lines ? 100.0 * (double)s.rows / (double)s.total_lines : 0.0;
    if (s.columns == 0) {
        std::cout << "  detected shape: general (arbitrary bytes) -- would fall back to compress()\n";
        std::cout << "  numeric-table match rate: " << s.rows << " / " << s.total_lines << " lines ("
                  << std::fixed << std::setprecision(1) << confidence << "% -- below the 95% threshold squeeze() requires)\n";
        std::cout << "  recommendation: NOT a good fit for this codec's specialty (no clean 2/3/7-column numeric\n"
                      "    table detected) -- general-purpose compression (zstd/gzip/lzma) is the fair comparison\n"
                      "    here, not CSA's own lossy modes. See WHY_NOT_ZSTD.md.\n";
        return 0;
    }
    const char* shape = s.columns == 2 ? "geo2d" : s.columns == 3 ? "geo3d" : "pose (6-DOF)";
    std::cout << "  detected shape: " << shape << " (" << s.columns << " numeric columns)\n";
    std::cout << "  confidence: " << s.rows << " / " << s.total_lines << " lines matched ("
              << std::fixed << std::setprecision(1) << confidence << "%)\n";
    std::cout << "  decimal precision: position=" << s.decimals_a << (s.columns == 7 ? ", orientation=" + std::to_string(s.decimals_b) : "") << "\n";
    std::ostringstream max_abs_line;
    max_abs_line << std::fixed << std::setprecision(6) << "  max abs value: position=" << s.max_abs_a;
    if (s.columns == 7) max_abs_line << ", orientation=" << s.max_abs_b;
    std::cout << max_abs_line.str() << "\n";
    std::cout << "  recommendation: a good fit for this codec's specialty (a clean, consistent " << s.columns
               << "-column numeric table was detected). `scissorc squeeze` will compress it losslessly by\n"
                  "    default; `scissorc optimize` can search for the strongest lossy configuration that\n"
                  "    still satisfies a real position/rotation error budget you state, if lossless isn't required.\n";
    return 0;
}

// Structural verification of a .csa file on its own, without the original
// input to diff against: attempts the same decode `unsqueeze` would, and
// reports success (with the real decoded record count) or the exact
// exception the decoder threw. This is NOT the same claim as "matches the
// original bit-for-bit" -- that already happens automatically inside
// squeeze() itself (see cmd_squeeze's own round-trip check) and needs the
// original file, which a standalone .csa doesn't carry. What this can
// honestly promise: the file decodes without the decoder detecting
// internal inconsistency -- exactly the guarantee FORMAT.md's own
// "known gap: no CRC" section describes, no more and no less.
int cmd_verify(const std::string& path) {
    auto file = read_file(path);
    try {
        if (file.size() >= 5 && file[0] == (u8)kGeoMagic[0] && file[1] == (u8)kGeoMagic[1] &&
            file[2] == (u8)kGeoMagic[2] && file[3] == (u8)kGeoMagic[3]) {
            u8 dims = file[4];
            size_t pos = 5;
            if (dims == 2) {
                i64 scale = (i64)get_u64(file.data(), file.size(), pos);
                (void)scale;
                std::vector<u8> blob(file.begin() + (long)pos, file.end());
                auto pts = decompress_geo2d(blob);
                std::cout << "verify: " << path << " -- OK, decoded geo2d (" << pts.size() << " points), " << file.size() << " bytes\n";
            } else if (dims == 3) {
                i64 scale = (i64)get_u64(file.data(), file.size(), pos);
                (void)scale;
                std::vector<u8> blob(file.begin() + (long)pos, file.end());
                auto pts = decompress_geo3d(blob);
                std::cout << "verify: " << path << " -- OK, decoded geo3d (" << pts.size() << " points), " << file.size() << " bytes\n";
            } else if (dims == 7) {
                i64 scale = (i64)get_u64(file.data(), file.size(), pos);
                i64 qscale = (i64)get_u64(file.data(), file.size(), pos);
                (void)scale; (void)qscale;
                std::vector<u8> blob(file.begin() + (long)pos, file.end());
                auto poses = decompress_pose(blob);
                std::cout << "verify: " << path << " -- OK, decoded pose (" << poses.size() << " poses), " << file.size() << " bytes\n";
            } else {
                std::cout << "verify: " << path << " -- FAILED: unrecognized CSAG dims byte (" << (int)dims << ")\n";
                return 1;
            }
        } else {
            auto restored = decompress(file);
            std::cout << "verify: " << path << " -- OK, decoded general (" << restored.size() << " bytes), " << file.size() << " bytes\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cout << "verify: " << path << " -- FAILED: " << e.what() << "\n";
        return 1;
    }
}

// Real gzip-equivalent baseline via vendored miniz (see CMakeLists.txt) --
// not an estimate, an actual mz_compress2()/mz_uncompress() round trip on
// the identical raw bytes CSA sees, timed with the same std::chrono
// pattern cmd_bench_transform already uses elsewhere in this file.
struct GzipResult { size_t compressed_bytes; double compress_ms; double decompress_ms; bool round_trip_ok; };
GzipResult gzip_equivalent_benchmark(const std::vector<u8>& raw) {
    mz_ulong bound = mz_compressBound((mz_ulong)raw.size());
    std::vector<u8> out(bound);
    mz_ulong out_len = bound;
    auto t0 = std::chrono::steady_clock::now();
    int rc = mz_compress2(out.data(), &out_len, raw.data(), (mz_ulong)raw.size(), MZ_BEST_COMPRESSION);
    auto t1 = std::chrono::steady_clock::now();
    if (rc != MZ_OK) throw std::runtime_error("miniz mz_compress2 failed (code " + std::to_string(rc) + ")");
    out.resize(out_len);

    std::vector<u8> back(raw.size());
    mz_ulong back_len = (mz_ulong)raw.size();
    auto t2 = std::chrono::steady_clock::now();
    rc = mz_uncompress(back.data(), &back_len, out.data(), (mz_ulong)out.size());
    auto t3 = std::chrono::steady_clock::now();
    bool ok = (rc == MZ_OK) && (back_len == raw.size()) && (back == raw);

    return {
        out.size(),
        std::chrono::duration<double, std::milli>(t1 - t0).count(),
        std::chrono::duration<double, std::milli>(t3 - t2).count(),
        ok,
    };
}

// `scissorc benchmark <file>`: a real, measured comparison table -- CSA's
// own auto-detected mode (whatever squeeze() would actually pick) against
// a real gzip-equivalent baseline (vendored miniz, not an estimate).
// Deliberately does NOT include an lzma column: a real LZMA implementation
// is a much larger dependency than the single-file miniz vendored for
// this feature (see CMakeLists.txt's comment on that trade-off) -- adding
// it is future scope, not something this command pretends to already do.
// For the project's own already-measured lzma/zstd/brotli comparisons on
// real datasets, see REAL_POSE_BENCHMARK.md / REAL_GEO_BENCHMARK.md
// instead, which this command does not attempt to reproduce or replace.
int cmd_benchmark(const std::string& path) {
    auto raw = read_file(path);
    SniffResult s = sniff_table(path);

    std::string shape_name = "general (arbitrary bytes)";
    std::vector<u8> csa_blob;
    double csa_compress_ms = 0.0, csa_decompress_ms = 0.0;
    bool csa_round_trip_ok = false;

    auto t0 = std::chrono::steady_clock::now();
    if (s.columns == 2 || s.columns == 3) {
        bool capped = false;
        i64 scale = safe_scale(s.decimals_a, s.max_abs_a, capped);
        shape_name = s.columns == 2 ? "geo2d" : "geo3d";
        if (s.columns == 2) {
            auto pts = read_points2d(path, scale);
            csa_blob = compress_geo2d(pts);
            auto t1 = std::chrono::steady_clock::now();
            csa_compress_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            auto t2 = std::chrono::steady_clock::now();
            auto back = decompress_geo2d(csa_blob);
            auto t3 = std::chrono::steady_clock::now();
            csa_decompress_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
            csa_round_trip_ok = back.size() == pts.size();
        } else {
            auto pts = read_points3d(path, scale);
            csa_blob = compress_geo3d(pts);
            auto t1 = std::chrono::steady_clock::now();
            csa_compress_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            auto t2 = std::chrono::steady_clock::now();
            auto back = decompress_geo3d(csa_blob);
            auto t3 = std::chrono::steady_clock::now();
            csa_decompress_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
            csa_round_trip_ok = back.size() == pts.size();
        }
    } else if (s.columns == 7) {
        bool capped_a = false, capped_b = false;
        i64 scale = safe_scale(s.decimals_a, s.max_abs_a, capped_a);
        i64 qscale = safe_scale(s.decimals_b, s.max_abs_b, capped_b);
        shape_name = "pose";
        auto poses = read_poses(path, scale, qscale);
        csa_blob = compress_pose(poses);
        auto t1 = std::chrono::steady_clock::now();
        csa_compress_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        auto t2 = std::chrono::steady_clock::now();
        auto back = decompress_pose(csa_blob);
        auto t3 = std::chrono::steady_clock::now();
        csa_decompress_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        csa_round_trip_ok = back.size() == poses.size();
    } else {
        csa_blob = compress(raw, false, kLzDefaultMaxChain, kLzDefaultNiceLength);
        auto t1 = std::chrono::steady_clock::now();
        csa_compress_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        auto t2 = std::chrono::steady_clock::now();
        auto back = decompress(csa_blob);
        auto t3 = std::chrono::steady_clock::now();
        csa_decompress_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        csa_round_trip_ok = back == raw;
    }

    GzipResult gz = gzip_equivalent_benchmark(raw);

    auto pct = [&](size_t compressed) {
        return raw.empty() ? 0.0 : 100.0 * (1.0 - (double)compressed / (double)raw.size());
    };
    std::cout << "benchmark: " << path << " (" << raw.size() << " bytes), detected: " << shape_name << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::left << std::setw(24) << "  algorithm" << std::right
               << std::setw(12) << "size (bytes)" << std::setw(11) << "reduction" << std::setw(15) << "compress (ms)"
               << std::setw(18) << "decompress (ms)" << std::setw(11) << "verified" << "\n";
    std::string csa_label = "  CSA (" + shape_name + (shape_name == "general" ? ", lossless)" : ", exact)");
    std::cout << std::left << std::setw(24) << csa_label << std::right
               << std::setw(12) << csa_blob.size() << std::setw(10) << pct(csa_blob.size()) << "%"
               << std::setw(15) << csa_compress_ms << std::setw(18) << csa_decompress_ms
               << std::setw(11) << (csa_round_trip_ok ? "yes" : "NO -- MISMATCH") << "\n";
    std::cout << std::left << std::setw(24) << "  gzip-equivalent" << std::right
               << std::setw(12) << gz.compressed_bytes << std::setw(10) << pct(gz.compressed_bytes) << "%"
               << std::setw(15) << gz.compress_ms << std::setw(18) << gz.decompress_ms
               << std::setw(11) << (gz.round_trip_ok ? "yes" : "NO -- MISMATCH") << "\n";
    std::cout << "  (lzma/zstd/brotli comparisons on real datasets already measured offline -- see "
                  "REAL_POSE_BENCHMARK.md / REAL_GEO_BENCHMARK.md; not reproduced live here)\n";

    if (!csa_round_trip_ok || !gz.round_trip_ok)
        return 1; // a benchmark whose own round-trip failed is not a number anyone should trust
    return 0;
}

// `scissorc scale-test`: a real thread-scaling measurement, not a claimed
// one. Measures the interleaved rANS format (src/rans_coder.cpp,
// csa::ThreadPool-backed as of this pass -- see docs/PARALLELISM.md) at
// 1/2/4/8/hardware_concurrency() lanes on a real (synthetically
// generated but not trivially compressible or trivially parallel-
// friendly) buffer, and reports actual measured speedup(N)=T(1)/T(N) and
// parallel efficiency(N)=speedup(N)/N -- not an estimate, not a formula
// applied to a single sample, but genuine repeated wall-clock
// measurement (median of several runs per lane count, to damp scheduler
// noise, all reported, not silently discarded). This is deliberately the
// *only* real data-parallel primitive in this codebase today: the
// adaptive order-1 range coder that compress()/compress_pose()/etc.
// actually use for their real output has a genuine sequential dependency
// (each symbol's coding depends on the running frequency table built
// from every symbol before it), so it is not a candidate for this kind
// of block-parallel speedup without changing the format -- see
// docs/PARALLELISM.md for that distinction spelled out in full, rather
// than silently parallelizing only the thing that was easy to
// parallelize and implying the whole codec scales this way.
// A real hardware/software fingerprint (see csa/system_info.hpp): every
// existing benchmark doc in this repository states its hardware by hand,
// in prose, once per document -- this is the first place that
// information is gathered by code, so a future benchmark run can attach
// it automatically instead of risking it drifting from the machine that
// actually produced the numbers.
int cmd_system(bool json) {
    SystemInfo info = query_system_info();
    auto bytes_to_gib = [](u64 b) { return (double)b / (1024.0 * 1024.0 * 1024.0); };

    if (json) {
        std::cout << "{\n"
                   << "  \"os\": \"" << info.os << "\",\n"
                   << "  \"compiler\": \"" << info.compiler << "\",\n"
                   << "  \"build_type\": \"" << info.build_type << "\",\n"
                   << "  \"logical_cores\": " << info.logical_cores << ",\n"
                   << "  \"cpu_brand\": \"" << info.cpu_brand << "\",\n"
                   << "  \"simd_backend\": \"" << info.simd_backend << "\",\n"
                   << "  \"ram_total_bytes\": " << info.ram_total_bytes << ",\n"
                   << "  \"cuda_available\": " << (info.cuda_available ? "true" : "false") << ",\n"
                   << "  \"cuda_device_name\": \"" << info.cuda_device_name << "\",\n"
                   << "  \"cuda_device_memory_bytes\": " << info.cuda_device_memory_bytes << "\n"
                   << "}\n";
        return 0;
    }

    std::cout << "OS               " << info.os << "\n"
              << "Compiler         " << info.compiler << " (" << info.build_type << ")\n"
              << "CPU              " << (info.cpu_brand.empty() ? "NOT AVAILABLE" : info.cpu_brand) << "\n"
              << "Logical cores    " << (info.logical_cores ? std::to_string(info.logical_cores) : "NOT AVAILABLE") << "\n"
              << "SIMD backend     " << info.simd_backend << "\n"
              << "RAM              " << (info.ram_total_bytes ? std::to_string(bytes_to_gib(info.ram_total_bytes)).substr(0, 5) + " GiB" : "NOT AVAILABLE") << "\n";
    if (info.cuda_available) {
        std::cout << "CUDA             AVAILABLE (" << info.cuda_device_name << ", "
                   << std::to_string(bytes_to_gib(info.cuda_device_memory_bytes)).substr(0, 5) << " GiB)\n";
    } else {
        std::cout << "CUDA             UNAVAILABLE (no device found, or built with WITH_CUDA=OFF)\n";
    }
    return 0;
}

int cmd_scale_test(size_t buffer_bytes, int repeats, bool json) {
    std::vector<u8> data(buffer_bytes);
    // A synthetic but non-degenerate byte stream: skewed-but-not-constant
    // frequency distribution (so the entropy coder does real, nontrivial
    // work per symbol, not a trivial all-zeros pass), matching the same
    // generator already used by test_rans_coder()'s own timing case.
    for (size_t i = 0; i < buffer_bytes; i++) data[i] = (u8)((i * 2654435761u) % 256);

    std::vector<int> lane_counts = {1, 2, 4};
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    if (std::find(lane_counts.begin(), lane_counts.end(), 8) == lane_counts.end() && hw >= 8) lane_counts.push_back(8);
    if (std::find(lane_counts.begin(), lane_counts.end(), (int)hw) == lane_counts.end()) lane_counts.push_back((int)hw);
    std::sort(lane_counts.begin(), lane_counts.end());
    lane_counts.erase(std::unique(lane_counts.begin(), lane_counts.end()), lane_counts.end());

    struct Row { int lanes; double encode_ms; double decode_ms; };
    std::vector<Row> rows;
    for (int lanes : lane_counts) {
        std::vector<double> enc_samples, dec_samples;
        std::vector<u8> coded;
        for (int r = 0; r < repeats; r++) {
            auto t0 = std::chrono::steady_clock::now();
            coded = encode_interleaved_rans(data, lanes);
            auto t1 = std::chrono::steady_clock::now();
            auto decoded = decode_interleaved_rans(coded);
            auto t2 = std::chrono::steady_clock::now();
            if (decoded != data) throw std::runtime_error("scale-test: round-trip mismatch at " + std::to_string(lanes) + " lanes -- refusing to report a speedup number next to a correctness failure");
            enc_samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            dec_samples.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
        }
        std::sort(enc_samples.begin(), enc_samples.end());
        std::sort(dec_samples.begin(), dec_samples.end());
        rows.push_back({lanes, enc_samples[enc_samples.size() / 2], dec_samples[dec_samples.size() / 2]});
    }

    double baseline_enc = rows[0].encode_ms, baseline_dec = rows[0].decode_ms;
    if (json) {
        std::cout << "{\n  \"buffer_bytes\": " << buffer_bytes << ",\n  \"repeats\": " << repeats
                   << ",\n  \"hardware_concurrency\": " << hw << ",\n  \"rows\": [\n";
        for (size_t i = 0; i < rows.size(); i++) {
            const auto& r = rows[i];
            std::cout << "    {\"lanes\": " << r.lanes << ", \"encode_ms\": " << r.encode_ms
                       << ", \"decode_ms\": " << r.decode_ms
                       << ", \"encode_speedup\": " << (baseline_enc / r.encode_ms)
                       << ", \"decode_speedup\": " << (baseline_dec / r.decode_ms)
                       << ", \"encode_efficiency\": " << (baseline_enc / r.encode_ms / r.lanes)
                       << ", \"decode_efficiency\": " << (baseline_dec / r.decode_ms / r.lanes) << "}"
                       << (i + 1 < rows.size() ? ",\n" : "\n");
        }
        std::cout << "  ]\n}\n";
    } else {
        std::cout << "scale-test: " << buffer_bytes << " bytes, median of " << repeats
                   << " runs/lane-count, hardware_concurrency()=" << hw << "\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::left << std::setw(8) << "lanes" << std::right << std::setw(14) << "encode (ms)"
                   << std::setw(10) << "speedup" << std::setw(12) << "efficiency"
                   << std::setw(14) << "decode (ms)" << std::setw(10) << "speedup" << std::setw(12) << "efficiency" << "\n";
        for (const auto& r : rows) {
            std::cout << std::left << std::setw(8) << r.lanes << std::right
                       << std::setw(14) << r.encode_ms << std::setw(10) << (baseline_enc / r.encode_ms)
                       << std::setw(12) << (baseline_enc / r.encode_ms / r.lanes)
                       << std::setw(14) << r.decode_ms << std::setw(10) << (baseline_dec / r.decode_ms)
                       << std::setw(12) << (baseline_dec / r.decode_ms / r.lanes) << "\n";
        }
        std::cout << "(this measures the interleaved-rANS entropy backend only -- the adaptive range coder\n"
                      " actually used by squeeze()/compress_pose()/etc. has a sequential dependency and is not\n"
                      " parallelized by this pass; see docs/PARALLELISM.md)\n";
    }
    return 0;
}

// Diagnostic: isolates lz_parse's match-finding time from lz_encode's
// entropy-coding time, to find out which stage actually dominates a slow
// compress() call instead of guessing.
int cmd_bench_lz(const std::string& path) {
    auto data = read_file(path);
    auto t0 = std::chrono::steady_clock::now();
    auto tokens = lz_parse(data);
    auto t1 = std::chrono::steady_clock::now();
    double parse_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    size_t n_literals = 0, n_matches = 0;
    for (auto& t : tokens) (t.is_match ? n_matches : n_literals)++;

    std::cout << "bench-lz: input=" << data.size() << " bytes, parse_time="
              << parse_ms << " ms (" << (data.size() / 1e6 / (parse_ms / 1000.0)) << " MB/s), "
              << "tokens=" << tokens.size() << " (" << n_literals << " literals, "
              << n_matches << " matches)\n";
    return 0;
}

// Isolates the Pantograph Lift forward transform itself (CPU or GPU) from
// the CPU-sequential entropy coding stage that always follows it in
// compress(), so CPU-vs-GPU timing reflects the actual thing the CUDA
// kernel accelerates rather than being diluted by a stage neither path
// speeds up.
//
// --repeat N runs N calls in the same process (same warm CUDA context,
// same driver/GPU clock state) and reports the first call separately from
// the rest: the first pays whatever one-time wake/context-creation cost
// the GPU owes (see GPU_BENCHMARKS.md), while calls 2..N show the
// steady-state per-call cost of whichever entry point is under test.
//
// use_session controls which GPU entry point is timed: the one-shot
// pantograph_lift_forward_cuda() (allocates/frees device+pinned buffers
// every single call) or a single CudaLiftSession reused across all
// `repeat` calls (buffers allocated once, only grown if a later call
// needs more capacity). Comparing the two directly, on the same input
// size and repeat count, is how the actual cudaMalloc/cudaFree overhead
// this session type exists to remove gets measured rather than assumed.
int cmd_bench_transform(size_t n, bool gpu, int repeat, bool use_session) {
    std::vector<i32> data(n);
    for (size_t i = 0; i < n; i++) {
        data[i] = (i32)(1000.0 * std::sin((double)i * 0.001) + (double)(i % 7));
    }

    std::vector<double> times_ms;
    bool used_gpu = false;
    CudaLiftSession session; // constructed even if unused; cheap when !gpu
    for (int rep = 0; rep < repeat; rep++) {
        auto t0 = std::chrono::steady_clock::now();
        LiftResult lr;
        if (gpu) {
            used_gpu = use_session ? session.forward(data, lr) : pantograph_lift_forward_cuda(data, lr);
            if (!used_gpu) lr = pantograph_lift_forward(data);
        } else {
            lr = pantograph_lift_forward(data);
        }
        auto t1 = std::chrono::steady_clock::now();
        times_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    const char* mode = use_session ? "session" : "one-shot";
    if (repeat == 1) {
        std::cout << "bench-transform: n=" << n << " gpu_requested=" << (gpu ? "yes" : "no")
                  << " gpu_used=" << (used_gpu ? "yes" : "no") << " mode=" << mode
                  << " time_ms=" << times_ms[0] << "\n";
        return 0;
    }

    double rest_sum = 0.0, rest_min = times_ms[1];
    for (int i = 1; i < repeat; i++) {
        rest_sum += times_ms[i];
        rest_min = std::min(rest_min, times_ms[i]);
    }
    double rest_avg = rest_sum / (double)(repeat - 1);
    std::cout << "bench-transform: n=" << n << " gpu_requested=" << (gpu ? "yes" : "no")
              << " gpu_used=" << (used_gpu ? "yes" : "no") << " mode=" << mode
              << " first_ms=" << times_ms[0]
              << " rest_avg_ms=" << rest_avg
              << " rest_min_ms=" << rest_min
              << " repeat=" << repeat << "\n";
    return 0;
}

void usage() {
    std::cerr <<
        "usage:\n"
        "  scissorc squeeze <in> [out] [--quality 1-9] [--explain] [--force] [--gpu] [--level fast|balanced|high]\n"
        "      Point it at any file. Detects whether it's a 2D/3D point trajectory,\n"
        "      a 6-DOF pose stream, or just bytes, and compresses it accordingly --\n"
        "      lossless by default. [out] defaults to <in>.csa. --quality trades size\n"
        "      for a small, bounded numeric error (9=least lossy .. 1=most lossy);\n"
        "      omit it for exact lossless output. --explain prints what was detected\n"
        "      and the real, measured round-trip error.\n"
        "  scissorc unsqueeze <in> [out] [--explain] [--force]\n"
        "      Reverses squeeze. [out] defaults to <in>.restored.\n"
        "  scissorc optimize <in> [out] --max-pos-error N [--max-quat-error N] [--explain] [--force]\n"
        "      Auto-Optimize: searches the codec's real quant_step parameter space\n"
        "      (via actual compress+decompress+measure at each candidate, not an\n"
        "      estimate) for the strongest compression whose measured error stays\n"
        "      within your stated budget. --max-pos-error is in the file's own\n"
        "      position units; --max-quat-error is raw quaternion-component error\n"
        "      (not degrees). Falls back to true lossless, honestly, if even the\n"
        "      finest lossy step exceeds your budget. Only applies to detected\n"
        "      geo2d/geo3d/pose data.\n"
        "  scissorc benchmark <file>\n"
        "      Real, measured comparison: CSA's own auto-detected mode (whatever\n"
        "      squeeze() would pick) vs. a real gzip-equivalent baseline (vendored\n"
        "      miniz) -- actual size, actual compress/decompress time, actual\n"
        "      round-trip verification for both, not estimates. Does not include\n"
        "      lzma/zstd/brotli (see REAL_POSE_BENCHMARK.md/REAL_GEO_BENCHMARK.md\n"
        "      for those, measured offline on real datasets instead).\n"
        "  scissorc system [--json]\n"
        "      Real hardware/software fingerprint: OS, compiler, CPU brand,\n"
        "      logical core count, SIMD backend actually detected on this\n"
        "      machine, RAM, and CUDA device (name + memory) if available.\n"
        "      Any field this platform can't report shows as NOT AVAILABLE\n"
        "      rather than a guess.\n"
        "  scissorc scale-test [--bytes N] [--repeats N] [--json]\n"
        "      Real measured thread-scaling: the interleaved-rANS entropy backend\n"
        "      (the only genuinely data-parallel primitive in this codebase --\n"
        "      see docs/PARALLELISM.md) at 1/2/4/8/hardware_concurrency() lanes,\n"
        "      median of --repeats runs each (default 5), on a --bytes-sized\n"
        "      (default 4000000) real generated buffer. Reports actual\n"
        "      speedup(N)=T(1)/T(N) and efficiency(N)=speedup(N)/N. Does NOT\n"
        "      measure the adaptive range coder squeeze()/compress_pose() etc.\n"
        "      actually use, which has a sequential dependency and isn't\n"
        "      parallelized by this pass.\n"
        "  scissorc inspect <file>\n"
        "      Reports what squeeze/unsqueeze would actually do with this file --\n"
        "      the detected shape and confidence for a raw input, or the real\n"
        "      CSAG/CSA1 header fields (dims, scale, mode) for a .csa file --\n"
        "      without compressing or decompressing anything.\n"
        "  scissorc verify <file.csa>\n"
        "      Attempts to decode a .csa file on its own (no original to diff\n"
        "      against) and reports success with the decoded record count, or\n"
        "      the exact decode error. Exit code 1 on failure.\n"
        "\n"
        "  scissorc compress <in> <out> [--gpu] [--level fast|balanced|high]\n"
        "  scissorc decompress <in> <out>\n"
        "  scissorc compress-geo2d <in.xy> <out> [--scale N]\n"
        "  scissorc decompress-geo2d <in> <out.xy>\n"
        "  scissorc compress-geo2d-lossy <in.xy> <out> --quant N --resync N [--scale N]\n"
        "  scissorc compress-geo3d <in.xyz> <out> [--scale N]\n"
        "  scissorc decompress-geo3d <in> <out.xyz>\n"
        "  scissorc compress-geo3d-lossy <in.xyz> <out> --quant N --resync N [--scale N]\n"
        "  scissorc compress-pose <in.pose> <out> [--scale N] [--qscale N]\n"
        "  scissorc decompress-pose <in> <out.pose>\n"
        "  scissorc compress-pose-lossy <in.pose> <out> --pos-quant N --pos-resync N --quat-quant N --quat-resync N [--scale N] [--qscale N]\n"
        "  scissorc info <file>\n"
        "  scissorc bench-transform <n> [--gpu] [--session] [--repeat N]\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string cmd = argv[1];
    try {
        if (cmd == "squeeze" && argc >= 3) {
            std::string in = argv[2];
            std::string out;
            bool have_quality = false;
            int quality = 9;
            bool explain = false, force = false, gpu = false;
            std::string level = "balanced";
            int i = 3;
            if (i < argc && std::string(argv[i]).rfind("--", 0) != 0) out = argv[i++];
            for (; i < argc; i++) {
                std::string a = argv[i];
                if (a == "--quality" && i + 1 < argc) { have_quality = true; quality = std::stoi(argv[++i]); }
                else if (a == "--explain") explain = true;
                else if (a == "--force") force = true;
                else if (a == "--gpu") gpu = true;
                else if (a == "--level" && i + 1 < argc) level = argv[++i];
            }
            return cmd_squeeze(in, out, have_quality, quality, explain, force, gpu, level);
        } else if (cmd == "unsqueeze" && argc >= 3) {
            std::string in = argv[2];
            std::string out;
            bool explain = false, force = false;
            int i = 3;
            if (i < argc && std::string(argv[i]).rfind("--", 0) != 0) out = argv[i++];
            for (; i < argc; i++) {
                std::string a = argv[i];
                if (a == "--explain") explain = true;
                else if (a == "--force") force = true;
            }
            return cmd_unsqueeze(in, out, explain, force);
        } else if (cmd == "optimize" && argc >= 3) {
            std::string in = argv[2];
            std::string out;
            bool explain = false, force = false;
            bool have_pos_budget = false, have_quat_budget = false;
            double pos_budget = 0.0, quat_budget = 0.0;
            int i = 3;
            if (i < argc && std::string(argv[i]).rfind("--", 0) != 0) out = argv[i++];
            for (; i < argc; i++) {
                std::string a = argv[i];
                if (a == "--max-pos-error" && i + 1 < argc) { have_pos_budget = true; pos_budget = std::stod(argv[++i]); }
                else if (a == "--max-quat-error" && i + 1 < argc) { have_quat_budget = true; quat_budget = std::stod(argv[++i]); }
                else if (a == "--explain") explain = true;
                else if (a == "--force") force = true;
            }
            return cmd_optimize(in, out, have_pos_budget, pos_budget, have_quat_budget, quat_budget, explain, force);
        } else if (cmd == "compress" && argc >= 4) {
            bool gpu = false;
            std::string level = "balanced";
            for (int i = 4; i < argc; i++) {
                std::string a = argv[i];
                if (a == "--gpu") gpu = true;
                else if (a == "--level" && i + 1 < argc) level = argv[++i];
            }
            return cmd_compress(argv[2], argv[3], gpu, level);
        } else if (cmd == "decompress" && argc >= 4) {
            return cmd_decompress(argv[2], argv[3]);
        } else if (cmd == "compress-geo2d" && argc >= 4) {
            i64 scale = 1000;
            if (argc >= 6 && std::string(argv[4]) == "--scale") scale = std::stoll(argv[5]);
            return cmd_compress_geo2d(argv[2], argv[3], scale);
        } else if (cmd == "decompress-geo2d" && argc >= 4) {
            return cmd_decompress_geo2d(argv[2], argv[3]);
        } else if (cmd == "compress-geo2d-lossy" && argc >= 4) {
            i64 scale = 1000;
            u32 quant_step = 1, resync_interval = 0;
            for (int i = 4; i + 1 < argc; i += 2) {
                std::string a = argv[i];
                if (a == "--scale") scale = std::stoll(argv[i + 1]);
                else if (a == "--quant") quant_step = (u32)std::stoul(argv[i + 1]);
                else if (a == "--resync") resync_interval = (u32)std::stoul(argv[i + 1]);
            }
            return cmd_compress_geo2d_lossy(argv[2], argv[3], scale, quant_step, resync_interval);
        } else if (cmd == "compress-geo3d" && argc >= 4) {
            i64 scale = 1000;
            if (argc >= 6 && std::string(argv[4]) == "--scale") scale = std::stoll(argv[5]);
            return cmd_compress_geo3d(argv[2], argv[3], scale);
        } else if (cmd == "decompress-geo3d" && argc >= 4) {
            return cmd_decompress_geo3d(argv[2], argv[3]);
        } else if (cmd == "compress-geo3d-lossy" && argc >= 4) {
            i64 scale = 1000;
            u32 quant_step = 1, resync_interval = 0;
            for (int i = 4; i + 1 < argc; i += 2) {
                std::string a = argv[i];
                if (a == "--scale") scale = std::stoll(argv[i + 1]);
                else if (a == "--quant") quant_step = (u32)std::stoul(argv[i + 1]);
                else if (a == "--resync") resync_interval = (u32)std::stoul(argv[i + 1]);
            }
            return cmd_compress_geo3d_lossy(argv[2], argv[3], scale, quant_step, resync_interval);
        } else if (cmd == "compress-pose" && argc >= 4) {
            i64 scale = 1000, qscale = 1000000;
            for (int i = 4; i + 1 < argc; i += 2) {
                std::string a = argv[i];
                if (a == "--scale") scale = std::stoll(argv[i + 1]);
                else if (a == "--qscale") qscale = std::stoll(argv[i + 1]);
            }
            return cmd_compress_pose(argv[2], argv[3], scale, qscale);
        } else if (cmd == "decompress-pose" && argc >= 4) {
            return cmd_decompress_pose(argv[2], argv[3]);
        } else if (cmd == "compress-pose-lossy" && argc >= 4) {
            i64 scale = 1000, qscale = 1000000;
            u32 pos_quant_step = 1, pos_resync_interval = 0, quat_quant_step = 1, quat_resync_interval = 0;
            for (int i = 4; i + 1 < argc; i += 2) {
                std::string a = argv[i];
                if (a == "--scale") scale = std::stoll(argv[i + 1]);
                else if (a == "--qscale") qscale = std::stoll(argv[i + 1]);
                else if (a == "--pos-quant") pos_quant_step = (u32)std::stoul(argv[i + 1]);
                else if (a == "--pos-resync") pos_resync_interval = (u32)std::stoul(argv[i + 1]);
                else if (a == "--quat-quant") quat_quant_step = (u32)std::stoul(argv[i + 1]);
                else if (a == "--quat-resync") quat_resync_interval = (u32)std::stoul(argv[i + 1]);
            }
            return cmd_compress_pose_lossy(argv[2], argv[3], scale, qscale, pos_quant_step, pos_resync_interval,
                                            quat_quant_step, quat_resync_interval);
        } else if (cmd == "info" && argc >= 3) {
            return cmd_info(argv[2]);
        } else if (cmd == "inspect" && argc >= 3) {
            return cmd_inspect(argv[2]);
        } else if (cmd == "verify" && argc >= 3) {
            return cmd_verify(argv[2]);
        } else if (cmd == "benchmark" && argc >= 3) {
            return cmd_benchmark(argv[2]);
        } else if (cmd == "system") {
            bool json = false;
            for (int i = 2; i < argc; i++) if (std::string(argv[i]) == "--json") json = true;
            return cmd_system(json);
        } else if (cmd == "scale-test") {
            size_t buffer_bytes = 4000000;
            int repeats = 5;
            bool json = false;
            for (int i = 2; i < argc; i++) {
                std::string a = argv[i];
                if (a == "--bytes" && i + 1 < argc) buffer_bytes = (size_t)std::stoull(argv[++i]);
                else if (a == "--repeats" && i + 1 < argc) repeats = std::stoi(argv[++i]);
                else if (a == "--json") json = true;
            }
            return cmd_scale_test(buffer_bytes, repeats, json);
        } else if (cmd == "bench-lz" && argc >= 3) {
            return cmd_bench_lz(argv[2]);
        } else if (cmd == "bench-transform" && argc >= 3) {
            size_t n = (size_t)std::stoull(argv[2]);
            bool gpu = false;
            bool use_session = false;
            int repeat = 1;
            for (int i = 3; i < argc; i++) {
                std::string a = argv[i];
                if (a == "--gpu") gpu = true;
                else if (a == "--session") use_session = true;
                else if (a == "--repeat" && i + 1 < argc) repeat = std::stoi(argv[++i]);
            }
            return cmd_bench_transform(n, gpu, repeat, use_session);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    usage();
    return 1;
}
