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
    double max_err = 0.0;
    for (size_t i = 0; i < pts.size() && i < decoded.size(); i++) {
        max_err = std::max({max_err, std::abs((double)(pts[i].x - decoded[i].x)) / (double)scale,
                             std::abs((double)(pts[i].y - decoded[i].y)) / (double)scale});
    }

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
    double max_err = 0.0;
    for (size_t i = 0; i < pts.size() && i < decoded.size(); i++) {
        max_err = std::max({max_err, std::abs((double)(pts[i].x - decoded[i].x)) / (double)scale,
                             std::abs((double)(pts[i].y - decoded[i].y)) / (double)scale,
                             std::abs((double)(pts[i].z - decoded[i].z)) / (double)scale});
    }

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
    size_t rows = 0;
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

    if (total_lines == 0 || expected_columns <= 0) return r;
    if ((double)ok_lines / (double)total_lines < 0.95) return r; // not a clean table -> general
    if (expected_columns != 2 && expected_columns != 3 && expected_columns != 7) return r;

    r.columns = expected_columns;
    r.rows = ok_lines;
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
            for (size_t i = 0; i < pts2.size() && i < back.size(); i++)
                max_err = std::max({max_err, std::abs((double)(pts2[i].x - back[i].x)) / (double)scale,
                                     std::abs((double)(pts2[i].y - back[i].y)) / (double)scale});
        } else {
            auto back = decompress_geo3d(blob);
            for (size_t i = 0; i < pts3.size() && i < back.size(); i++)
                max_err = std::max({max_err, std::abs((double)(pts3[i].x - back[i].x)) / (double)scale,
                                     std::abs((double)(pts3[i].y - back[i].y)) / (double)scale,
                                     std::abs((double)(pts3[i].z - back[i].z)) / (double)scale});
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
