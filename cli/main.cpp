// scissorc — command-line front end for the Collapsible Scissored Surfaces
// compression algorithm.
//
//   scissorc compress   <in> <out> [--gpu]
//   scissorc decompress <in> <out>
//   scissorc compress-geo2d   <in.xy>  <out> [--scale N]
//   scissorc decompress-geo2d <in> <out.xy>
//   scissorc compress-geo3d   <in.xyz> <out> [--scale N]
//   scissorc decompress-geo3d <in> <out.xyz>
//   scissorc info <file>
//
// Geo-mode point files are plain text, one point per line: "x y" (geo2d)
// or "x y z" (geo3d), floating point. They are quantized to fixed-point
// integers (value * scale, rounded) so the transform stays exact-integer
// lossless; --scale controls the quantization step (default 1000, i.e.
// 3 decimal digits of precision).
#include "csa/codec.hpp"
#include "csa/pantograph_lift_cuda.hpp"
#include <algorithm>
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

int cmd_compress(const std::string& in, const std::string& out, bool gpu) {
    auto data = read_file(in);
    auto blob = compress(data, gpu);
    write_file(out, blob);
    std::cout << "compress: " << data.size() << " -> " << blob.size() << " bytes ("
              << (data.empty() ? 0.0 : 100.0 * (1.0 - (double)blob.size() / (double)data.size()))
              << "% smaller)" << (gpu ? " [gpu]" : "") << "\n";
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

int cmd_info(const std::string& path) {
    auto data = read_file(path);
    std::cout << path << ": " << data.size() << " bytes\n";
    std::cout << "cuda available: " << (cuda_is_available() ? "yes" : "no") << "\n";
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
// steady-state per-call cost of this library's actual contract --
// pantograph_lift_forward_cuda allocates and frees its own device buffers
// every call, so this still isn't the best case a long-lived service
// reusing buffers across calls could achieve, but it is what today's API
// actually delivers under sustained use, measured honestly rather than
// assumed.
int cmd_bench_transform(size_t n, bool gpu, int repeat) {
    std::vector<i32> data(n);
    for (size_t i = 0; i < n; i++) {
        data[i] = (i32)(1000.0 * std::sin((double)i * 0.001) + (double)(i % 7));
    }

    std::vector<double> times_ms;
    bool used_gpu = false;
    for (int rep = 0; rep < repeat; rep++) {
        auto t0 = std::chrono::steady_clock::now();
        LiftResult lr;
        if (gpu) {
            used_gpu = pantograph_lift_forward_cuda(data, lr);
            if (!used_gpu) lr = pantograph_lift_forward(data);
        } else {
            lr = pantograph_lift_forward(data);
        }
        auto t1 = std::chrono::steady_clock::now();
        times_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    if (repeat == 1) {
        std::cout << "bench-transform: n=" << n << " gpu_requested=" << (gpu ? "yes" : "no")
                  << " gpu_used=" << (used_gpu ? "yes" : "no") << " time_ms=" << times_ms[0] << "\n";
        return 0;
    }

    double rest_sum = 0.0, rest_min = times_ms[1];
    for (int i = 1; i < repeat; i++) {
        rest_sum += times_ms[i];
        rest_min = std::min(rest_min, times_ms[i]);
    }
    double rest_avg = rest_sum / (double)(repeat - 1);
    std::cout << "bench-transform: n=" << n << " gpu_requested=" << (gpu ? "yes" : "no")
              << " gpu_used=" << (used_gpu ? "yes" : "no")
              << " first_ms=" << times_ms[0]
              << " rest_avg_ms=" << rest_avg
              << " rest_min_ms=" << rest_min
              << " repeat=" << repeat << "\n";
    return 0;
}

void usage() {
    std::cerr <<
        "usage:\n"
        "  scissorc compress <in> <out> [--gpu]\n"
        "  scissorc decompress <in> <out>\n"
        "  scissorc compress-geo2d <in.xy> <out> [--scale N]\n"
        "  scissorc decompress-geo2d <in> <out.xy>\n"
        "  scissorc compress-geo3d <in.xyz> <out> [--scale N]\n"
        "  scissorc decompress-geo3d <in> <out.xyz>\n"
        "  scissorc info <file>\n"
        "  scissorc bench-transform <n> [--gpu] [--repeat N]\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string cmd = argv[1];
    try {
        if (cmd == "compress" && argc >= 4) {
            bool gpu = (argc >= 5 && std::string(argv[4]) == "--gpu");
            return cmd_compress(argv[2], argv[3], gpu);
        } else if (cmd == "decompress" && argc >= 4) {
            return cmd_decompress(argv[2], argv[3]);
        } else if (cmd == "compress-geo2d" && argc >= 4) {
            i64 scale = 1000;
            if (argc >= 6 && std::string(argv[4]) == "--scale") scale = std::stoll(argv[5]);
            return cmd_compress_geo2d(argv[2], argv[3], scale);
        } else if (cmd == "decompress-geo2d" && argc >= 4) {
            return cmd_decompress_geo2d(argv[2], argv[3]);
        } else if (cmd == "compress-geo3d" && argc >= 4) {
            i64 scale = 1000;
            if (argc >= 6 && std::string(argv[4]) == "--scale") scale = std::stoll(argv[5]);
            return cmd_compress_geo3d(argv[2], argv[3], scale);
        } else if (cmd == "decompress-geo3d" && argc >= 4) {
            return cmd_decompress_geo3d(argv[2], argv[3]);
        } else if (cmd == "info" && argc >= 3) {
            return cmd_info(argv[2]);
        } else if (cmd == "bench-transform" && argc >= 3) {
            size_t n = (size_t)std::stoull(argv[2]);
            bool gpu = false;
            int repeat = 1;
            for (int i = 3; i < argc; i++) {
                std::string a = argv[i];
                if (a == "--gpu") gpu = true;
                else if (a == "--repeat" && i + 1 < argc) repeat = std::stoi(argv[++i]);
            }
            return cmd_bench_transform(n, gpu, repeat);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    usage();
    return 1;
}
