// Honest CPU-vs-GPU measurement for the Quaternion Joint's per-block-
// per-lag calibration search specifically (see
// include/csa/quaternion_calibration_cuda.hpp for why this can only ever
// speed up the calibration stage, not the sequential encoding stage that
// follows it). Same methodology as bench/gpu_crossover.py: isolate the
// exact same computation on both paths, measure with real wall-clock
// timing (best-of-N, warm GPU), across both real dataset sizes and much
// larger synthetic sizes to look for any crossover -- report whatever is
// actually measured, not an assumption.
//
// The CPU comparison function below (cpu_calibrate_all_blocks) computes
// the identical closed-form sums the CUDA kernel does, written as a plain
// CPU loop -- not quaternion_joint.cpp's own pick_block_lag (which is
// private to that translation unit, and whose CPU encoding step this
// isn't trying to measure anyway). test_quat_calibration_cuda_matches_cpu
// in tests/test_main.cpp already cross-checked the GPU kernel against the
// real production code path (quaternion_joint_forward's forced-lag mode)
// before this file's timing numbers are allowed to mean anything.
//
// Build manually against the built csa_core/csa_cuda static libraries,
// e.g. from an MSVC dev prompt:
//   cl /nologo /EHsc /O2 /MD /std:c++17 /I include
//      bench/quat_calibration_gpu_crossover.cpp
//      /Fe:build/quat_calibration_gpu_crossover.exe
//      /link build/csa_core.lib build/csa_cuda.lib
//      /LIBPATH:"<CUDA toolkit>\lib\x64" cudart.lib
#include "csa/quaternion_calibration_cuda.hpp"
#include "csa/quaternion_joint.hpp"
#include "csa/rod_joint_transform.hpp" // kRodJointCandidateLags
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>

using namespace csa;

namespace {

std::vector<i32> make_synthetic_component(size_t n, unsigned int seed) {
    std::vector<i32> v(n);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d(-(1 << 20), 1 << 20);
    for (size_t i = 0; i < n; i++) v[i] = d(rng);
    return v;
}

// Identical closed-form math to quat_calibrate_kernel, as a plain CPU
// loop -- the fair CPU-side comparison for this measurement.
void cpu_calibrate_all_blocks(const std::vector<i32>& qw, const std::vector<i32>& qx,
                               const std::vector<i32>& qy, const std::vector<i32>& qz,
                               size_t block_size, const std::vector<u32>& lags) {
    size_t n = qw.size();
    size_t nblocks = (n + block_size - 1) / block_size;
    for (size_t b = 0; b < nblocks; b++) {
        size_t start = b * block_size;
        size_t end = std::min(n, start + block_size);
        for (u32 lag : lags) {
            double a0 = 0, a1 = 0, a2 = 0, a3 = 0, norm2 = 0;
            for (size_t i = std::max(start, (size_t)lag); i < end; i++) {
                double aw = qw[i - lag], ax = qx[i - lag], ay = qy[i - lag], az = qz[i - lag];
                double bw = qw[i], bx = qx[i], by = qy[i], bz = qz[i];
                a0 += aw * bw + ax * bx + ay * by + az * bz;
                a1 += -ax * bw + aw * bx - az * by + ay * bz;
                a2 += -ay * bw + az * bx + aw * by - ax * bz;
                a3 += -az * bw - ay * bx + ax * by + aw * bz;
                norm2 += aw * aw + ax * ax + ay * ay + az * az;
            }
            volatile double sink = (norm2 > 1e-9) ? (a0 + a1 + a2 + a3) / norm2 : 0.0;
            (void)sink; // prevent the optimizer from deleting the "unused" result
        }
    }
}

double time_ms(const std::function<void()>& f) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

} // namespace

int main() {
    if (!cuda_is_available()) {
        std::printf("No CUDA device available -- cannot run this comparison.\n");
        return 1;
    }

    std::vector<u32> lags(kRodJointCandidateLags, kRodJointCandidateLags + kRodJointNumCandidateLags);
    const size_t block_size = 128;

    // Real dataset sizes (from REAL_POSE_BENCHMARK.md) plus much larger
    // synthetic sizes purely to look for any crossover -- reported
    // honestly as synthetic-only beyond real measured dataset sizes,
    // exactly like GPU_BENCHMARKS.md's own methodology.
    size_t sizes[] = {4541, 16702, 20957, 100000, 1000000, 10000000, 50000000};

    std::printf("%-12s %10s %14s %14s %10s %16s %10s\n",
                 "n", "nblocks", "CPU-calib(ms)", "GPU-calib(ms)", "GPU/CPU", "CPU-full(ms)", "calib%");
    for (size_t n : sizes) {
        auto qw = make_synthetic_component(n, 1);
        auto qx = make_synthetic_component(n, 2);
        auto qy = make_synthetic_component(n, 3);
        auto qz = make_synthetic_component(n, 4);
        size_t nblocks = (n + block_size - 1) / block_size;

        // Warm-up call (pays any one-time CUDA context cost here, not in
        // the timed loop -- same discipline as GPU_BENCHMARKS.md).
        std::vector<std::array<i64, 4>> delta;
        std::vector<double> sse;
        quat_calibrate_all_blocks_cuda(qw, qx, qy, qz, block_size, lags, delta, sse);

        int repeat = (n < 2000000) ? 3 : 1;
        double cpu_best = -1, gpu_best = -1;
        for (int r = 0; r < repeat; r++) {
            double t = time_ms([&]() { cpu_calibrate_all_blocks(qw, qx, qy, qz, block_size, lags); });
            if (cpu_best < 0 || t < cpu_best) cpu_best = t;
        }
        for (int r = 0; r < repeat; r++) {
            double t = time_ms([&]() { quat_calibrate_all_blocks_cuda(qw, qx, qy, qz, block_size, lags, delta, sse); });
            if (gpu_best < 0 || t < gpu_best) gpu_best = t;
        }

        // What fraction of the FULL CPU pipeline (calibration + the
        // sequential encoding pass that must stay on the CPU regardless)
        // does calibration actually represent? Real-valued quaternions
        // aren't required for this timing-only question -- reuse the
        // same synthetic (non-unit) component arrays as ordinary Quat4i
        // values, since quaternion_joint_forward's cost is data-
        // independent (same fixed amount of arithmetic regardless of
        // whether inputs are near-unit quaternions or not).
        std::vector<Quat4i> quats(n + 1);
        quats[0] = {1, 0, 0, 0};
        for (size_t i = 0; i < n; i++) quats[i + 1] = {qw[i], qx[i], qy[i], qz[i]};
        double full_best = -1;
        for (int r = 0; r < repeat; r++) {
            double t = time_ms([&]() { quaternion_joint_forward(quats); });
            if (full_best < 0 || t < full_best) full_best = t;
        }

        std::printf("%-12zu %10zu %14.3f %14.3f %10.2fx %16.3f %9.1f%%\n",
                     n, nblocks, cpu_best, gpu_best, gpu_best / cpu_best, full_best,
                     100.0 * cpu_best / full_best);
    }
    return 0;
}
