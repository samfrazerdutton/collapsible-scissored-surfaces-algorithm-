// GPU measurement of the Quaternion Joint's per-block-per-lag calibration
// search (see include/csa/quaternion_calibration_cuda.hpp for the honest
// framing: this can only ever accelerate the calibration stage, not the
// sequential residual-encoding stage that follows it).
//
// One CUDA thread block per (calibration block, candidate lag) pair, with
// a shared-memory tree reduction (same pattern as
// pantograph_lift_cuda.cu's calibrate_blocks_kernel) computing seven
// running sums per pair, from which both the calibrated delta D and its
// exact SSE fall out in closed form -- no second pass over the data is
// needed for SSE, via the standard least-squares identity
// SSE_min = sum|b|^2 - |acc|^2/sum|a|^2 (see quaternion_joint.hpp's
// header comment for where acc/sum|a|^2 come from). Verified against the
// CPU's own calibrate_block/block_predict_sse before being trusted for
// any timing claim -- see tests/test_main.cpp's
// test_quat_calibration_cuda_matches_cpu.
#include "csa/quaternion_calibration_cuda.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

namespace csa {

namespace {

#define CUDA_OK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__, __LINE__, cudaGetErrorName(_e), cudaGetErrorString(_e)); \
    goto cuda_fail; } } while (0)

constexpr int kThreadsPerBlock = 128;
constexpr long long kFixedOneDev = 1LL << 16;

__device__ long long clampd_dev(double v, double lo, double hi) {
    if (!isfinite(v)) return 0;
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return (long long)llround(v * (double)kFixedOneDev);
}

__global__ void quat_calibrate_kernel(const int* qw, const int* qx, const int* qy, const int* qz,
                                       unsigned long long n, unsigned long long block_size,
                                       const unsigned int* lags, int num_lags,
                                       long long* out_delta, double* out_sse) {
    int block_idx = (int)blockIdx.x;
    int lag_idx = (int)blockIdx.y;
    unsigned int lag = lags[lag_idx];

    unsigned long long start = (unsigned long long)block_idx * block_size;
    unsigned long long end = min((unsigned long long)n, start + block_size);

    __shared__ double sh_a0[kThreadsPerBlock];
    __shared__ double sh_a1[kThreadsPerBlock];
    __shared__ double sh_a2[kThreadsPerBlock];
    __shared__ double sh_a3[kThreadsPerBlock];
    __shared__ double sh_norm2[kThreadsPerBlock];
    __shared__ double sh_bsq_hi[kThreadsPerBlock];
    __shared__ double sh_bsq_lo[kThreadsPerBlock];

    int tid = (int)threadIdx.x;
    double a0 = 0, a1 = 0, a2 = 0, a3 = 0, norm2 = 0, bsq_hi = 0, bsq_lo = 0;

    for (unsigned long long i = start + (unsigned long long)tid; i < end; i += blockDim.x) {
        double bw = (double)qw[i], bx = (double)qx[i], by = (double)qy[i], bz = (double)qz[i];
        if (i >= lag) {
            double aw = (double)qw[i - lag], ax = (double)qx[i - lag], ay = (double)qy[i - lag], az = (double)qz[i - lag];
            a0 += aw * bw + ax * bx + ay * by + az * bz;
            a1 += -ax * bw + aw * bx - az * by + ay * bz;
            a2 += -ay * bw + az * bx + aw * by - ax * bz;
            a3 += -az * bw - ay * bx + ax * by + aw * bz;
            norm2 += aw * aw + ax * ax + ay * ay + az * az;
            bsq_hi += bw * bw + bx * bx + by * by + bz * bz;
        } else {
            bsq_lo += bw * bw + bx * bx + by * by + bz * bz;
        }
    }
    sh_a0[tid] = a0; sh_a1[tid] = a1; sh_a2[tid] = a2; sh_a3[tid] = a3;
    sh_norm2[tid] = norm2; sh_bsq_hi[tid] = bsq_hi; sh_bsq_lo[tid] = bsq_lo;
    __syncthreads();

    for (int stride = kThreadsPerBlock / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_a0[tid] += sh_a0[tid + stride]; sh_a1[tid] += sh_a1[tid + stride];
            sh_a2[tid] += sh_a2[tid + stride]; sh_a3[tid] += sh_a3[tid + stride];
            sh_norm2[tid] += sh_norm2[tid + stride];
            sh_bsq_hi[tid] += sh_bsq_hi[tid + stride]; sh_bsq_lo[tid] += sh_bsq_lo[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        double A0 = sh_a0[0], A1 = sh_a1[0], A2 = sh_a2[0], A3 = sh_a3[0], N = sh_norm2[0];
        double dw = 0, dx = 0, dy = 0, dz = 0;
        if (N > 1e-9) { dw = A0 / N; dx = A1 / N; dy = A2 / N; dz = A3 / N; }

        long long out_idx = (long long)block_idx * num_lags + lag_idx;
        out_delta[out_idx * 4 + 0] = clampd_dev(dw, -8.0, 8.0);
        out_delta[out_idx * 4 + 1] = clampd_dev(dx, -8.0, 8.0);
        out_delta[out_idx * 4 + 2] = clampd_dev(dy, -8.0, 8.0);
        out_delta[out_idx * 4 + 3] = clampd_dev(dz, -8.0, 8.0);

        double reg_sse = sh_bsq_hi[0] - (N > 1e-9 ? (A0 * A0 + A1 * A1 + A2 * A2 + A3 * A3) / N : 0.0);
        out_sse[out_idx] = sh_bsq_lo[0] + reg_sse;
    }
}

} // namespace

bool quat_calibrate_all_blocks_cuda(const std::vector<i32>& qw, const std::vector<i32>& qx,
                                     const std::vector<i32>& qy, const std::vector<i32>& qz,
                                     size_t block_size, const std::vector<u32>& lags,
                                     std::vector<std::array<i64, 4>>& out_delta,
                                     std::vector<double>& out_sse) {
    if (!cuda_is_available()) return false;
    size_t n = qw.size();
    if (block_size == 0) block_size = 1;
    if (n == 0 || lags.empty()) { out_delta.clear(); out_sse.clear(); return true; }

    size_t nblocks = (n + block_size - 1) / block_size;
    int num_lags = (int)lags.size();
    size_t total = nblocks * (size_t)num_lags;

    int *d_qw = nullptr, *d_qx = nullptr, *d_qy = nullptr, *d_qz = nullptr;
    unsigned int* d_lags = nullptr;
    long long* d_delta = nullptr;
    double* d_sse = nullptr;

    CUDA_OK(cudaMalloc(&d_qw, n * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_qx, n * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_qy, n * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_qz, n * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_lags, (size_t)num_lags * sizeof(unsigned int)));
    CUDA_OK(cudaMalloc(&d_delta, total * 4 * sizeof(long long)));
    CUDA_OK(cudaMalloc(&d_sse, total * sizeof(double)));

    CUDA_OK(cudaMemcpy(d_qw, qw.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_qx, qx.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_qy, qy.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_qz, qz.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_lags, lags.data(), (size_t)num_lags * sizeof(unsigned int), cudaMemcpyHostToDevice));

    {
        dim3 grid((unsigned int)nblocks, (unsigned int)num_lags);
        quat_calibrate_kernel<<<grid, kThreadsPerBlock>>>(
            d_qw, d_qx, d_qy, d_qz, (unsigned long long)n, (unsigned long long)block_size,
            d_lags, num_lags, d_delta, d_sse);
    }
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    {
        std::vector<long long> h_delta(total * 4);
        std::vector<double> h_sse(total);
        CUDA_OK(cudaMemcpy(h_delta.data(), d_delta, h_delta.size() * sizeof(long long), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(h_sse.data(), d_sse, h_sse.size() * sizeof(double), cudaMemcpyDeviceToHost));

        out_delta.resize(total);
        out_sse.resize(total);
        for (size_t i = 0; i < total; i++) {
            out_delta[i] = {(i64)h_delta[i * 4 + 0], (i64)h_delta[i * 4 + 1],
                             (i64)h_delta[i * 4 + 2], (i64)h_delta[i * 4 + 3]};
            out_sse[i] = h_sse[i];
        }
    }

    cudaFree(d_qw); cudaFree(d_qx); cudaFree(d_qy); cudaFree(d_qz);
    cudaFree(d_lags); cudaFree(d_delta); cudaFree(d_sse);
    return true;

cuda_fail:
    if (d_qw) cudaFree(d_qw);
    if (d_qx) cudaFree(d_qx);
    if (d_qy) cudaFree(d_qy);
    if (d_qz) cudaFree(d_qz);
    if (d_lags) cudaFree(d_lags);
    if (d_delta) cudaFree(d_delta);
    if (d_sse) cudaFree(d_sse);
    return false;
}

} // namespace csa
