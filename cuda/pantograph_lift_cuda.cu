#include "csa/pantograph_lift_cuda.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

namespace csa {

namespace {

#define CUDA_OK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    goto cuda_fail; } } while (0)

constexpr int kFixedShiftDev = 16;
constexpr int kBlockSizeDev = (int)kPantographBlockSize; // must match pantograph_lift.hpp

__device__ __forceinline__ long long fixed_mul_round_dev(long long value, long long fixed_ratio) {
    long long prod = value * fixed_ratio;
    long long half = 1LL << (kFixedShiftDev - 1);
    prod += (prod >= 0) ? half : -half;
    return prod >> kFixedShiftDev;
}

// One CUDA thread block per calibration block: each thread loads one pair
// (or none, if this is the last, partial block), and a shared-memory tree
// reduction produces that block's four sums with no atomics needed since
// each CUDA block owns a disjoint data range.
__global__ void calibrate_blocks_kernel(const int* cur, unsigned long long half,
                                         double* sum_a, double* sum_b,
                                         double* sum_ab, double* sum_aa) {
    __shared__ double sh_a[kBlockSizeDev];
    __shared__ double sh_b[kBlockSizeDev];
    __shared__ double sh_ab[kBlockSizeDev];
    __shared__ double sh_aa[kBlockSizeDev];

    unsigned long long i = (unsigned long long)blockIdx.x * kBlockSizeDev + threadIdx.x;
    double a = 0.0, b = 0.0;
    bool valid = i < half;
    if (valid) {
        a = (double)cur[2 * i];
        b = (double)cur[2 * i + 1];
    }
    sh_a[threadIdx.x] = a;
    sh_b[threadIdx.x] = b;
    sh_ab[threadIdx.x] = a * b;
    sh_aa[threadIdx.x] = a * a;
    __syncthreads();

    for (int stride = kBlockSizeDev / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sh_a[threadIdx.x] += sh_a[threadIdx.x + stride];
            sh_b[threadIdx.x] += sh_b[threadIdx.x + stride];
            sh_ab[threadIdx.x] += sh_ab[threadIdx.x + stride];
            sh_aa[threadIdx.x] += sh_aa[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        sum_a[blockIdx.x] = sh_a[0];
        sum_b[blockIdx.x] = sh_b[0];
        sum_ab[blockIdx.x] = sh_ab[0];
        sum_aa[blockIdx.x] = sh_aa[0];
    }
}

__global__ void transform_kernel(const int* cur, unsigned long long half,
                                  const long long* ratios, const int* offsets,
                                  int* next_low, int* residual) {
    unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (unsigned long long i = idx; i < half; i += stride) {
        unsigned long long blk = i / kBlockSizeDev;
        long long ratio = ratios[blk];
        int offset = offsets[blk];
        int a = cur[2 * i];
        int b = cur[2 * i + 1];
        long long predicted = fixed_mul_round_dev(a, ratio) + offset;
        int d = (int)((long long)b - predicted);
        int upd = d >> 1;
        int s = (int)((long long)a + upd);
        next_low[i] = s;
        residual[i] = d;
    }
}

size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

size_t num_blocks_for(size_t half) {
    return (half + kPantographBlockSize - 1) / kPantographBlockSize;
}

} // namespace

bool cuda_is_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

bool pantograph_lift_forward_cuda(const std::vector<i32>& input, LiftResult& out) {
    if (!cuda_is_available()) return false;
    out = LiftResult{};
    out.original_length = input.size();
    if (input.empty()) return true;

    size_t padded_len = next_pow2(input.size());
    std::vector<i32> host_padded(padded_len);
    for (size_t i = 0; i < input.size(); i++) host_padded[i] = input[i];
    for (size_t i = input.size(); i < padded_len; i++) host_padded[i] = input.back();

    int* cur_d = nullptr;
    double* sum_a_d = nullptr;
    double* sum_b_d = nullptr;
    double* sum_ab_d = nullptr;
    double* sum_aa_d = nullptr;
    long long* ratios_d = nullptr;
    int* offsets_d = nullptr;
    int* next_low_d = nullptr;
    int* residual_d = nullptr;
    size_t max_blocks_allocated = 0;
    bool ok = false;

    CUDA_OK(cudaMalloc(&cur_d, padded_len * sizeof(int)));
    CUDA_OK(cudaMemcpy(cur_d, host_padded.data(), padded_len * sizeof(int), cudaMemcpyHostToDevice));

    {
        size_t cur_size = padded_len;
        while (cur_size > 1) {
            size_t half = cur_size / 2;
            size_t nblocks = num_blocks_for(half);

            if (nblocks > max_blocks_allocated) {
                if (sum_a_d) cudaFree(sum_a_d);
                if (sum_b_d) cudaFree(sum_b_d);
                if (sum_ab_d) cudaFree(sum_ab_d);
                if (sum_aa_d) cudaFree(sum_aa_d);
                if (ratios_d) cudaFree(ratios_d);
                if (offsets_d) cudaFree(offsets_d);
                CUDA_OK(cudaMalloc(&sum_a_d, nblocks * sizeof(double)));
                CUDA_OK(cudaMalloc(&sum_b_d, nblocks * sizeof(double)));
                CUDA_OK(cudaMalloc(&sum_ab_d, nblocks * sizeof(double)));
                CUDA_OK(cudaMalloc(&sum_aa_d, nblocks * sizeof(double)));
                CUDA_OK(cudaMalloc(&ratios_d, nblocks * sizeof(long long)));
                CUDA_OK(cudaMalloc(&offsets_d, nblocks * sizeof(int)));
                max_blocks_allocated = nblocks;
            }

            calibrate_blocks_kernel<<<(unsigned int)nblocks, kBlockSizeDev>>>(
                cur_d, half, sum_a_d, sum_b_d, sum_ab_d, sum_aa_d);
            CUDA_OK(cudaGetLastError());

            std::vector<double> sum_a(nblocks), sum_b(nblocks), sum_ab(nblocks), sum_aa(nblocks);
            CUDA_OK(cudaMemcpy(sum_a.data(), sum_a_d, nblocks * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(sum_b.data(), sum_b_d, nblocks * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(sum_ab.data(), sum_ab_d, nblocks * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(sum_aa.data(), sum_aa_d, nblocks * sizeof(double), cudaMemcpyDeviceToHost));

            std::vector<i64> ratios(nblocks);
            std::vector<i32> offsets(nblocks);
            for (size_t blk = 0; blk < nblocks; blk++) {
                size_t start = blk * kPantographBlockSize;
                size_t end = (start + kPantographBlockSize < half) ? start + kPantographBlockSize : half;
                double n = (double)(end - start);
                double mean_a = sum_a[blk] / n, mean_b = sum_b[blk] / n;
                double denom = sum_aa[blk] - n * mean_a * mean_a;
                double ratio = (std::abs(denom) > 1e-6) ? (sum_ab[blk] - n * mean_a * mean_b) / denom : 0.0;
                if (ratio > 8.0) ratio = 8.0;
                if (ratio < -8.0) ratio = -8.0;
                if (!std::isfinite(ratio)) ratio = 0.0;
                double offset_d = mean_b - ratio * mean_a;
                if (!std::isfinite(offset_d)) offset_d = 0.0;
                ratios[blk] = (i64)llround(ratio * (double)(1LL << kFixedShiftDev));
                offsets[blk] = (i32)llround(offset_d);
            }

            std::vector<long long> ratios_ll(ratios.begin(), ratios.end());
            CUDA_OK(cudaMemcpy(ratios_d, ratios_ll.data(), nblocks * sizeof(long long), cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(offsets_d, offsets.data(), nblocks * sizeof(int), cudaMemcpyHostToDevice));

            CUDA_OK(cudaMalloc(&next_low_d, half * sizeof(int)));
            CUDA_OK(cudaMalloc(&residual_d, half * sizeof(int)));

            int threads = 256;
            int blocks = (int)((half + threads - 1) / threads);
            if (blocks < 1) blocks = 1;
            if (blocks > 4096) blocks = 4096;
            transform_kernel<<<blocks, threads>>>(cur_d, half, ratios_d, offsets_d, next_low_d, residual_d);
            CUDA_OK(cudaGetLastError());
            CUDA_OK(cudaDeviceSynchronize());

            std::vector<i32> residual_host(half);
            CUDA_OK(cudaMemcpy(residual_host.data(), residual_d, half * sizeof(int), cudaMemcpyDeviceToHost));

            out.residuals.push_back(std::move(residual_host));
            out.block_ratios.push_back(std::move(ratios));
            out.block_offsets.push_back(std::move(offsets));

            CUDA_OK(cudaFree(cur_d));
            cur_d = next_low_d;
            next_low_d = nullptr;
            CUDA_OK(cudaFree(residual_d));
            residual_d = nullptr;

            cur_size = half;
        }

        i32 base_value = 0;
        CUDA_OK(cudaMemcpy(&base_value, cur_d, sizeof(int), cudaMemcpyDeviceToHost));
        out.base.push_back(base_value);
    }

    ok = true;

cuda_fail:
    if (cur_d) cudaFree(cur_d);
    if (sum_a_d) cudaFree(sum_a_d);
    if (sum_b_d) cudaFree(sum_b_d);
    if (sum_ab_d) cudaFree(sum_ab_d);
    if (sum_aa_d) cudaFree(sum_aa_d);
    if (ratios_d) cudaFree(ratios_d);
    if (offsets_d) cudaFree(offsets_d);
    if (next_low_d) cudaFree(next_low_d);
    if (residual_d) cudaFree(residual_d);
    return ok;
}

} // namespace csa
