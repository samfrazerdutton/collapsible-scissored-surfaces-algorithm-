// GPU-resident Pantograph Lift forward transform.
//
// The first version of this file did a small round trip to the host
// every level (copy calibration sums back, compute ratio/offset on the
// CPU, copy them back to the device) purely to do ~12 FLOPs of division
// and rounding per block. That's the same PCIe/launch-overhead-bound
// mistake this repo's design docs call out honestly elsewhere: with
// ~20-30 levels for a realistic input, that's dozens of small blocking
// transfers before any real work happens on the next level.
//
// This version keeps everything device-resident from the first upload to
// the last download: calibration sums, the tiny ratio/offset arithmetic
// (now its own kernel instead of round-tripping to the host), and the
// transform itself all run back-to-back in the same CUDA stream with no
// host synchronization in between -- the driver orders same-stream work
// automatically. Per-level results are written directly into pre-sized
// slices of three buffers (residuals, ratios, offsets) allocated once for
// the whole forward pass, so there is exactly one H2D transfer (the input)
// and a handful of D2H transfers (the three result buffers, once each) for
// the entire multi-level decomposition, regardless of how many levels it
// has.
#include "csa/pantograph_lift_cuda.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

namespace csa {

namespace {

#define CUDA_OK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__, __LINE__, cudaGetErrorName(_e), cudaGetErrorString(_e)); \
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
    if (i < half) {
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

// Turns each block's four sums into its ratio+offset, entirely on-device
// -- one thread per block, a dozen FLOPs each. This used to be a host
// round trip; it's now just another kernel in the same stream.
__global__ void compute_block_params_kernel(const double* sum_a, const double* sum_b,
                                             const double* sum_ab, const double* sum_aa,
                                             unsigned long long half,
                                             long long* ratios_out, int* offsets_out) {
    unsigned long long nblocks = (half + kBlockSizeDev - 1) / kBlockSizeDev;
    unsigned long long blk = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (blk >= nblocks) return;

    unsigned long long start = blk * (unsigned long long)kBlockSizeDev;
    unsigned long long end = (start + kBlockSizeDev < half) ? start + kBlockSizeDev : half;
    double n = (double)(end - start);

    double mean_a = sum_a[blk] / n, mean_b = sum_b[blk] / n;
    double denom = sum_aa[blk] - n * mean_a * mean_a;
    double ratio = (fabs(denom) > 1e-6) ? (sum_ab[blk] - n * mean_a * mean_b) / denom : 0.0;
    ratio = fmax(-8.0, fmin(8.0, ratio));
    if (!isfinite(ratio)) ratio = 0.0;
    double offset_d = mean_b - ratio * mean_a;
    if (!isfinite(offset_d)) offset_d = 0.0;

    ratios_out[blk] = (long long)llround(ratio * (double)(1LL << kFixedShiftDev));
    offsets_out[blk] = (int)llround(offset_d);
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

int launch_blocks_for(size_t n, int threads) {
    size_t b = (n + threads - 1) / threads;
    if (b < 1) b = 1;
    if (b > 65535u * 64u) b = 65535u * 64u; // generous cap, well under grid.x limits
    return (int)b;
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

    // Precompute the whole level structure up front: pair counts and block
    // counts for every level, deterministic from padded_len (mirroring the
    // same derivation codec.cpp uses on decode).
    size_t num_levels = 0;
    for (size_t n = padded_len; n > 1; n /= 2) num_levels++;

    std::vector<size_t> level_pairs(num_levels), level_blocks(num_levels);
    {
        size_t half = padded_len / 2;
        for (size_t lvl = 0; lvl < num_levels; lvl++) {
            level_pairs[lvl] = half;
            level_blocks[lvl] = (half + kPantographBlockSize - 1) / kPantographBlockSize;
            half /= 2;
        }
    }

    // Past a certain point, a level's total work is too small for three
    // more kernel launches' fixed dispatch overhead to pay for itself --
    // the tail of levels below kGpuTailCutoff elements is handed off to
    // the CPU path instead (already independently tested), rather than
    // paying that overhead many times over on levels doing almost no
    // work. This is a hand-off of *data*, not of the algorithm: the CPU
    // path recursively runs the identical Pantograph Lift on whatever
    // array the GPU stopped at, and its levels are simply appended after
    // the GPU-produced ones.
    constexpr size_t kGpuTailCutoff = 65536;
    size_t gpu_levels = 0;
    while (gpu_levels < num_levels && level_pairs[gpu_levels] * 2 > kGpuTailCutoff) gpu_levels++;

    std::vector<size_t> level_resid_off(gpu_levels), level_param_off(gpu_levels);
    size_t total_residuals = 0, total_blocks = 0, max_blocks_any_level = 0;
    for (size_t lvl = 0; lvl < gpu_levels; lvl++) {
        level_resid_off[lvl] = total_residuals;
        level_param_off[lvl] = total_blocks;
        total_residuals += level_pairs[lvl];
        total_blocks += level_blocks[lvl];
        if (level_blocks[lvl] > max_blocks_any_level) max_blocks_any_level = level_blocks[lvl];
    }

    int* bufA_d = nullptr;
    int* bufB_d = nullptr;
    double* sum_a_d = nullptr;
    double* sum_b_d = nullptr;
    double* sum_ab_d = nullptr;
    double* sum_aa_d = nullptr;
    long long* ratios_d = nullptr;
    int* offsets_d = nullptr;
    int* residual_d = nullptr;
    i32* residual_pinned = nullptr;
    long long* ratios_pinned = nullptr;
    i32* offsets_pinned = nullptr;
    bool ok = false;
    LiftResult tail; // populated later if gpu_levels < num_levels; declared here
                     // (before any goto-capable CUDA_OK call) so the jump to
                     // cuda_fail on an early failure doesn't skip its init.

    // A single-element input has num_levels == 0, making several of these
    // sizes 0; cudaMalloc(0) is not reliably well-defined across CUDA
    // versions, so always allocate at least one element (the corresponding
    // cudaMemcpy calls below use the real, possibly-0, sizes, which *is*
    // well-defined and simply copies nothing).
    auto alloc_n = [](size_t n) { return n > 0 ? n : (size_t)1; };

    CUDA_OK(cudaMalloc(&bufA_d, padded_len * sizeof(int)));
    CUDA_OK(cudaMalloc(&bufB_d, padded_len * sizeof(int)));
    CUDA_OK(cudaMemcpy(bufA_d, host_padded.data(), padded_len * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&sum_a_d, alloc_n(max_blocks_any_level) * sizeof(double)));
    CUDA_OK(cudaMalloc(&sum_b_d, alloc_n(max_blocks_any_level) * sizeof(double)));
    CUDA_OK(cudaMalloc(&sum_ab_d, alloc_n(max_blocks_any_level) * sizeof(double)));
    CUDA_OK(cudaMalloc(&sum_aa_d, alloc_n(max_blocks_any_level) * sizeof(double)));
    CUDA_OK(cudaMalloc(&ratios_d, alloc_n(total_blocks) * sizeof(long long)));
    CUDA_OK(cudaMalloc(&offsets_d, alloc_n(total_blocks) * sizeof(int)));
    CUDA_OK(cudaMalloc(&residual_d, alloc_n(total_residuals) * sizeof(int)));

    {
        int* cur_d = bufA_d;
        int* next_d = bufB_d;
        for (size_t lvl = 0; lvl < gpu_levels; lvl++) {
            size_t half = level_pairs[lvl];
            size_t nblocks = level_blocks[lvl];
            long long* ratios_here = ratios_d + level_param_off[lvl];
            int* offsets_here = offsets_d + level_param_off[lvl];
            int* residual_here = residual_d + level_resid_off[lvl];

            calibrate_blocks_kernel<<<(unsigned int)nblocks, kBlockSizeDev>>>(
                cur_d, half, sum_a_d, sum_b_d, sum_ab_d, sum_aa_d);
            CUDA_OK(cudaGetLastError());

            int param_threads = 256;
            int param_blocks = launch_blocks_for(nblocks, param_threads);
            compute_block_params_kernel<<<param_blocks, param_threads>>>(
                sum_a_d, sum_b_d, sum_ab_d, sum_aa_d, half, ratios_here, offsets_here);
            CUDA_OK(cudaGetLastError());

            int threads = 256;
            int blocks = launch_blocks_for(half, threads);
            transform_kernel<<<blocks, threads>>>(cur_d, half, ratios_here, offsets_here, next_d, residual_here);
            CUDA_OK(cudaGetLastError());

            int* tmp = cur_d;
            cur_d = next_d;
            next_d = tmp;
        }

        if (gpu_levels < num_levels) {
            size_t tail_size = (gpu_levels > 0) ? level_pairs[gpu_levels - 1] : padded_len;
            std::vector<i32> tail_host(tail_size);
            CUDA_OK(cudaMemcpy(tail_host.data(), cur_d, tail_size * sizeof(int), cudaMemcpyDeviceToHost));
            tail = pantograph_lift_forward(tail_host);
        } else {
            i32 base_value = 0;
            CUDA_OK(cudaMemcpy(&base_value, cur_d, sizeof(int), cudaMemcpyDeviceToHost));
            out.base.push_back(base_value);
        }
    }

    // Pinned (page-locked) host staging buffers make the final bulk
    // transfers meaningfully faster than the pageable std::vector storage
    // the old per-level code copied into directly.
    CUDA_OK(cudaMallocHost(&residual_pinned, alloc_n(total_residuals) * sizeof(i32)));
    CUDA_OK(cudaMallocHost(&ratios_pinned, alloc_n(total_blocks) * sizeof(long long)));
    CUDA_OK(cudaMallocHost(&offsets_pinned, alloc_n(total_blocks) * sizeof(i32)));
    CUDA_OK(cudaMemcpy(residual_pinned, residual_d, total_residuals * sizeof(i32), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(ratios_pinned, ratios_d, total_blocks * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(offsets_pinned, offsets_d, total_blocks * sizeof(i32), cudaMemcpyDeviceToHost));

    out.residuals.resize(gpu_levels);
    out.block_ratios.resize(gpu_levels);
    out.block_offsets.resize(gpu_levels);
    for (size_t lvl = 0; lvl < gpu_levels; lvl++) {
        out.residuals[lvl].assign(residual_pinned + level_resid_off[lvl],
                                   residual_pinned + level_resid_off[lvl] + level_pairs[lvl]);
        out.block_offsets[lvl].assign(offsets_pinned + level_param_off[lvl],
                                       offsets_pinned + level_param_off[lvl] + level_blocks[lvl]);
        out.block_ratios[lvl].resize(level_blocks[lvl]);
        for (size_t b = 0; b < level_blocks[lvl]; b++)
            out.block_ratios[lvl][b] = (i64)ratios_pinned[level_param_off[lvl] + b];
    }

    // Append the CPU-computed tail levels (if any) after the GPU ones,
    // and adopt its base value -- pantograph_lift_inverse doesn't care
    // which device produced which level, only that residuals/ratios/
    // offsets/base are present and in the right order, which this is.
    if (gpu_levels < num_levels) {
        for (auto& lvl_res : tail.residuals) out.residuals.push_back(std::move(lvl_res));
        for (auto& lvl_r : tail.block_ratios) out.block_ratios.push_back(std::move(lvl_r));
        for (auto& lvl_o : tail.block_offsets) out.block_offsets.push_back(std::move(lvl_o));
        out.base = tail.base;
    }

    ok = true;

cuda_fail:
    if (residual_pinned) cudaFreeHost(residual_pinned);
    if (ratios_pinned) cudaFreeHost(ratios_pinned);
    if (offsets_pinned) cudaFreeHost(offsets_pinned);
    if (bufA_d) cudaFree(bufA_d);
    if (bufB_d) cudaFree(bufB_d);
    if (sum_a_d) cudaFree(sum_a_d);
    if (sum_b_d) cudaFree(sum_b_d);
    if (sum_ab_d) cudaFree(sum_ab_d);
    if (sum_aa_d) cudaFree(sum_aa_d);
    if (ratios_d) cudaFree(ratios_d);
    if (offsets_d) cudaFree(offsets_d);
    if (residual_d) cudaFree(residual_d);
    return ok;
}

} // namespace csa
