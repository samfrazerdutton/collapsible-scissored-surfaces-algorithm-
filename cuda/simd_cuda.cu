// GPU implementation of max_abs_diff_i32 (see csa/simd_cuda.hpp) -- the
// third backend for the same reduction csa/simd.hpp implements as a
// scalar CPU reference and an AVX2 kernel. One-shot: allocates its own
// device buffers and frees them before returning, matching
// pantograph_lift_forward_cuda's simplicity (this kernel has no
// multi-level state to amortize across calls the way a CudaLiftSession
// is worth building for, so it doesn't get one).
#include "csa/simd_cuda.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

namespace csa {

namespace {

#define CUDA_OK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__, __LINE__, cudaGetErrorName(_e), cudaGetErrorString(_e)); \
    goto cuda_fail; } } while (0)

constexpr int kBlockSize = 256;

// Each thread reduces a strided run of elements into a local max, then a
// shared-memory tree reduction collapses one block's threads down to one
// value, and exactly one atomicMax per block updates the single global
// result -- so contention on *out is O(num_blocks), not O(n).
__global__ void max_abs_diff_kernel(const int* a, const int* b, size_t n, unsigned int* out) {
    __shared__ unsigned int sh[kBlockSize];

    unsigned int local = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)blockDim.x * gridDim.x) {
        // Widened to long long, exactly like the scalar CPU reference, so
        // this kernel shares that path's zero-precondition guarantee
        // rather than the AVX2 kernel's narrower documented one -- a GPU
        // thread pays for the same memory transaction regardless of
        // whether the arithmetic after it is 32-bit or 64-bit wide.
        long long d = (long long)a[i] - (long long)b[i];
        unsigned int ad = (unsigned int)(d < 0 ? -d : d);
        local = local > ad ? local : ad;
    }
    sh[threadIdx.x] = local;
    __syncthreads();

    for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sh[threadIdx.x] = max(sh[threadIdx.x], sh[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(out, sh[0]);
}

} // namespace

bool max_abs_diff_i32_cuda(const i32* a, const i32* b, size_t n, u32& out_max) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return false;
    out_max = 0;
    if (n == 0) return true;

    int* a_d = nullptr;
    int* b_d = nullptr;
    unsigned int* out_d = nullptr;
    bool ok = false;

    CUDA_OK(cudaMalloc(&a_d, n * sizeof(int)));
    CUDA_OK(cudaMalloc(&b_d, n * sizeof(int)));
    CUDA_OK(cudaMalloc(&out_d, sizeof(unsigned int)));
    CUDA_OK(cudaMemcpy(a_d, a, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b_d, b, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(out_d, 0, sizeof(unsigned int)));

    {
        int num_blocks = (int)std::min((size_t)4096, (n + kBlockSize - 1) / kBlockSize);
        if (num_blocks < 1) num_blocks = 1;
        max_abs_diff_kernel<<<num_blocks, kBlockSize>>>(a_d, b_d, n, out_d);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
    }

    {
        unsigned int host_max = 0;
        CUDA_OK(cudaMemcpy(&host_max, out_d, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        out_max = (u32)host_max;
    }
    ok = true;

cuda_fail:
    if (a_d) cudaFree(a_d);
    if (b_d) cudaFree(b_d);
    if (out_d) cudaFree(out_d);
    return ok;
}

} // namespace csa
