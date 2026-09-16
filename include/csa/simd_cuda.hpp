// csa/simd_cuda.hpp — optional GPU-accelerated max_abs_diff_i32, the same
// reduction include/csa/simd.hpp implements as a scalar CPU reference and
// (behind runtime dispatch) an AVX2 kernel. This is the third backend for
// the identical operation: max(|a[i]-b[i]|) over two int32 arrays, the
// hot loop cli/main.cpp uses to measure real geo2d/geo3d round-trip error.
//
// Unlike the AVX2 kernel (documented precondition: |a[i]-b[i]| fits in
// int32), this widens to i64 per element on the device, matching the
// scalar CPU reference's zero-precondition guarantee -- a GPU kernel pays
// for a global memory transaction per element regardless of the compute
// width, so there is no real throughput reason to take on AVX2's narrower
// contract here.
//
// Returns false (and leaves out_max unspecified) if no CUDA device is
// available at runtime, or CSA was built with WITH_CUDA=OFF, exactly like
// pantograph_lift_forward_cuda -- callers fall back to max_abs_diff_i32()
// (the CPU dispatcher in simd.hpp) transparently.
#pragma once
#include "csa/common.hpp"
#include <cstddef>

namespace csa {

bool max_abs_diff_i32_cuda(const i32* a, const i32* b, size_t n, u32& out_max);

} // namespace csa
