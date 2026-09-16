// csa/simd.hpp — runtime-dispatched SIMD for the one clean, honestly
// vectorizable hot kernel in this codebase's actual verification path:
// max(|a[i] - b[i]|) over two same-length i32 arrays. This exact
// reduction runs on every squeeze()/optimize()/benchmark round-trip
// check for geo2d/geo3d data (see cli/main.cpp) -- and inside
// cmd_optimize's binary search, up to ~20 times per call -- so it is a
// real hot path, not a synthetic microbenchmark invented to have
// something to vectorize.
//
// Scoped deliberately to the *contiguous, single-scale* case (geo2d/
// geo3d, where every element in the flattened array shares one scale
// and "one overall max error across all components" is exactly what the
// existing scalar loop already computes). Pose data's flat array is
// interleaved as [x,y,z,qw,qx,qy,qz] per row with two different scales
// for the two groups (see FORMAT.md) -- a stride-7 access pattern that
// doesn't divide evenly into AVX2's 8-wide i32 lanes without a gather
// (historically inconsistent performance across microarchitectures) or
// a de-interleaving copy (extra allocation + copy defeating some of the
// benefit). Not attempted in this pass; the pose error loops in
// cli/main.cpp remain scalar, honestly, rather than forcing a
// technique onto a shape it doesn't fit well.
//
// This is an integer max-reduction, not a floating-point sum: reduction
// order doesn't affect the result (max is associative/commutative over
// exact integers), so the AVX2 and scalar paths are required to produce
// bit-identical output for the same input -- verified directly in
// tests/test_main.cpp, not assumed from "it's just SIMD."
#pragma once
#include "csa/common.hpp"
#include <cstddef>

// True only when compiling for an actual x86/x64 target -- AVX2 is an
// x86 instruction-set extension, meaningless on ARM, and Emscripten's
// WASM target (this project's browser build; see docs/) is neither: it
// has no 256-bit vector hardware to lower AVX2 intrinsics onto, and
// __builtin_cpu_supports("avx2") is itself an x86-only builtin. Every
// declaration/definition below that's specific to the real AVX2 kernel
// is gated on this, so the WASM and any future ARM build simply never
// compile or reference it -- max_abs_diff_i32() below still works
// everywhere, it just always resolves to the scalar path off x86/x64.
#if !defined(__EMSCRIPTEN__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#define CSA_X86_SIMD 1
#else
#define CSA_X86_SIMD 0
#endif

namespace csa {

enum class SimdBackend { Scalar, AVX2 };

// What max_abs_diff_i32() below will actually use on this run, decided
// once via a real CPUID check (see simd.cpp), not assumed from the
// compiler's target flags -- a binary built with AVX2 codegen enabled
// can still run on a CPU that doesn't have it, so the dispatch has to be
// a runtime fact, not a compile-time one.
SimdBackend detect_simd_backend();

// Reference implementation. Always available, always correct, used
// directly when detect_simd_backend() reports Scalar and as the ground
// truth every correctness test compares the AVX2 path against.
u32 max_abs_diff_i32_scalar(const i32* a, const i32* b, size_t n);

#if CSA_X86_SIMD
// AVX2 implementation. Safe to call unconditionally -- internally
// checked cheaply enough it doesn't need callers to gate it themselves,
// but max_abs_diff_i32() below already does that via
// detect_simd_backend() so this is mainly exposed for the correctness
// test to call directly on hardware that has AVX2. Only declared at all
// on x86/x64 (see CSA_X86_SIMD above) -- there is nothing to call on
// WASM/ARM builds, by design, not by omission.
u32 max_abs_diff_i32_avx2(const i32* a, const i32* b, size_t n);
#endif

// The one callers should actually use: dispatches to AVX2 or the scalar
// fallback based on detect_simd_backend(), once per call (cheap; the
// CPUID result itself is cached after the first call).
u32 max_abs_diff_i32(const i32* a, const i32* b, size_t n);

} // namespace csa
