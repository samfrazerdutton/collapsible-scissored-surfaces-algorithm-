#include "csa/simd.hpp"
#include <algorithm>

#if CSA_X86_SIMD
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace csa {

u32 max_abs_diff_i32_scalar(const i32* a, const i32* b, size_t n) {
    u32 result = 0;
    for (size_t i = 0; i < n; i++) {
        // Widened to i64 specifically so this reference has no precondition
        // on the input magnitudes at all (see simd_avx2.cpp's narrower,
        // documented precondition) -- this is the function every
        // correctness test and every non-x86 build actually trusts.
        i64 d = (i64)a[i] - (i64)b[i];
        u32 ad = (u32)(d < 0 ? -d : d);
        result = std::max(result, ad);
    }
    return result;
}

#if CSA_X86_SIMD
namespace {

// Real CPUID-based detection, not a compile-time assumption -- a binary
// built with AVX2 codegen enabled (this project's simd_avx2.cpp
// specifically) can still be *run* on a CPU that doesn't have it, so
// this has to be checked at runtime, once, and cached.
bool cpu_supports_avx2() {
#if defined(_MSC_VER)
    int regs[4];
    __cpuid(regs, 0);
    int max_leaf = regs[0];
    if (max_leaf < 7) return false;

    // AVX2 requires both the CPU to advertise it (leaf 7, EBX bit 5) and
    // the OS to have enabled XSAVE for the YMM register state (leaf 1,
    // ECX bits 27 [OSXSAVE] and 28 [AVX]) -- checking CPU support alone
    // without the OSXSAVE/XCR0 check is a real, documented way to crash
    // on an OS that hasn't opted into saving those registers, even on
    // hardware that physically supports AVX2.
    __cpuid(regs, 1);
    bool osxsave = (regs[2] & (1 << 27)) != 0;
    bool avx = (regs[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) return false;

    unsigned long long xcr0 = _xgetbv(0);
    bool os_saves_ymm = (xcr0 & 0x6) == 0x6; // XCR0[2:1] = SSE + AVX state
    if (!os_saves_ymm) return false;

    __cpuidex(regs, 7, 0);
    bool avx2 = (regs[1] & (1 << 5)) != 0;
    return avx2;
#else
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#endif
}

} // namespace
#endif // CSA_X86_SIMD

SimdBackend detect_simd_backend() {
#if CSA_X86_SIMD
    static const SimdBackend backend = cpu_supports_avx2() ? SimdBackend::AVX2 : SimdBackend::Scalar;
    return backend;
#else
    return SimdBackend::Scalar; // no AVX2 kernel exists to dispatch to on this target -- see CSA_X86_SIMD
#endif
}

u32 max_abs_diff_i32(const i32* a, const i32* b, size_t n) {
#if CSA_X86_SIMD
    return detect_simd_backend() == SimdBackend::AVX2 ? max_abs_diff_i32_avx2(a, b, n) : max_abs_diff_i32_scalar(a, b, n);
#else
    return max_abs_diff_i32_scalar(a, b, n);
#endif
}

} // namespace csa
