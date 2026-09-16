// The one AVX2-using translation unit in this codebase. Compiled with a
// wider instruction-set target than the rest of the project (see
// CMakeLists.txt: /arch:AVX2 on MSVC, -mavx2 elsewhere, applied to this
// file only) so the rest of the codebase's codegen is unaffected and
// still runs on CPUs without AVX2 -- runtime dispatch (csa::
// detect_simd_backend(), in simd.cpp) decides whether this function is
// ever called at all.
#include "csa/simd.hpp"

#if CSA_X86_SIMD
#include <algorithm>
#include <immintrin.h>

namespace csa {

// Precondition (see simd.hpp): |a[i] - b[i]| must fit in a 32-bit signed
// difference for every i -- i.e. a[i]-b[i] doesn't overflow int32_t. Real
// enough for this kernel's actual callers (geo2d/geo3d round-trip error
// between an encoded and decoded value, both produced by safe_scale()'s
// own int32-overflow-avoiding cap -- see cli/main.cpp) that this isn't a
// theoretical hand-wave: it's the same bound safe_scale() already
// enforces on its callers, just not re-derived here. _mm256_sub_epi32
// wraps silently on overflow with no trap, unlike the scalar reference
// below (which widens to i64 specifically to avoid needing this
// precondition at all) -- so this kernel is intentionally narrower than
// its scalar counterpart, not equally general. If this is ever reused
// for a caller that can't make this guarantee, use the scalar path.
u32 max_abs_diff_i32_avx2(const i32* a, const i32* b, size_t n) {
    __m256i vmax = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m256i diff = _mm256_sub_epi32(va, vb);
        __m256i absdiff = _mm256_abs_epi32(diff);
        vmax = _mm256_max_epi32(vmax, absdiff);
    }

    // Horizontal max of the 8 lanes in vmax.
    alignas(32) i32 lanes[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes), vmax);
    u32 result = 0;
    for (int j = 0; j < 8; j++) result = std::max(result, (u32)lanes[j]);

    // Scalar tail for n not a multiple of 8 -- identical arithmetic to
    // max_abs_diff_i32_scalar's loop body, just for the leftover elements.
    for (; i < n; i++) {
        i64 d = (i64)a[i] - (i64)b[i];
        u32 ad = (u32)(d < 0 ? -d : d);
        result = std::max(result, ad);
    }
    return result;
}

} // namespace csa
#endif // CSA_X86_SIMD
