#include "csa/pantograph_lift.hpp"
#include <algorithm>
#include <cmath>

namespace csa {

namespace {

struct RatioOffset { i64 ratio; i32 offset; };

// Affine least-squares fit b ~= m*a + c over pairs [start, end) at this
// level. `m` is the calibrated pivot ratio (a pure scaling joint); `c` is
// a small additive correction so ordinary affine/ramp-like data (which a
// scaling-only joint cannot represent) still predicts well.
RatioOffset calibrate_range(const std::vector<i32>& cur, size_t start, size_t end) {
    double sum_a = 0.0, sum_b = 0.0, sum_ab = 0.0, sum_aa = 0.0;
    for (size_t i = start; i < end; i++) {
        double a = (double)cur[2 * i];
        double b = (double)cur[2 * i + 1];
        sum_a += a; sum_b += b; sum_ab += a * b; sum_aa += a * a;
    }
    double n = (double)(end - start);
    double mean_a = sum_a / n, mean_b = sum_b / n;
    double denom = sum_aa - n * mean_a * mean_a;
    double m = (std::abs(denom) > 1e-6) ? (sum_ab - n * mean_a * mean_b) / denom : 0.0;
    if (m > 8.0) m = 8.0;
    if (m < -8.0) m = -8.0;
    if (!std::isfinite(m)) m = 0.0;
    double c = mean_b - m * mean_a;
    if (!std::isfinite(c)) c = 0.0;

    RatioOffset ro;
    ro.ratio = (i64)llround(m * (double)kFixedOne);
    ro.offset = (i32)llround(c);
    return ro;
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

LiftResult pantograph_lift_forward(const std::vector<i32>& input) {
    LiftResult result;
    result.original_length = input.size();

    if (input.empty()) {
        return result;
    }

    size_t padded_len = next_pow2(input.size());
    std::vector<i32> cur(padded_len);
    for (size_t i = 0; i < input.size(); i++) cur[i] = input[i];
    for (size_t i = input.size(); i < padded_len; i++) cur[i] = input.back();

    while (cur.size() > 1) {
        size_t half = cur.size() / 2;
        size_t nblocks = num_blocks_for(half);
        std::vector<i64> ratios(nblocks);
        std::vector<i32> offsets(nblocks);
        std::vector<i32> next_low(half);
        std::vector<i32> residual(half);

        for (size_t blk = 0; blk < nblocks; blk++) {
            size_t start = blk * kPantographBlockSize;
            size_t end = std::min(half, start + kPantographBlockSize);
            RatioOffset ro = calibrate_range(cur, start, end);
            ratios[blk] = ro.ratio;
            offsets[blk] = ro.offset;

            for (size_t i = start; i < end; i++) {
                i32 a = cur[2 * i];
                i32 b = cur[2 * i + 1];
                i64 predicted = fixed_mul_round(a, ro.ratio) + ro.offset;
                i32 d = (i32)(b - predicted);
                i32 upd = d >> 1; // arithmetic shift: floor(d/2)
                i32 s = (i32)(a + upd);
                next_low[i] = s;
                residual[i] = d;
            }
        }

        result.residuals.push_back(std::move(residual));
        result.block_ratios.push_back(std::move(ratios));
        result.block_offsets.push_back(std::move(offsets));
        cur = std::move(next_low);
    }

    result.base = cur; // single element
    return result;
}

std::vector<i32> pantograph_lift_inverse(const LiftResult& result) {
    if (result.original_length == 0) return {};

    std::vector<i32> cur = result.base;
    int levels = (int)result.residuals.size();

    for (int level = levels - 1; level >= 0; level--) {
        const std::vector<i32>& residual = result.residuals[level];
        const std::vector<i64>& ratios = result.block_ratios[level];
        const std::vector<i32>& offsets = result.block_offsets[level];
        size_t half = cur.size();
        std::vector<i32> next(half * 2);

        for (size_t i = 0; i < half; i++) {
            size_t blk = i / kPantographBlockSize;
            i64 ratio = ratios[blk];
            i32 offset = offsets[blk];
            i32 s = cur[i];
            i32 d = residual[i];
            i32 upd = d >> 1;
            i32 a = (i32)(s - upd);
            i64 predicted = fixed_mul_round(a, ratio) + offset;
            i32 b = (i32)(predicted + d);
            next[2 * i] = a;
            next[2 * i + 1] = b;
        }
        cur = std::move(next);
    }

    cur.resize((size_t)result.original_length);
    return cur;
}

} // namespace csa
