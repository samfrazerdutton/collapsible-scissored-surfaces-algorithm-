// csa/pantograph_lift.hpp — the general-purpose "Pantograph Lift" transform.
//
// This is the data-domain analogue of a scissor/pantograph linkage: the
// Harvard/Tokyo "collapsible scissored surfaces" work showed that the whole
// geometry of a deployed surface can be encoded as a small set of local
// design parameters (fixed pivot ratios) plus a local additive construction
// rule, so the structure can collapse to a compact line and re-deploy
// exactly. Here, each decomposition level plays the role of one row of
// scissor units sharing a single calibrated pivot ratio: it predicts one
// half of the data from the other via that ratio, and only the exact
// integer residual is stored. Recursing halves the array each level, just
// like the linkage's compact collapsed line encodes an entire unfolded
// surface via a handful of numbers.
//
// The transform is a standard reversible integer lifting scheme (predict +
// update, à la LeGall 5/3) reparametrized so the "predict" step uses a
// calibrated affine fit (ratio + offset) instead of a fixed +1/-1
// coefficient: predicted = round(ratio * a) + offset. The ratio captures
// proportional/self-similar structure (the literal pantograph pivot
// ratio); the offset is a small additive correction so an ordinary affine
// sequence (b ~= a + constant) — which a pure scaling joint cannot
// represent — is still predicted well.
//
// Calibration is done per fixed-size block (see kBlockSize) rather than
// once for an entire level: a single global ratio/offset can only track
// data that is affine across the *whole* array, which fails for anything
// that curves (e.g. one period of a sine wave). A pivot ratio recalibrated
// every kBlockSize samples still keeps the parameter count tiny relative
// to the data (matching the source paper's "small set of design
// parameters" property) while tracking local curvature. Because the
// residual is always `actual - predicted`, correctness never depends on
// the calibration being good — only the compression ratio does.
#pragma once
#include "csa/common.hpp"
#include <vector>

namespace csa {

constexpr size_t kPantographBlockSize = 1024; // pairs per calibration block

// Lossless and lossy share one code path, the same unification every
// other lossy mode in this codebase uses: a level's residual is
// quantized to the nearest multiple of `quant_step` (q=1 is exact, so
// lossless is simply the q=1 special case). The quantized *index*
// (residual/q, rounded) is what's actually stored in `residuals`, and
// critically, the "update" half that becomes the *next* level's input
// (`s = a + upd`) is computed from the quantized/reconstructed residual,
// never the true one -- the same closed-loop DPCM design Rod-Joint's
// lossy mode uses, generalized across this transform's O(log n) cascaded
// levels instead of one flat rod sequence. Every level shares the same
// quant_step; there's no per-level resync mechanism the way Rod-Joint has
// one, because the failure mode resync exists to bound (drift
// accumulating over a long flat sequence of rods) doesn't have a direct
// analogue here -- error here can only compound across O(log n) levels,
// not across the sequence length, a fundamentally smaller and
// differently-shaped effect (see DESIGN.md for what was actually
// measured).
struct LiftResult {
    std::vector<i32> base;                          // final single (or few) low-pass value(s)
    std::vector<std::vector<i32>> residuals;         // per level, finest (level 0) first; quantized index when quant_step > 1
    std::vector<std::vector<i64>> block_ratios;      // per level, per block, Q16.16 fixed-point
    std::vector<std::vector<i32>> block_offsets;     // per level, per block, integer intercept
    u64 original_length = 0;                         // length before padding
    u32 quant_step = 1;                              // 1 == lossless
};

LiftResult pantograph_lift_forward(const std::vector<i32>& input, u32 quant_step = 1);
std::vector<i32> pantograph_lift_inverse(const LiftResult& result);

} // namespace csa
