// csa/adaptive_partition.hpp — variable-resolution block segmentation for
// the Rod-Joint/Quaternion-Joint family, layered on top of each
// transform's existing per-block calibration rather than replacing it.
//
// Motivation (see DESIGN.md and bench/pose_breakdown_tool.cpp): measuring
// per-block residual energy on real 6-DOF tracking data showed it varies
// a lot from block to block within a single file (coefficient of
// variation up to ~5-6 on real datasets) -- a fixed block size is forcing
// one calibration resolution onto stretches that are actually very
// different in how predictable they are (a camera held steady vs. a
// sudden jerk). Separately, calibration-parameter overhead was measured
// at under 5% of compressed size everywhere, so a hierarchical scheme
// aimed at shrinking *that* wouldn't have paid off -- this instead targets
// the *residual* bytes (95%+ of the total) by letting block boundaries
// adapt to where the data actually needs a fresh calibration.
//
// The mechanism is bottom-up greedy merging, not top-down splitting: an
// earlier top-down design (bisect, recurse only if the split helps) had a
// real failure mode -- if the exact midpoint doesn't land on a genuine
// regime change, both halves look equally "mixed" and the split gets
// rejected at the very first level, so it never recurses deep enough to
// find the real boundary even though one exists nearby. Bottom-up merging
// doesn't have that problem in principle: start at maximal resolution
// (every `min_block`-sized chunk is its own block) and greedily extend
// the current block by one more chunk at a time, left to right, so every
// candidate boundary gets tried directly instead of only ever the
// halfway point.
//
// The merge test itself needs care, though -- an earlier version of this
// compared the new chunk's cost against the *accumulated* block's total
// SSE (sse(whole so far) + sse(new chunk) vs. sse(merged)), which has its
// own failure mode: as the accumulated block grows large and easy, its
// own near-zero SSE stops contributing anything to the comparison, so
// admitting one bad chunk barely moves the ratio and merging never stops
// -- everything collapses into one giant block regardless of the data.
// The fix is to measure the *marginal* cost of forcing the new chunk
// specifically to share the merged calibration, not a comparison diluted
// by however much history is already accumulated: fit the merged
// calibration over the whole combined range, then score *only the new
// chunk's own slice* under that shared calibration, and compare that
// against the new chunk's independently-optimized SSE. That comparison
// doesn't get diluted by the accumulated block's size, so a genuine
// regime change is still caught even deep into a long, easy run.
//
// SSE is a cheap, already-proven proxy (the same one each transform's own
// per-block lag search already uses) for the real objective (entropy-
// coded bytes) -- and exactly like every other SSE-based search in this
// codebase, the *final* choice between this and the existing fixed-block
// scheme is decided by comparing actual serialized bytes (see codec.cpp),
// so an imperfect merge decision here can only cost ratio, never
// correctness or a regression below the fixed-block baseline.
#pragma once
#include "csa/common.hpp"
#include <algorithm>
#include <vector>

namespace csa {

template <typename Calib>
struct AdaptiveBlock {
    size_t start, end;
    Calib calib;
};

// CalibFn: (size_t start, size_t end) -> std::pair<Calib, double sse> --
// the best calibration for treating [start, end) as one block, and the
// SSE it achieves.
// ScoreFn: (size_t start, size_t end, const Calib&) -> double sse -- the
// SSE a *given* calibration achieves over [start, end), used to measure
// the new chunk's marginal cost under the candidate merged calibration.
// Both reuse each transform's own existing calibration+lag-search logic;
// this function only decides where to place boundaries.
template <typename Calib, typename CalibFn, typename ScoreFn>
std::vector<AdaptiveBlock<Calib>> adaptive_partition(size_t n, size_t min_block, double merge_ratio,
                                                      CalibFn&& calibrate, ScoreFn&& score) {
    std::vector<AdaptiveBlock<Calib>> out;
    if (n == 0) return out;
    if (min_block == 0) min_block = 1;

    size_t cur_start = 0;
    size_t cur_end = std::min(n, min_block);
    auto cur = calibrate(cur_start, cur_end);

    while (cur_end < n) {
        size_t next_end = std::min(n, cur_end + min_block);
        auto next = calibrate(cur_end, next_end);       // the candidate chunk, scored on its own
        auto merged = calibrate(cur_start, next_end);    // the whole combined range as one block
        // How much worse does the new chunk do if forced to share the
        // merged calibration instead of its own best one?
        double marginal_sse = score(cur_end, next_end, merged.first);
        bool merge_ok = marginal_sse <= next.second * merge_ratio;
        if (merge_ok) {
            cur_end = next_end;
            cur = merged;
        } else {
            out.push_back({cur_start, cur_end, cur.first});
            cur_start = cur_end;
            cur_end = next_end;
            cur = next;
        }
    }
    out.push_back({cur_start, cur_end, cur.first});
    return out;
}

} // namespace csa
