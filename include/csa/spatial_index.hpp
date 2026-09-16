// csa/spatial_index.hpp — a real KD-tree over decompressed Point3i data
// (the geo3d point-cloud shape this codec already round-trips), so a
// caller can run a spatial query (a bounding-box range query, or
// k-nearest-neighbors) against a decoded point cloud without a full
// linear scan every time. This is a query structure built *on top of*
// decompress_geo3d's output, not a change to the codec or wire format --
// nothing here affects compression, and no .csa file format changes.
//
// Scoped to Point3i (the geo3d shape) only. A Point2i equivalent would
// share the same algorithm structurally (a 2-axis cycle instead of a
// 3-axis one) but is not built here -- real, disclosed follow-up work,
// not a silent gap; see docs/SPATIAL_INDEX.md.
//
// Implicit, index-permutation-based tree (no per-node heap allocations,
// no pointers): the input points are copied once, in their original
// order, into `points_`; a separate permutation array `order_` is
// recursively partitioned via std::nth_element (median-of-range split,
// cycling x/y/z by depth) into an implicit balanced binary tree over
// `order_`'s index range -- the classic array-backed KD-tree layout.
// Every query result is reported as an index into the *original* input
// vector passed to build(), never an internal position, so callers don't
// need to know anything about how the tree stores things.
#pragma once
#include "csa/common.hpp"
#include "csa/rod_joint_transform.hpp" // Point3i
#include <cstddef>
#include <vector>

namespace csa {

class KdTree3i {
public:
    KdTree3i() = default;

    // Copies `points` and builds the tree. O(n log^2 n) (std::nth_element
    // is O(n) per level, log n levels). Safe to call again to rebuild
    // from a different point set; discards any previous tree.
    void build(const std::vector<Point3i>& points);

    size_t size() const { return points_.size(); }

    // All original indices i such that points_[i]'s x/y/z each fall
    // within [lo.<axis>, hi.<axis>] inclusive. Returned in no particular
    // order (a range query has no natural single ordering the way
    // nearest-neighbor does). Requires lo.<axis> <= hi.<axis> per axis --
    // not checked, since it's a precondition on the caller's own query
    // bounds, not on untrusted wire data.
    std::vector<u32> range_query(const Point3i& lo, const Point3i& hi) const;

    // The k original indices whose points are nearest `query` by
    // Euclidean distance, nearest first. Returns fewer than k only if
    // the tree itself holds fewer than k points. Squared distances are
    // computed as doubles, not accumulated in an integer type: two i32
    // coordinates at opposite ends of the full int32 range differ by up
    // to ~2^32, and squaring that alone reaches the i64/u64 overflow
    // boundary before the three axes are even summed. See
    // src/spatial_index.cpp's dist_sq() for the full reasoning.
    std::vector<u32> k_nearest(const Point3i& query, size_t k) const;

private:
    void build_range(size_t lo, size_t hi, int depth);
    void range_query_rec(size_t lo, size_t hi, int depth, const Point3i& qlo, const Point3i& qhi,
                          std::vector<u32>& out) const;

    struct Candidate { double dist_sq; u32 index; };
    void knn_rec(size_t lo, size_t hi, int depth, const Point3i& query, size_t k,
                 std::vector<Candidate>& heap) const;

    std::vector<Point3i> points_; // original order, never permuted
    std::vector<u32> order_;      // permutation of [0, points_.size()), recursively partitioned in place
};

} // namespace csa
