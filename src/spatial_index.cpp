#include "csa/spatial_index.hpp"
#include <algorithm>
#include <numeric>

namespace csa {

namespace {

i32 coord(const Point3i& p, int axis) {
    return axis == 0 ? p.x : (axis == 1 ? p.y : p.z);
}

// Squared Euclidean distance as a double, not i64: a straightforward i64
// accumulation (widen each axis difference, square, sum) can overflow --
// two i32 coordinates at opposite ends of the full int32 range differ by
// up to ~2^32, and squaring that alone reaches the u64/i64 overflow
// boundary before the three axes are even summed. Real point-cloud data
// this codec actually decodes never approaches that magnitude, but a
// general-purpose spatial index shouldn't rely on that to avoid
// undefined behavior -- especially not in a codebase that spent real
// effort this session finding and fixing exactly this class of overflow
// bug elsewhere (see docs/SANITIZERS.md). Doubles can't overflow at any
// of these magnitudes; the only cost is that a difference near the full
// int32 range loses some precision once squared (doubles exactly
// represent integers only up to 2^53), which affects nothing but nearest-
// neighbor tie-breaking in an already-extreme, unrealistic case -- not
// the final index results' correctness for any input this codec's own
// codec paths would ever actually produce.
double dist_sq(const Point3i& a, const Point3i& b) {
    double dx = (double)a.x - (double)b.x;
    double dy = (double)a.y - (double)b.y;
    double dz = (double)a.z - (double)b.z;
    return dx * dx + dy * dy + dz * dz;
}

} // namespace

void KdTree3i::build(const std::vector<Point3i>& points) {
    points_ = points;
    order_.resize(points_.size());
    std::iota(order_.begin(), order_.end(), 0u);
    build_range(0, order_.size(), 0);
}

void KdTree3i::build_range(size_t lo, size_t hi, int depth) {
    if (hi - lo <= 1) return;
    int axis = depth % 3;
    size_t mid = lo + (hi - lo) / 2;
    std::nth_element(order_.begin() + lo, order_.begin() + mid, order_.begin() + hi,
                      [&](u32 a, u32 b) { return coord(points_[a], axis) < coord(points_[b], axis); });
    build_range(lo, mid, depth + 1);
    build_range(mid + 1, hi, depth + 1);
}

std::vector<u32> KdTree3i::range_query(const Point3i& lo, const Point3i& hi) const {
    std::vector<u32> out;
    if (!order_.empty()) range_query_rec(0, order_.size(), 0, lo, hi, out);
    return out;
}

void KdTree3i::range_query_rec(size_t lo, size_t hi, int depth, const Point3i& qlo, const Point3i& qhi,
                                std::vector<u32>& out) const {
    if (lo >= hi) return;
    size_t mid = lo + (hi - lo) / 2;
    u32 idx = order_[mid];
    const Point3i& p = points_[idx];

    if (p.x >= qlo.x && p.x <= qhi.x && p.y >= qlo.y && p.y <= qhi.y && p.z >= qlo.z && p.z <= qhi.z)
        out.push_back(idx);

    int axis = depth % 3;
    i32 pc = coord(p, axis);
    if (coord(qlo, axis) <= pc) range_query_rec(lo, mid, depth + 1, qlo, qhi, out);
    if (coord(qhi, axis) >= pc) range_query_rec(mid + 1, hi, depth + 1, qlo, qhi, out);
}

std::vector<u32> KdTree3i::k_nearest(const Point3i& query, size_t k) const {
    std::vector<Candidate> heap; // max-heap by dist_sq: front() is the current worst-of-the-best
    if (k > 0 && !order_.empty()) knn_rec(0, order_.size(), 0, query, k, heap);

    std::sort(heap.begin(), heap.end(), [](const Candidate& a, const Candidate& b) { return a.dist_sq < b.dist_sq; });
    std::vector<u32> out;
    out.reserve(heap.size());
    for (const Candidate& c : heap) out.push_back(c.index);
    return out;
}

void KdTree3i::knn_rec(size_t lo, size_t hi, int depth, const Point3i& query, size_t k,
                        std::vector<Candidate>& heap) const {
    if (lo >= hi) return;
    size_t mid = lo + (hi - lo) / 2;
    u32 idx = order_[mid];
    const Point3i& p = points_[idx];
    double d = dist_sq(p, query);

    auto cmp = [](const Candidate& a, const Candidate& b) { return a.dist_sq < b.dist_sq; };
    if (heap.size() < k) {
        heap.push_back({d, idx});
        std::push_heap(heap.begin(), heap.end(), cmp);
    } else if (d < heap.front().dist_sq) {
        std::pop_heap(heap.begin(), heap.end(), cmp);
        heap.back() = {d, idx};
        std::push_heap(heap.begin(), heap.end(), cmp);
    }

    int axis = depth % 3;
    i32 pc = coord(p, axis);
    i32 qc = coord(query, axis);
    bool go_left_first = qc < pc;
    size_t near_lo = go_left_first ? lo : mid + 1, near_hi = go_left_first ? mid : hi;
    size_t far_lo = go_left_first ? mid + 1 : lo, far_hi = go_left_first ? hi : mid;

    knn_rec(near_lo, near_hi, depth + 1, query, k, heap);

    // Only worth exploring the far side if the heap isn't full yet, or the
    // splitting plane itself is closer than the current worst kept
    // candidate -- the standard KD-tree nearest-neighbor pruning rule.
    double axis_dist = (double)qc - (double)pc;
    double axis_dist_sq = axis_dist * axis_dist;
    if (heap.size() < k || axis_dist_sq < heap.front().dist_sq)
        knn_rec(far_lo, far_hi, depth + 1, query, k, heap);
}

} // namespace csa
