#include "csa/rod_joint_transform.hpp"
#include <algorithm>
#include <cmath>

namespace csa {

namespace {

inline i64 clampd(double v, double lo, double hi) {
    if (!std::isfinite(v)) return 0;
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return (i64)llround(v * (double)kFixedOne);
}

// complex multiply: (cr + i*ci) * (ex + i*ey), all fixed-point/int inputs,
// result rounded to nearest integer pair.
inline void complex_mul_round(i64 cr, i64 ci, i32 ex, i32 ey, i32& out_re, i32& out_im) {
    i64 re = fixed_mul_round(ex, cr) - fixed_mul_round(ey, ci);
    i64 im = fixed_mul_round(ey, cr) + fixed_mul_round(ex, ci);
    out_re = (i32)re;
    out_im = (i32)im;
}

// Least-squares calibration of c over rod pairs (e[i-1] -> e[i]) for
// i in [start, end) (0-indexed into ex/ey), skipping i==0 (the virtual
// zero seed rod, which carries no rotation information).
void calibrate_block(const std::vector<i32>& ex, const std::vector<i32>& ey,
                      size_t start, size_t end, i64& out_re, i64& out_im) {
    double sum_re = 0.0, sum_im = 0.0, sum_norm = 0.0;
    for (size_t i = std::max(start, size_t(1)); i < end; i++) {
        double pex = ex[i - 1], pey = ey[i - 1];
        double cex = ex[i], cey = ey[i];
        sum_re += pex * cex + pey * cey;
        sum_im += pex * cey - pey * cex;
        sum_norm += pex * pex + pey * pey;
    }
    double cr = 0.0, ci = 0.0;
    if (sum_norm > 1e-9) {
        cr = sum_re / sum_norm;
        ci = sum_im / sum_norm;
    }
    out_re = clampd(cr, -8.0, 8.0);
    out_im = clampd(ci, -8.0, 8.0);
}

} // namespace

RodJoint2DResult rod_joint_2d_forward(const std::vector<Point2i>& points) {
    RodJoint2DResult r;
    r.count = points.size();
    if (points.empty()) return r;
    r.anchor = points[0];
    size_t m = points.size();
    if (m < 2) return r;

    // Build rod (edge) vectors e[1..m-1] = P[i] - P[i-1].
    std::vector<i32> ex(m - 1), ey(m - 1);
    for (size_t i = 1; i < m; i++) {
        ex[i - 1] = points[i].x - points[i - 1].x;
        ey[i - 1] = points[i].y - points[i - 1].y;
    }

    size_t nrods = m - 1;
    size_t nblocks = (nrods + kRodJointBlockSize - 1) / kRodJointBlockSize;
    r.block_ratio_re.resize(nblocks);
    r.block_ratio_im.resize(nblocks);
    r.residual_x.resize(nrods);
    r.residual_y.resize(nrods);

    i32 prev_ex = 0, prev_ey = 0;
    for (size_t blk = 0; blk < nblocks; blk++) {
        size_t start = blk * kRodJointBlockSize;
        size_t end = std::min(nrods, start + kRodJointBlockSize);
        i64 cr, ci;
        calibrate_block(ex, ey, start, end, cr, ci);
        r.block_ratio_re[blk] = cr;
        r.block_ratio_im[blk] = ci;

        for (size_t i = start; i < end; i++) {
            i32 pred_re, pred_im;
            complex_mul_round(cr, ci, prev_ex, prev_ey, pred_re, pred_im);
            r.residual_x[i] = ex[i] - pred_re;
            r.residual_y[i] = ey[i] - pred_im;
            prev_ex = ex[i];
            prev_ey = ey[i];
        }
    }
    return r;
}

std::vector<Point2i> rod_joint_2d_inverse(const RodJoint2DResult& r) {
    std::vector<Point2i> points;
    if (r.count == 0) return points;
    points.resize((size_t)r.count);
    points[0] = r.anchor;
    if (r.count < 2) return points;

    i32 prev_ex = 0, prev_ey = 0;
    for (size_t i = 0; i < r.count - 1; i++) {
        size_t blk = i / kRodJointBlockSize;
        i64 cr = r.block_ratio_re[blk];
        i64 ci = r.block_ratio_im[blk];
        i32 pred_re, pred_im;
        complex_mul_round(cr, ci, prev_ex, prev_ey, pred_re, pred_im);
        i32 ex = pred_re + r.residual_x[i];
        i32 ey = pred_im + r.residual_y[i];
        points[i + 1].x = points[i].x + ex;
        points[i + 1].y = points[i].y + ey;
        prev_ex = ex;
        prev_ey = ey;
    }
    return points;
}

RodJoint3DResult rod_joint_3d_forward(const std::vector<Point3i>& points) {
    std::vector<Point2i> xy(points.size());
    std::vector<i32> zs(points.size());
    for (size_t i = 0; i < points.size(); i++) {
        xy[i] = {points[i].x, points[i].y};
        zs[i] = points[i].z;
    }
    RodJoint3DResult r;
    r.xy = rod_joint_2d_forward(xy);
    r.lift_z = pantograph_lift_forward(zs);
    return r;
}

std::vector<Point3i> rod_joint_3d_inverse(const RodJoint3DResult& r) {
    std::vector<Point2i> xy = rod_joint_2d_inverse(r.xy);
    std::vector<i32> zs = pantograph_lift_inverse(r.lift_z);
    std::vector<Point3i> points(xy.size());
    for (size_t i = 0; i < xy.size(); i++) {
        points[i] = {xy[i].x, xy[i].y, zs[i]};
    }
    return points;
}

} // namespace csa
