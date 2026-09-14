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

// Least-squares calibration of c over rod pairs (e[i-lag] -> e[i]) for
// i in [start, end) (0-indexed into ex/ey), skipping i < lag (rods with
// no rod that far back yet, which predict from a virtual zero).
void calibrate_block(const std::vector<i32>& ex, const std::vector<i32>& ey,
                      size_t start, size_t end, u32 lag, i64& out_re, i64& out_im) {
    double sum_re = 0.0, sum_im = 0.0, sum_norm = 0.0;
    for (size_t i = std::max(start, (size_t)lag); i < end; i++) {
        double pex = ex[i - lag], pey = ey[i - lag];
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

// Sum of squared true-rod prediction error a given (lag, cr, ci) achieves
// over [start, end) -- the score per-block lag search minimizes. Rods
// with i < lag (globally) have no predecessor and predict 0 (matching
// the actual encode loop's `prev = 0` case), so their own squared
// magnitude counts as error too; this keeps the score comparable across
// candidate lags that skip different numbers of rods at the very start
// of the sequence (only ever nonempty for the first block, since every
// later block's start already exceeds every candidate lag).
double block_predict_sse_2d(const std::vector<i32>& ex, const std::vector<i32>& ey,
                             size_t start, size_t end, u32 lag, i64 cr, i64 ci) {
    double sse = 0.0;
    for (size_t i = start; i < std::min(end, (size_t)lag); i++)
        sse += (double)ex[i] * ex[i] + (double)ey[i] * ey[i];
    for (size_t i = std::max(start, (size_t)lag); i < end; i++) {
        i32 pred_re, pred_im;
        complex_mul_round(cr, ci, ex[i - lag], ey[i - lag], pred_re, pred_im);
        double dx = (double)ex[i] - pred_re, dy = (double)ey[i] - pred_im;
        sse += dx * dx + dy * dy;
    }
    return sse;
}

// Picks the best lag for one calibration block: force_lag != 0 skips the
// search and uses that lag uniformly (see rod_joint_2d_forward's doc
// comment); force_lag == 0 tries every candidate and keeps whichever
// achieves the lowest prediction SSE.
u32 pick_block_lag_2d(const std::vector<i32>& ex, const std::vector<i32>& ey,
                       size_t start, size_t end, u32 force_lag, i64& out_re, i64& out_im) {
    if (force_lag != 0) {
        calibrate_block(ex, ey, start, end, force_lag, out_re, out_im);
        return force_lag;
    }
    u32 best_lag = kRodJointCandidateLags[0];
    double best_sse = -1.0;
    i64 best_re = 0, best_im = 0;
    for (u32 lag : kRodJointCandidateLags) {
        i64 cr, ci;
        calibrate_block(ex, ey, start, end, lag, cr, ci);
        double sse = block_predict_sse_2d(ex, ey, start, end, lag, cr, ci);
        if (best_sse < 0.0 || sse < best_sse) {
            best_sse = sse;
            best_lag = lag;
            best_re = cr;
            best_im = ci;
        }
    }
    out_re = best_re;
    out_im = best_im;
    return best_lag;
}

// 3x3 (row-major, Q16.16 fixed point) matrix-vector multiply, rounded to
// nearest integer.
inline void mat3_mul_round(const std::array<i64, 9>& m, i32 x, i32 y, i32 z,
                            i32& ox, i32& oy, i32& oz) {
    ox = (i32)(fixed_mul_round(x, m[0]) + fixed_mul_round(y, m[1]) + fixed_mul_round(z, m[2]));
    oy = (i32)(fixed_mul_round(x, m[3]) + fixed_mul_round(y, m[4]) + fixed_mul_round(z, m[5]));
    oz = (i32)(fixed_mul_round(x, m[6]) + fixed_mul_round(y, m[7]) + fixed_mul_round(z, m[8]));
}

// Horn's closed-form absolute-orientation method, specialized to fit a
// single similarity (rotation + uniform scale) predicting rod[i] from
// rod[i-lag] over i in [start, end). Rather than a general eigensolver, the
// optimal rotation quaternion is found as the dominant eigenvector of the
// symmetric 4x4 "profile" matrix N via shifted power iteration: N has
// trace 0 (so its eigenvalues aren't all one sign), so it is shifted by
// its Frobenius norm before iterating, which guarantees the eigenvalue we
// want (the most positive one) becomes the largest in magnitude too.
// Calibration quality only affects compression ratio -- forward/inverse
// use whatever matrix comes out, exactly, so this never affects
// correctness even in a degenerate/non-converged case.
std::array<i64, 9> calibrate_3d_block(const std::vector<i32>& ex, const std::vector<i32>& ey,
                                       const std::vector<i32>& ez, size_t start, size_t end, u32 lag) {
    double H[3][3] = {{0}};
    for (size_t i = std::max(start, (size_t)lag); i < end; i++) {
        double p[3] = {(double)ex[i - lag], (double)ey[i - lag], (double)ez[i - lag]};
        double c[3] = {(double)ex[i], (double)ey[i], (double)ez[i]};
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                H[a][b] += p[a] * c[b];
    }

    double Sxx = H[0][0], Sxy = H[0][1], Sxz = H[0][2];
    double Syx = H[1][0], Syy = H[1][1], Syz = H[1][2];
    double Szx = H[2][0], Szy = H[2][1], Szz = H[2][2];

    double N[4][4] = {
        {Sxx + Syy + Szz, Syz - Szy,        Szx - Sxz,        Sxy - Syx},
        {Syz - Szy,       Sxx - Syy - Szz,  Sxy + Syx,        Szx + Sxz},
        {Szx - Sxz,       Sxy + Syx,       -Sxx + Syy - Szz,  Syz + Szy},
        {Sxy - Syx,       Szx + Sxz,        Syz + Szy,       -Sxx - Syy + Szz},
    };

    double frob = 0.0;
    for (int a = 0; a < 4; a++)
        for (int b = 0; b < 4; b++)
            frob += N[a][b] * N[a][b];
    frob = std::sqrt(frob);
    for (int a = 0; a < 4; a++) N[a][a] += frob; // shift: guarantees a non-negative spectrum

    double v[4] = {1.0, 1.0, 1.0, 1.0};
    for (int iter = 0; iter < 40; iter++) {
        double nv[4] = {0, 0, 0, 0};
        for (int a = 0; a < 4; a++)
            for (int b = 0; b < 4; b++)
                nv[a] += N[a][b] * v[b];
        double norm = std::sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2] + nv[3] * nv[3]);
        if (norm < 1e-12) { v[0] = 1; v[1] = v[2] = v[3] = 0; break; }
        for (int a = 0; a < 4; a++) v[a] = nv[a] / norm;
    }

    double q0 = v[0], q1 = v[1], q2 = v[2], q3 = v[3];
    double qn = std::sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    std::array<i64, 9> m{};
    if (qn < 1e-12) {
        // Degenerate (e.g. an all-zero block): identity, scale 0 -> pure
        // "predict nothing", still perfectly correctable via the residual.
        m = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        return m;
    }
    q0 /= qn; q1 /= qn; q2 /= qn; q3 /= qn;

    double R[3][3] = {
        {q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2-q0*q3),         2*(q1*q3+q0*q2)},
        {2*(q1*q2+q0*q3),         q0*q0-q1*q1+q2*q2-q3*q3, 2*(q2*q3-q0*q1)},
        {2*(q1*q3-q0*q2),         2*(q2*q3+q0*q1),         q0*q0-q1*q1-q2*q2+q3*q3},
    };

    // Optimal uniform scale given R (Umeyama): s = sum(cur . R*prev) / sum(|prev|^2).
    double num = 0.0, den = 0.0;
    for (size_t i = std::max(start, (size_t)lag); i < end; i++) {
        double p[3] = {(double)ex[i - lag], (double)ey[i - lag], (double)ez[i - lag]};
        double c[3] = {(double)ex[i], (double)ey[i], (double)ez[i]};
        double rp[3];
        for (int a = 0; a < 3; a++) rp[a] = R[a][0]*p[0] + R[a][1]*p[1] + R[a][2]*p[2];
        num += c[0]*rp[0] + c[1]*rp[1] + c[2]*rp[2];
        den += p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
    }
    double s = (den > 1e-9) ? (num / den) : 0.0;
    if (s > 8.0) s = 8.0;
    if (s < -8.0) s = -8.0;
    if (!std::isfinite(s)) s = 0.0;

    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
            m[a * 3 + b] = clampd(s * R[a][b], -8.0, 8.0);
    return m;
}

// 3D counterpart of block_predict_sse_2d.
double block_predict_sse_3d(const std::vector<i32>& ex, const std::vector<i32>& ey, const std::vector<i32>& ez,
                             size_t start, size_t end, u32 lag, const std::array<i64, 9>& M) {
    double sse = 0.0;
    for (size_t i = start; i < std::min(end, (size_t)lag); i++)
        sse += (double)ex[i] * ex[i] + (double)ey[i] * ey[i] + (double)ez[i] * ez[i];
    for (size_t i = std::max(start, (size_t)lag); i < end; i++) {
        i32 px, py, pz;
        mat3_mul_round(M, ex[i - lag], ey[i - lag], ez[i - lag], px, py, pz);
        double dx = (double)ex[i] - px, dy = (double)ey[i] - py, dz = (double)ez[i] - pz;
        sse += dx * dx + dy * dy + dz * dz;
    }
    return sse;
}

// 3D counterpart of pick_block_lag_2d.
u32 pick_block_lag_3d(const std::vector<i32>& ex, const std::vector<i32>& ey, const std::vector<i32>& ez,
                       size_t start, size_t end, u32 force_lag, std::array<i64, 9>& out_M) {
    if (force_lag != 0) {
        out_M = calibrate_3d_block(ex, ey, ez, start, end, force_lag);
        return force_lag;
    }
    u32 best_lag = kRodJointCandidateLags[0];
    double best_sse = -1.0;
    std::array<i64, 9> best_M{};
    for (u32 lag : kRodJointCandidateLags) {
        std::array<i64, 9> M = calibrate_3d_block(ex, ey, ez, start, end, lag);
        double sse = block_predict_sse_3d(ex, ey, ez, start, end, lag, M);
        if (best_sse < 0.0 || sse < best_sse) {
            best_sse = sse;
            best_lag = lag;
            best_M = M;
        }
    }
    out_M = best_M;
    return best_lag;
}

} // namespace

RodJoint2DResult rod_joint_2d_forward(const std::vector<Point2i>& points, u32 force_lag,
                                       u32 quant_step, u32 resync_interval) {
    RodJoint2DResult r;
    r.count = points.size();
    if (points.empty()) return r;
    r.anchor = points[0];
    r.quant_step = (quant_step == 0) ? 1 : quant_step;
    r.resync_interval = resync_interval;
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
    r.block_lag.resize(nblocks);
    r.block_ratio_re.resize(nblocks);
    r.block_ratio_im.resize(nblocks);
    r.residual_x.resize(nrods);
    r.residual_y.resize(nrods);

    // Reconstructed (possibly lossy) rod history -- prediction is always
    // based on this, never on the true `ex`/`ey`, so the encoder stays in
    // lockstep with what the decoder will independently reconstruct. For
    // quant_step == 1 this is provably identical to `ex`/`ey` (see the
    // header comment), so lossless behavior is unchanged.
    std::vector<i32> rec_ex(nrods), rec_ey(nrods);

    // Running reconstructed absolute position (what a decoder would have
    // accumulated so far). A resync rod's target is *not* the true rod
    // ex[i]/ey[i] -- using that would only stop drift from growing
    // further, it would not undo drift already accumulated from earlier
    // lossy rods, since it would still be added to an already-off
    // rec_p{x,y}. The target that actually lands exactly on the true
    // absolute point points[i+1] is (points[i+1] - rec_p), computed
    // against whatever rec_p currently is.
    i64 rec_px = points[0].x, rec_py = points[0].y;

    for (size_t blk = 0; blk < nblocks; blk++) {
        size_t start = blk * kRodJointBlockSize;
        size_t end = std::min(nrods, start + kRodJointBlockSize);
        i64 cr, ci;
        // Calibration (and lag selection) may use true rods; only
        // prediction needs closed-loop reconstructed history.
        u32 lag = pick_block_lag_2d(ex, ey, start, end, force_lag, cr, ci);
        r.block_lag[blk] = lag;
        r.block_ratio_re[blk] = cr;
        r.block_ratio_im[blk] = ci;

        for (size_t i = start; i < end; i++) {
            i32 prev_ex = (i >= lag) ? rec_ex[i - lag] : 0;
            i32 prev_ey = (i >= lag) ? rec_ey[i - lag] : 0;
            i32 pred_re, pred_im;
            complex_mul_round(cr, ci, prev_ex, prev_ey, pred_re, pred_im);

            bool is_resync = (r.resync_interval > 0) && (((i + 1) % r.resync_interval) == 0);
            u32 q_here = is_resync ? 1 : r.quant_step;

            i64 target_dx = is_resync ? ((i64)points[i + 1].x - rec_px) : (i64)ex[i];
            i64 target_dy = is_resync ? ((i64)points[i + 1].y - rec_py) : (i64)ey[i];
            i64 raw_dx = target_dx - pred_re;
            i64 raw_dy = target_dy - pred_im;

            i64 qx = quant_round_div(raw_dx, (i64)q_here);
            i64 qy = quant_round_div(raw_dy, (i64)q_here);
            r.residual_x[i] = (i32)qx;
            r.residual_y[i] = (i32)qy;

            rec_ex[i] = (i32)(pred_re + qx * (i64)q_here);
            rec_ey[i] = (i32)(pred_im + qy * (i64)q_here);
            rec_px += rec_ex[i];
            rec_py += rec_ey[i];
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

    size_t nrods = (size_t)r.count - 1;
    u32 quant_step = (r.quant_step == 0) ? 1 : r.quant_step;
    std::vector<i32> rec_ex(nrods), rec_ey(nrods);
    for (size_t i = 0; i < nrods; i++) {
        size_t blk = i / kRodJointBlockSize;
        u32 lag = r.block_lag[blk];
        i64 cr = r.block_ratio_re[blk];
        i64 ci = r.block_ratio_im[blk];
        i32 prev_ex = (i >= lag) ? rec_ex[i - lag] : 0;
        i32 prev_ey = (i >= lag) ? rec_ey[i - lag] : 0;
        i32 pred_re, pred_im;
        complex_mul_round(cr, ci, prev_ex, prev_ey, pred_re, pred_im);

        bool is_resync = (r.resync_interval > 0) && (((i + 1) % r.resync_interval) == 0);
        u32 q_here = is_resync ? 1 : quant_step;

        i64 qx = r.residual_x[i];
        i64 qy = r.residual_y[i];
        rec_ex[i] = (i32)(pred_re + qx * (i64)q_here);
        rec_ey[i] = (i32)(pred_im + qy * (i64)q_here);

        points[i + 1].x = points[i].x + rec_ex[i];
        points[i + 1].y = points[i].y + rec_ey[i];
    }
    return points;
}

RodJoint3DResult rod_joint_3d_forward(const std::vector<Point3i>& points, u32 xy_force_lag,
                                       u32 xy_quant_step, u32 xy_resync_interval, u32 z_quant_step) {
    std::vector<Point2i> xy(points.size());
    std::vector<i32> zs(points.size());
    for (size_t i = 0; i < points.size(); i++) {
        xy[i] = {points[i].x, points[i].y};
        zs[i] = points[i].z;
    }
    RodJoint3DResult r;
    r.xy = rod_joint_2d_forward(xy, xy_force_lag, xy_quant_step, xy_resync_interval);
    r.lift_z = pantograph_lift_forward(zs, z_quant_step);
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

RodJoint3DSimResult rod_joint_3d_similarity_forward(const std::vector<Point3i>& points, u32 force_lag,
                                                     u32 quant_step, u32 resync_interval) {
    RodJoint3DSimResult r;
    r.count = points.size();
    if (points.empty()) return r;
    r.anchor = points[0];
    r.quant_step = (quant_step == 0) ? 1 : quant_step;
    r.resync_interval = resync_interval;
    size_t m = points.size();
    if (m < 2) return r;

    std::vector<i32> ex(m - 1), ey(m - 1), ez(m - 1);
    for (size_t i = 1; i < m; i++) {
        ex[i - 1] = points[i].x - points[i - 1].x;
        ey[i - 1] = points[i].y - points[i - 1].y;
        ez[i - 1] = points[i].z - points[i - 1].z;
    }

    size_t nrods = m - 1;
    size_t nblocks = (nrods + kRodJoint3DBlockSize - 1) / kRodJoint3DBlockSize;
    r.block_lag.resize(nblocks);
    r.block_matrix.resize(nblocks);
    r.residual_x.resize(nrods);
    r.residual_y.resize(nrods);
    r.residual_z.resize(nrods);

    // Reconstructed (possibly lossy) rod history, used for prediction --
    // see RodJoint2DResult's forward comment for why. quant_step == 1
    // makes this provably identical to ex/ey/ez.
    std::vector<i32> rec_ex(nrods), rec_ey(nrods), rec_ez(nrods);
    i64 rec_px = points[0].x, rec_py = points[0].y, rec_pz = points[0].z;

    for (size_t blk = 0; blk < nblocks; blk++) {
        size_t start = blk * kRodJoint3DBlockSize;
        size_t end = std::min(nrods, start + kRodJoint3DBlockSize);
        std::array<i64, 9> M{};
        u32 lag = pick_block_lag_3d(ex, ey, ez, start, end, force_lag, M);
        r.block_lag[blk] = lag;
        r.block_matrix[blk] = M;

        for (size_t i = start; i < end; i++) {
            i32 prev_x = (i >= lag) ? rec_ex[i - lag] : 0;
            i32 prev_y = (i >= lag) ? rec_ey[i - lag] : 0;
            i32 prev_z = (i >= lag) ? rec_ez[i - lag] : 0;
            i32 px, py, pz;
            mat3_mul_round(M, prev_x, prev_y, prev_z, px, py, pz);

            bool is_resync = (r.resync_interval > 0) && (((i + 1) % r.resync_interval) == 0);
            u32 q_here = is_resync ? 1 : r.quant_step;

            i64 target_dx = is_resync ? ((i64)points[i + 1].x - rec_px) : (i64)ex[i];
            i64 target_dy = is_resync ? ((i64)points[i + 1].y - rec_py) : (i64)ey[i];
            i64 target_dz = is_resync ? ((i64)points[i + 1].z - rec_pz) : (i64)ez[i];
            i64 raw_dx = target_dx - px;
            i64 raw_dy = target_dy - py;
            i64 raw_dz = target_dz - pz;

            i64 qx = quant_round_div(raw_dx, (i64)q_here);
            i64 qy = quant_round_div(raw_dy, (i64)q_here);
            i64 qz = quant_round_div(raw_dz, (i64)q_here);
            r.residual_x[i] = (i32)qx;
            r.residual_y[i] = (i32)qy;
            r.residual_z[i] = (i32)qz;

            rec_ex[i] = (i32)(px + qx * (i64)q_here);
            rec_ey[i] = (i32)(py + qy * (i64)q_here);
            rec_ez[i] = (i32)(pz + qz * (i64)q_here);
            rec_px += rec_ex[i];
            rec_py += rec_ey[i];
            rec_pz += rec_ez[i];
        }
    }
    return r;
}

std::vector<Point3i> rod_joint_3d_similarity_inverse(const RodJoint3DSimResult& r) {
    std::vector<Point3i> points;
    if (r.count == 0) return points;
    points.resize((size_t)r.count);
    points[0] = r.anchor;
    if (r.count < 2) return points;

    size_t nrods = (size_t)r.count - 1;
    u32 quant_step = (r.quant_step == 0) ? 1 : r.quant_step;
    std::vector<i32> rec_ex(nrods), rec_ey(nrods), rec_ez(nrods);
    for (size_t i = 0; i < nrods; i++) {
        size_t blk = i / kRodJoint3DBlockSize;
        u32 lag = r.block_lag[blk];
        const std::array<i64, 9>& M = r.block_matrix[blk];
        i32 prev_x = (i >= lag) ? rec_ex[i - lag] : 0;
        i32 prev_y = (i >= lag) ? rec_ey[i - lag] : 0;
        i32 prev_z = (i >= lag) ? rec_ez[i - lag] : 0;
        i32 px, py, pz;
        mat3_mul_round(M, prev_x, prev_y, prev_z, px, py, pz);

        bool is_resync = (r.resync_interval > 0) && (((i + 1) % r.resync_interval) == 0);
        u32 q_here = is_resync ? 1 : quant_step;

        i64 qx = r.residual_x[i];
        i64 qy = r.residual_y[i];
        i64 qz = r.residual_z[i];
        rec_ex[i] = (i32)(px + qx * (i64)q_here);
        rec_ey[i] = (i32)(py + qy * (i64)q_here);
        rec_ez[i] = (i32)(pz + qz * (i64)q_here);

        points[i + 1].x = points[i].x + rec_ex[i];
        points[i + 1].y = points[i].y + rec_ey[i];
        points[i + 1].z = points[i].z + rec_ez[i];
    }
    return points;
}

} // namespace csa
