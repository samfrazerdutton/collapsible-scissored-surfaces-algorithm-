#include "csa/quaternion_joint.hpp"
#include "csa/adaptive_partition.hpp"
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

// Predicted = a (x) D (Hamilton product, D applied by right-multiplication),
// with `a` raw integer quaternion components and D given as a Q16.16
// fixed-point 4-vector -- see quaternion_joint.hpp's header comment for
// the R(a)*D matrix form this implements row by row, matching
// rod_joint_transform.cpp's mat3_mul_round/complex_mul_round style
// exactly (fixed_mul_round pairs one raw-integer operand with one
// Q16.16 operand, result rounded to the nearest integer).
inline void quat_right_mul_round(i32 aw, i32 ax, i32 ay, i32 az, const std::array<i64, 4>& d,
                                  i32& ow, i32& ox, i32& oy, i32& oz) {
    ow = (i32)(fixed_mul_round(aw, d[0]) - fixed_mul_round(ax, d[1]) - fixed_mul_round(ay, d[2]) - fixed_mul_round(az, d[3]));
    ox = (i32)(fixed_mul_round(ax, d[0]) + fixed_mul_round(aw, d[1]) + fixed_mul_round(az, d[2]) - fixed_mul_round(ay, d[3]));
    oy = (i32)(fixed_mul_round(ay, d[0]) - fixed_mul_round(az, d[1]) + fixed_mul_round(aw, d[2]) + fixed_mul_round(ax, d[3]));
    oz = (i32)(fixed_mul_round(az, d[0]) + fixed_mul_round(ay, d[1]) - fixed_mul_round(ax, d[2]) + fixed_mul_round(aw, d[3]));
}

// Closed-form least-squares fit of D over quaternion pairs (a=quats[i-lag],
// b=quats[i]) for i in [start, end) -- see quaternion_joint.hpp's header
// comment for the derivation (R(a)^T R(a) = |a|^2 I collapses the normal
// equations to a scalar divide, no matrix inversion needed).
std::array<i64, 4> calibrate_block(const std::vector<i32>& qw, const std::vector<i32>& qx,
                                    const std::vector<i32>& qy, const std::vector<i32>& qz,
                                    size_t start, size_t end, u32 lag) {
    double acc[4] = {0.0, 0.0, 0.0, 0.0};
    double sum_norm2 = 0.0;
    for (size_t i = std::max(start, (size_t)lag); i < end; i++) {
        double aw = qw[i - lag], ax = qx[i - lag], ay = qy[i - lag], az = qz[i - lag];
        double bw = qw[i], bx = qx[i], by = qy[i], bz = qz[i];
        // R(a)^T * b, from the transpose of R(a) (see header comment).
        acc[0] += aw * bw + ax * bx + ay * by + az * bz;
        acc[1] += -ax * bw + aw * bx - az * by + ay * bz;
        acc[2] += -ay * bw + az * bx + aw * by - ax * bz;
        acc[3] += -az * bw - ay * bx + ax * by + aw * bz;
        sum_norm2 += aw * aw + ax * ax + ay * ay + az * az;
    }
    std::array<i64, 4> d{};
    double dw = 0.0, dx = 0.0, dy = 0.0, dz = 0.0;
    if (sum_norm2 > 1e-9) {
        dw = acc[0] / sum_norm2;
        dx = acc[1] / sum_norm2;
        dy = acc[2] / sum_norm2;
        dz = acc[3] / sum_norm2;
    }
    // D is dimensionless (order 1 for a near-unit-quaternion input, see
    // header comment), so the same +-8 safety clamp Rod-Joint uses for its
    // own dimensionless ratio/matrix is appropriate here too.
    d[0] = clampd(dw, -8.0, 8.0);
    d[1] = clampd(dx, -8.0, 8.0);
    d[2] = clampd(dy, -8.0, 8.0);
    d[3] = clampd(dz, -8.0, 8.0);
    return d;
}

// Sum of squared true-quaternion prediction error a given (lag, D)
// achieves over [start, end) -- mirrors rod_joint_transform.cpp's
// block_predict_sse_2d/3d exactly, including the "no predecessor yet"
// convention for i < lag (predicts 0, so its own squared magnitude counts
// as error -- only ever nonempty in the first block).
double block_predict_sse(const std::vector<i32>& qw, const std::vector<i32>& qx,
                          const std::vector<i32>& qy, const std::vector<i32>& qz,
                          size_t start, size_t end, u32 lag, const std::array<i64, 4>& d) {
    double sse = 0.0;
    for (size_t i = start; i < std::min(end, (size_t)lag); i++)
        sse += (double)qw[i] * qw[i] + (double)qx[i] * qx[i] + (double)qy[i] * qy[i] + (double)qz[i] * qz[i];
    for (size_t i = std::max(start, (size_t)lag); i < end; i++) {
        i32 pw, px, py, pz;
        quat_right_mul_round(qw[i - lag], qx[i - lag], qy[i - lag], qz[i - lag], d, pw, px, py, pz);
        double dw = (double)qw[i] - pw, dx = (double)qx[i] - px, dy = (double)qy[i] - py, dz = (double)qz[i] - pz;
        sse += dw * dw + dx * dx + dy * dy + dz * dz;
    }
    return sse;
}

// Picks the best lag for one calibration block: force_lag != 0 skips the
// search (see quaternion_joint_forward's doc comment).
u32 pick_block_lag(const std::vector<i32>& qw, const std::vector<i32>& qx,
                    const std::vector<i32>& qy, const std::vector<i32>& qz,
                    size_t start, size_t end, u32 force_lag, std::array<i64, 4>& out_d) {
    if (force_lag != 0) {
        out_d = calibrate_block(qw, qx, qy, qz, start, end, force_lag);
        return force_lag;
    }
    u32 best_lag = kRodJointCandidateLags[0];
    double best_sse = -1.0;
    std::array<i64, 4> best_d{};
    for (u32 lag : kRodJointCandidateLags) {
        std::array<i64, 4> d = calibrate_block(qw, qx, qy, qz, start, end, lag);
        double sse = block_predict_sse(qw, qx, qy, qz, start, end, lag, d);
        if (best_sse < 0.0 || sse < best_sse) {
            best_sse = sse;
            best_lag = lag;
            best_d = d;
        }
    }
    out_d = best_d;
    return best_lag;
}

// Shared inner loop for quaternion_joint_forward/_forward_adaptive:
// predicts and encodes Q[start+1..end] (in the qw/qx/qy/qz index space,
// i.e. indices [start,end) there) using a calibrated (lag, D), updating
// reconstructed history and residuals -- exactly the fixed-block
// forward's original loop body, factored out so the adaptive variant
// shares it instead of duplicating the prediction math.
void encode_block_quat(const std::vector<i32>& qw, const std::vector<i32>& qx, const std::vector<i32>& qy,
                        const std::vector<i32>& qz, size_t start, size_t end, u32 lag, const std::array<i64, 4>& d,
                        u32 quant_step, u32 resync_interval,
                        std::vector<i32>& rec_w, std::vector<i32>& rec_x, std::vector<i32>& rec_y, std::vector<i32>& rec_z,
                        std::vector<i32>& residual_w, std::vector<i32>& residual_x,
                        std::vector<i32>& residual_y, std::vector<i32>& residual_z) {
    for (size_t i = start; i < end; i++) {
        i32 aw = (i >= lag) ? rec_w[i - lag] : 0;
        i32 ax = (i >= lag) ? rec_x[i - lag] : 0;
        i32 ay = (i >= lag) ? rec_y[i - lag] : 0;
        i32 az = (i >= lag) ? rec_z[i - lag] : 0;
        i32 pw, px, py, pz;
        quat_right_mul_round(aw, ax, ay, az, d, pw, px, py, pz);

        bool is_resync = (resync_interval > 0) && (((i + 1) % resync_interval) == 0);
        u32 q_here = is_resync ? 1 : quant_step;

        i64 raw_dw = (i64)qw[i] - pw;
        i64 raw_dx = (i64)qx[i] - px;
        i64 raw_dy = (i64)qy[i] - py;
        i64 raw_dz = (i64)qz[i] - pz;

        i64 rw = quant_round_div(raw_dw, (i64)q_here);
        i64 rx = quant_round_div(raw_dx, (i64)q_here);
        i64 ry = quant_round_div(raw_dy, (i64)q_here);
        i64 rz = quant_round_div(raw_dz, (i64)q_here);
        residual_w[i] = (i32)rw;
        residual_x[i] = (i32)rx;
        residual_y[i] = (i32)ry;
        residual_z[i] = (i32)rz;

        rec_w[i] = (i32)(pw + rw * (i64)q_here);
        rec_x[i] = (i32)(px + rx * (i64)q_here);
        rec_y[i] = (i32)(py + ry * (i64)q_here);
        rec_z[i] = (i32)(pz + rz * (i64)q_here);
    }
}

struct QuatBlockCalib { u32 lag; std::array<i64, 4> d; };

std::pair<QuatBlockCalib, double> calibrate_and_score_quat(const std::vector<i32>& qw, const std::vector<i32>& qx,
                                                            const std::vector<i32>& qy, const std::vector<i32>& qz,
                                                            size_t start, size_t end) {
    std::array<i64, 4> d{};
    u32 lag = pick_block_lag(qw, qx, qy, qz, start, end, /*force_lag=*/0, d);
    double sse = block_predict_sse(qw, qx, qy, qz, start, end, lag, d);
    return {QuatBlockCalib{lag, d}, sse};
}

std::vector<u32> block_index_from_lengths(size_t n, const std::vector<u32>& block_len) {
    std::vector<u32> idx(n);
    size_t pos = 0;
    for (size_t b = 0; b < block_len.size(); b++) {
        size_t end = std::min(n, pos + (size_t)block_len[b]);
        for (size_t i = pos; i < end; i++) idx[i] = (u32)b;
        pos = end;
    }
    return idx;
}

} // namespace

QuaternionJointResult quaternion_joint_forward(const std::vector<Quat4i>& quats, u32 force_lag,
                                                u32 quant_step, u32 resync_interval) {
    QuaternionJointResult r;
    r.count = quats.size();
    if (quats.empty()) return r;
    r.anchor = quats[0];
    r.quant_step = (quant_step == 0) ? 1 : quant_step;
    r.resync_interval = resync_interval;
    size_t m = quats.size();
    if (m < 2) return r;

    size_t n = m - 1; // predicted quaternions Q[1..m-1]
    std::vector<i32> qw(n), qx(n), qy(n), qz(n);
    for (size_t i = 1; i < m; i++) {
        qw[i - 1] = quats[i].w;
        qx[i - 1] = quats[i].x;
        qy[i - 1] = quats[i].y;
        qz[i - 1] = quats[i].z;
    }
    // Reconstructed (possibly lossy) history, indexed the same way as
    // qw/qx/qy/qz (index 0 == Q[1]) -- the anchor Q[0] is deliberately
    // *not* part of this array and is never used as a lag source, exactly
    // like Rod-Joint's anchor point is never itself treated as "rod -1"
    // (see rod_joint_transform.cpp): i < lag predicts a zero quaternion
    // instead, keeping calibration's SSE scoring and the actual per-sample
    // prediction in exact agreement at every index, including the first
    // block's boundary.
    std::vector<i32> rec_w(n), rec_x(n), rec_y(n), rec_z(n);

    size_t nblocks = (n + kQuatJointBlockSize - 1) / kQuatJointBlockSize;
    r.block_lag.resize(nblocks);
    r.block_delta.resize(nblocks);
    r.residual_w.resize(n);
    r.residual_x.resize(n);
    r.residual_y.resize(n);
    r.residual_z.resize(n);

    for (size_t blk = 0; blk < nblocks; blk++) {
        size_t start = blk * kQuatJointBlockSize;
        size_t end = std::min(n, start + kQuatJointBlockSize);
        // Calibration/lag search may use true quaternions (qw/qx/qy/qz,
        // shifted by 1 vs. hist); only the actual prediction step below
        // needs closed-loop reconstructed history (hist_*).
        std::array<i64, 4> d{};
        u32 lag = pick_block_lag(qw, qx, qy, qz, start, end, force_lag, d);
        r.block_lag[blk] = lag;
        r.block_delta[blk] = d;
        encode_block_quat(qw, qx, qy, qz, start, end, lag, d, r.quant_step, r.resync_interval,
                           rec_w, rec_x, rec_y, rec_z, r.residual_w, r.residual_x, r.residual_y, r.residual_z);
    }
    return r;
}

QuaternionJointResult quaternion_joint_forward_adaptive(const std::vector<Quat4i>& quats,
                                                         u32 quant_step, u32 resync_interval,
                                                         size_t min_block, double merge_ratio) {
    QuaternionJointResult r;
    r.count = quats.size();
    if (quats.empty()) return r;
    r.anchor = quats[0];
    r.quant_step = (quant_step == 0) ? 1 : quant_step;
    r.resync_interval = resync_interval;
    size_t m = quats.size();
    if (m < 2) return r;

    size_t n = m - 1;
    std::vector<i32> qw(n), qx(n), qy(n), qz(n);
    for (size_t i = 1; i < m; i++) {
        qw[i - 1] = quats[i].w;
        qx[i - 1] = quats[i].x;
        qy[i - 1] = quats[i].y;
        qz[i - 1] = quats[i].z;
    }
    std::vector<i32> rec_w(n), rec_x(n), rec_y(n), rec_z(n);

    auto blocks = adaptive_partition<QuatBlockCalib>(n, min_block, merge_ratio,
        [&](size_t start, size_t end) { return calibrate_and_score_quat(qw, qx, qy, qz, start, end); },
        [&](size_t start, size_t end, const QuatBlockCalib& c) { return block_predict_sse(qw, qx, qy, qz, start, end, c.lag, c.d); });

    r.block_len.resize(blocks.size());
    r.block_lag.resize(blocks.size());
    r.block_delta.resize(blocks.size());
    r.residual_w.resize(n);
    r.residual_x.resize(n);
    r.residual_y.resize(n);
    r.residual_z.resize(n);

    for (size_t b = 0; b < blocks.size(); b++) {
        r.block_len[b] = (u32)(blocks[b].end - blocks[b].start);
        r.block_lag[b] = blocks[b].calib.lag;
        r.block_delta[b] = blocks[b].calib.d;
        encode_block_quat(qw, qx, qy, qz, blocks[b].start, blocks[b].end, blocks[b].calib.lag, blocks[b].calib.d,
                           r.quant_step, r.resync_interval, rec_w, rec_x, rec_y, rec_z,
                           r.residual_w, r.residual_x, r.residual_y, r.residual_z);
    }
    return r;
}

std::vector<Quat4i> quaternion_joint_inverse(const QuaternionJointResult& r) {
    std::vector<Quat4i> quats;
    if (r.count == 0) return quats;
    quats.resize((size_t)r.count);
    quats[0] = r.anchor;
    if (r.count < 2) return quats;

    size_t n = (size_t)r.count - 1;
    u32 quant_step = (r.quant_step == 0) ? 1 : r.quant_step;
    std::vector<i32> rec_w(n), rec_x(n), rec_y(n), rec_z(n);
    std::vector<u32> block_of;
    if (!r.block_len.empty()) block_of = block_index_from_lengths(n, r.block_len);

    for (size_t i = 0; i < n; i++) {
        size_t blk = r.block_len.empty() ? (i / kQuatJointBlockSize) : (size_t)block_of[i];
        u32 lag = r.block_lag[blk];
        const std::array<i64, 4>& d = r.block_delta[blk];
        i32 aw = (i >= lag) ? rec_w[i - lag] : 0;
        i32 ax = (i >= lag) ? rec_x[i - lag] : 0;
        i32 ay = (i >= lag) ? rec_y[i - lag] : 0;
        i32 az = (i >= lag) ? rec_z[i - lag] : 0;
        i32 pw, px, py, pz;
        quat_right_mul_round(aw, ax, ay, az, d, pw, px, py, pz);

        bool is_resync = (r.resync_interval > 0) && (((i + 1) % r.resync_interval) == 0);
        u32 q_here = is_resync ? 1 : quant_step;

        i64 rw = r.residual_w[i], rx = r.residual_x[i], ry = r.residual_y[i], rz = r.residual_z[i];
        rec_w[i] = (i32)(pw + rw * (i64)q_here);
        rec_x[i] = (i32)(px + rx * (i64)q_here);
        rec_y[i] = (i32)(py + ry * (i64)q_here);
        rec_z[i] = (i32)(pz + rz * (i64)q_here);

        quats[i + 1] = {rec_w[i], rec_x[i], rec_y[i], rec_z[i]};
    }
    return quats;
}

} // namespace csa
