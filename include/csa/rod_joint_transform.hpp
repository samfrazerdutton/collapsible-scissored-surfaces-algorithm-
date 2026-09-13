// csa/rod_joint_transform.hpp — the literal geometric-mode transform.
//
// This is the most direct instantiation of the scissor-linkage idea: a
// point sequence (a polyline — GPS track, mesh silhouette, spiral/helix
// point cloud, toroidal cross-section, ...) is a chain of rigid "rods"
// (edge vectors between consecutive points) connected by "joints". In a
// real pantograph lattice a single fixed pivot ratio, applied joint after
// joint, is enough to deploy an entire helical/toroidal surface from a
// collapsed line. Here the same idea is a calibrated complex constant c
// (a 2D rotation+scale — a "similarity" ratio) that predicts each rod from
// an earlier one: rod[i] ~= c * rod[i-lag]. Only the exact integer
// residual is stored, so reconstruction is bit-exact regardless of how
// good the calibration is. For a perfect logarithmic spiral/helix
// projection, c alone (2 numbers) plus the anchor point predicts the
// entire shape and every residual collapses to zero.
//
// Like the Pantograph Lift, c is recalibrated every kRodJointBlockSize
// rods rather than once for the whole chain: a single global rotation
// can't track a path whose curvature itself drifts (e.g. a wobbling
// radius), so periodic recalibration trades a still-tiny parameter count
// for real local adaptivity.
//
// `lag` defaults to 1 (predict from the immediately preceding rod), but a
// path whose rods oscillate with a period shorter than any reasonable
// calibration block (e.g. a toroidal cross-section's radius, which can
// complete a full cycle every 2-3 rods) can never be tracked by an
// immediate-previous-rod predictor no matter how often it's recalibrated
// -- a lag matching that period aligns the prediction directly instead.
// codec.cpp's compress_geo2d/compress_geo3d search a small set of
// candidate lags (see kRodJointCandidateLags) and keep whichever actually
// encodes smallest; this is exactly the same kind of long-term/pitch
// prediction speech and audio codecs use for periodic signals, applied
// here to rod sequences instead of waveform samples.
//
// 3D point sequences have two candidate models, and the codec (see
// codec.cpp's compress_geo3d) tries both and keeps whichever encodes
// smaller, tagging the choice with one header byte:
//
//   (a) Composition: the (x, y) plane goes through the same rotation+
//       scale Rod-Joint predictor as the 2D case (this captures genuine
//       turning motion jointly — a rotating trajectory's x and y are
//       individually just two out-of-phase sine waves, each hard for a
//       simple predictor, but together they are exactly what the
//       complex-ratio joint models), and z goes through the Pantograph
//       Lift on its own (captures smooth/linear vertical motion, e.g. a
//       helix's constant climb rate). Wins when the path's radius in the
//       xy-plane is genuinely constant (a helix).
//
//   (b) True 3D similarity joint: a single calibrated 3x3 matrix
//       (rotation + uniform scale) predicts each 3D rod from the
//       previous one, rod[i] ~= M * rod[i-1]. M is fit per block via
//       Horn's closed-form absolute-orientation method: build the 3x3
//       cross-covariance of consecutive rods, form the corresponding
//       4x4 symmetric "profile" matrix, and take its dominant eigenvector
//       (the optimal rotation quaternion) via shifted power iteration —
//       a handful of 4x4 matrix-vector products, no general eigensolver
//       needed. Optimal scale then follows in closed form. This is the
//       genuinely-3D generalization of the 2D joint above, and can track
//       paths whose rotation axis isn't fixed to z (e.g. a path that
//       tumbles), including some of what defeats the xy+z composition.
//
// Both are recalibrated per block (kRodJointBlockSize / kRodJoint3DBlockSize
// rods) for the same reason as the 2D case, and both are exactly
// reversible regardless of calibration quality — the residual is always
// `actual - predicted`, so a bad fit only costs compression ratio.
#pragma once
#include "csa/common.hpp"
#include "csa/pantograph_lift.hpp"
#include <array>
#include <vector>

namespace csa {

struct Point2i { i32 x, y; };
struct Point3i { i32 x, y, z; };

constexpr size_t kRodJointBlockSize = 128;   // rods per calibration block (2D)
constexpr size_t kRodJoint3DBlockSize = 128; // rods per calibration block (3D similarity)

constexpr u32 kRodJointCandidateLags[] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32};
constexpr size_t kRodJointNumCandidateLags = sizeof(kRodJointCandidateLags) / sizeof(kRodJointCandidateLags[0]);

struct RodJoint2DResult {
    Point2i anchor{0, 0};
    u64 count = 0;                    // number of points
    u32 lag = 1;                      // rod[i] predicted from rod[i-lag]
    std::vector<i64> block_ratio_re;  // per block, Q16.16 fixed-point joint constant
    std::vector<i64> block_ratio_im;  // c = ratio_re + i*ratio_im
    std::vector<i32> residual_x;      // size count-1 (empty if count <= 1)
    std::vector<i32> residual_y;
};

RodJoint2DResult rod_joint_2d_forward(const std::vector<Point2i>& points, u32 lag = 1);
std::vector<Point2i> rod_joint_2d_inverse(const RodJoint2DResult& r);

struct RodJoint3DResult {
    RodJoint2DResult xy;
    LiftResult lift_z;
};

RodJoint3DResult rod_joint_3d_forward(const std::vector<Point3i>& points, u32 xy_lag = 1);
std::vector<Point3i> rod_joint_3d_inverse(const RodJoint3DResult& r);

struct RodJoint3DSimResult {
    Point3i anchor{0, 0, 0};
    u64 count = 0;
    u32 lag = 1;                      // rod[i] predicted from rod[i-lag]
    std::vector<std::array<i64, 9>> block_matrix; // per block, row-major 3x3, Q16.16
    std::vector<i32> residual_x, residual_y, residual_z; // size count-1 each
};

RodJoint3DSimResult rod_joint_3d_similarity_forward(const std::vector<Point3i>& points, u32 lag = 1);
std::vector<Point3i> rod_joint_3d_similarity_inverse(const RodJoint3DSimResult& r);

} // namespace csa
