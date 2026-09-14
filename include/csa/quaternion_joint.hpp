// csa/quaternion_joint.hpp — the Rod-Joint idea applied to orientation.
//
// A 6-DOF pose stream (a VR/AR headset or controller, a drone or ground
// robot's odometry, a SLAM camera trajectory) is a sequence of (position,
// orientation) samples at a steady rate. Position is already exactly what
// Rod-Joint/Pantograph Lift model (see rod_joint_transform.hpp); this file
// is the analogous model for the orientation half, represented the
// standard way real tracking systems represent it: a unit quaternion
// per sample (continuous, no gimbal lock, 4 numbers).
//
// The physical intuition driving this is identical to Rod-Joint's: real
// motion has *locally consistent angular velocity* -- a smoothly turning
// head, a drone banking through a turn, a robot arcing around a corner --
// so the rotation that carried orientation Q[i-1] into Q[i] is a decent
// predictor of the rotation that will carry Q[i-1+lag] into Q[i+lag] too.
// Concretely: predict Q[i] from an earlier Q[i-lag] via a single
// calibrated "delta" quaternion D, recalibrated per block exactly like
// Rod-Joint's ratio/matrix:
//
//     Q_pred[i] = Q[i-lag] (x) D          ((x) = Hamilton/quaternion product)
//
// D is applied via RIGHT multiplication deliberately, not left: real
// tracking data (gyroscope/IMU-integrated orientation) accumulates
// rotation in the object's own rotating (body) frame, i.e.
// Q[i] = Q[i-1] (x) dq_body, so D approximates the block's characteristic
// per-lag-step body-frame delta rotation. (Using left-multiplication
// instead would model rotation about a fixed world axis, which is the
// less common case for something like a head or drone actually turning.)
//
// Calibrating D has an unusually clean closed form, simpler than the 3D
// position joint's Horn's-method eigenvector iteration
// (rod_joint_transform.cpp's calibrate_3d_block): right-multiplication by
// a quaternion a is a *linear* map of the other operand
// (a (x) D = R(a) * D for the 4x4 matrix R(a) built from a's components),
// and quaternion multiplication is norm-multiplicative (|p (x) q| =
// |p||q| for every p, q, not just unit ones) -- which means
// R(a)^T R(a) = |a|^2 * I. Both facts together collapse the usual normal-
// equations matrix for least-squares fitting D over many (a_i, b_i) pairs
// down to a *scalar* multiple of the identity, so the fit is just:
//
//     D = ( sum_i R(a_i)^T b_i ) / ( sum_i |a_i|^2 )
//
// -- no matrix inversion or iteration needed at all, for any candidate
// lag. This makes lag search (reusing kRodJointCandidateLags, same
// mechanism and same period-drift benefit already proven for position
// rods on a toroidal path -- see DESIGN.md) essentially free per
// candidate, exactly like Pantograph Lift's own single-ratio calibration.
//
// Unlike position (where rods, not points, are what gets predicted, and
// reconstructed rods must be summed back into an absolute point), Q[i]
// *is* the absolute state already -- there is no accumulation step here.
// What does still need care is error compounding under lossy quantization:
// Q_pred[i] depends on the *reconstructed* Q[i-lag] (closed-loop DPCM,
// same discipline as every other lossy mode in this codebase), so a
// quantization error introduced at position i-lag can propagate forward
// into every later prediction that depends on it, transitively. This is
// exactly the same shape of problem Rod-Joint's `resync_interval` already
// solves, and it means the same thing here: every resync_interval-th
// quaternion is stored exactly (its own residual is never quantized),
// bounding how far this compounding can drift before the next reset.
// resync_interval == 0 means "never" (fine for lossless use).
//
// What this deliberately does NOT do: model smoothly *accelerating*
// angular velocity within a single calibration block (only per-block
// recalibration handles that, the same compromise Rod-Joint/Pantograph
// Lift already make for position), or use quaternion exponentiation/SLERP
// to compose a multi-step prediction -- D is fit directly against
// whatever (Q[i-lag], Q[i]) pairs actually occur in the data, which is
// well-posed and correct regardless of whether real angular velocity was
// exactly constant over that gap; it simply won't compress as well if it
// wasn't.
#pragma once
#include "csa/common.hpp"
#include "csa/rod_joint_transform.hpp" // reuses kRodJointCandidateLags
#include <array>
#include <vector>

namespace csa {

struct Quat4i { i32 w, x, y, z; };

constexpr size_t kQuatJointBlockSize = 128; // quaternions per calibration block

// Same lossless/lossy unification as RodJoint2DResult/RodJoint3DSimResult:
// quant_step <= 1 is exactly lossless (quant_round_div(d, 1) == d always).
// block_delta stores each block's calibrated D as a Q16.16 fixed-point
// 4-vector (w, x, y, z), applied via right-multiplication.
struct QuaternionJointResult {
    Quat4i anchor{0, 0, 0, 0}; // Q[0], stored exactly
    u64 count = 0;             // number of quaternions
    u32 quant_step = 1;        // 1 == lossless
    u32 resync_interval = 0;   // 0 == no periodic exact resync
    // Empty (the default): blocks are the fixed kQuatJointBlockSize
    // scheme. Non-empty: explicit per-block lengths (summing to
    // count-1) -- see quaternion_joint_forward_adaptive() and
    // adaptive_partition.hpp.
    std::vector<u32> block_len;
    std::vector<u32> block_lag;                 // per block: Q[i] predicted from Q[i-block_lag[blk]]
    std::vector<std::array<i64, 4>> block_delta; // per block, Q16.16 (w,x,y,z)
    std::vector<i32> residual_w, residual_x, residual_y, residual_z; // size count-1 each
};

// force_lag == 0 (default) searches kRodJointCandidateLags independently
// per calibration block, exactly like rod_joint_2d_forward/
// rod_joint_3d_similarity_forward; a nonzero value forces that lag
// uniformly across every block instead.
QuaternionJointResult quaternion_joint_forward(const std::vector<Quat4i>& quats, u32 force_lag = 0,
                                                u32 quant_step = 1, u32 resync_interval = 0);

// Variable-resolution alternative to the fixed-block forward() above --
// same rationale and mechanism as
// rod_joint_3d_similarity_forward_adaptive (see adaptive_partition.hpp):
// bottom-up greedy merging of min_block-sized chunks where doing so
// doesn't cost much prediction accuracy, instead of always using
// kQuatJointBlockSize.
QuaternionJointResult quaternion_joint_forward_adaptive(const std::vector<Quat4i>& quats,
                                                         u32 quant_step = 1, u32 resync_interval = 0,
                                                         size_t min_block = 16, double merge_ratio = 1.3);
std::vector<Quat4i> quaternion_joint_inverse(const QuaternionJointResult& r);

// A single 6-DOF pose sample (position + orientation), the unit a real
// tracking stream (VR/AR headset or controller, drone/robot odometry,
// SLAM camera path) is a sequence of. codec.cpp's compress_pose/
// decompress_pose compress the position and orientation streams
// independently -- position through the existing Geo3D auto-select
// (rod_joint_transform.hpp), orientation through the Quaternion Joint
// above -- since the two have unrelated structure (a rotating trajectory
// vs. a rotation sequence) and each already has its own best model.
struct Pose { Point3i position; Quat4i orientation; };

} // namespace csa
