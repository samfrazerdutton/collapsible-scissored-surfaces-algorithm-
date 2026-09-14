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

// Lossless and lossy share one code path: a rod's residual is quantized
// to the nearest multiple of `quant_step` (q=1 is exact -- round(d/1)*1
// == d always, so lossless is simply the q=1 special case, not a
// separate implementation). What's actually stored is the quantized
// *index* (residual/q, rounded), which is why lossy mode compresses much
// better: the values being entropy-coded are smaller by roughly a factor
// of q. Critically, prediction for rod i uses the *reconstructed* (lossy)
// rod i-lag, not the true original one -- the same closed-loop DPCM
// design real predictive codecs use, so the encoder and decoder always
// agree on history. Per-rod reconstruction error is bounded by q/2 in
// each coordinate, but because rods accumulate into an absolute point
// path (points[i+1] = points[i] + rod[i]), that per-rod error is a random
// walk over the sequence and *absolute* position error can drift further
// the longer the run since the last exact point. `resync_interval` bounds
// that drift: every `resync_interval` rods, one rod is stored exactly
// (lossless) regardless of `quant_step`, which resets accumulated error
// to zero there -- the same role a keyframe plays in a video codec.
// resync_interval == 0 means "never" (fine for lossless use, or for short
// sequences where drift never gets large enough to matter).
// `lag` used to be one value for the whole sequence, found by an outer
// per-file search over kRodJointCandidateLags (codec.cpp tried each lag as
// a whole separate forward() call and kept whichever encoded smallest).
// That works well when one oscillation period describes the entire path,
// but a path whose period itself drifts partway through (e.g. a toroidal
// scan whose pitch changes, or two concatenated shapes with different
// periods) can't be served by any single global lag. `block_lag` makes
// the choice per calibration block instead: each block searches
// kRodJointCandidateLags independently (by least-squares residual energy,
// the same criterion the ratio calibration itself already optimizes) and
// keeps whichever lag it predicts best with. A uniform whole-file lag is
// just the special case where every block happens to pick the same
// candidate, so this can only match or beat the old design, never lose to
// it, and it does the search in one pass instead of re-running the whole
// transform once per candidate.
struct RodJoint2DResult {
    Point2i anchor{0, 0};
    u64 count = 0;                    // number of points
    u32 quant_step = 1;                // 1 == lossless
    u32 resync_interval = 0;           // 0 == no periodic exact resync
    std::vector<u32> block_lag;       // per block: rod[i] predicted from rod[i-block_lag[blk]]
    std::vector<i64> block_ratio_re;  // per block, Q16.16 fixed-point joint constant
    std::vector<i64> block_ratio_im;  // c = ratio_re + i*ratio_im
    std::vector<i32> residual_x;      // size count-1 (empty if count <= 1); quantized index when quant_step > 1
    std::vector<i32> residual_y;
};

// `force_lag == 0` (the default) searches kRodJointCandidateLags
// independently per calibration block. A nonzero value instead forces
// that one lag uniformly across every block, skipping the search --
// mainly useful for tests/benchmarks that want to measure "what if lag
// search were disabled" against the auto-search result.
RodJoint2DResult rod_joint_2d_forward(const std::vector<Point2i>& points, u32 force_lag = 0,
                                       u32 quant_step = 1, u32 resync_interval = 0);
std::vector<Point2i> rod_joint_2d_inverse(const RodJoint2DResult& r);

// Reports the exact per-coordinate reconstruction error bound implied by
// a given quant_step: any lossy-mode coordinate is within this many units
// of its true value (0 for quant_step <= 1, i.e. lossless).
inline u32 rod_joint_2d_error_bound(u32 quant_step) {
    return quant_step <= 1 ? 0 : (quant_step / 2);
}

struct RodJoint3DResult {
    RodJoint2DResult xy;
    LiftResult lift_z;
};

// xy_quant_step/xy_resync_interval reach RodJoint2DResult's own already-
// lossy-capable xy plane directly; z_quant_step is new -- it's what lets
// this composition model's z-axis (a plain Pantograph Lift over the raw
// z-coordinates) go lossy too, closing the gap noted in DESIGN.md where
// this model previously couldn't participate in a lossy comparison
// against the true 3D similarity joint at all. quant_step <= 1 on either
// axis is exactly lossless for that axis (see RodJoint2DResult's and
// LiftResult's own comments).
RodJoint3DResult rod_joint_3d_forward(const std::vector<Point3i>& points, u32 xy_force_lag = 0,
                                       u32 xy_quant_step = 1, u32 xy_resync_interval = 0,
                                       u32 z_quant_step = 1);
std::vector<Point3i> rod_joint_3d_inverse(const RodJoint3DResult& r);

// Same lossless/lossy unification as RodJoint2DResult (quant_step == 1 is
// exactly lossless; resync_interval periodically forces an exact rod,
// computed to land on the true *absolute point* given wherever the
// reconstructed position currently is, not the true rod -- see
// RodJoint2DResult's comment for why that distinction matters). Lag is
// per-block for the same reason as RodJoint2DResult's `block_lag` --
// see that struct's comment.
struct RodJoint3DSimResult {
    Point3i anchor{0, 0, 0};
    u64 count = 0;
    u32 quant_step = 1;                // 1 == lossless
    u32 resync_interval = 0;           // 0 == no periodic exact resync
    // Empty (the default): blocks are the fixed kRodJoint3DBlockSize
    // scheme. Non-empty: explicit per-block lengths (summing to
    // count-1), overriding the fixed scheme -- see
    // rod_joint_3d_similarity_forward_adaptive() and
    // adaptive_partition.hpp for why and when this is used. Either way,
    // block_lag/block_matrix have one entry per actual block.
    std::vector<u32> block_len;
    std::vector<u32> block_lag;       // per block: rod[i] predicted from rod[i-block_lag[blk]]
    std::vector<std::array<i64, 9>> block_matrix; // per block, row-major 3x3, Q16.16
    std::vector<i32> residual_x, residual_y, residual_z; // size count-1 each; quantized index when quant_step > 1
};

// force_lag == 0 (default) searches kRodJointCandidateLags per block; a
// nonzero value forces that lag uniformly (see rod_joint_2d_forward).
RodJoint3DSimResult rod_joint_3d_similarity_forward(const std::vector<Point3i>& points, u32 force_lag = 0,
                                                     u32 quant_step = 1, u32 resync_interval = 0);

// Variable-resolution alternative to the fixed-block forward() above:
// bottom-up greedy merging of min_block-sized chunks where doing so
// doesn't cost much prediction accuracy (see adaptive_partition.hpp),
// instead of always using kRodJoint3DBlockSize. Always searches
// kRodJointCandidateLags per block (no force_lag option -- adaptive mode
// is about resolution, not lag). min_block/merge_ratio tune the
// segmentation; the defaults are the ones measured against real tracking
// data.
RodJoint3DSimResult rod_joint_3d_similarity_forward_adaptive(const std::vector<Point3i>& points,
                                                              u32 quant_step = 1, u32 resync_interval = 0,
                                                              size_t min_block = 16, double merge_ratio = 2.0);
std::vector<Point3i> rod_joint_3d_similarity_inverse(const RodJoint3DSimResult& r);

// Same per-coordinate error bound as rod_joint_2d_error_bound() (the
// designs are identical, just in 3D).
inline u32 rod_joint_3d_error_bound(u32 quant_step) {
    return quant_step <= 1 ? 0 : (quant_step / 2);
}

} // namespace csa
