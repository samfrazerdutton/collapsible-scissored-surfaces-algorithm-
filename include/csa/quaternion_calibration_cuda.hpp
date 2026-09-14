// csa/quaternion_calibration_cuda.hpp — GPU measurement of whether the
// Quaternion Joint's per-block-per-lag calibration search benefits from
// parallelism, following the exact same honest-measurement discipline as
// pantograph_lift_cuda.hpp: this is a real, correctness-verified kernel,
// not an assumption, and DESIGN.md reports whatever it actually measures
// (win or loss) rather than a guess.
//
// Architectural note this measurement exists to test: unlike the
// Pantograph Lift's forward transform (already GPU-accelerated), the
// Quaternion Joint's *calibration* is the only part of it that's
// embarrassingly parallel across blocks -- calibration uses the TRUE
// quaternion sequence, which is fully known up front, so every
// (block, candidate lag) pair is independent. The *encoding* pass that
// follows (turning a calibration into residuals) is NOT parallel across
// blocks: it depends on the *reconstructed* history of earlier blocks
// (the closed-loop DPCM design every lossy mode in this codebase uses),
// so it stays sequential on the CPU regardless of what this measures.
// That means even a large win here can only ever speed up the
// calibration fraction of total compress_pose time, not the whole call.
//
// Returns false (leaves outputs unspecified) if no CUDA device is
// available or CSA was built with WITH_CUDA=OFF.
#pragma once
#include "csa/common.hpp"
#include "csa/pantograph_lift_cuda.hpp" // reuses cuda_is_available()
#include <array>
#include <vector>

namespace csa {

// Computes, for every (calibration block, candidate lag) pair, the
// closed-form calibrated delta quaternion D and the exact SSE it
// achieves -- the same (D, SSE) pick_block_lag's per-candidate search
// computes on the CPU (quaternion_joint.cpp), just for every candidate
// at once instead of one at a time.
//
// qw/qx/qy/qz are the quaternion component streams already shifted the
// way quaternion_joint.cpp's internals expect (qw[i] == the (i+1)-th
// quaternion's w component, etc. -- see quaternion_joint_forward's own
// comment for why). `lags` is the candidate lag list (typically
// kRodJointCandidateLags). block_size is normally kQuatJointBlockSize.
//
// out_delta/out_sse are resized to nblocks * lags.size(); entry
// [block * lags.size() + lag_index] holds that (block, lag) pair's
// result, mirroring how the CPU search iterates.
bool quat_calibrate_all_blocks_cuda(const std::vector<i32>& qw, const std::vector<i32>& qx,
                                     const std::vector<i32>& qy, const std::vector<i32>& qz,
                                     size_t block_size, const std::vector<u32>& lags,
                                     std::vector<std::array<i64, 4>>& out_delta,
                                     std::vector<double>& out_sse);

} // namespace csa
