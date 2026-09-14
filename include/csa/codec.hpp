// csa/codec.hpp — top-level container format tying the transforms and the
// range coder together into a single compress()/decompress() API, plus a
// raw-storage fallback mode so the codec can never blow up incompressible
// input by more than a small fixed header.
#pragma once
#include "csa/common.hpp"
#include "csa/lz_matcher.hpp"
#include "csa/quaternion_joint.hpp"
#include "csa/rod_joint_transform.hpp"
#include <vector>

namespace csa {

enum class Mode : u8 { Raw = 0, General = 1, Geo2D = 2, Geo3D = 3, GeneralLZ = 4, GeneralBWT = 5, Pose = 6 };

// General-purpose byte-stream compression. Up to four candidates are
// tried -- raw storage, the Pantograph Lift (predictive, good on smooth/
// self-similar numeric data), an LZ77-style dictionary matcher with
// adaptive range coding (repeated-substring redundancy, good on text and
// structured files), and (for inputs up to a few MB; see codec.cpp's
// kBwtMaxInputSize) a Burrows-Wheeler Transform + move-to-front mode
// (local byte-context statistics, good on ordinary prose -- the kind of
// redundancy bz2 targets and the other candidates don't attack directly)
// -- and whichever encodes smallest wins, so callers never need to guess
// which model suits their data. When use_gpu is true
// and a CUDA device is available, the Pantograph Lift candidate's forward
// transform runs on the GPU (level-parallel); otherwise it transparently
// falls back to the CPU implementation. The bitstream is identical either
// way.
//
// lz_max_chain/lz_nice_length are a real speed-vs-ratio knob for the LZ
// candidate's match search (see lz_matcher.hpp), the same tradeoff every
// production LZ compressor exposes as a "level". They're a pure
// encoder-side search parameter with no bitstream effect, so decompress()
// needs no matching parameter.
std::vector<u8> compress(const std::vector<u8>& input, bool use_gpu = false,
                          int lz_max_chain = kLzDefaultMaxChain,
                          size_t lz_nice_length = kLzDefaultNiceLength);
std::vector<u8> decompress(const std::vector<u8>& blob);

// Literal geometric mode for 2D point sequences (Rod-Joint Transform).
std::vector<u8> compress_geo2d(const std::vector<Point2i>& points);
std::vector<Point2i> decompress_geo2d(const std::vector<u8>& blob);

// Lossy variant: quantizes rod residuals to the nearest multiple of
// quant_step (see rod_joint_transform.hpp for the closed-loop design and
// rod_joint_2d_error_bound() for the resulting per-coordinate error
// bound), with an exact resync rod every resync_interval rods to bound
// how far absolute position error can drift (0 = never resync). Decoding
// is via the same decompress_geo2d -- the blob carries its own
// quant_step/resync_interval, so the decoder needs no special "lossy
// mode" of its own. quant_step <= 1 is exactly compress_geo2d (lossless).
std::vector<u8> compress_geo2d_lossy(const std::vector<Point2i>& points, u32 quant_step, u32 resync_interval);

// 3D point sequences: tries the xy+z composition and the true 3D
// similarity joint, keeps whichever encodes smaller.
std::vector<u8> compress_geo3d(const std::vector<Point3i>& points);
std::vector<Point3i> decompress_geo3d(const std::vector<u8>& blob);

// Lossy variant, same design as compress_geo2d_lossy. Tries both the
// composition and the true 3D similarity joint (the composition's z-axis
// Pantograph Lift supports quantization too -- see DESIGN.md) and keeps
// whichever encodes smaller, the same full auto-select compress_geo3d
// already does losslessly. quant_step <= 1 is exactly compress_geo3d
// (lossless, full auto-select).
std::vector<u8> compress_geo3d_lossy(const std::vector<Point3i>& points, u32 quant_step, u32 resync_interval);

// 6-DOF pose stream (position + orientation, e.g. VR/AR tracking, drone/
// robot odometry, SLAM camera paths): position goes through the same
// Geo3D auto-select as compress_geo3d, orientation through the
// Quaternion Joint (see quaternion_joint.hpp) -- independent models for
// independent structure, sharing one container.
std::vector<u8> compress_pose(const std::vector<Pose>& poses);
std::vector<Pose> decompress_pose(const std::vector<u8>& blob);

// Lossy variant: pos_quant_step/pos_resync_interval reach compress_geo3d_lossy's
// existing lossy position support directly; quat_quant_step/
// quat_resync_interval are the analogous knobs for the Quaternion Joint
// (see QuaternionJointResult's own comment for the closed-loop design and
// what resync_interval bounds here). Either quant_step <= 1 is exactly
// lossless for that half.
std::vector<u8> compress_pose_lossy(const std::vector<Pose>& poses,
                                     u32 pos_quant_step, u32 pos_resync_interval,
                                     u32 quat_quant_step, u32 quat_resync_interval);

} // namespace csa
