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
// the previous one: rod[i] ~= c * rod[i-1]. Only the exact integer
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
// 3D point sequences compose the two pieces above rather than using a full
// 3D similarity joint: the (x, y) plane goes through the same rotation+
// scale Rod-Joint predictor as the 2D case (this captures genuine turning
// motion jointly — a rotating trajectory's x and y are individually just
// two out-of-phase sine waves, each hard for a simple predictor, but
// together they are exactly what the complex-ratio joint models), and z
// goes through the Pantograph Lift on its own (captures smooth/linear
// vertical motion, e.g. a helix's constant climb rate). A full 3D
// similarity joint (rotation+scale via a calibrated quaternion, for
// genuinely 3D-tumbling paths) is noted as future work in DESIGN.md.
#pragma once
#include "csa/common.hpp"
#include "csa/pantograph_lift.hpp"
#include <vector>

namespace csa {

struct Point2i { i32 x, y; };
struct Point3i { i32 x, y, z; };

constexpr size_t kRodJointBlockSize = 128; // rods per calibration block

struct RodJoint2DResult {
    Point2i anchor{0, 0};
    u64 count = 0;                    // number of points
    std::vector<i64> block_ratio_re;  // per block, Q16.16 fixed-point joint constant
    std::vector<i64> block_ratio_im;  // c = ratio_re + i*ratio_im
    std::vector<i32> residual_x;      // size count-1 (empty if count <= 1)
    std::vector<i32> residual_y;
};

RodJoint2DResult rod_joint_2d_forward(const std::vector<Point2i>& points);
std::vector<Point2i> rod_joint_2d_inverse(const RodJoint2DResult& r);

struct RodJoint3DResult {
    RodJoint2DResult xy;
    LiftResult lift_z;
};

RodJoint3DResult rod_joint_3d_forward(const std::vector<Point3i>& points);
std::vector<Point3i> rod_joint_3d_inverse(const RodJoint3DResult& r);

} // namespace csa
