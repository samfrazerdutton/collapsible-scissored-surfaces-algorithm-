// csa/codec.hpp — top-level container format tying the transforms and the
// range coder together into a single compress()/decompress() API, plus a
// raw-storage fallback mode so the codec can never blow up incompressible
// input by more than a small fixed header.
#pragma once
#include "csa/common.hpp"
#include "csa/rod_joint_transform.hpp"
#include <vector>

namespace csa {

enum class Mode : u8 { Raw = 0, General = 1, Geo2D = 2, Geo3D = 3 };

// General-purpose byte-stream compression (Pantograph Lift + range coding),
// with automatic raw-storage fallback if that would be smaller. When
// use_gpu is true and a CUDA device is available, the forward transform
// runs on the GPU (level-parallel); otherwise it transparently falls back
// to the CPU implementation. The bitstream is identical either way.
std::vector<u8> compress(const std::vector<u8>& input, bool use_gpu = false);
std::vector<u8> decompress(const std::vector<u8>& blob);

// Literal geometric mode for 2D point sequences (Rod-Joint Transform).
std::vector<u8> compress_geo2d(const std::vector<Point2i>& points);
std::vector<Point2i> decompress_geo2d(const std::vector<u8>& blob);

// Axis-wise geometric mode for 3D point sequences.
std::vector<u8> compress_geo3d(const std::vector<Point3i>& points);
std::vector<Point3i> decompress_geo3d(const std::vector<u8>& blob);

} // namespace csa
