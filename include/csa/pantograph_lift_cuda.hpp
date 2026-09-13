// csa/pantograph_lift_cuda.hpp — optional GPU-accelerated Pantograph Lift
// forward transform. Each decomposition level processes N/2 independent
// scissor-joint pairs; that independence is exactly what makes the level
// parallel across GPU threads (no cross-thread dependency within a level,
// only between levels, which stay sequential on the host).
//
// Returns false (and leaves `out` unspecified) if no CUDA device is
// available at runtime, or CSA was built with WITH_CUDA=OFF, so callers can
// transparently fall back to pantograph_lift_forward() on the CPU.
#pragma once
#include "csa/pantograph_lift.hpp"

namespace csa {

bool pantograph_lift_forward_cuda(const std::vector<i32>& input, LiftResult& out);
bool cuda_is_available();

} // namespace csa
