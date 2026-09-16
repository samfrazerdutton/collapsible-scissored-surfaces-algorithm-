// Built only when CSA is compiled without CUDA (WITH_CUDA=OFF or no CUDA
// compiler found). Keeps the API available so callers don't need #ifdefs.
#include "csa/pantograph_lift_cuda.hpp"
#include "csa/quaternion_calibration_cuda.hpp"

namespace csa {

bool cuda_is_available() { return false; }

bool cuda_device_info(std::string&, u64&) { return false; }

bool pantograph_lift_forward_cuda(const std::vector<i32>&, LiftResult&) {
    return false;
}

struct CudaLiftSession::Impl {};

CudaLiftSession::CudaLiftSession() : impl_(nullptr) {}
CudaLiftSession::~CudaLiftSession() {}

bool CudaLiftSession::forward(const std::vector<i32>&, LiftResult&) {
    return false;
}

// Never actually reached (every caller checks cuda_is_available() first --
// see tests/test_main.cpp's test_quat_calibration_cuda_matches_cpu()), but
// the symbol still has to exist for a CPU-only build to *link*: a runtime
// guard doesn't help the linker resolve a reference to a function that
// exists in only one of two possible translation units. This gap was
// invisible on this project's own Windows/MSVC dev machine, which always
// had CUDA available -- caught only by reproducing a real CPU-only Linux
// build (WSL) after this project's first real CI run failed here.
bool quat_calibrate_all_blocks_cuda(const std::vector<i32>&, const std::vector<i32>&,
                                     const std::vector<i32>&, const std::vector<i32>&,
                                     size_t, const std::vector<u32>&,
                                     std::vector<std::array<i64, 4>>&,
                                     std::vector<double>&) {
    return false;
}

} // namespace csa
