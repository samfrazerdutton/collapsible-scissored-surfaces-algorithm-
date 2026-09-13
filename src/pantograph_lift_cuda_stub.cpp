// Built only when CSA is compiled without CUDA (WITH_CUDA=OFF or no CUDA
// compiler found). Keeps the API available so callers don't need #ifdefs.
#include "csa/pantograph_lift_cuda.hpp"

namespace csa {

bool cuda_is_available() { return false; }

bool pantograph_lift_forward_cuda(const std::vector<i32>&, LiftResult&) {
    return false;
}

struct CudaLiftSession::Impl {};

CudaLiftSession::CudaLiftSession() : impl_(nullptr) {}
CudaLiftSession::~CudaLiftSession() {}

bool CudaLiftSession::forward(const std::vector<i32>&, LiftResult&) {
    return false;
}

} // namespace csa
