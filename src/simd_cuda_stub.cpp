// Built only when CSA is compiled without CUDA (WITH_CUDA=OFF or no CUDA
// compiler found) -- mirrors src/pantograph_lift_cuda_stub.cpp's reasoning:
// keeps the API available so callers don't need #ifdefs, and gives the
// linker a symbol to resolve even though the runtime cuda_is_available()
// guard means this body is never actually reached on such a build.
#include "csa/simd_cuda.hpp"

namespace csa {

bool max_abs_diff_i32_cuda(const i32*, const i32*, size_t, u32&) {
    return false;
}

} // namespace csa
