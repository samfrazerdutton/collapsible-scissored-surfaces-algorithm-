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

// One-shot call: allocates its own device (and pinned host) buffers and
// frees them before returning. Simple and correct, but every call pays
// cudaMalloc/cudaFree overhead regardless of whether the caller is about
// to call it again with a similarly-sized input right after.
bool pantograph_lift_forward_cuda(const std::vector<i32>& input, LiftResult& out);
bool cuda_is_available();

// A reusable GPU session: device and pinned-host buffers are allocated
// once and grown (never shrunk) as needed across calls, so repeated
// forward() calls on inputs no larger than a previous one pay zero
// cudaMalloc/cudaFree/cudaMallocHost/cudaFreeHost calls -- just the
// actual H2D/D2H transfers and kernel launches. The constructor also pays
// CUDA's one-time context-creation cost up front (see GPU_BENCHMARKS.md's
// "first call" finding) rather than folding it into the first forward().
//
// Not thread-safe: a session's buffers aren't safe to drive from two
// threads at once. Give each thread its own session (e.g. thread_local),
// the same way codec.cpp's compress() does.
class CudaLiftSession {
public:
    CudaLiftSession();
    ~CudaLiftSession();
    CudaLiftSession(const CudaLiftSession&) = delete;
    CudaLiftSession& operator=(const CudaLiftSession&) = delete;

    // Same contract as pantograph_lift_forward_cuda(); returns false if no
    // CUDA device is available (the session still exists and is safe to
    // reuse/destroy, it just never allocated anything).
    bool forward(const std::vector<i32>& input, LiftResult& out);

private:
    struct Impl;
    Impl* impl_; // pimpl: keeps cuda_runtime.h out of this public header
};

} // namespace csa
