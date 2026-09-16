// csa/numerical.hpp — a small, real numerical-computing lab: finite-
// difference stencils, gradient/divergence/Laplacian, and an explicit
// diffusion simulation, each with a scalar reference implementation and
// a row-blocked parallel one (via csa::parallel_for). This is not an
// attempt to make CSA a CFD solver -- it exists to demonstrate the same
// loop this codebase's actual codec kernels already went through
// (mathematical formulation -> scalar reference -> parallel
// implementation -> validation -> benchmark) on a workload whose math is
// small enough to state in one line and check by hand.
//
// All grids use periodic (wraparound) boundary conditions, chosen
// deliberately for one concrete reason: a periodic domain has no special
// edge cases in the stencil formula (every cell, including row/column 0
// and the last one, uses the exact same 5-point/3-point form via modular
// indexing), and periodic diffusion exactly conserves total mass
// (sum of all cell values) up to floating-point rounding -- a real,
// checkable physical invariant this header's own tests verify, not just
// "doesn't crash." A Dirichlet (frozen-edge) or Neumann (reflective-edge)
// boundary would each need separate edge-handling code and would not
// give this same clean conservation property to test against.
//
// Parallelism model: every kernel here computes each output cell from
// only the *input* grid (never partially-updated output), so splitting
// rows across threads via csa::parallel_for is embarrassingly parallel
// with zero risk of a data race or a numerically-different result --
// the parallel and scalar paths are required to produce bit-identical
// output for the same input, not just "close," and this is verified
// directly in tests/test_main.cpp, not assumed from the design.
#pragma once
#include "csa/common.hpp"
#include <cstddef>
#include <vector>

namespace csa {

// Row-major dense scalar field on a periodic width x height grid.
struct Grid2D {
    size_t width = 0, height = 0;
    std::vector<double> data;

    Grid2D() = default;
    Grid2D(size_t w, size_t h) : width(w), height(h), data(w * h, 0.0) {}

    double& at(size_t x, size_t y) { return data[y * width + x]; }
    double at(size_t x, size_t y) const { return data[y * width + x]; }

    // Periodic (wraparound) neighbor access -- the one place the
    // boundary condition described above is actually implemented; every
    // kernel below reads neighbors exclusively through this, so there is
    // exactly one place to change if a different boundary condition is
    // ever needed.
    double at_wrapped(long x, long y) const {
        long wx = ((x % (long)width) + (long)width) % (long)width;
        long wy = ((y % (long)height) + (long)height) % (long)height;
        return data[(size_t)wy * width + (size_t)wx];
    }

    double sum() const {
        double s = 0.0;
        for (double v : data) s += v;
        return s;
    }
};

// ---- 1D stencil: the same idea as the 2D Laplacian below, at the
// simplest possible dimensionality (the standard 3-point second-
// derivative stencil, periodic). ----
std::vector<double> laplacian_1d_scalar(const std::vector<double>& f, double dx);

// ---- 2D Laplacian: the standard 5-point stencil, del^2 f = (f[x+1,y] +
// f[x-1,y] + f[x,y+1] + f[x,y-1] - 4*f[x,y]) / dx^2. ----
Grid2D laplacian_scalar(const Grid2D& f, double dx);
Grid2D laplacian_parallel(const Grid2D& f, double dx);

// ---- Gradient: central difference, df/dx = (f[x+1]-f[x-1])/(2*dx),
// same for df/dy. Returns the two component grids. ----
void gradient_scalar(const Grid2D& f, double dx, Grid2D& out_gx, Grid2D& out_gy);
void gradient_parallel(const Grid2D& f, double dx, Grid2D& out_gx, Grid2D& out_gy);

// ---- Divergence of a 2D vector field (fx, fy): div F = dfx/dx + dfy/dy,
// each term the same central difference as gradient() above. ----
Grid2D divergence_scalar(const Grid2D& fx, const Grid2D& fy, double dx);
Grid2D divergence_parallel(const Grid2D& fx, const Grid2D& fy, double dx);

// ---- Explicit-Euler diffusion: du/dt = alpha * del^2 u, discretized as
// u_next = u + dt * alpha * laplacian(u). The classical stability bound
// for this exact scheme (2D, 5-point stencil, explicit Euler) is
// alpha*dt/dx^2 <= 1/4 -- checked and thrown on violation, not silently
// left to produce a numerically blown-up result. Runs `steps` iterations
// and returns the final grid (ping-ponging between two buffers
// internally so no in-place read/write hazard exists within a step).
Grid2D diffuse_scalar(const Grid2D& initial, double alpha, double dt, double dx, int steps);
Grid2D diffuse_parallel(const Grid2D& initial, double alpha, double dt, double dx, int steps);

} // namespace csa
