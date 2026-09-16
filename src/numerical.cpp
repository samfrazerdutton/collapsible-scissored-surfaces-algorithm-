#include "csa/numerical.hpp"
#include "csa/thread_pool.hpp"
#include <stdexcept>

namespace csa {

std::vector<double> laplacian_1d_scalar(const std::vector<double>& f, double dx) {
    size_t n = f.size();
    std::vector<double> out(n);
    double inv_dx2 = 1.0 / (dx * dx);
    for (size_t i = 0; i < n; i++) {
        double left = f[(i + n - 1) % n];
        double right = f[(i + 1) % n];
        out[i] = (left + right - 2.0 * f[i]) * inv_dx2;
    }
    return out;
}

namespace {

void laplacian_row(const Grid2D& f, double inv_dx2, size_t y, Grid2D& out) {
    for (size_t x = 0; x < f.width; x++) {
        double c = f.at(x, y);
        double sum4 = f.at_wrapped((long)x + 1, (long)y) + f.at_wrapped((long)x - 1, (long)y) +
                      f.at_wrapped((long)x, (long)y + 1) + f.at_wrapped((long)x, (long)y - 1);
        out.at(x, y) = (sum4 - 4.0 * c) * inv_dx2;
    }
}

void gradient_row(const Grid2D& f, double inv_2dx, size_t y, Grid2D& gx, Grid2D& gy) {
    for (size_t x = 0; x < f.width; x++) {
        gx.at(x, y) = (f.at_wrapped((long)x + 1, (long)y) - f.at_wrapped((long)x - 1, (long)y)) * inv_2dx;
        gy.at(x, y) = (f.at_wrapped((long)x, (long)y + 1) - f.at_wrapped((long)x, (long)y - 1)) * inv_2dx;
    }
}

void divergence_row(const Grid2D& fx, const Grid2D& fy, double inv_2dx, size_t y, Grid2D& out) {
    for (size_t x = 0; x < fx.width; x++) {
        double dfx_dx = (fx.at_wrapped((long)x + 1, (long)y) - fx.at_wrapped((long)x - 1, (long)y)) * inv_2dx;
        double dfy_dy = (fy.at_wrapped((long)x, (long)y + 1) - fy.at_wrapped((long)x, (long)y - 1)) * inv_2dx;
        out.at(x, y) = dfx_dx + dfy_dy;
    }
}

// Classical stability bound for explicit-Euler diffusion with the 2D
// 5-point Laplacian stencil: the update multiplies each neighbor's
// contribution by k = alpha*dt/dx^2, and the scheme is stable (errors
// don't grow unboundedly) only while 4k <= 1, i.e. k <= 1/4. Checked
// here rather than left to silently blow up into NaN/Inf over many
// steps -- a real numerical-stability assumption, stated and enforced,
// not just assumed true.
void check_diffusion_stability(double alpha, double dt, double dx) {
    double k = alpha * dt / (dx * dx);
    if (k > 0.25)
        throw std::runtime_error("csa: diffusion parameters violate the explicit-Euler stability bound "
                                  "(alpha*dt/dx^2 must be <= 0.25 for this 5-point stencil)");
}

} // namespace

Grid2D laplacian_scalar(const Grid2D& f, double dx) {
    Grid2D out(f.width, f.height);
    double inv_dx2 = 1.0 / (dx * dx);
    for (size_t y = 0; y < f.height; y++) laplacian_row(f, inv_dx2, y, out);
    return out;
}

Grid2D laplacian_parallel(const Grid2D& f, double dx) {
    Grid2D out(f.width, f.height);
    double inv_dx2 = 1.0 / (dx * dx);
    parallel_for((int)f.height, [&](int y) { laplacian_row(f, inv_dx2, (size_t)y, out); });
    return out;
}

void gradient_scalar(const Grid2D& f, double dx, Grid2D& out_gx, Grid2D& out_gy) {
    out_gx = Grid2D(f.width, f.height);
    out_gy = Grid2D(f.width, f.height);
    double inv_2dx = 1.0 / (2.0 * dx);
    for (size_t y = 0; y < f.height; y++) gradient_row(f, inv_2dx, y, out_gx, out_gy);
}

void gradient_parallel(const Grid2D& f, double dx, Grid2D& out_gx, Grid2D& out_gy) {
    out_gx = Grid2D(f.width, f.height);
    out_gy = Grid2D(f.width, f.height);
    double inv_2dx = 1.0 / (2.0 * dx);
    parallel_for((int)f.height, [&](int y) { gradient_row(f, inv_2dx, (size_t)y, out_gx, out_gy); });
}

Grid2D divergence_scalar(const Grid2D& fx, const Grid2D& fy, double dx) {
    Grid2D out(fx.width, fx.height);
    double inv_2dx = 1.0 / (2.0 * dx);
    for (size_t y = 0; y < fx.height; y++) divergence_row(fx, fy, inv_2dx, y, out);
    return out;
}

Grid2D divergence_parallel(const Grid2D& fx, const Grid2D& fy, double dx) {
    Grid2D out(fx.width, fx.height);
    double inv_2dx = 1.0 / (2.0 * dx);
    parallel_for((int)fx.height, [&](int y) { divergence_row(fx, fy, inv_2dx, (size_t)y, out); });
    return out;
}

Grid2D diffuse_scalar(const Grid2D& initial, double alpha, double dt, double dx, int steps) {
    check_diffusion_stability(alpha, dt, dx);
    double inv_dx2 = 1.0 / (dx * dx);
    Grid2D cur = initial, next(initial.width, initial.height);
    for (int s = 0; s < steps; s++) {
        for (size_t y = 0; y < cur.height; y++) {
            for (size_t x = 0; x < cur.width; x++) {
                double c = cur.at(x, y);
                double sum4 = cur.at_wrapped((long)x + 1, (long)y) + cur.at_wrapped((long)x - 1, (long)y) +
                              cur.at_wrapped((long)x, (long)y + 1) + cur.at_wrapped((long)x, (long)y - 1);
                next.at(x, y) = c + dt * alpha * (sum4 - 4.0 * c) * inv_dx2;
            }
        }
        std::swap(cur, next);
    }
    return cur;
}

Grid2D diffuse_parallel(const Grid2D& initial, double alpha, double dt, double dx, int steps) {
    check_diffusion_stability(alpha, dt, dx);
    double inv_dx2 = 1.0 / (dx * dx);
    Grid2D cur = initial, next(initial.width, initial.height);
    for (int s = 0; s < steps; s++) {
        parallel_for((int)cur.height, [&](int yi) {
            size_t y = (size_t)yi;
            for (size_t x = 0; x < cur.width; x++) {
                double c = cur.at(x, y);
                double sum4 = cur.at_wrapped((long)x + 1, (long)y) + cur.at_wrapped((long)x - 1, (long)y) +
                              cur.at_wrapped((long)x, (long)y + 1) + cur.at_wrapped((long)x, (long)y - 1);
                next.at(x, y) = c + dt * alpha * (sum4 - 4.0 * c) * inv_dx2;
            }
        });
        std::swap(cur, next);
    }
    return cur;
}

} // namespace csa
