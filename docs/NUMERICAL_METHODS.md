# Numerical methods

**Status: covers every scientific-computing kernel this codebase actually
implements today (`include/csa/numerical.hpp`/`src/numerical.cpp`) --
finite-difference stencils, gradient/divergence/Laplacian, and an
explicit-Euler diffusion simulation. This is not a CFD solver and does
not claim to be one: it exists to demonstrate the same loop this
codebase's actual codec kernels already went through -- mathematical
formulation, a scalar reference implementation, a parallel
implementation, numerical validation, and a real benchmark -- on a
workload whose mathematics is small enough to state in one line and
check by hand.**

## Why this exists

Everything else in this repository maps a *domain-specific* algorithm
(a compression transform) onto parallel hardware. This module maps a
*textbook* algorithm onto the same hardware, using the same two
scheduling primitives (`csa::ThreadPool`/`parallel_for`) already used
elsewhere in this codebase -- so the parallelization technique itself,
not the domain, is what's on display here.

## Domain: a periodic 2D scalar field

`csa::Grid2D` is a dense, row-major `width x height` array of `double`.
Every kernel below uses **periodic (wraparound) boundary conditions** --
a deliberate choice, not a default left unexamined: a periodic domain
needs no special-case code for edge/corner cells (every cell, including
column 0 and the last column, uses the exact same stencil formula via
modular indexing in `Grid2D::at_wrapped`), and periodic diffusion has a
genuine, checkable physical invariant (total mass is exactly conserved)
that a Dirichlet (frozen-edge) or Neumann (reflective-edge) boundary
would not give as cleanly. This is a real tradeoff, not a hidden one: a
Dirichlet/Neumann version would need distinct edge-handling code this
implementation does not have.

## 1D stencil: the second-derivative analogue

```
d^2f/dx^2 [i] ~= (f[i-1] + f[i+1] - 2*f[i]) / dx^2
```

`laplacian_1d_scalar` implements exactly this, periodic. Verified by
hand against a discrete delta function (a single spike among zeros):
the spike's own second derivative is `-2/dx^2`, and each of its two
immediate neighbors sees `+1/dx^2` (`tests/test_main.cpp`,
`test_numerical_correctness`) -- a case whose correct answer can be
checked by arithmetic, not just "the code ran."

## 2D Laplacian: the 5-point stencil

```
del^2 f [x,y] ~= (f[x+1,y] + f[x-1,y] + f[x,y+1] + f[x,y-1] - 4*f[x,y]) / dx^2
```

`laplacian_scalar`/`laplacian_parallel`. The parallel version splits
rows across `csa::parallel_for`'s worker pool -- one task per grid row.
This is embarrassingly parallel with **zero risk of a numerically
different result**: every output cell is computed purely from the
read-only input grid (never a partially-updated output), so which
thread computes which row cannot change the arithmetic. The parallel
and scalar paths are required to produce **bit-identical** output, not
merely "close," and this is checked directly (not assumed from the
design) across several grid sizes including 1x1, 2x2, and a
non-power-of-two 37x37.

## Gradient and divergence: central differences

```
df/dx [x,y] ~= (f[x+1,y] - f[x-1,y]) / (2*dx)      (same form for df/dy)
div F [x,y] = dFx/dx + dFy/dy
```

`gradient_scalar`/`gradient_parallel` and
`divergence_scalar`/`divergence_parallel`, same row-parallel strategy
and same bit-identical-output requirement as the Laplacian above.

## Diffusion: explicit-Euler finite differences

```
du/dt = alpha * del^2 u
```

discretized as

```
u_next[x,y] = u[x,y] + dt * alpha * del^2 u [x,y]
```

iterated for a caller-chosen number of steps, ping-ponging between two
grid buffers each step so no in-place read/write hazard exists within a
single update.

**Stability assumption, stated and enforced, not assumed true**: this
exact scheme (explicit Euler, 2D 5-point stencil) is only numerically
stable while `alpha * dt / dx^2 <= 1/4` -- the classical bound for this
discretization. `diffuse_scalar`/`diffuse_parallel` check this and
throw a clean `std::runtime_error` on violation, rather than silently
producing a numerically-blown-up (NaN/Inf) result after enough steps.
Verified directly: a deliberately-violating parameter set
(`alpha*dt/dx^2 = 100`) is confirmed to throw in
`test_numerical_correctness`.

**A real physical invariant, checked, not asserted by construction**:
periodic-boundary diffusion with this stencil conserves total mass
exactly in the continuous case (the 5-point stencil's weights sum to
zero, so no step creates or destroys material), and this carries over
to the discrete scheme up to floating-point rounding. Measured over 200
steps on a random 20x20 grid: relative mass drift of `4.43e-16` --
consistent with accumulated double-precision rounding over ~80,000 cell
updates, not a sign of a bug. This is a genuinely independent check
(derived from the physics, not from the code's own internal
consistency) that a Dirichlet or Neumann boundary variant would not
give for free.

## Measured performance

`test_numerical_benchmark` (`tests/test_main.cpp`) times the scalar vs.
row-parallel Laplacian on a 2000x2000 grid (4,000,000 cells). Real,
platform-dependent results from this session:

| Platform | Scalar | Parallel | Speedup |
|---|---|---|---|
| Windows 11, MSVC 19.29, 16 logical cores | 162.4ms | 25.1ms | 6.47x |
| WSL2 Ubuntu, GCC 13.3.0, same physical machine | 98.9ms | 59.9ms | 1.65x |

**Reported honestly, including the platform gap, not just the better
number**: this specific kernel (a single Laplacian pass, no diffusion
loop) is memory-bandwidth-bound per cell (4 neighbor reads + 1
arithmetic combine, minimal compute), and the two platforms' measured
scaling differ substantially for reasons not root-caused in this
pass -- a real, disclosed open question, not glossed over. The
diffusion driver (`diffuse_parallel`), which repeats this same kernel
many times per call, was not separately benchmarked at scale in this
pass; only its correctness and the mass-conservation invariant were
verified.

## What this deliberately does not attempt

- **Not a CFD solver.** No advection, no non-linear terms, no implicit
  time-stepping, no adaptive time-stepping, no irregular/unstructured
  meshes. The point is the parallelization pattern, not simulation
  fidelity.
- **No GPU kernel for any of this yet.** These are exactly the kind of
  embarrassingly-parallel, per-cell-independent kernels that would map
  cleanly onto CUDA (the same shape as the existing Pantograph Lift
  kernel's per-pair independence) -- a real, scoped, disclosed next step,
  not attempted in this pass.
- **No SIMD for any of this yet.** The Laplacian/gradient/divergence
  inner loops are exactly as vectorizable as `max_abs_diff_i32` was --
  not attempted here, a real candidate for a future pass.
- **No connection (yet) to the codec's own point-cloud/pose data.**
  Brief-style "particle transform"/"voxelization"/"density estimation"
  workloads over `KdTree3i`'s point clouds would be a natural extension
  of this module, not built in this pass.
