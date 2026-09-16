# Parallelism in CSA

**Status: covers exactly one real data-parallel primitive in this codebase
today (the interleaved rANS entropy backend), plus an explicit account of
why the entropy coder actually used by `compress()`/`compress_pose()`/
`squeeze()` is not parallelized the same way. This is not a roadmap
document dressed up as a status report -- everything below either has
code and a measurement behind it, or is explicitly labeled as not yet
built.**

## What's actually parallel, and why it's safe

`src/rans_coder.cpp`'s interleaved rANS format (`encode_interleaved_rans`/
`decode_interleaved_rans`, exposed via the C ABI as `csa_rans_encode`/
`csa_rans_decode`) splits its input into `num_lanes` contiguous,
independent chunks, each encoded or decoded against a **shared, static,
read-only** frequency table built once up front. That's what makes this
specific format safe to parallelize with plain OS threads and no
synchronization inside the hot loop: no lane's encode/decode state
depends on any other lane's progress or on the order lanes finish in.

As of this pass, that parallel dispatch (`run_lanes()`) goes through a
real, reusable thread pool (`include/csa/thread_pool.hpp`,
`csa::ThreadPool` / `csa::parallel_for` / `csa::default_thread_pool()`)
instead of spawning and joining raw `std::thread` objects on every single
call, which is what this function did before. The pool's worker threads
are created once (lazily, on first use, sized to
`std::thread::hardware_concurrency()`) and reused for the life of the
process -- real OS thread creation/teardown cost is paid once, not once
per `compress`/`decompress` call, which matters for any long-running
process (a server, a batch job, this project's own `webapp/server.py`)
compressing many files in sequence.

**Determinism**: re-running `encode_interleaved_rans` with the same
input and lane count produces byte-identical output regardless of
thread-scheduling order, because each lane only ever writes to its own
disjoint output range. Verified directly in `tests/test_main.cpp`
(`test_rans_coder`'s determinism check at 1/4/8 lanes) rather than
assumed from the design.

## What's *not* parallel, and why -- honestly

The entropy coder `compress()`, `compress_pose()`, `compress_geo2d()`,
`compress_geo3d()` and their lossy variants actually use in their real
output (`range_encode_bytes`, an adaptive order-1 range coder --
`include/csa/range_coder.hpp`) has a genuine sequential dependency: each
symbol's code depends on a running frequency table built from every
symbol that came before it in the stream. That adaptivity is exactly
what makes it competitive with the interleaved rANS format in the first
place (see the `test_rans_coder` benchmark output: on real skewed data,
the adaptive range coder is *smaller* than 4-lane rANS, not just close --
splitting into independent lanes throws away the cross-lane adaptation).
Parallelizing it block-by-block the way the rANS lanes are split would
mean either (a) resetting the adaptive model at each block boundary,
which changes the actual compressed output and ratio, not just its
speed -- a real format change, not a free optimization -- or (b) finding
a way to parallelize maintaining one shared adaptive model across
threads, which is a substantially harder synchronization problem with no
existing solution in this codebase. Neither is attempted here. This is
the direct answer to "why doesn't squeeze() get faster with more
threads": **it doesn't, on purpose, because the honest alternative would
either cost compression ratio or requires real synchronization design
work not yet done.**

Practical consequence: `scissorc scale-test` (below) measures the
interleaved rANS backend specifically, because it is the only thing in
this codebase there currently is a real scaling story to tell. Wiring
interleaved rANS in as a selectable, opt-in entropy backend for the main
codec paths -- trading a small ratio cost (see the benchmark above) for
real multi-core throughput on large payloads -- is a real, concrete,
currently-unimplemented next step; it is not done in this pass and
nothing in the CLI, Python bindings, or browser build claims otherwise.

## Real measured scaling

`scissorc scale-test [--bytes N] [--repeats N] [--json]` runs the
interleaved rANS backend at 1/2/4/8/`hardware_concurrency()` lanes on a
real generated buffer (a skewed-but-nontrivial byte distribution, not
all-zeros), median of several repeated runs per lane count, and reports
real `speedup(N) = T(1)/T(N)` and `efficiency(N) = speedup(N)/N`. Every
number below is that command's real output, not a formula applied to a
single guessed baseline -- reproduce it yourself with the command shown.

**Windows 11, MSVC 19.29 Release, 16 logical cores** (`scissorc
scale-test`, default 4,000,000-byte buffer):

| lanes | encode (ms) | encode speedup | encode efficiency | decode (ms) | decode speedup | decode efficiency |
|---|---|---|---|---|---|---|
| 1  | 21.80 | 1.00 | 1.00 | 19.21 | 1.00 | 1.00 |
| 2  | 14.68 | 1.49 | 0.74 | 10.93 | 1.76 | 0.88 |
| 4  | 11.14 | 1.96 | 0.49 |  6.64 | 2.89 | 0.72 |
| 8  |  9.77 | 2.23 | 0.28 |  4.44 | 4.32 | 0.54 |
| 16 |  9.23 | 2.36 | 0.15 |  3.45 | 5.56 | 0.35 |

**WSL2 Ubuntu, GCC 13.3.0 Release (`-DWITH_CUDA=OFF`), same physical
machine, 16 logical cores** (identical command):

| lanes | encode (ms) | encode speedup | encode efficiency | decode (ms) | decode speedup | decode efficiency |
|---|---|---|---|---|---|---|
| 1  | 22.59 | 1.00 | 1.00 | 20.32 | 1.00 | 1.00 |
| 2  | 18.66 | 1.21 | 0.61 | 13.62 | 1.49 | 0.75 |
| 4  |  8.85 | 2.55 | 0.64 |  6.32 | 3.21 | 0.80 |
| 8  |  8.24 | 2.74 | 0.34 |  4.41 | 4.61 | 0.58 |
| 16 |  6.19 | 3.65 | 0.23 |  3.21 | 6.34 | 0.40 |

**The real, honest pattern in both runs, on both platforms**: decode
scales substantially better than encode, and neither scales linearly
past 4-8 lanes. Two concrete, explainable (not hand-waved) reasons:

1. **Encode has a serial preprocessing step decode doesn't.**
   `encode_interleaved_rans` builds its byte-frequency histogram with a
   single sequential pass over the *entire* input (`for (u8 b : data)
   hist[b]++;`) *before* `run_lanes()` even starts -- that scan is not
   parallelized in this pass, and by Amdahl's law it puts a hard ceiling
   on encode's achievable speedup regardless of lane count. Decode reads
   the already-built frequency table straight out of the blob header, so
   it has no equivalent serial phase, which is the direct, measured
   reason its speedup and efficiency numbers are consistently better
   than encode's at every lane count on both platforms above.
2. **Diminishing efficiency past 4-8 lanes is a real, expected
   consequence of a small, fixed 4,000,000-byte buffer split across more
   lanes than that**: each lane's chunk shrinks as lane count grows, so
   fixed per-task overhead (pool dispatch, `std::future` bookkeeping, the
   table-build cost inside each lane) becomes a proportionally larger
   share of a smaller unit of real work -- classic strong-scaling
   behavior, not a defect in the pool. A `--bytes` value large enough
   that each lane still does substantial work at 16 lanes would be
   expected to show flatter efficiency; that specific claim is not yet
   separately measured in this pass and should not be assumed true
   without running `scale-test --bytes <larger>` and looking.

## Heterogeneous computing that already existed before this pass

CUDA acceleration (`cuda/pantograph_lift_cuda.cu`,
`cuda/quaternion_calibration_cuda.cu`, `CudaLiftSession`,
`cuda_is_available()`) is real, pre-existing GPU work in this codebase,
with a CPU fallback stub (`src/pantograph_lift_cuda_stub.cpp`) so the
project builds and runs correctly with `WITH_CUDA=OFF` (the default on
CI's `ubuntu-latest`, and on any machine without an NVIDIA GPU). See
`GPU_BENCHMARKS.md` for its own real measured numbers (warm-up cost vs.
steady-state per-call cost, one-shot vs. persistent-session comparison)
-- not reproduced here.

## What this pass does not add (disclosed, not silently skipped)

- **SIMD/vectorization**: none added. The clearest candidate loops
  (`pick_block_lag_2d`'s per-candidate-lag SSE search in
  `src/rod_joint_transform.cpp`/`src/quaternion_joint.cpp`) are called
  very frequently on small, fixed-size candidate sets -- exactly the
  shape of workload where OS-thread-level parallelism would lose to its
  own dispatch overhead (this is *why* they aren't in `scale-test`
  above), but they are a real, plausible future SIMD target. Not
  attempted in this pass; would need real correctness verification
  against the scalar path before being trusted.
- **Work-stealing scheduler**: not built. The current workload (a
  handful of same-sized independent lane tasks per call) has no measured
  load-imbalance problem for a work-stealing queue to solve; see
  `include/csa/thread_pool.hpp`'s own header comment for the reasoning
  against building one speculatively.
- **Wiring interleaved rANS into the main compress paths as a
  throughput-optimized alternative to the adaptive range coder**: a real,
  concrete, unimplemented next step (see above), not attempted here.
- **Distributed/multi-process execution, GPU-accelerated quantization/
  error-measurement kernels beyond the existing Pantograph Lift/
  quaternion-calibration CUDA kernels, out-of-core/streaming processing**:
  none of this exists yet. Treat any claim otherwise, anywhere in this
  repository, as a documentation bug -- please file it.
