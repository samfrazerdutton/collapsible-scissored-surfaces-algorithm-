# Engineering Audit

**Status: Phase 0 of the "CSA Spatial Lab" transformation brief. Every
claim below was checked in this pass -- built, run, or read directly --
not carried over from older docs on trust. Where a check wasn't
re-verified this pass, it's marked "not re-verified this pass" rather
than silently assumed current. Dated 2026-09-16, against commit
`a2d6477` (the most recent pushed commit at audit time).**

This document exists to drive what gets built next (per the brief's own
Phase 1-16 ordering), not to stand alone as a report. See the bottom
section for the prioritized gap list this audit produced.

## 1. Current architecture

A three-layer C ABI design, not a single monolith:

- **`csa_core`** (static lib): the actual codec -- range coder, Pantograph
  Lift, Rod-Joint/Quaternion-Joint transforms, interleaved rANS, LZ/BWT
  candidates, pose streaming, packet transport, a spatial index (KD-tree),
  SIMD (AVX2 + CUDA backends for one kernel), a thread pool and a
  work-stealing pool. Pure C++17, no C ABI boundary inside this layer.
- **`csa_capi`** (shared lib, C ABI): `include/csa/csa_capi.h` is the one
  contract every non-C++ consumer (Python, Rust, C#, Go, WASM) depends on.
  `struct csa_buffer { unsigned char* data; size_t size; }` plus
  `csa_compress`/`csa_decompress`/`csa_compress_geo2d`/etc. No C++ types
  cross this boundary, by design (see `CMakeLists.txt` line ~43).
- **`scissorc`** (CLI executable): links `csa_core` directly (not through
  the C ABI, since it's C++ itself) -- `cli/main.cpp`, 1,515 lines, the
  thickest single file in the repo. Implements the actual sniffing/
  auto-scale/round-trip-verify logic that every other frontend (the
  browser demo's JS, `webapp/`) either shells out to or re-implements.
- **Four language bindings** (`bindings/{python,rust,csharp,go}/`), all
  thin wrappers over `csa_capi` per `CONTRIBUTING.md`'s explicit rule
  ("no reimplemented logic that could drift from the C++ core").
- **Two browser frontends**: `docs/index.html` (zero-backend, GitHub
  Pages-hosted, the actively-maintained one) and `demo/csa_demo.html`
  (an Artifact-sandboxed version with its own WebGPU rANS decode port).
  Both embed a compiled WASM build of `csa_core` + `csa_capi` +
  `wasm_shim.cpp` (a WASM-specific re-export layer, see section 14).
- **`webapp/`**: a thin Flask server that shells out to the built
  `scissorc` binary rather than reimplementing anything -- "exactly one
  place that logic lives and exactly one place it's tested" (DESIGN.md).

## 2. Current build graph

Single top-level `CMakeLists.txt` (~220 lines). Key facts, verified by
rebuilding from scratch this pass:

- `WITH_CUDA` option, `ON` by default, auto-detects via
  `check_language(CUDA)` and falls back to `CSA_HAVE_CUDA=0` with a
  warning if no CUDA compiler is found -- not a hard failure.
- `CSA_SANITIZE` cache option (`address`, `undefined`, `thread`, or
  empty) -- GCC/Clang only, MSVC warns if something other than `address`
  is requested.
- `miniz` is pulled via `FetchContent` (deprecated `FetchContent_Populate`
  call -- CMake emits a real warning about this every configure; a real,
  minor, easy fix not yet made) purely for `scissorc benchmark`'s
  gzip-equivalent baseline -- not linked into `csa_core` or `csa_capi`.
- Native targets verified building clean this pass: `csa_core`,
  `csa_capi`, `csa_capi_test`, `csa_tests`, `scissorc`,
  `quat_calibration_gpu_crossover` (Windows/MSVC, CUDA available).
- WASM is **not** part of this CMake project at all -- built via a
  separate, manual `em++` invocation (`bench/build_wasm.sh`, which did
  not exist until this session; see section 17).

## 3. Current dependencies

- **Build-time, vendored**: miniz (FetchContent, CLI-only).
- **Build-time, system**: CMake >= 3.18ish (uses `check_language`), a
  C++17 compiler, optionally CUDA toolkit + `nvcc`, optionally Emscripten
  (`em++`) for the WASM build (not CMake-driven).
- **Runtime, native**: none beyond the OS/libc and (optionally) the CUDA
  runtime (`cudart`, dynamically resolved via `cuda_is_available()`'s
  `cudaGetDeviceCount` check, never a hard link-time requirement for the
  CPU-only fallback path).
- **Runtime, browser**: none -- `docs/index.html` is genuinely
  zero-backend (SINGLE_FILE WASM inlined as base64, no CDN scripts, no
  external fetch for the core codec path).
- **Python bindings**: stdlib `ctypes` only (verified by reading
  `bindings/python/csa.py`'s imports) -- no numpy/scipy dependency, which
  keeps the binding genuinely thin but also means no numpy-array
  interop convenience exists yet (a real, disclosed gap, not
  investigated further this pass).

## 4. Current executable targets

| Target | Verified this pass | Notes |
|---|---|---|
| `csa_tests` | Built + run clean, both platforms | 1824 checks (Windows/MSVC+CUDA), 1773 (WSL/GCC, no CUDA) |
| `csa_capi_test` | Built + run clean, both platforms | 30 checks both platforms |
| `scissorc` | Built + smoke-tested this pass | `squeeze`/`unsqueeze`/`inspect`/`verify`/`optimize`/`benchmark` all exercised against a real dataset (`bench/datasets/lidar_ring.xyz`) in this audit |
| `quat_calibration_gpu_crossover` | Builds; not re-run this pass | A standalone GPU-vs-CPU benchmark, only built when CUDA is available |
| `fuzz_decompress` | Builds via `fuzz/build_fuzzer.sh` (clang, WSL only) | Not part of the CMake project (deliberately -- fuzzing is a manual/scheduled activity, not a per-commit cost) |

## 5. Current libraries (by responsibility, not by file count)

20 `.cpp` files in `src/`, 23 headers in `include/csa/`. By area:
range coding + Order-1/OrderN adaptive models (`range_coder`), the four
geometric/pose transforms (`pantograph_lift`, `rod_joint_transform`,
`quaternion_joint`, `pose_stream`), interleaved rANS (`rans_coder`), an
LZ+BWT candidate pipeline (`lz_matcher`/`lz_codec`/`bwt_transform`/
`bwt_codec`), the top-level dispatcher (`codec.cpp`, the largest file at
last count, ~800+ lines -- the one file every deserializer's security
patches this session landed in), SIMD (`simd`/`simd_avx2`/
`simd_cuda_stub`), two scheduling primitives (`thread_pool.hpp`,
`work_stealing_pool.hpp` -- header-only, no `.cpp`), a packet transport
layer (`crc32`/`packet_transport`), and a spatial index
(`spatial_index`). CUDA-only sources live in `cuda/` (`.cu`), with a
combined CPU-fallback stub (`pantograph_lift_cuda_stub.cpp`) and a
separate one for the newer SIMD-CUDA kernel (`simd_cuda_stub.cpp`).

## 6. Current data formats

`FORMAT.md` (129 lines) documents the real, two-layer wire format:
Layer 1 (`"CSA1"` + a `Mode` enum byte: Raw/General/Geo2D/Geo3D/
GeneralLZ/GeneralBWT/Pose) is what the C++ library itself produces;
Layer 2 (`"CSAG"` + a dims byte + scale) is a CLI/browser-only framing
convention wrapping a Layer 1 blob, invented so `scissorc`/`docs/`
files carry the scale/dims metadata the raw library API takes as
separate parameters. **`FORMAT.md` explicitly documents a "known gap:
no CRC"** -- this is real and current: neither Layer 1 nor Layer 2 has
a built-in integrity check on the *codec's own* container (the newer
`packet_transport.hpp` layer added this session has its own CRC-32, but
that's a separate, optional layer above the format, not a change to
`.csa` files themselves). No explicit format version field beyond the
`Mode` byte -- brief section 33's "explicit versioning" ask is not
currently met structurally, only informally (a new `Mode` value is
itself a compatible extension, but there's no independent version
counter for non-mode-related format evolution).

## 7. Current benchmarks

Real, not synthetic-only: `REAL_GEO_BENCHMARK.md` (a real 693,895-point
Autzen LiDAR scan, LASzip-comparable), `REAL_POSE_BENCHMARK.md` (KITTI
vehicle trajectories, EuRoC/TUM-style pose datasets),
`ADVERSARIAL_BENCHMARK.md` (synthetic pathological inputs, explicitly
labeled as such), `GPU_BENCHMARKS.md` (measured GPU warm-up/crossover
costs, including an honest "couldn't reproduce the original number"
finding rather than re-asserting a stale one), `docs/PARALLELISM.md`
(thread-scaling, SIMD, work-stealing -- real numbers, two platforms),
`docs/SANITIZERS.md` (fuzzing/sanitizer results). `BENCHMARKS.md` at
the repo root is a shorter index/summary -- **not re-diffed against the
`REAL_*` docs this pass to confirm it isn't stale; a real candidate for
the next audit pass, not confirmed current or stale here.**

## 8. Current test coverage

`tests/test_main.cpp` (a single hand-rolled `CHECK()`-macro test
runner, not a framework like GoogleTest/Catch2) -- 1824 checks on
Windows/MSVC with CUDA, 1773 on WSL/GCC without, both 0 failures as of
this pass. `tests/test_capi.cpp` covers the C ABI specifically (30
checks both platforms). `bindings/python/test_bindings.py` (38 checks,
verified passing this pass). Rust/C#/Go binding tests exist per
`CONTRIBUTING.md`'s own text (`cargo test`, "the C#/Go test projects")
but **were not run this pass** -- a real gap in this audit, not a
claim of "all four bindings verified equally."

## 9. Current performance bottlenecks (measured, not guessed)

- The adaptive order-1 range coder (`range_encode_bytes`, the entropy
  backend `compress()`/`compress_pose()` actually use) has a genuine
  sequential dependency and is not parallelized -- documented explicitly
  in `docs/PARALLELISM.md` as a real, current limitation, not a gap
  someone forgot.
- `encode_interleaved_rans`'s encode path has a serial histogram-build
  pass over the entire input before any parallel work starts -- Amdahl's
  law caps its achievable speedup regardless of thread count (measured:
  encode scales worse than decode at every lane count, on both
  platforms).
- `max_abs_diff_i32_cuda` (this session's own addition) is a *known,
  measured, kept* GPU loss (~6x slower than AVX2) for exactly this
  reason -- documented as a deliberate negative result, not silently
  removed.

## 10. Current memory behavior

No dedicated memory profiling has been run (no Massif/Valgrind/ETW
capture in this repo's history that this audit found). What *has* been
verified: ASan's own allocation tracking caught the real decompression-
bomb bug this session (bug #2 in `docs/SANITIZERS.md`) and the
subsequent `kMaxRangeDecodedBytes`/`kMaxLiftLevels`/
`check_reasonable_count` caps bound worst-case allocation size for
every untrusted-input deserializer. No general "peak RSS per operation"
instrumentation exists in the CLI or library -- brief section 144's
"expose peak memory" ask is not met.

## 11. Current parallelism

Two real primitives, not one, each scoped to where it actually helps
(the deliberate contrast is itself documented in three places:
`docs/PARALLELISM.md`, `include/csa/thread_pool.hpp`'s header comment,
`include/csa/work_stealing_pool.hpp`'s header comment):

- `csa::ThreadPool` / `parallel_for` / `default_thread_pool()` --
  backs the interleaved rANS lane dispatch. Measured scaling: encode up
  to ~2.4-3.7x at 16 lanes, decode up to ~5.6-6.9x, on a real
  16-logical-core machine, both Windows/MSVC and WSL/GCC.
- `csa::WorkStealingPool` -- built, tested, measured (3.5-3.85x on a
  genuinely imbalanced synthetic workload), **not wired into any
  production call site** -- the rANS workload it could replace has no
  measured load imbalance to justify it. This is a real, current,
  disclosed gap between "exists" and "used."

## 12. Current SIMD/vectorization

`include/csa/simd.hpp`: one real kernel (`max_abs_diff_i32`, the geo2d/
geo3d round-trip error measurement used by `squeeze`/`optimize`/
`benchmark`), runtime CPUID-dispatched (AVX2 vs. scalar), 2.6-6.4x
measured speedup. `CSA_X86_SIMD` guards every AVX2-specific line so the
WASM build (wasm32) and any future ARM build compile correctly and fall
back to scalar. **Only one kernel is vectorized** -- `pick_block_lag_2d`
(the per-candidate-lag search in `rod_joint_transform.cpp`/
`quaternion_joint.cpp`) was identified as a plausible future SIMD
target in a prior pass and never attempted (still true today).

## 13. Current GPU support

Three CUDA kernels, all verified building and passing correctness
tests against CPU on this pass's Windows/CUDA run:

1. Pantograph Lift forward transform (`cuda/pantograph_lift_cuda.cu`) --
   GPU-resident across all levels, `CudaLiftSession` for
   amortized-allocation repeated calls. Has a real, measured CPU/GPU
   crossover story in `GPU_BENCHMARKS.md`.
2. Quaternion calibration (`cuda/quaternion_calibration_cuda.cu`) --
   cross-checked against the CPU path (24 blocks x 14 lags, 0
   mismatches, verified this pass).
3. `max_abs_diff_i32_cuda` (this session) -- a real, honest **GPU loss**
   (~6x slower than AVX2), kept and documented rather than hidden.

`WITH_CUDA=OFF` (or no CUDA compiler found) is a fully supported,
tested configuration -- CPU-fallback stubs exist for every CUDA entry
point (`pantograph_lift_cuda_stub.cpp`, `simd_cuda_stub.cpp`), and this
is in fact what CI's `ubuntu-latest` runner builds by default.

## 14. Current WASM architecture

`src/wasm_shim.cpp`: a thin re-export of `csa_capi.h` functions, with
one systematic change (struct-by-value returns become an explicit
out-param pointer, since Emscripten's struct lowering is invisible to
JS's `cwrap`/`ccall`). 17 exports as of this pass (14 pre-existing +
3 new this session for `KdTree3i`). **`bench/build_wasm.sh` did not
exist before this pass** -- the module had only ever been rebuilt by
hand, an undocumented, unreproducible process; this is now fixed (see
section 17's gap list, item now closed). Verified this pass: a from-
scratch WASM rebuild (`em++`, `MODULARIZE=1`, `SINGLE_FILE=1`,
`ALLOW_MEMORY_GROWTH=1`) produces a module where all 17 exports
round-trip correctly against both a pure-JS brute-force reference (new
KD-tree exports) and the pre-existing general/geo3d compress-decompress
paths (`bench/verify_wasm.mjs`, 4/4 checks pass).

**Real, current gap**: the freshly-built module (with the 3 new
KD-tree exports) is **not yet embedded** in `docs/index.html` or
`demo/csa_demo.html` -- both pages still run the older, 14-export
build. Verified independently via Node, not yet via a live browser page.

## 15. Current browser architecture

`docs/index.html` (1,917 lines, one file, no build step) is the
actively-maintained "product" surface -- a WASM core running inside a
Web Worker (message-passing: `cmd: 'squeeze'|'unsqueeze'|'profile'|
'optimize'`, progress messages for real sequential stages), a Format
Inspector, a Compression Lab, Auto-Optimize, Profile-before-compress,
Break the Codec, an Edge Link Simulator, a Local Trust Boundary
section, Research Benchmarks, Wins/Losses, and a CLI-parity section.
`demo/csa_demo.html` is a separate, Artifact-sandboxed page with its
own real WebGPU compute-shader rANS decode port (a genuine, working
in-browser parallel execution demo -- not a mockup) -- this exists
**already**, predating this session, and is a real, if under-surfaced,
asset the brief's Phase 10/140-142 asks partially duplicate.

**Real, current gap**: neither page exposes any of this session's new
backend work (work-stealing, GPU SIMD honest-loss result, packet
transport, spatial index) in its UI. The brief's "Parallel Computing
Lab" (section 4), "GPU Page" (section 77), "Numerical Page" (section
78), and most of the visualization/replay asks (sections 61-66) do not
exist in either page today.

## 16. Current API surface

C ABI (`csa_capi.h`): compress/decompress for general/geo2d/geo3d/pose
(lossy and lossless variants), rANS encode/decode, `cuda_available()`.
Python (`bindings/python/csa.py`): a `ctypes`-based wrapper matching
the C ABI 1:1, plus `inspect()`/`verify()` header-parsing helpers added
in an earlier phase. **No numerical/scientific-computing API exists**
(brief section 87's `csa.numerical.gradient/divergence/laplacian` --
none of this exists; see the gap list). **No `Dataset`-object-oriented
API** (brief section 86's `csa.load(...).analyze().compress(...)`
fluent style) -- the current API is a flat set of functions, not an
object model.

## 17. Current technical debt (real items, found and either fixed or logged this pass)

- ~~`wasm_shim.cpp` referenced `bench/build_wasm.sh` by name; the file
  didn't exist~~ **fixed this session** (script now exists, verified).
- ~~Several source comments referenced `docs/SANITIZERS.md` before it
  existed~~ **fixed in an earlier phase this session**.
- CMake's `FetchContent_Populate(miniz)` call uses a deprecated API
  (real warning on every configure) -- not fixed yet, low-risk, low-effort.
- No CRC/integrity field in the `.csa` format itself (`FORMAT.md`'s own
  documented gap) -- not fixed, would be a real format-version bump.
- `BENCHMARKS.md` (root) vs. the `REAL_*.md` docs -- not cross-checked
  for staleness this pass.
- Rust/C#/Go binding test suites exist but weren't run this pass.

## 18. Current broken functionality

**None found this pass.** Every target that was built, built clean;
every command that was run, ran correctly; every test suite that was
run, passed. This is a meaningfully different starting position than
the brief assumes by default ("do not assume existing documentation is
correct") -- the documentation's claims of a working, tested system
were checked, not just trusted, and held up.

## 19. Current misleading UI

Not found in this pass's spot-checks of `docs/index.html`'s described
feature set (Format Inspector, Auto-Optimize, Profile, Break the
Codec), but a full UI audit against every button/slider (brief
sections 183-186) was **not performed this pass** -- this is a real,
scoped item for the next audit pass, not a clean bill of health for
every pixel.

## 20. Current undocumented behavior

- The exact `em++` flags used to build the WASM module were undocumented
  (now fixed, see section 17).
- `docs/SANITIZERS.md`'s "What fuzzing explicitly does not cover yet"
  section already discloses several real undocumented-by-omission gaps
  (CLI header parser, Python header parser, JS header parser, and the
  interleaved rANS format's own length-prefixed fields -- none fuzzed).

## 21. Current claims that require verification (not re-checked this pass)

- `BENCHMARKS.md`'s numbers against the `REAL_*.md` sources.
- Rust/C#/Go binding correctness (tests exist, not run this pass).
- `demo/csa_demo.html`'s WebGPU path (requires a WebGPU-capable browser
  context this audit did not have available; not re-run this pass).
- Any claim in `PITCH_ONE_PAGER.md`/`USE_CASES.md`/`WHY_NOT_ZSTD.md` --
  not read this pass.

## 22. Opportunities for parallelization

- `WorkStealingPool` exists, tested, unused in production -- the
  highest-leverage "already built, not yet applied" opportunity.
- `pick_block_lag_2d`'s SIMD target (identified, not attempted).
- The numerical-computing-lab kernels the brief asks for (stencil,
  diffusion, gradient/divergence/laplacian) don't exist at all yet --
  each is a genuinely new, embarrassingly-parallel-shaped workload that
  would give `WorkStealingPool`/SIMD/CUDA real new targets beyond the
  codec's own narrow kernel set.

## 23. Opportunities for algorithmic improvement

- The two abandoned "optimal LZ parsing" attempts (documented honestly
  in `DESIGN.md`'s "Honest limitations" section) remain open --  a real
  optimal parser needs a cost model with information genuinely
  independent of the parse it's improving, which neither past attempt had.
- No format-level CRC (see section 6) is the clearest "small,
  well-scoped, real improvement" candidate in the whole codebase.

## 24. Production risks

- `packet_transport.cpp`'s `deserialize_packet` is not in the fuzzing
  harness (a real untrusted-input parser, same class of surface the
  fuzzer already found 8 bugs in elsewhere) -- the clearest concrete
  security-relevant risk this audit surfaces.
- No CI job runs sanitizers or the fuzzer on a schedule -- a regression
  in either area would only be caught by another manual session like
  this one, not automatically.
- The `.csa` format's missing CRC (section 6) means silent bit-flip
  corruption in a `.csa` file (not a packet-transport-wrapped stream)
  has no detection mechanism at all before hitting the same class of
  decode-time bugs this session's fuzzing pass fixed -- those fixes
  make corruption *fail safely* (a clean exception), but do not
  *detect* it as corruption specifically before that point.

---

## Prioritized gap list (drives what gets built next)

Ranked by (brief-stated value) x (how unbuilt it currently is), not
strictly by brief section order, per section 295's "implement the
highest-value missing functionality" instruction:

1. **Numerical Computing Lab** (brief sections 19-24, Phase 6) -- a
   complete, real gap. Nothing in this codebase currently maps
   "mathematics -> algorithm -> parallel kernel" for a non-codec
   workload, which is the brief's own stated central thesis (section
   294). Highest narrative and technical value of anything on this list.
2. **Hardware fingerprint / `scissorc system`** (brief sections 29, 200) --
   small, concrete, and a real prerequisite for making every existing
   benchmark doc's numbers properly comparable/reproducible (brief
   sections 121-122's "experiment manifest" concept depends on this
   existing first).
3. **Fuzz `packet_transport.cpp`'s `deserialize_packet`** -- closes the
   clearest concrete security gap this audit found (section 24), and is
   a small, bounded extension of infrastructure that already exists
   (`fuzz/fuzz_decompress.cpp`'s pattern).
4. **Wire the already-built `KdTree3i` WASM export into `docs/index.html`'s
   UI** -- the nearest-to-done item on the whole list (library done,
   WASM export done and independently verified; only the UI/message-
   protocol layer remains).
5. **`.csa` format CRC** -- real, well-scoped, but a genuine format-
   version bump (a bigger compatibility decision than the others on
   this list, so ranked lower despite being conceptually simple).

Everything else the 297-section brief asks for (the full Parallel
Computing Lab UI, Pareto frontier visualization, dataset adapters for
public data sources, the experiment/reproducibility database, the CLI's
`analyze`/`compare`/`experiment`/`report` subcommands, etc.) is real,
legitimate scope, not dismissed -- but this audit's job is to name the
next one or two highest-value slices, not to schedule all 297 sections
in one document. This list will be revisited after the next slice
lands.
