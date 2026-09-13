# Collapsible Scissored Surfaces Algorithm (CSA)

A lossless compression codec built directly from a mechanical-metamaterials
idea: Harvard SEAS's "pantograph lattices" -- networks of scissor-jointed
rods that collapse into a compact line and deploy into helical, toroidal,
and doubly-curved surfaces from a *small set of design parameters*
(`seas.harvard.edu/news/folds-and-cuts-linkages`).

That's a description of a reversible predictive transform. This repo
implements it as one: a compact representation plus a handful of
calibrated parameters that regenerate the full data exactly, built by a
local rule applied repeatedly across scale -- plus a real LZ77-style
dictionary matcher for the kind of redundancy (repeated substrings, not
predictable samples) that a predictive transform can't touch. See
`DESIGN.md` for the full technical mapping, `BENCHMARKS.md` for real,
regenerable compression-ratio measurements, `USE_CASES.md` for how it does
on realistic (not maximally repetitive) text/log/telemetry files, and
`GPU_BENCHMARKS.md` for a dedicated CPU-vs-GPU crossover measurement from
100K to 256M elements (nothing in any of those files is hand-typed).

## What's actually here

- **Pantograph Lift** -- a general-purpose reversible integer lifting
  transform (C++17), the byte-stream analogue of the mechanism.
- **Rod-Joint Transform** -- the literal geometric analogue for point
  sequences (GPS tracks, spirals, helices, LiDAR scans): each edge vector
  is predicted from an earlier one (not always the immediately preceding
  one -- a small set of candidate lags is searched and whichever encodes
  smallest wins, letting quasi-periodic paths align with their own
  oscillation period) via a calibrated rotation+scale constant. 3D point
  sequences try two models per file automatically (a 2D rotation joint on
  (x,y) + affine fit on z, and a true 3D similarity joint fit via Horn's
  closed-form quaternion method) and keep whichever encodes smaller.
- **LZ dictionary matcher** -- a real, working LZ77-style compressor
  (unbounded-window hash-chain matching with lazy/one-step-lookahead
  parsing, the same technique zlib's higher levels use) for the
  repeated-substring redundancy in text/log/structured files that no
  predictive transform can exploit. `compress()` tries this alongside
  Pantograph Lift and raw storage and keeps whichever encodes smallest.
- **Adaptive order-1 range coder** -- the entropy-coding backend shared by
  every mode (generalized to arbitrary alphabet sizes for the LZ matcher's
  literal/length/distance streams, not just the original 256-byte case).
- **A real lossy mode** for the 2D Rod-Joint Transform -- quantized
  prediction residuals with periodic exact resync to bound absolute-
  position drift, not just lossless. `quant_step <= 1` is provably
  identical to lossless; larger steps trade a bounded, measured
  coordinate error for real compression gains (see `DESIGN.md`).
- **A stable C ABI** (`include/csa/csa_capi.h`, built as `csa.dll` /
  `libcsa.so` / `libcsa.dylib`) with no C++ types crossing the boundary,
  plus a **Python `ctypes` binding** (`bindings/python/csa.py`) on top of
  it -- both tested against the actual built shared library, not mocked.
- **CUDA kernel** for the Pantograph Lift's forward transform (genuinely
  parallel: every pair within a decomposition level is independent),
  GPU-resident across the whole multi-level pass (one upload, a handful of
  downloads, no per-level round trips), tested against the CPU path
  bit-for-bit.
- **1306 round-trip correctness checks** (`tests/test_main.cpp` +
  `tests/test_capi.cpp`, the latter linking the real shared library to
  catch actual symbol-export problems), including a dedicated check that
  the GPU path actually succeeds (not just that the overall call
  round-trips via CPU fallback), all passing.
- **A benchmark suite** (`bench/`) comparing against gzip/bz2/lzma on both
  synthetic shape classes and domain-realistic data (a simulated single-ring
  LiDAR scan, synthetic server logs, JSON telemetry, sensor CSV exports),
  with honest wins *and* honest losses reported.

## Quick start

```powershell
# Build (Windows, MSVC + CUDA 13.x; CPU-only build works without CUDA too)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Round-trip a file
build/scissorc.exe compress   input.bin  out.csa
build/scissorc.exe decompress out.csa    roundtrip.bin

# Geometric mode: text file of "x y" (or "x y z") floats, one point per line
build/scissorc.exe compress-geo2d   track.xy   track.csa
build/scissorc.exe decompress-geo2d track.csa  track_out.xy

# Run the test suite
build/csa_tests.exe

# Regenerate the benchmark datasets and BENCHMARKS.md
python bench/gen_datasets.py
python bench/run_benchmarks.py

# Measure the CPU-vs-GPU transform crossover on your own machine
python bench/gpu_crossover.py

# Realistic (not maximally repetitive) use-case datasets and results
python bench/use_cases.py

# Lossy geometric mode: quantized residuals, bounded error, real ratio gain
build/scissorc.exe compress-geo2d-lossy track.xy track_lossy.csa --quant 20 --resync 64

# Python bindings (ctypes, on top of the C ABI in include/csa/csa_capi.h)
python bindings/python/test_bindings.py
python -c "import sys; sys.path.insert(0, 'bindings/python'); import csa; print(csa.compress(b'hello world'))"
```

## Headline results

**Geometric shape classes** (see `BENCHMARKS.md` for the full table):

| dataset | shape class | vs. best of gzip/bz2/lzma |
|---|---|---|
| `spiral.xy` | logarithmic spiral | **92% smaller** than raw; beats lzma by ~8x |
| `helix.xyz` | helical point cloud | **95% smaller**; beats lzma by ~12x (true 3D similarity joint) |
| `gps_track.xy` | noisy real-world-style path | beats lzma by ~1.3x |
| `toroidal.xyz` | doubly-curved (wobbling radius) | **32% smaller; beats lzma** -- see below for how |

**Realistic use cases** (see `USE_CASES.md` for the full picture, including bz2/lzma comparisons):

| dataset | scenario | vs. gzip -9 |
|---|---|---|
| server access log | 5,000 synthetic nginx-format lines | **beats gzip** (55.5KB vs 56.9KB) |
| JSON telemetry events | 5,000 IoT/analytics events | **beats gzip** (70.6KB vs 73.2KB) |
| sensor CSV export | 20,000 rows, smooth+noisy columns | close to gzip (166.9KB vs 164.4KB) |

## Honesty, not hype

- The Rod-Joint Transform wins big specifically on the shape classes the
  source paper calls out (helical, spiral, toroidal-adjacent paths) --
  because those are exactly what a rotation+scale joint models. It is not
  a general-purpose point-cloud compressor and doesn't try to be.
- `toroidal.xyz` used to be reported as an honest loss against lzma: its
  radius oscillates roughly every 3 rods, far faster than any calibration
  block predicting from the immediately preceding rod could track. The
  actual fix wasn't a fancier per-block fit -- both Geo3D models now
  search a small set of candidate lags (predict rod `i` from rod `i-lag`
  for lag in {1..32}) and keep whichever encodes smallest, the same idea
  speech/audio codecs use for periodic signals. Lag 3 aligns almost
  exactly with this shape's oscillation period, turning a 15%-smaller
  loss into a 32%-smaller win. See `DESIGN.md` for the mechanism.
- General mode tries three candidates per file now -- raw storage,
  Pantograph Lift, and a real LZ77-style dictionary matcher -- and keeps
  whichever encodes smallest. On `text_repetitive.bin` (a single sentence
  repeated thousands of times) the LZ matcher's *unbounded* window beats
  gzip, bz2, *and* lzma outright; that's a real structural edge on very
  long-range repetition specifically, not a claim that CSA beats mature LZ
  compressors on arbitrary text in general. `USE_CASES.md` tests
  realistic, non-trivially-repetitive text/log/JSON/CSV instead: CSA beats
  gzip on most of them and is within reach of (without matching) bz2/lzma,
  which use more sophisticated parsing and entropy modeling (see
  `DESIGN.md`'s future work for exactly what's missing).
- GPU timing is reported honestly, at two different fidelities. Launching
  a fresh process per measurement (which pays its own allocation, and a
  real ~1.2s wake/context-creation cost if this laptop GPU had idled since
  the last call), GPU-vs-CPU time converges from ~128x slower at 100K
  elements to ~0.6-0.8x at 256M elements -- the CPU path still wins every
  size measured that way. But that understates real deployment: a service
  handles many requests from one long-lived process, not one process per
  input, and measuring *that* directly (`scissorc bench-transform --repeat
  N`, steady-state average once the GPU's clocks have ramped up) shows GPU
  reaching **~0.98-1.00x of CPU time at 16M-256M elements, and an outright
  win at 64M** in the runs recorded in `GPU_BENCHMARKS.md`. Genuine parity
  for a mid-range laptop GPU against a modern CPU, reported as measured --
  not the fixed one-shot number, and not oversold as a definitive win
  everywhere either. This mirrors the same PCIe/warm-up-bound pattern this
  repo owner's other GPU-resident projects (a GPU-resident CKKS
  homomorphic-encryption library, and a production LiDAR CUDA pipeline)
  already measured and documented.

## Repo layout

```
include/csa/      public headers (transforms, range coder, codec, CUDA API, C ABI)
src/              CPU implementation (+ csa_capi.cpp, the C ABI shim)
cuda/             CUDA kernel (built only if a CUDA compiler is found)
cli/              scissorc command-line tool
bindings/python/  ctypes bindings (csa.py) + test_bindings.py, on the C ABI
tests/            round-trip test suite (test_main.cpp) + C-ABI test that
                  links the real shared library (test_capi.cpp), no external deps
bench/            dataset generator + benchmark runner (writes BENCHMARKS.md)
                  + GPU crossover benchmark (writes GPU_BENCHMARKS.md)
                  + realistic use-case scenarios (writes USE_CASES.md)
DESIGN.md          full technical design and honest limitations/future work
BENCHMARKS.md      regenerated by bench/run_benchmarks.py -- not hand-edited
GPU_BENCHMARKS.md  regenerated by bench/gpu_crossover.py -- not hand-edited
USE_CASES.md       regenerated by bench/use_cases.py -- not hand-edited
```
