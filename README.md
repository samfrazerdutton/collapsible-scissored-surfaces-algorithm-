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
on realistic (not maximally repetitive) text/log/telemetry files,
`REAL_CORPUS_BENCHMARK.md` for the harshest test -- 18MB of real,
unmodified C++ source code, not text written for this project -- and
`GPU_BENCHMARKS.md` for a dedicated CPU-vs-GPU crossover measurement from
100K to 256M elements (nothing in any of those files is hand-typed).

## What's actually here

- **Pantograph Lift** -- a general-purpose reversible integer lifting
  transform (C++17), the byte-stream analogue of the mechanism.
- **Rod-Joint Transform** -- the literal geometric analogue for point
  sequences (GPS tracks, spirals, helices, LiDAR scans): each edge vector
  is predicted from an earlier one (not always the immediately preceding
  one -- a small set of candidate lags is searched **per calibration
  block**, not once for the whole file, so a path whose oscillation period
  itself drifts partway through is tracked correctly -- 71% less residual
  than the best single whole-file lag on a period-drifting test shape) via
  a calibrated rotation+scale constant. 3D point sequences try two models
  per file automatically (a 2D rotation joint on (x,y) + affine fit on z,
  and a true 3D similarity joint fit via Horn's closed-form quaternion
  method) and keep whichever encodes smaller.
- **LZ dictionary matcher** -- a real, working LZ77-style compressor
  (unbounded-window hash-chain matching with lazy/one-step-lookahead
  parsing, the same technique zlib's higher levels use) for the
  repeated-substring redundancy in text/log/structured files that no
  predictive transform can exploit. `compress()` tries this alongside
  Pantograph Lift and raw storage and keeps whichever encodes smallest.
  Exposes the same speed/ratio "level" every production LZ compressor
  does (`scissorc compress --level fast|balanced|high`, gzip -1..-9 /
  zstd -1..-22's tradeoff) -- `fast` beats gzip and bz2 while compressing
  faster than lzma on real source code; `high` closes to within 13% of
  lzma's ratio at real cost in time. See `REAL_CORPUS_BENCHMARK.md`.
- **Adaptive order-1 range coder** -- the entropy-coding backend shared by
  every mode (generalized to arbitrary alphabet sizes for the LZ matcher's
  literal/length/distance streams, not just the original 256-byte case).
- **A real lossy mode** for both the 2D and 3D (true similarity joint)
  Rod-Joint Transforms -- quantized prediction residuals with periodic
  exact resync to bound absolute-position drift, not just lossless.
  `quant_step <= 1` is provably identical to lossless; larger steps trade
  a bounded, measured coordinate error for real compression gains (see
  `DESIGN.md`).
- **A stable C ABI** (`include/csa/csa_capi.h`, built as `csa.dll` /
  `libcsa.so` / `libcsa.dylib`) with no C++ types crossing the boundary,
  plus **Python `ctypes`** (`bindings/python/csa.py`) and **Rust FFI**
  (`bindings/rust/`) bindings on top of it -- both tested end-to-end
  against the actual built shared library, not mocked.
- **CUDA kernel** for the Pantograph Lift's forward transform (genuinely
  parallel: every pair within a decomposition level is independent),
  GPU-resident across the whole multi-level pass (one upload, a handful of
  downloads, no per-level round trips), tested against the CPU path
  bit-for-bit.
- **Persistent GPU sessions** (`CudaLiftSession`) that reuse device/pinned
  buffers across calls instead of allocating and freeing them every time --
  `compress()` keeps one per thread automatically. Measured, not assumed:
  ~3.5x faster sustained per-call time at 50K elements, ~1.7x at 2M (see
  `DESIGN.md`).
- **1349 round-trip correctness checks** (`tests/test_main.cpp` +
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

# Round-trip a file (--level fast|balanced|high trades speed for ratio)
build/scissorc.exe compress   input.bin  out.csa --level balanced
build/scissorc.exe decompress out.csa    roundtrip.bin

# Build a real (non-synthetic) benchmark corpus from any large local C/C++
# project, and compare CSA against gzip/bz2/lzma on it at every level
python bench/build_real_corpus.py /path/to/some/large/cpp/project
python bench/real_corpus_benchmark.py

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

# Compare one-shot vs. persistent-session GPU buffer reuse (see DESIGN.md)
build/scissorc.exe bench-transform 2000000 --gpu --repeat 20
build/scissorc.exe bench-transform 2000000 --gpu --session --repeat 20

# Realistic (not maximally repetitive) use-case datasets and results
python bench/use_cases.py

# Lossy geometric mode: quantized residuals, bounded error, real ratio gain
build/scissorc.exe compress-geo2d-lossy track.xy track_lossy.csa --quant 20 --resync 64
build/scissorc.exe compress-geo3d-lossy track.xyz track_lossy.csa --quant 20 --resync 64

# Python bindings (ctypes, on top of the C ABI in include/csa/csa_capi.h)
python bindings/python/test_bindings.py
python -c "import sys; sys.path.insert(0, 'bindings/python'); import csa; print(csa.compress(b'hello world'))"

# Rust bindings (FFI, on top of the same C ABI)
cd bindings/rust && cargo test
```

## Headline results

**Geometric shape classes** (see `BENCHMARKS.md` for the full table):

| dataset | shape class | vs. best of gzip/bz2/lzma |
|---|---|---|
| `spiral.xy` | logarithmic spiral | **92% smaller** than raw; beats lzma by ~8x |
| `helix.xyz` | helical point cloud | **95% smaller**; beats lzma by ~12x (true 3D similarity joint) |
| `gps_track.xy` | noisy real-world-style path | beats lzma by ~1.3x |
| `toroidal.xyz` | doubly-curved (wobbling radius) | **32% smaller; beats lzma** -- see below for how |

**Realistic use cases** (`--level balanced`, the default; see `USE_CASES.md` for the full picture, including bz2/lzma comparisons):

| dataset | scenario | vs. gzip -9 |
|---|---|---|
| server access log | 5,000 synthetic nginx-format lines | close to gzip (57.9KB vs 56.9KB) |
| JSON telemetry events | 5,000 IoT/analytics events | **beats gzip** (72.4KB vs 73.2KB) |
| sensor CSV export | 20,000 rows, smooth+noisy columns | close to gzip (168.3KB vs 164.4KB) |

**A real, non-synthetic corpus** (18MB of unmodified C++ source code, vs. real market compressors -- see `REAL_CORPUS_BENCHMARK.md`):

| level | size | notable comparisons | compress time |
|---|---:|---|---:|
| `--level fast` | 2.04MB | **beats gzip (3.10MB), bz2 (2.40MB), zstd -3 (2.93MB)** | 1.4s |
| `--level balanced` | 1.95MB | **beats brotli -11 (2.09MB)**, still ahead of gzip/bz2/zstd -3 | 5.2s |
| `--level high` | 1.91MB | within 13% of lzma -9 (1.68MB) -- but **zstd -19 beats it on size *and* speed** (1.71MB in 5.2s vs 30.2s) | 30.2s |

## Honesty, not hype

- The Rod-Joint Transform wins big specifically on the shape classes the
  source paper calls out (helical, spiral, toroidal-adjacent paths) --
  because those are exactly what a rotation+scale joint models. It is not
  a general-purpose point-cloud compressor and doesn't try to be.
- `toroidal.xyz` used to be reported as an honest loss against lzma: its
  radius oscillates roughly every 3 rods, far faster than any calibration
  block predicting from the immediately preceding rod could track. Both
  Geo3D models now search a small set of candidate lags (predict rod `i`
  from rod `i-lag` for lag in {1..32}) **independently per calibration
  block**, not once for the whole file -- the same idea speech/audio codecs
  use for periodic signals, generalized so a path whose period itself
  drifts partway through still gets tracked (71% less residual than the
  best single whole-file lag on a test shape built to prove it). Lag 3
  aligns almost exactly with `toroidal.xyz`'s oscillation period, turning a
  15%-smaller loss into a 32%-smaller win. See `DESIGN.md` for the
  mechanism.
- General mode tries three candidates per file now -- raw storage,
  Pantograph Lift, and a real LZ77-style dictionary matcher -- and keeps
  whichever encodes smallest. On `text_repetitive.bin` (a single sentence
  repeated thousands of times) the LZ matcher's *unbounded* window beats
  gzip, bz2, *and* lzma outright; that's a real structural edge on very
  long-range repetition specifically, not a claim that CSA beats mature LZ
  compressors on arbitrary text in general.
- The harshest, most credible test in this repo is `REAL_CORPUS_BENCHMARK.md`:
  18MB of real, unmodified C++ source code, not text written or generated
  for this project, benchmarked against gzip/bz2/lzma *and* zstd and
  brotli -- the compressors "the market" actually means today, not just
  the textbook trio. `--level fast` beats gzip, bz2, *and* zstd's default
  level; `--level balanced` beats brotli's max level too. But the more
  important, more humbling number: **zstd -19 beats CSA `--level high` on
  both size *and* speed at once** (1.71MB in 5.2s vs 1.91MB in 30.2s) --
  not just a better ratio, a better ratio *and* 6x faster. That's the
  honest measure of the gap to a real modern production compressor: the
  whole speed/ratio curve, not one axis. Real source code has exactly the
  kind of structure LZMA/zstd's optimal-ish parsing and bz2's
  Burrows-Wheeler Transform are built to exploit, and this codec's
  lazy-matching LZ + order-1 entropy model currently isn't (see
  `DESIGN.md`'s future work for exactly what closing that gap would
  require: optimal parsing, a BWT mode, and the kind of performance
  engineering zstd has had years of). The first version of this matcher
  took ~30s to compress that
  same 18MB at only 0.6 MB/s regardless of level -- a real performance bug
  (an unbounded lazy-lookahead search on highly repetitive text), found by
  profiling and fixed, not glossed over.
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
                  + real (non-synthetic) corpus benchmark
                    (build_real_corpus.py + real_corpus_benchmark.py,
                    writes REAL_CORPUS_BENCHMARK.md)
DESIGN.md               full technical design and honest limitations/future work
BENCHMARKS.md           regenerated by bench/run_benchmarks.py -- not hand-edited
GPU_BENCHMARKS.md       regenerated by bench/gpu_crossover.py -- not hand-edited
USE_CASES.md            regenerated by bench/use_cases.py -- not hand-edited
REAL_CORPUS_BENCHMARK.md  regenerated by bench/real_corpus_benchmark.py -- not hand-edited
```
