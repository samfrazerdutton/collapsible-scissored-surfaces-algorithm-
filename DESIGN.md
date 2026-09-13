# Design: Collapsible Scissored Surfaces Algorithm (CSA)

## Source idea

Bertoldi/Reis-style deployable metamaterial work at Harvard SEAS ("Folds and
cuts and linkages", `seas.harvard.edu/news/folds-and-cuts-linkages`)
describes "pantograph lattices": networks of scissor-jointed rods that
collapse into a compact line and deploy into helical, toroidal, and
doubly-curved surfaces. The paper's central claim, in its own words:

> "the geometry of an entire deployable surface can be encoded in a small
> set of design parameters" via a "local additive construction process" --
> new linkages added one at a time while preserving compatibility,
> deployability, and collapsibility.

That is a description of a **reversible predictive transform**: a compact
representation (the collapsed line) plus a small parameter set that
regenerates the full structure (the deployed surface) exactly, built by a
local rule applied repeatedly. This repo takes that idea at face value and
implements it as an actual lossless compression codec, not a metaphor.

## Two transforms, one entropy backend

### 1. Pantograph Lift (`include/csa/pantograph_lift.hpp`)

General-purpose, operates on an arbitrary byte/integer stream. It is a
reversible integer lifting scheme (the same family as JPEG2000's 5/3
wavelet), but the predict step is a calibrated **affine fit** instead of a
fixed coefficient:

```
predicted = round(ratio * a) + offset
d = b - predicted            # exact residual, any integer
upd = d >> 1                 # arithmetic shift = floor(d/2)
s = a + upd                  # goes to the next (coarser) level
```

- `ratio` is the literal pantograph pivot ratio: a single scaling
  constant. `offset` is a small additive correction so ordinary
  affine/ramp data (`b ~= a + k`), which a pure scaling joint cannot
  represent, still predicts well.
- Both are recalibrated every `kPantographBlockSize` (1024) pairs rather
  than once per level: one global ratio can only track data that is
  affine across the *entire* array, which fails for anything that curves
  (e.g. a sine wave). Recalibrating periodically keeps the parameter
  count tiny relative to the data (still matching the paper's "small set
  of design parameters" property) while tracking local curvature.
- **Correctness is independent of calibration quality.** `predicted` is
  computed the same way on encode and decode from data already
  transmitted; the residual is always `actual - predicted`, so a bad
  fit only costs compression ratio, never correctness. This mirrors the
  source structure's actual engineering property: compatibility is
  guaranteed by construction, not tuned.
- Recursion: pair up the array, halve, repeat until one value remains
  (padding to the next power of two, replication-padded). log2(N) levels,
  each an independent calibration + a fully parallel transform pass.

### 2. Rod-Joint Transform (`include/csa/rod_joint_transform.hpp`)

The literal instantiation: a point sequence (polyline) is a chain of rigid
rods (edge vectors) connected by joints. A pantograph lattice deploys an
entire helical/toroidal surface from one fixed pivot ratio applied joint
after joint; here, one calibrated **complex constant** `c = ratio_re +
i*ratio_im` (a 2D rotation+scale, i.e. a similarity transform) predicts
each rod from the previous one:

```
rod[i] ~= c * rod[i-1]
residual[i] = rod[i] - round(c * rod[i-1])     # exact complex-integer residual
```

For a perfect logarithmic spiral, `c` alone (2 numbers) plus the anchor
point predicts the *entire* shape and every residual is zero -- the
compression-domain analogue of the paper's helical/toroidal deployment
claim. `c` is recalibrated every `kRodJointBlockSize` (128) rods, same
rationale as above.

3D point sequences have two candidate models, and `compress_geo3d` tries
both and keeps whichever encodes smaller (tagged with one sub-mode byte,
still exactly reversible either way):

- **Composition**: the (x, y) plane goes through the 2D Rod-Joint
  transform (this captures genuine turning motion jointly -- a rotating
  trajectory's x and y are individually just two out-of-phase sine waves,
  each hard to predict alone, but together are exactly what a rotation
  constant models), and z goes through the Pantograph Lift by itself
  (captures smooth/linear vertical motion, e.g. a helix's constant climb
  rate).
- **True 3D similarity joint**: a single calibrated 3x3 matrix (rotation +
  uniform scale) predicts each 3D rod from the previous one,
  `rod[i] ~= M * rod[i-1]`. `M` is fit per block via Horn's closed-form
  absolute-orientation method (see below) rather than the axis-split
  composition, so it can track paths that genuinely tumble (no fixed "up"
  axis) -- and, as a bonus, it turns out to *subsume* the composition's
  best case too: its bottom row collapses to `(0,0,1)` whenever a rod's
  z-component is constant (a helix's climb), reproducing that case and
  then improving on it (see `BENCHMARKS.md`: the true 3D joint wins
  outright on `helix.xyz`, pushing it to ~12x smaller than lzma).

**Calibrating the 3D similarity joint without a general eigensolver.**
Horn's method (1987) finds the optimal rotation between two sets of
corresponding vectors as the dominant eigenvector of a symmetric 4x4
"profile" matrix `N` built from the 3x3 cross-covariance of the vector
pairs. Rather than implement a general eigensolver, `calibrate_3d_block`
(in `src/rod_joint_transform.cpp`) uses **shifted power iteration**: `N`
has trace 0 (so its eigenvalues aren't all one sign), so it's shifted by
its Frobenius norm before iterating (`N' = N + ||N||_F * I`), which
guarantees the eigenvalue Horn's method wants -- the most positive one --
becomes the largest in magnitude too, so ~40 iterations of `v <- N'v /
|N'v|` converge to the right eigenvector. The optimal uniform scale then
follows in closed form (Umeyama's formula) given that rotation. As with
every other calibration in this codebase, a bad or non-converged fit only
costs compression ratio -- the residual is still `actual - predicted`,
computed identically on encode and decode, so correctness never depends
on it.

**Predicting from `rod[i-lag]`, not always `rod[i-1]`.** A toroidal
cross-section's xy-radius oscillates roughly every 3 rods -- far faster
than any calibration block (even recalibrated every 128 rods) predicting
from the *immediately preceding* rod could ever track, no matter how good
the rotation+scale fit is. The fix isn't a richer per-block model; it's
recognizing that "predict from `rod[i-1]`" was an unnecessary assumption.
Both `rod_joint_2d_forward` and `rod_joint_3d_similarity_forward` take a
`lag` parameter: rod `i` is predicted from rod `i-lag` (the calibration
math is unchanged, just re-indexed). `compress_geo2d`/`compress_geo3d`
(in `codec.cpp`) search `kRodJointCandidateLags` = {1, 2, 3, 4, 5, 6, 7, 8,
10, 12, 16, 20, 24, 32} and keep whichever lag actually encodes smallest —
each candidate is a full, cheap encode (a few thousand points takes
milliseconds), so comparing true output size beats guessing from a
residual-magnitude proxy. This is the same idea as long-term/pitch
prediction in speech and audio codecs, applied to rod sequences instead
of waveform samples. For `toroidal.xyz`, lag 3 aligns almost exactly with
the oscillation period and turns a 15%-smaller loss against lzma into a
32%-smaller win (see `BENCHMARKS.md`); every other shape ticked up
slightly too, since lag=1 usually isn't *exactly* optimal even when it's
close.

### 3. Entropy backend: adaptive order-1 range coder (`include/csa/range_coder.hpp`)

Both transforms produce residual/parameter streams that are then
zigzag+varint encoded and range-coded with a Fenwick-tree-backed adaptive
order-1 model (256 contexts, one per preceding byte). This is a standard,
well-understood carryless range coder (Subbotin-style normalization) --
deliberately *not* a hand-rolled rANS. rANS is the more fashionable choice
(it backs zstd/JPEG XL/AV1) and its interleaved-stream variants are what
would make entropy coding itself GPU-parallel, but it is also considerably
easier to get subtly wrong. Given the choice between a flashier entropy
coder and one that is provably bit-exact under test, correctness won:
the range coder here is simple enough to reason about completely, and all
1293 round-trip checks (see `tests/test_main.cpp`) pass, including the
CUDA path. Interleaved-stream rANS for GPU-parallel entropy decode is a
natural next step (see Future Work).

### 4. LZ dictionary matcher (`include/csa/lz_matcher.hpp`, `include/csa/lz_codec.hpp`)

The two transforms above are *predictive*: good when a sample or rod
relates to an earlier one by a small calibrated rule. Ordinary text and
structured files have a different kind of redundancy instead -- the same
substring recurring far apart (a word, a log-line template, a repeated
JSON key) -- which no predictive transform can exploit but a dictionary
matcher is built for. This is what lets `Mode::GeneralLZ` compete with
gzip/bz2/lzma on that kind of data, rather than leaving it entirely to
Pantograph Lift (which was never going to beat them there, and honestly
didn't -- see `BENCHMARKS.md`'s git history for the numbers before this
was added).

**Matching** (`lz_matcher.cpp`): an unbounded-window hash-chain matcher --
a position's 4-byte hash indexes a chain of every earlier position with
the same hash (classic zlib-style hash chains), walked up to a bounded
depth (1024) to find the longest match. "Unbounded window" means a match
can reference any earlier position in the whole buffer, not a fixed
32KB-ish window like gzip -- a real, structural advantage specifically on
data with very long-range repetition (see `text_repetitive.bin` in
`BENCHMARKS.md`, which collapses smaller than gzip, bz2, *and* lzma).
Matching is **lazy** (one-step lookahead, the same technique zlib's higher
compression levels use): before committing to a match found at position
`i`, the matcher checks whether position `i+1` has a strictly longer one;
if so, `i` is emitted as a literal and the better match at `i+1` wins
instead. This alone improved the realistic-text results in `USE_CASES.md`
enough to flip several from losing against gzip to beating it. Every
position an accepted match covers is still inserted into the hash chains
(not just skipped over), so a later match can reference into the middle
of an earlier one -- important for highly repetitive data.

**Entropy coding** (`lz_codec.cpp`): literal bytes and the "a match starts
here" flag share one order-1 adaptive model over a 257-symbol alphabet
(reusing the same `RangeEncoder`/`RangeDecoder` as everything else, just
with a newly-generalized `FenwickFreqN`/`Order1ModelN` that support an
arbitrary alphabet size instead of the original hard-coded 256). Match
length and distance are each split into a magnitude class ("bucket",
`bucket = floor(log2(v+1))`, entropy-coded via its own small adaptive
model) and the bits distinguishing values within that bucket (packed raw
via `bitpacker.hpp`'s `BitWriter`/`BitReader`, since those bits are ~
uniform by construction and entropy coding them would gain nothing) --
the same length/distance coding strategy DEFLATE and LZMA-style codecs
use, adapted to this project's range coder instead of Huffman coding.

**Honest limits**: this is a real, working LZ77-style compressor, not a
toy -- but it is not LZMA. There is no optimal parsing (the lazy
lookahead is one step, not a full cost-based search over parse choices),
no context-mixed high-order entropy modeling, and no BWT (which is a
large part of why bz2 wins on ordinary prose). See `USE_CASES.md` for
where this actually lands against gzip/bz2/lzma on realistic (not
maximally repetitive) text, log, and structured-data files -- it beats
gzip on most of them, and is within reach of bz2/lzma without matching
them outright.

## Container format

One shared bitstream, `include/csa/codec.hpp`:

- `Mode::Raw` -- passthrough with an 13-byte header, chosen automatically
  whenever a transformed representation would be *larger* (e.g.
  incompressible random data), so CSA never inflates input by more than a
  small fixed overhead.
- `Mode::General` -- Pantograph Lift over a byte stream.
- `Mode::GeneralLZ` -- the LZ dictionary matcher over a byte stream.
- `Mode::Geo2D` / `Mode::Geo3D` -- Rod-Joint Transform over point streams.

`compress()` tries `Raw`/`General`/`GeneralLZ` for any byte-stream input
and keeps whichever encodes smallest, so callers never need to know in
advance whether their data is more "smooth/predictive" or more
"repeated-substring" in nature.

## GPU acceleration (`cuda/pantograph_lift_cuda.cu`)

The Pantograph Lift's per-level transform is embarrassingly parallel: given
the previous level's array, every pair's predict/residual/update is
independent of every other pair. The CUDA path is **GPU-resident for the
entire multi-level forward pass**: everything from the first upload to the
last download runs in one CUDA stream with no host synchronization in
between, and there is exactly one H2D transfer (the padded input) and a
handful of D2H transfers (the flat residual/ratio/offset buffers, each
copied once for the whole pass, plus the final scalar base value) —
regardless of how many decomposition levels the input has.

1. `calibrate_blocks_kernel`: one CUDA thread block per calibration block,
   shared-memory tree reduction, no atomics (each thread block owns a
   disjoint data range).
2. `compute_block_params_kernel`: turns each block's four sums into its
   ratio+offset **on the device** — one thread per block, a dozen FLOPs
   each. (The first version of this code did this arithmetic on the host,
   which meant a small blocking round trip every level purely to do
   ~12 FLOPs of division and rounding; see the comment at the top of
   `pantograph_lift_cuda.cu` for why that was the wrong tradeoff.)
3. `transform_kernel`: applies predict+residual+update to every pair in
   parallel, reading the block parameters `compute_block_params_kernel`
   already wrote, and writes directly into a pre-sized slice of one
   whole-pass residual buffer (no per-level allocation).
4. Two fixed-size device buffers ping-pong as the "current level" array
   across levels; nothing is malloc'd or freed inside the level loop.
   Pinned (page-locked) host staging buffers make the handful of final
   bulk transfers faster than the pageable `std::vector` storage an
   earlier version copied into directly.
5. **Tail hand-off to CPU**: past `kGpuTailCutoff` (65536 elements), a
   level's total work is too small for three more kernel launches' fixed
   dispatch overhead to pay for itself. Once a level's array shrinks to
   that size, the GPU loop stops, the array is copied back once, and the
   remaining (small) levels are computed by the ordinary CPU
   `pantograph_lift_forward` — the *same* transform, just finishing on
   the device better suited to a workload that small. `pantograph_lift_
   inverse` doesn't need to know or care which device produced which
   level; it just needs `residuals`/`block_ratios`/`block_offsets` present
   and in order, which this hand-off preserves by simple concatenation.

This mirrors the GPU-resident philosophy from this repo owner's other CUDA
projects (a GPU-resident CKKS homomorphic-encryption library, and a
GPU-resident LiDAR preprocessing pipeline): keep the working data
on-device across the whole pipeline stage, minimize host round trips, and
measure honestly rather than assume the GPU path wins.

**Measured, not assumed**: see `GPU_BENCHMARKS.md` for the dedicated
transform-only CPU-vs-GPU crossover measurement on this machine (an RTX
2060 Max-Q laptop GPU), isolating the forward transform from the
CPU-sequential entropy coding stage that follows it and runs identically
either way. Headline findings from that file:

- This laptop GPU idles down to a low-power state between uses, and the
  first CUDA call after idling pays a real, fixed wake/context-creation
  cost (~1.2s on this machine) independent of input size — a genuine
  reason a single ad-hoc `--gpu` call on one small file can look far
  slower than the CPU path.
- Comparing a fresh `scissorc` process per measurement (paying its own
  allocation, and the wake cost if the GPU had idled), GPU vs. CPU time
  converges steadily as input size grows, from ~0.01x (128x slower) at
  100K elements to ~0.6-0.8x at 256M elements — the CPU path was still
  faster at every size tested this way.
- That one-shot-per-process comparison understates real deployment,
  though: a service handling many requests runs from one long-lived
  process, not one process per input. Measuring that directly (`scissorc
  bench-transform <n> --repeat N`, same warm CUDA context and GPU clock
  state across calls, reporting steady-state average of calls 2+) tells a
  materially different story — GPU reaches **~0.98-1.00x of CPU time at
  16M-256M elements, and outright wins at 64M** in the runs recorded in
  `GPU_BENCHMARKS.md`. This is genuine parity for a mid-range laptop GPU
  against a modern CPU on a task CPUs are naturally efficient at (simple,
  cache-friendly, branch-predictable sequential array passes) — reported
  as measured, not oversold as a definitive win everywhere.
- A one-off probe past the practical VRAM ceiling (400M elements) hit a
  genuine CUDA resource error, and the `pantograph_lift_forward_cuda` →
  automatic CPU fallback path handled it transparently — the
  graceful-degradation design being exercised by a real failure, not just
  a theoretical code path.

## Honest limitations / future work

- **Per-block lag search** (rather than one global lag per file) would
  help paths whose oscillation period itself drifts over the sequence --
  currently `compress_geo2d`/`compress_geo3d` pick one lag for the whole
  file. This is the natural next step if a real-world dataset shows a
  period that changes partway through.
- **Interleaved-stream rANS** would let the entropy-coding stage itself
  run in parallel on GPU (unlike the current sequential adaptive range
  coder), closing the loop on an end-to-end GPU-resident codec.
- **A CPU-vs-GPU crossover past this GPU's ~256M-element practical VRAM
  ceiling** hasn't been measured (see `GPU_BENCHMARKS.md`) -- testing on a
  GPU with more VRAM, or reducing per-buffer memory (e.g. processing in
  chunks instead of one padded_len-sized allocation), would extend the
  measurement further.
- **Optimal (cost-based) LZ parsing** would likely close more of the
  remaining gap to bz2/lzma on realistic text (see `USE_CASES.md`): the
  current matcher's lookahead is one step (lazy matching), not a full
  search over parse choices weighted by their actual entropy-coded cost,
  which is what LZMA-class compressors do.
- **A BWT-based mode** would be the more direct way to challenge bz2
  specifically on ordinary prose, since a large part of bz2's advantage
  there comes from the Burrows-Wheeler Transform rearranging the data
  into long runs of similar bytes before entropy coding, not from its LZ
  stage.

## Build gotcha: adding a new `__global__` kernel

If you add a new `__global__` CUDA kernel and its launch fails at runtime
with `cudaErrorSymbolNotFound` ("named symbol not found") despite building
without errors, this is a known CMake+Ninja+CUDA incremental-build issue
with relocatable device code (`CUDA_SEPARABLE_COMPILATION ON`): the
per-executable device-link object (`cmake_device_link.obj`) isn't always
regenerated correctly when only the static library's `.cu` file changed.
`rm -rf build && ` reconfigure `+` rebuild from scratch fixes it. This bit
the first version of the GPU-resident rewrite below (adding
`compute_block_params_kernel` triggered exactly this), and the test suite
did not catch it initially because `pantograph_lift_forward_cuda` returning
`false` transparently falls back to the CPU path -- masking the failure as
a passing round-trip test. `tests/test_main.cpp`'s `test_codec()` now
calls `pantograph_lift_forward_cuda` directly and asserts it returns
`true` when a CUDA device is available, specifically to catch this class
of silent-fallback bug in the future.
