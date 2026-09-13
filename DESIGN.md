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

See `BENCHMARKS.md` for what these models win and where they still fall
short (toroidal paths, whose xy-radius itself oscillates roughly every 3
samples -- far faster than any per-block calibration, even the true 3D
joint's, can track).

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
644 round-trip checks (see `tests/test_main.cpp`) pass, including the CUDA
path. Interleaved-stream rANS for GPU-parallel entropy decode is a natural
next step (see Future Work).

## Container format

One shared bitstream, `include/csa/codec.hpp`:

- `Mode::Raw` -- passthrough with an 13-byte header, chosen automatically
  whenever a transformed representation would be *larger* (e.g.
  incompressible random data), so CSA never inflates input by more than a
  small fixed overhead.
- `Mode::General` -- Pantograph Lift over a byte stream.
- `Mode::Geo2D` / `Mode::Geo3D` -- Rod-Joint Transform over point streams.

All four modes share the same range coder and the same varint/zigzag
residual encoding.

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
- Once warm, GPU vs. CPU transform time converges steadily as input size
  grows: from ~0.01x (128x slower) at 100K elements to ~0.8-0.98x (roughly
  parity, varying run to run) at 256M elements, the largest size that
  reliably fits this card's 6GB VRAM. The CPU path was still faster at
  every size actually tested here — an honest negative result, reported
  because it's what was measured, not because it's the fun answer. A GPU
  with more VRAM (to test past ~256M elements) or higher memory bandwidth
  could plausibly cross over; that is future work, not a claim made here.
- A one-off probe past the practical VRAM ceiling (400M elements) hit a
  genuine CUDA resource error, and the `pantograph_lift_forward_cuda` →
  automatic CPU fallback path handled it transparently — the
  graceful-degradation design being exercised by a real failure, not just
  a theoretical code path.

## Honest limitations / future work

- **Toroidal-style paths** (where the rod magnitude itself oscillates, not
  just its direction) aren't well modeled by a single rotation+scale
  joint, even recalibrated -- not even by the true 3D similarity joint,
  since the oscillation period here (~3 samples) is far shorter than any
  reasonable calibration block. A genuine fix needs either a richer
  per-block model (e.g. a second harmonic term) or accepting that this
  shape class is out of scope for a "small parameter set" model.
- **Interleaved-stream rANS** would let the entropy-coding stage itself
  run in parallel on GPU (unlike the current sequential adaptive range
  coder), closing the loop on an end-to-end GPU-resident codec.
- **A CPU-vs-GPU crossover past this GPU's ~256M-element practical VRAM
  ceiling** hasn't been measured (see `GPU_BENCHMARKS.md`) -- testing on a
  GPU with more VRAM, or reducing per-buffer memory (e.g. processing in
  chunks instead of one padded_len-sized allocation), would extend the
  measurement further.

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
