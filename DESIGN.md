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

3D point sequences compose two already-correct pieces rather than
introducing a new one: the (x, y) plane goes through the 2D Rod-Joint
transform (this captures genuine turning motion jointly -- a rotating
trajectory's x and y are individually just two out-of-phase sine waves,
each hard to predict alone, but together are exactly what a rotation
constant models), and z goes through the Pantograph Lift by itself
(captures smooth/linear vertical motion, e.g. a helix's constant climb
rate). See `BENCHMARKS.md` for what this composition wins and where it
still falls short (toroidal paths, whose xy-radius itself oscillates).

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
independent of every other pair. The CUDA path:

1. One kernel per level does a block-per-calibration-block shared-memory
   reduction (`calibrate_blocks_kernel`) to compute the four sums needed
   for each block's affine fit -- no atomics needed, since each CUDA
   thread block owns a disjoint data range.
2. The (tiny) per-block ratio/offset arithmetic runs on the host --
   there are only `padded_len / kPantographBlockSize` values per level,
   not one per sample, so this is negligible CPU time.
3. A second kernel (`transform_kernel`) applies predict+residual+update
   to every pair in parallel, reading the small per-block parameter
   arrays.
4. The `s`-array (low-pass) stays device-resident across levels; only the
   residual array (needed by the CPU-side entropy coder) and the tiny
   per-level parameter arrays come back to the host each level.

This deliberately mirrors the GPU-resident philosophy from this repo
owner's other CUDA projects (a GPU-resident CKKS homomorphic-encryption
library, and a GPU-resident LiDAR preprocessing pipeline): keep the
working array on-device across the whole pipeline stage, and be honest
about where PCIe/kernel-launch overhead dominates rather than assuming the
GPU path always wins. See `BENCHMARKS.md` for the measured CPU-vs-GPU
crossover on this machine (RTX 2060) -- at the input sizes tested here, it
does not win, exactly the same PCIe-bound pattern those other projects
already documented. It should start winning once a single compress() call
processes tens of megabytes, where the fixed per-level launch/copy
overhead amortizes over far more parallel work; that crossover point has
not yet been measured here and is future work.

## Honest limitations / future work

- **Toroidal-style paths** (where the rod magnitude itself oscillates, not
  just its direction) aren't well modeled by a single rotation+scale
  joint, even recalibrated. A genuine fix needs either a richer per-block
  model (e.g. a second harmonic term) or accepting that this shape class
  is out of scope for a "small parameter set" model.
- **True 3D similarity joints** (a calibrated quaternion / 3x3
  rotation+scale predicting one 3D rod from the previous one) would
  likely beat the current xy-plane + z-axis composition for paths that
  genuinely tumble in 3D (not just climb steadily), and is a natural
  next step.
- **Interleaved-stream rANS** would let the entropy-coding stage itself
  run in parallel on GPU (unlike the current sequential adaptive range
  coder), closing the loop on an end-to-end GPU-resident codec.
- **Large-input GPU crossover** has not been measured -- `BENCHMARKS.md`
  only covers inputs up to ~300KB, where per-level PCIe/launch overhead
  dominates. Confirming (or refuting) a crossover at multi-megabyte sizes
  is the natural next benchmarking step.
