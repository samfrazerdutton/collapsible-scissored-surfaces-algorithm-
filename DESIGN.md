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

## Four models, one entropy backend

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
`RodJoint2DResult`/`RodJoint3DSimResult` carry a `block_lag` vector (one
entry per calibration block) rather than a single file-wide value: rod `i`
in block `blk` is predicted from rod `i - block_lag[blk]` (the calibration
math is unchanged, just re-indexed). Each block searches
`kRodJointCandidateLags` = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32}
independently and keeps whichever lag gives the lowest prediction SSE for
*that block* -- this is a strict generalization of "one lag for the whole
file" (a uniform choice is just the degenerate case where every block
happens to agree), so it can track not only a fast, *constant* oscillation
period but one that **drifts partway through the sequence** -- two
concatenated shapes with different periods, or a scan whose pitch changes
-- which no single whole-file lag ever could. A dedicated test
(`test_rod_joint_lag_search_period_drift`) builds exactly that case (a
toroidal-style path whose period changes at the midpoint) and measures a
**71% reduction in total residual magnitude** versus the best possible
single forced lag for the whole file.

Per-block search picks its lag from *raw residual SSE on the true rods*,
which is cheap (no quantization, no entropy coding needed to score a
candidate) but is only a *proxy* for the real objective -- actual
entropy-coded bytes after quantization -- and a proxy can occasionally
diverge from the real thing, particularly once lossy quantization is
involved (an SSE-optimal lag can produce a residual distribution that
happens to compress worse after quantization than a slightly-less-SSE-
optimal one). This was a real regression caught during development: an
early version made per-block auto search the *only* strategy, and a
lossy-mode round-trip test that had been passing regressed to a
meaningfully larger blob. The fix keeps both strategies and compares
**actual serialized bytes**: `compress_geo2d`/`compress_geo3d` (and their
lossy variants, in `codec.cpp`) try the per-block auto search *and* the
best single whole-file-uniform forced lag (the older design, one full
encode per `kRodJointCandidateLags` candidate), and keep whichever
genuinely serializes smaller. This guarantees the new per-block search can
only ever match or beat the old whole-file design, never quietly regress
it -- the same "compare real measured output, don't trust a proxy" ethos
this project applies everywhere else. For `toroidal.xyz`, this still turns
a 15%-smaller loss against lzma into a 32%-smaller win (see
`BENCHMARKS.md`); every other shape ticked up slightly too, since lag=1
usually isn't *exactly* optimal even when it's close. This is the same
idea as long-term/pitch prediction in speech and audio codecs, applied to
rod sequences instead of waveform samples.

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
1528 round-trip checks (see `tests/test_main.cpp`) pass, including the
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
them outright. `REAL_CORPUS_BENCHMARK.md` runs the same comparison
against 18MB of real, unmodified C++ source code (not text written or
generated for this project) -- the harshest, most credible test in this
repo, since nobody wrote that data to flatter a particular compressor.

**Speed/ratio levels** (`compress(..., lz_max_chain, lz_nice_length)`,
`scissorc compress --level fast|balanced|high`): the match search's
`max_chain` (how many hash-chain candidates to examine per position) and
`nice_length` (the match length at which the search stops early and
skips the lazy lookahead entirely, since checking for something even
better than an already-excellent match is the single most expensive
thing this matcher does) are a real speed-vs-ratio knob, exactly the
"level" every production LZ compressor exposes (gzip -1..-9, zstd
-1..-22). This was not a hypothetical concern: the first version of this
matcher took **~30s to compress 18MB** of real source code at only 0.6
MB/s, because highly repetitive real text produces very long hash chains
that got walked in full, twice per position (once for the current
position, once for the lazy lookahead), for every match no matter how
already-excellent it was. Adding the early-exit and tuning the defaults
against both `REAL_CORPUS_BENCHMARK.md` and `USE_CASES.md` together (not
just one or the other) brought the default ("balanced") level to ~5s
(~6x faster) for a ~2% ratio cost; `--level fast` compresses the same
18MB in ~1.5s -- faster than lzma -9, while still beating both gzip and
bz2 on size. `--level high` recovers close to the original thorough
search's ratio (within 13% of lzma) at the original ~30s cost. None of
this affects the bitstream format -- it's a pure encoder-side search
parameter, so decoding is identical regardless of which level compressed
a file.

### 5. BWT + move-to-front (`include/csa/bwt_transform.hpp`, `include/csa/bwt_codec.hpp`)

A third, distinct kind of redundancy from the two above: the Pantograph
Lift/Rod-Joint transforms are predictive (good when nearby *samples*
relate by a small rule) and the LZ matcher finds exact repeated
*substrings* far apart; the Burrows-Wheeler Transform instead exploits
local *context* statistics -- "this byte tends to follow this preceding
context" -- without needing an exact repeat at all. This is bz2's core
technique, and it's a genuinely different lever: BWT permutes the input
so every byte migrates next to every other occurrence sharing its
context, turning that statistical tendency into long runs of identical
or near-identical bytes; a move-to-front pass then turns those runs into
mostly-small numbers -- runs of the resulting all-dominant rank-0 are
further collapsed by a RUNA/RUNB zero-run encoding (see below) -- which
the same adaptive range coder (reused as-is, just over a 258-symbol
alphabet via `FenwickFreqN`) compresses well.

**Block-based, not whole-file.** `bwt_encode_block` builds each block's
suffix array via SA-IS (Nong/Zhang/Chen's linear-time construction by
induced sorting) -- but even at O(n), BWT operates on fixed 256KB blocks
(`kBwtDefaultBlockSize`) rather than the whole file at once, the same
reason bz2 itself caps its own block size at 900KB: smaller, independent
blocks bound memory and let the transform's local-context clustering
work over a bounded window rather than needing to hold and index an
entire multi-hundred-MB file's suffix array at once.

**Linear time, trusted through cross-validation, not just hand-checked
examples.** SA-IS is intricate enough (LMS-substring classification,
induced sorting in two passes, a recursive reduction when LMS substrings
tie) that no amount of hand-checking a small example would be enough to
trust it directly. This codebase keeps its original, much simpler O(n
log^2 n) prefix-doubling construction (`build_suffix_array`, "Manber-
Myers" style: repeated rank-doubling rounds, each an `std::sort` keyed by
the current `(rank[i], rank[i+k])` pair) specifically as a reference
implementation, and `test_bwt_sais_matches_reference` cross-validates
SA-IS against it on ~150+ randomized cases -- deliberately including
tiny alphabets (which force deep recursion, the likeliest place for a
subtle bug to hide), fully periodic/constant blocks, and the hand-
derived `BWT("banana")` case -- before SA-IS was ever trusted to replace
the reference construction in `bwt_encode_block` itself. Switching paid
off immediately: the same 312KB `text_repetitive.bin` used in
`BENCHMARKS.md` compresses in ~25ms now, down from ~174ms with the old
construction (~7x faster), and trying BWT on the full 18MB real-source-
code corpus (see below) now costs ~1.1-1.2s instead of ~10s.

**Sentinel-terminated, not raw cyclic rotations.** The classic textbook
BWT sorts a block's *cyclic rotations* directly, which requires careful
tie-breaking for periodic blocks (e.g. one repeated byte, where many
rotations are literally identical strings). This implementation instead
appends one unique sentinel symbol -- smaller than every real byte --
before building the suffix array, which makes every suffix distinct by
construction (no periodicity special-casing needed at all), at the cost
of one extra symbol per block. `BwtBlockResult`'s alphabet is therefore
257 values (0 = sentinel, 1..256 = original byte + 1), and decoding is
the standard LF/`next`-mapping inverse: build a `next[]` array from the
symbol counts, walk it starting from the sentinel's row for exactly
`block_size` steps, and read off the original bytes. This was derived
and hand-verified against the classic `BWT("banana")` textbook example
before being trusted with anything larger --
`test_bwt_transform` encodes that exact derivation as a permanent
regression test.

**RUNA/RUNB zero-run encoding.** bz2's real pipeline follows MTF with a
specialized zero-run encoding, and this implementation now has one too:
MTF's rank-0 symbol dominates its output on BWT-clustered data (every
byte matching the most-recently-seen one produces rank 0), so a run of N
consecutive zeros is replaced by O(log N) reserved-symbol tokens instead
of N individual rank-0 symbols through the entropy coder. The run length
is encoded in bijective base-2 (digit set {1, 2}, least-significant
digit first: repeatedly take `d = ((N-1) mod 2) + 1`, emit RUNA for d=1
or RUNB for d=2, then `N = (N-d)/2`, until N reaches 0) -- a bijection,
so every N >= 1 has exactly one such digit sequence and decoding
(`N = sum(d_i * 2^i)`) is unambiguous. This adds two reserved symbols to
the alphabet (RUNA=0, RUNB=1) and shifts every nonzero MTF rank up by
one to make room (258 symbols total, up from 257). Verified with the
same rigor as the transform itself: hand-derived and checked against
worked examples before trusting it, then measured. The decoder can't
just loop a fixed number of times per block anymore (a run compresses
multiple ranks into one entropy symbol), so it decodes one RLE symbol at
a time and expands runs as they're recognized until the block's known
rank count is reached -- including the case where a run ends exactly at
a block boundary with no real symbol following it in that block.

**Measured, not assumed, and gated on that measurement.** `compress()`
tries this as a fourth candidate alongside raw/Pantograph-Lift/LZ. On
the 18MB real-source-code corpus (`REAL_CORPUS_BENCHMARK.md`), it never
actually wins -- the LZ candidate's exact-repeat matching already covers
that corpus's redundancy better -- and that outcome hasn't changed since
switching to SA-IS; what changed is the *cost* of finding that out: back
when suffix-array construction was O(n log^2 n), trying it unconditionally
cost roughly 10 extra seconds at the `fast` level alone, purely from
construction cost that didn't care whether the result would win. With
SA-IS, that same corpus now costs only ~1.1-1.2s extra -- roughly the
same order of magnitude as the LZ candidate's own cost, not a
disproportionate outlier anymore -- so `compress()` now tries BWT on it
unconditionally rather than skipping it. On realistic few-hundred-KB-
to-1MB files, though (`USE_CASES.md`'s synthetic server-log, JSON-
telemetry, and sensor-CSV datasets), it won outright every time even
before RUNA/RUNB was added; adding RUNA/RUNB shrank those same outputs by
a further **14-22%** (605KB server log: 53.4KB -> 41.8KB; 616KB JSON
events: 54.7KB -> 47.3KB; 800KB sensor CSV: 134.5KB -> 114.8KB), enough
to newly beat lzma on the server-log case too, on top of the JSON case
it already beat. The one honest cost found: a tiny (4KB) synthetic file
got a few percent *larger* (1,718 -> 1,768 bytes) -- plausibly the larger
258-symbol alphabet's adaptive model taking marginally longer to warm up
on a block too short to have many long zero-runs to amortize that
against. `compress()`'s real-size comparison against the other three
candidates means this never costs anything beyond that specific measured
difference; it isn't hidden.

`compress()` still uses a plain size cutoff (`kBwtMaxInputSize`, raised
from 4MB to 64MB once SA-IS made the per-byte cost proportionate to the
other candidates' rather than a disproportionate outlier) rather than
removing the cap outright: this codebase has only directly measured
SA-IS's real-world cost up to the 18MB corpus, and a generous-but-finite
bound is the honest position pending measurement further out, not a
claim of unbounded confidence. It also still isn't gated on the LZ-ratio
signal the Pantograph Lift skip heuristic uses, since BWT's strength
doesn't correlate with it (it beat an already-strong LZ result by 24% on
the JSON case) -- that signal simply isn't a valid predictor for BWT.
This is the same "compare real measured behavior, gate on what was
actually found" discipline applied everywhere else in this codebase, not
a tuned constant chosen to make a benchmark look good.

## Container format

One shared bitstream, `include/csa/codec.hpp`:

- `Mode::Raw` -- passthrough with an 13-byte header, chosen automatically
  whenever a transformed representation would be *larger* (e.g.
  incompressible random data), so CSA never inflates input by more than a
  small fixed overhead.
- `Mode::General` -- Pantograph Lift over a byte stream.
- `Mode::GeneralLZ` -- the LZ dictionary matcher over a byte stream.
- `Mode::GeneralBWT` -- the BWT + move-to-front mode over a byte stream.
- `Mode::Geo2D` / `Mode::Geo3D` -- Rod-Joint Transform over point streams.
- `Mode::Pose` -- 6-DOF pose streams (position via Geo3D + orientation via
  the Quaternion Joint); see its own section below.

`compress()` tries `Raw`/`General`/`GeneralLZ`/`GeneralBWT` for any
byte-stream input and keeps whichever encodes smallest (`GeneralBWT` is
skipped above `kBwtMaxInputSize`; see its own section below), so callers
never need to know in advance whether their data is more
"smooth/predictive", "repeated-substring", or "local-context-statistics"
in nature.

### Adaptive heuristic: skipping Pantograph Lift on measured evidence

`compress()` computes the LZ candidate first, and only runs Pantograph
Lift at all if the input is at least `kAdaptiveSizeThreshold` (100KB) and
LZ's result wasn't already very strong (`< kAdaptiveLzStrongRatio`, 35% of
the input). Below that size, trying every candidate costs milliseconds
regardless, so there's no reason to skip anything. Above it, an LZ result
that good is strong *measured* evidence (not a guessed "this looks like
text" content-type classifier) that the data has exploitable repeated-
substring structure Pantograph Lift's predictive model is very unlikely to
beat. This is an honest tradeoff, not a free lunch: it is possible for an
adversarial file to have LZ do reasonably well *and* have Pantograph Lift
do better still, in which case this heuristic gives up a small, unmeasured
amount of ratio for a real, measured amount of speed on large inputs. See
`tests/test_main.cpp`'s `test_codec_adaptive_skip` for the regression
guard.

### Lossy mode: quantized residuals with periodic exact resync

The Rod-Joint Transform's 2D path (`rod_joint_2d_forward`/`_inverse`) and
`codec.cpp`'s `compress_geo2d_lossy` support genuine lossy compression,
not just lossless -- "flexible joints that approximate the target shape"
in the pantograph-lattice metaphor, versus rigid/exact ones. Lossless and
lossy share one code path parameterized by `quant_step`: a rod's residual
is quantized to the nearest multiple of `quant_step` (rather than stored
exactly), and `quant_step == 1` is provably identical to the original
lossless behavior (`round(d/1)*1 == d` always), so this was a safe,
non-breaking generalization of the existing transform, not a parallel
implementation.

The design is closed-loop (the same idea DPCM and video codecs use):
prediction for rod `i` is always based on the *reconstructed* (possibly
lossy) rod `i-lag`, never the true original one, so the encoder computes
exactly what the decoder will independently reconstruct. Per-rod error is
therefore bounded by `quant_step/2` (see `rod_joint_2d_error_bound()`) --
but because rods accumulate into an absolute point path, that per-rod
error is a random walk over the sequence, and *absolute* position error
can drift further the longer a run goes since the last exact point.
`resync_interval` bounds that: every `resync_interval` rods, one rod is
computed to land the reconstructed *absolute point* exactly on the true
one, not just to reproduce the true *rod* value. This distinction mattered
in practice: an earlier version made the resync rod itself exact (equal
to the true rod) but added it to whatever position had already drifted,
which only stops drift from growing further without ever undoing it --
caught by `tests/test_main.cpp`'s `test_rod_joint_2d_lossy`, which checks
that error resets to exactly zero *at* resync points, not just that it
stays "small".

`scissorc compress-geo2d-lossy <in> <out> --quant N --resync N` exposes
this from the CLI, reporting the real *measured* max coordinate error
(by actually decoding and comparing), not just the theoretical bound.

The same design is generalized to 3D on the true similarity joint
(`rod_joint_3d_similarity_forward`/`_inverse`, `compress_geo3d_lossy`):
prediction still uses the reconstructed rod history, quantization and
error bound (`rod_joint_3d_error_bound()`) are identical in form, and
resync still targets the true absolute *point* given wherever
reconstruction currently sits, not the true rod.
`test_rod_joint_3d_lossy` in `tests/test_main.cpp` checks the same
properties as the 2D test (bounded per-rod error away from resync
points, exact reset at resync points, bounded absolute drift, a real
compression win over lossless, and exact lossless behavior at
`quant_step <= 1`) on a synthetic curving-and-climbing path.

`compress_geo3d_lossy` also tries the xy+z composition model, not just
the similarity joint: the composition's z-axis Pantograph Lift now
supports the same closed-loop quantization (`LiftResult::quant_step`,
threaded through `pantograph_lift_forward`/`_inverse`), so
`rod_joint_3d_forward` takes an independent `z_quant_step` alongside the
xy plane's own `xy_quant_step`/`xy_resync_interval` (see
`include/csa/rod_joint_transform.hpp`). This closes what used to be a
known gap: previously the composition model could only ever be tried
losslessly, so a helix with a very constant climb rate -- exactly the
shape the composition wins on -- would lose out to the similarity joint
once `quant_step > 1` engaged lossy mode, even though the composition
would have compressed it better. `compress_geo3d_lossy` now builds both
candidates with the same `quant_step`/`resync_interval` and keeps
whichever serializes smaller, the same auto-select `compress_geo3d`
already does losslessly. `test_pantograph_lift_lossy` in
`tests/test_main.cpp` exercises this end-to-end on a synthetic helix
(constant-radius rotation with a steady, noisy z climb) through the
public `compress_geo3d`/`compress_geo3d_lossy`/`decompress_geo3d` API,
confirming a real compression win, bounded coordinate error, and exact
lossless behavior at `quant_step <= 1`. On a synthetic 20,000-point helix
(constant-radius rotation, noisy steady climb -- the shape class the
composition model is meant for), `compress-geo3d-lossy --quant 20
--resync 64` measures 71,908 -> 44,329 bytes (~38% smaller than lossless
`compress-geo3d`) at a measured max coordinate error of 0.567 units,
confirming the composition model's new lossy z-axis is a real win, not
just a theoretical one.
`scissorc compress-geo3d-lossy <in.xyz> <out> --quant N --resync N`
exposes it from the CLI the same way, and it's wired through the C ABI
(`csa_compress_geo3d_lossy`) and Python bindings
(`csa.compress_geo3d_lossy`) alongside the 2D lossy path.

### C ABI and language bindings (`include/csa/csa_capi.h`, `libcsa`)

`src/csa_capi.cpp` wraps the core `compress`/`decompress`/`compress_geo2d`/
`compress_geo3d` functions (plus the lossy variant) behind a pure C
interface -- no C++ types (`std::vector`, `std::string`, `Point2i`, ...)
cross the boundary, only plain pointers/sizes and a `csa_buffer{data,
size}` struct, which is what lets a shared library built with one compiler
toolchain (MSVC) link correctly from a program built with a different one
(GCC, Clang) or a completely different language. This is the actual
prerequisite for any language binding; the C++ headers alone are not
ABI-stable across compilers.

Errors are reported via a thread-local last-error string
(`csa_last_error()`) rather than exceptions or error codes mixed into the
return value, since a `csa_buffer{nullptr, 0}` is ambiguous between "empty
result" (e.g. decompressing an empty file) and "failed" -- callers check
`csa_last_error()` to disambiguate, and it's cleared on every success.

`bindings/python/csa.py` is a `ctypes` wrapper on top of this ABI --
`import csa; csa.compress(data)` -- tested end-to-end against the actual
built shared library (`bindings/python/test_bindings.py`), not mocked.
`tests/test_capi.cpp` similarly links against the real shared library
(not `csa_core` directly), specifically to catch real symbol-export/
linking problems that testing the C++ core alone never would.

`bindings/rust/` is the same idea via `extern "C"` FFI instead of
`ctypes`: a `build.rs` locates the built shared library (same
`CSA_LIB_PATH`-env-var-or-relative-`build/`-directory rule as the Python
bindings), links against it, and copies the runtime `.dll`/`.so`/`.dylib`
next to whatever binaries `cargo` produces (Windows needs the library in
the launching executable's own directory or on `PATH` to find it at
runtime, which a link-only `cargo:rustc-link-lib` doesn't handle by
itself). Multi-byte fields are read from the returned buffer with
`read_unaligned` rather than a direct pointer cast, since a malloc'd
buffer's alignment isn't part of the C ABI's contract even though it
happens to work out in practice. `bindings/rust/tests/integration.rs`
mirrors `test_bindings.py` case-for-case (general round-trip, geo2d
round-trip, 2D and 3D lossy bounded-error checks, error-path handling) and
runs via `cargo test` against the real library, the same testing
discipline as the Python and C-ABI test suites.

`bindings/csharp/` (a `Csa` class library plus a `Csa.Tests` console app)
does the same via P/Invoke. One real bug surfaced building it: the managed
assembly can't be named `Csa.dll`, because it collides case-insensitively
with the native `csa.dll` once both land in the same build output
directory on Windows -- whichever one the loader picked up second would
either overwrite the other on disk or get loaded in its place, and the
actual failure mode was a `BadImageFormatException` when the CLR tried to
read the native DLL's bytes as a managed assembly. Fixed by setting
`<AssemblyName>CsaSharp</AssemblyName>` (the C# namespace stays `Csa` for
ergonomic `using Csa;`) -- a namespace and a native library sharing an
on-disk name is a real, project-specific gotcha worth documenting, not
just working around silently.

`bindings/go/` is the most different of the four: this environment has no
C compiler available (no MinGW-w64 gcc; a chocolatey install attempt
failed on a sandbox permissions error), so `cgo` -- the usual way Go binds
to a C library -- isn't an option here. Instead it calls into `csa.dll`
directly via `syscall.LoadDLL`/`Proc.Call`, the same mechanism the Go
standard library itself uses for arbitrary Windows DLLs (no C compiler or
static linking against the MSVC-built import library required at all,
which sidesteps the cross-toolchain MSVC-vs-MinGW `.lib` compatibility
problems a cgo build would have hit anyway). The one genuinely tricky part
this technique exposes directly: `csa_buffer` is 16 bytes and returned
*by value*, and the Microsoft x64 calling convention returns any such
"non-trivial" struct via a **hidden pointer** passed as an implicit first
argument (the caller allocates the 16 bytes and passes its address; the
callee writes `{data, size}` there) -- every buffer-returning call in
`bindings/go/csa/csa.go` passes that hidden pointer as `args[0]` before
the function's real arguments. This is exactly the kind of ABI detail a C
compiler normally handles invisibly; getting it right by hand was
verified by cross-checking output byte-for-byte against the Rust/C#/Python
bindings on identical inputs (all four report the same compressed sizes
on the same test data), not just by the Go tests passing in isolation.
`go vet` still flags one `uintptr`-to-`unsafe.Pointer` conversion in
`goStringFromCStr` as a possible misuse -- documented in code as a known,
reviewed false positive (the address in question is a `syscall.Proc.Call`
return value pointing into libcsa's C heap, never Go-GC-managed memory in
the first place, so the moving-GC hazard that rule exists to prevent
doesn't apply), not a bug.

`dll_unix.go` (behind `//go:build !windows`) extends the same no-cgo
approach to Linux/macOS using
[`purego`](https://github.com/ebitengine/purego) (pinned to v0.8.4
specifically because it's the newest release whose own `go.mod` still
says `go 1.18`, rather than the current `v0.11.0`, which would have
forced this module's minimum Go version up to `1.25` for every
consumer including Windows-only ones, just to gain access to a Unix code
path most of them will never compile) for `dlopen`/`dlsym`, mirroring
`dll_windows.go`'s `syscall.LoadDLL`/`Proc.Call` approach one level down.
The two platforms' loaders can't share `csa.go`'s buffer-returning call
path, though, because they disagree on how `csa_buffer` (two eightbytes:
a pointer and a size) comes back from a function call: the Microsoft x64
ABI uses the hidden-out-pointer convention described above, while the
System V AMD64 ABI (Linux) and the equivalent Darwin ABI instead pack
both eightbytes directly into the RAX:RDX return registers -- no hidden
pointer at all. `purego.SyscallN` conveniently exposes both registers as
`(r1, r2)`, so `dll_unix.go`'s own `bufferReturningCall` reconstructs
`csaBuffer{data: r1, size: r2}` directly instead of reusing
`dll_windows.go`'s hidden-pointer version; each platform file now owns
its own `bufferReturningCall`, not just the smaller `lazyProc`/
`mustLoadLibrary`/`envOr`/`fileExists` surface originally anticipated.
**Honesty caveat**: this file was written carefully and cross-compiled
clean for `linux/amd64`, `linux/arm64`, `darwin/amd64`, and
`darwin/arm64` (`GOOS=... GOARCH=... CGO_ENABLED=0 go build ./...`), but
this development environment is Windows-only, so unlike literally
everything else in this repository, it has never actually been run --
no real machine loaded a real `libcsa.so`/`.dylib` and exercised
`csa_test.go` against it. The ABI reasoning is the standard, well-
documented behavior for both platforms, and the Windows path (same
loader shape, different ABI convention) is proof the overall design
works, but this is the one part of the codebase resting on review rather
than measurement, and it should be treated that way until someone runs
it on real Linux/macOS hardware.

## Quaternion Joint and 6-DOF pose streams (`quaternion_joint.hpp`/`.cpp`)

Everything above models *position* sequences. A real 6-DOF pose stream
(a VR/AR headset or controller track, a drone or robot's odometry, a SLAM
camera path) also has an *orientation* half -- a unit quaternion per
sample -- and `REAL_GEO_BENCHMARK.md`'s finding (Rod-Joint's calibrated-
rotation model genuinely fits continuous paths, just not LiDAR-style
point clouds) is exactly the reasoning that motivated building an
orientation-domain analogue: real tracking data has *locally consistent
angular velocity* (a smoothly turning head, a drone banking through a
turn) the same way real GPS/vehicle paths have locally consistent
direction+speed, so the same "calibrate a small transform per block,
predict from reconstructed history" idea should transfer.

The model: predict `Q[i]` from an earlier `Q[i-lag]` via a single
calibrated "delta" quaternion `D`, applied by **right** multiplication --
`Q_pred[i] = Q[i-lag] (x) D` -- recalibrated per block exactly like Rod-
Joint's ratio/matrix. Right-multiplication is a deliberate choice, not
arbitrary: real gyroscope/IMU-integrated orientation accumulates rotation
in the object's own body frame (`Q[i] = Q[i-1] (x) dq_body`), so `D`
approximates that block's characteristic per-lag-step body-frame delta.
(Left-multiplication would instead model rotation about a fixed *world*
axis -- the less common case for something like a head or drone actually
turning.)

Calibrating `D` turned out to have an unusually clean closed form,
simpler than the 3D position joint's Horn's-method eigenvector iteration
(`calibrate_3d_block`): right-multiplication by a quaternion `a` is a
*linear* map of the other operand (`a (x) D = R(a) * D` for a 4x4 matrix
`R(a)` built from `a`'s components), and quaternion multiplication is
norm-multiplicative (`|p (x) q| = |p||q|` for *every* `p, q`, not just
unit ones) -- together these mean `R(a)^T R(a) = |a|^2 * I`, which
collapses the usual least-squares normal-equations matrix down to a
*scalar* multiple of the identity for any number of calibration pairs:

```
D = ( sum_i R(a_i)^T b_i ) / ( sum_i |a_i|^2 )
```

No matrix inversion or iteration needed, for any candidate lag -- which
makes lag search (reusing `kRodJointCandidateLags`, the exact same
mechanism and period-drift benefit already proven for position rods on
the toroidal path) essentially free per candidate. On a synthetic
oscillating-angular-velocity test case (two different periods, first vs.
second half of the sequence -- the orientation analogue of
`test_rod_joint_lag_search_period_drift`), per-block auto search beat the
best single whole-file lag by **79.5%** less residual magnitude, even
better than the 71% position rods achieved on the analogous shape.

Unlike position (where *rods*, not points, get predicted, and
reconstructed rods must be summed back into an absolute point), `Q[i]`
*is* the absolute state already -- there's no accumulation step. What
still needs care under lossy quantization is the same compounding-error
shape every other lossy mode in this codebase handles: `Q_pred[i]`
depends on the *reconstructed* `Q[i-lag]` (closed-loop DPCM), so a
quantization error at `i-lag` can propagate forward through every later
prediction that depends on it. `resync_interval` bounds this exactly like
Rod-Joint's own: every `resync_interval`-th quaternion is stored exactly.
`quant_step <= 1` is exactly lossless, the same unification every other
lossy mode here uses.

What this deliberately does not do: model smoothly *accelerating*
angular velocity within one calibration block (only per-block
recalibration handles that, same compromise as everywhere else in this
codebase), or use quaternion exponentiation/SLERP to compose a multi-step
prediction -- `D` is fit directly against whatever `(Q[i-lag], Q[i])`
pairs actually occur, which is well-posed regardless of whether real
angular velocity was exactly constant over that gap; it just won't
compress as well if it wasn't.

`codec.cpp`'s `compress_pose`/`decompress_pose`/`compress_pose_lossy`
combine this with the existing Geo3D position auto-select into one
container (`Mode::Pose`): position and orientation are compressed
independently (unrelated structure -- a rotating trajectory vs. a
rotation sequence) and concatenated, position as a complete self-
contained `compress_geo3d` blob (length-prefixed, since it isn't last),
orientation as a `serialize_quat_joint` payload (not self-contained --
only ever read back immediately after the position sub-blob, so it needs
no magic header of its own). `scissorc compress-pose`/`decompress-pose`/
`compress-pose-lossy` expose it from the CLI, reading/writing plain-text
"x y z qw qx qy qz" lines; `tests/test_main.cpp`'s `test_quaternion_joint`/
`test_quaternion_joint_lossy`/`test_pose_codec` check round-trip
correctness (including an adversarial random-quaternion case), the lag-
search claim above, bounded lossy error, and a real compression win on a
synthetic drone-circling-while-yawing pose stream (**83.3%** smaller than
a fair raw-packed baseline, lossless; a further **42%** smaller in lossy
mode with modest bounded error). `csa_capi.h`/`csa_capi.cpp` expose
`csa_compress_pose`/`csa_compress_pose_lossy`/`csa_decompress_pose` the
same way as the existing geo2d/geo3d functions (interleaved int32, 7 per
pose), and all four language bindings (Python/Rust/C#/Go) wrap them the
same way they wrap `compress_geo3d`.

**Real-data validation** (`REAL_POSE_BENCHMARK.md`): three real ground-
truth 6-DOF trajectories -- EuRoC MAV Vicon Room 1 "02" (a real drone
flight, 16,702 poses), TUM RGB-D freiburg2/desk (a real handheld camera,
20,957 poses), and KITTI odometry sequence 00 (a real car driving through
Karlsruhe, 4,541 poses, rotation matrices converted to quaternions and
independently verified to ~1e-7 reconstruction accuracy). `compress_pose`
beats lzma -9 (the strongest general-purpose reference tested) on **all
three**: 30.6% smaller on the drone flight, 4.8% smaller on the handheld
camera, 4.8% smaller on the vehicle drive -- largest on the smoothest
motion, smallest on the least smooth, exactly as the calibrated-rotation
model predicts. Orientation is the *larger* half of the compressed size
on all three (51-60%), not a minor add-on. No specialized 6-DOF pose
competitor exists yet to test against (unlike LASzip for LiDAR), so this
result says CSA beats general-purpose compression on real tracking data,
not that it beats a best-in-class specialized codec that doesn't exist
yet.

### Adaptive-resolution block calibration (`adaptive_partition.hpp`)

Measuring where `compress_pose`'s bytes actually go on the three real
datasets above (`bench/pose_breakdown_tool.cpp`, a diagnostic, not part
of the shipped codec) found two things that reshaped this work: per-block
calibration-parameter overhead was under 5% of compressed size
everywhere (so a hierarchical scheme aimed at shrinking *that* -- the
literal reading of "make the geometric transforms as recursively
multi-scale as the Pantograph Lift already is" -- wouldn't have paid
off), but per-block *residual* energy varied a lot within a single file
(coefficient of variation up to ~5.8 on real data) -- a fixed block size
was forcing one calibration resolution onto stretches that are actually
very different in how predictable they are.

`adaptive_partition.hpp` targets that directly: a bottom-up greedy merge
that starts at maximal resolution (every `min_block` samples its own
block) and extends the current block by one more chunk at a time,
keeping the merge only if forcing the new chunk to share the combined
calibration doesn't cost much accuracy specifically *on that chunk*
(not diluted by however much easy history has already accumulated --
an earlier version of this compared against the whole accumulated
block's total SSE and collapsed everything into one giant block as a
result, since one bad chunk's cost becomes negligible against an
ever-growing accumulated total; see the header comment for the full
postmortem). `rod_joint_3d_similarity_forward_adaptive`/
`quaternion_joint_forward_adaptive` apply this to the 3D similarity
joint and the Quaternion Joint respectively; `codec.cpp`'s
`best_geo3d_similarity_encoding`/`best_quat_encoding` try it alongside
every existing fixed-block candidate and keep whichever actually
serializes smaller, so it can only ever help or tie, never regress.

Real, measured effect on the same three real datasets (re-running
`bench/real_pose_benchmark.py` after adding this): EuRoC (drone flight)
improved from 160,151 to 146,999 bytes (its lzma -9 margin widened from
30.6% to 36.3%), KITTI (vehicle driving) improved from 74,590 to 72,093
bytes (4.8% to 8.0% margin over lzma -9), and TUM (handheld camera)
barely moved (212,295 to 212,162 bytes) -- despite TUM having the
*highest* measured residual-energy CV of the three. That's a real,
somewhat counterintuitive result worth stating plainly rather than
smoothing over: high block-to-block variability doesn't automatically
mean adaptive resolution can exploit it, if the variability comes from
motion that's just uniformly hard to predict (a handheld camera's
constant small jitter) rather than distinct easy/hard *regimes* the way
EuRoC's stabilized flight and KITTI's mostly-straight driving both have.

## Streaming/incremental pose API (`pose_stream.hpp`/`.cpp`)

`compress_pose`/`decompress_pose` are batch APIs: the whole sequence must
be in memory before any output exists. A real drone, vehicle, or camera
produces a continuous feed, not one pre-collected buffer, so
`PoseStreamEncoder`/`PoseStreamDecoder` add a genuinely incremental path
on top: the encoder buffers at most `chunk_size` poses (not the whole
stream) and emits a complete, independently-decodable, length-prefixed
chunk the moment one fills; the decoder is fed bytes as they arrive (in
any chunking a real transport happens to deliver, not just one write()
per chunk) and dispatches each pose as soon as its chunk has fully
arrived and been decoded.

The implementation deliberately does not re-derive any calibration or
serialization logic: each chunk is compressed via an ordinary
`compress_pose()` call on just that chunk's poses. This works correctly
because the batch codec's own first block already handles "no history
yet" (predicts from zero until a candidate lag's lookback is satisfied) --
a chunked stream just pays that cost once per chunk instead of once per
file. That is a real, measured ratio cost, not a free abstraction:
`bench/pose_stream_demo.cpp` on a synthetic 2,000-pose feed measured
+142.6% at `chunk_size=32`, +39.8% at 128, and +13.9% at 512, against
`compress_pose` on the identical data -- `chunk_size` is a genuine
latency-vs-ratio knob (smaller chunks mean data is available to a
consumer sooner but cost more bytes; larger chunks approach batch ratio
at the cost of higher per-chunk latency), not a tuning parameter with a
free lunch at one end.

`tests/test_main.cpp`'s `test_pose_stream` validates round-trip
correctness under adversarial delivery (bytes fed to the decoder in
61-byte pieces that don't align to any chunk boundary, proving it
correctly buffers a partial chunk rather than assuming aligned delivery)
and reports the same real ratio-cost measurement.
**Honest scope note**: this trades away compress_pose's own additional
batch-only optimizations -- there is no adaptive-resolution block
calibration and no per-file composition-vs-similarity-joint model
selection in the streaming path, both because they require seeing the
whole sequence (or committing to one choice for the whole file) before
they can be applied. Each chunk always uses the fixed-block, per-block-
auto-lag-search encoding, applied independently.

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
either way. `bench/gpu_crossover.py` measures three distinct things and
keeps them separate rather than conflating them; headline findings from
its latest run:

- **The one-time GPU wake/context-creation cost.** This laptop GPU idles
  down to a low-power state between uses, and an earlier run of this
  script measured a real, fixed ~1.2s cost on the first CUDA call after
  idling, independent of input size. That specific number hasn't been
  reliably reproducible since, though: the script now polls `nvidia-smi`
  for the GPU's actual reported P-state rather than assuming a fixed
  sleep is long enough (a real gap in the earlier version, caught when a
  fixed 8s sleep once silently measured an already-warm call and called
  it "cold"), but even after confirming the GPU reached its idle
  P-state, the first call afterward has since measured only ~10ms. The
  P-state evidently isn't a reliable proxy for whatever actually causes
  the larger cost — plausibly a one-time per-driver-session cost rather
  than a per-idle-period one, which would need lower-level instrumentation
  than this script has to actually confirm. `GPU_BENCHMARKS.md` reports
  whichever outcome each run actually measured, honestly, rather than
  re-asserting the older number as settled.
- **Warm, back-to-back per-process throughput** (best of 3 fresh
  `scissorc` processes per size, GPU already awake from the prior call):
  ~0.25x at 100K elements, climbing to ~0.86-1.00x by 4M-16M elements,
  and sitting at ~0.99x (essentially parity, within run-to-run noise) at
  64M-256M in the latest run.
- **Sustained throughput within one long-lived process** (`scissorc
  bench-transform <n> --repeat N`, the realistic shape of a batch/service
  workload, reporting the steady-state average of calls 2+): a materially
  more favorable picture, reaching **~0.98-1.05x of CPU time at
  16M-256M elements**, an outright GPU win at 64M and 256M in the latest
  run. This is genuine near-parity-to-a-slight-win for a mid-range laptop
  GPU against a modern CPU on a task CPUs are naturally efficient at
  (simple, cache-friendly, branch-predictable sequential array passes) —
  reported as measured, not oversold as a definitive win everywhere, and
  not identical run to run (the exact crossover point moves by a few
  percent between runs, consistent with ordinary system noise at this
  margin).
- A one-off probe past the practical VRAM ceiling (400M elements) hit a
  genuine CUDA resource error, and the `pantograph_lift_forward_cuda` →
  automatic CPU fallback path handled it transparently — the
  graceful-degradation design being exercised by a real failure, not just
  a theoretical code path.

### Persistent GPU sessions (`CudaLiftSession`): removing the remaining per-call allocation cost

Everything above is GPU-resident *within* one `pantograph_lift_forward_cuda`
call — but that function still calls `cudaMalloc`/`cudaMallocHost` for
every device and pinned-host buffer it needs, and frees all of them again
before returning. That's the right contract for a single ad-hoc
conversion, but it's real, avoidable overhead for a caller making many
calls back to back (a long-lived worker thread inside a service, or
`compress()` processing a stream of inputs) — `cudaMalloc`/`cudaFree` are
actual driver calls with real latency, not free bookkeeping.

`CudaLiftSession` (`include/csa/pantograph_lift_cuda.hpp`,
`cuda/pantograph_lift_cuda.cu`) is the same forward pass wrapped around
buffers that persist for the session's lifetime instead of per call: its
constructor forces CUDA context creation up front (`cudaFree(0)`, the
standard idiom, so that one-time cost is paid once and explicitly rather
than folding into whichever `forward()` call happens to run first), and
each device/pinned buffer only grows — via an `ensure_capacity` helper,
the same amortized-growth idea `std::vector` uses — when a call needs more
room than any call before it; it's never shrunk or freed until the
session itself is destroyed. Reusing a larger buffer for a smaller
subsequent call is safe: every kernel indexes by the *current* call's
element counts (`half`, `padded_len`, ...), never by the buffer's
capacity, so leftover data from a previous larger call in the unused tail
of a buffer is simply never read. `codec.cpp`'s `compress()` now keeps one
`thread_local` `CudaLiftSession` and reuses it across calls automatically
— not thread-safe to share one session across threads (its buffers aren't
synchronized), so each thread gets its own, which is also exactly what a
`thread_local` gives for free.

**Measured, not assumed**: `scissorc bench-transform <n> --gpu --session
--repeat N` reuses one session across all `N` calls; dropping `--session`
times the old one-shot path on the same input/repeat count for a direct
comparison. On this machine (RTX 2060 Max-Q):

| input size | one-shot (steady-state avg) | session (steady-state avg) | speedup |
|---|---:|---:|---:|
| 50,000 elements | 1.85 ms | 0.53 ms | ~3.5x |
| 2,000,000 elements | 14.59 ms | 8.46 ms | ~1.7x |

The smaller input shows a much bigger relative win, as expected: at 50K
elements the actual transform work is tiny, so allocation overhead was a
larger fraction of the total to begin with; at 2M elements real kernel
work dominates more, so removing allocation overhead still helps
substantially but by a smaller multiple. Correctness is covered by
`tests/test_main.cpp`'s dedicated `CudaLiftSession` check, which drives one
session through small → large → small inputs and requires the result to
be bit-identical to `pantograph_lift_forward_cuda` at every step —
specifically to catch the failure mode this design has to avoid (a buffer
grown for a larger call silently corrupting or truncating a later,
smaller call's result).

## Honest limitations / future work

- **Interleaved-stream rANS** would let the entropy-coding stage itself
  run in parallel on GPU (unlike the current sequential adaptive range
  coder), closing the loop on an end-to-end GPU-resident codec.
- **A CPU-vs-GPU crossover past this GPU's ~256M-element practical VRAM
  ceiling** hasn't been measured (see `GPU_BENCHMARKS.md`) -- testing on a
  GPU with more VRAM, or reducing per-buffer memory (e.g. processing in
  chunks instead of one padded_len-sized allocation), would extend the
  measurement further.
- **The ~1.2s one-time GPU wake/context-creation cost hasn't been
  reliably reproducible** since it was first recorded -- `bench/
  gpu_crossover.py` now polls `nvidia-smi`'s reported P-state instead of
  assuming a fixed sleep is enough (an earlier version's fixed 8s sleep
  silently measured a warm call once and called it "cold"), but even
  after confirming the GPU reached its idle P-state, repeated runs since
  measured only ~10ms for the first call afterward -- not the ~1.2s
  originally observed. `GPU_BENCHMARKS.md` reports this honestly each
  run rather than re-asserting the older number. The likely explanation:
  the real cost is a one-time per-driver-session thing (paid once after
  a fresh driver load / system boot), not a per-idle-period one, which
  would need instrumentation below what a benchmark script driving
  `nvidia-smi` and `scissorc` can observe to actually confirm.
- **Optimal (cost-based) LZ parsing** was tried twice now, both times
  reverted, both documented here in full rather than quietly dropped
  (the actual code for both remains recoverable from git history, not
  just this summary):
  1. A greedy per-candidate cost heuristic: `find_match` and the lazy-
     deferral decision selected matches by an estimated encoded-bit cost
     (bytes covered per bit, a cheap integer proxy built from
     `bucket_encode`'s own bucket math) instead of pure greedy-longest,
     on the theory that a much closer, slightly shorter match often
     costs fewer bits than a farther, marginally longer one. A real,
     honest, *mixed* result: helped on the 18MB real-source-code corpus
     (~0.8-0.9% smaller at every level) and a synthetic numeric-CSV case
     (~5.7% smaller), but hurt a synthetic server-log case (~7.4%
     *larger*) and JSON telemetry (~2.1% larger), while costing 20-30%
     more compression time everywhere. Traced to the cost estimate being
     evaluated greedily per candidate with no visibility into how that
     choice affects whatever comes *after* it in the parse -- exactly
     the failure mode a true DP-based optimal parser is built to avoid.
  2. A genuine dynamic-programming optimal parser (`lz_parse_optimal`),
     built specifically to fix (1)'s blind spot: minimum-cost path over
     the whole remaining input, using a real Shannon-entropy (`-log2(p)`)
     cost model empirically derived from a baseline lazy parse of the
     same input -- literal byte frequencies, the match-vs-literal split,
     and length/distance bucket frequencies, not an ad hoc formula.
     Correctly implemented (round-trip correctness held throughout) and,
     after finding and fixing a real ~13x performance bug (the DP calls
     `find_match` at *every* position, unlike the lazy parser, which
     skips ahead once a match commits -- reusing the caller's
     `max_chain`/`nice_length` directly at that calling frequency made a
     1MB test file take 30-40s at the "high" level, fixed by giving the
     DP its own small, fixed internal search budget decoupled from the
     level the caller requested), ran at a reasonable speed. Gated to
     only run at the "high" level (its own fixed budget still means a
     second full parse, not worth paying unconditionally on "fast"/
     "balanced") and compared against the plain lazy parse by actual
     encoded size, so it could only ever help or be a no-op, never
     regress anything. Measured honestly, it never actually won: byte-
     identical output on the 18MB real-source-code corpus (at 2.4x the
     "high"-level compress time) and on synthetic templated server-log
     data. Likely cause: the cost model is derived from the lazy parse's
     *own* output statistics, so it's implicitly calibrated toward
     reproducing the lazy parser's own preferences rather than measuring
     what a genuinely different, better parse would look like -- a
     subtler, second-order version of (1)'s same "not enough independent
     signal" problem, not an implementation bug. Reverted in full rather
     than kept as a correctly-working feature that simply never helps in
     practice. A real optimal parser -- one whose cost model has some
     source of information genuinely independent of the parse it's
     trying to improve on, e.g. an actual simulated adaptive-coder state
     rather than a static empirical proxy -- remains the honest
     future-work item here; both of these attempts were real, careful
     tries at it, not strawmen, and neither paid off.
- **Cross-modal position-orientation coupling** (predicting a vehicle's
  orientation from its direction of travel, instead of from its own
  earlier orientation) was tried and reverted -- a real, carefully-
  measured negative result, not a strawman. The motivating measurement
  was real: rotating the canonical forward axis by KITTI's own ground-
  truth quaternions and comparing to the actual direction of travel
  (from consecutive reconstructed positions) gave a median misalignment
  of ~1.3 degrees -- a genuinely tight coupling, confirmed independently
  (not just asserted) by testing multiple candidate forward-axis
  conventions against the raw data before trusting the result. Built on
  that: `reference_quat_from_rod`, the "shortest arc" quaternion from a
  canonical axis to a position rod's direction (an unnormalized version
  first, revised to a fixed-magnitude-normalized version after measuring
  that an unnormalized reference's magnitude tracking the vehicle's own
  varying speed was actively hurting the fit -- a real bug caught and
  fixed during development, not a footnote), and a new predictor,
  `quaternion_joint_forward_velocity_referenced`, using the *same*
  closed-form right-multiplication calibration the lag-based Quaternion
  Joint already uses, just against this externally-supplied reference
  instead of an earlier orientation sample. Wired into `compress_pose` as
  a third candidate (alongside the existing lag-based and adaptive-block
  Quaternion Joint encodings), kept only if it actually won on real bytes
  -- the same never-regress discipline as every other candidate in this
  codebase.
  It measured decisively *better* than the lag-based model on a
  synthetic adversarial case built to test the hypothesis directly
  (erratic steering, defeating the lag-based model's "locally consistent
  angular velocity" assumption while keeping orientation tightly coupled
  to direction of travel: 73.3% smaller residual). But on real KITTI data
  -- the exact dataset the ~1.3-degree measurement came from -- it showed
  no improvement at all, losing to the lag-based model by roughly 12x in
  residual magnitude. Diagnosed, not just observed: `reference_quat_from_rod`
  derives its signal from *already-quantized* position deltas (`--scale
  1e6` in `REAL_POSE_BENCHMARK.md`), and when a vehicle isn't actively
  turning, consecutive integer rods round to the *identical* direction
  for several samples in a row even though the true orientation keeps
  changing smoothly and continuously underneath (Vicon/RTK-grade
  precision, much finer than the position stream's own quantization
  floor) -- so the derived reference is a coarse staircase next to a
  smooth signal, and a single per-block calibrated delta can't bridge
  that gap. The real ~1.3-degree directional correlation is genuine; the
  bottleneck turned out to be *achievable precision*, not correlation
  strength, and continuity-based (lag) prediction simply has access to
  far more precision (the previous orientation sample, not a position-
  derived proxy) than this predictor ever could. Reverted in full rather
  than kept as a correctly-implemented feature that only helps on
  synthetic data.
- **A genuinely cross-vendor GPU backend** (Vulkan Compute, WebGPU, or
  similar) would let the parallel block-coding kernels run on non-NVIDIA
  hardware and non-Windows/Linux platforms (macOS/Metal, mobile, WASM).
  This is a full second GPU backend in a different API, not an
  incremental addition to the existing CUDA path, and is out of scope for
  what a single-repository research project can responsibly claim to have
  built alongside everything else here.
- **A specialized 6-DOF pose competitor to benchmark against** --
  `REAL_POSE_BENCHMARK.md` could only compare `compress_pose` against
  general-purpose compressors because no dominant specialized codec for
  this domain exists yet (unlike LASzip for LiDAR or Google's polyline
  format for 2D GPS traces); if one emerges, that would be the more
  demanding comparison to run.

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
