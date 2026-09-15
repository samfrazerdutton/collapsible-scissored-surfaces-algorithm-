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

### GPU calibration for the Quaternion Joint (`quaternion_calibration_cuda.cu`)

The Pantograph Lift's GPU path above accelerates a *transform*; the
Quaternion Joint has no transform to accelerate in the same sense --
its expensive part is *calibration* (the per-block, per-candidate-lag
least-squares search), which is embarrassingly parallel across blocks
(it operates on the true, fully-known quaternion sequence, unlike the
*encoding* pass that follows, which depends on each block's
*reconstructed* history and must stay sequential regardless of what
runs the calibration). This is a genuinely different question from the
Pantograph Lift's own GPU result above, not an assumption extrapolated
from it: `quat_calibrate_all_blocks_cuda` computes, for every
(block, candidate lag) pair, the same closed-form calibrated delta and
its SSE `pick_block_lag` computes one candidate at a time on the CPU
(one CUDA thread block per pair, shared-memory tree reduction over up
to seven running sums, from which both the delta and its exact SSE fall
out via the standard least-squares identity `SSE_min = sum|b|^2 -
|acc|^2/sum|a|^2` -- no second data pass needed). Verified bit-for-bit
(within +-1 in the Q16.16 integer, for floating-point summation-order
differences between the CPU's sequential accumulation and the GPU's
tree reduction) against the CPU's own production code path
(`quaternion_joint_forward`'s forced-lag mode) across 24 blocks x 14
lags before any timing number below was trusted --
`tests/test_main.cpp`'s `test_quat_calibration_cuda_matches_cpu`.

**Measured, not assumed** (`bench/quat_calibration_gpu_crossover.cpp`,
same RTX 2060 Max-Q): calibration alone, isolated from encoding, on real
dataset sizes and much larger synthetic ones to look for any crossover:

| n (elements) | CPU calibration | GPU calibration | GPU vs CPU | calibration's share of full CPU time |
|---:|---:|---:|---:|---:|
| 4,541 (KITTI) | 0.270 ms | 0.815 ms | 3.02x slower | 25.1% |
| 16,702 (EuRoC) | 1.008 ms | 1.123 ms | 1.11x slower | 24.8% |
| 20,957 (TUM) | 1.265 ms | 1.187 ms | 0.94x (roughly even) | 25.2% |
| 100,000 | 6.075 ms | 3.567 ms | 1.70x faster | 24.0% |
| 1,000,000 | 61.144 ms | 26.848 ms | 2.28x faster | 23.6% |
| 10,000,000 | 611.114 ms | 149.582 ms | 4.09x faster | 23.8% |
| 50,000,000 | 3092.657 ms | 980.417 ms | 3.15x faster | 23.5% |

This is a genuinely different result from the Pantograph Lift's, not
the same story again: at every real dataset size this project has
actually measured (all under 21K poses), GPU calibration is a wash or a
real (if modest) loss -- but it shows a real, growing advantage starting
around 100K elements, well before the Pantograph Lift's own crossover
territory. The honest caveat is the "calibration's share" column:
calibration is a stable ~24-25% of total CPU orientation-encoding time
regardless of scale (both calibration and encoding are O(n), just with
different constants), so even the best-measured GPU speedup here (4.09x
at 10M elements) would only cut *total* orientation-encoding time by
roughly 18-19% -- a real, meaningful improvement at large enough scale,
not a dramatic one, because the sequential encoding pass this can't
touch is most of the work either way.

**Not wired into `compress_pose`**: given no real dataset measured in
this repo shows a clear win (all are under 21K poses, squarely in the
break-even-or-loss region), integrating this into production would add
real complexity (device buffer management, the same fallback plumbing
`CudaLiftSession` needed) for no benefit on the workloads actually
validated so far. The kernel and its correctness cross-check are real
and kept, should a future workload at real 100K+-pose scale (e.g.
aggregating calibration across many long trajectories at once, not one
file at a time) make it worth integrating.

### Interleaved rANS: a parallel, order-0 alternative entropy coder (built)

The adaptive order-1 range coder every mode in this codebase shares
(`range_coder.hpp`) is fundamentally sequential: each symbol's encoding
depends on the *adapted* frequency state left behind by every symbol
before it, so there is no way to parallelize one continuous range-coded
stream without changing what gets encoded. `rans_coder.hpp`/`rans_coder.cpp`
implement the standard technique for making entropy coding parallel
instead: **interleaving**. The whole input's byte histogram is quantized
once into a single static, shared frequency table (`M = 1 << scale_bits`,
`scale_bits = 14` by default), then the input is split into `num_lanes`
independent contiguous chunks, each encoded/decoded as its own
self-contained byte-oriented rANS stream (the standard structure
popularized by Fabian Giesen's public-domain `rans_byte.h`: a single
32-bit state, byte-wise renormalization, a `slot_to_symbol[M]` lookup
table for O(1) decode) against that one shared table. Because every
lane only needs the shared table plus its own slice of input, lanes have
zero cross-lane dependency -- exactly the property real GPU entropy
coders (nvcomp and friends) rely on, one lane per thread/warp. On CPU,
that same independence is exploited directly with `std::thread`: both
`encode_interleaved_rans` and `decode_interleaved_rans` run all lanes
concurrently.

Two things this was explicitly built to measure honestly rather than
assume, per `tests/test_main.cpp`'s `test_rans_coder` (1619 total C++
checks now include this):

- **Ratio cost of order-0-static vs. order-1-adaptive.** Expected to
  lose on most realistic (context-correlated) data, since the shared
  table can't exploit any position-dependent structure the adaptive
  coder tracks for free. Measured on a synthetic skewed-byte-frequency
  stream with *no* real inter-symbol structure (i.i.d. draws from a
  fixed skewed distribution): interleaved rANS actually came out
  **0.8% smaller** than `range_encode_bytes` on that data, not larger --
  because with no real context to exploit, the adaptive coder's
  per-symbol table-update overhead has nothing to pay for itself with,
  while the static table pays no such tax. The honest reading: this
  coder is competitive (not a guaranteed loss) specifically when the
  data doesn't actually have the order-1 structure the adaptive coder
  is designed to exploit -- e.g. post-transform residual streams that
  are already close to i.i.d. On real order-1-structured data this
  codebase's other formats produce, expect the adaptive coder to win;
  that comparison is exactly what any future caller should re-run on
  its own data before choosing between the two.
- **Real CPU parallel speedup.** On a 4,000,000-byte buffer: 1 lane =
  24.16ms encode / 19.59ms decode; 4 lanes = 12.27ms / 7.03ms (1.97x /
  2.79x); 8 lanes = 10.41ms / 5.37ms (2.32x / 3.65x) on this machine's
  core count. Real, not assumed -- and decode scales better than encode
  because `rans_encode_lane`'s reverse-order pass has less to overlap
  with the shared-table setup than `rans_decode_lane`'s forward pass.

**Not wired into `compress()`/`compress_pose()`**: since the ratio
comparison depends on whether the specific data has order-1 structure
(this codebase's transforms usually leave some), forcing it in as a
silent candidate would need to fall back to real per-call ratio
comparison against the existing coder anyway, and no measured workload
in this repo yet shows a clear win once threading overhead and the
extra per-lane header are accounted for at realistic buffer sizes.
Instead it ships as a standalone, fully tested capability -- exposed
through the C ABI as `csa_rans_encode`/`csa_rans_decode` (see
`csa_capi.h`) so any binding can reach for it directly when multi-
threaded CPU (or, in the future, GPU-lane) entropy coding matters more
than the last bit of ratio, e.g. compressing a very large buffer where
wall-clock time dominates. A GPU port of the decode path (the natural
target, since `rans_decode_lane`'s per-lane loop is embarrassingly
parallel) is the logical next step, following the same
validate-correctness-before-trusting-timing discipline used for the
Quaternion Joint calibration kernel above -- not yet done.

## In-browser demo (`demo/csa_demo.html`, `src/wasm_shim.cpp`)

The actual C++ core (range coder, Pantograph Lift, Rod-Joint/Quaternion
Joint transforms, LZ/BWT candidates, interleaved rANS -- everything
`compress()`/`compress_pose()`/`compress_geo3d()` use) compiles directly
to WebAssembly via Emscripten (`em++`, CPU-only: `pantograph_lift_cuda_stub.cpp`
takes the place of the real CUDA kernel, since there is no GPU in a
browser tab). `src/wasm_shim.cpp` is a thin re-export of `csa_capi.h`'s
functions with one change: every `csa_buffer`-by-value return becomes an
explicit `(pointer, out-size-param)` pair, because Emscripten lowers a
struct-by-value C return into a hidden pointer parameter that JS's
`cwrap`/`ccall` have no way to know about. `demo/csa_demo.html` embeds
the compiled module (`-s SINGLE_FILE=1` inlines the wasm binary as base64
directly in the JS, so the whole page is one self-contained file) plus
three real sample datasets -- a KITTI vehicle trajectory, a real captured
airborne LiDAR scan (`autzen/stadium-utm.laz` from PDAL's public
test-data repository, 693,895 points, full resolution, decoded via
`laspy`/`lazrs` and embedded as the same raw scaled integers LASzip
itself stores; see `REAL_GEO_BENCHMARK.md` for the specialized-competitor
comparison on this exact file), and a slice of real C++ source -- and
runs the actual codec against them live, client-side, alongside the
browser's native `CompressionStream('gzip')` as a live baseline. The
point-cloud canvas splats points directly into an `ImageData` buffer
rather than one `arc()` call per point -- the only way to draw 693,895
points at interactive speed, the same technique real point-cloud viewers
use for dense preview thumbnails.

**One real correctness bug this surfaced, fixed, and worth recording**:
`rans_coder.cpp`'s lane parallelism uses `std::thread`, which links fine
under Emscripten without `-pthread` but *aborts at runtime* the moment a
thread actually tries to start -- `-pthread` requires SharedArrayBuffer,
which requires cross-origin-isolation response headers this hosted page
doesn't control. Every other module in this codec is single-threaded
C++ with no such dependency, so this was the one place the WASM port
wasn't a free recompile. Fixed with a `run_lanes()` helper
(`src/rans_coder.cpp`) that keeps genuine `std::thread` parallelism on
every native target and only falls back to a plain sequential loop when
`__EMSCRIPTEN__` is defined without `__EMSCRIPTEN_PTHREADS__` -- verified
by rebuilding and re-running the full native test suite afterward (1619
checks, unchanged) to confirm the guard is truly a no-op off-WASM. The
practical honest consequence: the demo's rANS panel proves correctness
at every lane count live, in-browser, but not the ~2.3x/2.79x (4 lanes)
and ~2.32x/3.65x (8 lanes) encode/decode speedup already measured on the
native desktop build above -- the page says so directly rather than
implying a speedup that isn't actually happening in that tab.

### WebGPU rANS decode: a genuine in-browser parallel port (built)

The natural GPU target the CPU-thread-per-lane design was always meant
to generalize to: `rans_decode_lane`'s inner loop ported directly to a
WGSL compute shader (inlined in `demo/csa_demo.html`), one GPU thread
(invocation) per lane, each running the identical sequential decode over
its own byte range -- exactly the technique real GPU entropy coders use,
now actually running on a real GPU instead of only on CPU threads. Reuses
`encode_interleaved_rans`'s own wire format directly (parsed in JS --
`parseRansBlob`) rather than adding any new C++/WASM surface, since this
project already fully controls and tests that format.

Two real engineering problems solved to make this correct, not just
plausible:
- **Cross-lane write safety without atomics.** The GPU output buffer
  stores one `u32` per decoded *byte* (not four bytes packed per word),
  so no two lanes' writes can ever land in the same 32-bit word --
  trading 4x output memory for completely eliminating the sub-word
  read-modify-write hazard a packed byte buffer would need atomics to
  handle safely. The read-only input blob stays tightly packed (4
  bytes/word, unpacked in-shader with plain shifts), since concurrent
  reads of the same word are never a hazard.
- **`meta` is a reserved WGSL keyword.** A real compile-time catch from
  testing against the actual Dawn WebGPU implementation (via Node +
  the `webgpu` npm package) before ever shipping this to a browser tab --
  renamed to `laneInfo`.

**Correctness verified before any timing was trusted** (this project's
standing rule, applied here exactly as it was for the CUDA calibration
kernel): the WGSL shader's decode was run against real CSA-encoded blobs
at 1/4/8/32 lanes through Dawn (Google's WebGPU implementation, via the
`webgpu` npm package) in Node, byte-compared against the original input --
bit-exact at every lane count. The demo re-verifies this live, in
whatever browser is actually running it, comparing the GPU decode output
against the same buffer's WASM CPU decode before showing a single timing
number.

**Honest scope**: only the decode path is GPU-ported (the natural
target -- decode's per-lane loop is embarrassingly parallel across
lanes, same as encode, but decode is the side this session's WASM port
already couldn't parallelize at all in-browser). Encode stays WASM/CPU.
Whatever crossover point (or lack of one) a given browser/GPU shows
between CPU and GPU decode time is a real, live measurement the page
makes on its own device -- not asserted here, since it depends on GPU
dispatch/readback overhead relative to buffer size, the same kind of
crossover already measured (and reported honestly either way) for the
native CUDA calibration kernel.

## `squeeze`/`unsqueeze`: a real front door (`cli/main.cpp`)

Everything above this point in the CLI (`compress-geo2d`, `compress-pose`,
etc.) requires the caller to already know their file's shape and to pick
a `--scale` that preserves its precision -- fine for someone who already
knows this codec, a real barrier for anyone else, and the actual gap
between "a demo that proves the codec works" and "a tool someone would
reach for." `scissorc squeeze <file>` closes it: it sniffs the input
(does it parse as a whitespace-separated numeric table with a consistent
column count across >=95% of its lines? 2 columns -> geo2d, 3 -> geo3d,
7 -> pose; anything else, including any binary file, falls through to
general `compress()`) and, for the geo/pose cases, computes a `--scale`
(and `--qscale` for pose) from the *actual decimal digits present in the
file's own text* -- not a guessed default -- capping it only if needed to
keep `value * scale` inside a safe int32 range, and saying so honestly
(`capped to avoid overflow`) rather than silently losing precision.
`--quality 1-9` maps to a single `quant_step` (`1 << (10 - quality)`,
same fixed `resync_interval=64` for both position and orientation) for
anyone who wants smaller output and accepts a small bounded error without
learning this codec's four separate lossy knobs; omitting it stays
exactly lossless. Every squeeze run decodes its own output and compares
it against the input before printing anything (max coordinate/position/
orientation error, or a byte-identical check for general mode) -- so the
"lossless"/error number in the output is a fact checked on this exact
run, not something inherited from the test suite. `unsqueeze` reverses
it, auto-detecting shape from the file's own header bytes (`CSA1` =
general, `CSAG` + a dims byte = geo2d/geo3d/pose) rather than the
filename. Both refuse to silently overwrite an existing output path
without `--force` -- a real safety property, not present on the older
explicit commands (which the project's own benchmark scripts rerun
against fixed paths and depend on overwriting).

Every explicit command from before this addition is unchanged and still
exists for pipelines/scripts that already know their data's shape and
want to name the scale themselves rather than have it detected.

## Zero-backend browser app (`docs/`)

`webapp/` (below) is real, but needs Python/Flask running locally --
not something a stranger with a link can use. `docs/index.html`
is the version built for that: a single self-contained static HTML file
(the compiled WASM core inlined, same as `demo/csa_demo.html`) with no
server at all, so it can be hosted anywhere plain files are served
(GitHub Pages, Netlify, a plain `python -m http.server`) and, unlike the
Artifact demo, isn't sandboxed -- real drag-and-drop, real
`<a download>`, real files in and out, entirely client-side.

The one piece of logic that had to move: `squeeze`/`unsqueeze`'s
sniffing/auto-scale/round-trip-verify algorithm lives only in
`cli/main.cpp`, which doesn't run in a browser. `docs/index.html`
re-implements it in JS (`sniffTable`/`safeScale`/`qualityToQuantStep`),
deliberately mirroring `cli/main.cpp`'s algorithm line-for-line rather
than approximating it, plus the tiny `CSAG`-header framing
(`write_geo_header`/`write_pose_header` in the CLI) so files this page
produces are byte-compatible with ones the CLI/`webapp/` produce, and
vice versa. Required adding real WASM exports for `compress_geo2d`(`_lossy`)
(the CLI's `squeeze` needed it; the earlier WASM build only had
geo3d/pose/general) -- verified correct via Node against the real module
before being wired into anything (`csa_wasm_compress_geo2d` round-tripped
a synthetic 2D trajectory correctly at 8/32 scale/quant settings).

**Verified before being trusted, not just read through**: extracted the
actual `squeeze`/`unsqueeze` functions from the shipped HTML file (not a
copy) and ran them against the real compiled WASM module in Node,
against all three shapes plus general -- pose (500 rows, 87.9% smaller,
lossless, max error 0), geo3d lossy at quality 5 (97.1% smaller, measured
error 0.00381), geo2d (95.5% smaller, lossless), and general text
(byte-identical). Then, matching the same discipline used for `webapp/`,
did a *numeric* comparison (not byte-for-byte, since `unsqueeze` always
writes 9 decimal digits) of a squeezed-then-unsqueezed pose sample against
the original: max error 5e-6, exactly the rounding tolerance the
auto-picked scale implies -- a real, checked fact, not an assumption
carried over from the CLI's own tests.

**First real usability problem after deploying, fixed**: the page worked
correctly but felt inert -- it asked a first-time visitor to read a
paragraph, pick a sample, then click a button before showing anything,
so the most compelling thing this project can show (a huge, honest ratio
win on real structured data) was gated behind three actions nobody had a
reason to take yet. Fixed by restructuring the page, not by adding
features: the result panel moved to the very top and now auto-runs the
drone-pose sample the instant the WASM engine finishes loading (labeled
"Live, right now, in this tab" so it's clearly not a canned screenshot),
with a large animated headline stat (e.g. "3.5x smaller than gzip",
counting up with an eased transition rather than snapping into place), size bars that animate in with a staggered reveal (raw, then
gzip, then CSA, ~180ms apart) instead of appearing instantly, and the
trajectory preview now draws progressively over ~1.1s rather than all at
once, so watching it trace out reads as "this is what got reconstructed"
rather than a static illustration. All of it respects
`prefers-reduced-motion` (skips straight to final state, no motion) --
verified by running the *entire* page script (auto-run through the full
render path, not a truncated excerpt) inside a stubbed DOM in Node twice,
once with reduced-motion forced on and once with the real animation/rAF
timing loops actually exercised; both completed cleanly with the correct
final hero stat and result state.

**Hosted, live, for free**: this repo's GitHub Pages, serving
`docs/index.html` directly (repo Settings -> Pages -> Source: "Deploy
from a branch" -> Branch: `main`, folder: `/docs`), at
`https://<owner>.github.io/<repo>/` -- no server, no cost, updates
automatically on every push to `main` that touches `docs/`.

### Positioning pivot and Three.js/Web Worker rebuild

The deployed page still read as boring. The specific ask that followed
was a full "Silicon Valley startup" rebuild -- React/Next.js/Three.js,
an "Edge-Native Spatial Telemetry Codec" narrative for autonomous drone
swarms, a simulated network race against gzip, and a marketing-copy list
including claims like "vastly outperforming gzip" and "viable for
embedded RTOS on drone hardware." Adopted the real, buildable parts of
that; declined the parts that would have made the page say something
false:

- **Kept, because it's true and already measured**: framing this as
  compression for 6-DOF pose streams and point clouds -- the actual
  domain this project's real benchmarks (`REAL_POSE_BENCHMARK.md`) show
  it winning at, often by a wide margin. Real WebGL rendering of the
  reconstructed trajectory/point cloud (Three.js, ES modules from
  jsdelivr, no build step) instead of a flat 2D canvas line. A real Web
  Worker running the WASM engine off the main thread, so the 3D view's
  render loop doesn't freeze during compression -- genuinely necessary
  now that there's a continuous animation loop to protect, not present
  before. A simulated transmission-time "race" against gzip at a chosen
  bandwidth (LoRa/degraded-5G/satellite), explicitly labeled as
  illustrative/simulated, not implied as a live network test.
- **Declined**: "vastly outperforms gzip" as an unqualified claim --
  false; this codec loses to general-purpose compressors on ordinary
  files and loses decisively to the specialized LASzip codec on real
  LiDAR data (`REAL_GEO_BENCHMARK.md`). "Viable for embedded RTOS on
  drone hardware" and "zero-trust infrastructure" -- neither has been
  built or tested for that target. Replaced with a narrower, true,
  verifiable claim instead: the compression hot path runs in WASM linear
  memory with manual malloc/free, not JS object allocation, so there's
  no GC pause during compression -- a real fact about this codebase, not
  a leap to what it would take to run on real drone flight hardware. The
  full React/Next.js/Tailwind rewrite -- declined structurally: it would
  trade the "one static file, free GitHub Pages hosting, zero build
  step" property this page already has for a real build pipeline and
  likely a different host, for a page whose actual job (drop a file,
  watch it compress, download the result) doesn't need a framework.

**Architecture**: the compiled WASM module and every compute function
(`sniffTable`/`safeScale`/`squeeze`/`unsqueeze`, unchanged from the
earlier version) now live entirely inside a Web Worker, embedded as a
base64 string decoded via `TextDecoder` (not naive `atob`, since
Emscripten's `SINGLE_FILE` build embeds the wasm binary as a raw
binary-string literal with real non-ASCII byte values -- a UTF-8-unsafe
`atob`-only round-trip would have silently corrupted it) into a `Blob`
that becomes the Worker's source. The main thread only ever sees
`{blob, filename, summary, detail, shape, rawSize, gzipSize, csaSize,
preview}` come back over `postMessage`, with the compressed payload's
`ArrayBuffer` transferred (zero-copy). `preview` points gained a third
(z) coordinate for the 3D view (previously just x/y for the flat canvas
plot).

**Verified before being trusted, not just read through, including the
part added this round that's hardest to get right (Worker creation from
a Blob URL)**: built a Node harness that stubs `Worker`/`Blob`/
`URL.createObjectURL` to route through a *real* `worker_threads` process
running the *actual* embedded worker source (extracted from the shipped
HTML file, not a copy), stubbed Three.js/OrbitControls/DOM/canvas with
minimal fakes, and ran the entire page module script -- boot sequence,
auto-demo, hero stat, size bars, and the transmission-race arithmetic --
end to end. Caught one real stub-only issue along the way (a fake
`<select>` defaulting to `value:'0'` instead of its `selected` option,
producing `Infinity` in the race's time calculation) and confirmed it
was a test-harness gap, not a page bug, by seeding a realistic value and
re-running: the race then produced correct, real arithmetic (1.7s
gzip / 480ms CSA at 256kbps for the pose sample -- checks out exactly
against `55306*8/256000` and `15349*8/256000`). The one thing this
verification *cannot* cover without a real browser: whether the actual
WebGL rendering looks right. That's the one part of this change that's
unverified pending a real look.

## Local web app (`webapp/`)

The one thing `demo/csa_demo.html` (the WASM Artifact) structurally
cannot do: hand a real file back to whoever is using it -- Artifacts run
in a sandbox that blocks every download path, by design, regardless of
what the page's own JS tries. `webapp/` is a real, ordinary local web
app (Flask + a plain HTML/JS front end) that closes that gap: drop a
file in a real (non-sandboxed) browser tab, get a real compressed file
back via a normal download, or drop a `.csa` file in to restore it.

Deliberately a thin transport layer, not a second brain: `webapp/server.py`
shells out to the already-built `scissorc` binary (`squeeze`/`unsqueeze`)
rather than reimplementing the sniffing/scale-picking logic in Python, so
there is exactly one place that logic lives and exactly one place it's
tested. The server's only real jobs are (1) save the upload to a temp
file, (2) run the CLI, (3) relay its stdout report (with the server's own
temp-directory paths swapped back out for the filename the user actually
typed, so no local filesystem detail leaks into the browser) and its
output file back as a normal HTTP response with a real
`Content-Disposition: attachment` header.

Run it:
```
cd webapp
pip install -r requirements.txt
python server.py
```
then open `http://127.0.0.1:8000/` in a browser. Verified end-to-end
against real files of all three detected shapes (general text, a real
LiDAR-shaped point file, both lossless and `--quality`-lossy) via
`curl` directly against the running server before this was called done --
upload, download, and the restored/lossy file's content checked against
the original (numerically, not byte-for-byte: `unsqueeze` always writes
9 decimal digits regardless of the source file's own precision, so a
lossless round-trip's *values* match to the scale's rounding tolerance
even though the restored text isn't byte-identical to the input -- a
pre-existing property of `cmd_unsqueeze`, not something this app changed).

**First real usability problem this surfaced and fixed**: the first
version of this just relayed scissorc's bare text report, so a genuinely
compressed general file (e.g. a PDF, 87.6% smaller) came back as a flat
percentage with nothing to compare it to -- and general files aren't
even this codec's specialty, so a first-time user trying whatever's on
their desktop got an unremarkable, uncontextualized result and no path
to the case this codec actually wins big on. Fixed with three real
additions, not cosmetic ones:
- **A live gzip -9 baseline**, computed server-side with the stdlib's own
  `gzip` module on every upload, shown as a real size-comparison bar
  chart (raw / gzip / CSA) instead of a lone percentage -- so "smaller"
  always has a number to be smaller *than*.
- **One-click synthetic sample datasets** (`/api/sample/<pose|points|gps>`,
  generated server-side, not bundled binary assets) -- a drone-pose
  spiral, a LiDAR-ring-style point cloud, a GPS walk -- so the first
  thing anyone can try shows this codec's actual domain instead of
  requiring them to already have a suitable file lying around. Measured
  live through the real endpoint before shipping: the pose sample beats
  gzip by CSA being smaller than gzip's *output*, not just smaller than
  raw (146,181 -> gzip 55,306 -> CSA 15,355 bytes); the point-cloud sample
  similarly (61,921 -> gzip 19,577 -> CSA 2,710).
- **A trajectory/point preview** (a small canvas line plot) whenever the
  detected shape is geo2d/geo3d/pose, built by having the server run
  `unsqueeze` on its own freshly-produced output and parsing the restored
  text back into points -- reusing the existing CLI round-trip rather
  than adding a second decode path, same "one brain" principle as the
  rest of this app. A general (non-numeric) upload now also gets an
  explicit on-page note that it isn't this codec's specialty and the
  gzip comparison is the fair baseline there, rather than silently
  implying every file type is an equally strong case.

**Not deployed anywhere.** This only runs locally (`127.0.0.1`) until a
real hosting decision is made deliberately -- that's a different kind of
decision (cost, exposure, which files a stranger could send it) than
anything in this section, and isn't one to make silently.

## Honest limitations / future work

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
