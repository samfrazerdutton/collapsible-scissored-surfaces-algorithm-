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
1360 round-trip checks (see `tests/test_main.cpp`) pass, including the
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
mostly-small numbers, which the same adaptive range coder (reused as-is,
just over a 257-symbol alphabet via `FenwickFreqN`) compresses well.

**Block-based, not whole-file.** A real suffix array is at best O(n log
n) to build, and this implementation deliberately uses the simpler,
easier-to-verify O(n log^2 n) prefix-doubling ("Manber-Myers") method
rather than the linear-time SA-IS algorithm -- correctness over
cleverness for a first working version (see Future Work). That cost is
why BWT operates on fixed 256KB blocks (`kBwtDefaultBlockSize`)
independently, the same reason bz2 itself caps its own block size at
900KB rather than transforming an entire file as one unit.

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

**No dedicated run-length stage.** bz2's real pipeline follows MTF with
a specialized RUNA/RUNB zero-run encoding before entropy coding; this
implementation skips that and relies on the adaptive entropy model's own
skew-handling to capture a real fraction of the same benefit, trading
some ratio for meaningfully less implementation risk (see Future Work).

**Measured, not assumed, and gated on that measurement.** `compress()`
tries this as a fourth candidate alongside raw/Pantograph-Lift/LZ. On
the 18MB real-source-code corpus (`REAL_CORPUS_BENCHMARK.md`), it never
actually won -- the LZ candidate's exact-repeat matching already covers
that corpus's redundancy better -- but trying it unconditionally still
cost roughly 10 extra seconds at the `fast` level alone (suffix-array
construction cost doesn't care whether the result ends up winning). On
realistic few-hundred-KB-to-1MB files, though (`USE_CASES.md`'s
synthetic server-log, JSON-telemetry, and sensor-CSV datasets), it won
outright every time -- 8-24% smaller than the next-best candidate,
enough to newly beat lzma on the JSON case. Since BWT's strength doesn't
correlate with the LZ-ratio signal the Pantograph Lift skip heuristic
already uses (it beat an already-strong LZ result by 24% on the JSON
case), that signal isn't a valid predictor for skipping BWT too. Instead,
`compress()` uses a plain size cutoff (`kBwtMaxInputSize`, 4MB): below
it, BWT is always tried since the cost is negligible in absolute terms
regardless of outcome; above it, the measured cost stopped being worth
paying for files that size actually measured. This is the same "compare
real measured behavior, gate on what was actually found" discipline
applied everywhere else in this codebase, not a tuned constant chosen to
make a benchmark look good.

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
reconstruction currently sits, not the true rod. It deliberately only
tries the similarity-joint model, not the xy+z composition -- the
composition's z-axis Pantograph Lift has no lossy mode, so
`compress_geo3d_lossy` can occasionally lose to `compress_geo3d` on
shapes the composition would have won (a helix with a very constant
climb rate, say); that's a known, documented tradeoff, not a bug, and it
only matters once `quant_step > 1` actually engages lossy mode.
`test_rod_joint_3d_lossy` in `tests/test_main.cpp` checks the same
properties as the 2D test (bounded per-rod error away from resync
points, exact reset at resync points, bounded absolute drift, a real
compression win over lossless, and exact lossless behavior at
`quant_step <= 1`) on a synthetic curving-and-climbing path.
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
- **Optimal (cost-based) LZ parsing** would likely close more of the
  remaining gap to bz2/lzma on realistic text (see `USE_CASES.md`): the
  current matcher's lookahead is one step (lazy matching), not a full
  search over parse choices weighted by their actual entropy-coded cost,
  which is what LZMA-class compressors do. **A greedy (non-optimal)
  version of this was actually tried and reverted**: `find_match` and the
  lazy-deferral decision were changed to select matches by an estimated
  encoded-bit cost (bytes covered per bit, a cheap integer proxy built
  from `bucket_encode`'s own bucket math -- see the reverted code's
  comments, kept in git history) instead of pure greedy-longest, on the
  theory that a much closer, slightly shorter match often costs fewer
  bits than a farther, marginally longer one. Measured honestly, it was a
  real but *mixed* result, not a net win: on the 18MB real-source-code
  corpus it shrank output by ~0.8-0.9% at every speed level (narrowing
  the gap to lzma from 13.4% to 12.4%) and on a synthetic numeric-CSV use
  case by ~5.7%, but on a synthetic server-log use case it made output
  ~7.4% *larger*, and on JSON telemetry events ~2.1% larger -- while also
  costing 20-30% more compression time everywhere, since the cost
  estimate has to be computed for every still-viable chain candidate
  instead of a cheap length comparison. The likely cause of the
  regressions: the cost estimate is evaluated greedily, per candidate,
  with no visibility into how that choice affects the cost of whatever
  comes *after* it in the parse -- so on template-like repeated data
  (e.g. a log line repeating a whole template with only a few fields
  differing), it can prefer a shorter, locally-cheaper-looking match that
  forces a worse parse of the remaining bytes. This is exactly the
  failure mode true optimal parsing (a dynamic-programming pass scoring
  the *whole* remaining parse, not one candidate in isolation) is built
  to avoid, and why a cost-aware-but-still-greedy heuristic isn't a
  substitute for it. Reverted rather than kept as an ambiguous, mixed-
  result default; a real DP-based optimal parser remains the actual
  future-work item here, not the greedy approximation.
- **BWT mode has no dedicated run-length stage for MTF's zero-runs**
  (bz2's RUNA/RUNB scheme) -- the adaptive entropy model's own skew-
  handling captures a real fraction of that benefit already (see the BWT
  section above), but a proper zero-run encoding would likely close more
  of the remaining gap to real bz2 on the cases where BWT already wins.
- **BWT's suffix-array construction is O(n log^2 n) prefix-doubling, not
  linear-time SA-IS** -- correctness-over-cleverness for a first working
  version (see the BWT section above), but it's also why BWT mode is
  capped at a few MB via `kBwtMaxInputSize`; a linear-time construction
  would remove the reason for that cap entirely, letting BWT compete on
  large files too instead of being skipped there.
- **Lossy mode's 3D path only tries the similarity joint, not the xy+z
  composition** (see the lossy-mode section above) -- extending the
  composition's z-axis Pantograph Lift to support quantization too would
  close that gap, at the cost of another lossy code path to maintain.
- **A Go build tag for non-Windows platforms** -- `bindings/go/` currently
  only implements `dll_windows.go` (this environment had no C compiler
  available to build a portable `cgo` version); a `cgo`-based
  `dll_unix.go` behind a `//go:build !windows` tag, or a `purego`-based
  one to stay cgo-free everywhere, would extend it to Linux/macOS's
  `.so`/`.dylib` without changing `csa.go`'s public API at all.
- **A genuinely cross-vendor GPU backend** (Vulkan Compute, WebGPU, or
  similar) would let the parallel block-coding kernels run on non-NVIDIA
  hardware and non-Windows/Linux platforms (macOS/Metal, mobile, WASM).
  This is a full second GPU backend in a different API, not an
  incremental addition to the existing CUDA path, and is out of scope for
  what a single-repository research project can responsibly claim to have
  built alongside everything else here.

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
