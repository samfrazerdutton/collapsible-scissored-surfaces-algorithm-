# Sanitizers and fuzzing

**Status: this documents runs that actually happened, on this exact
codebase, with the exact commands to reproduce them -- not a claim that
"the project supports sanitizers" in the abstract. Every finding below
was a real bug (fixed) or a real, deliberately-scoped absence (named as
such), not a hypothetical.**

## What was actually run

| Sanitizer | Toolchain | Target | Result |
|---|---|---|---|
| ThreadSanitizer (TSan) | WSL2 Ubuntu, GCC 13.3.0 | `csa_tests` (full suite, incl. `csa::ThreadPool`/`csa::WorkStealingPool`) | 0 races detected |
| AddressSanitizer + UndefinedBehaviorSanitizer | WSL2 Ubuntu, GCC 13.3.0 | `csa_tests`, `csa_capi_test` (full suite) | 2 real bugs found and fixed (below); clean after |
| libFuzzer + ASan + UBSan | WSL2 Ubuntu, clang 18.1.3 | `fuzz/fuzz_decompress.cpp` against `decompress()`/`decompress_geo2d()`/`decompress_geo3d()`/`decompress_pose()`, plus (later pass) `deserialize_packet` | 8 real bugs found and fixed in the codec paths (below); `deserialize_packet` fuzzed separately and found clean (95,257 executions, 5 minutes) -- see fuzz/README.md for exact durations |

Not run: MSVC's `/fsanitize=address` (CMake wires it up for MSVC too --
see `CSA_SANITIZE` in `CMakeLists.txt` -- but this project's actual
sanitizer/fuzzing work happened on the WSL/Linux toolchain, since TSan
and libFuzzer aren't available on MSVC at all, and it was simpler to run
everything through one consistent toolchain rather than half on each).
If you run the MSVC ASan build yourself, treat it as unverified until
someone actually has.

### Reproducing the sanitizer builds

```bash
# ThreadSanitizer (GCC or Clang; NOT combinable with ASan in one binary)
cmake -S . -B build_tsan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DWITH_CUDA=OFF -DCSA_SANITIZE=thread
cmake --build build_tsan --target csa_tests
# Some kernels (this one included, on some distros) need reduced ASLR for
# TSan's fixed shadow-memory mapping to succeed -- if you see
# "FATAL: ThreadSanitizer: unexpected memory mapping", this is why:
setarch $(uname -m) -R ./build_tsan/csa_tests

# AddressSanitizer + UndefinedBehaviorSanitizer (combinable)
cmake -S . -B build_asan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DWITH_CUDA=OFF -DCSA_SANITIZE=address,undefined
cmake --build build_asan --target csa_tests csa_capi_test
./build_asan/csa_tests && ./build_asan/csa_capi_test
```

## Real bugs found and fixed

### 1. Left-shift of a negative signed value (UBSan) -- `include/csa/common.hpp`

`zigzag_encode32`/`zigzag_encode64` computed `(v << 1) ^ (v >> N)` directly
on the signed input `v`: left-shifting a negative signed integer is
undefined behavior pre-C++20 (this project targets C++17). Fixed by
casting to the unsigned type first and reconstructing the sign mask via
an unsigned shift (`0u - (uv >> 31)`), which produces the exact same bit
pattern on every two's-complement platform with no UB and no
implementation-defined step. Verified bit-for-bit unchanged output
(same compressed byte sizes, same round-trip results) across the entire
test suite before and after -- this touches the actual wire format's
residual encoding, so a silent behavior change here would have been a
real compatibility regression, not just a style fix.

### 2. Decompression bomb via unvalidated element counts (fuzzer-found) -- `src/codec.cpp`

`deserialize_geo2d`/`deserialize_geo3d_sim`/`deserialize_quat_joint`/
`deserialize_lift` each read at least one count field (record count,
block count, pantograph-lift level count) directly from untrusted bytes
and used it to sizetok `std::vector::resize()` calls with no upper
bound. The fuzzer found this within seconds of its first real run: a
~60-byte crafted input drove a `resize()` call requesting a
multi-terabyte allocation (`0x16000000004` bytes), aborting the process.
Fixed with two real checks, not one:
- `check_reasonable_count()`: an absolute sanity cap (100 million
  elements -- generous headroom over the largest real dataset this
  project's own benchmarks decode, the 693,895-point Autzen LiDAR scan)
  applied to counts used *before* any range-decoded payload exists to
  cross-check them against.
- `check_count_feasible()`: a mathematically exact bound (requested
  element count can never exceed the actual decoded payload's byte
  length, since every element takes at least one byte to varint-encode)
  applied once the real payload is available -- this one can never
  false-positive on honestly-produced data, it's a necessary condition.

`range_decode_bytes()` (`src/range_coder.cpp`) got the same treatment
independently: its own `output_length` parameter (also read directly
from untrusted bytes, and shared by all four deserializers above) is now
capped at `kMaxRangeDecodedBytes` (1 GiB) before it's used to `reserve()`
anything.

### 3. Integer-overflow bypassing a bounds check (fuzzer-found) -- `src/codec.cpp`

All four deserializers checked `pos + coded_len > size` to validate a
claimed compressed-sub-stream length before reading it. `coded_len` is
untrusted (read directly from the stream) and this addition can
overflow `size_t`: a `coded_len` near `UINT64_MAX` wraps `pos +
coded_len` around to a small value, passing the check even though the
real length is absurd. The fuzzer found this as a real heap-buffer-
overflow *read* inside `RangeDecoder::next_byte()` -- `size_` had been
set to the wrapped (enormous) `coded_len`, so the decoder's own
`pos_ < size_` bounds check was permanently true and it walked off the
end of the real buffer. Fixed by rewriting the check as
`coded_len > (u64)(size - pos)` (subtraction instead of addition --
`pos <= size` is a maintained invariant, so this can't itself overflow)
in all four call sites.

### 4. Out-of-bounds block index from an inconsistent stream (fuzzer-found) -- `src/rod_joint_transform.cpp`, `src/quaternion_joint.cpp`

`deserialize_geo3d_sim` and `deserialize_quat_joint` read their block
count (`nblocks`) as an *independent* field from the record count
(`count`) -- needed for the adaptive-block-size path, where block count
genuinely isn't a fixed function of record count. But
`rod_joint_3d_similarity_inverse`/`quaternion_joint_inverse`'s
*fixed*-block-size code path (`blk = i / kBlockSize`) assumed
`r.block_lag`/`r.block_matrix`/`r.block_delta` were always large enough
to cover every rod/sample, with no cross-check against the independently
supplied `nblocks`. An adversarial stream claiming many records but few
blocks indexed those vectors out of bounds -- a real SEGV, reproduced
directly. Fixed by validating, once, before the reconstruction loop,
that every block index the loop could ever compute (in both the fixed
and adaptive sub-paths) stays within the actual array sizes; throws a
clean exception otherwise. `rod_joint_2d_inverse`'s equivalent block
count is *derived* from record count with no independent field, so it
was never exposed to this specific inconsistency -- checked directly,
not assumed safe by analogy.

### 5. Signed integer overflow in delta accumulation (fuzzer-found) -- `src/rod_joint_transform.cpp`

`rod_joint_2d_inverse`/`rod_joint_3d_similarity_inverse` reconstruct each
point as `points[i+1] = points[i] + residual[i]`, accumulating in plain
`i32` arithmetic. Adversarial (or corrupted) residual values can drive
this running sum past `INT32_MAX`/below `INT32_MIN` -- real signed
overflow, UB, caught directly by UBSan during fuzzing. Fixed by widening
to `i64` for the addition and narrowing back afterward, matching the
pattern this same file already used for its predicted-value computation
two lines above -- not a new technique introduced just for this fix.

### 6. num_levels sanity cap too loose for what it actually bounds (fuzzer-found) -- `src/codec.cpp`

`deserialize_lift`'s `num_levels` field was checked against the same
generic 100-million-element sanity cap used everywhere else in this
file (`check_reasonable_count`). That cap is correct for fields that
bound a single flat array, but `num_levels` instead bounds *four*
separate per-level allocations (`level_pair_counts`, `level_block_counts`,
`block_ratios`, `block_offsets`), each sized directly off it before any
payload-size cross-check is possible -- and a real encoder never emits
more than about 64 levels (each level halves `padded_length`, itself
bounded). The fuzzer found a crafted `num_levels` of roughly 33 million
(comfortably under the 100-million cap) that drove those four
allocations to ~750 MB-792 MB each, aborting the process with an
out-of-memory error. Fixed by replacing the generic cap, for this one
field, with a tight, purpose-specific `kMaxLiftLevels = 64` -- the real
structural bound, not a borrowed one.

### 7. Absolute output-length cap still permits a CPU-time amplification attack (fuzzer-found) -- `include/csa/range_coder.hpp`

Bug #2's `kMaxRangeDecodedBytes` cap (originally 1 GiB) closes the
memory-exhaustion class, but `range_decode_bytes` still does real,
unavoidable work -- one adaptive order-1 model step (a Fenwick-tree
query and update) *per output byte* -- regardless of how small the
compressed input claiming that output length is. The fuzzer found this
directly during a follow-up campaign: several ~60-byte inputs claiming
an output length approaching the 1 GiB cap took 17-51 real CPU-seconds
each to decode, without tripping any memory-bomb defense. This is a
genuine, if modest, algorithmic-complexity finding -- distinct from an
outright memory bomb, since it never crashes or exceeds a byte budget,
just burns CPU disproportionate to the input size. Mitigated (not fully
closed) by lowering `kMaxRangeDecodedBytes` to 128 MiB -- still several
times more headroom than this project's own benchmark suite ever needs
(the largest real dataset, the 693,895-point Autzen LiDAR scan, decodes
to a few tens of MB), while cutting worst-case decode time by ~8x.
Verified: all three of the original slow inputs, replayed against the
fix, now execute in 1-3 ms instead of 17-51 seconds. Fully closing this
class would mean bounding decode work by the *compressed* input size as
well (e.g. a claimed-expansion-ratio check) -- not done in this pass;
see "What fuzzing explicitly does not cover yet" below.

### 8. Independently-decoded sub-streams composed without a length check (fuzzer-found) -- `src/rod_joint_transform.cpp`

`rod_joint_3d_inverse` (the rigid, non-similarity geo3d variant --
distinct from `rod_joint_3d_similarity_inverse`, which already had the
block-index bounds check from bug #4) decodes its XY sub-stream (via
`rod_joint_2d_inverse`) and its Z sub-stream (via
`pantograph_lift_inverse`) independently -- each has its own record-
count field in the wire format -- then zips them together index-by-
index with no check that the two actually agree in length. A real
encoder (`rod_joint_3d_forward`) always produces equal lengths, since
both come from splitting one `points` array, but an adversarial stream
can claim two different lengths; the fuzzer found this as a direct SEGV
reading `zs[i]` past the end of the shorter-than-claimed Z array. Fixed
by checking `xy.size() == zs.size()` once, before the zip, and throwing
a clean exception otherwise. Notably, this exact composition-mismatch
pattern was already recognized and guarded against elsewhere in the
codebase -- `decompress_pose` in `src/codec.cpp` already checks
`positions.size() != orientations.size()` for the same reason -- so
this fix brings `rod_joint_3d_inverse` in line with an existing project
convention rather than introducing a new one.

## `deserialize_packet` (packet_transport.hpp) -- fuzzed, clean

A forensic audit of this repository (`docs/ENGINEERING_AUDIT.md`)
named `packet_transport.cpp`'s `deserialize_packet` -- a second real
untrusted-input parser this codebase gained, for the loss/corruption/
reorder-tolerant transport layer underneath pose streaming -- as the
clearest concrete, not-yet-fuzzed attack surface it found. Added as a
fifth case to the existing harness (`fuzz/fuzz_decompress.cpp`,
selector `% 5` now instead of `% 4`), seeded with one real, correctly-
constructed packet (`serialize_packet`'s exact wire format, hand-built
and cross-checked against Python's `zlib.crc32` to confirm the CRC
matches this project's own implementation before ever fuzzing it). A
5-minute campaign (95,257 executions) found nothing -- a real, honest
"clean" result, not evidence of exhaustive coverage, but a genuine
answer to a previously entirely untested question. Consistent with this
function's design (see its own header comment): every length read from
the wire is cross-checked against the actual received byte count before
any slice/copy touches it, so there was no obvious bug shape for the
fuzzer to have been expected to find here the way there was in
`codec.cpp`'s deserializers before this session's earlier fixes.

## What fuzzing explicitly does not cover yet

- Only the four `decompress_*` entry points and (as of this pass)
  `deserialize_packet` -- the real untrusted-input boundaries for a
  `.csa` file's *library-level* payload and the packet-transport layer.
  Not fuzzed: the CLI's own CSAG-header parsing (`cmd_unsqueeze`/
  `cmd_inspect` in `cli/main.cpp`), the Python bindings' `inspect()`/
  `verify()` header parsing, or the browser worker's JS port of the
  same header logic -- each is a real, separate parser with its own
  untrusted-input surface, not yet given the same treatment.
- The interleaved rANS format (`encode_interleaved_rans`/
  `decode_interleaved_rans`, `src/rans_coder.cpp`) is not in the fuzz
  harness at all yet, despite also parsing untrusted length-prefixed
  fields per lane.
- No corpus minimization/coverage-guided long-running campaign has been
  run in CI -- what's documented above is what ran interactively during
  this development session (see `fuzz/README.md` for exact durations and
  commands), not a continuously-running fuzzing service.
- CI does not yet run any sanitizer build or the fuzzer on a schedule --
  everything above was run manually. Wiring at least the ASan+UBSan test
  suite into a scheduled CI job (mirroring `benchmarks.yml`'s cadence,
  not `ci.yml`'s per-commit one, since sanitizer builds are slow) is real
  future work, not done in this pass.
