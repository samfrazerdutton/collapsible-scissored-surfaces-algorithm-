# Fuzzing `decompress()` / `decompress_geo2d()` / `decompress_geo3d()` / `decompress_pose()`

This is a real libFuzzer harness (`fuzz_decompress.cpp`), not a stub --
it found 6 genuine bugs in this codec's untrusted-input decode path
(decompression bomb, out-of-bounds indexing, integer-overflow-bypassing
a bounds check, signed-overflow UB, an over-loose sanity cap, a
CPU-time amplification path, and a missing sub-stream length check),
each fixed and verified. See `docs/SANITIZERS.md` for the technical
writeup of every finding; this file is the "how to actually run it"
reference.

## Prerequisites

clang with libFuzzer support. Everything here was run inside WSL2
Ubuntu using its default `clang++` (18.1.3) -- MSVC has no libFuzzer
support, so this only runs under WSL/Linux, not natively on Windows.

## Building

```bash
bash fuzz/build_fuzzer.sh
# -> /tmp/fuzz_decompress
```

This compiles the fuzzer-instrumented harness plus every real core
source file (plain ASan/UBSan-instrumented, no `-fsanitize=fuzzer` on
those) separately, then links them -- see the script for the exact file
list. `src/simd_avx2.cpp` gets `-mavx2` like the normal CMake build
does, since it's the one translation unit that actually uses AVX2
intrinsics.

## Seed corpus

A seed corpus was built from real `scissorc squeeze` output (stripping
the CLI's CSAG wrapper header down to the raw Layer-1 CSA1 blob, then
prepending a selector byte 0-3 to route it to the matching
`decompress_*` function): one seed each for a real geo2d, geo3d, pose,
and general/lossless squeeze, in `/tmp/fuzz_corpus/` (not committed --
regenerate it from `scissorc squeeze` output plus a prepended selector
byte if you want to reproduce this from scratch). Every crash/OOM input
this harness ever found was copied back into that corpus before the
next run, so later campaigns kept exercising every previously-found
edge case as a mutation seed, not just the four original hand-built
seeds.

## Running

```bash
cd /tmp
setarch $(uname -m) -R ./fuzz_decompress -max_total_time=<seconds> -rss_limit_mb=2048 fuzz_corpus
```

(`setarch ... -R` disables ASLR -- see docs/SANITIZERS.md's TSan note;
it isn't strictly required for the ASan+UBSan fuzzer build the way it
is for TSan, but was used consistently here anyway.)

Replaying one specific saved input (a crash, an OOM repro, or a slow
unit) instead of fuzzing:

```bash
./fuzz_decompress <path-to-saved-input>
```

## What was actually run, and what it found

| Campaign | Duration | Result |
|---|---|---|
| Initial run | first few seconds | bug #2 (decompression bomb in `deserialize_geo2d`) |
| Interactive follow-ups | seconds-to-minutes each | bugs #1 (UBSan, found via the plain sanitizer test-suite run, not this harness), #3, #4, #5 -- each found within the same session as the fix before it, corpus reseeded with every crash file each time |
| 15-minute campaign, full corpus incl. all then-known crash reproducers | 900s | 1 out-of-memory abort (bug #6: `num_levels` sanity cap too loose) plus 3 "slow unit" inputs (17s, 38s, 51s each -- not crashes, but became bug #7 once investigated: the same absolute output-length cap that prevents memory bombs still let those ~60-byte inputs burn double-digit seconds of real CPU time) |
| 2-minute campaign, after bugs #6/#7 fixes | 120s | 1 new SEGV (bug #8: `rod_joint_3d_inverse` composing two independently-decoded sub-streams with no length check) |
| 5-minute campaign, after bug #8 fix | 300s | nothing further; corpus plateaued at 26/27 coverage/feature counts, `DONE` |
| 5-minute campaign, `deserialize_packet` added as a 5th selector case (harness now `% 5`, seeded with one real, hand-built, CRC-cross-checked-against-Python's-`zlib.crc32` valid packet) | 300s | nothing found (95,257 executions); a real, honest clean result on a previously entirely-untested parser, not evidence of exhaustive coverage |

Every crash/OOM/slow-unit file the fuzzer wrote out during these runs
was: (1) hex-dumped and reasoned about to find the real root cause --
never patched by guessing, (2) fixed, (3) replayed against the fixed
binary to confirm it now behaves correctly (rejected quickly with a
clean exception, for the correctness bugs; now fast instead of slow,
for the CPU-amplification finding), and (4) the entire non-adversarial
test suite (`csa_tests`, `csa_capi_test`) was re-run on both native
Windows/MSVC and WSL/GCC to confirm zero regressions to legitimate
functionality, before moving to the next campaign.

## Known limitations of this harness (see also docs/SANITIZERS.md)

- Only the four library-level `decompress_*` entry points are fuzzed.
  The CLI's CSAG-header parsing, the Python bindings' header parsing,
  and the browser worker's JS port of the same header logic are each a
  separate, real parser with their own untrusted-input surface, not
  covered here.
- The interleaved rANS format's own length-prefixed per-lane fields are
  not fuzzed at all yet.
- `-max_len` was never set, so libFuzzer capped generated inputs at
  4096 bytes across every campaign above -- larger malformed inputs are
  untested territory.
- No coverage-guided campaign here has run for more than 15 minutes at
  a stretch, and none has run in CI. This is what was actually run
  during interactive development, not a continuously-running fuzzing
  service.
