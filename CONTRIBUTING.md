# Contributing to CSA

Thanks for looking at this. A few things that will make a contribution
land faster, and a few things this project actually cares about that
aren't the usual boilerplate.

## The one rule that matters most: honesty over a good-looking number

This project's whole credibility rests on `REAL_POSE_BENCHMARK.md`,
`REAL_GEO_BENCHMARK.md`, and `ADVERSARIAL_BENCHMARK.md` being real,
reproducible, and including the cases where CSA loses. If you're adding a
benchmark, a comparison, or a claim anywhere in the docs:

- **Run it, don't estimate it.** Every number in this repo's docs is
  regenerable from a script in `bench/`. If you add a claim, add (or
  extend) the script that produces it.
- **Report losses as prominently as wins.** If your change makes something
  worse on some real dataset, say so in the same commit, not in a follow-up
  someday.
- **Real data over synthetic where it exists.** Synthetic test cases are
  fine for unit tests and CI; benchmark *claims* in the `REAL_*`/
  `ADVERSARIAL_*` docs need real, sourced, publicly-fetchable data (or an
  honestly-labeled synthetic fallback if the real data can't be
  redistributed).

## Before opening a PR

1. **Build and run the full test suite.** `cmake --build build` then run
   `build/csa_tests.exe` and `build/csa_capi_test.exe` (or the equivalent
   without `.exe` on Linux/macOS). Both should report `0 failures`. If
   you touched a language binding, run that binding's own tests too
   (`bindings/python/test_bindings.py`, `cargo test` in `bindings/rust/`,
   the C#/Go test projects).
2. **If you touched the wire format** (`csa_capi.h`, the `CSA1`/`CSAG`
   magic bytes, anything `cli/main.cpp`'s `squeeze`/`unsqueeze` or
   `docs/index.html`'s JS port of the same logic depends on), read
   `FORMAT.md` first and update it in the same PR. Format changes need a
   version bump per that doc's stability promise -- see below.
3. **If you're adding a new transform or codec mode**, it needs a
   correctness test that round-trips real or realistic data before any
   ratio/performance claim about it goes in a doc -- this project's
   established pattern (see `DESIGN.md`'s many "verified before being
   trusted" sections) is to test correctness first, measure second, and
   never assume a novel algorithm is bit-exact without checking.
4. Keep commits focused. A bug fix doesn't need a drive-by refactor of
   nearby code; a new feature doesn't need to also reorganize the file it
   lives in.

## Code style

- C++17, matching the existing style in `src/`/`include/csa/` -- comments
  explain *why*, not *what* (the code should be readable enough that "what"
  doesn't need narrating).
- No new abstractions or config flags for hypothetical future needs. Three
  similar lines beat a premature abstraction.
- Python/Rust/C#/Go bindings should stay thin wrappers over the C ABI
  (`include/csa/csa_capi.h`) -- no reimplemented logic that could drift
  from the C++ core. If you find yourself reimplementing an algorithm in a
  binding, that's a sign the C ABI is missing an export, not a reason to
  duplicate the logic.

## Reporting a bug

Please include: what you ran, what you expected, what actually happened,
and (if it's a correctness issue) the smallest input that reproduces it.
If it's a benchmark/ratio discrepancy, include which script you ran and
its full output -- see the bug report issue template.

## Code of conduct

This project follows the Contributor Covenant -- see `CODE_OF_CONDUCT.md`.

## License

By contributing, you agree your contribution is licensed under this
project's Apache License 2.0 (see `LICENSE`).
