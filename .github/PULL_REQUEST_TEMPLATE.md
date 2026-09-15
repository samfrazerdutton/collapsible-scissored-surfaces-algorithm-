## What does this change and why

## Testing

- [ ] `build/csa_tests.exe` and `build/csa_capi_test.exe` (or equivalents)
      pass with 0 failures
- [ ] If a language binding was touched, that binding's own tests pass
- [ ] If the wire format (`CSA1`/`CSAG` magic bytes, anything
      `squeeze`/`unsqueeze` or `docs/index.html`'s JS port depend on) was
      touched, `FORMAT.md` was updated in this PR and the version was
      bumped per its stability promise
- [ ] If this adds or changes a benchmark claim, it's backed by a script
      in `bench/` that regenerates it -- not a one-off number typed into a doc

## Honest notes

Anything this makes worse, anywhere it's untested, or any claim in this
PR that's aspirational rather than measured -- say so here rather than in
a follow-up later.
