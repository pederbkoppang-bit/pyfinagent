---
name: mutate-the-flag-read-not-just-the-guard
description: Dark-launch steps monkeypatch the `_<flag>_enabled()` helper in every test, so the production flag-read executes in ZERO tests; mutate the helper, not just the guard body
metadata:
  type: feedback
---

On any flag-gated (dark-launch) step, mutate the **flag-read helper** itself, not only the
guard body it gates. The house idiom is a function-local
`bool(getattr(get_settings(), "<flag>", False))` helper, and the tests almost always
`monkeypatch.setattr(module, "_<flag>_enabled", lambda: True/False)` — which means the
helper body is executed by **zero** tests. Misspelling the settings attribute or replacing
the body with `return False` keeps the whole suite green, and the operator's eventual flip
would be a silent no-op.

Two mutations that catch it:
- `getattr(..., "<flag>")` -> `getattr(..., "<flag>XX")`
- helper body -> `return False`

A `test_flag_defaults_off_in_settings`-style test does NOT cover this: it reads
`get_settings()` directly and never calls the tool's helper.

Also mutate guard **operand completeness** (`not all(isfinite(v) for v in (a,b,c))` ->
`not isfinite(a)`). Suites that feed a single all-poisoned fixture cannot detect a guard
that checks one of three operands — and that gap is usually the same root as a real
coverage hole in the shipped code.

**Why:** phase-80.27 (2026-07-25) — Main's own 11-mutation matrix was 11/11 killed, but 7
of 10 mutations I authored survived, including all three flag-activation ones. The guard
bodies were genuinely behavioural (removing them went red); the *activation path* and the
*operand set* were the blind spots. The partial-operand mutation surviving pointed straight
at a demonstrable criterion failure (non-finite values leaking into the LLM prompt on
partially-poisoned input).

**How to apply:** whenever a step ships behind `Field(False, ...)` and the tests contain
`monkeypatch.setattr(<module>, "_..._enabled", lambda: ...)`, run these mutations before
issuing a verdict. Positive-control the helper too (stub `get_settings` with the flag True
and confirm the real helper returns True) so you can distinguish a test-coverage gap
(WARN) from a live defect (blocking). See [[mutate-the-library-for-upstream-pins]] and
[[derived-scope-lint-use-xargs]].
