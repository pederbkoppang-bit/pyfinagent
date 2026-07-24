# live_check_75.4.2 — skill_optimizer post-write delivery invariant

All output verbatim from live runs 2026-07-24. Offline-only step (deterministic
invariant, $0, no UI surface).

## 1. Verification command (immutable) — exit 0

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_4_2_optimizer_invariant.py -q
....                                                                     [100%]
4 passed
$ .venv/bin/python -m pytest backend/tests/test_phase_75_4_2_optimizer_invariant.py backend/tests/test_phase_75_skill_delivery.py -q
31 passed, 1 warning in 2.69s     (the 75.4 delivery suite stays green — regression proof)
```

## 2. Fail-CLOSED evidence (rejected write leaves the file byte-identical)

T1 (heading promotion) and T2 (placeholder drop) both compute
`sha256(read_bytes())` BEFORE the apply_modification call and AFTER the refusal
and assert equality (test file :103-113 / :129-142); T2 additionally asserts
`read_bytes()` equality directly. Both pass — and under mutations M1/M2 both
FAIL, so the byte-identity assertion is load-bearing, not decorative.

Also proven: the fixture trap — the bare `{{quant_model_data}}` occurs TWICE in
quant_model_agent.md, so every fixture asserts `content.count(old_text) == 1`
before use; mutation M5 (degrading T2's unique 2-line old_text to the bare
placeholder) is KILLED by that assert.

## 3. Mutation matrix — 5 mutations, 5 killed, 0 survivors

Runner `run_mutations_75_4_2.py`, verbatim log `mutation_matrix_75_4_2.txt`
(scratchpad). Summary verbatim:

```
SUMMARY: 5 mutations, 5 killed, survivors: NONE
=== post-restore sanity: pytest exit 0 ===
```

| # | Mutation | Killed by |
|---|---|---|
| M1 | invariant call removed (back to load-succeeds-only) — criterion-4 required | T1 + T2 + T4 (3 failed) |
| M2 | helper weakened to always-ok — criterion-4 required | T1 + T2 + T4 (3 failed) |
| M3 | placeholder-subset guard dropped | T2 only (T1 survives on the length guard — each guard independently load-bearing) |
| M4 | length-retention guard dropped | T4 only |
| M5 | **FIXTURE**: T2's unique old_text degraded to the bare twice-occurring placeholder | T2's own `count==1` trap-assert |

## 4. git diff --stat (change surface: exactly the contract's 2 files)

```
 backend/agents/skill_optimizer.py                        (+import re, +consts, +_delivery_invariant_ok, apply_modification baseline+postcondition)
 backend/tests/test_phase_75_4_2_optimizer_invariant.py   (new, 4 tests)
```

prompts.py / skills/*.md / the 71.4 flag-gated review: untouched (absent from diff).

## 5. Lint

New test file: `All checks passed!` (2 PLR0402 import-alias findings auto-fixed).
skill_optimizer.py finding-class census vs `git show HEAD:` baseline: the ONLY
delta is +1 BLE001 (the new pre-write baseline `except Exception` — the same
documented fail-closed-skip pattern as the file's 6 existing instances; an
executor-introduced ISC004 was fixed by Main during review).

## 6. Execution-model note

GENERATE was delegated to a Sonnet executor (sonnet-tagged step, operator tiering
directive); Main independently re-ran both suites, reviewed the full diff, fixed
the two lint nits, and ran the mutation matrix itself. The invariant is
UNCONDITIONAL and checked BEFORE the git commit; revert is the byte-exact
in-function write-back (never revert_modification's git checkout).
