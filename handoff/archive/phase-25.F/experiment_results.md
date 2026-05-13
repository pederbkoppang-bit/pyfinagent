---
step: phase-25.F
cycle: 91
cycle_date: 2026-05-13
result: PASS_PENDING_QA
---

# Experiment Results -- phase-25.F

## What was built/changed

Closed audit bucket 24.4 F-6 by adding two pytest regression tests that
lock the 25.B aliasing-cleanup:

1. **`test_lite_path_byte_identical_flagged`** -- exercises the edge case
   where the RiskJudge `reasoning` is byte-identical to the Trader
   rationale (the historical collision case that the old `is_lite_dup`
   block flagged). Asserts the resulting RiskJudge entry has the
   canonical 4 keys (`agent, role, rationale, weight`) and NO
   `lite_path` field.
2. **`test_full_path_distinct_rationale`** -- exercises the normal
   post-25.A path with distinct RiskJudge reasoning + Trader trader_note.
   Asserts rationale is verbatim, weight is `recommended_position_pct`,
   role is `gate`.

## Files changed

| File | Action |
|------|--------|
| `tests/services/test_signal_attribution.py` | Added 2 regression tests |
| `tests/verify_phase_25_F.py` | NEW verifier (4 claims via pytest -k) |

## Verification command + output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_F.py

=== phase-25.F verification ===

[PASS] 1. test_lite_path_byte_identical_flagged_defined
        -> test name found
[PASS] 2. test_full_path_distinct_rationale_defined
        -> test name found
[PASS] 3. pytest_test_lite_path_byte_identical_flagged_passes
        -> exit=0 matched=True
[PASS] 4. pytest_test_full_path_distinct_rationale_passes
        -> exit=0 matched=True

ALL 4 CLAIMS PASS
```

Direct pytest invocation also confirms (with PYTHONPATH=.):

```
$ PYTHONPATH=. .venv/bin/python -m pytest tests/services/test_signal_attribution.py \
    -k "test_lite_path_byte_identical_flagged or test_full_path_distinct_rationale" -v

collected 22 items / 20 deselected / 2 selected
test_lite_path_byte_identical_flagged PASSED
test_full_path_distinct_rationale PASSED
2 passed in 0.01s
```

## Success criteria -> evidence

1. `pytest_test_lite_path_byte_identical_flagged_passes` -- Claim 3 PASS:
   the test definition is present (claim 1) and pytest invocation reports
   the test PASSED (claim 3).
2. `pytest_test_full_path_distinct_rationale_passes` -- Claim 4 PASS:
   same shape for the distinct-rationale case.

## Out-of-scope / deferred

- Additional aliasing-detection regression tests for cosmetic fields
  beyond `lite_path` (e.g., a future `cosmetic_match` field). The current
  tests assert the canonical 4-key shape, which would catch any new
  cosmetic field by exact-match.
- Frontend Signal-interface regression test: TS strict mode already
  blocks `lite_path` reintroduction at compile time.

## References

- `handoff/archive/phase-25.B/` (the 25.B cleanup these tests lock)
- `tests/services/test_signal_attribution.py:106-167` (new test sites)
- `backend/services/signal_attribution.py:128-141` (post-25.B shape)
