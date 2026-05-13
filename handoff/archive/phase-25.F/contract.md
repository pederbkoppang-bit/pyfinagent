---
step: 25.F
slug: byte-identical-aliasing-regression-tests
status: in_progress
cycle_date: 2026-05-13
parent_research_brief: handoff/current/research_brief.md
---

# Contract -- phase-25.F

## Step ID + masterplan reference

`25.F` -- "Byte-identical regression test for aliasing detection"
(P2, harness_required, no dep).

## Research-gate summary

Tier=simple. Brief at `handoff/current/research_brief.md`,
`gate_passed=true`.

## Hypothesis

Cycle 85 (25.B) removed the `is_lite_dup` aliasing-detection block
from `signal_attribution.py`. Without a regression test, a future
PR could silently re-introduce the dead-code branch (or a similar
`lite_path` field on the RiskJudge entry). Adding two pytest cases
that lock the post-25.B shape prevents this.

## Success criteria (verbatim from masterplan.json)

> `pytest_test_lite_path_byte_identical_flagged_passes`
>
> `pytest_test_full_path_distinct_rationale_passes`

## Plan steps

1. **Add `test_lite_path_byte_identical_flagged`** to
   `tests/services/test_signal_attribution.py`:
   - Build an analysis dict where the RiskJudge `reasoning` is byte-identical
     to what the Trader rationale would be.
   - Call `extract_signals_from_analysis`.
   - Assert the RiskJudge entry IS present.
   - Assert it has NO `lite_path` key.
   - Assert it has only the 4 canonical keys `{agent, role, rationale, weight}`.
2. **Add `test_full_path_distinct_rationale`** to the same file:
   - Build an analysis dict with distinct RiskJudge `reasoning` and
     Trader `trader_note`.
   - Call `extract_signals_from_analysis`.
   - Assert the RiskJudge rationale is the verbatim `reasoning` value
     (no patching, no fallback to `Decision:`).
   - Assert weight comes from `recommended_position_pct`.
3. **Create `tests/verify_phase_25_F.py`** that runs:
   ```
   pytest tests/services/test_signal_attribution.py \
     -k "test_lite_path_byte_identical_flagged or test_full_path_distinct_rationale" \
     -v --no-header
   ```
   and asserts exit code 0 + both test names appear in PASSED output.

## Files

| File | Action |
|------|--------|
| `tests/services/test_signal_attribution.py` | Add 2 regression tests |
| `tests/verify_phase_25_F.py` | NEW verifier (pytest -k filter) |

## Verification command (immutable)

```
source .venv/bin/activate && python3 tests/verify_phase_25_F.py
```

## Live-check

`pytest tests/test_signal_attribution.py passes both regression tests`.
Will write `handoff/current/live_check_25.F.md`.

## Risks + mitigations

- **Risk**: test naming collision with existing tests.
  **Mitigation**: Both names are unique; current tests don't use
  "byte_identical" or "distinct_rationale" in any test name.
- **Risk**: pytest subprocess might fail to locate venv.
  **Mitigation**: verifier invokes pytest via `subprocess.run` with the
  full venv interpreter path.

## References

- `handoff/current/research_brief.md`
- `handoff/archive/phase-25.B/contract.md` (the 25.B cleanup these tests lock)
- `backend/services/signal_attribution.py:128-141` (post-25.B shape)
- `.claude/masterplan.json::25.F`
