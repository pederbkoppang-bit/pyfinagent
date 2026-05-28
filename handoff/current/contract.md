# Contract — cycle 19 / phase-43.0 DoD-5 pytest follow-up

**Cycle:** 19 | **Date:** 2026-05-28 | **Sub-step of:** phase-43.0 (P1, H) | **Author:** Main

---

## Research-Gate Summary

- Researcher: `aca2fe1722a1259eb`
- Brief: `handoff/current/research_brief_phase_43_0_dod_5_pytest.md`
- `gate_passed: true` — 6 sources in full, 19 URLs, recency scan, 3-variant queries.
- Skeleton ready: 2 `@pytest.mark.parametrize` test functions × 2 cases each = 4 logical cases.

## Hypothesis

Adding the pytest suite covering the cycle-14 `_STRING_DATE_TIMESTAMP_COLS` type-branch (in `_bq_max_event_age`) provides CI-runnable regression coverage equivalent to cycle 17's pytest for DoD-2. Defends against future SAFE.TIMESTAMP regression.

## Immutable success criteria

1. `backend/tests/test_phase_43_dod5_freshness.py` exists with 2 parametrized test functions (4 logical cases total).
2. `pytest backend/tests/test_phase_43_dod5_freshness.py -v` exits 0.
3. Tests use MagicMock pattern matching the canonical chain mock at `test_dod4_tier1_coverage_investment.py:417-424`.
4. NO modification to `backend/services/cycle_health.py`.

## Plan Steps

1. Write `backend/tests/test_phase_43_dod5_freshness.py` per brief skeleton.
2. Run pytest.
3. Write experiment_results.md.
4. Spawn tight Q/A.
5. Append harness_log.
6. Commit + push.

## What this cycle will NOT do

- NOT modify cycle_health.py.
- NOT add conftest.py or shared fixtures.
- NOT close a DoD (pure quality improvement).

## References

- Cycle 19 brief: `handoff/current/research_brief_phase_43_0_dod_5_pytest.md`
- Cycle 14 fix: `backend/services/cycle_health.py:414-462`
- Test pattern reference: `backend/tests/test_dod4_tier1_coverage_investment.py:417-424`
- Mirror pattern: `backend/tests/test_phase_43_dod2_window.py` (cycle 17)
