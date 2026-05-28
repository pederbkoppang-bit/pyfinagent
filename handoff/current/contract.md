# Contract — cycle 17 / phase-43.0 DoD-2 pytest follow-up

**Cycle:** 17 | **Date:** 2026-05-28 | **Sub-step of:** phase-43.0 (P1, H) | **Author:** Main

---

## Research-Gate Summary

- Researcher subagent: `ad51c00cf2fe7d075`
- Brief: `handoff/current/research_brief_phase_43_0_dod_2_pytest_followup.md`
- `gate_passed: true` — 5 sources read in full (floor 5: unittest.mock docs, OneUptime pytest mocking Feb 2026, pytest-with-eric MagicMock raises, Carpentries Edge Cases, Wikipedia Boundary Testing), 15 URLs, recency scan, 3-variant queries.
- Test skeleton ready-to-write in brief §7 (4 test functions).
- Pattern: `backend/tests/test_phase_43_dod2_window.py` (NOT top-level `tests/`) per existing precedent at `backend/tests/test_dod4_tier1_coverage_investment.py`.

## Hypothesis

Adding the 4-case pytest closes the cycle-16 Q/A NOTE (`financial-logic-without-behavioral-test` heuristic owed a follow-up pytest). Live-BQ smoke output in cycle-16 `experiment_results.md` already proved the helpers work end-to-end against real BQ; this cycle commits a CI-runnable test so future regressions are caught automatically.

## Immutable success criteria

1. `backend/tests/test_phase_43_dod2_window.py` exists with 4 test functions matching the brief §7 skeleton.
2. `pytest backend/tests/test_phase_43_dod2_window.py -v` exits 0 with 4 passed.
3. Tests use `MagicMock()` mocking pattern (no live BQ dependency for CI).
4. No modifications to `backend/services/perf_metrics.py` (this is a pure test cycle).

**Verification commands:**
```bash
source .venv/bin/activate
test -f backend/tests/test_phase_43_dod2_window.py && echo OK
pytest backend/tests/test_phase_43_dod2_window.py -v
git diff --stat backend/services/perf_metrics.py  # expect: empty (no changes)
```

## Plan Steps

1. Write `backend/tests/test_phase_43_dod2_window.py` per brief §7 verbatim.
2. Run pytest. Expect 4/4 passed.
3. Write `experiment_results.md` with verbatim pytest output.
4. Spawn Q/A (tight prompt; this is a small confirming cycle).
5. Append harness_log.
6. Commit + push.

## What this cycle will NOT do

- NOT modify perf_metrics.py.
- NOT add new test infrastructure (no conftest.py, no fixtures, no CI yaml).
- NOT close any DoD (purely follow-up; cycle 16 already shipped the instrument).

## References

- Cycle 17 brief: `handoff/current/research_brief_phase_43_0_dod_2_pytest_followup.md`
- Cycle 16 evidence: `backend/services/perf_metrics.py:118-169` (helper), `:240-349` (extended gap fn)
- Q/A cycle 16 NOTE: `financial-logic-without-behavioral-test` heuristic
- Existing test precedent: `backend/tests/test_dod4_tier1_coverage_investment.py:312-705`
