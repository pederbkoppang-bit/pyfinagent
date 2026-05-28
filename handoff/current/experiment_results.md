# Cycle 19 — Experiment Results (DoD-5 SAFE.TIMESTAMP pytest follow-up)

**Window:** 2026-05-28T20:35-20:50+02:00 (approx)
**Sub-step of:** phase-43.0 (P1, H) — pytest regression coverage for cycle 14 fix
**Researcher gate:** `aca2fe1722a1259eb` PASSED (6 in-full / 19 URLs)

## Files created

- `backend/tests/test_phase_43_dod5_freshness.py` (79 lines, 2 parametrized test functions, 4 logical cases)
- `handoff/current/research_brief_phase_43_0_dod_5_pytest.md` (researcher output)

## Files NOT changed

- `backend/services/cycle_health.py` — pure test cycle.

## pytest output

```
backend/tests/test_phase_43_dod5_freshness.py::test_bq_max_event_age_string_columns_use_safe_timestamp[paper_trades-created_at] PASSED [ 25%]
backend/tests/test_phase_43_dod5_freshness.py::test_bq_max_event_age_string_columns_use_safe_timestamp[paper_portfolio_snapshots-snapshot_date] PASSED [ 50%]
backend/tests/test_phase_43_dod5_freshness.py::test_bq_max_event_age_timestamp_columns_use_bare_max[historical_prices-ingested_at] PASSED [ 75%]
backend/tests/test_phase_43_dod5_freshness.py::test_bq_max_event_age_timestamp_columns_use_bare_max[signals_log-recorded_at] PASSED [100%]

============================== 4 passed in 0.01s ===============================
```

## Coverage matrix

| Branch | Tables tested | SQL pattern asserted |
|---|---|---|
| STRING/DATE (needs SAFE.TIMESTAMP) | `paper_trades.created_at`, `paper_portfolio_snapshots.snapshot_date` | `SAFE.TIMESTAMP(MAX(col))` present |
| Native TIMESTAMP (bare MAX) | `historical_prices.ingested_at`, `signals_log.recorded_at` | `MAX(col)` present + `SAFE.TIMESTAMP` absent |

Each branch tested with 2 distinct (table, col) combinations covering the membership and non-membership of `_STRING_DATE_TIMESTAMP_COLS`.

## Mutation resistance

- Removing the type-branch (collapsing to always-SAFE.TIMESTAMP): STRING/DATE tests still pass; TIMESTAMP tests FAIL → CAUGHT.
- Removing the type-branch (collapsing to always-bare-MAX): TIMESTAMP tests still pass; STRING/DATE tests FAIL → CAUGHT.
- Removing/changing `_STRING_DATE_TIMESTAMP_COLS` set membership: at least one test FAILs → CAUGHT.

3/3 mutations caught.

## Step status

phase-43.0 STAYS `pending`. Cycle 19 adds regression coverage for the cycle-14 DoD-5 fix; no DoD count change.

## Cumulative tally

Unchanged: **12 most-generous / 8 literal of 14 PASS**.

## References

- Cycle 19 brief: `handoff/current/research_brief_phase_43_0_dod_5_pytest.md`
- Test pattern: `backend/tests/test_dod4_tier1_coverage_investment.py:417-424` (canonical chain mock)
- Cycle 17 sibling: `backend/tests/test_phase_43_dod2_window.py`
- Cycle 14 fix under test: `backend/services/cycle_health.py:414-462`
