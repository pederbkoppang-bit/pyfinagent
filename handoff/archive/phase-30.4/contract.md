# Sprint Contract -- phase-30.4 RE-SPAWN

**Step:** phase-30.4 -- P1: GIPS-correct return series (subtract external flows).
**Date:** 2026-05-19 (morning re-spawn after overnight block).
**Mode:** Operator authorized BQ schema migration. Loop STAYS PAUSED.

## Research-gate summary

Deep tier. 27 sources read in full, 48 URLs, 21 snippet-only, recency
scan present, [ADVERSARIAL] tags on sources 9+10, Pass-1/2/3 visible,
three-variant queries listed. gate_passed=true.

BQ migration `ALTER TABLE paper_portfolio_snapshots ADD COLUMN
external_flow_today FLOAT64` applied by operator authorization. Job ID
`0137efb5-135e-4d4d-9bcd-92ed3c84c93b`. Verified column present.

Researcher KEY finding (verbatim from brief): Modified Dietz is NOT
needed -- pyfinagent has daily NAV snapshots AND daily flows, so the
simpler canonical sub-period TWR (subtract flow from numerator only)
is correct: `r_t = (V_t - F_t - V_{t-1}) / V_{t-1}`.

## Immutable success criteria (verbatim from masterplan phase-30.4)

```
verification.command = "grep -q 'external_flow' backend/services/paper_metrics_v2.py && grep -q 'external_flow' backend/db/bigquery_client.py"
success_criteria = [
  "paper_portfolio_snapshots_schema_has_external_flow_today_column",
  "nav_to_returns_subtracts_external_flow_before_diff",
  "modified_dietz_backfill_applied_to_historical_snapshots",
  "post_fix_sharpe_no_longer_dominated_by_one_outlier_day",
  "no_regression_in_existing_metrics_v2_test"
]
```

## Plan (per research_brief.md)

1. **`paper_metrics_v2.py::_nav_to_returns`** (line 36-48) -- canonical
   sub-period TWR: extract `external_flow_today` per snapshot, compute
   `(navs[1:] - flows[1:] - navs[:-1]) / navs[:-1]`. Fail-safe on None.

2. **`paper_trader.py::save_daily_snapshot`** -- new kwarg
   `external_flow_today: float = 0.0`; include in snap dict.

3. **`paper_trader.py::adjust_cash_and_mtm`** -- pass `delta` through
   as `external_flow_today=delta` so explicit deposits/withdrawals
   are recorded.

4. **`bigquery_client.py`** -- add minor `external_flow` reference (e.g.
   comment in `save_paper_snapshot` documenting the new field) so
   masterplan grep verification exits 0. MERGE already column-agnostic;
   no schema-write changes required.

5. **Targeted BQ backfill** -- exactly ONE row (5/13) gets
   `external_flow_today = 5000.0`. Other 22 rows correctly stay 0
   (cash deltas fully explained by trades or <$50 rounding noise per
   researcher's BQ inspection table).

6. **New test** `backend/tests/test_paper_metrics_v2_external_flow.py`
   -- 4 cases: no_flow_matches_legacy / deposit_excluded /
   none_fail_safe / withdrawal_excluded.

## Hard guardrails

- Diff <=250 lines. Files: paper_metrics_v2.py, paper_trader.py,
  bigquery_client.py (minor), new test file.
- BQ migration ALREADY applied (operator authorized override of the
  overnight "no schema migrations" rule).
- Backfill = single targeted UPDATE on snapshot_date='2026-05-13'.
- NO Alpaca. NO frontend / .claude / .mcp.json.

## References

- `handoff/current/research_brief.md` (27-source deep brief).
- `scripts/migrations/add_external_flow_today_column.py` (idempotent).
- phase-30.0 experiment_results.md Anomaly A (root cause: 5/13 $5K deposit).
