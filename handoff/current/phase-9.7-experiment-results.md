# Experiment Results — phase-9 / 9.7 (weekly data integrity) — REMEDIATION v1

**Step:** 9.7 **Remediation cycle:** 1 **Date:** 2026-04-20

## What was done

1. Fresh researcher: 6 sources in full; brief authored; gate passed.
2. Contract authored with cross-phase carry to 9.9 (scheduler-wiring bug).
3. Re-verified immutable criterion.
4. No code changes.

## Verification (verbatim)

```
$ python -c "import ast; ast.parse(open('backend/slack_bot/jobs/weekly_data_integrity.py').read())" && pytest tests/slack_bot/test_weekly_data_integrity.py -q
...                                                                      [100%]
3 passed in 0.01s
(exit 0)
```

Tests: test_drift_above_threshold_alerts, test_drift_below_threshold_no_alert, test_missing_prior_baseline_skipped.

## Artifact shape

- `weekly_data_integrity.py` — 57 lines, `run()` with DI (`current_counts`, `prior_counts`, `alert_fn`, `store`, `iso_year_week`, `drift_threshold`).
- `DRIFT_THRESHOLD = 0.20`; weekly idempotency; alert-fn fail-open.
- `_compute_drifts()` pure function: `abs(cur_n - prev_n) / prev_n > threshold`.

## Carry-forwards

1. **Cross-phase to 9.9:** `scheduler.py:374` lacks `current_counts`/`prior_counts` args → empty-dict run. Must fix in phase-9.9 remediation.
2. Direction-blind drift (drop ≠ growth severity)
3. Single-week baseline → rolling 4-week median (Acceldata 2025)
4. Schema/null-rate/freshness-SLA checks deferred

## Success criteria

| # | Criterion | Status |
|---|---|---|
| 1 | ast.parse OK | PASS |
| 2 | pytest 3/3 | PASS |
