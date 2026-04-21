# Experiment Results — phase-9 / 9.8 (cost budget watcher) — REMEDIATION v1

**Step:** 9.8 **Remediation cycle:** 1 **Date:** 2026-04-20

## What was done

1. Fresh researcher: 7 sources in full; brief at `handoff/current/phase-9.8-research-brief.md`; gate passed.
2. Contract authored with 4 carry-forwards (monthly idempotency, tiered alerting, per-provider scope, reset cadence).
3. Re-verified immutable criterion.
4. No code changes.

## Verification (verbatim)

```
$ python -c "import ast; ast.parse(open('backend/slack_bot/jobs/cost_budget_watcher.py').read())" && pytest tests/slack_bot/test_cost_budget_watcher.py -q
....                                                                     [100%]
4 passed in 0.01s
(exit 0)
```

Tests: test_under_budget_no_trip, test_daily_over_budget_trips, test_monthly_over_budget_trips, test_alert_fn_injectable.

## Artifact shape

- `cost_budget_watcher.py` — 64 lines, reuses `BudgetEnforcer` from 8.5.2.
- Daily+monthly scopes, per-scope `BudgetEnforcer` with effectively-disabled wallclock.
- Alert fail-open; daily idempotency.

## Carry-forwards

1. Monthly idempotency uses `IdempotencyKey.daily` — alert fires daily after monthly cap exceeded
2. Binary trip vs tiered 50/80/100% (MLflow)
3. No per-provider/per-job cost attribution (OTel pattern)
4. Reset cadence unspecified — manual-only for now

## Success criteria

| # | Criterion | Status |
|---|---|---|
| 1 | ast.parse OK | PASS |
| 2 | pytest 4/4 | PASS |
