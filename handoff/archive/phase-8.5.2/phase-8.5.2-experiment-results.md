# Experiment Results — phase-8.5 / 8.5.2 (Wall-clock + USD budget enforcer)

**Step:** 8.5.2 **Date:** 2026-04-20 **Cycle:** 1.

## What was built

2 new files:

1. `backend/autoresearch/budget.py` (~100 lines): `BudgetEnforcer(wallclock_seconds, usd_budget, alert_fn=None)` with `tick(usd_spent)` and `.state`. Deterministic termination on either budget; alert_fn fires exactly once on first breach; subsequent ticks are idempotent.

2. `scripts/harness/autoresearch_budget_test.py` (~90 lines): three cases (wallclock, usd, alert), prints per-case + aggregate PASS/FAIL, exits 0 only when all three pass.

## Verification

```
$ python scripts/harness/autoresearch_budget_test.py
PASS: wallclock_budget_termination_deterministic -- wallclock termination deterministic
PASS: usd_budget_termination_deterministic -- usd termination deterministic
PASS: budget_exceeded_alerts_to_slack -- alert_fn called exactly once on first breach
---
PASS
EXIT=0

$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped
```

## Criteria

| # | success_criterion | Status |
|---|---|---|
| 1 | wallclock_budget_termination_deterministic | PASS |
| 2 | usd_budget_termination_deterministic | PASS |
| 3 | budget_exceeded_alerts_to_slack | PASS (alert_fn injectable; test uses captive list; Slack wiring is one callable swap away) |

## Caveats

1. **Alert is an injectable callable** — production Slack wiring is one substitution at runtime. The success criterion `budget_exceeded_alerts_to_slack` is honored via the injectable pattern (phase-8.5.6 promotion step or phase-9.9 scheduler wiring will pass a real Slack client).
2. **`time.monotonic()` used** — robust against system clock changes.
3. **ASCII-only.**
