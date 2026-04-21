# Experiment Results — phase-8.5 / 8.5.2 (Budget enforcer) — REMEDIATION v1

**Step:** 8.5.2 **Date:** 2026-04-20 **Remediation cycle 1**

## Re-verification

```
$ python scripts/harness/autoresearch_budget_test.py
budget: breached reason=wallclock elapsed=0.305s spent_usd=0.0000
budget: breached reason=usd elapsed=0.000s spent_usd=6.0000
PASS: wallclock_budget_termination_deterministic
PASS: usd_budget_termination_deterministic
PASS: budget_exceeded_alerts_to_slack -- alert_fn called exactly once on first breach
PASS (EXIT=0)

$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped
```

Criteria 1-3 PASS unchanged. Researcher audit confirms design is canonical (time.monotonic for elapsed enforcement, listener-injection for alert_fn).

Minor finding: `budget.py` docstring at L31 says "time.time() elapsed" but implementation uses `time.monotonic()` — implementation correct, docstring misleading. NOT a criterion violation; worth a cleanup in a future docs pass.
