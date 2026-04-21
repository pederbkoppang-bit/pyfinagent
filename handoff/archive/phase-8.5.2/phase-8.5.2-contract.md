# Sprint Contract — phase-8.5 / 8.5.2 (Wall-clock + USD budget enforcer)

**Step id:** 8.5.2 **Cycle:** 1 **Date:** 2026-04-20 **Tier:** simple

## Research-gate summary
Closure brief. `gate_passed: true`. Brief at `handoff/current/phase-8.5.2-research-brief.md`.

## Hypothesis
Write `backend/autoresearch/budget.py` with `BudgetEnforcer` + `scripts/harness/autoresearch_budget_test.py` that exercises wall-clock termination, USD termination, and an injectable slack-alert callable. Test exits 0 (prints "PASS") iff all three success criteria hold.

## Immutable criterion
- `python scripts/harness/autoresearch_budget_test.py` exits 0.

## Plan
1. `backend/autoresearch/budget.py` (~80 lines): `BudgetEnforcer(wallclock_seconds, usd_budget, alert_fn=None)` + `.tick(usd_spent)` + `.state` property.
2. `scripts/harness/autoresearch_budget_test.py` (~120 lines): 3 in-script test cases + "PASS"/"FAIL" stdout + exit code.
3. Verify + regression + Q/A + log + flip.

## Out of scope
- No BQ logging.
- No real Slack send (alert_fn is injectable; test passes a captive list).
- ASCII-only.
