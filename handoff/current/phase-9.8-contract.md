# Sprint Contract — phase-9 / 9.8 (cost budget watcher) — REMEDIATION v1

**Step id:** 9.8 **Remediation cycle:** 1 **Date:** 2026-04-20 **Tier:** simple

## Why remediation

Previous cycle inline-authored. Fresh MAS re-run.

## Research-gate summary

Fresh researcher: `handoff/current/phase-9.8-research-brief.md` — 7 sources in full, 17 URLs, three-variant queries, recency scan (GetOnStack $47K agentic loop incident 2026, MLflow window-aligned reset, per-agent OTel z-score kill switch), gate_passed=true.

Validated: reuse of `BudgetEnforcer` from 8.5.2 + fail-open `alert_fn` + circuit-breaker pattern is Fowler-canonical. Default `daily=$5, monthly=$50` reasonable for research harness.

Carry-forwards (NOT in 9.8 scope):
1. **Monthly idempotency absent** — `IdempotencyKey.daily(JOB_NAME, ...)` means monthly-over alert would fire daily after cap exceeded, not once per month. Consider `IdempotencyKey.monthly` or composite key for monthly scope (MLflow window-aligned reset pattern).
2. Binary trip vs tiered 50/80/100% alerting (industry standard per MLflow + skywork.ai)
3. No per-provider (OpenAI vs Anthropic vs Vertex) or per-job cost attribution — deferred to hardening with OTel-per-agent pattern
4. Reset cadence unspecified — once tripped, no automatic reset path (requires manual intervention for now)

## Immutable criterion

`python -c "import ast; ast.parse(open('backend/slack_bot/jobs/cost_budget_watcher.py').read())" && pytest tests/slack_bot/test_cost_budget_watcher.py -q`
Expected: exit 0, 4/4 pass.

## Plan

1. Re-verify.
2. Capture output.
3. Spawn fresh Q/A.
4. Log and confirm.

## References

- `handoff/current/phase-9.8-research-brief.md`
- `backend/slack_bot/jobs/cost_budget_watcher.py` (64 lines)
- `tests/slack_bot/test_cost_budget_watcher.py` (4 tests)
- `backend/autoresearch/budget.py` (BudgetEnforcer from 8.5.2)
- `.claude/masterplan.json` → phase-9 / 9.8
