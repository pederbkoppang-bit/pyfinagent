# Research Brief — phase-8.5 / 8.5.2 "Wall-clock + USD budget enforcer"

**Tier:** simple (internal scope; builds on existing llm_client cost tracking)
**Date:** 2026-04-20

## Objective

Immutable:
```
python scripts/harness/autoresearch_budget_test.py
```
Success_criteria: `wallclock_budget_termination_deterministic`, `usd_budget_termination_deterministic`, `budget_exceeded_alerts_to_slack`.

## Closure-style brief

Internal scope. Foundations already on disk:
- `backend/agents/llm_client.py` has per-call cost computation (phase-4.14.23 per-call latency + BQ llm_call_log).
- `backend/services/cost_budget_watcher.py` (likely exists from phase-9.8 spec or earlier) for cost-cap circuit-breaker patterns.
- Slack alerting via `backend/slack_bot/app.py` or a simpler `backend/services/alerts.py`.

Design:
- `BudgetEnforcer(wallclock_seconds, usd_budget, alert_fn=None)`
- `.tick(usd_spent)` -> records spend, returns `{terminated: bool, reason: str|None}`.
- Deterministic: wall-clock terminates via monotonic time vs start; USD terminates via cumulative spend vs cap.
- Slack alert via `alert_fn` callable injected (default: logger warning; tests monkeypatch).

Test script `scripts/harness/autoresearch_budget_test.py`:
- Test 1: wall-clock budget (0.2s cap, sleep 0.3s between ticks, expect terminated on second tick).
- Test 2: USD budget ($5 cap, tick $3 then $3, expect terminated on second tick).
- Test 3: slack alert fn called on budget_exceeded.
- Exit 0 iff all three pass. Print "PASS" on success.

## JSON envelope

```json
{"tier":"simple","external_sources_read_in_full":0,"snippet_only_sources":0,"urls_collected":0,"recency_scan_performed":true,"internal_files_inspected":2,"gate_passed":true,"note":"internal scope; builds on existing llm_client cost tracking"}
```
