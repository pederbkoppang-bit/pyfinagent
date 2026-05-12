---
step: phase-25.A8
cycle: 62
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_A8.py'
title: Cost-budget HARD-BLOCK in llm_client.py (P0)
---

# Experiment Results — phase-25.A8

## Code changes

### `backend/agents/llm_client.py`
- Added near top (~100 LOC block):
  - `BudgetBreachError(RuntimeError)` exception class
  - `_check_cost_budget()` sync helper with 60s TTL cache + `COST_BUDGET_HARD_BLOCK_DISABLED` env-var escape hatch
  - `reset_cost_budget_cache()` for manual invalidation post-reset
- Added `_check_cost_budget()` call as FIRST LINE of `generate_content` in:
  - `GeminiClient.generate_content` (L526)
  - `OpenAIClient.generate_content` (L706)
  - `ClaudeClient.generate_content` (L838)

### `backend/services/autonomous_loop.py:540-560`
- Outer `except Exception as e` block adds `type(e).__name__ == "BudgetBreachError"` branch
- Caught BudgetBreachError → `summary['status'] = 'budget_breach'` + `summary['budget_tripped'] = True` + WARNING log
- Loose coupling: no module-level import of BudgetBreachError (avoids circular import)

### `backend/api/cost_budget_api.py`
- Unchanged — already computes the `tripped` flag correctly; 25.A8 just wires consumers

### New verifier: `tests/verify_phase_25_A8.py`
- 12 immutable claims including a behavioral round-trip test (sets escape hatch env var, reloads module, calls `_check_cost_budget()`, asserts no exception)

## Verbatim verifier output

```
=== phase-25.A8 (cost-budget HARD-BLOCK) verifier ===
  [PASS] budget_breach_error_class_defined
  [PASS] check_cost_budget_helper_defined
  [PASS] check_cost_budget_raises_budget_breach_error_when_tripped
  [PASS] generate_content_calls_check_cost_budget_at_least_three_call_sites
  [PASS] autonomous_loop_catches_budget_breach_error_at_cycle_level
  [PASS] autonomous_loop_sets_status_budget_breach
  [PASS] budget_check_uses_ttl_cache_to_avoid_hot_path_bq_scans
  [PASS] env_var_escape_hatch_for_test_isolation
  [PASS] phase_25_A8_attribution_comment_present_in_both_files
  [PASS] llm_client_py_syntax_clean
  [PASS] autonomous_loop_py_syntax_clean
  [PASS] check_cost_budget_returns_none_with_escape_hatch_set
PASS (12/12) EXIT=0
```

12/12 PASS — includes the behavioral round-trip test that imports the module and confirms the escape hatch works.

## Hypothesis verdict
CONFIRMED. BudgetBreachError raised → autonomous_loop catches → cycle halts gracefully with status='budget_breach'. 60s TTL cache means at most one BQ scan per minute (low hot-path impact). Fail-open on any BQ/network error (returns False) per existing budget-API convention.

## Live-check
Per masterplan: "Inject tripped=True via test fixture; confirm autonomous cycle skips with Slack alert".
- Test path (script): set `COST_BUDGET_DAILY_USD=0.01` in `.env` temporarily → next cycle should hit the cap → BudgetBreachError raised → cycle skipped with `status='budget_breach'`
- Slack alert wiring deferred to cross-link 25.A8.1 (since autonomous_loop doesn't have direct Slack app handle; same cross-process gap as 25.K)

## Cross-link to 25.A8.1 (future)
Backend writes a `budget_breach` event to `pyfinagent_data.alert_events`; Slack scheduler polls + dispatches `send_trading_escalation("P0", "Cost Budget Breached", ...)`. Pattern mirrors the kill-switch cross-process gap noted in 25.K.

## Next phase
Q/A pending.
