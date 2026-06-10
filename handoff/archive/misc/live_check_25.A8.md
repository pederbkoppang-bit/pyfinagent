# Live-check placeholder — phase-25.A8

**Step:** 25.A8 — Cost-budget HARD-BLOCK in llm_client.py
**Date:** 2026-05-12

## Live-check field
> "Inject tripped=True via test fixture; confirm autonomous cycle skips with Slack alert"

## Pre-deployment evidence
- 12/12 verifier PASS including behavioral round-trip
- BudgetBreachError class module-public
- 3 generate_content call sites patched (Gemini, OpenAI, Claude); abstract method skipped
- autonomous_loop catches via type(e).__name__ (loose coupling, no circular import)
- 60s TTL cache prevents hot-path BQ scans
- COST_BUDGET_HARD_BLOCK_DISABLED env-var escape hatch for tests
- Fail-open: returns False on any BQ/network/config error (never halts trading on broken budget API)

## Post-deployment operator test
1. Temporarily set `COST_BUDGET_DAILY_USD=0.01` in `.env` (or `daily_cap` in cost_budget_api)
2. Run autonomous cycle: `python -c "import asyncio; from backend.services.autonomous_loop import run_daily_cycle; print(asyncio.run(run_daily_cycle()))"`
3. Expected: cycle returns with `status='budget_breach'` + `budget_tripped=True` + error string
4. Restore `.env`; call `reset_cost_budget_cache()` or wait 60s; next cycle runs normally

## Deferred (phase-25.A8.1)
Slack-alert wiring on BudgetBreachError. Requires cross-process pattern (same as 25.K):
- Backend writes `budget_breach` event to `pyfinagent_data.alert_events` BQ table
- Slack scheduler polls + dispatches `send_trading_escalation("P0", "Cost Budget Breached", ...)`

**Audit anchor for next bucket:** 25.2 (backfill missing stops, depends on 25.1 DONE).
