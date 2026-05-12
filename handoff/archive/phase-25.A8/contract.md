# Sprint Contract — phase-25.A8 — Cost-budget HARD-BLOCK

**Cycle:** phase-25 cycle 6
**Date:** 2026-05-12
**Step ID:** 25.A8
**Priority:** P0
**Depends on:** 25.A9 (DONE) — needs accurate cost data for budget check to be meaningful

## Research-gate
Reuses phase-24.8 cycle 8 + phase-24.13 cycle 14 researcher gates. Audit basis: F-4 (llm_client never checks cost_budget.tripped).

## Hypothesis
Adding `BudgetBreachError` + `_check_cost_budget()` at top of every `generate_content` will hard-block LLM API calls when daily/monthly BQ spend exceeds caps.

## Success criteria (verbatim)
1. llm_client_raises_budget_breach_error_when_tripped_true
2. autonomous_loop_catches_budget_breach_skips_cycle_emits_slack
3. manual_reset_via_post_cost_budget_reset_clears_block

## Plan
1. Add `BudgetBreachError` exception + `_check_cost_budget()` helper + `reset_cost_budget_cache()` in `backend/agents/llm_client.py` near top
2. Add `_check_cost_budget()` call at top of each of the 3 concrete `generate_content` methods (Gemini, OpenAI, Claude — line ~358 is abstract method skipped)
3. Use TTL cache (60s) to avoid hot-path BQ scans
4. Add `COST_BUDGET_HARD_BLOCK_DISABLED` env-var escape hatch for tests
5. `autonomous_loop.py` catches by name (loose coupling) → sets `status='budget_breach'`
6. Verifier `tests/verify_phase_25_A8.py` (12 claims incl. behavioral round-trip)
7. experiment_results.md, Q/A, harness_log Cycle 62, flip 25.A8

## References
- `docs/audits/phase-24-2026-05-12/24.8-observability-findings.md` F-4
- `docs/audits/phase-24-2026-05-12/24.13-redline-synthesis-findings.md` F-4
- `backend/agents/llm_client.py:42-150` (added budget block)
- `backend/agents/llm_client.py:526,706,838` (generate_content patch sites)
- `backend/services/autonomous_loop.py:540-560` (BudgetBreachError catch)
- `backend/api/cost_budget_api.py` (tripped flag computation — unchanged)
