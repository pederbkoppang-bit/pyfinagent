---
name: strategy-rotation-infra
description: Strategy-rotation/promotion machinery already largely EXISTS (phase-25 + optimizer categorical); phase-47.6 only adds the per-strategy DSR selector. Anchors to avoid rebuilding.
metadata:
  type: project
---

The dynamic-strategy-rotation North Star ("shift capital to whichever strategy is making
the most money") is mostly already BUILT. Do not let a future cycle rebuild it.

**Already exists (file:line):**
- `backend/backtest/quant_optimizer.py:108` — optimizer rotates `strategy` as a categorical
  search param across `AVAILABLE_STRATEGIES` (the 5 + `blend`); `lock_strategy` flag at :409.
- `backend/backtest/analytics.py:239` — `compute_deflated_sharpe(observed_sr, num_trials, ...)`
  with the multiple-testing `num_trials` deflation already implemented.
- `backend/autoresearch/gate.py:19` — `PromotionGate(min_dsr=0.95, max_pbo=0.20)`.
- `backend/autoresearch/friday_promotion.py:32` — WEEKLY promotion, ranks DSR desc / PBO asc,
  top-N, writes BQ. Operates on `candidates` (trials), not the 5 named strategies.
- `backend/autoresearch/promoter.py:61` — `write_to_registry` = atomic supersede prior active
  -> write new active -> P0 Slack (phase-25.R "red-line goal-c" auto-switch).
- `backend/autoresearch/monthly_champion_challenger.py` — monthly Sortino gate + 48h HITL;
  `actual_replacement` gated on `Settings.real_capital_enabled` (default False, SR 11-7).
- `backend/db/bigquery_client.py:702-845` — `promoted_strategies` table: save (MERGE on
  week_iso+strategy_id) / get_latest (status filter, DSR-desc) / update_status (supersede).
- `backend/services/autonomous_loop.py:46` — `load_promoted_params(bq)`: 3-tier fallback
  BQ-promoted -> optimizer_best.json -> {}. **This is the loop's strategy/params read path.**
- `backend/db/bigquery_client.py:403` + `autonomous_loop.py:1082` — `strategy_decisions`
  audit table + per-cycle heartbeat writer (phase-30.7). Comment at loop:1077 literally says
  "Full router activation deferred to phase-31."

**The actual GAP (phase-47.6):** no pure function does a *bake-off across the 5 named
STRATEGY_REGISTRY strategies* — compute per-strategy DSR (deflated by N=5), apply DSR>=0.95
guard + anti-churn min-improvement (Δ-DSR) + incumbent-tie rule, pick the winner. The
existing pipeline promotes optimizer-TRIAL params (one search trajectory), not a 5-way
named-strategy comparison.

**Data reality:** per-strategy DSR is NOT readable from existing artifacts.
`optimizer_best.json` = single incumbent only (triple_barrier, Sharpe 1.17, DSR 0.9526,
stamped 2026-04-06). `quant_results.tsv` has `strategy` only inside `params_json`, sparse /
triple_barrier-dominated. A REAL selection needs 5 quant-only ($0-LLM) walk-forward
backtests; the SELECTOR FUNCTION itself is pure and unit-testable WITHOUT backtests.

**Why:** phase-47.6 research gate, 2026-05-29. The prompt assumed friday_promotion etc. were
"aspirational"; they are real (phase-25/26/30). 
**How to apply:** when extending rotation, extend `friday_promotion`/`promoted_strategies`/
`load_promoted_params` and the existing test idiom (`tests/autoresearch/test_friday_promotion.py`,
`tests/verify_phase_25_*.py`) — do NOT invent a parallel promotion mechanism. See
[[psr-dsr-formulas]] for the DSR formula details.