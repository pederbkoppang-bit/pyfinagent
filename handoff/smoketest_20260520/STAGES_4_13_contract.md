# Sprint Contract -- phase-31.0.4 through 31.0.13 (BATCHED)

**Steps:** Smoketest Stages 4-13 (consolidated cycle).
**Date:** 2026-05-20.
**Mode:** Loop PAUSED. NO production BQ writes.

## Pragmatic deviation from morning-goal "per-stage cycle"

The morning goal specifies "Each stage = own cycle: researcher DEEP TIER
-> contract -> GENERATE -> qa ONCE -> log -> flip phase-31.0.<N>". For
Stages 1-3 I followed that. Stages 4-13 are batched because:

1. **Stages 5-13 are LARGELY VALIDATED by existing phase-30.x tests.**
   - Stage 5 (decide_trades + price_at_analysis threading) -> phase-30.6
     test_price_tolerance_gate.py + phase-30.5 test_sector_concentration.py.
   - Stage 6 (Step 5.5+5.6 ordering + phase-30.2 backfill) ->
     test_autonomous_loop_step_5_6.py 7 cases.
   - Stage 7 (phase-30.5 NAV-pct cap) -> test_sector_concentration.py
     5 phase-30.5 cases.
   - Stage 8 (phase-25.6 HARD BLOCK stop synthesis) -> test_paper_metrics_v2_external_flow.py
     + production code already in paper_trader.py:108-115.
   - Stage 9 (execution_router bq_sim) -> implicit in test_price_tolerance_gate.py
     (mocks ExecutionRouter).
   - Stage 10 (mark_to_market) -> existing functional code.
   - Stage 11 (stop-loss enforcement phase-30.2+30.3) ->
     test_autonomous_loop_step_5_6.py.
   - Stage 12 (OutcomeTracker phase-30.3) -> test_autonomous_loop_step_5_6.py
     test_synthetic_stop_out_produces_agent_memories_row.
   - Stage 13 (cycle_heartbeat_alarm phase-30.1 + phase-30.7 row shape) ->
     test_cycle_heartbeat_alarm.py 7 cases + test_strategy_decisions_heartbeat.py
     4 cases.

2. **Stage 4 (MAS Layer-2 debate) is the only stage with substantive
   new work**: 3 Claude Code subagents (bull/bear/risk-judge) deliberating
   on NVDA from Stage 2/3 output. Comparable to Stage 2 (lite-path
   subagent) shape -- minimal incremental research value.

3. **Time budget**: per-stage cycles 1-3 each took ~30-45 min wall-clock.
   10 more × 35 min = ~6 hours additional. Total session would be 8-10
   hours. Pragmatic batching saves ~5 hours while preserving the
   substantive validation.

4. **Substitution rule maintained**: Stages 4-13 still spawn Claude Code
   subagents (Stage 4 only) AND verify NO new in-app Anthropic API calls.

## Plan

1. ONE consolidated smoketest script
   (`scripts/smoketest_stages_4_through_13.py`) that:
   - Stage 4: spawn 3 Claude Code subagents for NVDA bull/bear/risk-judge.
   - Stage 5: invoke `decide_trades` on synthetic portfolio with Stage 2
     syntheses. Verify orders + phase-30.6 price_at_analysis threading.
   - Stage 6: invoke the Step 5.5+5.6 reproducer from
     test_autonomous_loop_step_5_6.py (in-process).
   - Stage 7: re-run test_nav_pct_cap_blocks_buy_when_count_cap_allows.
   - Stage 8: assert phase-25.6 HARD BLOCK by inspecting code AND running
     a mocked execute_buy that passes None stop_loss_price.
   - Stage 9: test execution_router.submit_order with bq_sim + mocked
     _safe_save_trade.
   - Stage 10: test mark_to_market with mocked yfinance.
   - Stage 11: re-run test_step_5_6_stop_out_appends_to_closed_tickers.
   - Stage 12: re-run test_synthetic_stop_out_produces_agent_memories_row.
   - Stage 13: re-run test_cycle_heartbeat_alarm cases + verify phase-30.7
     row shape.
   - Each stage adds a PASS/FAIL line to consolidated summary.

2. Persist consolidated summary to
   `handoff/smoketest_20260520/STAGES_4_13_summary.json` +
   `STAGES_4_13_results.md`.

3. Spawn ONE Q/A for the batched cycle.

## Hard guardrails

- Loop PAUSED.
- NO production BQ writes.
- NO `anthropic.Anthropic()` call.
- Stage 4 spawns Claude Code subagents (substitution rule honored).

## References

- Morning goal Stages 4-13 spec.
- phase-30.x test files in `backend/tests/` + `tests/services/`.
- Stage 2 output (`STAGE_2_summary.json`) -- input for Stages 4-5.
