---
name: learn-loop-state
description: phase-47.7 learn-loop root cause — outcome_tracking + agent_memories empty; flag-off + return_pct/realized_pnl_pct field bug; both tables in financial_reports us-central1
metadata:
  type: project
---

Learn-loop (per-sell-close outcome + reflection writes) was BROKEN as of phase-47.7
(2026-05-29). Both `financial_reports.outcome_tracking` and `financial_reports.agent_memories`
had 0 rows ever. Both live in dataset `financial_reports` @ region **us-central1** (NOT
pyfinagent_data, NOT US) — query at the wrong location returns NotFound. Neither table has a
`cycle_id` column, so the DoD-6 probe `WHERE cycle_id IS NOT NULL` is invalid SQL.

Two root causes (the cycle_block_summary "swap-SELL bypasses the path" hypothesis was WRONG —
all sells incl. swap reach `_learn_from_closed_trades` via autonomous_loop.py:962-975,1035-1038):
1. `paper_learn_loop_enabled=False` (settings.py:32 default; operator gate per /goal gate 3) ->
   autonomous_loop.py:1970-1971 `if not learn_loop_enabled: continue` fires before any write.
   The "OutcomeTracker reflection-model constructed" log (line 1915) is a RED HERRING — it logs
   before the flag check, so it appears even when nothing is written.
2. **Field-name bug:** the fallback writer reads `trade.get("return_pct")` (autonomous_loop.py:1981)
   but `paper_trader.execute_sell` writes the P&L as **`realized_pnl_pct`** (paper_trader.py:364);
   `paper_trades` table has `realized_pnl_pct`, NOT `return_pct` (verified live: 22 rows). So even
   with the flag ON, every fallback row would record return_pct=0.0. The phase-35.1 test masked
   this by hand-setting `return_pct=17.89` in its mock fixture (a field the real row never has).

**Why:** This is the textbook "silent memory-write failure" trap (Memory for Autonomous LLM Agents
survey arXiv:2603.07670 §4.4/§7.7 — writes fail with no exception/log; fix is per-write logging +
write-assertion tests).
**How to apply:** When verifying any BQ-write path, grep the ACTUAL producer for the field name
and check the live table schema — do not trust mock-fixture field names. `save_outcome` is
append-only (`insert_rows_json`), NOT an UPSERT despite the autonomous_loop.py:1977 comment.
Live confirmation needs the flag ON + a real sell-close cycle (~1h45m) — unit tests only prove
the code path calls the writers. Related: [[metric-source-paths]] (DESC-order trap, same
"trust the real shape" lesson).
