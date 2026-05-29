# Contract — phase-47.7: Learn-loop correctness fix (read real paper_trades P&L field)

**Cycle:** 8 of the production-ready+money push (priority 6 / DoD-6). FREE of LLM spend.
**Step:** 47.7 | **Phase:** phase-47 | **Status:** in-progress | **Harness:** required | **Tier:** moderate-complex.

## Research-gate summary (PASSED)
Researcher `adf9093569a532de8`, tier=moderate-complex, `gate_passed: true`. 6 sources in full, 17 URLs,
recency scan, 8 internal files (+ live BQ schema). Brief: `research_brief_phase_47_7_learn_loop.md`.

Both `financial_reports.outcome_tracking` AND `financial_reports.agent_memories` are EMPTY ever (0 rows,
us-central1). TWO root causes:
- RC#1 (flag OFF): `settings.py:32 paper_learn_loop_enabled` defaults False; `autonomous_loop.py:1970-71
  if not learn_loop_enabled: continue`. The flag's own description says "operator flips to true" -- it is
  OPERATOR-GATED by design. NOT flipped here.
- RC#3 (FIELD BUG): `autonomous_loop.py:1981` read `trade.get("return_pct")`, but paper_trades rows carry
  `realized_pnl_pct` (paper_trader.execute_sell, :350-369; get_paper_trades SELECT * no remap). So the
  fallback wrote 0.0 return for EVERY sell-close. The existing test masked it by hand-setting return_pct
  in its mock (test line 39). Also a stale comment claimed save_outcome is an UPSERT (it APPENDS).

## Hypothesis
Reading `realized_pnl_pct` (with `return_pct` fallback) makes the learn-loop record the TRUE realized
return instead of 0.0, so when the operator enables the (operator-gated) flag, outcome_tracking/agent_
memories carry real P&L. De-masking the test mock (use realized_pnl_pct) turns the fallback-path test
into a genuine regression guard. The flag stays operator-gated; no LLM spend incurred this cycle.

## Immutable success criteria (verbatim from masterplan.json phase-47.7)
1. autonomous_loop.py sell-close fallback reads the REAL paper_trades P&L field (realized_pnl_pct, with return_pct as fallback), NOT the non-existent return_pct -- so a sell-close writes the true return to outcome_tracking instead of 0.0
2. the regression test mock (test_phase_35_1_learn_loop_writer.py) uses realized_pnl_pct (the real field) so it now GUARDS the bug (the fallback-path test fails on the pre-fix code, passes post-fix); all tests in the file pass
3. the operator-gated paper_learn_loop_enabled flag is NOT flipped (stays default False per its design; test_field_default_off still passes); the misleading save_outcome 'UPSERT' comment corrected to append-only
4. ast.parse clean; pytest green

## Plan steps
1. `autonomous_loop.py:1981` -- read realized_pnl_pct (fallback return_pct, then 0.0). DONE.
2. `autonomous_loop.py:1976-78` -- correct the UPSERT->append comment. DONE.
3. `test_phase_35_1_learn_loop_writer.py:39` -- mock uses realized_pnl_pct (real field) so the fallback
   test guards the bug. DONE.
4. Verify: ast + `pytest test_phase_35_1_learn_loop_writer.py` (all pass post-fix; fallback test fails
   pre-fix). Mutation proof inline. Fresh Q/A.

## Blast radius
Learn-loop outcome-write path (autonomous_loop fallback) + its test. No flag flip (operator-gated, no
LLM spend). No trade-execution change. Additive correctness fix.

## Deferred (documented)
- LIVE outcome_tracking row from a real sell-close: needs the operator to flip paper_learn_loop_enabled
  ("operator flips to true") + a sell-close cycle. The reflection fan-out (agent_memories) uses an
  Anthropic LLM (claude-sonnet-4-6) = metered spend = operator-gated.
- save_outcome composite-dedup (currently append-only). The DoD-6 probe references a `cycle_id` column
  that neither outcome_tracking nor agent_memories has -- DoD-6 criterion/probe needs reconciliation.

## References
- `research_brief_phase_47_7_learn_loop.md` (gate); `backend/services/autonomous_loop.py:1880-2046`
- `backend/services/paper_trader.py:350-369` (realized_pnl_pct source); `backend/db/bigquery_client.py:375-392` (save_outcome)
- `backend/config/settings.py:32` (operator-gated flag); `backend/services/outcome_tracker.py`
