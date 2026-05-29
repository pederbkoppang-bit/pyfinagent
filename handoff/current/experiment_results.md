# Experiment Results — phase-47.7: Learn-loop correctness fix (read real paper_trades P&L field)

**Cycle:** 8 (priority 6 / DoD-6). FREE of LLM spend. **Result:** ready for Q/A.

## What was changed (3 edits: 1 code fix + 1 comment + 1 test de-mask)
Root cause (research-validated): the learn-loop sell-close fallback read `trade.get("return_pct")`, but
paper_trades rows carry **`realized_pnl_pct`** (paper_trader.execute_sell). So every sell-close recorded
**0.0 return** -- the learn-loop's core signal (true realized P&L) was always zero. The existing test
masked it by hand-setting `return_pct` in its mock.

1. `backend/services/autonomous_loop.py:1981` -- now reads `realized_pnl_pct` first, falls back to
   `return_pct`, then 0.0 (`_rp = trade.get("realized_pnl_pct"); if _rp is None: _rp =
   trade.get("return_pct"); pnl_pct = float(_rp or 0.0)`).
2. `autonomous_loop.py:1976-78` -- corrected the stale "bq.save_outcome is an UPSERT" comment to
   "APPENDS (not an upsert); re-running could duplicate the (ticker, analysis_date) row; dedup is a follow-up".
3. `backend/tests/test_phase_35_1_learn_loop_writer.py:39` -- mock now uses `realized_pnl_pct` (the REAL
   field), so the fallback-path test is a genuine regression guard against the field bug.

**NOT changed:** the operator-gated `paper_learn_loop_enabled` flag (settings.py:32, default False -- its
own description says "operator flips to true"). Enabling it incurs per-sell-close Anthropic reflection
spend (claude-sonnet-4-6) = operator-gated. Left for the operator.

## Verbatim verification output
```
$ python -c "import ast; ast.parse('backend/services/autonomous_loop.py')"  -> ast OK
$ python -m pytest backend/tests/test_phase_35_1_learn_loop_writer.py -q     -> 5 passed in 2.18s

Mutation proof (de-masked mock trade = {realized_pnl_pct: 17.89, no return_pct}):
  PRE-FIX  read return_pct        -> 0.0    (test asserts 17.89 -> FAILS)
  POST-FIX read realized_pnl_pct  -> 17.89  (test asserts 17.89 -> PASSES)
  mutation guard CONFIRMED.

$ grep "paper_learn_loop_enabled: bool = Field(False" settings.py  -> present (flag still operator-gated, NOT flipped)
```

## Success-criteria mapping (masterplan phase-47.7)
1. fallback reads real field realized_pnl_pct (fallback return_pct) not the non-existent return_pct -- **MET**.
2. test mock uses realized_pnl_pct -> guards the bug (fails pre-fix on the fallback path, passes post-fix); all pass -- **MET** (5 passed; mutation proof shows pre-fix 0.0 vs post-fix 17.89).
3. operator-gated flag NOT flipped (test_field_default_off passes); UPSERT comment corrected -- **MET** (flag still Field(False)).
4. ast clean; pytest green -- **MET**.

## Scope honesty / deferred
Fixes the learn-loop CORRECTNESS (it will now record true P&L when enabled). Does NOT flip the
operator-gated flag (no LLM spend incurred) and does NOT claim a LIVE outcome_tracking row -- the live
evidence needs the operator to enable the flag + a sell-close cycle (and the reflection fan-out is
Anthropic-metered = operator-gated). save_outcome append-only dedup + the DoD-6 probe's non-existent
cycle_id column are flagged follow-ups, not fixed here.

## Files
backend/services/autonomous_loop.py, backend/tests/test_phase_35_1_learn_loop_writer.py,
.claude/masterplan.json (phase-47.7), handoff/current/{contract.md, research_brief_phase_47_7_learn_loop.md}.
