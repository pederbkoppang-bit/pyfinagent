# Contract -- Cycle 42 -- Phase 4.4.2.1 Paper Trading Runtime >= 2 Weeks

## Target
Checklist item 4.4.2.1: "Continuous paper trading run time on the latest parameter set reaches the 2-week wall-clock floor"

## Hypothesis
Paper trading has been running since 2026-03-20 (inception_date in BQ paper_portfolio). Today is 2026-04-22, giving a delta of 33 days -- well above the 14-day floor. Prior NOOP cycles (31-41) incorrectly classified this as "wall-clock gated" because they conflated "the item is about wall-clock time" with "the wall-clock hasn't elapsed yet." The clock HAS elapsed. A BQ-querying drill can verify this mechanically.

## Success Criteria (from checklist HOW)
- Query BigQuery `paper_portfolio` / `paper_portfolio_snapshots` for inception and latest snapshot timestamps
- Compute delta >= 14 days
- Identify the current parameter cohort

## Plan
1. Write `scripts/go_live_drills/paper_runtime_test.py` (BQ-querying, 8 checks)
2. Run drill, confirm exit 0
3. Flip checklist item 4.4.2.1 `[ ]` -> `[x]` with evidence line
4. Save evidence snapshot to `backend/backtest/experiments/results/`

## Checks
- S0: Paper portfolio exists in BQ
- S1: Valid inception date parsed
- S2: Wall-clock delta >= 14 days (HARD GATE)
- S3: Snapshot coverage exists (multiple dates)
- S4: Parameter cohort identified (optimizer_best.json)
- S5: Starting capital > 0
- S6: Latest update within 48h (soft note if stale)
- S7: Trade count (informational)
