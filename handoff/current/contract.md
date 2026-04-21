# Contract: 4.4.2.4 No missed trading days (signal generation reliable)

## Step ID
4.4.2.4

## Target
Verify that every US market open day in the paper trading window has a
signal-generation log entry in BigQuery.

## Verification criteria (from checklist)
- Every US market open day in the paper trading window has a signal-generation
  log entry
- Query BigQuery `signals_log` with `event_kind = "publish"` grouped by day
- Compare distinct days against NYSE trading calendar for the window
- Zero gaps is the gate

## BQ data assessment (2026-04-21)
- `signals_log` table does NOT exist in any dataset (migration scaffolded in
  Cycle 5 but never executed against BQ)
- Fallback: `financial_reports.analysis_results` has `analysis_date` column
- Paper trading inception: 2026-03-20
- Signal generation days since inception: 2 (Mar 20: 1 analysis, Mar 21: 2)
- Approximate US trading days in window (Mar 20 - Apr 21): ~22
- Coverage: 2/22 = ~9% -- far below 100% gate

## Expected outcome
BLOCKED -- signal generation pipeline not running daily. Drill will be written
for future re-verification when daily signal generation is activated.

## Drill plan
1. Save BQ evidence snapshot to `backend/backtest/experiments/results/`
2. Write stdlib-only drill at `scripts/go_live_drills/signal_reliability_test.py`
3. Drill checks: evidence file exists, signals_log table status, signal days
   vs NYSE calendar, gap count, coverage percentage
4. Exit 0 only if zero gaps (will exit 1 this cycle)
