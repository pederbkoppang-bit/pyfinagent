# Experiment Results -- Phase 4.4.2.4 No Missed Trading Days

**Date:** 2026-04-21
**Cycle:** 31
**Item:** 4.4.2.4 No missed trading days (signal generation reliable)
**Outcome:** BLOCKED (drill exits 1, evidence insufficient)

## What was built

1. **BQ evidence snapshot**: `backend/backtest/experiments/results/signal_generation_evidence_20260421.json`
   - Queried all BQ datasets for `signals_log` table: NOT FOUND
   - Fallback: `financial_reports.analysis_results` used as signal proxy
   - Paper trading window: 2026-03-20 to 2026-04-21 (32 calendar days)
   - Signal generation days: 2 (Mar 20: 1 SNDK/Hold, Mar 21: 2 SNDK/Hold)
   - Mar 21 is a Saturday (non-trading day)

2. **Drill**: `scripts/go_live_drills/signal_reliability_test.py`
   - stdlib-only, follows kill_switch_test.py pattern
   - Loads evidence JSON, computes NYSE trading days (Mon-Fri minus US holidays)
   - Compares signal generation dates against trading calendar
   - 7-check battery: evidence load, signals_log status, trading day count,
     signal day count, coverage gate (100%), gap list, non-trading-day check

## Drill output

```
4.4.2.4 No Missed Trading Days Drill
  [+] S0: Evidence loaded, query_date=2026-04-21
  [X] S1: signals_log table missing
  [+] S2: NYSE trading days in window: 22
  [+] S3: Signal generation days in BQ: 2
  [X] S4: Coverage: 1/22 = 4.5% (gate: 100%)
  [X] S5: 21 missed trading days
  [i] S6: 1 signal day outside trading calendar (2026-03-21 = Saturday)
DRILL FAIL: 3/7
```

## Root causes

1. **signals_log migration never executed.** `scripts/migrations/migrate_signals_log.py`
   was scaffolded in Cycle 5 (Phase 4.2.4) but the `CREATE TABLE IF NOT EXISTS`
   was never run against BQ. The table does not exist.
2. **Signal generation pipeline not running daily.** The autonomous loop
   (`backend/services/autonomous_loop.py`) needs to be scheduled and running
   to generate daily signals. Only 2 analyses were recorded since inception.
3. **All analyses were for a single ticker (SNDK) with Hold recommendation.**
   No BUY/SELL signals were generated that would trigger paper trades.

## Files changed

| File | Action |
|------|--------|
| `scripts/go_live_drills/signal_reliability_test.py` | NEW -- drill for 4.4.2.4 |
| `backend/backtest/experiments/results/signal_generation_evidence_20260421.json` | NEW -- BQ snapshot |
| `handoff/current/contract.md` | Updated for 4.4.2.4 |
| `handoff/current/experiment_results.md` | This file |
| `handoff/current/evaluator_critique.md` | Updated |
