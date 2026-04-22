# Phase 4.4.2.1 -- Experiment Results

## What was built

BQ-querying drill at `scripts/go_live_drills/paper_runtime_test.py` that verifies paper trading has been running >= 14 days (the 2-week wall-clock floor from checklist item 4.4.2.1).

## Files changed
- `scripts/go_live_drills/paper_runtime_test.py` (NEW, ~130 lines)
- `docs/GO_LIVE_CHECKLIST.md` (checkbox flip + evidence line)
- `backend/backtest/experiments/results/paper_runtime_evidence_20260422.json` (evidence snapshot)

## Drill output (verbatim)

```
  [+] S0: Paper portfolio: NAV=$9,499.50, PnL=-5.0%
  [+] S1: Inception: 2026-03-20 14:01 UTC
  [+] S2: Running 32 days >= 14-day floor (18 days margin)
  [+] S3: 11 snapshots, 5 distinct dates (2026-04-14 to 2026-04-21)
  [+] S4: optimizer_best.json: Sharpe=1.1705, file=?
  [+] S5: Starting capital $10,000.00
  [+] S6: Last updated 13.6h ago (2026-04-21 12:01 UTC)
  [+] S7: 1 paper trades executed

  DRILL PASS: 8/8
```

## BQ data sources
- `sunny-might-477607-p8.financial_reports.paper_portfolio` (1 row, inception 2026-03-20)
- `sunny-might-477607-p8.financial_reports.paper_portfolio_snapshots` (11 rows, 5 distinct dates)
- `sunny-might-477607-p8.financial_reports.paper_trades` (1 row, XOM test trade 2026-03-28)

## Soft notes
- SN1: Paper trading has been running 32 days but with only 1 trade (XOM test trade). The zero-orders bug in `decide_trades` means the system is running but not generating live trades. This is a known issue documented in Session Note 2026-04-16 and Cycles 31-41 NOOPs.
- SN2: Item 4.4.2.1 is about wall-clock runtime, not trade quality. Trade quality is covered by 4.4.2.2 (Sharpe), 4.4.2.4 (no missed days), and 4.4.2.5 (divergence).
- SN3: WHO is "joint" -- Peder should verify calendar alignment at launch-week.
