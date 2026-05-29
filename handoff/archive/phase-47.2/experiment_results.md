# Experiment Results — phase-47.2: First autonomous trade end-to-end

**Cycle:** 6 (resumes parked 47.2). **Step:** 47.2 | **Result:** ready for Q/A. **THE FIRST TRADES EXECUTED.**

## What happened
Fired `POST /api/paper-trading/run-now?dry_run=false` (cycle `6a6b548c`, started 23:08:45 UTC). It ran
the full pipeline (Screen -> Analyze 12 tickers via claude_code rail, scores 6-8 -> Decide -> Execute)
and at 00:53 UTC reached Step 7 and **executed 2 trades via the sector-rotation swap path**:

```
Swap fired (1/2): SELL KEYS (score=5.0) -> BUY STX (score=7.0) delta=40.0%   (> 25% min_delta)
Paper trading: Step 7 -- Executing 2 trades
```

The book was 7 Tech vs cap 2, so direct BUYs were sector-blocked + "queued for swap check"; the swap
sold the weakest Tech holding (KEYS) and bought the top candidate (STX). AMD/CIEN/HPE were correctly
swap-skipped (their delta vs the next-weakest holding MU was 16.7% < 25%). All safety intact.

## Verbatim evidence (BQ persistence -- the canonical proof)
```
financial_reports.paper_trades, rows created last 3h (us-central1):
  BUY  STX  qty=0.537481  px=880.72  created_at=2026-05-29T00:53:17
  SELL KEYS qty=4.229682  px=339.13  created_at=2026-05-29T00:53:02
```
2 fresh rows dated TODAY (2026-05-29). The "no trades" problem is SOLVED.

## Root cause (resolved)
Validated cause (47.2 research): per-sector COUNT cap blocked 100% of buys with no rotation; the
swap-rotation path existed (commit 69c710ec) but the RUNNING backend predated it. The cycle-2/4 backend
restarts loaded the swap code; this cycle fired it. NO code change was needed for the swap itself
(code-verified ready); the fix was operational (restart to load the committed path) + running a cycle.
The diagnostic's "empty new_candidates / stale-prices-block-trading / sod_date" hypotheses were all
REFUTED by the research gate; the real cause was the sector-cap-without-rotation, now resolved.

## Success-criteria mapping (masterplan phase-47.2)
1. run-now returns + cycle produces n_trades >= 1 with non-empty candidate analyses -- **MET** (2 trades; 12 analyses).
2. fresh financial_reports.paper_trades row dated today -- **MET** (BUY STX + SELL KEYS, 2026-05-29).
3. root cause identified + fixed without disabling safety -- **MET** (sector-cap-without-rotation; swap respects min_delta + NAV-pct + count cap + max_per_cycle; kill-switch/stop-loss intact).
4. live_check_47.2.md captures run-now + BQ paper_trades row -- **MET** (live_check_47.2.md).

## Honesty / follow-ups (non-blocking)
- `cycle_history.jsonl` for 6a6b548c still shows started/n_trades=0 (completion-write lag); the trades
  are BQ-confirmed regardless. Follow-up: reliable cycle_history completion write (DoD-9 observability).
- Cycle took ~1h44m (slow claude_code rail, lite_mode=False) -- works but slow for a daily loop;
  lite_mode / faster-rail optimization queued (cycle_block_summary.md). Does NOT block the trade.
- This SELL-close (KEYS) should trigger the learn-loop (outcome_tracking) -- priority 6, now unblocked.

## Files
backend (no code change this cycle -- operational: restart loaded the committed swap path),
.claude/masterplan.json (47.2), handoff/current/{contract.md, live_check_47.2.md,
research_brief_phase_47_2_first_trade.md}.
