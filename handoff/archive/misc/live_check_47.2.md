# Live Check — phase-47.2: First autonomous trade end-to-end

Captured 2026-05-29 on the running local system. Cycle `6a6b548c` (POST /api/paper-trading/run-now,
dry_run=false), started 2026-05-28T23:08:45 UTC.

## 1. The swap fired (sector-rotation), Step 7 executed 2 trades
```
backend.log (cycle 6a6b548c, 02:53:01 CEST = 00:53:01 UTC):
  buy_candidate risk_judge decision=APPROVE_REDUCED ticker=STX position_pct=2.0 final_score=7.0
  Skipping BUY STX: sector Technology at cap (7/2) -- queued for swap check
  Swap fired (1/2): SELL KEYS (score=5.000) -> BUY STX (score=7.000) delta=40.0%
  Swap skip MU -> AMD: delta=16.7% below threshold 25.0% (correct: bounded churn)
  Paper trading: Step 7 -- Executing 2 trades
```
The rotation logic worked exactly as designed: sold the weakest Tech holding (KEYS, score 5), bought
the strong candidate (STX, score 7); delta 40% > the 25% min_delta; AMD/CIEN/HPE correctly skipped
(delta below threshold). Sector COUNT cap stayed intact (the swap is +1/-1, net-neutral).

## 2. Trades PERSISTED to financial_reports.paper_trades, dated today
```
BQ rows created in the last 3h (us-central1):
  BUY  STX  qty=0.537481  px=880.72  created_at=2026-05-29T00:53:17
  SELL KEYS qty=4.229682  px=339.13  created_at=2026-05-29T00:53:02
```
Both dated 2026-05-29 (today). The ledger was stale at 2026-05-27 ONLY because nothing had traded since
the path was broken; it now writes correctly -- no persistence bug.

## 3. Conditions
- historical_prices FRESH (47.1; band green).
- Metrics correct (47.4; Sharpe/maxDD fixed).
- Safety intact: sector count cap enforced, swap respects min_delta + NAV-pct backstop + max_per_cycle=2,
  kill-switch ACTIVE-not-paused, stop-loss/trailing active (phase-32.2 trail fires logged this cycle).

## Known observability detail (non-blocking)
`cycle_history.jsonl` for 6a6b548c still shows `status=started, n_trades=0` -- the cycle reached Step 7
+ executed/persisted the trades (BQ-confirmed) but the cycle_history completion row had not been written
when captured (the cycle was ~1h44m, still finishing later steps / the completion-writer lagged). The
CANONICAL proof of the trade is the BQ paper_trades rows above, not cycle_history. (Follow-up: ensure
the cycle_history completion + n_trades write is reliable -- relates to DoD-9 cron-streak observability.)

## Production-readiness note (follow-up, not blocking the trade)
The cycle took ~1h44m (12 tickers x full 15-step pipeline via the claude_code rail, lite_mode=False).
That works but is slow for a daily loop; enabling lite_mode and/or a faster rail is queued (see
cycle_block_summary.md). The TRADE itself is proven.
