# Contract -- Cycle 50

## Step

Phase 4.4.2.4 infrastructure prep: wire autonomous_loop.py to log signals to BQ signals_log.

## Problem

The autonomous loop (`run_daily_cycle`) executes trades via `paper_trader.execute_buy/sell()` but never writes to `signals_log` in BQ. The `publish_signal()` method on `SignalsServer` has the BQ write path (via `_append_signal_history`), but it's only reachable through the MCP tool interface -- the autonomous loop doesn't call it. Root cause identified in Cycle 49.

The 4.4.2.4 drill (`signal_reliability_test.py`) queries `signals_log WHERE event_kind = 'publish' GROUP BY signal_date` and expects at least one row per NYSE trading day. With zero rows, it exits with code 2 (SKIP).

## Hypothesis

Adding a `_log_cycle_signals_to_bq()` helper that writes one `signals_log` row per trade order (or a HOLD heartbeat on no-order days) after Step 7 in `run_daily_cycle()` will populate the BQ audit trail and unblock the 4.4.2.4 drill once data accumulates.

## Plan

1. Add `import hashlib` to autonomous_loop.py
2. Add `_log_cycle_signals_to_bq(bq, orders, today_str)` function:
   - For each BUY/SELL order: write a publish event with signal_id, ticker, signal_type, factors, price
   - If no orders: write a single HOLD heartbeat with ticker="$CYCLE"
   - Best-effort: never raises, logs warnings on failure
3. Call the helper after Step 7 (execute trades) in `run_daily_cycle()`
4. Also call with a HOLD record after kill-switch halt (Step 5.5)
5. Verify with `python -c "import ast; ast.parse(...)"`

## Success criteria

- SC1: `autonomous_loop.py` parses without errors
- SC2: `_log_cycle_signals_to_bq` writes to BQ via `bq.save_signal()`
- SC3: Each daily cycle produces >= 1 signals_log row with event_kind="publish"
- SC4: No duplicate trade execution (publish_signal NOT called)
- SC5: Kill-switch halt path also logs a HOLD heartbeat
- SC6: Best-effort write -- never raises on BQ failure
