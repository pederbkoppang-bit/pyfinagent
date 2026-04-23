# Cycle 50 -- Experiment Results

## What was built

Wired `autonomous_loop.py` to log signal events to BQ `signals_log` table after trade execution. This is the infrastructure fix identified in Cycle 49's root-cause analysis (blocking code path #2: "autonomous loop never calls `publish_signal()`").

## Changes

| File | Lines changed | Description |
|------|---------------|-------------|
| `backend/services/autonomous_loop.py` | +70 / -1 | Added `import hashlib`, `_log_cycle_signals_to_bq()` helper, two call sites |

## Design decisions

1. **Did NOT call `publish_signal()`** -- that method does risk_check + paper_trader.execute_buy/sell + Slack posting internally. Calling it from the autonomous loop would double-execute trades. Instead, wrote directly to BQ via `bq.save_signal()`.

2. **HOLD heartbeat on no-order days** -- when `decide_trades()` produces zero BUY/SELL orders, a single HOLD row with ticker="$CYCLE" is written. This ensures every daily cycle produces >= 1 `signals_log` row, which is what the 4.4.2.4 drill checks.

3. **Kill-switch path covered** -- the early-return at Step 5.5 (kill-switch halt) also logs a HOLD heartbeat, preventing reliability gaps during risk events.

4. **Best-effort write** -- each `save_signal()` call is wrapped in try/except. Failures are logged but never raise. Matches the pattern in `_append_signal_history()`.

5. **Signal ID generation** -- SHA1-16 prefix of `"{ticker}:{date}:{action}"` for trade signals, `"HOLD:{date}:daily_cycle"` for heartbeats. Deterministic, so re-running the same day dedupes naturally (same signal_id).

## Verification

```
$ python3 -c "import ast; ast.parse(open('backend/services/autonomous_loop.py').read()); print('SYNTAX OK')"
SYNTAX OK
```

## What this does NOT do

- Does not implement `generate_signal()` (still a stub) -- that's a separate concern
- Does not flip the 4.4.2.4 checkbox -- data needs to accumulate over >= 14 NYSE trading days first
- Does not touch `publish_signal()` or `_append_signal_history()` in signals_server.py
