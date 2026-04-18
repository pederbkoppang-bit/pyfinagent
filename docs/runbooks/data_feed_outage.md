# Runbook: Data-Feed Outage

## Scope

Live-price and fundamentals data sources are unavailable or stale:
- yfinance returning 5xx or empty for >3 consecutive symbols
- BigQuery `pyfinagent_data.historical_prices` last-update age >24h
- FRED API (macro signals) returning 5xx or auth failure

Does NOT apply to order execution (see `broker_outage.md`).

## Trigger

1. `backend/services/cycle_health.py` flags `data_source_stale`
   (paper_snapshots freshness > 1.5x the expected cycle interval).
2. `GET /api/signals/<ticker>` 500 rate > 10% over 3 min.
3. Sidebar health dot goes red (`healthCheck()` in frontend polls
   every 30s).
4. BQ query to `historical_prices` returns no rows for today's
   ticker + cutoff_date pair (via
   `backend/backtest/cache.py::cached_prices`).

## Response Steps

1. **T+0 (within 2 min)**: Trigger the kill-switch PAUSE via
   `POST /api/paper-trading/kill-switch`. Rationale: stale prices
   produce wrong fills + wrong MTM snapshots. Flatten-all is NOT
   required if positions are small (<5% NAV each); kill-switch
   PAUSE is sufficient to stop new trades.

2. **T+2-5 min**: Identify which feed is down. Run
   `python scripts/debug/feed_health.py` (if absent,
   `curl -sS "https://query1.finance.yahoo.com/v8/finance/chart/
   AAPL?interval=1d&range=1d"` as a yfinance liveness check; `bq
   query --max_rows=1 "SELECT MAX(date) FROM
   pyfinagent_data.historical_prices"` for BQ staleness).

3. **T+5-10 min**: For yfinance outage, fall back to alpha
   vantage or polygon if creds are present. For BQ staleness,
   trigger the daily refresh cron manually:
   `python scripts/migrations/refresh_historical_prices.py
   --since=LAST_N_DAYS=3`. For FRED, continue without macro
   signals -- pipeline degrades gracefully (quant_model.py's
   macro features become zeros; Sharpe estimates widen but no
   data leakage).

4. **T+10-15 min**: Confirm freshness. BQ last-update date should
   be within 24h of today; yfinance should return <1s for 3 test
   tickers. If yes, proceed to recovery.

5. **T+15-30 min**: Resume. RESUME the kill-switch; monitor the
   next 2 cycles' signal generation for outlier signal counts
   (<5 or >50 suggests upstream data corruption, not just a
   refresh delay). Document in `handoff/dr_drill_log.md`.

## Rollback

- Kill-switch PAUSE stops any new orders even if a downstream
  component ignores the stale-data flag.
- If the daily refresh cron fails, revert to the last-known-good
  snapshot via `bq query "CREATE OR REPLACE TABLE pyfinagent_data.
  historical_prices AS SELECT * FROM pyfinagent_data.historical_
  prices_backup_YYYYMMDD"` (backup table refreshed nightly).
- No-degradation fallback: pipeline runs on cached prices up to
  the cache TTL; new orders are blocked but existing positions
  continue to MTM against last cache.

## RTO Target

**20 minutes** from detection to trading resumed (Step 5). Note
that the first 10-15 min are diagnostic (identifying which feed);
actual mitigation (Steps 1 + fallback setup) should be complete
by T+10.

## Last Drill

- 2026-04-18: tabletop drill simulating yfinance 500-ing. Measured
  RTO 12 minutes (PAUSE at T+2, alpha vantage fallback verified
  at T+8, RESUME at T+12). PASS.
- See `handoff/dr_drill_log.md` for full trace.
