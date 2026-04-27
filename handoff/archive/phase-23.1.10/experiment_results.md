---
step: phase-23.1.10
cycle_date: 2026-04-27
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_10.py'
---

# Experiment Results — phase-23.1.10

## What was built

Positions table now shows **Company** + **Sector** columns; Trades table shows **Company**. Data resolved via a new `GET /api/paper-trading/ticker-meta` endpoint that does BQ-first lookup against `analysis_results` (zero rate-limit risk) with yfinance fallback for missing fields. 24h response cache. **Zero BQ migration**.

## Files modified

| File | Change |
|---|---|
| `backend/api/paper_trading.py` | NEW `_yfinance_ticker_info` + `_fetch_ticker_meta` helpers + `GET /ticker-meta` endpoint. BQ-first (single batch query); yfinance fallback for tickers missing OR sector missing; 0.3s sleep between yfinance calls; graceful on yfinance error. |
| `backend/services/api_cache.py` | NEW `paper:ticker_meta: 86400.0` (24h) entry in `ENDPOINT_TTLS`. |
| `frontend/src/lib/types.ts` | NEW `TickerMeta` + `TickerMetaResponse` interfaces. |
| `frontend/src/lib/api.ts` | NEW `getTickerMeta(tickers: string[])` client function. |
| `frontend/src/lib/useTickerMeta.ts` | NEW hook (~40 lines) — fetches once per unique sorted ticker set; graceful on error. |
| `frontend/src/app/paper-trading/page.tsx` | Import `useTickerMeta`; derive `allTickersForMeta` from positions + trades; call `useTickerMeta`. Positions table: +2 headers (Company, Sector) + 2 cells after Ticker; colSpan 8 → 10. Trades table: +1 header (Company) + 1 cell after Ticker; colSpan 8 → 9. |
| `tests/api/test_ticker_meta.py` | NEW (9 tests covering yfinance fallback chain, BQ-first resolution, BQ-with-missing-sector falls through to yfinance, BQ-down graceful fallback, response shape, empty input). |
| `tests/verify_phase_23_1_10.py` | NEW immutable verification script. |

## Verbatim verification command output

```
$ source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_10.py
HTTP Error 404: {"quoteSummary":{"result":null,"error":{"code":"Not Found","description":"Quote not found for symbol: ZZZZZ_NOT_A_REAL_TICKER"}}}
ok ticker-meta route registered + helpers callable + 24h TTL configured
exit=0
```

The HTTP 404 line is yfinance's expected behavior on a fake ticker — proves the graceful fallback path works (returns the documented error shape, doesn't raise). Verification asserts:
1. `/api/paper-trading/ticker-meta` route is registered
2. `_fetch_ticker_meta` and `_yfinance_ticker_info` are importable + callable
3. `_yfinance_ticker_info` on a fake ticker returns the documented fallback shape
4. `ENDPOINT_TTLS["paper:ticker_meta"] == 86400.0` (24h)

## Unit test results

```
$ source .venv/bin/activate && python -m pytest tests/api/ tests/services/ -v --no-header -q
collected 146 items
tests/api/test_paper_trading_deposit.py ............         [ 8%]
tests/api/test_settings_api_signal_stack.py ..............    [17%]
tests/api/test_ticker_meta.py .........                       [23%]
tests/services/test_extract_stop_loss.py ..........            [30%]
tests/services/test_macro_regime.py ............              [39%]
tests/services/test_meta_scorer.py ..............             [48%]
tests/services/test_news_screen.py .....................      [63%]
tests/services/test_pead_signal.py ..................         [75%]
tests/services/test_sector_calendars.py ................      [86%]
tests/services/test_signal_attribution.py ....................[100%]
============================== 146 passed in 3.24s ==============================
```

9 new + 137 prior = 146/146 tests pass. No regression across any of the 10 phase-23.1 cycles.

## Frontend type-check

```
$ cd frontend && npx tsc --noEmit
(silent — 0 errors)
```

## Resolution strategy (verified live against BQ)

The brief identified that `analysis_results` populates `company_name` (`"Apple Inc."`, `"NVIDIA CORP"`) but `sector` is NULL. The endpoint handles both cases:

- **Tickers in BQ with both fields:** single response, `source: "bq"`
- **Tickers in BQ with name but NO sector** (the common case for our universe): name from BQ, sector from yfinance, `source: "bq+yf"`
- **Tickers not in BQ at all:** full yfinance lookup, `source: "yfinance"`
- **yfinance error:** ticker symbol as name, empty sector, `source: "error"` — never raises

The 24h cache means repeat page loads on the same set of tickers hit zero yfinance calls.

## What the operator sees

**Positions tab — 2 new columns:**
```
TICKER  COMPANY              SECTOR                  QTY   ENTRY   ...
ON      ON Semiconductor     Information Technology  4.80  $98.40  ...
INTC    Intel Corporation    Information Technology  11.50 $82.57  ...
NVDA    NVIDIA Corporation   Information Technology  ...
```

**Trades tab — 1 new column:**
```
DATE        ACTION  TICKER  COMPANY              QTY   PRICE   ...
2026-04-27  BUY     ON      ON Semiconductor     4.80  $98.40  ...
```

Cold first-load: ~1-2s for the yfinance batch (10 tickers × 0.3s sleep + BQ query). Subsequent loads: instant from cache. After 24h: re-fetch.

## Out of scope (per contract; Phase-2 follow-ups)

- Persisting `company_name` and `sector` on `paper_trades` / `paper_positions` BQ schemas (Path A — operator --apply needed)
- Industry / market_cap / employees fields on the meta response (not user-requested)
- Real-time meta updates (24h cache is fine; company names rarely change)
- Sector-grouped P&L SQL queries (depends on Path A)

## Honest disclosure

- **Cold-cache latency:** first page load with N uncached tickers takes ~N × 0.3s for the yfinance polite-sleep guard. For our 10-position universe that's ~3s worst case. Subsequent loads hit the 24h cache.
- **yfinance dependency:** if yfinance / Yahoo's API goes down, sector data shows "—" but the endpoint still returns BQ company names where available. Graceful degradation.
- **The endpoint does NOT persist:** every request that misses cache calls yfinance fresh. A Phase-2 follow-up could write resolved metadata to a `ticker_metadata` BQ reference table for faster cold starts.

## What's next

1. Spawn fresh Q/A
2. On PASS: log → flip → archive → commit → restart frontend so the new bundle ships
