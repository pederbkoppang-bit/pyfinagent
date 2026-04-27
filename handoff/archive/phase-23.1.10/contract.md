---
step: phase-23.1.10
title: Show Company name + Sector on Positions and Trades tables (ticker-meta endpoint)
cycle_date: 2026-04-27
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_10.py'
research_brief: handoff/current/phase-23.1.10-research-brief.md
---

# Contract — phase-23.1.10

## Hypothesis

A new `GET /api/paper-trading/ticker-meta?tickers=...` endpoint with BQ-first / yfinance-fallback resolution returns `{ticker → {company_name, sector}}`, cached 24h. Frontend Positions table shows two new columns (Company, Sector); Trades table shows one (Company). Zero BQ migration; existing rows render correctly immediately.

## Plan

1. **Backend `backend/api/paper_trading.py`** — NEW `GET /ticker-meta` endpoint:
   - Query param `tickers: str = Query(...)` (comma-separated, max 50)
   - Cache key `paper:ticker_meta:{sorted_tickers_joined}` with 24h TTL
   - Resolution: query `analysis_results` BQ for most-recent `company_name + sector` per ticker; for tickers not found, call `yf.Ticker(t).info` once each (with sleep between calls) and extract `shortName / longName / sector`
   - Returns `{meta: {ticker: {company_name, sector, source}}, ttl_sec, count}`
   - Graceful: yfinance error → `{company_name: ticker, sector: "", source: "error"}`

2. **Backend `backend/services/api_cache.py`** — add `"paper:ticker_meta": 86400.0` to `ENDPOINT_TTLS` (24h TTL).

3. **Frontend `frontend/src/lib/types.ts`** — NEW `TickerMeta` interface.

4. **Frontend `frontend/src/lib/api.ts`** — NEW `getTickerMeta(tickers: string[])` client function.

5. **Frontend `frontend/src/lib/useTickerMeta.ts`** — NEW hook that takes a `tickers: string[]` array, computes a stable sorted key, and fetches once per unique key (re-fetches when ticker set changes). Returns `{ meta }`. Graceful on error (returns empty `{}`).

6. **Frontend `frontend/src/app/paper-trading/page.tsx`** — derive `allTickers` from positions + trades, call `useTickerMeta(allTickers)`. 
   - Positions table: add `<th>Company</th>` and `<th>Sector</th>` after Ticker; cells render `tickerMeta[pos.ticker]?.company_name ?? "—"` and similar for sector. Update colSpan from 8 to 10.
   - Trades table: add `<th>Company</th>` after Ticker; cell renders `tickerMeta[t.ticker]?.company_name ?? "—"`. Update colSpan accordingly.

7. **Tests** at `tests/api/test_ticker_meta.py`:
   - Endpoint accepts comma-separated tickers
   - Endpoint rejects empty list (400)
   - Endpoint rejects >50 tickers (400)
   - BQ-first resolution wins over yfinance (mocked)
   - Cache hit on second call (mocked)
   - Graceful fallback when yfinance raises (mocked)
   - `_yfinance_ticker_info` returns `{company_name: ticker, sector: ""}` on missing fields

8. **Verification script** `tests/verify_phase_23_1_10.py` — imports the endpoint + helper, asserts they're callable and the route is registered.

## Out of scope

- Persisting `company_name` and `sector` on `paper_trades` / `paper_positions` (Path A — Phase 2; needs operator BQ migration)
- Industry / market_cap fields on the meta response (Phase 2 — not user-requested)
- Real-time meta updates (24h cache is fine; company names rarely change)
- Sector-grouped P&L SQL queries (depends on Path A)

## Verification

The verification script asserts:
1. `get_ticker_meta` route is registered on the router
2. `_fetch_ticker_meta` helper is importable + callable
3. `_yfinance_ticker_info` returns the documented fallback shape on a fake/error ticker
4. Cache key construction is deterministic for sorted input

## Files modified

- `backend/api/paper_trading.py` — NEW endpoint + 2 helpers
- `backend/services/api_cache.py` — 1 new entry in `ENDPOINT_TTLS`
- `frontend/src/lib/types.ts` — NEW `TickerMeta` interface
- `frontend/src/lib/api.ts` — NEW `getTickerMeta()` + response type
- `frontend/src/lib/useTickerMeta.ts` — NEW hook (~40 lines)
- `frontend/src/app/paper-trading/page.tsx` — `useTickerMeta` call + 3 new column renders + colSpan adjustments
- `tests/api/test_ticker_meta.py` — NEW (7 tests)
- `tests/verify_phase_23_1_10.py` — NEW immutable verification script

## References

- `handoff/current/phase-23.1.10-research-brief.md` — full brief (386 lines, 3 sources read in full, gate_passed: true)
- `backend/services/autonomous_loop.py:481-482` — existing yfinance.info usage shows the pattern
- `backend/services/api_cache.py` — existing TTL cache helper
- `frontend/src/lib/useLivePrices.ts` — existing batch-by-tickers hook pattern
- `frontend/src/app/paper-trading/page.tsx` — Positions render around line 540, Trades render around line 630
