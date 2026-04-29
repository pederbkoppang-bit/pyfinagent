# Internal Codebase Audit — phase-23.1.16
## /paper-trading Positions tab latency: company_name + sector columns

---

## 1. Confirmed latency root causes

### 1a. Serial yfinance loop with sleep (PRIMARY)
File: `backend/api/paper_trading.py:728-742`

```python
# Step 2: yfinance fallback for tickers missing OR missing sector
import time
for t in tickers:
    existing = out.get(t)
    needs_yf = existing is None or not existing.get("sector")
    if not needs_yf:
        continue
    info = _yfinance_ticker_info(t)
    ...
    time.sleep(0.3)  # polite rate-limit guard
```

The loop is fully serial. Each iteration calls `_yfinance_ticker_info(t)` (line 675-685) which does `yf.Ticker(ticker).info` — a blocking HTTP round-trip to Yahoo Finance. Empirical cost: ~1s per ticker + 0.3s sleep = ~1.3s/ticker. With 14 tickers all missing sector in BQ: **~18s**.

`_fetch_ticker_meta` is declared `def` (sync), not `async def`. It is called at line 767 via `await asyncio.to_thread(_fetch_ticker_meta, raw, settings, bq)`, which is correct — it runs on a threadpool thread and does not block the event loop. But the serial-within-thread latency is the wall-clock problem.

### 1b. BQ `analysis_results` sector column is frequently NULL
File: `backend/api/paper_trading.py:693-694`

```
# company_name is well-populated; sector OFTEN NULL on this table.
```

The BQ step (lines 702-726) queries `analysis_results` with `ANY_VALUE(sector)`. The `analysis_results` table's sector column is populated from yfinance at analysis time (via `backend/db/bigquery_client.py:56`), but only when a full analysis run has been done for that ticker. Portfolio tickers that haven't been analyzed recently return NULL sector, forcing the yfinance fallback for every such ticker on every cold-cache hit.

### 1c. Set-level cache key busts on any position change
File: `backend/api/paper_trading.py:761-764`

```python
cache_key = f"paper:ticker_meta:{','.join(sorted(raw))}"
cached = cache.get(cache_key)
```

The cache key is a single string combining ALL sorted tickers. Adding or removing one position from the portfolio produces a new string, busting the entire 24h TTL cache for the set. This means every portfolio change (buy/sell) forces a full re-fetch of metadata for all tickers, including ones already resolved.

---

## 2. Cache layer: `backend/services/api_cache.py`

Read in full. Key findings:

- `APICache` is a `threading.Lock`-protected in-memory TTL dict (lines 26-101).
- `get()` and `set()` both acquire `self._lock` — thread-safe for concurrent reads/writes from multiple threads.
- `set(key, value, ttl_seconds)` is fully synchronous. No async wrapper needed.
- The module-level singleton `_api_cache` is shared across all route handlers (line 106).
- **Implication for Fix B (per-ticker keys)**: `cache.set(f"paper:ticker_meta:{ticker}", ...)` from multiple ThreadPoolExecutor threads is safe. The lock serializes writes; no race condition.
- `invalidate(pattern)` supports glob patterns (e.g., `"paper:ticker_meta:*"`) — useful if we need to flush all per-ticker keys on demand (line 59-73).
- TTL for `paper:ticker_meta` is 86400s (24h) per `ENDPOINT_TTLS` at line 132.

---

## 3. Startup hooks: `backend/main.py`

Read through the full `lifespan` context manager (lines 109-196). Findings:

- `lifespan` is an `@asynccontextmanager` registered at `FastAPI(lifespan=lifespan)` (line 203). This is the modern FastAPI startup hook — `@app.on_event("startup")` is deprecated but still works.
- On startup, lifespan currently runs: governance limits loader, APScheduler for paper trading cron, Slack monitor init, ticket queue processor APScheduler (lines 119-183).
- **The existing pattern for background work**: `asyncio.create_task(...)` or APScheduler. The paper trading scheduler is already started inside lifespan via `AsyncIOScheduler` (line 135).
- **Fix C integration point**: A prewarm task can be added at line ~183 (after the queue scheduler block, before `yield`). Pattern:
  ```python
  asyncio.create_task(_prewarm_ticker_meta())
  ```
  Where `_prewarm_ticker_meta` is an `async def` coroutine that fetches current positions and calls `_fetch_ticker_meta` via `asyncio.to_thread`, storing per-ticker results in the cache. Must be fire-and-forget (non-blocking) so it does not delay the server becoming ready.
- No existing prewarm task for ticker metadata exists.

---

## 4. `paper_positions` BQ schema — sector column status

Migration source: `scripts/migrations/migrate_paper_trading.py:36-51`

```python
PAPER_POSITIONS_SCHEMA = [
    SchemaField("position_id", "STRING", mode="REQUIRED"),
    SchemaField("ticker", "STRING", mode="REQUIRED"),
    SchemaField("quantity", "FLOAT64", mode="REQUIRED"),
    SchemaField("avg_entry_price", "FLOAT64", mode="REQUIRED"),
    SchemaField("cost_basis", "FLOAT64", mode="NULLABLE"),
    SchemaField("current_price", "FLOAT64", mode="NULLABLE"),
    SchemaField("market_value", "FLOAT64", mode="NULLABLE"),
    SchemaField("unrealized_pnl", "FLOAT64", mode="NULLABLE"),
    SchemaField("unrealized_pnl_pct", "FLOAT64", mode="NULLABLE"),
    SchemaField("entry_date", "STRING", mode="REQUIRED"),
    SchemaField("last_analysis_date", "STRING", mode="NULLABLE"),
    SchemaField("recommendation", "STRING", mode="NULLABLE"),
    SchemaField("risk_judge_position_pct", "FLOAT64", mode="NULLABLE"),
    SchemaField("stop_loss_price", "FLOAT64", mode="NULLABLE"),
]
```

**CONFIRMED: `paper_positions` has NO `sector` column.** The schema is 14 fields; sector is absent. `mfe_pct` and `mae_pct` were added later via `scripts/migrations/add_round_trip_schema.py:46` but sector was never added. Fix E (reading sector from paper_positions) is NOT viable without a schema migration. Recommendation: skip Fix E as caller requested.

Additionally, `autonomous_loop.py:320` has a comment: "BQ paper_positions rows predating the sector column migration" — suggesting a sector column migration was contemplated but never executed. This is a dormant TODO.

---

## 5. `BigQueryClient` paper_positions methods

File: `backend/db/bigquery_client.py:529-596`

- `get_paper_positions()` (line 531): `SELECT * FROM paper_positions` — returns all rows as dicts. Does NOT include sector (not in schema).
- `save_paper_position(row)` (line 549): MERGE on `ticker` natural key (phase-23.1.15). This is the MERGE pattern already in use for positions.
- `delete_paper_position()`, `update_paper_position()` also exist.
- No `ticker_meta`-style table or query exists in this file. The existing MERGE pattern at line 549-588 is directly analogous to what Fix D would need for a `ticker_meta` table.

---

## 6. Tests for `_fetch_ticker_meta`

File: `tests/api/test_ticker_meta.py` (136 lines, read in full)

Tests present:
1. `test_yfinance_info_handles_exception_gracefully` — error fallback
2. `test_yfinance_info_uses_short_name_when_present` — happy path
3. `test_yfinance_info_falls_through_to_long_name`
4. `test_yfinance_info_falls_through_to_ticker_when_no_name`
5. `test_fetch_ticker_meta_empty_input_returns_empty`
6. `test_fetch_ticker_meta_bq_hit_skips_yfinance` — BQ short-circuit
7. `test_fetch_ticker_meta_bq_missing_sector_falls_back_to_yfinance`
8. `test_fetch_ticker_meta_bq_query_failure_falls_back_to_yfinance`
9. `test_fetch_ticker_meta_response_shape`

All tests mock yfinance via `patch("yfinance.Ticker", ...)` and mock `bq.client.query`. No tests exercise the serial loop timing, per-ticker cache keying, or the concurrent ThreadPoolExecutor pattern. Tests for Fix A would need to mock `concurrent.futures.ThreadPoolExecutor` or verify that `_yfinance_ticker_info` is called concurrently. The existing test harness is compatible — `_fetch_ticker_meta` is a sync function, tests call it directly.

Also found: `tests/verify_phase_23_1_10.py`, `tests/verify_phase_23_1_13.py`, `tests/verify_phase_23_1_14.py` — phase-specific verification scripts for prior phases.

---

## 7. Call site inventory

`_fetch_ticker_meta` is called from two places:

| Call site | Context | Async wrapping |
|-----------|---------|---------------|
| `paper_trading.py:767` | HTTP route `GET /ticker-meta` | `await asyncio.to_thread(...)` |
| `paper_trading.py:184` (comment reference) | `run_daily_cycle` sector breakdown at portfolio level | `await asyncio.to_thread(...)` |

Both call sites already use `asyncio.to_thread`. Refactoring to `async def` is NOT required if we use `concurrent.futures.ThreadPoolExecutor` within the sync function body (Fix A option c). Using `asyncio.run()` inside the sync function would raise `RuntimeError: This event loop is already running` — confirmed dangerous in this context.

---

## Summary: Three confirmed latency causes

| Cause | File:Line | Magnitude |
|-------|-----------|-----------|
| Serial yfinance loop + sleep(0.3) | `paper_trading.py:728-742` | ~1.3s/ticker, ~18s for 14 |
| sector NULL in analysis_results BQ | `paper_trading.py:693-694` | Forces yfinance for all tickers |
| Set-level cache key (whole-set bust) | `paper_trading.py:761` | Any buy/sell resets 24h cache |

## Fix viability matrix

| Fix | Viable | Blocker | Notes |
|-----|--------|---------|-------|
| A — parallel yfinance via ThreadPoolExecutor | YES | None | Use sync-compatible concurrent.futures, not asyncio.gather |
| B — per-ticker cache keys | YES | None | APICache.set() is thread-safe; lock protects concurrent writes |
| C — startup prewarm | YES | None | lifespan hook at main.py:183; asyncio.create_task fires after yield is crossed |
| D — ticker_meta BQ persistence table | YES (Phase 2) | New table + migration needed | Google warns against single-row MERGE high-frequency on BQ OLAP |
| E — sector from paper_positions | NO | No sector column in schema | Migration needed; scope beyond phase-23.1.16 |
