---
step: phase-23.1.16
title: ticker-meta latency fix (parallel yfinance + per-ticker cache + startup prewarm)
cycle_date: 2026-04-29
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_16.py'
research_brief: handoff/current/phase-23.1.16-external-research.md (also see phase-23.1.16-internal-codebase-audit.md)
---

# Contract — phase-23.1.16

## Hypothesis

User reports COMPANY + SECTOR columns on the Positions tab show
"—" placeholders for ~15-20s after page load. Three compounding
causes confirmed in the internal audit:

1. **Serial yfinance loop** in `_fetch_ticker_meta`
   (`backend/api/paper_trading.py:728-742`): `for t in tickers: ...
   yfinance.Ticker(t).info ... time.sleep(0.3)`. Per-ticker
   ~1.3s × 14 tickers = ~18s wall clock.
2. **Sector NULL in `analysis_results`**: BQ Step 1 returns
   `company_name` but `sector` is usually NULL, forcing yfinance
   fallback for every ticker on cold cache.
3. **Set-level cache key**: `f"paper:ticker_meta:{','.join(sorted(raw))}"`
   — adding/removing one position busts the 24h cache for the
   entire ticker set.

If we (A) parallelize the yfinance loop with a bounded
ThreadPoolExecutor (max_workers=5), (B) switch to per-ticker
cache keys so partial cache hits work, and (C) prewarm the cache
at backend startup for current paper_positions tickers, then the
first user landing on the page after a backend restart sees
populated COMPANY + SECTOR columns within 1-2s instead of 15-20s,
and ongoing position changes incur near-zero refetch cost.

## Research-gate summary

- External brief: `handoff/current/phase-23.1.16-external-research.md`
  — 10 sources read in full (yfinance rate limiting #2431,
  thread-safety #2557, Sling Academy yfinance practices, FastAPI
  SWR patterns, ThreadPoolExecutor best practices, BQ MERGE/upsert
  guidance, BQ compute best practices, yfinance DeepWiki multi-ticker,
  FastAPI middleware tricks, FastAPI edge-caching). 22 URLs
  collected. Recency scan 2024-2026 performed. `gate_passed: true`.
- Internal audit: `handoff/current/phase-23.1.16-internal-codebase-audit.md`
  — 7 files inspected with file:line anchors and concrete patch
  sketches.

Key findings:
- yfinance has no real batch-info API; ThreadPoolExecutor with
  separate `Ticker` instances per thread is the canonical
  parallelism pattern. Empirical rate ceiling ~100 requests /
  30s. `max_workers=5` is safe for 14-50 ticker batches.
- `APICache` is thread-safe (threading.Lock), so per-ticker
  concurrent writes are safe.
- `backend/main.py` already uses `lifespan` async context manager
  with `asyncio.create_task` patterns — natural place to attach
  the prewarm.
- `paper_positions` BQ schema has NO sector column → Fix E
  (read-from-positions) skipped.
- Researcher recommends A + B + C for phase-23.1.16. Defer Fix D
  (dedicated `ticker_meta` BQ table) to Phase 2 — single-row
  MERGE is BQ anti-pattern (per Google Cloud docs); if pursued
  later, must be a batched multi-row MERGE.

## Plan steps

1. **Fix A — Parallel yfinance via ThreadPoolExecutor.** In
   `backend/api/paper_trading.py:728-742` replace the serial
   loop with `ThreadPoolExecutor(max_workers=5)` + `as_completed`.
   Drop `time.sleep(0.3)`. Keep the function `def` (sync) — the
   pool is created/destroyed inside the function, no event-loop
   coupling. Both call sites (`/ticker-meta` route and
   `autonomous_loop`) already wrap via `asyncio.to_thread`.

2. **Fix B — Per-ticker cache keys.** In the `/ticker-meta`
   route handler, look up each ticker individually from the
   cache; collect a `missing` list; call `_fetch_ticker_meta`
   only for the missing subset; merge results; write each
   resolved ticker back to its own cache slot. Cache key shape:
   `paper:ticker_meta:single:{TICKER}` with the same 24h TTL.

3. **Fix C — Startup prewarm.** In `backend/main.py` lifespan,
   after backend boot, fire `asyncio.create_task(_prewarm_ticker_meta())`
   that reads current `paper_positions` tickers and calls
   `_fetch_ticker_meta` to warm the cache. Non-blocking; failure
   logged non-fatal. Skipped if `paper_positions` is empty.

4. **Tests** (`tests/api/test_ticker_meta_perf.py`): 3 new tests:
   - Per-ticker cache hits return without calling
     `_fetch_ticker_meta` for the cached subset.
   - ThreadPoolExecutor max_workers cap is respected (count
     concurrent calls).
   - Prewarm task short-circuits when `paper_positions` is
     empty.

5. **Immutable verification** (`tests/verify_phase_23_1_16.py`):
   greps for the ThreadPoolExecutor block, per-ticker cache
   key shape, prewarm task, runs the new tests.

## Immutable verification command

```bash
source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_16.py
```

Must exit 0 with one ok-line.

## Acceptance criteria

- `pytest tests/api/test_ticker_meta.py tests/api/test_ticker_meta_perf.py -q` passes.
- `python tests/verify_phase_23_1_16.py` exits 0.
- `cd frontend && npx tsc --noEmit` exit 0.
- Manual smoke: backend restart → wait 5s → GET /api/paper-trading/ticker-meta?tickers=<14-tickers> returns within 3-4s on cold cache (down from 15-20s).
- Backend startup logs show "Prewarming ticker-meta cache for N tickers..." line.

## Backwards compatibility

- Per-ticker cache keys land alongside the legacy set-level key — a
  miss on the per-ticker key still falls through to `_fetch_ticker_meta`.
- ThreadPoolExecutor parallel fetch keeps the same return shape;
  callers see no API change.
- Prewarm failure is logged non-fatal — backend boots normally even
  if BQ or yfinance is unavailable.

## References

- `handoff/current/phase-23.1.16-external-research.md`
- `handoff/current/phase-23.1.16-internal-codebase-audit.md`
- `backend/api/paper_trading.py:670-769` (the slow path)
- `backend/services/api_cache.py` (APICache thread-safe)
- `backend/main.py:109-196` (lifespan with asyncio.create_task patterns)
