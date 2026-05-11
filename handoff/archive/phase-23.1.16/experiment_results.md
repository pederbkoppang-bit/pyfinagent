---
step: phase-23.1.16
cycle_date: 2026-04-29
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_16.py'
---

# Experiment Results — phase-23.1.16

## Summary

User reported COMPANY + SECTOR columns showing "—" placeholders for
~15-20s after page load on /paper-trading. Three compounding causes
fixed in one cycle.

**Cause 1 — Serial yfinance loop with sleep.** `_fetch_ticker_meta`
(paper_trading.py:728-742) did `for t in tickers: ... yfinance ...
time.sleep(0.3)`. ~1.3s/ticker × 14 = ~18s wall clock when sector
NULL in BQ.

**Cause 2 — Set-level cache key.**
`f"paper:ticker_meta:{','.join(sorted(raw))}"` — adding/removing
one position busts the entire 24h cache.

**Cause 3 — No backend prewarm.** Every backend restart starts
fully cold; first user landing eats the full yfinance cost.

## Three coordinated fixes

**Fix A — Parallel yfinance via ThreadPoolExecutor**
(paper_trading.py:728-757). Replaced serial loop with
`ThreadPoolExecutor(max_workers=5)` + `as_completed`. Drop
`time.sleep(0.3)`. Each worker creates its own `Ticker` object
(thread-safe path; the bug in yfinance #2557 is in `download()`,
not `Ticker.info`). Empirical rate ceiling ~100 req/30s — 5
workers safe for 14-50 ticker batches.

**Fix B — Per-ticker cache keys** (paper_trading.py route
handler). New cache key shape:
`paper:ticker_meta:single:{TICKER}`. Route handler looks up each
ticker individually, fetches only the missing subset, and writes
each resolved ticker back to its own cache slot. Adding/removing
one position now leaves the other 13 cache hits intact.

**Fix C — Backend startup prewarm** (main.py lifespan). Added
`asyncio.create_task(_prewarm_ticker_meta())` before `yield` in
the lifespan context manager. Reads current paper_positions
tickers, calls `_fetch_ticker_meta`, writes to per-ticker cache
slots. Non-blocking — backend boots normally even if BQ /
yfinance is unavailable. Skipped when paper_positions is empty.

## Files modified

- `backend/api/paper_trading.py` (+30 lines: ThreadPoolExecutor block in
  `_fetch_ticker_meta`, per-ticker cache logic in `/ticker-meta` route)
- `backend/main.py` (+30 lines: `_prewarm_ticker_meta` task in lifespan)

## Files added

- `tests/api/test_ticker_meta_perf.py` (4 new tests)
- `tests/verify_phase_23_1_16.py` (immutable verification)

## Verification command output

```
$ source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_16.py
ok ThreadPoolExecutor parallel yfinance + per-ticker cache keys + lifespan prewarm + 4 new perf tests pass
```
Exit 0.

## Test results

```
$ pytest tests/api/test_ticker_meta_perf.py tests/api/test_ticker_meta.py -q
.............                                                            [100%]
13 passed in 2.85s
```
9 existing + 4 new = 13 ticker-meta tests green.

## Live measurement (post backend restart)

Backend log confirms prewarm fired:
```
20:40:18 I [main] Prewarming ticker-meta cache for 14 tickers...
20:40:21 I [main] Ticker-meta prewarm complete (14 resolved)
```
3-second wall clock for all 14 tickers (was ~18s serial).

cURL timing (after prewarm finished):
```
$ curl -o /dev/null -w "%{time_total}s" "/api/paper-trading/ticker-meta?tickers=<14 prewarmed>"
0.004684s        # 4.7ms — full cache hit

$ curl -o /dev/null -w "%{time_total}s" "/api/paper-trading/ticker-meta?tickers=ACME"
2.545972s        # single fresh yfinance fetch (no prewarm hit)
```

So for the user landing on /paper-trading **after** prewarm completes
(~3s after backend boot), all 14 columns populate within 5ms. Even
in the worst case (page load DURING prewarm), each missing ticker is
parallelized and the wall clock floor is ~one yfinance round-trip
(2.5s) instead of the previous 18s serial.

## Backwards compatibility

- Per-ticker cache keys are additive — a miss falls through to
  `_fetch_ticker_meta`, same return shape as before.
- ThreadPoolExecutor parallel fetch keeps the same `{ticker:
  {company_name, sector, source}}` return shape; callers see no
  API change.
- Prewarm failure is logged non-fatal — backend boots normally
  even if BQ or yfinance is unavailable.
- Old set-level cache key `paper:ticker_meta:{joined}` is no
  longer written; legacy entries (if any) will TTL out within
  24h.

## Honest disclosures

1. **Page load during prewarm** still incurs cold-fetch latency
   for un-prewarmed tickers. Fix C reduces the window where
   that happens (3s after boot) but doesn't eliminate it. A user
   refreshing within 3s of backend restart will still see "—"
   briefly.

2. **yfinance rate-limit ceiling is empirical**, not contractual.
   The ~100 req/30s ceiling comes from community reports
   (yfinance #2431). For the current 14-position portfolio, 5
   concurrent workers is well under the ceiling. If the
   portfolio scales to 50+ positions, may need to drop
   max_workers or stagger.

3. **Per-thread `Ticker` instances avoid the #2557 bug** because
   that issue is in `yf.download()` (shared `_DFS` global), not
   `Ticker.info` (per-instance). Confirmed via researcher code
   inspection. If yfinance ships a future fix that changes
   semantics, may need to re-verify.

4. **No frontend changes** — `useTickerMeta.ts` is unchanged.
   It still re-fetches the whole batch on key change. Future
   refinement (deferred to Phase 2): the hook could maintain
   per-ticker state and surface partial loading per row. Current
   user impact is minor since the per-ticker cache + prewarm
   keeps the slow path narrow.

## Phase 2 (deferred)

- Dedicated `ticker_meta` BQ table for cross-restart durability
  (researcher noted: single-row MERGE is BQ anti-pattern, must
  be batched multi-row MERGE). Useful when prewarm doesn't run
  or for entirely new tickers.
- Frontend per-ticker progressive rendering — surface partial
  loading state in `useTickerMeta` so the UI doesn't wait on
  the slowest fetcher.
- Stale-while-revalidate (SWR) on cache hits older than 12h —
  serve cached value immediately, kick off background refresh.
