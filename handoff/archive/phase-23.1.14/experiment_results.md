---
step: phase-23.1.14
cycle_date: 2026-04-29
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_14.py'
---

# Experiment Results — phase-23.1.14

## Summary

Two coordinated bug fixes shipped under one harness cycle.

**Bug A (sector cap blind to legacy positions) — fixed.** Existing 11
positions in BQ `paper_positions` predate the sector column, so
`pos.get("sector")` returned None for all of them, sending every
legacy row into the "Unknown" bucket. New Tech BUYs (MU, KEYS) were
trivially passing the `paper_max_per_sector=2` cap.

**Fix site:** `backend/services/autonomous_loop.py` between
`positions = trader.get_positions()` (post-MTM refresh) and
`orders = decide_trades(...)`. New block:
1. Computes `legacy_tickers` — positions whose `sector` field is empty.
2. Calls `_fetch_ticker_meta` via `asyncio.to_thread` (same pattern
   already used at line 179 for top-N candidate enrichment;
   `_fetch_ticker_meta` is sync def with 24h cache, BQ-first /
   yfinance-fallback).
3. Mutates each position dict with the resolved sector before passing
   to `decide_trades`.
4. Skipped when `paper_max_per_sector == 0` (zero extra cost when
   cap disabled).
5. Best-effort: failure logged non-fatal, cycle continues.

**Bug B (stale NAV scoreboards) — fixed.** Hero metric cards (NAV,
Total P&L) read `status?.portfolio.nav` / `pnl_pct` — both BQ snapshot
fields updated only at end-of-cycle. Position table immediately below
already derived live values from `useLivePrices` on every 30s
yfinance tick, producing a $329.49 visible discrepancy
($13,952.25 hero vs $14,281.74 table sum).

**Fix sites in `frontend/src/app/paper-trading/page.tsx`:**
1. `useLivePrices` — lifted the `tab === "positions"` gate. Live ticks
   now flow regardless of which tab is active.
2. New `useMemo` for `liveNav = cash + sum(livePrice * qty)` and
   `liveTotalPnlPct = (liveNav - starting_capital) / starting_capital
   * 100`. Falls back to BQ snapshot when no ticks are available
   (initial paint, empty positions).
3. `SummaryHero` extended with `liveNav` + `liveTotalPnlPct` props.
   Renders them when present; falls back to BQ snapshot otherwise.
   Cash, Sharpe, Positions stay from BQ snapshot (correct — they
   don't change between ticks).

## Files modified

- `backend/services/autonomous_loop.py` (+38 lines: legacy-position
  sector enrichment block before decide_trades)
- `frontend/src/app/paper-trading/page.tsx` (+30 lines: lifted
  useLivePrices gate, two useMemo hooks, SummaryHero props)

## Files added

- `tests/services/test_sector_concentration.py` (2 new tests)
- `tests/verify_phase_23_1_14.py` (immutable verification, exits 0
  with one ok-line covering 5 distinct claims)

## Verification command output

```
$ source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_14.py
ok autonomous_loop legacy-position sector enrichment + page.tsx live-derived NAV/Total-P&L scoreboards + useLivePrices gate lifted + 2 new sector-concentration tests pass
```

Exit 0.

## Test results

```
$ pytest tests/services/test_sector_concentration.py tests/services/test_screener_sector_propagation.py -q
............                                                             [100%]
12 passed in 0.32s
```

8 tests in `test_sector_concentration.py` (6 from phase-23.1.13 +
2 new for phase-23.1.14) + 4 tests in `test_screener_sector_propagation.py`
all green.

## Frontend type-check

```
$ cd frontend && npx tsc --noEmit
(silent, exit 0)
```

## Backwards compatibility

- `paper_max_per_sector=0` short-circuits the new enrichment block
  (zero extra yfinance calls when cap disabled).
- When `livePrices` is empty (initial paint, no positions), `liveNav`
  falls back to `status?.portfolio.nav` — no UI regression.
- `SummaryHero` accepts `liveNav: null` / `liveTotalPnlPct: null`
  and renders the BQ snapshot as before.
- 24h `_fetch_ticker_meta` cache means subsequent cycles incur
  near-zero overhead for already-resolved tickers.

## What we did NOT change

- Did not migrate the BQ `paper_positions` schema to add a `sector`
  column. The runtime enrichment is the bridge until a future schema
  migration; cost is bounded by the 24h cache.
- Did not derive Sharpe live — Sharpe needs a return series, so it
  stays from the BQ snapshot. This is consistent with the position
  table which also doesn't show a live Sharpe.
- Did not derive `Cash` live — cash only changes at trade execution,
  which still produces a BQ snapshot update.
- Did not refactor `decide_trades` to be async — would have ripple
  effects across the autonomous_loop call graph. The sync wrapper
  via `asyncio.to_thread` in the caller is the cleaner path
  (confirmed by the external research brief: BBC Engineering +
  Sentry FastAPI both recommend `asyncio.to_thread` over caller
  refactor for one-off bridges).

## Honest disclosures

1. **Cannot directly verify Bug A fix in production until tomorrow's
   cycle.** The autonomous loop runs daily at market open; today's
   cycle already completed. I can confirm the code path is correct
   via unit tests + verification script, but the live "tomorrow's
   MU + KEYS BUYs are blocked because 11 Tech positions counted
   correctly" assertion only runs at next cycle.

2. **`_fetch_ticker_meta` cache is in-memory** (`api_cache`). If the
   backend restarts mid-day, the next cycle pays the BQ-first /
   yfinance-fallback latency for legacy tickers (one-time cost,
   then re-cached for 24h).

3. **Bug B fix uses `livePrices[t].price ?? pos.current_price ??
   pos.avg_entry_price` fallback chain** — when a single ticker has
   no live price, it gracefully degrades to the last-known BQ price.
   This means `liveNav` always reflects the freshest available data
   per ticker, not all-or-nothing.

4. **`tab === "positions"` gate lifted means yfinance ticks fire on
   every tab.** Cost: at most `positions.length` extra polls per 30s
   when user is on Manage / Trades / Reality-gap tabs. Bounded by
   `paper_max_positions` setting (default 12). Net cost: negligible
   (yfinance is free).

## Phase 2 (deferred)

- BQ schema migration: add `sector` column to `paper_positions`,
  backfill from yfinance, drop the runtime enrichment block as
  scaffolding.
- Live Sharpe via rolling-window return series from
  `paper_snapshots`.
- Risk Monitor live sector concentration (already partially live in
  phase-23.1.13 via tickerMeta — could surface real-time alerts when
  a sector exceeds the cap mid-day).
