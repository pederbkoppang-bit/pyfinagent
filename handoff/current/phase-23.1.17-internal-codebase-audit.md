# Phase-23.1.17 Internal Codebase Audit
# Home-page vs Paper-trading NAV Disagreement

Accessed: 2026-04-29

---

## 1. Home Page Hero Math — Line by Line

File: `frontend/src/app/page.tsx`

### Data fetching (lines 91-123)

The home page fires four parallel `Promise.allSettled` calls:

| Call | API endpoint | Sets state |
|------|-------------|------------|
| `listReports(5)` | `/api/reports` | `reports` |
| `getPaperTradingStatus()` | `/api/paper-trading/status` | `ptStatus` |
| `getPaperPortfolio()` | `/api/paper-trading/portfolio` | `positions` |
| `getPaperTrades(5)` | `/api/paper-trading/trades` | `trades` |

A separate `useEffect` (lines 125-139) calls `getSovereignRedLine(redLineWindow)` which hits `/api/sovereign/red-line?window=30d`.

### Hero card derivation (lines 141-153)

```
nav = ptStatus?.portfolio          // the entire portfolio sub-object from /status
navValue = nav?.nav                // portfolio.total_nav from BQ paper_portfolio row
pnl = nav?.pnl_pct                 // portfolio.total_pnl_pct from BQ paper_portfolio row
benchmark = nav?.benchmark_return_pct
alpha = pnl - benchmark            // vs SPY card

navSeries = redLineSeries.map((p) => ({ date: p.date, nav: p.nav }))
today = dailyDelta(navSeries)      // P&L (today) card — from RED LINE SERIES, not BQ
sharpe90 = kpiSharpe(navSeries)    // Sharpe (90d) card — from RED LINE SERIES
dd30 = maxDrawdownPct(navSeries)   // Max DD (30d) card — from RED LINE SERIES
```

**Critical split**: The NAV dollar card comes from `ptStatus.portfolio.nav` (BQ `paper_portfolio.total_nav`). The P&L-today, Sharpe, and MaxDD cards all come from the sovereign red-line series, which reads `financial_reports.paper_portfolio_snapshots`.

### What each card sources

| Card | Source | Freshness |
|------|--------|-----------|
| NAV ($14,153.03 in screenshot) | `paper_portfolio.total_nav` via `/api/paper-trading/status` | Updated only when `mark_to_market()` runs or a cash mutation writes `upsert_paper_portfolio` |
| P&L (today) | `dailyDelta(navSeries)` — last two rows of `paper_portfolio_snapshots` | Snapshot-age (written once per autonomous cycle) |
| vs SPY | `ptStatus.portfolio.pnl_pct - benchmark_return_pct` | Same BQ staleness as NAV |
| Sharpe (90d) | `kpiSharpe(navSeries)` from snapshot series | Snapshot-age |
| Max DD (30d) | `maxDrawdownPct(navSeries)` from snapshot series | Snapshot-age |
| Positions (count) | `positions.length` from `/api/paper-trading/portfolio` | Relatively fresh (per-request BQ scan) |

---

## 2. redLineSeries Source and Freshness

`getSovereignRedLine()` calls `/api/sovereign/red-line?window=<N>d`.

In `backend/api/sovereign_api.py` (lines 122-147), `_fetch_snapshots()` runs:

```sql
SELECT
  snapshot_date AS d,
  ANY_VALUE(total_nav) AS nav
FROM `sunny-might-477607-p8.financial_reports.paper_portfolio_snapshots`
WHERE PARSE_DATE('%Y-%m-%d', snapshot_date)
      >= DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY)
GROUP BY snapshot_date
ORDER BY snapshot_date
```

This reads from `financial_reports.paper_portfolio_snapshots`, NOT from `paper_portfolio` (which holds the live/mutated row).

**Key consequence**: The snapshot table only gets a new row when `save_daily_snapshot()` is called (line 449 of `paper_trader.py`), which in turn is only called by the autonomous loop at the end of a daily cycle. A manual cash mutation (`current_cash += $1,451.40`) does NOT write a new snapshot row; it only updates the `paper_portfolio` table. Therefore:

- The home page NAV tile reads `paper_portfolio.total_nav` — the latest value in the mutable portfolio row.
- The chart and derived KPIs (P&L today, Sharpe, Max DD) read `paper_portfolio_snapshots` — which does NOT have yesterday's cash refund reflected.
- The paper-trading page liveNav is computed from `cash + sum(livePrice * qty)` — which reads `paper_portfolio.current_cash` (post-refund) and live prices.

**Was total_nav recomputed after the $1,451.40 refund?** Almost certainly NO. The phase-23.1.15 cleanup script touched `current_cash` directly. `mark_to_market()` (line 384) recomputes `nav = portfolio["current_cash"] + total_positions_value` and writes it back via `upsert_paper_portfolio`. If `mark_to_market()` was NOT called after the cash bump, then `paper_portfolio.total_nav` still reflects the pre-refund sum. The paper-trading page sidesteps this by recomputing `liveNav = cash + sum(livePrice * qty)` inline — it reads the fresh `current_cash` even if `total_nav` is stale.

The NAV discrepancy ($14,153.03 home vs $15,664.69 paper-trading) = $1,511.66 is larger than the refund alone ($1,451.40). The delta suggests both a stale `total_nav` AND a live-price drift from the last MtM. This is expected: the paper-trading page always adds real-time live prices.

---

## 3. `paper_portfolio.total_nav` Refresh Cadence

`total_nav` in the `paper_portfolio` BQ table is written by `upsert_paper_portfolio()` in two code paths:

1. `mark_to_market()` — called during autonomous loop step 5. Updates every position with live prices, sums to `nav = current_cash + total_positions_value`, writes back.
2. Any explicit trade execution (buy/sell calls `_update_portfolio_cash` which writes `current_cash`; but does NOT rewrite `total_nav` separately — `total_nav` only gets updated on the `mark_to_market` call).
3. `depositPaperFunds` endpoint (paper_trading.py lines 808-854) — this IS a special path: it computes `new_nav = portfolio.total_nav + amount` and writes `total_nav` back. However, the phase-23.1.15 cleanup was a raw BQ UPDATE, not this endpoint.

**Conclusion**: After a raw SQL `UPDATE paper_portfolio SET current_cash = current_cash + 1451.40`, `total_nav` is stale. The only way to fix it is to call `mark_to_market()` or invoke the deposit endpoint (which adds to `total_nav` proportionally). The paper-trading page avoids this by computing liveNav fresh on every render cycle.

---

## 4. `/api/paper-trading/performance` Endpoint

File: `backend/api/paper_trading.py`, lines 253-299.

```python
portfolio = await asyncio.to_thread(bq.get_paper_portfolio, "default")
snapshots = await asyncio.to_thread(bq.get_paper_snapshots, limit=365)

sharpe = compute_sharpe_from_snapshots(snapshots)   # from snapshot series
pnl_pct = portfolio.get("total_pnl_pct", 0)        # from paper_portfolio row
bench_pct = portfolio.get("benchmark_return_pct", 0)
```

The Sharpe on this endpoint is computed from `paper_portfolio_snapshots`. The P&L uses `total_pnl_pct` from the mutable `paper_portfolio` row. Neither uses live-derived NAV. Critically, the paper-trading page's SummaryHero Sharpe is sourced from `perf?.sharpe_ratio` which comes from this endpoint — so it reflects the stale BQ snapshot series Sharpe, not the live-derived NAV Sharpe. This is why paper-trading page shows Sharpe -0.71: the snapshot series has the refund-corrupted history in it.

---

## 5. Paper-Trading Page useLiveNav Logic

File: `frontend/src/app/paper-trading/page.tsx`, lines 433-451.

```typescript
// liveNav useMemo (lines 433-443)
const liveNav = useMemo(() => {
  if (positions.length === 0) return status?.portfolio.nav ?? null;
  const cash = status?.portfolio.cash ?? 0;
  const hasAnyLive = positions.some((p) => livePrices[p.ticker]?.price != null);
  if (!hasAnyLive) return status?.portfolio.nav ?? null;
  const positionsValue = positions.reduce((sum, pos) => {
    const lp = livePrices[pos.ticker]?.price ?? pos.current_price ?? pos.avg_entry_price;
    return sum + lp * pos.quantity;
  }, 0);
  return cash + positionsValue;
}, [positions, livePrices, status]);

// liveTotalPnlPct useMemo (lines 445-451)
const liveTotalPnlPct = useMemo(() => {
  const startingCapital = status?.portfolio.starting_capital ?? null;
  if (liveNav == null || startingCapital == null || startingCapital <= 0) {
    return status?.portfolio.pnl_pct ?? null;
  }
  return ((liveNav - startingCapital) / startingCapital) * 100;
}, [liveNav, status]);
```

Dependencies:
- `status?.portfolio.cash` — from `/api/paper-trading/status` (reads `paper_portfolio.current_cash`, which IS post-refund)
- `status?.portfolio.starting_capital` — from `/api/paper-trading/status`
- `livePrices[ticker].price` — from `useLivePrices` hook (30s yfinance polling)
- `positions` — from `getPaperPortfolio()` 

**Can this be cleanly extracted into a shared hook?** Yes, with minor refactoring. The hook requires `positions`, `livePrices`, and `status` as inputs. It does not read BQ directly. Extraction into `frontend/src/lib/useLiveNav.ts` is straightforward. The hook would accept these three inputs and return `{ liveNav, liveTotalPnlPct }`. The home page would need to:
1. Import and call `useLivePrices` (already exists at `frontend/src/lib/useLivePrices.ts`)
2. Import and call `useLiveNav`
3. Use `liveNav` for the NAV hero tile instead of `navValue`

One complication: the home page currently imports `positions` from `getPaperPortfolio()`, so that fetch is already there. The `useLivePrices` import would add one new hook call.

---

## 6. Dead / Stale Code Observations

- `navValue = nav?.nav` (home page line 142) is read from `ptStatus.portfolio.nav`. This comes from the stale BQ `total_nav` column. After the paper-trading page was updated in phase-23.1.14 to use liveNav, the home page was NOT updated in parallel — this is the root of the divergence.
- The home page imports `getPaperPortfolio` (line 13) AND calls it (line 98) to populate `positions`, but those positions are only used for `posBreakdown` (line 154 / the Positions card). They are not used for NAV. This means all the ingredients for `useLiveNav` (positions, status) are already fetched on the home page.
- `useLivePrices` is not imported on the home page. It would need to be added.

---

## 7. Concrete Fix Sketches

### Fix A — Extract `useLiveNav` shared hook

**File to create**: `frontend/src/lib/useLiveNav.ts`

**Anchor in paper-trading/page.tsx**: lines 433-451 — the two `useMemo` blocks.

**Sketch**:
```typescript
import { useMemo } from "react";
import type { PaperPosition, PaperTradingStatus } from "@/lib/types";

export function useLiveNav(
  positions: PaperPosition[],
  livePrices: Record<string, { price: number | null; age_sec: number | null } | undefined>,
  status: PaperTradingStatus | null,
): { liveNav: number | null; liveTotalPnlPct: number | null } {
  const liveNav = useMemo(() => {
    if (positions.length === 0) return status?.portfolio.nav ?? null;
    const cash = status?.portfolio.cash ?? 0;
    const hasAnyLive = positions.some((p) => livePrices[p.ticker]?.price != null);
    if (!hasAnyLive) return status?.portfolio.nav ?? null;
    const positionsValue = positions.reduce((sum, pos) => {
      const lp = livePrices[pos.ticker]?.price ?? pos.current_price ?? pos.avg_entry_price;
      return sum + lp * pos.quantity;
    }, 0);
    return cash + positionsValue;
  }, [positions, livePrices, status]);

  const liveTotalPnlPct = useMemo(() => {
    const startingCapital = status?.portfolio.starting_capital ?? null;
    if (liveNav == null || startingCapital == null || startingCapital <= 0)
      return status?.portfolio.pnl_pct ?? null;
    return ((liveNav - startingCapital) / startingCapital) * 100;
  }, [liveNav, status]);

  return { liveNav, liveTotalPnlPct };
}
```

**For/against**:
- FOR: eliminates the divergence at the source. One place to change.
- FOR: paper-trading/page.tsx can replace its inline `useMemo` blocks with the shared hook with zero behavior change.
- AGAINST: adds one new file; minimal concern.
- AGAINST: home page still needs `useLivePrices` import added (one line).

### Fix B — Backfill snapshot row for 2026-04-29

Write a fresh row to `paper_portfolio_snapshots` with the post-refund NAV via `save_daily_snapshot()`. This makes the sovereign red-line chart and the home-page derived KPIs (Sharpe, MaxDD, P&L today) reflect the corrected value.

**For/against**:
- FOR: fixes the chart and all downstream derived stats.
- FOR: corrects the historical record; future Sharpe computations will no longer be distorted by the gap.
- AGAINST: does not fix the home-page NAV tile (which reads `total_nav`, not snapshots) unless `mark_to_market` is also run.
- AGAINST: writing a snapshot row with an artificial NAV (not freshly computed from live prices) may introduce a new inconsistency if done without first running MtM.

**Correct sequence**: run `mark_to_market()` first (updates `total_nav` in `paper_portfolio`), THEN call `save_daily_snapshot()` (which reads `total_nav` from the freshly-updated row). This gives a coherent snapshot.

### Fix C — Make `/api/paper-trading/status.portfolio.nav` return live-derived value

In `paper_trading.py` lines 115-153, the status endpoint reads `bq.get_paper_portfolio("default")` and returns `portfolio.get("total_nav")` raw. To return live NAV instead, the endpoint would need to:
1. Call `trader.mark_to_market()` (or a lighter price-fetch-only variant) on every request
2. Return the computed `nav` from that call

**For/against**:
- FOR: every consumer of `/status` gets the same live NAV. Home page, paper-trading page, any future consumer — all agree.
- FOR: cleanest architectural fix long-term.
- AGAINST: `mark_to_market()` calls `_get_live_price(ticker)` for each position, which hits yfinance. With 5-10 positions, that's 5-10 yfinance fetches per `/status` call. The status endpoint is polled by `OpsStatusBar` on every page.
- AGAINST: yfinance calls are already batched in `useLivePrices` on the frontend (30s TTL); doing it again on the backend on every status call is redundant and expensive.
- MITIGATION: a cached lite variant (e.g., use the last known live prices from the 30s frontend polling) could work but requires wiring.
- VERDICT: architecturally correct but latency cost is non-trivial for a polled endpoint. More appropriate as a post-phase optimization.

### Fix D — Trigger mark_to_market in cash-mutating scripts

The phase-23.1.15 cleanup script mutated `current_cash` directly via BQ UPDATE. Adding a `mark_to_market()` call at the end of any script that touches cash would ensure `total_nav` stays in sync.

**For/against**:
- FOR: prevents this class of bug in future cleanup scripts.
- FOR: low cost (one Python call).
- AGAINST: does not fix the current stale state; requires also running MtM NOW.
- VERDICT: good prophylactic; should be documented in a comment in any future cleanup script.

### Fix E — Make home-page NAV tile use live-derived value

Replace `navValue = nav?.nav` (home page line 142) with `liveNav` from the shared hook (Fix A). Use the same Sharpe/P&L tile values as paper-trading page — i.e., derived from the live-computed NAV rather than the snapshot series.

**For/against**:
- FOR: pages agree by construction; both read from the same code path.
- FOR: eliminates the two-tier data source split (BQ snapshot vs live price).
- AGAINST: home page does not currently poll `useLivePrices`; adding it means a new 30s background polling interval on the home page.
- VERDICT: worth it. The polling is cheap (yfinance batch for 5-10 tickers) and the cost of user confusion from disagreeing dashboards is higher.

### Recommended combination

**A + E + B (with MtM first)**

1. **Fix A**: Extract `useLiveNav` into `frontend/src/lib/useLiveNav.ts`.
2. **Fix E**: Import `useLivePrices` and `useLiveNav` on home page; replace `navValue` with `liveNav` for the NAV tile.
3. **Fix B** (sequenced): Run `mark_to_market()` in a one-shot Python call, THEN call `save_daily_snapshot()` to write a corrected 2026-04-29 row. This aligns the snapshot table with the corrected live state.
4. **Fix D** (prophylactic): Add a `# NOTE: call mark_to_market() after mutating current_cash` comment to any future cleanup script.

Fix C (backend live-derived status endpoint) is the architecturally purest long-term answer but should be a separate phase due to the latency implications on a polled endpoint. It is noted as a future improvement.

**Against the preference for "A + E + F"**: Fix F was described as "regenerate the snapshot row for 2026-04-29." This is Fix B above. The ordering matters: MtM must precede snapshot write. If MtM is skipped, the snapshot row gets a stale `total_nav` value from `paper_portfolio` — the same bug just written to a different table. The correct sequence is: MtM -> snapshot write -> then Fix E updates the home page to use live values going forward.

---

## 8. File-Line Anchor Summary

| Claim | File | Lines |
|-------|------|-------|
| Home page NAV reads `ptStatus.portfolio.nav` (BQ `total_nav`) | `frontend/src/app/page.tsx` | 141-142 |
| Home page Sharpe/MaxDD read from `redLineSeries` (snapshot table) | `frontend/src/app/page.tsx` | 149-153 |
| Sovereign red-line endpoint reads `paper_portfolio_snapshots` | `backend/api/sovereign_api.py` | 122-147 |
| `mark_to_market()` computes `nav = cash + positions_value` | `backend/services/paper_trader.py` | 384 |
| `mark_to_market()` writes result to `paper_portfolio.total_nav` | `backend/services/paper_trader.py` | 389-395 |
| `save_daily_snapshot()` reads `portfolio.total_nav` to write snapshot | `backend/services/paper_trader.py` | 426 |
| `/status` endpoint returns `portfolio.get("total_nav")` raw | `backend/api/paper_trading.py` | 136 |
| `/performance` Sharpe computed from `paper_portfolio_snapshots` | `backend/api/paper_trading.py` | 261-273 |
| Paper-trading page liveNav useMemo | `frontend/src/app/paper-trading/page.tsx` | 433-443 |
| Paper-trading page liveTotalPnlPct useMemo | `frontend/src/app/paper-trading/page.tsx` | 445-451 |
| SummaryHero falls back to `status?.portfolio.nav` if no live prices | `frontend/src/app/paper-trading/page.tsx` | 192-193 |
