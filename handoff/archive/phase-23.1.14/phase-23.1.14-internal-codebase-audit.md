# Internal Codebase Audit — phase-23.1.14

Two coordinated bugs: Bug A (sector cap blind to legacy positions) and Bug B (stale NAV scoreboards).

---

## Internal Code Inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/services/portfolio_manager.py` | 319 | Sell-first-then-buy trade decider, sector cap enforcer | Active — Bug A site |
| `backend/services/autonomous_loop.py` | 808 | Daily cycle orchestrator, caller of decide_trades | Active — Bug A upstream fix site |
| `backend/api/paper_trading.py` | ~850 | REST API, `_fetch_ticker_meta` helper (sync, BQ+yfinance, 24h cache) | Active — reference for fix |
| `frontend/src/app/paper-trading/page.tsx` | ~1000+ | Paper trading dashboard, hero cards, position table with useLivePrices | Active — Bug B site |
| `tests/services/test_sector_concentration.py` | 145 | 6 existing sector-cap tests (phase-23.1.13) | Active — new tests needed |

---

## Bug A — Sector Cap Blind to Legacy Positions

### Exact location of the broken loop

`backend/services/portfolio_manager.py`, lines 194-200:

```python
max_per_sector = int(getattr(settings, "paper_max_per_sector", 0) or 0)
sector_counts: dict[str, int] = {}
if max_per_sector > 0:
    for pos in current_positions:
        if pos["ticker"] in selling_tickers:
            continue
        s = (pos.get("sector") or "").strip() or "Unknown"   # line 199
        sector_counts[s] = sector_counts.get(s, 0) + 1
```

Line 199 reads `pos.get("sector")`. The 11 legacy BQ rows in `paper_positions` have no `sector` column populated, so every legacy position falls into `"Unknown"`. `sector_counts["Technology"]` is therefore 0, and new Tech BUY candidates at line 212 trivially pass the `>= max_per_sector` guard.

### The fix — two options, Option B is strongly preferred

**Option A (fix inside decide_trades — requires sync-to-async bridge):**

`decide_trades` is sync (plain `def`). `_fetch_ticker_meta` is also sync (plain `def`), so actually this bridge concern only arises because `autonomous_loop.py::run_daily_cycle` is an `async def` and cannot pass an awaited result into a sync call. The cleaner and safer fix is **Option B**.

**Option B (enrich current_positions in the async caller, pass pre-resolved sector_lookup — PREFERRED):**

In `autonomous_loop.py`, at Step 6 "Decide trades" (line 317-329), positions are already refreshed. The loop is async. Add an enrichment call analogous to the existing candidate enrichment at lines 175-192:

```python
# After: positions = trader.get_positions()  (line 317)
# Before: orders = decide_trades(...)         (line 322)

# Phase-23.1.14: build sector_lookup for legacy positions lacking sector.
# _fetch_ticker_meta is sync; asyncio.to_thread is the correct bridge here.
legacy_tickers = [
    p["ticker"] for p in positions
    if not (p.get("sector") or "").strip()
]
if legacy_tickers and max_per_sector_enabled(settings):
    try:
        from backend.api.paper_trading import _fetch_ticker_meta
        meta_response = await asyncio.to_thread(
            _fetch_ticker_meta, legacy_tickers, settings, bq,
        )
        meta_map = (meta_response or {}).get("meta", {})
        for p in positions:
            if not (p.get("sector") or "").strip():
                sector = (meta_map.get(p["ticker"], {}) or {}).get("sector") or ""
                if sector:
                    p["sector"] = sector
    except Exception as e:
        logger.warning("Legacy position sector enrichment failed (non-fatal): %s", e)
```

Then pass the already-enriched `positions` list (with sector fields filled in) to `decide_trades`. **No change to `decide_trades` signature or internals needed.** The existing line 199 loop in `portfolio_manager.py` reads `pos.get("sector")` which will now be populated for legacy positions.

Helper function `max_per_sector_enabled`:

```python
def max_per_sector_enabled(settings) -> bool:
    return int(getattr(settings, "paper_max_per_sector", 0) or 0) > 0
```

Or inline as `int(getattr(settings, "paper_max_per_sector", 0) or 0) > 0`.

**Why Option B is preferred over Option A:**
- `decide_trades` stays sync — no event loop concern, no breaking change.
- The enrichment pattern is identical to the existing candidate enrichment at autonomous_loop.py:175-192 — same `asyncio.to_thread` call, same `meta_map` access, same non-fatal guard. Zero new idioms.
- The fix is in the async orchestrator where `bq` is already constructed and in scope.
- `_fetch_ticker_meta` has a 24h internal cache (api_cache keyed by sorted tickers) so subsequent cycles incur near-zero latency for already-resolved tickers.

**Do NOT use `asyncio.run()` inside `decide_trades`** — it would raise `RuntimeError: This event loop is already running` because `run_daily_cycle` is an `async def` context. The Python event loop docs (docs.python.org/3) confirm: `asyncio.run()` raises `RuntimeError` if called inside an already-running loop.

### Backward compatibility

When all positions already have `sector` populated (future cycles after the schema migration), `legacy_tickers` will be empty and the enrichment block is a no-op. Zero overhead.

### Async-in-sync bridge summary (for decide_trades specifically)

`decide_trades` is and should remain a plain `def`. It is called from `run_daily_cycle` which is `async`. The correct bridge when you need to call an async helper from inside a sync function that is itself called from an async context is:
- If the sync function is called via `asyncio.to_thread`, the thread has no running event loop — `asyncio.run()` would work there but is heavyweight.
- The cleanest solution (and the one already used in this codebase) is to resolve all async data **before** calling the sync function, in the async caller. This is what Option B does.

---

## Bug B — Stale NAV Scoreboards

### Hero card component location

`frontend/src/app/paper-trading/page.tsx`, lines 177-199 (`SummaryHero` function):

```tsx
function SummaryHero({ status, perf }: ...) {
  const pnl = status?.portfolio.pnl_pct ?? 0;
  const bench = status?.portfolio.benchmark_return_pct ?? 0;
  return (
    <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
      <MetricCard label="NAV"><Dollar value={status?.portfolio.nav} /></MetricCard>       // line 188: stale BQ value
      <MetricCard label="Cash"><Dollar value={status?.portfolio.cash} /></MetricCard>      // line 189: stale
      <MetricCard label="Total P&L"><PnlBadge value={status?.portfolio.pnl_pct} /></MetricCard>  // line 190: stale
      <MetricCard label="vs SPY"><PnlBadge value={pnl - bench} /></MetricCard>
      ...
    </div>
  );
}
```

All three stale fields (`nav`, `cash`, `pnl_pct`) read from `status?.portfolio` which comes from the BQ snapshot row updated only at cycle end.

### Live-price data already available at the hero level

`useLivePrices` is already wired at the page level, lines 411-415:

```tsx
const positionTickers = useMemo(() => positions.map((p) => p.ticker), [positions]);
const { prices: livePrices } = useLivePrices(
  positionTickers,
  tab === "positions" && positions.length > 0,
);
```

**Problem:** `useLivePrices` is enabled only when `tab === "positions"` (line 414). The `SummaryHero` is rendered regardless of which tab is active (line 746). So when the user is on the Trades tab, `livePrices` will be empty `{}` and the hero NAV will always be stale.

### Live Market Value derivation pattern (already works in position table)

Lines 791-801 in the position table loop show the exact live-derivation idiom:

```tsx
const livePrice = live?.price ?? null;
const liveMarketValue =
  livePrice != null ? livePrice * pos.quantity : pos.market_value;
const liveCostBasis =
  pos.cost_basis != null && pos.cost_basis > 0
    ? pos.cost_basis
    : pos.avg_entry_price * pos.quantity;
const livePnlPct = livePrice != null && liveCostBasis > 0
  ? ((livePrice * pos.quantity - liveCostBasis) / liveCostBasis) * 100
  : pos.unrealized_pnl_pct;
```

### Minimal patch to fix Bug B

**Step 1:** Change the `useLivePrices` enable condition to always-on when positions exist:

```tsx
// Line 413-415: change from:
const { prices: livePrices } = useLivePrices(
  positionTickers,
  tab === "positions" && positions.length > 0,
);
// To:
const { prices: livePrices } = useLivePrices(
  positionTickers,
  positions.length > 0,   // always poll when positions exist, not just on Positions tab
);
```

**Step 2:** Compute `liveNav` once in the page component body (after `positions` and `livePrices` are available), before the JSX return:

```tsx
// Derive live NAV from current tick prices (matches position table derivation)
// Falls back to status?.portfolio.nav when no live prices available.
const liveNav = useMemo(() => {
  if (positions.length === 0) return status?.portfolio.nav ?? null;
  const cash = status?.portfolio.cash ?? 0;
  const positionsValue = positions.reduce((sum, pos) => {
    const live = livePrices[pos.ticker];
    const price = live?.price ?? pos.current_price ?? pos.avg_entry_price;
    return sum + price * pos.quantity;
  }, 0);
  const nav = cash + positionsValue;
  // Guard: only use live NAV when we have at least one live price tick
  const hasLiveTick = positions.some((p) => livePrices[p.ticker]?.price != null);
  return hasLiveTick ? nav : (status?.portfolio.nav ?? null);
}, [positions, livePrices, status]);

const startingCapital = status?.portfolio.starting_capital ?? 10000;
const liveTotalPnlPct = liveNav != null
  ? ((liveNav - startingCapital) / startingCapital) * 100
  : (status?.portfolio.pnl_pct ?? null);
```

**Step 3:** Pass `liveNav` and `liveTotalPnlPct` to `SummaryHero`:

```tsx
function SummaryHero({
  status,
  perf,
  liveNav,
  liveTotalPnlPct,
}: {
  status: PaperTradingStatus | null;
  perf: PaperPerformance | null;
  liveNav: number | null;
  liveTotalPnlPct: number | null;
}) {
  const bench = status?.portfolio.benchmark_return_pct ?? 0;
  const pnl = liveTotalPnlPct ?? 0;
  return (
    <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
      <MetricCard label="NAV"><Dollar value={liveNav} /></MetricCard>
      <MetricCard label="Cash"><Dollar value={status?.portfolio.cash} /></MetricCard>
      <MetricCard label="Total P&L"><PnlBadge value={liveTotalPnlPct} /></MetricCard>
      <MetricCard label="vs SPY"><PnlBadge value={pnl - bench} /></MetricCard>
      <MetricCard label="Sharpe">
        <span className="text-slate-200">{perf?.sharpe_ratio?.toFixed(2) ?? "—"}</span>
      </MetricCard>
      <MetricCard label="Positions">
        <span className="text-slate-200">{status?.position_count ?? 0}</span>
      </MetricCard>
    </div>
  );
}
```

And update the call site (line 746):

```tsx
<SummaryHero status={status} perf={perf} liveNav={liveNav} liveTotalPnlPct={liveTotalPnlPct} />
```

**Sharpe stays stale** — correctly, because Sharpe needs the return series, not just a spot price.
**Cash stays stale** — cash only changes after a trade executes; the stale BQ cash value is correct between cycles.

### Discrepancy this fixes

The described discrepancy ($13,952.25 BQ NAV vs $14,281.74 live position MV sum) occurs because `status?.portfolio.nav` is a stale BQ snapshot. After the patch, `liveNav` = `status.portfolio.cash` + `sum(livePrice * quantity)` = live-accurate value on every 30s poll.

---

## New Tests for `tests/services/test_sector_concentration.py`

Two tests to propose (append to existing file):

### Test 1: legacy position without sector gets correctly attributed via pre-resolved lookup

```python
def test_legacy_position_without_sector_gets_enriched_before_decide():
    """Simulate the upstream enrichment: a legacy position dict whose sector
    is empty-string gets its sector field filled in before decide_trades is
    called. After enrichment the cap is correctly applied."""
    # Legacy position with no sector (simulates BQ row lacking sector column)
    existing = [
        {"ticker": "NVDA", "sector": "", "quantity": 10,
         "current_price": 500, "avg_entry_price": 500,
         "market_value": 5000, "recommendation": "BUY"},
        {"ticker": "INTC", "sector": "", "quantity": 5,
         "current_price": 30, "avg_entry_price": 30,
         "market_value": 150, "recommendation": "BUY"},
    ]
    # Simulate enrichment: autonomous_loop fills in sector from _fetch_ticker_meta
    for p in existing:
        if not p.get("sector"):
            p["sector"] = "Technology"  # what _fetch_ticker_meta returns

    # Now Tech is at cap=2 before deciding new trades
    candidates = [_analysis("AMD", "Technology")]
    orders = decide_trades(
        current_positions=existing,
        candidate_analyses=candidates,
        holding_analyses=[],
        portfolio_state=_portfolio_state(cash=8000),
        settings=_settings(max_per_sector=2),
    )
    buys = [o for o in orders if o.action == "BUY"]
    # After enrichment Tech count=2 (NVDA+INTC), AMD should be blocked
    assert len(buys) == 0, "AMD should be blocked: Technology sector at cap after enrichment"
```

### Test 2: empty-string sector falls into Unknown bucket (no enrichment case — regression guard)

```python
def test_empty_sector_without_enrichment_falls_into_unknown_bucket():
    """Without prior enrichment, a position with sector='' or sector=None
    falls into the 'Unknown' bucket. This tests the fallback so we can
    confirm the enrichment fix is what makes the real difference."""
    existing = [
        {"ticker": "NVDA", "sector": None, "quantity": 10,
         "current_price": 500, "avg_entry_price": 500,
         "market_value": 5000, "recommendation": "BUY"},
        {"ticker": "INTC", "sector": None, "quantity": 5,
         "current_price": 30, "avg_entry_price": 30,
         "market_value": 150, "recommendation": "BUY"},
    ]
    # Both positions have sector=None => both counted as "Unknown"
    # Tech sector_counts["Technology"] = 0 => new Tech BUY passes the cap
    candidates = [_analysis("AMD", "Technology")]
    orders = decide_trades(
        current_positions=existing,
        candidate_analyses=candidates,
        holding_analyses=[],
        portfolio_state=_portfolio_state(cash=8000),
        settings=_settings(max_per_sector=2),
    )
    buys = [o for o in orders if o.action == "BUY"]
    # Without enrichment: both legacy positions fall in Unknown, Tech count=0,
    # AMD passes the cap check (this is the Bug A behavior)
    assert len(buys) == 1, (
        "Without enrichment, empty-sector legacy positions go to Unknown bucket "
        "and new Tech BUYs slip through -- this is the known bug"
    )
    assert buys[0].ticker == "AMD"
```

---

## Key file:line anchors

| Claim | File:Line |
|-------|----------|
| sector_counts loop reads pos.get("sector") | `portfolio_manager.py:199` |
| sector cap check against max_per_sector | `portfolio_manager.py:212-219` |
| existing candidate enrichment via _fetch_ticker_meta | `autonomous_loop.py:175-192` |
| decide_trades call site in async loop | `autonomous_loop.py:322-329` |
| _fetch_ticker_meta signature (sync def) | `paper_trading.py:688` |
| _fetch_ticker_meta BQ+yfinance strategy | `paper_trading.py:688-744` |
| 24h cache on ticker_meta route | `paper_trading.py:761-768` |
| useLivePrices wiring with tab gate | `page.tsx:412-415` |
| SummaryHero reading stale BQ nav | `page.tsx:188-190` |
| live Market Value derivation idiom in position table | `page.tsx:791-801` |
| SummaryHero rendered unconditionally above tab content | `page.tsx:746` |

---

## Research Gate Checklist (Internal Half)

- [x] Internal exploration covered every relevant module (5 files read in full)
- [x] file:line anchors for every internal claim
- [x] Contradictions / consensus noted (Bug A: the existing loop pattern is correct, only the sector data is missing; Bug B: the live-price derivation idiom already exists in the position table)
- [x] All claims cited per-claim

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/phase-23.1.14-internal-codebase-audit.md",
  "gate_passed": false,
  "note": "This file covers the internal half only. gate_passed is evaluated on the combined external brief in phase-23.1.14-external-research.md"
}
```
