# Experiment Results — Multi-Market UX (Phases A + B + C)

**Step:** goal-multimarket-ux · **Date:** 2026-06-01 · **Status:** complete (A+B+C done; all 7 immutable criteria addressed in code; visual-browser pass pending — see caveat)

## What was built

### A — Foundation
- **NEW `frontend/src/lib/format.ts`** — pure, dependency-free (type-only import of
  NumberFlow `Format`). Mirrors backend `MARKET_CONFIG` + `market_for_symbol`:
  - `MARKET_CURRENCY` (US→USD, EU→EUR, KR→KRW, NO→NOK, SE→SEK, DK→DKK, FI→EUR, IS→ISK, CA→CAD)
  - `MARKET_BENCHMARK_LABEL` (US→SPY, EU→DAX, KR→KOSPI, …)
  - `CURRENCY_LOCALE` (USD→en-US, EUR→en-IE, KRW→ko-KR, …)
  - `MARKET_DOT_CLASS` (static JIT-safe Tailwind map; US sky / EU amber / KR violet / …)
  - `MARKET_EXCHANGE` + `MARKET_EXCHANGE_SHORT` (chip tooltip + compact tag)
  - `MARKET_ORDER` (canonical display order for filter + session strip)
  - `marketForSymbol` (TS port; suffix is source of truth; bare→US)
  - `resolveMarket`, `resolveCurrency`, `formatCurrency` (Intl, narrowSymbol, no
    forced fraction digits → KRW 0dp), `formatUsd`, `numberFlowFormat`, `numberFlowLocale`
  - `isMarketOpen` (session heuristic: weekday + local cash-session window per exchange tz;
    holiday-blind by design — backend `exchange_calendars` owns the authoritative gate)
- **`frontend/src/lib/types.ts`** — added optional `market?`/`base_currency?` to
  `PaperPosition` + `PaperPortfolio`; optional `market?`/`currency?` to `PaperTrade`.
  Optional ⇒ backward-compatible; documented that price/entry/stop are LOCAL while
  cost_basis/market_value/total_value/fee are USD.

### B — Currency-aware rendering + Market column
- **`cockpit-helpers.tsx`** — `Dollar` now takes optional `currency` (default "USD";
  USD branch keeps the EXACT legacy format object + default locales → byte-identical).
  New shared `MarketChip` (colored dot + market code [+ optional short exchange tag];
  NO flag emoji; `aria-hidden` dot so the code carries the meaning — WCAG, not color-only).
- **`positions-columns.tsx`** — new MARKET column after Ticker; ENTRY / CURRENT /
  STOP-LOSS render in LOCAL currency (USD branch byte-identical); MARKET VALUE and P&L%
  use the backend USD / `unrealized_pnl_pct` for non-US (the `livePrice × qty` recompute
  is LOCAL notional, valid only for US — `resolveMarket==='US'` is both the do-no-harm
  guard and the no-client-FX guard).
- **`trades-columns.tsx`** — new MARKET column (derived from ticker suffix); PRICE in
  LOCAL; VALUE + FEE stay USD.
- **`LatestTransactionsBox.tsx`** (home cockpit) — market dot before the ticker; PRICE
  in LOCAL (USD path byte-identical).

### C — Filter + dynamic benchmark + session strip
- **NEW `components/paper-trading/MarketFilter.tsx`** — WAI-ARIA APG `radiogroup`
  segmented control (All · US · EU · KR · …). Roving tabindex, Arrow/Home/End keyboard
  nav with selection-follows-focus, `aria-checked`, focus-visible ring. Options are
  data-driven (core US/EU/KR + any held/traded market, in `MARKET_ORDER`). Colored dot +
  code; exchange name in `title`. NO flag emoji.
- **NEW `components/paper-trading/MarketSessionStrip.tsx`** — per-market OPEN/CLOSED dot
  (emerald/slate) from `isMarketOpen`. Mounts to `null`-then-`Date` to avoid SSR
  hydration mismatch; re-evaluates every 60s. Holiday-blind UI hint (documented).
- **`paper-trading/layout.tsx`** — owns `activeMarket` state (default "ALL"); computes
  `availableMarkets` (core set ∪ markets present in positions/trades, in canonical order);
  auto-resets to "ALL" if the active market disappears; mounts the filter + session strip
  as a Tier-4 global control; threads `positions`+`activeMarket` into `SummaryHero`; shows a
  "Filtered to X — NAV/Cash/Sharpe are fund-level USD; table/allocation/sector show X only"
  hint when a single market is selected. Exposes `activeMarket`/`setActiveMarket` via
  `PaperTradingDataContext`.
- **`positions/page.tsx`** — filters the table to `activeMarket`; donut + sector-bar scope
  to the filtered set using a USD market-value helper (`mvUsd`: US = legacy livePrice×qty,
  non-US = backend USD `market_value`, no client FX); single-market denominator = that
  market's USD holdings (sectors sum to ~100% within the market); Cash slice only in "All";
  donut center + title reflect the filtered market.
- **`trades/page.tsx`** — filters the trades table to `activeMarket` (suffix-derived).
- **`cockpit-helpers.tsx` `SummaryHero`** — Positions tile count filters to the active
  market; **dynamic benchmark label** ("vs SPY" / "vs DAX" / "vs KOSPI"); for a specific
  non-US market (no per-market index in the API) shows that market's USD-consistent
  holdings return with an explanatory tooltip rather than inventing FX-converted excess.

## Files changed
NEW: `frontend/src/lib/format.ts`,
`frontend/src/components/paper-trading/MarketFilter.tsx`,
`frontend/src/components/paper-trading/MarketSessionStrip.tsx`.
MODIFIED: `frontend/src/lib/types.ts`, `frontend/src/lib/paper-trading-context.tsx`,
`frontend/src/app/paper-trading/layout.tsx`,
`frontend/src/app/paper-trading/positions/page.tsx`,
`frontend/src/app/paper-trading/trades/page.tsx`,
`frontend/src/components/LatestTransactionsBox.tsx`,
`frontend/src/components/paper-trading/cockpit-helpers.tsx`,
`frontend/src/components/paper-trading/positions-columns.tsx`,
`frontend/src/components/paper-trading/trades-columns.tsx`.

## Verification (verbatim)

TypeScript (frontend, strict):
```
$ node_modules/.bin/tsc --noEmit -p tsconfig.json
TSC_EXIT=0
```

Production build (`npm run build`) — clean; modified routes compiled:
```
├ ○ /paper-trading/positions               11 kB         137 kB
├ ○ /paper-trading/trades                3.39 kB         129 kB
○  (Static)   prerendered as static content
ƒ  (Dynamic)  server-rendered on demand
```
NOTE: the FIRST build run failed prerendering `/404` + `/_error` with
"`<Html>` should not be imported outside of pages/_document". This is a stale-`.next`
cache symptom UNRELATED to this work — proved: zero `next/document` imports in `src/`,
no `pages/` dir (pure App Router), no custom 404/error/not-found/global-error pages, and
the diff touches none of those paths. A second build (regenerating the stale chunk)
completed clean, as shown above.

Deterministic `format.ts` proof (transpiled REAL module via `tsc src/lib/format.ts`,
then `node` — tests the shipped code, not a reimplementation):
```
PASS USD 971.55 -> "$971.55"
PASS EUR 243.1 (en-IE) -> "€243.10"
PASS KRW 71200 0dp -> "₩71,200"
PASS formatUsd 1694 -> "$1,694.00"
PASS null -> dash -> "—"
PASS bare AAPL -> US      PASS SAP.DE -> EU      PASS 005930.KS -> KR      PASS EQNR.OL -> NO
PASS resolveCurrency SAP.DE -> EUR    PASS resolveCurrency bare -> USD
PASS resolveMarket explicit wins -> "KR"
PASS bench label EU -> "DAX"    PASS bench label KR -> "KOSPI"
PASS US Sat closed -> false    PASS US Mon 10:00 ET open -> true    PASS US Mon 19:00 ET closed -> false
ALL_FORMAT_OK   (17/17)
```
KRW renders 0 decimals (₩71,200), EUR/USD 2 — confirms not forcing minimumFractionDigits.

## Criteria status (all 7 immutable)
1. Global market filter (All·US·EU·KR) filters tables/KPI tile/donut/sector bar; "All" =
   combined USD: **DONE** (MarketFilter in layout; positions+trades filtered; donut+sector
   scoped; Positions tile count filtered).
2. Market column + chip (dot+code, NO flag emoji): **DONE** (positions + trades).
3. Dual currency (local price/entry/stop; USD value/fee/cost-basis): **DONE**.
4. Locale-correct Intl (KRW 0dp): **DONE + verified**.
5. Dynamic benchmark label (vs SPY/DAX/KOSPI): **DONE** (SummaryHero `benchLabel`).
6. Market-session strip: **DONE** (MarketSessionStrip in layout).
7. Latest Transactions market chip + local price: **DONE**. (Reports History is
   score-based — no price/currency — so "local price" is N/A there.)

## Do-no-harm evidence
Every USD path branches `currency === "USD" ? <legacy exact> : <Intl>`, so US rows are
byte-identical (proved: $971.55/$880.72/$1,694.00 match legacy; tsc clean; build clean).
`marketForSymbol` returns US for bare tickers, so the current all-US live portfolio is
unchanged. The market filter defaults to "ALL"; with an all-US book, "ALL" and "US" both
show the full set, and the donut/sector/KPI math reduces to the pre-change formulas.

## Browser-verification caveat (for Q/A and operator)
The live paper portfolio is currently **all-US** (first multi-market cycle scheduled
Mon 14:00 UTC). Therefore:
- US-unchanged + filter/session-strip rendering CAN be visually confirmed now at
  `localhost:3000/paper-trading` (operator review, or computer-use once macOS
  Accessibility + Screen-Recording permissions are granted and Claude Code restarted —
  currently NOT granted).
- EU/KR € / ₩ row rendering cannot be seen with live data until EU/KR positions exist
  (Monday's cycle or a seeded fixture). It is proved deterministically by the Intl check
  above (real module).

Per `.claude/rules/frontend.md` rule 5, this color-coded UI was marked **visual
verification pending operator review** until one of the above paths runs.

## Visual verification — DONE (2026-06-01, computer-use screenshot, all-US live book)
Confirmed live at `localhost:3000/paper-trading` (after a `launchctl kickstart` of
`com.pyfinagent.frontend` — the earlier `npm run build` had clobbered the dev `.next`,
causing a transient 500; kickstart regenerated the dev build and the route returned 200):
- **Market filter** renders `All · US · EU · KR` (All selected) with the correct dot
  colors — US sky, EU amber, KR violet. (#1, #2)
- **Market-session strip** shows `US CLOSED · EU OPEN · KR CLOSED`, and the states are
  ACCURATE for the capture time (~13:49 CEST Mon): XETRA open, NYSE + KRX closed. (#6)
- **Dynamic benchmark label** shows `VS SPY` for the All/US view. (#5)
- **Market column** on the positions table shows the `● US · NYSE` chip (sky dot + code
  + short exchange) on every row — NO flag emoji. (#2)
- **All KPIs + values in USD** (NAV 24,098.52 USD, Cash 15,983.75 USD) — do-no-harm holds.
- Allocation donut, sector concentration, Risk Monitor all render cleanly; no layout break.

DO-NO-HARM confirmed: US rows render as before. Note the live/animated USD cells
(CURRENT, MARKET VALUE) show e.g. `880,03 USD` (comma decimal, "USD" suffix) rather than
`$880.03` — this is PRE-EXISTING legacy behavior, NOT introduced here: the `Dollar` /
`CurrentPriceCell` USD branch uses NumberFlow with no explicit locale, so it renders in
the browser locale (this machine is Norwegian `nb-NO` → comma + "USD"). This cycle kept
that USD branch byte-identical (`locales={undefined}`); the entry/stop `$X.XX` cells were
always hardcoded. (Optional future polish, out of THIS contract's scope: pin the USD
NumberFlow path to `en-US` for a consistent `$` — but that intentionally breaks the
do-no-harm byte-identity, so it's a deliberate product decision, not a bug fix.)

STILL PENDING (cannot be done now): EU/KR € / ₩ row rendering — the live book is all-US
until Monday's first multi-market cycle (14:00 UTC). Proven deterministically via the Intl
check above; re-confirm visually once non-US positions exist or via a seeded fixture.
