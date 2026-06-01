# Contract — Multi-Market UX (US / EU / KR)

**Step id:** goal-multimarket-ux (session /goal, not a masterplan phase id)
**Date:** 2026-06-01
**Cycle driver:** Main (Claude Code) + Researcher (done) + Q/A (pending)

## Research-gate summary
Researcher PASSED (8 sources read in full, 18 URLs, recency scan done). Brief at
`handoff/current/research_brief.md`. Decisive findings:
- Backend already ships `market`, `base_currency`, and LOCAL `current_price` for
  positions via `SELECT *` (`bigquery_client.py`); `/portfolio` builds positions from
  `trader.get_positions()` (`api/paper_trading.py:188`). The frontend DROPS these at
  the type boundary (`types.ts` `PaperPosition`/`PaperPortfolio` omit them).
- `market_value` / `cost_basis` are USD; `current_price` is LOCAL (phase-50.2).
- `paper_trades` has NO market/currency column → derive market from the yfinance
  ticker suffix (`markets.py::market_for_symbol`, suffix is source of truth).
- `MARKET_CONFIG` (`markets.py:26-62`) maps market→currency+benchmark
  (US=USD/SPY, EU=EUR/^GDAXI, KR=KRW/^KS11).
- No frontend currency formatter exists; `Dollar`/`CurrentPriceCell` hardcode USD.
- Recommended: locale map USD→en-US, EUR→en-IE, KRW→ko-KR; `currencyDisplay:
  'narrowSymbol'`; do NOT force `minimumFractionDigits:2` (breaks KRW 0-dp); market
  filter = ARIA `radiogroup`; extend `PaperTradingDataContext` with `activeMarket`.

## Hypothesis
Because the local price + market already reach the API for positions, the bulk of the
work is frontend: a pure `lib/format.ts` (Intl-based, market→currency map, suffix
derivation) + currency-aware rendering + a Market column + a market filter. Deriving
market from the ticker suffix makes US (bare tickers) resolve to US and stay
byte-identical, while EU/KR (.DE/.KS) resolve correctly — no hard dependency on the API
passing `market`.

## Immutable success criteria (verbatim from the /goal — all must hold, browser-verified)
1. Global market filter (segmented control All·US·EU·KR) in the header filters every
   table, KPI tile, donut, and sector bar. "All" = combined view in USD base.
2. Positions & trades tables carry a MARKET column — chip = colored dot + code
   (US sky, EU amber, KR violet). NO flag emoji.
3. Dual currency, truthful to backend: per-share PRICE/ENTRY/STOP-LOSS in LOCAL;
   every VALUE/NAV/COST-BASIS/FEE in USD. Mirror backend; never invent client-side FX.
4. Locale-correct Intl.NumberFormat: USD/EUR 2dp, KRW 0dp, correct symbol+separators.
   No hardcoded "$" or {currency:'USD'} left anywhere money is market-dependent.
5. "vs SPY" benchmark label dynamic: All/US→vs SPY, EU→vs DAX, KR→vs KOSPI.
6. Market-session strip shows each active market's open/closed state.
7. Cockpit "Latest Transactions" + Reports widgets show market chip + local price.

Plus design-system constraints (navy/slate, Phosphor, NO emoji/flags, Recharts dark,
reuse Dollar/NumberFlow parameterized, WCAG AAA, ARIA radiogroup filter, loading/empty/
error states) and DO-NO-HARM (US filter / single-market path byte-identical).

## Plan (incremental, dependency-ordered)
- **A — Foundation (this increment):** `lib/format.ts` (pure: MARKET_CURRENCY,
  MARKET_BENCHMARK_LABEL, CURRENCY_LOCALE, MARKET_DOT_CLASS, `marketForSymbol`,
  `resolveMarket`, `resolveCurrency`, `formatCurrency`, `formatUsd`,
  `numberFlowFormat`, `numberFlowLocale`). Add optional `market`/`base_currency`/
  `currency` to `PaperPosition`/`PaperPortfolio`/`PaperTrade`.
- **B — Currency-aware rendering + Market column (this increment):** parameterize
  `Dollar` by currency (default USD = byte-identical); `MarketChip` component;
  Market column on positions+trades; entry/current/stop-loss → LOCAL; market_value &
  P&L use backend USD/pct for non-US (no client FX); LatestTransactions → local price +
  market dot. Every USD path branches `currency==='USD'? legacy : Intl` for byte-identity.
- **C — Filter + dynamic benchmark + session strip (next increment):** ARIA radiogroup
  market filter wired through `PaperTradingDataContext.activeMarket`; filter tables/KPIs/
  donut/sector bar; dynamic "vs SPY/DAX/KOSPI"; market-session strip from MARKET_CONFIG tz.
- **D — Verify:** `tsc`/`npm run build`; deterministic `format.ts` assertion (€243.10 /
  ₩71,200 / $971.55); real-browser check that the (currently all-US) live pages are
  unchanged. Then spawn Q/A.

## References
- `handoff/current/research_brief.md` (research gate)
- backend: `backtest/markets.py`, `api/paper_trading.py`, `db/bigquery_client.py`
- frontend rules: `.claude/rules/frontend.md`, `.claude/rules/frontend-layout.md`
- design precedent: `components/BudgetDashboard.tsx` (currency-from-API; NOT the Intl model to copy)
