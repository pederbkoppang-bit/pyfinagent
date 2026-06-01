# Research Brief: Multi-Market UX (US/EU/KR positions & trades, global filter, local + USD currency)

Tier: MODERATE | Date accessed: 2026-06-01 | Step: multi-market UX cycle
Feeds: `handoff/current/contract.md` for the multi-market UX cycle.

## Read in full (>=5 required; counts toward the gate) — 8 read

| # | URL | Accessed | Kind | Fetched how | Key finding |
| --- | --- | --- | --- | --- | --- |
| 1 | https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/NumberFormat/NumberFormat | 2026-06-01 | Official doc (MDN) | WebFetch full | `currencyDisplay`: `symbol`(default)/`narrowSymbol`(`$` not `US$`)/`code`(`EUR 123`)/`name`. Fraction digits default = ISO 4217 minor units; JPY=0; KRW=0; USD/EUR=2. `undefined` locale -> runtime default. de-DE renders `123.456,79 €` (period thousands, comma decimal, trailing symbol). |
| 2 | https://www.w3.org/WAI/ARIA/apg/patterns/radio/ | 2026-06-01 | W3C/WAI-ARIA APG | WebFetch full | Radiogroup = "exactly one option from a set of mutually exclusive choices." Roles `radiogroup`+`radio`+`aria-checked`. Keyboard: Tab/Shift+Tab move in/out; Arrow keys move AND select (selection follows focus); Space selects. Roving tabindex OR aria-activedescendant. |
| 3 | https://www.w3.org/WAI/ARIA/apg/patterns/toolbar/ | 2026-06-01 | W3C/WAI-ARIA APG | WebFetch full | `role=toolbar` groups 3+ controls; single Tab stop; Left/Right arrows move between controls (roving tabindex); requires `aria-label`/`aria-labelledby`. Inside a toolbar, a nested radiogroup does NOT auto-select on focus (toolbar owns arrows). |
| 4 | https://number-flow.barvian.me/ | 2026-06-01 | Official doc (lib) | WebFetch full | `format` prop accepts `Intl.NumberFormatOptions` incl. `style:'currency'`+`currency`. `locales` prop accepts `Intl.LocalesArgument` (BCP-47 string/array). "**Non-Latin digits and RTL locales aren't currently supported.**" `respectMotionPreference` default true; `willChange` default false. |
| 5 | https://react-aria.adobe.com/blog/how-we-internationalized-our-numberfield | 2026-06-01 | Authoritative eng blog (Adobe) | WebFetch full | "in the US we use '.' as the decimal point, while in Germany ',' is used." Uses `Intl.NumberFormat` to avoid shipping locale data ("relies on data the browser already has"). Browser formats but does not parse — parsing needs `formatToParts` digit-mapping. |
| 6 | https://w3c.github.io/i18n-drafts/questions/qa-number-format.en.html | 2026-06-01 | W3C i18n | WebFetch full | `1,234.56` (US) vs `1.234,56` (EU) vs `1 234,56` (space). India 2-digit grouping `12,34,567`. "**Hardcoding formats is a brittle and unsustainable approach.**" Symbol before (`$100`) or after (`1 000 ₫`), with/without space. Use `Intl`. |
| 7 | https://polaris-react.shopify.com/foundations/formatting-localized-currency | 2026-06-01 | Industry practitioner (Shopify design system) | WebFetch full | Two formats: **short** (`$12.50`) vs **explicit** (`$12.50 CAD`). Rule: "Use explicit format when showing total amounts... for merchants who deal with unfamiliar currencies in multi-currency stores." CLDR auto-handles decimals (no yen cents). |
| 8 | https://tc39.es/proposal-intl-numberformat-v3/ (via TC39 search detail) | 2026-06-01 | Standard (ECMA-402) | WebSearch detail + spec | NumberFormat V3 (now shipped baseline): `roundingMode` (default `halfExpand`), `roundingPriority` (auto/morePrecision/lessPrecision), `roundingIncrement`, `trailingZeroDisplay` (`stripIfInteger`). All available in current evergreen browsers. |

Note on #8: the TC39 spec page returned its content via the search-detail expansion (full proposal text), so it is counted as read; sources 1-7 are full WebFetch page reads. The gate floor (>=5 full WebFetch reads) is met by 1-7 alone.

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
| --- | --- | --- |
| https://www.w3.org/WAI/ARIA/apg/patterns/ | W3C APG index | Index page; specific patterns (radio/toolbar) fetched instead |
| https://w3c.github.io/aria/ | WAI-ARIA 1.3 editor's draft | Spec draft; APG pattern pages are the actionable layer |
| https://elementor.com/blog/apg/ | Blog | Lower-tier; APG primary sources used instead |
| https://theosoti.com/short/tabular-nums/ | Blog | Confirms `font-variant-numeric: tabular-nums` + right-align for column alignment; corroborates #6 |
| https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Properties/font-variant-numeric | MDN | `tabular-nums` CSS reference; repo already uses `tabular-nums` class |
| https://dev.to/josephciullo/simplify-currency-formatting-in-react-a-zero-dependency-solution-with-intl-api-3kok | Blog | Reusable-formatter pattern; covered better by #5 |
| https://react.dev (use/useContext) via search | Official React | React 19 `use()` hook can read context conditionally; covered in findings |
| https://www.contentful.com/blog/react-localization-internationalization-i18n/ | Blog | General i18n; not needed (we don't translate, only format) |
| https://lokalise.com/blog/react-i18n-intl/ | Blog | react-intl library overview; we stay zero-dep on native Intl |
| https://www.w3.org/WAI/ARIA/apg/patterns/tabs/ (tablist, via search) | W3C APG | Tablist rejected for filter (see Consensus); not a content-panel switch |

URLs collected total: 18 unique (8 read in full + 10 snippet-only).

## Recency scan (2024-2026)

Searched 2024-2026 for: Intl.NumberFormat new options / ECMA-402 changes; WAI-ARIA APG 2025-2026 updates; React 19 context patterns; React i18n currency best practices. **Findings that complement (none supersede) the canonical sources:**

1. **ECMA-402 Intl.NumberFormat V3** (source #8) is now Baseline in evergreen browsers: `roundingMode`, `roundingPriority`, `roundingIncrement`, `trailingZeroDisplay: 'stripIfInteger'`. Relevant: `trailingZeroDisplay` is the clean way to strip `.00` if we ever want compact KPI display, but for a dense financial table the default 2-fraction-digit alignment is preferred — no action required, just available.
2. **React 19 `use()` hook**: can read Context inside conditionals/loops (relaxes Rules-of-Hooks vs `useContext`). For our currency case `useContext` is sufficient; `use()` is not required.
3. **WAI-ARIA APG (2025/2026 reads)** surfaced a nuance NOT in the bare pattern pages: **a radiogroup nested inside a `role=toolbar` does NOT auto-select on focus** because the toolbar captures Left/Right arrows. This directly informs the role recommendation below — use a STANDALONE radiogroup (not a radiogroup-in-toolbar) so selection-follows-focus works for the filter.
4. No 2024-2026 source contradicts the locale-per-currency or `Intl.NumberFormat` approach; it remains the universal recommendation (MDN, W3C, Adobe React Aria, Shopify Polaris all align).

## Search queries run (3-variant discipline)

- Current-frontier / recency (2025-2026): "Intl.NumberFormat 2025 2026 new options ... ECMA-402 changes"; "WAI-ARIA APG 2025 2026 updates radiogroup toolbar"; "React 19 Context vs prop drilling currency formatter ... 2025".
- Year-less canonical: "Intl.NumberFormat currency formatting multi-currency financial UI currencyDisplay narrowSymbol"; "WAI-ARIA APG segmented control single-select filter radiogroup vs tablist vs toolbar"; "@number-flow/react format currency prop locale"; "tabular-nums financial table column alignment".
- Practitioner cross-check: "React i18n currency formatting best practices dark theme dashboard segmented control filter" -> Shopify Polaris.

## Key findings (external)

1. **Locale-per-currency, not user-locale.** A multi-currency financial UI should format each amount with the locale conventional for THAT currency, so the number reads naturally (`€1.234,56`, `₩1,234,567`, `$1,234.56`). MDN #1 + W3C #6 show separators and symbol placement are locale-driven; hardcoding is "brittle and unsustainable" (W3C #6). Recommended map below.
2. **KRW has 0 fraction digits by default.** ISO 4217 minor units drive `minimum/maximumFractionDigits`; KRW=0, JPY=0, USD/EUR=2 (MDN #1). Do NOT hardcode `minimumFractionDigits: 2` globally — that would render `₩1,234,567.00` (wrong). Let the currency default decide, OR set per-currency.
3. **`currencyDisplay: 'narrowSymbol'`** renders `$` instead of `US$` and is the compact choice for a dense table (MDN #1). But narrowSymbol does NOT disambiguate currencies that share `$`. KRW=`₩`, EUR=`€`, USD=`$` are all distinct, so symbol alone is unambiguous here — `narrowSymbol` is safe and compact.
4. **Show ISO code for the non-native/base value.** Shopify Polaris #7: use "explicit format" (`$12.50 CAD`) when an amount could be confused across a multi-currency view. Our UI shows per-share LOCAL price (€/₩/$) AND a USD value; the USD value should carry a `USD` disambiguator (or a column header) so a EUR position's `$214.50` isn't misread as the local price.
5. **The market filter is a `radiogroup`, not a tablist or toolbar.** APG #2: radiogroup = single mutually-exclusive selection with selection-follows-focus — exactly a single-select "All / US / EU / KR" filter. Tablist is for switching *content panels* (wrong semantics: a filter doesn't swap panels, it narrows one list). Toolbar is for a *group of independent action controls* and, when wrapping a radiogroup, breaks selection-follows-focus (recency finding #3). Use roving tabindex within the radiogroup.
6. **NumberFlow already supports everything we need** (#4): pass `format={{style:'currency', currency, currencyDisplay:'narrowSymbol'}}` and `locales={localeForCurrency}`. The current hardcoded `currency:'USD'` (cockpit-helpers.tsx:63, positions-columns.tsx:52) is a 2-line parameterization. CAVEAT: NumberFlow does not support non-Latin digits/RTL — en-US/de-DE/ko-KR all use Latin digits so we are safe; never pass `ar`/`fa`/`bn` locales.
7. **Context over prop-drilling for the formatter** (#search + React 19): the repo already has `PaperTradingDataContext` (paper-trading-context.tsx). A currency formatter / active-market-filter is a "read-often, write-rarely" value — the canonical Context use case. Do NOT prop-drill `currency` through every TanStack column cell.
8. **tabular-nums + right-align preserves column alignment** despite locale-variable separators (tabular-nums snippet sources + #6). The repo already applies `tabular-nums` + `meta:{align:'right'}` on numeric columns (positions-columns.tsx:109,118,133). Locale-correct separators (`.` vs `,`) are single-glyph and tabular figures keep columns aligned.

## Internal code inventory

| File | Lines | Role | Status |
| --- | --- | --- | --- |
| backend/backtest/markets.py | 26-66 | `MARKET_CONFIG`: per-market `currency`+`benchmark`+`timezone`+`exchange` | EXISTS — US=USD/SPY, EU=EUR/^GDAXI, KR=KRW/^KS11 (also NO=NOK, CA=CAD) |
| backend/backtest/markets.py | 91-115 | `YF_SUFFIX` + `detect_market_from_symbol` (`.DE/.PA/.AS/.F`->EU, `.KS/.KQ`->KR, bare->US) | EXISTS — usable to derive market from a trade ticker |
| backend/services/paper_trader.py | 111 | `get_positions()` — the getter the `/portfolio` endpoint actually calls | EXISTS — emits `market`, `base_currency`, `current_price` (LOCAL) per phase-50.2 (lines 298-333, 468-477, 537) |
| backend/api/paper_trading.py | 171-241 | `GET /portfolio` — returns `{portfolio, positions, sector_breakdown}` | EXISTS — positions come from `trader.get_positions()` so they ALREADY carry market/base_currency/current_price; `portfolio` from BQ `SELECT *` carries market/base_currency too |
| backend/api/paper_trading.py | 244-275 | `GET /trades` — returns `{trades, count}` | EXISTS — trades from `bq.get_paper_trades` (`SELECT *`); no market/currency column on the table |
| backend/db/bigquery_client.py | 517-530 | `get_paper_portfolio` — `SELECT *` | EXISTS — pass-through; table HAS `market`,`base_currency` |
| backend/db/bigquery_client.py | 571-576 | `get_paper_positions` — `SELECT *` ORDER BY entry_date DESC | EXISTS — pass-through; table HAS `market`,`base_currency`,`current_price` |
| backend/db/bigquery_client.py | 674-700 | `get_paper_trades` — `SELECT *` ORDER BY created_at DESC | EXISTS — pass-through; table LACKS `market`,`base_currency` |
| (BQ schema) financial_reports.paper_positions | — | columns | HAS: `market`, `base_currency`, `current_price` (+ ticker, quantity, avg_entry_price, cost_basis, market_value, ...) |
| (BQ schema) financial_reports.paper_portfolio | — | columns | HAS: `market`, `base_currency` (+ total_nav, current_cash, benchmark_return_pct, ...) |
| (BQ schema) financial_reports.paper_trades | — | columns | LACKS `market`/`base_currency`. HAS: trade_id, ticker, action, quantity, price, total_value, transaction_cost, created_at, ... |
| frontend/src/lib/types.ts | 626-641 | `PaperPosition` interface | MISSING `market`, `base_currency` (backend sends them; TS doesn't declare them) |
| frontend/src/lib/types.ts | 608-624 | `PaperPortfolio` interface | MISSING `market`, `base_currency` |
| frontend/src/lib/types.ts | 643-655 | `PaperTrade` interface | No market/currency (table has none either) |
| frontend/src/lib/api.ts | 276-285 | `getPaperPortfolio` / `getPaperTrades` / `getPaperSnapshots` | EXISTS — return-typed to current interfaces; no market arg |
| frontend/src/lib/paper-trading-context.tsx | 1-63 | `PaperTradingDataContext` + `usePaperTradingData()` | EXISTS — the in-repo Context precedent; publishes positions/trades/portfolio once, no prop-drilling. Natural home for an active-market filter + currency formatter |
| frontend/src/components/paper-trading/cockpit-helpers.tsx | 55-74 | `Dollar` helper | HARDCODES `currency:'USD'` + `minimumFractionDigits:2`; used by SummaryHero, positions, trades |
| frontend/src/components/paper-trading/positions-columns.tsx | 29-64 | `CurrentPriceCell` | HARDCODES `currency:'USD'` on the live-price NumberFlow |
| frontend/src/components/paper-trading/positions-columns.tsx | 111-119 | `Entry` column | HARDCODES `$` prefix string (`$${avg_entry_price.toFixed(2)}`) — not even Intl |
| frontend/src/components/paper-trading/positions-columns.tsx | 184-196 | `Stop Loss` column | HARDCODES `$` prefix string |
| frontend/src/components/BudgetDashboard.tsx | 117-144 | variable-currency precedent | EXISTS — reads `data.currency_symbol` from API and prefixes it; the in-repo "currency from backend" pattern (string concat, NOT Intl) |

### Internal: what EXISTS vs what is MISSING

**EXISTS (no backend change needed for positions/portfolio):**
- `MARKET_CONFIG` (markets.py:26) already maps every market -> `currency` + `benchmark`. The UI can hardcode a tiny mirror, or a new endpoint can expose it. Mirror in TS is lowest-risk.
- `paper_positions` and `paper_portfolio` BQ tables HAVE `market` + `base_currency`; `paper_positions` also has `current_price` (LOCAL per phase-50.2). Because both getters `SELECT *` and `/portfolio` builds positions from `trader.get_positions()`, **the `/portfolio` API already returns `market`, `base_currency`, and local `current_price` on each position today** — the frontend simply discards them (TS interface omits them).
- `current_price` on a position is the LOCAL per-share price (markets.py/paper_trader.py:298 comment: "LOCAL price; market_value below is USD"). So per-share LOCAL currency + USD `market_value` are BOTH already on the wire.
- A working Context (`PaperTradingDataContext`) that fans out positions/trades to all sub-routes without prop-drilling.
- `tabular-nums` + right-align already on numeric columns (alignment groundwork done).

**MISSING (the actual work):**
- `frontend/src/lib/types.ts`: `PaperPosition` and `PaperPortfolio` interfaces do NOT declare `market` / `base_currency` (data arrives but is untyped/dropped). Add `market?: string; base_currency?: string;` to both.
- No frontend currency formatter util. There is NO `lib/format.ts`, no `formatCurrency`, no shared `Intl.NumberFormat` wrapper. `BudgetDashboard` does naive `sym + value.toFixed(0)` string concat; `Dollar`/`CurrentPriceCell` hardcode USD via NumberFlow. **Create one shared formatter** (locale-per-currency map + Intl/NumberFlow options builder).
- No market->currency / market->benchmark map in the frontend. Mirror `MARKET_CONFIG`'s `currency`+`benchmark` (a ~6-line const).
- `paper_trades` table has NO `market`/`base_currency` column. Trade rows must derive market from the ticker suffix client-side (port `detect_market_from_symbol`: `.DE/.PA/.AS/.F`->EU, `.KS/.KQ`->KR, else US) OR backend adds a derived field. Lowest-risk: derive in the UI from `ticker`.
- No global market filter UI control exists anywhere.
- `Dollar`, `CurrentPriceCell`, and the two hardcoded `$`-prefix columns (Entry, Stop Loss) need to consume the per-row currency. Entry/Stop Loss currently bypass Intl entirely (plain `$` template strings) — these are the riskiest to leave (would show `$` on a EUR position's entry price).

**US-only regression guard:** Default the formatter to `currency='USD'`, `locale='en-US'` whenever `market`/`base_currency` is absent or `=== 'US'`. With NumberFlow `currency:'USD'` + `minimumFractionDigits:2` is the exact current behavior, so a US position renders byte-identically. The filter defaults to "All" (or "US"), preserving the current single-market view.

## Consensus vs debate (external)

- **Consensus:** Use `Intl.NumberFormat` (never hardcode separators/symbols) — MDN, W3C, Adobe React Aria, Shopify all agree. Format per-currency. Use Context for read-often/write-rarely values. tabular-nums + right-align for table number alignment.
- **Debate / judgment calls:**
  - *Which locale per currency?* No single authority dictates; the convention is the currency's primary-market locale. EUR is the one real choice: `de-DE` (`1.234,56 €`) vs `en-IE` (`€1,234.56`). For a dense table where US numbers dominate, `en-IE` keeps `.`-decimal/`,`-thousands consistent with USD/KRW columns and only swaps the symbol — LESS visual churn across columns. `de-DE` is more "authentic EU" but flips separators. Recommendation below picks `en-IE` for column consistency; flag as a reversible product choice.
  - *Symbol vs code?* narrowSymbol is compact and unambiguous here (`$`/`€`/`₩` differ). Shopify argues for explicit ISO code on totals in multi-currency contexts — apply that ONLY to the USD "base value" column/label to distinguish it from the local price, not to every cell.
  - *radiogroup vs toolbar?* APG 2026 nuance settles it: standalone radiogroup (selection-follows-focus); avoid radiogroup-in-toolbar (breaks that).

## Pitfalls (from literature)

1. **Hardcoding `minimumFractionDigits: 2` breaks KRW** — `₩1,234,567.00` is wrong (KRW minor units = 0). Let the per-currency default decide, or set fraction digits per currency. (MDN #1)
2. **NumberFlow can't render non-Latin digits / RTL** (#4) — en-US/de-DE/ko-KR use Latin digits (safe); never pass Arabic/Persian/Bengali locales to NumberFlow.
3. **Locale-variable separators can misalign columns** if not tabular — must keep `font-variant-numeric: tabular-nums` + right-align (already present). (W3C #6 + tabular-nums sources)
4. **Ambiguous USD value vs local price** — a EUR position shows local `€214,50` and USD `$231.10`; without a `USD` label/column-header the reader can misattribute. Use explicit ISO code on the base column. (Shopify #7)
5. **`paper_trades` has no market column** — deriving market from a bare US ticker is fine (no suffix -> US), but a non-suffixed intl ticker would misclassify; the repo's own `detect_market_from_symbol` treats the suffix as source of truth, so port it verbatim rather than re-inventing. (markets.py:96-115)
6. **Context re-render storms** — wrap the formatter/value in `useMemo`/`useCallback` (search finding). A market-filter value changes rarely, so risk is low, but memoize the formatter factory.
7. **Tailwind JIT + runtime currency** — if any currency-driven color/class is introduced, use a static lookup map (frontend.md rule), not template-string classes. (Repo rule, not literature.)

## Application to pyfinagent (external -> file:line)

- Locale-per-currency map -> consume in a NEW `frontend/src/lib/format.ts`; replace hardcoded `currency:'USD'` at `cockpit-helpers.tsx:63` and `positions-columns.tsx:52`, and the plain `$` strings at `positions-columns.tsx:116,191`.
- `currencyDisplay:'narrowSymbol'` + per-currency fraction digits -> the same formatter util.
- radiogroup market filter -> a new component (e.g. `MarketFilter.tsx`); roles `radiogroup`/`radio`/`aria-checked`, roving tabindex, dark-navy AAA contrast (`text-slate-100` selected, `text-slate-400` idle per frontend.md). Filter state lives in `paper-trading-context.tsx` (extend `PaperTradingDataValue`).
- Context-vs-formatter -> extend `PaperTradingDataContext` (paper-trading-context.tsx:30) with `activeMarket` + a memoized `formatLocal(value, currency)` / a `currencyForMarket(market)` helper; consume via `usePaperTradingData()` in columns instead of prop-drilling.
- Types -> add `market?: string; base_currency?: string;` to `PaperPosition` (types.ts:626) and `PaperPortfolio` (types.ts:608). `PaperTrade` (types.ts:643) gets market via derivation, not a field.
- market->currency / market->benchmark mirror of `MARKET_CONFIG` (markets.py:26-66) -> a const in `format.ts` or a tiny new endpoint; mirror is lowest-risk.
- US regression guard -> formatter defaults `USD`/`en-US` when market absent/`'US'`; filter defaults to All/US.

## Recommended approach (actionable)

- **Locale-per-currency map:**
  ```ts
  // currency -> BCP-47 locale (column-consistency choice for a USD-dominant table)
  const CURRENCY_LOCALE: Record<string, string> = {
    USD: "en-US",  // $1,234.56   (2 frac)
    EUR: "en-IE",  // €1,234.56   (en-IE keeps '.'/',' like USD; de-DE would flip to 1.234,56 €)
    KRW: "ko-KR",  // ₩1,234,567  (0 frac, Latin digits — NumberFlow-safe)
    NOK: "nb-NO",  // present in MARKET_CONFIG; future
    CAD: "en-CA",
  };
  ```
  Use `currencyDisplay:'narrowSymbol'`; do NOT force `minimumFractionDigits` — let the currency default (USD/EUR=2, KRW=0) apply, or set `{KRW:0}` explicitly. (Decision: `en-IE` for EUR is a reversible product choice to keep separators consistent across columns; switch to `de-DE` only if the operator wants authentic EU separators and accepts column-glyph variance.)
- **ARIA role for the market filter:** **`radiogroup`** (standalone, NOT inside a toolbar). Roles `radiogroup` + `radio` + `aria-checked`; roving tabindex; Tab in/out, Arrow keys move-and-select, Space selects; `aria-label="Market filter"`. Rationale: single mutually-exclusive selection with selection-follows-focus is the radiogroup contract; tablist is for content-panel switching; a radiogroup-in-toolbar would break selection-follows-focus (APG 2026 nuance). Dark-navy AAA: selected `bg-sky-500/10 text-sky-400` or `text-slate-100`; idle `text-slate-400 hover:text-slate-200`.
- **Context vs formatter:** **Both, via the existing Context.** Put the rarely-changing `activeMarket` filter state in `PaperTradingDataContext`, and expose a memoized formatter (`formatLocal(value, currency)` + `currencyForMarket(market)`), implemented in a new `lib/format.ts` and surfaced through the context value. Columns call `usePaperTradingData()` — no `currency` prop drilled through TanStack column defs. This matches the repo's own no-prop-drilling rationale (paper-trading-context.tsx header comment).

## Research Gate Checklist

Hard blockers — all satisfied:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (8: MDN, ARIA Radio, ARIA Toolbar, NumberFlow, React Aria, W3C i18n, Shopify Polaris, TC39 V3)
- [x] 10+ unique URLs total (18: 8 full + 10 snippet-only)
- [x] Recency scan (last 2 years) performed + reported (4 findings; APG-in-toolbar nuance is load-bearing)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (API, BQ getters, BQ schemas, markets.py, types, api.ts, context, cockpit-helpers, positions-columns, BudgetDashboard precedent)
- [x] Contradictions / consensus noted (EUR locale + symbol-vs-code debates)
- [x] All claims cited per-claim with file:line or URL

## JSON envelope
```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 10,
  "urls_collected": 18,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "gate_passed": true
}
```
