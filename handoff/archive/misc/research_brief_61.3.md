# Research Brief — phase-61.3 (money-display + currency correctness)

**Status: COMPLETE — gate_passed: true (envelope at end).**

Tier: complex. Date: 2026-07-08. Author: researcher (Layer-3 harness MAS).
Pre-pay brief: step 61.3 has not started; its future contract cites and revalidates this.

## Scope (from caller)

1. Reproduce latent add-on-BUY USD-into-LOCAL averaging bug (paper_trader.py) on paper
2. Inventory positions price-column currency resolution (backend + frontend)
3. Inventory hardcoded toFixed/currency templates vs shared formatCurrency
4. Non-US P&L staleness: where live local price mixes with stale FX/mark; mark-timestamp surface
5. KRX/XETRA close times vs single 18:00 UTC cycle; what a ~07:00 UTC post-KRX-close mark job would touch

External: multi-currency average-cost accounting; Intl.NumberFormat locale pinning; cross-market mark-to-market timing/as-of labeling; mixed-currency table UX.

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://www.cpdbox.com/ias21-foreign-exchange-rates/ | 2026-07-08 | authoritative doc (IFRS specialist) | WebFetch full | "Initially, all foreign currency transactions shall be translated to functional currency by applying the **spot exchange rate** ... at the date of the transaction"; non-monetary items at historical cost keep the transaction-date rate — never restated at current FX |
| https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/NumberFormat/NumberFormat | 2026-07-08 | official docs (MDN) | WebFetch full | "The runtime's default locale is used when `undefined` is passed"; currency fraction digits default to ISO 4217 minor units (KRW=0); narrowSymbol = "$100" not "US$100" |
| https://nextjs.org/docs/messages/react-hydration-error | 2026-07-08 | official docs (Vercel) | WebFetch full | Hydration mismatch causes include time/environment-dependent rendering; fixes: deterministic output, useEffect client-only, suppressHydrationWarning ("escape hatch. Don't overuse it") |
| https://www.w3.org/International/questions/qa-number-format | 2026-07-08 | official docs (W3C i18n) | WebFetch full | Decimal/grouping separators vary by region ("1,234.56" vs "1.234,56"); "The same symbol might represent multiple currencies (e.g., `$` for US Dollar, Canadian Dollar, Mexican Peso...)" |
| https://medium.com/workday-design/the-ux-of-currency-display-whats-in-a-sign-6447cbc4fb88 | 2026-07-08 | authoritative design blog (Workday Design) | WebFetch full | "$" used in 28 countries; enterprise-table pattern = ISO code alongside amounts + separate column converted to the user's base currency — "gives users the much needed context ... reduces potential misunderstandings" |
| https://www.zigpoll.com/content/what-are-the-best-ux-practices-for-displaying-multicurrency-prices-dynamically-across-different-digital-touchpoints-without-overwhelming-the-user | 2026-07-08 | industry blog | WebFetch full | "Cache exchange rates and update them regularly (e.g., daily). Clearly inform users when rates were last updated"; "Ensure uniform currency formatting across product pages..." |
| https://public.econ.duke.edu/Papers/PDF/Time_Zone_Arbitrage.pdf (Donnelly & Tower 2008, Duke) | 2026-07-08 | academic working paper | curl + pypdf/pdfplumber extraction (research-gate PDF chain step 3) | "When a foreign market closes, the assets traded on that exchange will artificially freeze in value ... These NAV's if used hours later are termed 'stale prices'"; "The predictability of change in the stale prices when the foreign market opens creates an arbitrage opportunity" |

## Identified but snippet-only (does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://www.iasplus.com/en/standards/ias/ias21 | official summary (Deloitte) | fetched but page body returned empty (header only); CPDbox covers same content |
| https://ifrscommunity.com/knowledge-base/ias-21-effects-of-changes-in-foreign-exchange-rates/ | authoritative doc | HTTP 403 |
| https://www.bogleheads.org/wiki/Fair_value_pricing | community wiki | HTTP 403; Duke paper covers the mechanics at higher tier |
| https://www.ifrs.org/issued-standards/list-of-standards/ias-21-the-effects-of-changes-in-foreign-exchange-rates/ | official standard | registration-walled PDF |
| https://github.com/vercel/next.js/discussions/79397 | community (2025) | snippet confirms toLocaleString hydration error 418 across Node versions — recency evidence |
| https://github.com/vercel/next.js/discussions/19409 | community | server/client Intl locale misalignment ("locales in browser and Node are not the same... en picked by default in Node") |
| https://github.com/nuxt/nuxt/discussions/17629 | community | same failure class outside Next — framework-independent |
| https://github.com/amannn/next-intl/issues/528 | community | plural/locale hydration error |
| https://www.msci.com/documents/1296102/1335390/FV+Research+Bulletin_FINAL.pdf | industry (MSCI) | binary PDF, secondary to Duke paper |
| https://www.unescap.org/sites/default/d8files/11-CHA~1_0_0.PDF | institutional | binary PDF |
| https://www.sec.gov/files/rules/proposed/s71104/lmmetzger050704.pdf | regulator comment file | binary PDF |
| https://www.sciencedirect.com/science/article/abs/pii/S0378426699001260 | peer-reviewed | paywalled abstract only |
| https://www.wildnetedge.com/blogs/fintech-ux-design-best-practices-for-financial-dashboards | industry (2026) | generic dashboard UX; no currency-specific delta vs Workday |
| https://dart.deloitte.com/USDART/.../5-6-foreign-currency-matters | official (Deloitte DART) | login-walled |
| ~24 further unique hits (ICAEW, GAAP Dynamics, Moore Global, phrase.com, dev.to, coreui, lucanerlich, uxmatters, eleken, CurrencyFair Medium, Bogleheads forum, mutualfunds.com, ResearchGate, Hastings LJ, etc.) | mixed | lower marginal value; counted in urls_collected |

## Search queries run (3-variant discipline)

1. Year-less canonical: "IAS 21 average cost foreign currency securities functional currency transaction date rate"; "stale prices mutual fund fair value pricing international markets time zone"; "displaying mixed currency amounts table UX design multi-currency dashboard best practices"; "Intl.NumberFormat undefined locale pitfalls hydration mismatch Next.js toLocaleString"
2. Last-2-year window + current-year frontier (combined pass): "Intl.NumberFormat currency formatting best practices 2025 2026 JavaScript locale pinning"
3. Frontier hits also arrived organically: wildnetedge "[2026]" fintech-dashboard piece, vercel discussion #79397 (2025).

## Recency scan (2024-2026)

Searched the 2024-2026 window explicitly (query variant 2 above). Result: three findings, none superseding the canonical sources:
1. **Hydration/locale bugs remain live in 2025-2026** — vercel/next.js discussion #79397 (2025) documents `toLocaleString()` producing React error 418 hydration mismatches when Node build/revalidate environments differ; confirms the undefined-locale hazard is current, not historical.
2. **IAS 21 amendment "Lack of Exchangeability" became effective 2025-01-01** — changes spot-rate estimation only when a currency is not exchangeable; KRW and EUR are freely exchangeable, so no impact on the 61.3 design.
3. **2026 fintech-dashboard UX guidance** (wildnetedge) is consistent with the older Workday canonical pattern (codes + base-currency column); no new convention emerged.

## Key findings (external)

1. **Accounting standard: local cost and base-currency cost are separate ledgers.** IAS 21: transactions are recognized at the transaction-date spot rate, and historical-cost items KEEP that rate — you never re-derive local cost from base cost at a later FX rate (Source: CPDbox IAS 21, accessed 2026-07-08). Mapping: `avg_entry_price` (LOCAL) must be averaged from local prices; `cost_basis` (USD) from transaction-date USD amounts. The two are linked only through per-transaction FX, never through division of aggregates — which is precisely what paper_trader.py:291 does wrong.
2. **`locales: undefined` = nondeterminism by design.** "The runtime's default locale is used when undefined is passed" (MDN). Node's ICU default (typically en-US) vs the operator's nb-NO browser makes the same component render differently server vs client — the class of bug Next.js documents as hydration error; deterministic (pinned-locale) output is the non-escape-hatch fix (Next.js docs). GitHub #19409/#79397 show this failing in production stacks through 2025.
3. **Currency identity must be explicit in mixed-currency tables.** "$" is used in 28 countries (Workday); W3C documents separator/grouping divergence. The Workday enterprise pattern — local amount with explicit currency + a separate base-currency (USD) conversion column — is exactly the positions-table shape pyfinagent already has; the defect is only that resolution returns the wrong currency for LOCAL columns.
4. **Staleness must be labeled with time.** "Clearly inform users when rates were last updated" (Zigpoll); the mutual-fund literature calls values reused hours after a foreign close "stale prices" with predictable drift (Donnelly & Tower 2008). An unlabeled stale P&L next to a live price is the retail-facing version of the stale-NAV problem.
5. **Marking at/after the local close is the industry norm; staleness windows of hours are material.** European markets close by 11:00 ET, Pacific after midnight ET; funds that reuse those closes hours later carry predictable error (Donnelly & Tower 2008). A post-KRX-close mark (~07:00 UTC) aligns the KR book with local close, the same discipline mutual funds apply at 16:00 ET for US books.

## Internal code inventory

All anchors re-verified against working tree 2026-07-08 (the masterplan's 2026-06-11
audit_basis anchors still hold; a few frontend paths differ from audit shorthand —
`positions-columns.tsx` lives under `frontend/src/components/paper-trading/`).

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/services/paper_trader.py` | 286-302 | add-on BUY position update (THE bug) | live, latent |
| `backend/services/paper_trader.py` | 313, 334, 481 | `base_currency: "USD"` hardcode on every pos_row | live |
| `backend/services/paper_trader.py` | 393-396, 443 | SELL realized P&L consumers of avg_entry_price | live |
| `backend/services/paper_trader.py` | 462 | partial-sell cost_basis fallback `qty*avg_entry_price` | live |
| `backend/services/paper_trader.py` | 500-583 | `mark_to_market` (no timestamp persisted) | live |
| `backend/services/paper_trader.py` | 591-599 | `check_stop_losses` `current <= stop` (both LOCAL) | live |
| `backend/services/paper_trader.py` | 724, 756-760 | `backfill_missing_stops` from avg_entry_price | live |
| `backend/services/paper_trader.py` | 1071-1137 | `_advance_stop` breakeven+trail from avg_entry_price | live |
| `backend/services/fx_rates.py` | 53, 78-103, 153-178, 182-197 | FX live 6h cache + as-of BQ read | live |
| `backend/services/autonomous_loop.py` | 997-1001, 1041, 1083, 1268 | mark_to_market + check_stop_losses call sites | live |
| `backend/api/paper_trading.py` | 1289-1323 | `init_scheduler`/`_add_scheduler_job` cron (10:00 ET default, mon-fri, misfire 3600, coalesce) | live |
| `backend/config/settings.py` | 341 | `paper_trading_hour: int = 10` (ET) | live |
| `backend/backtest/markets.py` | 26-62, 168-211 | MARKET_CONFIG (XETR/XKRX) + `is_trading_day` (50.4 is_session fix) | live |
| `frontend/src/lib/format.ts` | 23, 86, 161-171, 173-175, 180-197, 199-201 | MARKET_CURRENCY, CURRENCY_LOCALE, resolveCurrency (explicit-first), localeForCurrency, formatCurrency, formatUsd | live |
| `frontend/src/components/paper-trading/positions-columns.tsx` | 64-79, 144-153, 169-180, 254-262 | CurrentPriceCell NumberFlow undefined-locale USD branch; Entry/Current/Stop cells | live |
| `frontend/src/components/paper-trading/cockpit-helpers.tsx` | 72-104 (locales at 96) | shared `Dollar` NumberFlow, USD branch `locales=undefined` | live |
| `frontend/src/lib/types.ts` | 649-658 | PaperPosition currency contract (LOCAL vs USD columns) | live |

### 1. Latent add-on-BUY averaging bug — reproduced on paper

Mechanics (`backend/services/paper_trader.py`): quantity is sized in LOCAL shares from
USD at :212 (`quantity = (amount_usd * _usd_to_local) / price`). First BUY stores
`avg_entry_price = price` (LOCAL, :321) but `cost_basis = amount_usd` (USD, :322).
The add-on branch then averages ACROSS scales:

```python
old_cost = existing["cost_basis"] or (old_qty * existing["avg_entry_price"])  # :288 USD
new_cost = old_cost + amount_usd                                              # :290 USD
new_avg  = new_cost / new_qty                                                 # :291 USD/share
"avg_entry_price": round(new_avg, 4),                                         # :297 stored as LOCAL
```

For US (`_usd_to_local == 1`) this is byte-identical; for KR/EU it replaces the LOCAL
average with a USD-per-share number.

Numeric reproduction (script run 2026-07-08, mirrors :286-302 exactly):
KR name, fx = 1350 KRW/USD. Buy 1: 50,000 KRW x $500 -> q=13.5, avg=50,000 KRW,
cost_basis=$500. Add-on: 52,000 KRW x $300 -> q2=7.7885.
`new_avg = 800/21.2885 = 37.5790` stored as avg_entry_price — true LOCAL average is
**50,731.71 KRW**; corruption ratio = 1350x = exactly the FX rate.

Downstream blast radius (all consume `avg_entry_price` believing it LOCAL):
- **Realized P&L** — SELL at 51,000 KRW: `realized_pnl_pct` (:393-396) = **135,614%**
  (true +0.53%); `realized_pnl_usd` (:443) = **$803.64** (true $4.23). Poisons
  paper_trades, round-trips (exit-quality analytics 4.5.9), Slack notifications.
- **Breakeven ratchet** (:1137) — returns `(entry_price, iso)`: stop_loss_price is
  OVERWRITTEN with 37.58 "KRW". `check_stop_losses` (:598) tests `51,000 <= 37.58` ->
  never fires. The position's previously-valid LOCAL stop is destroyed; downside
  protection silently removed.
- **Trailing branch** (:1105-1125) — `peak_price = entry_price*(1+mfe/100)` also
  USD-scale; every trail value untriggerable.
- **Stop backfill** (:756-760) — `stop = avg_entry_price * (1-pct/100)` -> garbage.
- **Fallback cost bases** (:288, :462, :524) — `qty * avg_entry_price` post-corruption
  coincidentally equals the USD cost (new_avg = USD cost/qty), so cost_basis-derived
  USD P&L stays right; the damage is confined to LOCAL-scale consumers — which is
  exactly the stop/realized-P&L machinery.
- **EU is the insidious case**: EURUSD~1.08 -> stored avg 109.58 vs true 101.46 EUR
  (**+8.0%, silently plausible**). Realized P&L understated ~8pp; breakeven stop set
  ABOVE true entry (fires ~8% early). KR screams; EU whispers.

Trigger condition: any BUY for a ticker with an existing non-US position (`existing`
branch :286). Confirmed unfired so far (masterplan audit_basis); portfolio is 100%
cash since 2026-07-03, so the fix window is open NOW — no live position can trip it
mid-fix.

Secondary note (do not scope-creep): the add-on/first-buy pos_row stores requested
`price`, while the trade row stores `exec_price` (:255-266). Identical under bq_sim
fills; flag only.

### 2. Positions price-column currency resolution

Contract (`frontend/src/lib/types.ts:649-658`): `avg_entry_price` / `current_price` /
`stop_loss_price` are LOCAL; `cost_basis` / `market_value` are USD.
`base_currency` is documented as the currency of the USD-based columns — and the
backend hardcodes `"USD"` on every row (paper_trader.py:313, :334, :481).

`resolveCurrency` (`format.ts:161-171`) is **explicit-first**: `baseCurrency ?? currency`
wins over market. Every LOCAL-price cell passes `baseCurrency: row.base_currency`:
- Entry: positions-columns.tsx:144-148 -> `$50,000.00`-style USD render of a KRW price
- Current: :169-173 (comment at :167 even says "LOCAL currency (phase-50.2)")
- Stop Loss: :254-258

So a KR row's LOCAL columns always resolve USD. **Fix design: market-first resolution
for LOCAL price columns** — either (a) stop passing `baseCurrency` into resolveCurrency
at these three call sites (market/ticker suffix alone resolves KR->KRW via
MARKET_CURRENCY, format.ts:23), or (b) add a `resolveLocalCurrency()` helper that
ignores explicit base_currency by contract. Option (b) is self-documenting and keeps
resolveCurrency for genuinely-explicit surfaces. US rows: resolveMarket->US->USD,
byte-identical. Note `market_value`/`cost_basis`/P&L cells correctly stay USD
(positions-columns.tsx:186-213 fallback logic + formatUsd) — do not touch.

### 3. Hardcoded currency formatting vs formatCurrency

Shared util exists and is locale-correct: `formatCurrency` (format.ts:180-197) pins
locale via `CURRENCY_LOCALE` (:86) -> `localeForCurrency` (:173-175, default "en-US"),
uses narrowSymbol, no forced fraction digits (KRW=0dp).

In-scope offenders (paper-trading surfaces, the criterion's target):
- positions-columns.tsx:152 `` `$${...toFixed(2)}` `` (Entry USD branch)
- positions-columns.tsx:261 `` `$${sl.toFixed(2)}` `` (Stop USD branch)
- positions-columns.tsx:64-79 CurrentPriceCell NumberFlow USD branch: inline format
  object + `locales={isUsd ? undefined : ...}` (:74) -> **browser default locale**.
  On the operator's nb-NO browser this renders "50 000,00 USD"-shape while :152
  renders "$50000.00" — the byte-exact mixed-locale mismatch in the screenshots.
- cockpit-helpers.tsx:72-104 shared `Dollar`: same `locales={isUsd ? undefined : ...}`
  pattern (:96). Consumed by positions/trades tables + SummaryHero MetricCards.

Repo-wide inventory (for scoping honesty): **41** hardcoded `` `$${...}` `` template
sites across frontend/src (page.tsx:68, backtest/page.tsx x5, ReportHeader.tsx x5,
CostDashboard, AltDataPanel, ComputeCostBreakdown, GlassBoxCards, LatestTransactionsBox,
performance, agents, manage ...) and **9** `toLocaleString(undefined, ...)` sites
(undefined locale = browser default). These are ALL genuinely-USD surfaces (LLM cost,
backtest equity, filings market-cap) — the money-CORRECTNESS fix only needs the
paper-trading positions surface; the rest is a locale-CONSISTENCY sweep. Recommend:
criterion 3 scope = positions/cockpit/trades components (replace templates with
formatCurrency/formatUsd, pin NumberFlow USD branches to en-US), plus a lint-style
regex test to freeze new offenders; defer the other ~35 dashboard sites to a follow-up
or fix mechanically in the same pass if cheap (they all reduce to formatUsd).

### 4. Non-US P&L staleness / mark-timestamp surface

- Stored `unrealized_pnl`/`unrealized_pnl_pct`/`market_value` are written ONLY by
  `mark_to_market` (paper_trader.py:539-556), called from the daily cycle
  (autonomous_loop.py:1001 step "mark_to_market", :1041 final_state, :1268) — i.e.
  once per trading day at the cycle hour.
- The frontend P&L cell (positions-columns.tsx:215-244) live-recomputes ONLY for US
  (`isUs && livePrice != null`); non-US rows render `pos.unrealized_pnl_pct` — the
  frozen mark. Meanwhile the Current cell (:169-180) shows the LIVE local price with
  an age band. **A KR row therefore shows a live price beside a P&L up to ~24h
  (weekend: ~72h) stale, unlabeled.**
- FX staleness compounds: `_fx_local_to_usd` live path caches 6h (fx_rates.py:53) and
  the stored mark embeds the FX of mark time.
- **The mark timestamp is not persisted anywhere per-position**: the `updates` dict
  (:539-546) carries no timestamp; `last_analysis_date` is stamped only on BUY (:308);
  portfolio-level `updated_at` (:576) is the closest existing surface but is
  portfolio-wide, not per-position.
- Fix design: (a) add `marked_at` (ISO) to the mark_to_market `updates` dict + ship it
  through the positions API (passthrough of BQ row) + `PaperPosition` type; render an
  as-of chip on non-US P&L cells (reuse the existing `bandFromAgeSec` idiom,
  positions-columns.tsx:170); `_safe_save_position`'s schema-retry pattern
  (`_POSITION_RT_FIELDS`, :1149) is the established idiom for new-column tolerance +
  a `scripts/migrations/` column add. Or (b) frontend-only: label non-US P&L with the
  live-price age band and a "marked at last cycle" tooltip — zero schema change but
  the timestamp is inferred, not honest. Recommend (a); it is the only path that makes
  the criterion's "as-of indicator (mark timestamp)" literally true.

### 5. KRX/XETRA close vs the single cycle; post-close mark job

- Cycle: APScheduler cron at `paper_trading_hour`:00 ET mon-fri
  (paper_trading.py:1300-1322), default 10 ET (settings.py:341) = 14:00 UTC in DST /
  15:00 UTC in winter. **The goal doc's "18:00 UTC" does not match the settings
  default; `backend/.env` may override `PAPER_TRADING_HOUR` — researcher sandbox is
  denied backend/.env, so Main must verify the live value at contract time.**
- Exchange closes: KRX 15:30 KST = **06:30 UTC** (no DST). XETRA 17:30 CET/CEST =
  16:30 UTC winter / **15:30 UTC summer**.
- At a 14:00-15:00 UTC cycle: KR marks use prices ~7.5-8.5h after KRX close (that
  day's close — acceptable same-day) but the NEXT KR session runs 00:00-06:30 UTC
  BEFORE the next cycle, so **stops are checked against marks that are a full KR
  session behind**; gap risk realized at the KR open is invisible until ~8h later.
  XETRA summer close 15:30 UTC lands almost exactly ON a 14:00 UTC cycle — EU marks
  can be intraday-of-close; winter close 16:30 UTC means the mark precedes the close
  by ~1.5h. A ~07:00 UTC job cleanly post-dates both the KRX close and nothing else.
- What a `~07:00 UTC post-KRX-close mark job` touches (all existing idioms):
  1. `_add_scheduler_job` (paper_trading.py:1300) — add a second `add_job` with its
     own id (registry pattern `_scheduler_job_id`), `timezone=ZoneInfo("UTC")` or
     Asia/Seoul, `misfire_grace_time=3600, coalesce=True` (copy :1313-1322 rationale).
  2. Job body: `mark_to_market()` alone refreshes marks/MFE/stop-advancement
     (:500-583). If the gap is to be CLOSED (stops enforced, not just marked), the
     job must also run the stop-execution block — that logic lives inline in
     autonomous_loop (:1052-1105, backfill + `check_stop_losses` + sell loop), NOT in
     a reusable method: closing fully requires extracting it or accepting mark-only.
  3. Gate on `is_trading_day(<local date>, "KR")` (markets.py:192-211, the 50.4
     is_session fix) — use the KR-local date, per phase-50.4 lunar-holiday finding.
  4. Introspection is free: the job appears in `/api/jobs/all` via
     `_register_cron_scheduler("main", ...)` (main.py:276).
  5. Restart safety: MemoryJobStore + forward-only next-fire (phase-61.1 finding)
     applies equally; no double-fire risk on kickstart.
- Decision input the criterion asks for: mark-only at 07:00 UTC is low-risk (pure
  refresh; MFE/trail advancement uses fresher data = strictly better) but does NOT
  execute stops — KR stop EXECUTION still waits for the main cycle. Full closure
  needs the sell-loop extraction (bigger diff into "untouchable except money-safety"
  territory: goal bans touching the stop engine beyond the averaging fix). Reasoned
  recommendation below in Application.

## Consensus vs debate (external)

**Consensus:** transaction-date FX for historical cost, never current-FX re-derivation (IAS 21); pin the locale for deterministic currency rendering (MDN + Next.js + GitHub issue corpus); make currency identity explicit in mixed-currency views (Workday, W3C); label the freshness of any stale/converted value (Zigpoll, stale-price literature).

**Debate:** (a) mutual funds ADJUST stale foreign closes with fair-value factors vs merely labeling them — for a paper book, adjustment is overkill; labeling + a better-timed mark captures the value at near-zero risk. (b) UX: Zigpoll leans "one currency at a time with a toggle"; Workday endorses simultaneous local+base display with codes. For a trading positions table, local price columns + USD aggregate columns is the domain convention (broker statements do exactly this) — keep the existing shape.

## Pitfalls (from literature)

1. Forcing `minimumFractionDigits: 2` on KRW renders "₩1,234,567.00" — already correctly avoided in `formatCurrency` (format.ts:184-186 comment); do not reintroduce via NumberFlow inline format objects.
2. `suppressHydrationWarning`/useEffect-gating are escape hatches (Next.js docs) — the correct fix for locale nondeterminism is pinning, not suppression.
3. Fix ORDER matters: if display resolution is fixed before the averaging bug and the bug then fires, a corrupted 37.58 avg renders as "₩38" — visibly absurd (good for detection) — but the stop is still silently destroyed. The averaging fix is the money-safety item; display is honesty. Ship both in one step (they are independent diffs).
4. If the averaging bug has EVER fired, stored rows need repair, not just code fix. Pre-GENERATE check: BQ `SELECT ticker, market, avg_entry_price, cost_basis/quantity AS usd_ps FROM financial_reports.paper_positions WHERE market != 'US'` — corruption signature is `avg_entry_price ≈ cost_basis/quantity` (ratio ~1) instead of ~fx. Portfolio has been 100% cash since 2026-07-03, so expected result is zero rows; confirm and record in the contract.
5. US byte-identity: every change must be a no-op for US rows (`_usd_to_local == 1`, resolveMarket == "US"); tests must assert the US path unchanged (phase-50.2 discipline).
6. A second APScheduler job must copy the misfire_grace_time=3600/coalesce=True rationale (paper_trading.py:1313-1322) and gate on the KR-LOCAL calendar date (phase-50.4 lunar-holiday finding), else it fires on KR holidays or double-marks after downtime.
7. `-k` test-selection trap (phase-59.1 lesson): the immutable verification command selects `-k 'addon or avg_entry or currency or 61_3'` — name the test file `test_phase_61_3_*.py` with `addon`/`currency` in test function names so the filter provably matches.

## Application to pyfinagent (recommended fix design per criterion)

**Criterion 1 — LOCAL-currency add-on averaging (paper_trader.py:286-302).**
Keep `cost_basis`/`new_cost` in USD exactly as-is (IAS-21-consistent: sums of transaction-date USD amounts). Change ONLY the avg_entry_price computation to a quantity-weighted LOCAL average:
`new_avg_local = (old_qty * existing["avg_entry_price"] + quantity * exec_price) / new_qty`.
US: prices==USD, so this equals the old result only when the old result was right; strictly it now weights by price paid — assert US equivalence in tests via fx=1 rounding tolerance. Do NOT touch `_advance_stop`/stop engine (goal declares it untouchable beyond this fix; the fix lands upstream so stops receive a sane LOCAL entry).

**Criterion 2 — market-first currency resolution on LOCAL price columns.**
Add `resolveLocalCurrency({market, ticker})` to format.ts (market-first by contract, ignores base_currency, which describes the USD columns per types.ts:653-657 and is hardcoded "USD" at paper_trader.py:313/:334/:481) and use it at positions-columns.tsx:144-148 (Entry), :169-173 (Current), :254-258 (Stop). Leave `resolveCurrency` for genuinely-explicit surfaces. Backend alternative (persisting a `local_currency` column) is more invasive for zero display gain — rejected.

**Criterion 3 — one en-US USD locale policy.**
(a) Replace `` `$${x.toFixed(2)}` `` at positions-columns.tsx:152/:261 with `formatCurrency(x, cur)` (drop the USD special-case ternary entirely — formatCurrency already pins en-US for USD via CURRENCY_LOCALE).
(b) Pin NumberFlow USD branches: CurrentPriceCell (positions-columns.tsx:74) and `Dollar` (cockpit-helpers.tsx:96) — change `locales={isUsd ? undefined : numberFlowLocale(cur)}` to always pass `numberFlowLocale(cur)` (which returns "en-US" for USD via localeForCurrency), and use `numberFlowFormat(cur)` for both branches. This also removes a live SSR-hydration hazard (Node en-US vs browser nb-NO — MDN/Next.js finding 2).
(c) Scope: the paper-trading surfaces only (positions-columns, cockpit-helpers, trades table if it shares Dollar). The repo-wide 41 `` `$${...}` ``/9 `toLocaleString(undefined,...)` sites are genuinely-USD dashboards — sweep mechanically only if cheap, else record as explicit deferral in the contract (scope honesty).

**Criterion 4 — non-US P&L staleness honesty.**
Add `marked_at` (ISO UTC) to the mark_to_market `updates` dict (paper_trader.py:539-546); add the column via `scripts/migrations/` (established pattern) and to `_POSITION_RT_FIELDS`-style schema-retry pruning (:1149) so pre-migration saves don't break; pass through the positions API (row passthrough) + `PaperPosition` type (types.ts) ; render an as-of chip/tooltip on non-US P&L cells (reuse the `bandFromAgeSec` age-band idiom at positions-columns.tsx:170). Frontend-only inference was considered and rejected — it cannot make the "as-of indicator (mark timestamp)" criterion literally true.

**Criterion 5 — per-market mark-to-market decision (researcher-grounded input).**
Recommend: **mark-only job at 07:00 UTC gated on `is_trading_day(KR-local-date, "KR")`** (markets.py:192-211), added as a second `add_job` in `_add_scheduler_job` (paper_trading.py:1300) with the :1313-1322 misfire/coalesce rationale copied. Rationale: KRX closes 06:30 UTC; the next KR session (00:00-06:30 UTC) completes BEFORE the next default cycle (10:00 ET = 14:00/15:00 UTC), so stop checks run against marks a full session old. A 07:00 UTC mark refreshes marks+MFE+stop-advancement (mark_to_market already does all three, :500-583) with fresher data — strictly better, no new execution authority. What it does NOT do: EXECUTE stops (that sell-loop lives inline in autonomous_loop.py:1052-1105, not reusable); extracting it violates the goal's "trailing-stop engine untouchable" boundary. Therefore: close the MARK gap now, explicitly DEFER stop-execution-at-07:00 with this rationale — the criterion text permits exactly this ("closing or explicitly deferring ... with rationale"). XETRA (15:30/16:30 UTC close) is adequately served by the existing cycle. NOTE for contract: goal text says "18:00 UTC cycle" but settings default is 10:00 ET (settings.py:341) — Main must read `PAPER_TRADING_HOUR` in backend/.env (researcher sandbox is denied that file) and pin the true cycle time in the contract.

**Execution risk for the live_check:** the criterion demands a Playwright capture of "the live KR position" — the portfolio is 100% cash since 2026-07-03. If no KR position exists when 61.3 runs, the live_check needs either the loop to re-enter a KR name first, or an operator-acknowledged seeded paper position; flag in the contract rather than discovering at EVALUATE.

## Regression-test shapes (per the immutable criteria)

1. **KR add-on KRW-scale** (`backend/tests/test_phase_61_3_addon_averaging.py`, names containing `addon`): monkeypatch `_fx_local_to_usd`/`_fx_usd_to_local` (1/1350, 1350) + stub BQ; `execute_buy` twice on an `.KS` ticker (50,000 then 52,000 KRW); assert `avg_entry_price == pytest.approx(quantity-weighted KRW mean)` and `min(p1,p2) <= avg <= max(p1,p2)` (KRW-scale bound); then drive `mfe_pct` past `paper_default_stop_loss_pct` and assert `_advance_stop` returns a KRW-scale stop `== avg_entry_price`. Companion test: US add-on with fx=1 asserts byte-identical old-vs-new averaging.
2. **No-USD-symbol-on-KRW regex** (vitest, mirrors the 60.3 prompt regex test): render Entry/Current/Stop cells with a KR row that carries `base_currency: "USD"` (exactly what the backend ships) and a KRW-magnitude value; assert rendered text matches `/₩|KRW/` and does NOT match `/\$\s?\d/`.
3. **Forced nb-NO default locale** (vitest): wrap/mock global `Intl.NumberFormat` so `locales === undefined` resolves as `nb-NO`, then assert `formatCurrency(1234.56, "USD") === "$1,234.56"` and that CurrentPriceCell/Dollar USD branches still render en-US shape (i.e., they pass an explicit locale, never undefined). Prop-level assertion on NumberFlow `locales="en-US"` is an acceptable stricter variant.
4. **marked_at persistence**: pytest asserts `mark_to_market` writes `marked_at` (fresh ISO within test tolerance) on every position row; frontend test asserts the as-of chip renders for a non-US row and NOT for a US live-priced row.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL (7: 4 via WebFetch, 1 W3C via WebFetch, 1 Workday via WebFetch, 1 Duke paper via curl+PDF-extraction per the sanctioned chain)
- [x] 10+ unique URLs total (~44 collected across 5 searches)
- [x] Recency scan (2024-2026) performed + reported (3 findings, none superseding)
- [x] Full pages/papers read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (re-verified 2026-07-08 working tree)

Soft checks:
- [x] Internal exploration covered every relevant module (paper_trader, fx_rates, autonomous_loop, scheduler/api, settings, markets, format.ts, positions-columns, cockpit-helpers, types)
- [x] Contradictions/consensus noted (fair-value adjust-vs-label; toggle-vs-simultaneous UX)
- [x] Claims cited per-claim
- Gap noted honestly: `backend/.env` cycle-hour override unreadable from this sandbox (denied); Main must verify `PAPER_TRADING_HOUR` at contract time. ifrscommunity/iasplus/bogleheads fetches failed (403/empty) — substituted same-tier alternatives.

## JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 37,
  "urls_collected": 44,
  "recency_scan_performed": true,
  "internal_files_inspected": 13,
  "report_md": "handoff/current/research_brief_61.3.md",
  "gate_passed": true
}
```
