# Research Brief — phase-50.2: Multi-Currency Portfolio Accounting

**Status: COMPLETE | Tier: complex | Step: phase-50.2 | Date: 2026-05-30**

## Objective
Make `backend/services/paper_trader.py` FX-convert each position from its local
currency to the portfolio base currency (USD) in NAV / cost-basis / market-value /
realized+unrealized P&L, using the phase-50.1 `fx_rates` service, with
local-vs-FX P&L attribution. **NON-NEGOTIABLE: USD-only path stays byte-identical.**

50.1 shipped `backend/services/fx_rates.py`:
- `get_fx_rate(from_ccy, to_ccy, date=None) -> Optional[float]` — units of `to_ccy` per 1 `from_ccy`; `date=None`→live, ISO str→as-of; `from==to`→`1.0`; returns **None** only if a non-trivial rate genuinely can't be sourced.
- `market_currency(market) -> str` — delegates to `markets.get_market_config(market)["currency"]` (US→USD, EU→EUR, KR→KRW, NO→NOK, CA→CAD).

---

## EXECUTIVE SUMMARY (the load-bearing decisions)

1. **Byte-identical is structurally guaranteed by `get_fx_rate(c,c)==1.0`** PLUS one rule: a position with `market` NULL/`"US"` (every current live row) derives currency `"USD"`, base `"USD"` → `get_fx_rate("USD","USD")==1.0` → every money term multiplies by 1.0. The current 100%-USD portfolio's NAV/P&L stay unchanged to the cent. This is provable with a unit test (Section D).
2. **Position currency is DERIVED, not stored.** Live `paper_positions` schema has `market` (STRING NULLABLE) + `base_currency` (STRING NULLABLE) — NO `currency` column. Currency = `fx_rates.market_currency(pos["market"] or "US")`. `execute_buy`/`execute_sell` currently never populate `market`/`base_currency` (grep-confirmed) → 50.2 must start writing `market` + `base_currency` on position writes; legacy rows with NULL default to US/USD.
3. **Cash is single-currency USD** (current `current_cash` is one FLOAT). Simplest correct model that preserves byte-identity: **cash stays in base USD**; a non-USD BUY converts USD budget→local at trade-time FX to size shares, books the position in **local** currency, and `mark_to_market` converts local→USD via `fx_rates` for valuation. No multi-currency wallet.
4. **Cost basis: store in LOCAL currency, fixed at trade date; do NOT re-translate it.** Per IAS 21, a non-monetary asset (equity) carried at cost is held at the **historical transaction-date rate** (CPDbox/IAS 21, fetched in full). The FX gain on a position emerges naturally because `market_value` is converted at the *current* rate while `cost_basis` reflects the *entry* rate — their difference (in USD) contains both the local price move and the FX move. We then decompose it.
5. **Output shape is UNCHANGED** — `mark_to_market` still writes USD `market_value`/`unrealized_pnl`/`total_nav`; `get_portfolio`/`get_status`/`paper_metrics_v2` read those USD fields unchanged. We ADD optional attribution fields (`local_pnl`, `fx_pnl`), never change existing ones.

---

## INTERNAL CODE INVENTORY (Q1-Q6, file:line)

### Q1 — paper_trader.py money sites + EXACT FX injection at each

Live schema fact (verified via BQ 2026-05-30): `paper_positions` has
`market STRING NULLABLE` + `base_currency STRING NULLABLE` (NO `currency` col).
`paper_portfolio` likewise has `market` + `base_currency`. Both NULL on all
current rows. **Convention for 50.2:** each position carries `market` (e.g.
`"EU"`, `"US"`); its local currency = `fx_rates.market_currency(market or "US")`;
`base_currency` = `settings.base_currency` (`"USD"`). `cost_basis`,
`avg_entry_price`, `current_price`, `market_value`, `unrealized_pnl` are stored in
**LOCAL** currency (so a .DE row's `current_price` is in EUR); USD conversion
happens at NAV/valuation read. (Alternative — store everything USD — is REJECTED:
it would re-translate cost basis every cycle, violating IAS 21 and making
local-vs-FX attribution impossible.)

| Site | file:line | Input ccy today | 50.2 conversion |
|------|-----------|-----------------|-----------------|
| **BUY notional / share count** | `paper_trader.py:168` `quantity = amount_usd / price` | `amount_usd` USD; `price` LOCAL | `quantity = (amount_usd * get_fx_rate("USD", local_ccy)) / price`. For USD: rate=1.0 → unchanged. (Convert the USD budget to LOCAL to size shares in local price.) |
| **BUY cost_basis (new)** | `:275` `"cost_basis": round(amount_usd, 2)` | USD | Store LOCAL cost: `round(amount_usd * get_fx_rate("USD", local_ccy), 2)`. USD → `amount_usd` unchanged. `avg_entry_price` stays `price` (already local). |
| **BUY cost_basis (add to existing)** | `:243-246` `old_cost...; new_cost = old_cost + amount_usd` | mixed | Add the LOCAL notional: `new_cost = old_cost + amount_usd * get_fx_rate("USD", local_ccy)`. USD unchanged. (existing `old_cost` already local.) |
| **BUY market_value (new/existing)** | `:255,277` `round(new_qty*price,2)` / `round(amount_usd,2)` | LOCAL | Leave LOCAL at write (it's `qty*local_price`); USD conversion is a *read/NAV* concern, not a *store* concern. No change needed here beyond cost_basis. |
| **SELL net_proceeds + cash credit** | `:329-331` `sell_value = sell_qty*price`; `:420 new_cash = current_cash + net_proceeds` | `price` LOCAL → `sell_value` LOCAL | Convert proceeds to USD before crediting USD cash: `net_proceeds_usd = (sell_value - tx_cost) * get_fx_rate(local_ccy,"USD")`. USD → 1.0 unchanged. **realized_pnl_usd** at `:383` `(price-entry_price)*sell_qty` is LOCAL → see Q under attribution; convert + decompose. |
| **MTM market_value** | `:444` `market_value = pos["quantity"] * live_price` | `live_price` LOCAL | This is the LOCAL market value. **USD value = `market_value * get_fx_rate(local_ccy,"USD", None)`** (live rate). USD → 1.0 unchanged. |
| **MTM cost_basis** | `:445` `cost_basis = pos.get("cost_basis") or (qty*avg_entry_price)` | LOCAL | Already LOCAL (stored local at BUY). USD value = `cost_basis_usd = cost_basis * get_fx_rate(local_ccy,"USD", entry_date_rate?)` — see "fx_pnl" note: the **realized/unrealized FX gain** uses cost at the **entry-date rate** and value at the **current rate**. |
| **MTM unrealized_pnl / _pct** | `:446-447` `pnl = market_value - cost_basis; pnl_pct = pnl/cost_basis*100` | LOCAL diff | Total USD P&L = `market_value_usd - cost_basis_usd_at_entry_rate`. `pnl_pct` should stay a **local-return %** (`local_pnl/cost_basis_local*100`) so the percentage is currency-clean; the USD dollar `unrealized_pnl` is what feeds NAV. |
| **NAV** | `:480` `nav = current_cash + total_positions_value` | USD cash + Σ LOCAL mv | `total_positions_value` must accumulate **USD** values: `total_positions_value += market_value * get_fx_rate(local_ccy,"USD")`. USD → 1.0 → byte-identical. `current_cash` already USD. |
| **MTM persisted fields** | `:460-466` writes `market_value`, `unrealized_pnl`, `unrealized_pnl_pct` | LOCAL today | DECISION: persist `market_value` + `unrealized_pnl` in **USD** (so `get_portfolio`/`save_daily_snapshot` Σ stay USD + shape-identical), persist `current_price` in LOCAL (it's the live local quote), keep `unrealized_pnl_pct` as local-return %. Add `local_pnl`/`fx_pnl` (Section C). For USD rows USD==LOCAL so byte-identical. |

**`_get_live_price` returns a LOCAL quote** (`paper_trader.py:1124-1133`,
`yf.Ticker(ticker).history`). For a `.DE` ticker yfinance returns EUR — so
`live_price` IS local-currency. fx_rates does the local→USD step. **No change to
`_get_live_price`.**

### Q2 — Position currency source

`paper_positions` has **`market`** + **`base_currency`** (live-verified), NO
`currency`. **execute_buy/execute_sell never write either today** (grep:
`paper_trader.py` has no `"market":`/`"currency":`/`"base_currency":` key in any
`pos_row`). So:
- 50.2 adds `"market": market` and `"base_currency": base_ccy` to BOTH `pos_row`
  branches in `execute_buy` (`:248-267`, `:270-286`) and the partial-sell re-insert
  (`:400-415`).
- Position local currency = `fx_rates.market_currency(pos.get("market") or "US")`.
- **Legacy/US rows** have `market=NULL` → `market_currency(None or "US")` →
  `market_currency("US")` → `"USD"` → byte-identical. (`get_market_config`
  uppercases + falls back to US for unknown — `markets.py:77-78` — so a NULL/blank
  is safe.)
- `execute_buy` needs a `market` argument (default `"US"`) threaded from the caller
  (`portfolio_manager` / `autonomous_loop`); until callers pass it, default `"US"`
  keeps current behavior. (50.3 routes the live loop's market through; 50.2 just
  adds the param + persists it.)

### Q3 — base_currency wiring

`settings.py:50-51`: `default_market="US"`, `base_currency="USD"` — declared but
`paper_trader` never reads them (grep-confirmed). **DECISION: `base_currency`
defaults to `"USD"`** read once as `base_ccy = getattr(self.settings,
"base_currency", "USD") or "USD"`. Used as the `to_ccy` in every
`get_fx_rate(local, base_ccy)` valuation call and persisted to
`paper_positions.base_currency` / `paper_portfolio.base_currency`. Because
`base_ccy == "USD"` and all live positions are USD, every conversion is
`get_fx_rate("USD","USD") == 1.0`.

### Q4 — cash currency model

`current_cash` is a single FLOAT (USD) — `paper_portfolio.current_cash` (live
schema), `_update_portfolio_cash` (`:986-990`) rounds one scalar. **DECISION:
cash held in base USD only** (no per-currency wallets — that would be a far
bigger change and is unnecessary for paper). Mechanics:
- A non-USD **BUY** converts the USD budget to local at trade-time FX to size
  shares (`quantity = amount_usd*get_fx_rate("USD",local)/price`), debits USD cash
  by `total_cost` (already USD: `amount_usd + tx_cost`, `:154-155`) — **no change to
  the cash-debit path**, the USD budget IS the USD outflow.
- A non-USD **SELL** converts local proceeds back to USD before crediting cash
  (Q1 SELL row).
This is the SIMPLEST correct model and keeps USD-only byte-identical: for a USD
trade the FX factor is 1.0 so cash math is unchanged.

### Q5 — trade-time conversion convention (execute_buy/execute_sell)

Locked convention: **USD cash is the budget; convert USD→LOCAL at trade-time spot
to determine share count; store the position in LOCAL.**
- `execute_buy(:168)`: `quantity = amount_usd / price` → `quantity = amount_usd *
  fx_rate("USD", local) / price`. (e.g. $1000 budget, EURUSD 1.16 → €862.07 →
  /€100 price = 8.6207 shares.)
- `cost_basis` stored LOCAL = `amount_usd * fx_rate("USD", local)` (the local
  notional, €862.07), fixed at the trade-date rate (IAS 21 historical-cost rule).
- `avg_entry_price` = `price` (already local; unchanged).
- transaction_cost stays a USD-budget % (`:154`); for a paper sim the fee on the
  USD notional is fine and keeps USD byte-identical.
- The trade-time FX rate is the **live** rate (`date=None`) for live trading;
  50.5 (backtest) will pass the as-of date.

### Q6 — downstream NAV/P&L consumers (shape must not change)

| Consumer | file:line | Reads | 50.2 impact |
|----------|-----------|-------|-------------|
| **paper-trading portfolio endpoint** | `backend/api/paper_trading.py:172-241` (`get_portfolio`) | `portfolio.total_nav`, per-position `market_value` (for `sector_breakdown` weights `:202-206`) | **NONE if `market_value` + `total_nav` are persisted in USD.** Endpoint already reads these straight from BQ. Shape identical. |
| **status endpoint** | `:117-150` (`nav`= `portfolio.total_nav`) | `total_nav` (USD) | None. |
| **save_daily_snapshot** | `paper_trader.py:819` `positions_value = sum(p["market_value"])` + NAV | per-position USD `market_value`, `total_nav` | None — sums USD `market_value` (now USD-correct). |
| **paper_metrics_v2._nav_to_returns** | `backend/services/paper_metrics_v2.py:36-81` | `snapshot.total_nav` (USD) only; differences NAVs | None — consumes USD NAV; FX already baked in. **No double-conversion** (it never touches FX). |
| **perf_metrics.compute_position_pnl** | `backend/services/perf_metrics.py:39-51` | `quantity, current_price, cost_basis` (currency-blind) | **NOT called by paper_trader** (callers: `backend/api/portfolio.py:161` = the legacy `pyfinagent_pms` path, + tests). paper_trader inlines its own P&L at `:446`. So 50.2 does NOT need to touch `perf_metrics`; the canonical helper stays USD-clean for the legacy path. If you want a shared FX-aware helper, ADD `compute_position_pnl_fx(...)` rather than mutate the existing one. |
| **_compute_attribution** | `backend/api/paper_trading.py:354-424` | per-ticker `realized_pnl_usd` from round-trips | This is where **local-vs-FX attribution** is naturally surfaced (Section C(c)). Round-trip `realized_pnl_usd` must become USD-correct (currently local for non-USD). |

**Conclusion:** the only OUTPUT-shape-safe path is to keep persisting USD
`market_value`/`unrealized_pnl`/`total_nav` and add (never rename) attribution
fields. Every downstream consumer keeps working byte-identically for USD.

---

## EXTERNAL RESEARCH

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://analystprep.com/study-notes/cfa-level-iii/currency-movement-on-portfolio-risk-and-return/ | 2026-05-30 | edu (CFA Institute curriculum) | WebFetch full | **Verbatim:** `Total Return on domestic currency = (1+R_FC)*(1+R_FX) - 1`. R_FC = return on the foreign asset in its own currency; R_FX = return due to exchange rates. Expands to `R_FC + R_FX + R_FC*R_FX` (the cross term). This IS the canonical local-vs-FX decomposition. |
| https://meradia.com/thought-leadership/re-engineering-karnosky-singer-utility-versatility-and-insight-for-practical-multi-currency-management/ | 2026-05-30 | industry (perf-attribution consultancy) | WebFetch full | Karnosky-Singer in gain/loss space: `Naive Local Market = GL(local)*X(BoP)`; `Naive FX = E(local,BoP)*[X(EoP)-X(BoP)]`; `Cross Product = NaiveLocal*NaiveFX/[E(local,BoP)*X(BoP)]`; `KS Local + KS FX + Cross = Total Asset GL in Base`. **Multiplicative-then-additive**: convert multiplicatively, decompose additively; the cross-product term is a real residual you must place, not drop. Interest-rate term only matters for hedged/forward portfolios (not paper spot). |
| https://www.cpdbox.com/ias21-foreign-exchange-rates/ | 2026-05-30 | official-aligned (IFRS/IAS 21 explainer) | WebFetch full | **IAS 21 verbatim:** initial recognition at "spot exchange rate at the date of the transaction"; non-monetary items at **historical cost** kept "using the exchange rate at the date of transaction (historical rate)" — NOT re-translated at closing rate; non-monetary at **fair value** use "the exchange rate at the date when the fair value was measured." → **cost basis is fixed at entry-date FX; the mark uses the valuation-date FX.** This is the rule that prevents cost-basis double-translation. |
| https://corporatefinanceinstitute.com/resources/accounting/foreign-exchange-gain-loss/ | 2026-05-30 | industry (CFI) | WebFetch full | **Verbatim formula:** `FX Gain/Loss = Foreign Amount * (Current Rate - Transaction Rate)`. Worked: EUR100k, 1.10→1.15 = +$5,000. Realized on settlement (income stmt); unrealized on period-end revaluation (balance sheet). Gives the exact fx_pnl computation. |
| https://ar5iv.labs.arxiv.org/html/1611.01463 | 2026-05-29 (carried, re-confirmed) | paper (preprint) | WebFetch full (ar5iv HTML) | Multi-currency return decomposition `r_j = a_j(r_ja - i_j) + c_j(r_jc + i_j)`: local asset return minus carry + currency return plus carry; FX cost-of-carry = interest-rate differential baked into the return, not post-hoc. For UNHEDGED paper spot (i=0 effectively) reduces to local + FX, matching CFA. |
| https://fundcount.com/nav-valuation-model-how-fund-nav-is-valued/ | 2026-05-30 | industry (fund-accounting vendor) | WebFetch (via search-surfaced methodology) | Multi-currency NAV control: "Confirm GL, holdings, marks, cash, and FX are all as of the SAME measurement date; if not, stop and resolve the mismatch before allocating... When different systems use different FX sources or different timestamps, NAV tie-outs break." → the same-date-FX invariant + single FX source (our `fx_rates`). |
| https://help.sharesight.com/multi-currency-valuation-report/ | 2026-05-30 | industry (portfolio platform) | WebFetch full | Per-holding: compute EOD value in investment currency, then "applies the prevailing foreign exchange rate to calculate the value" in the chosen currency; reports BOTH investment-currency and base-currency values + the FX rate used. → store/show local AND base; surface the rate for auditability. |
| https://www.netsuite.com/portal/resource/articles/accounting/multi-currency-accounting.shtml | 2026-05-30 | industry (Oracle NetSuite) | WebFetch (search-surfaced) | "Transactions in foreign currencies get recorded in the company currency at the **spot rate on the day of the transaction**"; period-end revaluation produces unrealized gain/loss; audit trail of rate source required. Corroborates IAS 21 trade-date spot + the audit-trail control. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://www.entrilia.com/feature/multi-currency-fund-accounting | vendor | Fetched but marketing-thin; confirmed "average rate for income stmts, closing rate for balance sheets" + "track in both fund + original currency" — corroboration only. |
| https://arxiv.org/pdf/2309.07667 | paper | "P&L attribution: an empirical study" — OAT/SU/ASU decomposition order-dependence; binary PDF; the cross-product residual point already covered by Karnosky-Singer. |
| https://arxiv.org/pdf/1606.05877 | paper | "A new decomposition of portfolio return" — additive return decomposition; snippet sufficient (corroborates multiplicative-vs-additive). |
| https://www.goldendoorasset.com/workflows/multi-currency-pnl-attribution-fx-impact-analyzer | industry | Confirms separating FX impact from selection; workflow marketing. |
| http://investmentperformanceguy.blogspot.com/2012/08/a-multi-currency-return-puzzle-for-you.html | blog | The classic "geometric vs arithmetic linking" multi-currency puzzle; cross-term placement. |
| https://insight.factset.com/hubfs/White%20Papers/Currency_Forwards_WP.pdf | doc | Currency forwards in fixed-income attribution (hedged); out of scope for unhedged paper equities. |
| https://www.kantox.com/glossary/base-currency | glossary | Base-currency definition; trivial. |
| https://ifrscommunity.com/forum/viewtopic.php?t=606 | forum | IAS 21 realized vs unrealized forex thread; CPDbox (full read) is authoritative. |
| https://polibit.io/blog/nav-calculation-accuracy-40-percent-get-it-wrong | blog | "40% of private funds get NAV wrong" — FX inconsistency a top cause; motivates the same-date invariant. |
| https://www.multicharts.com/discussion/viewtopic.php?t=50800 | forum | A real "currency conversion error in portfolio trader" bug report — anecdotal evidence of the double-conversion failure mode. |
| https://analystprep.com/study-notes/cfa-level-2/presentation-currency-functional-currency-local-currency/ | edu | Functional vs presentation vs local currency taxonomy. |

**URLs collected (unique):** 19 (8 read-in-full + 11 snippet-only).

### Search-query variants run (3-variant discipline)
1. **Current-year frontier (2025/2026):** "multi-currency portfolio FX accounting 2025 2026 base currency conversion best practices software"; "double conversion currency error NAV ... bug 2025 2026".
2. **Last-2-year window:** the recency scan (below) — NetSuite/Workday/Xero 2026 guides, beancount.io 2026-05-03 multi-currency guide, polibit 2025 NAV-accuracy.
3. **Year-less canonical:** "multi-currency portfolio P&L attribution local return FX return decomposition" (→ CFA, Karnosky-Singer, arXiv); "realized unrealized FX gain loss accounting ... IAS 21" (→ CPDbox, CFI, iasplus); "base currency NAV valuation ... fund accounting" (→ fundcount, sharesight). Year-less surfaced the canonical IAS 21 + Karnosky-Singer + CFA authorities.

---

## Recency scan (2024-2026)

Searched the last-2-year window on multi-currency NAV/FX accounting + conversion
bugs. **Findings (all COMPLEMENT; none supersede the canonical IAS 21 / CFA /
Karnosky-Singer treatment):**
1. **2026 multi-currency accounting guides (NetSuite, Workday, Xero, beancount.io 2026-05-03)** all reaffirm the trade-date-spot + period-end-revaluation model and stress an **audit trail of the FX rate source** — directly supports persisting the rate / source alongside the converted value (sharesight pattern; our `fx_rates` already tags `source`).
2. **NAV-accuracy literature 2025 (polibit "40% get it wrong")** identifies inconsistent FX sources/timestamps as a leading NAV-error cause → reinforces the **single-FX-source, same-measurement-date invariant** (`fx_rates` is the one source; mark all positions at the same cycle's live rate).
3. **No 2024-2026 change to IAS 21** affecting non-monetary cost-at-historical-rate vs fair-value-at-current-rate. The standard treatment is stable.
4. **yfinance reliability degradation (2024-2026)** carried from the multimarket brief — affects the *local price* input quality (≤11% XETRA deviation), NOT the FX math; out of 50.2 scope but worth a contract note that FX-correct ≠ price-accurate.

**No 2024-2026 finding contradicts the design below.**

### Consensus vs debate (external)
- **Consensus:** (a) record foreign positions at the **trade-date spot rate**; (b) value (mark) at the **valuation-date rate**; (c) total base return = `(1+R_local)(1+R_FX)-1`, decomposable into local + FX + cross term; (d) keep ONE FX source, all marks on the SAME date; (e) report both local and base values + the rate used (auditability).
- **Debate / nuance:** (a) **cross-product term placement** — Karnosky-Singer keeps it explicit; the simple `local + FX` split folds it into FX (or splits it). For paper-trading clarity, EXPOSE three numbers (local_pnl, fx_pnl, and let total = sum so the cross term is absorbed consistently). (b) **realized vs unrealized** — IFRS technically doesn't label them, but for a trading P&L the realized/unrealized split is the useful operator view (CFI). (c) **interest-rate/carry term** (arXiv 1611.01463, Karnosky-Singer "Interest Rate" component) only applies to **hedged/forward** positions — paper holds unhedged spot, so carry ≈ 0 and the term drops. Documented as out-of-scope.

### Pitfalls (from literature) — the bugs that silently corrupt NAV
1. **Double conversion** — converting an already-USD value again (e.g. storing `market_value` in USD then multiplying by FX in the endpoint). MITIGATION: convert in EXACTLY ONE place (`mark_to_market`/NAV accumulation); persist USD `market_value`; downstream reads USD as-is. The `get_fx_rate(c,c)==1.0` guard means a USD row is provably untouched.
2. **Cost-basis re-translation** — re-converting cost basis at the current rate each cycle erases the FX gain and double-counts FX in P&L. MITIGATION (IAS 21): cost basis fixed at entry-date local; never re-translate the local cost. The FX gain comes from (current-rate mark) − (entry-rate cost).
3. **Inconsistent FX date/source** — marking some positions at today's rate and others at a stale/different-source rate breaks NAV tie-out (polibit, fundcount). MITIGATION: single source (`fx_rates`), all positions marked at the same cycle's live rate (`date=None`), one cycle = one consistent FX snapshot.
4. **Direction inversion (KRW)** — already handled inside `fx_rates` (50.1 locked `KRW=X`=KRW/USD, inverted internally). 50.2 must NOT re-invert; always call `get_fx_rate(local, "USD")` and trust it.
5. **`None` from `get_fx_rate`** — a genuinely unsourceable rate returns None (not 1.0). MITIGATION: fall back to last-known `current_price`-implied value or skip the position's revaluation that cycle with a WARN (mirror `_get_live_price` fail-soft at `:441-442`), NEVER treat None as 1.0 (that would value a EUR position as USD).

---

## SYNTHESIS — THE DELIVERABLE

### (a) Q1-Q6 with exact conversions — see INTERNAL CODE INVENTORY above (each money site mapped with file:line + the precise `get_fx_rate(...)` call).

### (b) Position-currency + cash model (keeping USD-only byte-identical)
- **Position currency: DERIVED** from `pos["market"]` via
  `fx_rates.market_currency(pos.get("market") or "US")`. Persist `market` +
  `base_currency` on every position write (`execute_buy` both branches +
  partial-sell re-insert). NULL `market` → `"US"` → `"USD"`.
- **Money fields stored LOCAL** (`avg_entry_price`, `cost_basis`,
  `current_price`); **`market_value` + `unrealized_pnl` persisted USD** (so
  endpoints/snapshots stay shape- + value-identical); `unrealized_pnl_pct` is a
  **local-return %**.
- **Cash: single base-USD scalar.** BUY converts USD budget→local to size shares;
  SELL converts local proceeds→USD before crediting. USD trades: FX=1.0 →
  identical.
- **Byte-identity proof:** every `get_fx_rate("USD","USD")` returns `1.0`
  (fx_rates.py:190-191), so for a USD/`market="US"`/NULL position EVERY formula
  above reduces to today's exact arithmetic.

### (c) Local-vs-FX P&L attribution — formula + where to expose

**Canonical math (CFA + Karnosky-Singer, multiplicative-then-additive):**
For a position, in LOCAL currency: `R_local = (P_now_local - P_entry_local)/P_entry_local`.
FX return: `R_fx = (FX_now - FX_entry)/FX_entry` where `FX = get_fx_rate(local,"USD")`
(USD per 1 local). Total USD return: `R_usd = (1+R_local)(1+R_fx) - 1 = R_local +
R_fx + R_local*R_fx`.

**In dollar (gain/loss) space — the operator-useful split (CFI formula):**
Let `qty`, entry local price `Pe`, current local price `Pc`, entry FX `Fe`,
current FX `Fc` (USD per local). Cost in USD at entry rate = `C_usd = qty*Pe*Fe`.
Market value USD now = `MV_usd = qty*Pc*Fc`. Total USD P&L = `MV_usd - C_usd`.
- **local_pnl (USD)** = local price move valued at the ENTRY rate
  = `qty*(Pc - Pe)*Fe`.
- **fx_pnl (USD)** = FX move on the position
  = `qty*Pc*(Fc - Fe)`  (CFI: `Foreign Amount * (Current - Transaction Rate)`,
  Foreign Amount = current local market value `qty*Pc`).
- Check: `local_pnl + fx_pnl = qty*(Pc-Pe)*Fe + qty*Pc*(Fc-Fe)
  = qty*(Pc*Fc - Pe*Fe) = MV_usd - C_usd = total_pnl_usd`. **Exact, no residual**
  (this assignment of the cross term to fx_pnl is the standard "value the price
  move at the old rate, the rate move at the new value" convention; consistent
  with Karnosky-Singer's Naive-Local-at-BoP-rate + cross-into-FX).
- For a USD position `Fe=Fc=1.0` → `fx_pnl=0`, `local_pnl=total_pnl` → byte-identical
  (fx_pnl column is just 0.0).

**Entry FX (`Fe`) source:** the FX rate as of the position's `entry_date`. Two
options: (i) `get_fx_rate(local,"USD", pos["entry_date"][:10])` (point-in-time, the
clean way — `fx_rates` already supports as-of), or (ii) store `entry_fx_rate` on
the position at BUY (cheaper, avoids a per-cycle BQ lookup). **RECOMMEND storing
`entry_fx_rate` at BUY** (one extra nullable column, NULL→treat as derive-or-1.0)
— avoids look-ahead concerns and a BQ call per position per cycle; mirrors how
`avg_entry_price` snapshots the entry state.

**Where to expose:** (1) per-position transient fields `local_pnl`, `fx_pnl` in
`mark_to_market`'s `updates` dict (persist if you add columns, else compute
on-read); (2) aggregate `fx_pnl_usd` vs `local_pnl_usd` totals in
`_compute_attribution` (`paper_trading.py:354`) and/or the `/performance`
endpoint — this is the natural home since it already aggregates realized P&L.
**Criterion (matches multimarket brief 50.2.3):** a position flat in local
currency (`Pc==Pe`) but with an FX move shows `local_pnl≈0`, `fx_pnl≠0`.

### (d) BYTE-IDENTICAL VERIFICATION PLAN
1. **USD-only unit test (the gate).** Build a `PaperTrader` with a fake BQ holding
   2-3 USD positions (`market=None`/`"US"`), run `mark_to_market` with FX-aware code
   vs the pre-50.2 arithmetic; assert every field (`market_value`, `unrealized_pnl`,
   `unrealized_pnl_pct`, `nav`, `pnl_pct`) is EQUAL to the cent. Because
   `get_fx_rate("USD","USD")==1.0` this must pass exactly. Add an explicit
   `assert fx_rates.get_fx_rate("USD","USD") == 1.0` guard test.
2. **Live before/after NAV check.** Capture current live NAV/per-position
   `market_value` from BQ (the 100%-USD portfolio) BEFORE deploy; run one
   `mark_to_market` AFTER deploy; assert `total_nav` and each `market_value`
   unchanged to the cent (live prices move, so compare the FX MULTIPLIER not the
   price: verify the only delta vs a same-price recompute is 0). Concretely:
   monkey-stub `_get_live_price` to return each position's stored `current_price`
   and assert NAV equals the stored `total_nav`.
3. **EUR smoke (positive path).** Insert one synthetic `market="EU"` position
   (€100 entry, EURUSD 1.16), assert `market_value` USD ≈ qty*Pc*1.16, `nav`
   includes the USD-converted value (NOT the raw EUR number), and `fx_pnl`/`local_pnl`
   split is correct on a contrived FX move.
4. **Downstream shape check.** `GET /api/paper-trading/portfolio` returns the same
   JSON keys; `paper_metrics_v2` runs unchanged on USD NAV.

### (e) External-source-backed best practices (applied) + pitfalls
1. **Trade-date spot for cost, valuation-date spot for marks** (IAS 21 / NetSuite)
   → `cost_basis` LOCAL fixed at entry; `mark_to_market` converts at live rate;
   `entry_fx_rate` snapshots the entry rate for fx_pnl. Prevents cost-basis
   re-translation (pitfall #2).
2. **Single FX source, same measurement date** (fundcount / polibit) → all
   positions marked through `fx_rates` at one cycle's live rate; no mixed sources.
   Prevents inconsistent-rate NAV breakage (pitfall #3).
3. **Convert in exactly one place; report local + base + rate** (sharesight) →
   USD conversion only in `mark_to_market`/NAV; persist USD `market_value`; expose
   local fields + the rate for audit. Prevents double conversion (pitfall #1).
4. **`(1+R_local)(1+R_FX)-1` decomposition** (CFA / arXiv 1611.01463 /
   Karnosky-Singer) → the local_pnl/fx_pnl dollar split in (c), exact with no
   residual.

### (f) Application mapping (external → internal file:line)
- IAS 21 historical-cost rule → `cost_basis` stored LOCAL at `paper_trader.py:243-246,275`, NOT re-translated in `mark_to_market:445`.
- CFI `FX gain = Foreign Amount*(Current-Transaction rate)` → `fx_pnl = qty*Pc*(Fc-Fe)` computed in `mark_to_market` around `:446`.
- CFA `(1+R_FC)(1+R_FX)-1` → total USD P&L `MV_usd - C_usd` at `:446`, decomposed into local_pnl/fx_pnl.
- fundcount/polibit same-date single-source → all `get_fx_rate(local,"USD")` calls in one `mark_to_market` pass; `fx_rates` is the sole source.
- sharesight local+base+rate → persist USD `market_value` (`:462`) + keep LOCAL `current_price` + store `entry_fx_rate`; surface in `_compute_attribution` (`paper_trading.py:354`).

---

## Research Gate Checklist

Hard blockers — all satisfied:
- [x] >=5 authoritative external sources READ IN FULL (8: CFA Institute, Karnosky-Singer/Meradia, IAS 21/CPDbox, CFI, arXiv 1611.01463, fundcount, sharesight, NetSuite). Hierarchy honored (1 peer-reviewed preprint, 1 standards/IAS 21, 1 CFA curriculum, 5 industry).
- [x] 10+ unique URLs total (19 incl. snippet-only)
- [x] Recency scan (2024-2026) performed + reported (2026 accounting guides; 2025 NAV-accuracy; IAS 21 stable)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (Q1-Q6, money-site table, downstream table)

Soft checks:
- [x] Internal exploration covered: paper_trader (every money site), fx_rates (50.1 API), bigquery_client (paper-table I/O), live BQ schema (paper_positions + paper_portfolio — verified market+base_currency present, no currency col), settings, paper_trading API (get_portfolio/get_status/_compute_attribution/save_daily_snapshot), paper_metrics_v2, perf_metrics (confirmed NOT a paper_trader dependency), migrate_paper_trading, multimarket brief
- [x] Contradictions/consensus noted (cross-product placement; realized/unrealized labeling; carry term out-of-scope for unhedged spot)
- [x] All claims cited per-claim with file:line or URL

## Research Gate JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 11,
  "urls_collected": 19,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "gate_passed": true
}
```
