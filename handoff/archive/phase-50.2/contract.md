# Contract -- phase-50.2: Multi-currency portfolio accounting

**Step id:** 50.2 | **Priority:** P3 (phase-50; MONEY-CRITICAL) | **depends_on:** 50.1
**Date:** 2026-05-30 | **harness_required:** true | **$0 LLM** | no pip

## Research-gate summary (PASSED)
`handoff/current/research_brief.md` (gate: **8 sources read in full, recency scan, 19 URLs, 11 internal files, gate_passed=true**). Load-bearing decisions:
- **Byte-identity STRUCTURALLY guaranteed:** `fx_rates.get_fx_rate("USD","USD")==1.0`; every current live position is market NULL/"US" -> currency "USD" -> base "USD" -> every money term x1.0 -> the 100%-USD +20% portfolio's NAV/P&L are unchanged to the cent.
- **Position currency is DERIVED, not stored:** live `paper_positions` has `market` + `base_currency` (both NULL today), NO `currency` column. currency = `fx_rates.market_currency(pos.get("market") or "US")`. 50.2 starts WRITING `market` + `base_currency` (+ `entry_fx_rate`) on the buy pos_row branches (paper_trader.py:248-267, :270-286) + the partial-sell re-insert (:400-415); `execute_buy` gets a `market="US"` default arg.
- **Cash = single base-USD scalar** (no wallets). Non-USD BUY converts the USD budget->local at trade-time FX to size shares; non-USD SELL converts local proceeds->USD before crediting. USD trades: FX=1.0 -> cash math unchanged.
- **Cost basis stored LOCAL, fixed at trade date, NEVER re-translated** (IAS 21). `market_value` + `unrealized_pnl` persisted in **USD** (downstream shape-identical); `current_price` stays LOCAL; `unrealized_pnl_pct` is a local-return %.
- **Output shape UNCHANGED:** get_portfolio/get_status/save_daily_snapshot/paper_metrics_v2 read USD total_nav/market_value straight -> NO endpoint change; ADD attribution fields, never rename.
- **Attribution (exact, no residual):** with qty, entry/current local price Pe/Pc, entry/current FX Fe/Fc (USD per local): `local_pnl = qty*(Pc-Pe)*Fe`; `fx_pnl = qty*Pc*(Fc-Fe)`; sum = `qty*(Pc*Fc - Pe*Fe)` = `MV_usd - C_usd` exactly. Store `entry_fx_rate` at BUY (avoids per-cycle lookup + look-ahead). Expose in `_compute_attribution` (paper_trading.py:354). USD: Fe=Fc=1.0 -> fx_pnl=0.
- **Pitfalls:** double-conversion (convert in exactly ONE place per term), cost-basis re-translation (keep local fixed), inconsistent FX source (single source = fx_rates), and **`None` from get_fx_rate must NOT be coerced to 1.0** (fail-soft to last-known / log WARN, mirroring _get_live_price:441-442).

## Money-site conversions (file:line)
- BUY share count (:168): `quantity = amount_usd * get_fx_rate("USD", local) / price_local`
- BUY cost_basis (:243-246/:275): store local = `amount_usd * get_fx_rate("USD", local)`; also store `entry_fx_rate = get_fx_rate(local,"USD")`, `market`, `base_currency="USD"`
- MTM market_value->USD (:444): `mv_usd = qty * current_price_local * get_fx_rate(local,"USD")`
- NAV (:480): `total_positions_value` accumulates **USD** mvs; `current_cash` already USD
- SELL proceeds (:329-331/:420): `net_proceeds_usd = net_proceeds_local * get_fx_rate(local,"USD")` before crediting cash

## Hypothesis
Injecting `fx_rates` conversion at the 5 paper_trader money sites (each in exactly ONE place), deriving position currency from `market`, storing `entry_fx_rate`/`market`/`base_currency` at BUY, and exposing local-vs-FX attribution -- makes the paper portfolio currency-correct for non-USD positions while leaving the all-USD path byte-identical (every conversion x1.0).

## Success criteria (IMMUTABLE -- verbatim from masterplan step 50.2)
1. paper_trader NAV / cost_basis / market_value / realized+unrealized P&L FX-convert each position from its local currency to the portfolio base_currency (USD) using fx_rates (50.1)
2. USD-only portfolio behaviour is byte-identical to pre-50.2 (regression test: a US-only NAV/P&L computation matches the current value exactly)
3. a non-USD position (e.g. a EUR holding) values into USD NAV at the correct FX rate; P&L is decomposed into local-return vs FX-return per the arXiv model
4. live or fixture evidence: a EUR position's USD NAV contribution + the local/FX P&L split shown numerically

**Verification command:** ast.parse(paper_trader.py) + get_fx_rate('USD','USD')==1.0 + import paper_trader + `pytest backend/tests/test_phase_50_2_multicurrency.py` + test -f live_check_50.2.md.
**live_check:** REQUIRED -- numeric evidence: USD-only path unchanged + a EUR position FX-converted into USD NAV with the local/FX P&L split.

## Plan steps
1. **paper_trader.py** -- add a small `_to_usd(value_local, market, date=None)` helper using `fx_rates.get_fx_rate(market_currency(market), "USD", date)` with fail-soft on None (WARN + last-known, NOT 1.0). Inject at the 5 money sites above, each ONCE. `execute_buy(..., market="US")`; write `market`/`base_currency`/`entry_fx_rate` on the 3 pos_row sites. mark_to_market market_value -> USD; NAV sums USD; SELL proceeds -> USD. Guard: USD/US -> x1.0 (byte-identical).
2. **paper_trading.py `_compute_attribution` (:354)** -- add `local_pnl` + `fx_pnl` per the formula (using stored entry_fx_rate + current FX); USD positions -> fx_pnl=0. ADDITIVE fields only.
3. **`backend/tests/test_phase_50_2_multicurrency.py`** (NEW) -- (a) USD-only byte-identical: a synthetic USD position's mark_to_market market_value/nav/pnl == the pre-50.2 arithmetic (qty*price, etc.) to the cent + get_fx_rate('USD','USD')==1.0; (b) EUR position: mv_usd == qty*price_eur*fx, and local_pnl+fx_pnl == mv_usd-cost_usd exactly; mock fx_rates.get_fx_rate so the test is deterministic + offline.
4. **Verify:** ast.parse; the pytest; a LIVE before/after on the current portfolio (stub _get_live_price to stored current_price -> assert total_nav == the stored/pre-50.2 value, proving byte-identical on the real all-USD portfolio); a synthetic EUR numeric example. Capture into live_check_50.2.md.
5. **EVALUATE:** fresh qa (no self-eval). Then harness_log.md (LAST), then flip masterplan 50.2 -> done.

## Safety / scope notes
- **The working +20% engine is provably untouched** (all-USD -> x1.0). The byte-identical regression test + the live before/after are the proof.
- Convert in exactly ONE place per money term (no double-conversion). Cost basis local + fixed (no re-translation). `None` FX -> fail-soft (WARN + last-known), never silently 1.0 (that would mis-value a non-USD position as if USD).
- Output JSON shape unchanged (USD NAV); attribution fields are additive.
- No new pip; no owner approval (no DROP/DELETE; FX is free). The historical_fx_rates streaming-buffer junk rows from 50.1 remain harmless.

## References
- handoff/current/research_brief.md (50.2 gate) + research_brief_multimarket.md
- backend/services/paper_trader.py:168,221,243-246,275,329-331,400-415,420,432,444,480 (money sites + pos_row writes)
- backend/services/fx_rates.py (get_fx_rate, market_currency) [50.1]
- backend/api/paper_trading.py:354 (_compute_attribution)
- backend/config/settings.py:50-51 (base_currency/default_market)
- backend/db/bigquery_client.py:512-1075 (paper-table I/O), paper_metrics_v2.py / perf_metrics.py (downstream, shape-unchanged)
- IAS 21 (cost basis fixed); CFA/CFI/Meradia (local-vs-FX attribution); arXiv 1611.01463; FundCount/Sharesight (base-ccy NAV)
