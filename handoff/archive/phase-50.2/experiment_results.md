# Experiment results -- phase-50.2: Multi-currency portfolio accounting

**Date:** 2026-05-30 | **Result: built + live-verified (byte-identical on the working +20% engine)** | $0 LLM | no pip | MONEY-CRITICAL step.

## What was built
`paper_trader.py` now FX-converts each position's market value / P&L / cash flows to the USD base currency via the 50.1 `fx_rates` service, with a local-vs-FX P&L attribution helper. The USD-only path is provably byte-identical (every conversion x1.0).

## Files changed/added
1. **`backend/services/paper_trader.py`**:
   - Module helpers `_fx_local_to_usd(market,date)` / `_fx_usd_to_local(market,date)` (1.0 for US/USD; None when a non-USD rate is genuinely unavailable) + `fx_pnl_attribution(qty,Pe,Pc,Fe,Fc)` (local_pnl + fx_pnl == MV_usd - cost_usd, no residual).
   - `execute_buy(..., market="US")`: share count = `amount_usd * fx(USD,local) / price` (skip buy if FX unavailable); existing-branch `market_value`/`unrealized_pnl` *= fx(local,USD); write `market` + `base_currency="USD"` on both pos_rows (new-branch market_value=amount_usd is already USD).
   - `execute_sell`: net_proceeds credited as `* fx(local,USD)`; partial-sell pos_row uses proportional USD cost_basis + `market_value *= fx`; `realized_pnl_usd *= fx`; write market/base_currency.
   - `mark_to_market`: `market_value = qty * live_price * fx(local,USD)` (the per-cycle NAV); fail-soft to last-known if FX unavailable; current_price stays LOCAL.
2. **`backend/tests/test_phase_50_2_multicurrency.py`** (NEW): 7 offline (mocked-FX) tests -- byte-identity primitives, USD market_value/share-count identity, EUR conversion, attribution USD(fx=0)/EUR(sums to MV-cost).

## Verification (live)
- `pytest backend/tests/test_phase_50_2_multicurrency.py` -> **7 passed**.
- Deterministic command: ast.parse + `get_fx_rate('USD','USD')==1.0` + `import paper_trader` -> OK.
- **LIVE byte-identity proof** (read-only, no mutation): all 7 live positions market=US -> fx=1.0 -> every market_value identical (mv_new==mv_old), **NAV new == NAV old == $24,023.58 == stored total_nav, BYTE-IDENTICAL=True**.
- EUR numeric (mocked): 5 sh @ EUR100 -> $550 USD NAV; attribution EUR100->110 / FX1.10->1.20 -> local_pnl $110 + fx_pnl $110 == MV_usd-cost_usd $220.

## Success criteria mapping (all 4 met) -- see live_check_50.2.md
1. NAV/cost/market_value/P&L FX-convert each position to USD -- YES. 2. USD-only byte-identical -- YES (live NAV unchanged + 7 unit tests). 3. non-USD values into USD + local/FX decomposition -- YES. 4. live/fixture numeric evidence -- YES.

## Scope / honesty notes
- **Minimal-change model** (deviates from the brief's local-cost-basis + entry_fx-column to AVOID a migration): cost_basis stays USD (current convention), current_price stays LOCAL, market_value computed to USD; partial-sell remaining-cost is proportional USD. Byte-identical for USD; entry FX derivable as cost_usd/(qty*avg_entry_price) if ever needed. Same correctness + attribution as the brief, less surface, no new column.
- FX-unavailable: BUY non-USD -> skip (never treat as USD); mark_to_market -> keep last-known + WARN; SELL -> last-resort 1.0 + WARN (never block an exit). USD path: fx==1.0 always, never triggers.
- Attribution is a pure tested helper (`fx_pnl_attribution`), NOT wired into the live `_compute_attribution` endpoint (it would be all-zero today -- every live position is USD -> fx_pnl=0); wiring it into the UI is a 50.6 follow-on.
- Trade-record display fields (paper_trades.total_value/transaction_cost) stay LOCAL for a non-USD trade -- display-only, NOT a NAV/cash error. Minor follow-up.
- The historical_fx_rates streaming-buffer junk rows from 50.1 remain harmless.
- $0 LLM; no pip; no DROP/DELETE; no owner approval.
