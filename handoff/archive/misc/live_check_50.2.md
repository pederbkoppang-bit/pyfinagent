# live_check_50.2 -- multi-currency accounting (evidence)

Verified 2026-05-30. The load-bearing guarantee: every money term is x1.0 for USD/US.

## 1. Unit test (criteria #2, #3) -- backend/tests/test_phase_50_2_multicurrency.py
`pytest -q` -> **7 passed in 0.81s**:
- `_fx_local_to_usd("US")==1.0`, `(None)==1.0` (legacy rows), `_fx_usd_to_local("US")==1.0`, `get_fx_rate("USD","USD")==1.0`
- market_value formula `qty*price*fx == qty*price` for USD (byte-identical); share-count `amount_usd*u2l/price == amount_usd/price` for USD
- EUR (mocked fx=1.10): mv_usd = 5*100 EUR * 1.10 = $550; buy $1100/1.10/100 EUR = 10 shares
- attribution USD: fx_pnl==0, local_pnl==qty*(Pc-Pe); EUR: local_pnl+fx_pnl == MV_usd-cost_usd exactly (no residual)

## 2. LIVE byte-identity proof (criterion #2) -- the working +20% engine
Read-only recompute of every live position's market_value the NEW way (`qty*current_price*_fx_local_to_usd(market)`) vs the pre-50.2 way (`qty*current_price`):
```
STX  market=US fx=1.0 mv_new=468.34   mv_old=468.34   identical=True
MU   market=US fx=1.0 mv_new=989.71   mv_old=989.71   identical=True
ON   market=US fx=1.0 mv_new=575.74   mv_old=575.74   identical=True
INTC market=US fx=1.0 mv_new=1362.64  mv_old=1362.64  identical=True
DELL market=US fx=1.0 mv_new=1834.24  mv_old=1834.24  identical=True
SNDK market=US fx=1.0 mv_new=1582.76  mv_old=1582.76  identical=True
WDC  market=US fx=1.0 mv_new=1226.40  mv_old=1226.40  identical=True
NAV new=24023.58  NAV old=24023.58  BYTE-IDENTICAL=True
```
NAV new == NAV old == the stored total_nav $24,023.58 -> the live all-USD +20% portfolio is provably unchanged to the cent.

## 3. Non-USD correctness (criteria #3, #4) -- numeric (from the test, mocked EUR fx=1.10)
- A EUR position 5 sh @ EUR100 values into USD NAV at $550 (= 5*100*1.10).
- A EUR buy of $1100 budget at EUR100/sh = 10 shares (= 1100/1.10/100).
- Attribution example (EUR price 100->110, FX 1.10->1.20): local_pnl = 10*(110-100)*1.10 = $110; fx_pnl = 10*110*(1.20-1.10) = $110; sum $220 == MV_usd(1320) - cost_usd(1100) exactly.

## 4. Deterministic masterplan command
```
ast.parse(paper_trader.py) -> OK
get_fx_rate('USD','USD')==1.0 + import backend.services.paper_trader -> "import OK, USD/USD=1.0"
pytest backend/tests/test_phase_50_2_multicurrency.py -> 7 passed
test -f handoff/current/live_check_50.2.md -> present
```

## Success criteria mapping (all 4 met)
1. NAV/cost_basis/market_value/realized+unrealized P&L FX-convert each position to USD via fx_rates -- YES (mark_to_market market_value *= fx; buy share-count + existing-branch market_value + partial-sell pos_row + sell proceeds + realized_pnl_usd all *= fx; cost_basis is USD).
2. USD-only byte-identical -- YES (live NAV new==old==$24,023.58; unit test confirms x1.0 at every site).
3. non-USD values into USD NAV + local-vs-FX P&L decomposition -- YES (EUR $550 example; fx_pnl_attribution sums to MV-cost with no residual).
4. live/fixture numeric evidence of a EUR position's USD NAV + the local/FX split -- YES (above).

## Scope / honesty notes
- **Minimal-change model** (vs the brief's local-cost-basis + entry_fx column): cost_basis stays USD (the current convention), current_price stays LOCAL, market_value computed to USD via fx_rates; remaining-cost on partial sell is proportional USD (byte-identical for USD). Avoids a schema migration/new column; achieves the same correctness + attribution (entry FX derivable as cost_usd/(qty*avg_entry_price) when needed).
- `market` + `base_currency="USD"` now written on all buy/partial-sell pos_rows.
- FX-unavailable handling: BUY of a non-USD stock -> skip (never silently treat as USD); mark_to_market -> keep last-known USD market_value + WARN; SELL -> last-resort 1.0 + WARN (never block an exit). USD path: fx always 1.0, these never trigger.
- attribution exposed as a pure tested helper `paper_trader.fx_pnl_attribution` (not wired into the live _compute_attribution endpoint, where it would be all-zero today since every live position is USD -> fx_pnl=0; wiring it in is a 50.6-UI follow-on).
- Trade-record display fields (paper_trades.total_value / transaction_cost) remain in local currency for a non-USD trade -- a display-only nicety, NOT a NAV/cash error (NAV uses positions in USD + cash credited in USD). Flagged as a minor follow-up.
- $0 LLM; no pip; no DROP/DELETE; no owner approval.
