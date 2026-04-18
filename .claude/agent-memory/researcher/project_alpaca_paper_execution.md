---
name: Alpaca paper execution — phase-3.7.5 research
description: SDK choice, feature-flag pattern, shadow-mode drift threshold, paper gotchas, and rollback for Alpaca execution swap in pyfinagent
type: project
---

alpaca-py (v0.43.2, Nov 2025) is the only actively maintained Alpaca Python SDK. `alpaca-trade-api-python` is deprecated since end-of-2022. TradingClient(paper=True) is the paper-mode entry point; keys must be paper-account keys.

**Why:** alpaca-trade-api-python README explicitly directs migration. alpaca-py supports Python 3.8–3.14.

**Feature flag shape:** Ops-toggle pattern (Fowler) via a single env-var `EXECUTION_BACKEND` with values `bq_sim | alpaca_paper | alpaca_live`. Read once at startup into a module-level constant; inject into the order router. No external service needed for a single-operator shop.

**Shadow-mode drift threshold:** <=1% fill-price drift is at the tight end but feasible for S&P liquid names using BQ close-price sim. Literature (IBKR, LuxAlgo, QuantConnect) puts acceptable simulation slippage at 0.1–2% for liquid equities; 1% is defensible. Alpaca paper fills at NBBO price; BQ sim uses close price — overnight gap can exceed 1% on event days, so test on non-event days or widen threshold to 2%.

**Alpaca paper gotchas:**
- Fractional shares: supported in paper by default ($1 min notional, must set `fractionable=true` on asset; TimeInForce must be DAY for fractional market orders).
- Market hours: orders submitted outside market hours queue or reject depending on extended-hours flag; paper does NOT simulate pre/post fills automatically.
- Settlement: T+1 since May 2024 (same as live); margin accounts settle immediately.
- Symbol coverage: ~2,000+ fractionable equities; not all S&P 500 tickers are fractionable — must check asset.fractionable before submitting notional orders.
- Partial fills: ~10% random partial fill rate in paper mode by design.
- Market impact / queue position: NOT simulated.

**Rollback pattern:** Circuit-breaker wrapper (pybreaker or fabfuel/circuitbreaker) around the Alpaca call; on open-circuit, fall back to BQ sim path. State is stored in BQ (position ledger), so both paths write to the same table — no data corruption if the Alpaca path is mid-order; treat unconfirmed Alpaca orders as rejected and log for reconciliation.

**How to apply:** Implement `EXECUTION_BACKEND` env-var in `backend/.env`, read in `backend/services/autonomous_loop.py` or a new `backend/services/execution_router.py`. Shadow mode runs both paths simultaneously and logs `alpaca_fill_price` vs `bq_sim_price` to BQ for 5-day comparison before flag promotion.
