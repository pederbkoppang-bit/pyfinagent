---
step: phase-5.4
title: Multi-Asset Risk Engine Extension (equity + options + FX + futures)
cycle_date: 2026-04-26
harness_required: true
verification: 'source .venv/bin/activate && python -c "from backend.markets.risk_engine import RiskEngine; r=RiskEngine(); eq=r.compute_position_size(''AAPL'',''equity'',100000,0.2); opt=r.compute_position_size(''AAPL240119C00150000'',''option'',100000,0.3,delta=0.5); fx=r.compute_position_size(''EUR_USD'',''fx'',100000,0.08); assert all(x>0 for x in [eq,opt,fx]); print(''ok'')"'
research_brief: handoff/current/phase-5.4-research-brief.md
---

# Contract -- phase-5.4

## Step ID

`phase-5.4` -- "Multi-Asset Risk Engine Extension (equity + options + FX + futures)" (`.claude/masterplan.json` phase-5).

## Research-gate summary

Spawned `researcher` (simple tier). Brief at
`handoff/current/phase-5.4-research-brief.md`. Gate: 5 external
sources read in full (QuantPedia vol-targeting, AccountingInsights
delta-adjusted, OANDA micro-lot, ConcretumGroup, RobotWealth), 15
URLs, recency scan, 8 internal files. `gate_passed: true`.

Decisive findings:
- Equity vol-targeting: `notional = equity * target_vol / asset_vol`, clamped at `3.0 * equity` (matches existing `BacktestTrader.size_position` at `backend/backtest/backtest_trader.py:80-92`)
- Option delta-adjustment: `option_notional = base_notional * abs(delta)` (delta=0.5 -> half; signed delta only for P&L)
- FX micro-lot floor: 1 micro lot = 1000 units (OANDA universal); `lots = max(1, round(raw / 1000))`; `final = lots * 1000`
- Default `target_vol = 0.15` (matches `BacktestTrader` + `BacktestEngine` defaults)
- Asset class enum: `Literal["equity","option","fx","future"]` + raise ValueError on `"crypto"` (owner directive 2026-04-19)
- Regression anchor: `RiskEngine.compute_position_size('AAPL','equity',100000,0.20)` matches `BacktestTrader(max_positions=1).size_position(1.0, 0.20, 100000)` within 1e-6
- Greenfield file: no existing `backend/markets/risk_engine.py`; do NOT modify `BacktestTrader` or `portfolio_manager` (those are separate sizing paths)

## Hypothesis

A new `backend/markets/risk_engine.py` with a stateless `RiskEngine`
class implementing `compute_position_size(symbol, asset_class, equity,
asset_vol, **kwargs)` provides the multi-asset sizing contract that
phase-5.6+ option/FX/future execution paths plug into. Existing
equity sizing in `BacktestTrader` and Risk Judge in `portfolio_manager`
remain untouched -- this is additive, like 5.1.

## Immutable success criteria (verbatim from masterplan)

```
source .venv/bin/activate && python -c "from backend.markets.risk_engine import RiskEngine; r=RiskEngine(); eq=r.compute_position_size('AAPL','equity',100000,0.2); opt=r.compute_position_size('AAPL240119C00150000','option',100000,0.3,delta=0.5); fx=r.compute_position_size('EUR_USD','fx',100000,0.08); assert all(x>0 for x in [eq,opt,fx]); print('ok')"
```

Plus 5 success_criteria from masterplan (positive notional, delta-adjusted, FX micro-lot, regression on equity, no crypto).

## Plan steps

1. Create `backend/markets/risk_engine.py` (~150 LOC):
   - Module-level constants: `DEFAULT_TARGET_VOL = 0.15`, `MAX_LEVERAGE = 3.0`, `FX_MICRO_LOT = 1000`
   - `class RiskEngine` with `__init__(self, *, target_vol=DEFAULT_TARGET_VOL, max_leverage=MAX_LEVERAGE)`
   - `compute_position_size(symbol, asset_class, equity, asset_vol, *, delta=None, **kwargs) -> float`:
     - Normalize asset_class with `.lower()`; raise `ValueError` for `"crypto"`
     - Compute `base_notional = equity * (target_vol / max(asset_vol, 1e-6))` clamped at `max_leverage * equity`
     - equity: return base_notional
     - option: require delta (else assume 1.0); return `base_notional * abs(delta)`
     - fx: micro-lot floor: `lots = max(1, round(base_notional / FX_MICRO_LOT))`; return `lots * FX_MICRO_LOT`
     - future: same as equity for now (placeholder; no contract-multiplier table yet)
     - Default branch: raise `ValueError` listing supported classes
   - ASCII-only docstrings/logs

2. Create `tests/markets/test_risk_engine.py` (~150 LOC, 8+ tests per research plan):
   1. `test_equity_basic` -- 100000, vol=0.20, target=0.15 -> 75000.0
   2. `test_equity_clamp` -- vol=0.01 -> <= 3.0 * equity
   3. `test_option_delta_half` -- delta=0.5 -> 0.5 * base_notional
   4. `test_option_delta_negative` -- delta=-0.5 -> same as +0.5
   5. `test_fx_micro_lot_min` -- small base notional -> >= 1000
   6. `test_fx_rounding` -- base ~= 4500 -> rounds to 5000
   7. `test_no_crypto` -- raises ValueError
   8. `test_unknown_asset_class_raises` (defensive)
   9. `test_regression_vs_backtest_trader_equity` (when feasible without coupling)
   10. `test_immutable_verification_inline` -- runs the masterplan's exact python -c assertions

3. Run immutable verification command.

## References

- `handoff/current/phase-5.4-research-brief.md`
- `backend/backtest/backtest_trader.py:54, 80-92` (existing target_vol=0.15 + size_position formula)
- `backend/backtest/backtest_engine.py:163` (target_vol=0.15 default)
- `backend/markets/{__init__,broker_base,alpaca_broker}.py` (phase-5.1 pattern to follow)
- QuantPedia: https://quantpedia.com/an-introduction-to-volatility-targeting/
- OANDA micro-lot: https://help.oanda.com/us/en/faqs/micro-lots.htm

## Out of scope

- Changes to `BacktestTrader` / `portfolio_manager` / Risk Judge / `kelly_allocator` (separate sizing paths)
- Wiring `RiskEngine` into `paper_trader.py` (becomes load-bearing in 5.7+ when FX/option execution comes online)
- Future-asset contract-multiplier table (placeholder return `base_notional`; properly modeled in 5.8 IBKR cycle)
- BQ schema changes
- Frontend changes
