# Experiment Results — Cycle 1 (Phase 1 Complete)

## Summary
- **Cycle dates:** 2026-03-25 to 2026-03-28
- **Total experiments:** 40+ across 4 optimizer runs
- **Starting Sharpe:** 0.905 (initial audit baseline)
- **Final Sharpe:** 1.1705 (+29% improvement)
- **DSR:** 0.9984 (statistically significant)

## Changes Made (Phase 1 improvements)

### Code changes:
1. Cross-sectional percentile ranking in `_build_training_data()` (backtest_engine.py)
2. Turbulence index integration in `_predict_and_trade()` (backtest_engine.py)
3. Kelly Criterion position sizing in `size_position()` (backtest_trader.py)
4. Volatility targeting + trailing stops enabled in optimizer params (quant_optimizer.py)
5. Dynamic risk-free rate via FRED in aggregate Sharpe computation (backtest_engine.py)
6. Regime-aware strategy selection in `_compute_label()` (backtest_engine.py)
7. Performance-based position scaling in `mark_to_market()` (backtest_trader.py)
8. Sector diversification penalty in `execute_trades()` (backtest_trader.py)
9. Market microstructure costs in `_compute_commission()` (backtest_trader.py)

### Best params (from optimizer_best.json):
- strategy: triple_barrier
- sl_pct: 12.92 (optimizer found asymmetric barriers helpful)
- min_samples_leaf: 20
- n_estimators: 200, max_depth: 4, learning_rate: 0.1
- holding_days: 90, tp_pct: 10.0

## Key Experiment Patterns

### What worked:
- Asymmetric barriers (wider SL, tighter TP) — Sharpe jumped from 1.039 to 1.17
- Feature caching for ML-only changes — 80× speedup
- Early stopping at window 10 for obviously bad experiments

### What didn't work:
- target_vol perturbations consistently decreased Sharpe (5 attempts, all discarded)
- qm_weight additions showed no improvement (strategy is already triple_barrier)
- n_estimators changes beyond 200-220 range showed no benefit
- Holding days changes (90→97) degraded performance

## MDA Feature Importance (top 5, stable across windows):
1. annualized_volatility
2. treasury_10y
3. amihud_illiquidity
4. cpi_yoy
5. consumer_sentiment

## Observations
- Macro features (treasury, CPI, sentiment) dominate — strategy is macro-driven
- Volatility is the single most important feature — makes sense given inverse-vol sizing
- Momentum features ranked lower than expected — possible opportunity
- The optimizer hit diminishing returns around experiment 10 of the last run
- Need structural changes (new features, strategy logic) not more param tweaking
