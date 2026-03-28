# Research Plan — Cycle 2

## Context
Phase 1 optimization is complete. Sharpe 1.1705 achieved through parameter tuning and
structural code improvements. The optimizer hit diminishing returns on parameter-level
changes — last 8 experiments all discarded. Need structural research directions.

## Analysis of Experiment Log Patterns

### Signal: Macro features dominate MDA importance
- treasury_10y, cpi_yoy, consumer_sentiment are top features
- This means our strategy is primarily a macro-timing strategy
- **Opportunity:** Improve macro feature quality (more FRED series, leading indicators)

### Signal: target_vol perturbations always hurt
- 5 attempts to add volatility targeting, all decreased Sharpe
- This suggests our current inverse-vol sizing is already near-optimal
- **Action:** Stop trying target_vol changes, focus elsewhere

### Signal: Asymmetric barriers were the biggest single win
- sl_pct moved to 12.92 while tp_pct stayed at 10.0
- Wider stop-loss + tighter take-profit = let winners run, cut losers
- **Opportunity:** Explore volatility-adjusted barriers (per-stock, not fixed)

### Signal: Optimizer plateau after experiment 10
- All experiments 11-15 discarded with declining DSR
- Parameter space is exhausted at current architecture
- **Action:** Structural changes needed, not more param sweeps

## Proposed Research Directions (prioritized)

### Direction A: Enhanced Macro Features (highest ROI, zero cost)
**Hypothesis:** Adding leading economic indicators will improve the macro-timing signal.
**Changes:**
- Add ISM Manufacturing PMI from FRED (leading indicator)
- Add initial jobless claims (weekly, high-frequency)
- Add VIX as a feature (free from Yahoo Finance)
- Add yield curve slope (10Y-2Y spread, already partially implemented)
**Expected impact:** +0.1 to +0.2 Sharpe
**Risk:** More features → more overfitting risk. Evaluator must check ablation.

### Direction B: Momentum Feature Upgrade (medium ROI, zero cost)
**Hypothesis:** Momentum features rank low because they're raw values, not cross-sectional ranks.
**Changes:**
- Already have percentile versions — verify they're being used by the model
- Add momentum acceleration (change in momentum) as new feature
- Add sector-relative momentum (stock momentum vs sector ETF)
**Expected impact:** +0.05 to +0.15 Sharpe
**Risk:** Low risk, these are well-established academic features.

### Direction C: Walk-Forward Robustness (no Sharpe impact, required by evaluator)
**Hypothesis:** Current results may not hold across sub-periods.
**Changes:**
- Run backtest on 3 sub-periods independently
- Run with 5 different random seeds
- Run with 2× transaction costs
**Expected impact:** No Sharpe improvement, but validates existing results
**Risk:** May discover current Sharpe is overstated. Better to know now.

## Recommended Execution Order
1. **Direction C first** — validate before optimizing further (evaluator requirement)
2. **Direction A** — highest expected ROI with low risk
3. **Direction B** — if A succeeds and evaluator passes

## Success Criteria for Evaluator
- Direction C: All sub-period Sharpes > 0.3, seed std < 0.10
- Direction A: Sharpe ≥ 1.20 with DSR ≥ 0.95 and evaluator PASS on all criteria
- Direction B: Marginal Sharpe improvement ≥ +0.05 with ablation confirmation
