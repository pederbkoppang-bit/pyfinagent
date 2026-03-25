# Phase 0: Audit & Validate — Progress Tracker

## Status: IN PROGRESS
Started: 2026-03-25

---

## 0.1 Formula Validation

### Sharpe Ratio
- [ ] Cross-check with Sharpe (1994) "The Sharpe Ratio" — Journal of Portfolio Management
- [ ] Cross-check with Lo (2002) "The Statistics of Sharpe Ratios" — Financial Analysts Journal
- [ ] Verify risk-free rate handling (4% annualized / periods_per_year)
- [ ] Verify √252 annualization is correct for daily returns
- [ ] Source: analytics.py `compute_sharpe()`
- **Finding**: (pending)

### Deflated Sharpe Ratio
- [ ] Cross-check with Bailey & López de Prado (2014) "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality" — Journal of Portfolio Management
- [ ] Verify E[max(SR)] formula (Euler-Mascheroni approximation)
- [ ] Verify standard error formula (non-normality adjustment)
- [ ] Verify T < 10 guard is appropriate
- [ ] Source: analytics.py `compute_deflated_sharpe()`
- **Finding**: (pending)

### Sample Weights (Average Uniqueness)
- [ ] Cross-check with López de Prado AFML Ch. 4 "Sample Weights"
- [ ] Verify overlap detection logic
- [ ] Verify normalization (weights sum to n)
- [ ] Check edge cases: non-overlapping labels should get weight ≈ 1
- [ ] Source: backtest_engine.py `_compute_sample_weights()`
- **Finding**: (pending)

### Fractional Differentiation
- [ ] Cross-check with López de Prado AFML Ch. 5 "Fractionally Differentiated Features"
- [ ] Verify fixed-width window implementation
- [ ] Verify weight computation formula
- [ ] Consider: should we add ADF test to verify stationarity?
- [ ] Source: historical_data.py `fractional_diff()`
- **Finding**: (pending)

### Triple Barrier Labels
- [ ] Cross-check with López de Prado AFML Ch. 3 "Labeling"
- [ ] Verify barrier hit logic (TP/SL/time)
- [ ] Check: are we using calendar days or trading days for holding_days?
- [ ] Document: labels don't account for transaction costs (known limitation)
- [ ] Source: backtest_engine.py `_compute_triple_barrier_label()`
- **Finding**: (pending)

### Monte Carlo VaR
- [ ] Verify GBM assumption appropriateness for equity returns
- [ ] Document fat-tail limitation
- [ ] Consider: Student-t or historical simulation as alternative/supplement
- [ ] Source: historical_data.py `_compute_monte_carlo_var()`
- **Finding**: (pending)

### Position Sizing (Inverse Volatility)
- [ ] Cross-check with AQR "Betting Against Beta" (Frazzini & Pedersen, 2014)
- [ ] Verify formula: probability × (target_vol / stock_vol) × nav / max_positions
- [ ] Check: is the 3x cap on vol_scale justified?
- [ ] Check: max_single_pct × nav cap
- [ ] Source: backtest_trader.py `size_position()`
- **Finding**: (pending)

### Scalar Metric
- [ ] Document theoretical justification for tx cost penalty
- [ ] Verify: risk_adjusted_return = avg_return × beat_benchmark_rate is meaningful
- [ ] Check: 0.3 cap on tx drag — is this appropriate?
- [ ] Source: perf_metrics.py `get_scalar_metric()`
- **Finding**: (pending)

---

## 0.2 Bug Fixes

- [x] Quality Score — expanded to full Asness (2019) QMJ 4-dimension composite: profitability (ROE + margin), growth (revenue YoY), safety (D/E + vol), payout (FCF yield + dividend yield). Replaces old ROE × margin × (1-D/E) formula.
- [x] Factor Model — replaced hardcoded min/max normalization with sigmoid-centered scoring. Switched value factor from P/E to P/B (Fama-French HML proxy). Added 12-1 momentum (Jegadeesh & Titman 1993). Uses updated QMJ quality_score. Weights: value 25%, momentum 25%, quality 25%, low-vol 15%, yield 10%.
- [x] Mean Reversion — rewired to use mr_holding_days as forward validation horizon. Now two-stage: (1) detect oversold/overbought via SMA+RSI, (2) validate actual reversion within mr_holding_days trading days. Academic basis: Lo & MacKinlay (1990), reversion at 1-4 week horizon.
- [x] Meta-Label — implemented 2-stage model per AFML Ch. 3.6. Primary model provides direction, secondary (meta) model predicts P(primary is correct). Uses 3-fold CV on training set to avoid leakage. Meta-model's probability replaces primary probability for position sizing. Predictions with meta_probability < 0.5 filtered to HOLD.
- [x] Transaction costs in TB labels — TP barrier shifted up and SL barrier shifted up by round-trip cost (2 × transaction_cost_pct). Prevents labeling marginal-profit trades as winners when costs would eat the gain. Per Almgren & Chriss (2000).
- [ ] Sample weights O(n²) → O(n log n) optimization

---

## 0.3 Overfitting Diagnostics

- [ ] Walk-forward leakage audit
- [ ] Label leakage check
- [ ] Feature importance stability test
- [ ] Multi-seed DSR validation
- [ ] Backtest vs reality gap documentation

---

## Academic Sources Registry

| Paper | Authors | Year | Journal | Used For | Verified? |
|-------|---------|------|---------|----------|-----------|
| The Sharpe Ratio | Sharpe, W.F. | 1994 | J. Portfolio Mgmt | Sharpe formula | [ ] |
| The Statistics of Sharpe Ratios | Lo, A.W. | 2002 | Financial Analysts J. | √T annualization | [ ] |
| The Deflated Sharpe Ratio | Bailey, D.H. & López de Prado, M. | 2014 | J. Portfolio Mgmt | DSR overfitting guard | [ ] |
| Advances in Financial Machine Learning | López de Prado, M. | 2018 | Book (Wiley) | TB labels, sample weights, frac diff, walk-forward | [ ] |
| Quality Minus Junk | Asness, C., Frazzini, A. & Pedersen, L. | 2019 | J. Finance | Quality score definition | [ ] |
| A Five-Factor Model | Fama, E. & French, K. | 2015 | J. Financial Economics | Factor model construction | [ ] |
| Returns to Buying Winners/Selling Losers | Jegadeesh, N. & Titman, S. | 1993 | J. Finance | Momentum (12-1) | [ ] |
| ...and the Cross-Section of Expected Returns | Harvey, C., Liu, Y. & Zhu, H. | 2016 | Review of Financial Studies | Multiple testing (t>3.0 threshold) | [ ] |
| When are Contrarian Profits Due to Overreaction? | Lo, A.W. & MacKinlay, A.C. | 1990 | Review of Financial Studies | Mean reversion horizon | [ ] |
| Betting Against Beta | Frazzini, A. & Pedersen, L. | 2014 | J. Financial Economics | Inverse-vol sizing | [ ] |
| Optimal Execution of Portfolio Transactions | Almgren, R. & Chriss, N. | 2000 | J. Risk | Market impact / tx cost | [ ] |
| Machine Learning for Asset Managers | López de Prado, M. | 2020 | Book (Cambridge) | Additional ML techniques | [ ] |
| Why do tree-based models still outperform... | Grinsztajn, L. et al. | 2022 | NeurIPS | GBC vs deep learning for tabular | [ ] |
| TradingAgents | arXiv:2412.20138 | 2024 | arXiv | Multi-agent debate framework | [ ] |
| Ensemble Methods in Machine Learning | Dietterich, T. | 2000 | MCS Workshop | Strategy blending | [ ] |
