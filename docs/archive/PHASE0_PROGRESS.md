# Phase 0: Audit & Validate — Progress Tracker

## Status: IN PROGRESS
Started: 2026-03-25

---

## 0.1 Formula Validation — COMPLETE ✅

All 8 core formulas validated against original academic papers. Full findings in PHASE0_FINDINGS.md.

| Formula | Verdict | Source |
|---------|---------|--------|
| Sharpe Ratio | ✅ Correct | Sharpe (1994), Lo (2002) |
| Deflated Sharpe Ratio | ✅ Correct | Bailey & López de Prado (2014) |
| Sample Weights | ⚠️ Simplified (directionally correct) | AFML Ch. 4 |
| Fractional Differentiation | ✅ Correct | AFML Ch. 5 |
| Triple Barrier Labels | ✅ Correct (now with tx cost adjustment) | AFML Ch. 3 |
| Monte Carlo VaR | ✅ Correct (fat-tail limitation documented) | GBM standard |
| Position Sizing | ⚠️ Custom hybrid (documented) | AQR + Kelly inspired |
| Scalar Metric | ⚠️ Custom (functional, needs theoretical grounding) | Custom |

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

- [x] Walk-forward leakage audit — completed, no critical leakage found. Feature vectors correctly use cutoff_date bounds. BQ cache enforces temporal boundaries. See PHASE0_LEAKAGE_AUDIT.md.
- [x] Label leakage check — labels look forward by holding_days (by design). Embargo < holding_days means overlap with test window; this is standard practice per AFML Ch. 7. Documented as known limitation.
- [ ] Feature importance stability test — REQUIRES BACKTEST RUN (deferred to when BQ data is available)
- [ ] Multi-seed DSR validation — REQUIRES BACKTEST RUN (deferred)
- [x] Backtest vs reality gap documentation — completed. 10 gaps analyzed. Net assessment: backtest is moderately conservative. Survivorship bias is biggest unaddressed issue (Phase 1). See PHASE0_REALITY_GAP.md.

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
