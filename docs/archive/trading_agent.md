# trading_agent.md — Autonomous Trading Optimization

> Modeled after [karpathy/autoresearch](https://github.com/karpathy/autoresearch) `program.md`.
> This file is the instruction set for the autonomous optimization system.
> It is edited by humans. The optimizer modifies strategy params and skill prompts, never this file.

---

## Mission

Maximize risk-adjusted returns on US equities (S&P 500 universe) through autonomous
experimentation on quant strategy parameters and LLM agent prompts. Every decision
must be evidence-based, backed by academic literature, and statistically validated.

**Timeline**: Testing now → paper trading April → real money (Slack signals) May 2026.
**Budget**: BigQuery (low), GitHub Models (free via Copilot Pro), Claude Max (via OpenClaw). Vertex AI avoided unless proven high-ROI.

---

## What You CAN Modify

These are the "train.py" equivalents — fair game for autonomous experimentation:

### QuantOpt (fast loop, minutes/cycle, zero LLM cost)
- **21 strategy parameters** in `QuantStrategyOptimizer` (all bounds-constrained):
  - Triple Barrier: `tp_pct`, `sl_pct`, `holding_days`, `vol_barrier_multiplier`
  - Mean Reversion: `mr_holding_days`
  - Feature: `frac_diff_d`
  - ML: `n_estimators`, `max_depth`, `min_samples_leaf`, `learning_rate`
  - Portfolio: `target_vol`, `max_positions`, `top_n_candidates`
  - Screening: `momentum_weight`, `rsi_weight`, `volatility_weight`, `sma_weight`
  - Strategy: `strategy` (categorical: triple_barrier, quality_momentum, mean_reversion, factor_model, meta_label, blend)
  - Blend weights: `tb_weight`, `qm_weight`, `mr_weight`, `fm_weight`

### SkillOpt (slow loop, days/cycle, requires outcome data)
- `## Prompt Template`, `## Skills & Techniques`, `## Anti-Patterns` sections of each `backend/agents/skills/*.md`
- 29 agent skills total; optimize highest-MDA-impact agents first

### PerfOpt (fast loop, minutes/cycle)
- API cache TTL values in `ENDPOINT_TTLS`

## What You CANNOT Modify

Fixed harness — the "prepare.py" equivalent:

- Orchestrator pipeline order and step definitions (`orchestrator.py`)
- Output JSON schemas (`schemas.py`)
- BigQuery table schemas (88-column `analysis_results`, outcome tracking, paper trading)
- Evaluation formulas (`analytics.py`: Sharpe, DSR, baselines)
- Function signatures in `prompts.py`
- Data tool implementations (`backend/tools/*`)
- Walk-forward window generation logic (`walk_forward.py`)
- This file (`trading_agent.md`)

---

## The Metric

**Primary**: Annualized Sharpe ratio (Sharpe 1994, Lo 2002)
- Computed from daily NAV returns across walk-forward windows
- √252 annualization, 4% risk-free rate
- Source: `analytics.compute_sharpe()`

**Gate**: Deflated Sharpe Ratio ≥ 0.95 (Bailey & López de Prado 2014)
- Rejects overfitted improvements even when Sharpe looks better
- Accounts for number of trials, non-normality, sample length
- Source: `analytics.compute_deflated_sharpe()`

**Secondary** (for SkillOpt): `get_scalar_metric()` = risk-adjusted return × (1 − tx cost drag)

**Simplicity criterion** (from Karpathy): A small improvement that adds complexity is not worth it. Removing something and getting equal or better results is a great outcome.

---

## The Loop

```
LOOP FOREVER:
  1. BASELINE — run walk-forward backtest with current params, record Sharpe
  2. PROPOSE — one modification (random perturbation or LLM-guided)
  3. APPLY — set new param on engine
  4. MEASURE — run walk-forward backtest, compute Sharpe + DSR
  5. DECIDE:
     - IF Sharpe improved AND DSR ≥ 0.95 → KEEP (advance)
     - IF Sharpe improved BUT DSR < 0.95 → DSR_REJECT (revert)
     - IF Sharpe worse → DISCARD (revert)
     - IF crashed → CRASH (revert, log error)
  6. LOG — every experiment to quant_results.tsv (kept or discarded)
  7. NEVER STOP — run until externally stopped
```

**Warm start**: If `optimizer_best.json` exists, skip baseline and start from previous best.
**Warm cache**: BQ data preloaded once (`skip_cache_clear=True`), reused across iterations.
**Feature drift**: Top-5 MDA features logged per keep; WARNING on drift.

---

## Three Loops, One Coordinator

| Loop | Modifies | Speed | Guard | Metric |
|------|----------|-------|-------|--------|
| **QuantOpt** | 21 strategy params | Minutes | DSR ≥ 0.95 | Sharpe on backtest |
| **SkillOpt** | Agent skill prompts | Days | Simplicity criterion | Scalar metric from outcome_tracking |
| **PerfOpt** | Cache TTL values | Minutes | ≥5% latency gain | p95 latency |

**MetaCoordinator** sequencing (`backend/agents/meta_coordinator.py`):
1. Low Sharpe → QuantOpt
2. Low accuracy → SkillOpt (target agents by MDA importance)
3. High latency → PerfOpt
4. After QuantOpt keep → extract MDA → map to responsible agents → queue SkillOpt

**MDA→Agent Bridge**: QuantOpt discovers which features matter → MetaCoordinator targets the corresponding LLM agent's skill for optimization. Example: high MDA on `nlp_sentiment_score` → optimize NLP Sentiment Agent skill.

---

## 6 Strategies

| Strategy | Label Method | Research Basis |
|----------|-------------|----------------|
| **triple_barrier** | TP/SL/Time barriers (vol-adjusted) | López de Prado AFML Ch. 3; vol barriers per Ch. 3 recommendation |
| **quality_momentum** | 6M momentum + QMJ quality filter | Asness et al. (2019) "Quality Minus Junk", Novy-Marx (2013) |
| **mean_reversion** | SMA+RSI signal → forward reversion validation | Lo & MacKinlay (1990), Poterba & Summers (1988) |
| **factor_model** | 5-factor composite (P/B, 12-1 mom, vol, QMJ, yield) | Fama & French (2015), Carhart (1997), Jegadeesh & Titman (1993) |
| **meta_label** | TB labels + secondary model for bet sizing | López de Prado AFML Ch. 3.6 |
| **blend** | Weighted vote across TB, QM, MR, FM | Dietterich (2000) "Ensemble Methods in ML" |

---

## ~49 Feature Vector

| Category | Features | Source |
|----------|----------|--------|
| Price | price_at_analysis | BQ historical_prices |
| Momentum | momentum_1m/3m/6m/12m, momentum_12_1 | Jegadeesh & Titman (1993) |
| Technical | rsi_14, sma_50/200_distance, bb_upper/lower_distance, bb_pct_b, volume_ratio_20d | Bollinger (1992) |
| Volatility | annualized_volatility, daily_volatility | For inverse-vol sizing & vol-adjusted barriers |
| Risk | var_95/99_6m, expected_shortfall_6m, prob_positive_6m | GBM Monte Carlo (1K sims) |
| Anomaly | anomaly_count | Goldman Sachs 127-dim (Z-score proxy) |
| Liquidity | amihud_illiquidity | López de Prado AFML Ch. 18 |
| Fundamental | pe_ratio, pb_ratio, debt_equity, roe, profit_margin, market_cap, fcf_yield, dividend_yield, quality_score, revenue_growth_yoy | Fama-French (2015), Asness (2019) |
| Balance Sheet | total_revenue, net_income, total_debt, total_equity, total_assets | BQ historical_fundamentals |
| Macro | fed_funds_rate, cpi_yoy, unemployment_rate, yield_curve_spread, consumer_sentiment, treasury_10y | FRED 7-series |
| Derived | fractionally differenced: price, market_cap, revenue, debt, equity | López de Prado AFML Ch. 5 (d=0.4) |

---

## Position Sizing

**Inverse-volatility** (AQR / Frazzini & Pedersen 2014):
`size = probability × min(target_vol / stock_vol, 3.0) × nav / max_positions`

**Three scaling filters** (applied multiplicatively):
1. **Turbulence dampening** (FinRL): scale down when Mahalanobis distance exceeds threshold
2. **Amihud liquidity** (AFML Ch. 18): scale down for illiquid stocks (Amihud > 0.5)
3. **Meta-label confidence** (AFML Ch. 3.6): when strategy=meta_label, secondary model probability replaces primary

Capped at `max_single_pct × nav` (default 10%).

---

## Academic Sources

| Paper | Authors | Year | Used For |
|-------|---------|------|----------|
| The Sharpe Ratio | Sharpe, W.F. | 1994 | Sharpe formula (✅ validated) |
| The Statistics of Sharpe Ratios | Lo, A.W. | 2002 | √T annualization (✅ validated) |
| The Deflated Sharpe Ratio | Bailey, D.H. & López de Prado, M. | 2014 | DSR overfitting guard (✅ validated) |
| Advances in Financial Machine Learning | López de Prado, M. | 2018 | TB labels, sample weights, frac diff, walk-forward, MDA (✅ validated) |
| Quality Minus Junk | Asness, C. et al. | 2019 | Quality score: profitability + growth + safety + payout (✅ implemented) |
| A Five-Factor Model | Fama, E. & French, K. | 2015 | Factor model: value (P/B) + momentum + quality + low-vol + yield (✅ implemented) |
| Returns to Buying Winners/Selling Losers | Jegadeesh, N. & Titman, S. | 1993 | 12-1 momentum, momentum persistence (✅ implemented) |
| When are Contrarian Profits Due to Overreaction? | Lo, A.W. & MacKinlay, A.C. | 1990 | Mean reversion at 1-4 week horizon (✅ implemented) |
| Betting Against Beta | Frazzini, A. & Pedersen, L. | 2014 | Inverse-vol position sizing (✅ implemented) |
| Ensemble Methods in Machine Learning | Dietterich, T. | 2000 | Strategy blending via weighted vote (✅ implemented) |
| ...and the Cross-Section of Expected Returns | Harvey, C. et al. | 2016 | Multiple testing: t-stat > 3.0 for new factors |
| Optimal Execution of Portfolio Transactions | Almgren, R. & Chriss, N. | 2000 | Transaction cost in TB barriers (✅ implemented) |
| TradingAgents | arXiv:2412.20138 | 2024 | Bull/Bear/DA/Moderator debate + Risk Judge |
| Lopez-Lira & Tang | 2023 | Two-regime: quant-only historical, full LLM live |
| FinRL | arXiv:2011.09607 | 2020 | Data→Agent→Analytics architecture, turbulence index |
| Why tree-based models still outperform deep learning on tabular data | Grinsztajn, L. et al. | 2022 | Justification for GradientBoosting over deep learning |

---

## Data

- **Prices**: 299K rows, 149 tickers, 2018-01 to 2025-12 (BQ `historical_prices`)
- **Fundamentals**: 1,424 rows, 149 tickers (BQ `historical_fundamentals`, yfinance quarterly 2024+)
- **Macro**: 4,412 rows, 7 FRED series, 2018-2025 (BQ `historical_macro`)
- **Walk-forward**: ~24 expanding windows (12mo train + 3mo test + 5d embargo)
- **Baselines**: Buy-and-hold SPY, equal-weight top candidates, momentum-only top quartile

---

## What We Don't Do (and why)

- **No RL**: Data volume too low; prompt/param search is more sample-efficient at our scale
- **No fine-tuning**: Skills.md iteration is zero-cost vs GPU fine-tuning
- **No real-time streaming**: Daily cycle + 7-day outcome window → batch is correct
- **No optimizer self-modification**: Optimizers modify agents, never themselves (Karpathy: "modify train.py, never program.md")
- **No LLM in backtests**: Lopez-Lira (2023) contamination risk — quant-only for historical, full LLM for live

---

*Last updated: 2026-03-25 by Ford*
