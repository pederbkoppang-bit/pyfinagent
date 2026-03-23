# Quant Strategy — Optimizer Research Guide

## Goal
Provide research-backed strategy knowledge to the optimizer's LLM-proposal path. Increase Sharpe ratio and Deflated Sharpe Ratio by guiding parameter proposals toward academically validated configurations and away from known anti-patterns. This skill is loaded by `_propose_llm()` in `quant_optimizer.py`.

## Identity
Optimizer skill — not a pipeline agent. Loaded by `quant_optimizer.py` when generating LLM-guided parameter proposals. Provides research context for all 5 strategies + hybrid blending rules so the LLM can make informed proposals rather than blind perturbation.

## What You CAN Modify (Fair Game)
- Strategy documentation and research citations
- Parameter range recommendations with justification
- Anti-pattern descriptions
- Experiment suggestions and template proposals
- Data gap analysis

## What You CANNOT Modify (Fixed Harness)
- `_PARAM_BOUNDS` definitions in `quant_optimizer.py`
- `STRATEGY_REGISTRY` in `backtest_engine.py`
- Label function signatures and feature vector schema
- Walk-forward window generation logic
- DSR computation methodology

---

## Strategy Registry — 5 Strategies

### 1. Triple Barrier (primary) — Lopez de Prado AFML Ch. 3

**Implementation**: `_compute_triple_barrier_label()` — walks forward through prices from entry_date, checking TP barrier (entry × (1 + tp_pct/100)), SL barrier (entry × (1 - sl_pct/100)). Label +1 if TP hit first, -1 if SL, 0 if time expired at holding_days.

**Academic basis**: "Advances in Financial Machine Learning" (Lopez de Prado, 2018), Chapter 3: The triple-barrier method labels observations based on which barrier is hit first — take-profit (upper), stop-loss (lower), or time expiration (vertical). This avoids the fixed-horizon labeling bias that plagues traditional classification approaches.

**Research-aligned improvements (document for optimizer to consider)**:
- **Volatility-adjusted barriers**: Literature recommends `TP = daily_vol × multiplier`, `SL = daily_vol × multiplier` instead of fixed percentages. The feature vector already contains `annualized_volatility`. Daily vol = annualized / sqrt(252). Multipliers typically range 1.0-5.0. Current fixed tp_pct/sl_pct work but don't adapt to volatility regimes.
- **Event-driven sampling**: CUSUM filter on cumulative log returns would replace biweekly calendar sampling. Not currently implemented.
- **Sample weights**: Average uniqueness correctly implemented per Ch. 4. No changes needed.
- **Fractional differentiation**: Correctly applied to non-stationary features per Ch. 5. No changes needed.

**Optimal param ranges (research-backed)**:
| Param | Range | Rationale |
|-------|-------|-----------|
| tp_pct | 5-20% | Narrow TP catches more wins but caps upside. >20% rarely hit in 90d window. |
| sl_pct | 3-12% | Tighter SL controls downside. Asymmetric TP/SL (TP > SL by 1.5-2x) is standard. |
| holding_days | 30-120 | 30d for higher-freq, 120d for position traders. >120d approaches buy-and-hold. |
| frac_diff_d | 0.3-0.6 | d=0.4 is the default. d<0.3 loses too much memory, d>0.6 may still be non-stationary. |

**Key experiment**: Try asymmetric barriers — set tp_pct = 2 × sl_pct. The risk-reward asymmetry is a fundamental edge.

---

### 2. Quality Momentum — Asness, Frazzini & Pedersen 2019

**Implementation**: `_compute_quality_momentum_label()` — uses `momentum_6m` and `quality_score` with absolute thresholds: >5 momentum AND >0.3 quality → BUY, <-5 momentum AND <0.1 quality → SELL.

**Academic basis**: "Quality Minus Junk" (Asness, Frazzini & Pedersen, 2019, Journal of Finance). QMJ strategy goes long high-quality, high-momentum stocks and short low-quality, low-momentum. Quality is a multi-dimensional composite: profitability (gross profits/assets, ROE, ROA) + growth (5-year profitability trend) + safety (low leverage, low vol) + payout (dividend yield, net issuance).

**Data gap analysis**:
- **Absolute vs cross-sectional thresholds**: Current hardcoded thresholds (momentum >5, quality >0.3) break across market regimes. Asness uses **quartile ranking within the universe** — top 25% by quality AND momentum = BUY. This requires access to the full universe at training time.
- **Incomplete quality composite**: Current `quality_score = ROE × profit_margin × (1 - D/E_norm)` covers profitability + safety only. Missing: `fcf_yield` (cash flow proxy), `dividend_yield` (payout), `revenue_growth_yoy` (growth). All three exist in the feature vector but are not used in the quality composite.
- **12-1 momentum**: Asness recommends 12-month return minus most recent month (avoiding short-term reversal). Feature vector has `momentum_12m` and `momentum_1m` separately — could compute 12-1 as `momentum_12m - momentum_1m`.

**Optimal param ranges (research-backed)**:
| Param | Range | Rationale |
|-------|-------|-----------|
| holding_days | 60-180 | Momentum has 3-12 month persistence (Jegadeesh & Titman 1993). |
| tp_pct | 10-25% | Momentum profits come from letting winners run. |
| sl_pct | 5-15% | Standard risk management. |

**Key experiment**: The optimizer should try widening quality_score to include FCF yield and dividend yield. Since label functions use features, a composite feature could be proposed.

---

### 3. Mean Reversion — Lo & MacKinlay 1990

**Implementation**: `_compute_mean_reversion_label()` — uses `sma_50_distance` and `rsi_14` with hardcoded thresholds: SMA dist < -5% AND RSI < 35 → BUY (oversold), SMA dist > 10% AND RSI > 70 → SELL (overbought).

**Academic basis**: "When are Contrarian Profits Due to Stock Market Overreaction?" (Lo & MacKinlay, 1990). Also Poterba & Summers 1988 (variance ratio tests), Jegadeesh 1990 (short-term reversals).

**CRITICAL ISSUE — Holding period mismatch**:
- Mean reversion works at **short horizons (1-4 weeks)**. At 3-12 months, **momentum dominates** (Jegadeesh & Titman, 1993).
- The shared `holding_days` param (range 30-252) was designed for Triple Barrier. Running MR with 90-252 day holds is effectively a momentum bet, not mean reversion.
- New param `mr_holding_days` (range 5-30) added to `_PARAM_BOUNDS` for this purpose.

**Data gap analysis**:
- **Strategy-specific holding**: `mr_holding_days` now available (5-30 range). Future label function can use this for a forward-looking mean-reversion barrier: "did price revert to SMA within mr_holding_days?"
- **Missing Bollinger Bands**: Standard MR signal uses 20-day SMA ± 2σ bands. Not in feature vector; would need `bb_upper`, `bb_lower` features or compute inline.
- **Liquidity filter**: `amihud_illiquidity` exists in features but not used in MR label. High Amihud = low liquidity = wider bid-ask spreads eat reversion profits. MR should prefer liquid stocks.

**Optimal param ranges (research-backed)**:
| Param | Range | Rationale |
|-------|-------|-----------|
| mr_holding_days | 5-20 | Short reversion windows. >20d transitions to momentum territory. |
| holding_days | 30-60 | For the TB time barrier when strategy=mean_reversion. |
| tp_pct | 3-10% | Small, frequent wins are the MR edge. |
| sl_pct | 3-8% | Tight stops — MR trades that don't revert quickly are wrong. |

**Key experiment**: Try `mr_holding_days=10` with `tp_pct=5`, `sl_pct=4`. This captures the short-term reversion premium.

---

### 4. Factor Model — Fama-French 2015

**Implementation**: `_compute_factor_label()` — 5-factor composite with hardcoded weights: value (P/E) 0.25, momentum (6m) 0.25, low_volatility 0.20, quality_score 0.20, dividend_yield 0.10. Hardcoded normalization ranges. Composite > 0.6 → BUY, < 0.3 → SELL.

**Academic basis**: "A Five-Factor Model of Expected Stock Returns" (Fama & French, 2015). Original 5 factors: Market (Mkt-Rf), Size (SMB), Value (HML), Profitability (RMW), Investment (CMA). Also: Novy-Marx 2013 "The Other Side of Value" (gross profitability is more predictive), Carhart 1997 (adding momentum as 4th factor).

**Data gap analysis**:
- **Missing Size factor (SMB)**: `market_cap` exists in features but not in the factor composite. Small-cap premium is one of the original Fama-French factors. Implementation: inverse percentile rank of market_cap.
- **Missing Investment factor (CMA)**: Conservative vs aggressive asset growth. No `asset_growth` feature. `revenue_growth_yoy` could proxy. Conservative (low growth) firms earn higher expected returns.
- **Value metric**: Uses `pe_ratio` — literature prefers `pb_ratio` (Price-to-Book) which has fewer edge cases with negative earnings. `pb_ratio` exists in features.
- **Hardcoded normalization**: Fixed ranges (P/E 5-30, vol 0.10-0.60) break for growth stocks (P/E > 100) and negative-earnings stocks. Cross-sectional **percentile ranking** within the universe is the standard approach.

**Optimal param ranges (research-backed)**:
| Param | Range | Rationale |
|-------|-------|-----------|
| holding_days | 60-252 | Factor premiums are slow-moving, 6-12 month rebalancing typical. |
| tp_pct | 10-30% | Wide barriers — factor model is a long-term approach. |
| sl_pct | 8-20% | Matching wide TP with reasonable SL. |

**Key experiment**: Try replacing pe_ratio with pb_ratio for the value factor, and adding size_score (inverse market_cap rank). The 5-factor model explanation power increases significantly with proper factor construction.

---

### 5. Meta-Label — Lopez de Prado AFML Ch. 3.6

**Implementation**: `meta_label` in `STRATEGY_REGISTRY` maps to `_compute_triple_barrier_label` — **complete stub**. No secondary model, no bet sizing layer.

**Academic basis**: AFML Ch. 3.6 — Meta-Labeling. Two-stage approach:
1. **Primary model**: Generate directional signal (+1/-1). Can be ANY rule-based system.
2. **Secondary model**: Predict whether the primary signal will be CORRECT. Output = probability of correctness → used for position sizing (fractional Kelly criterion).

**Current state**: The spirit of meta-labeling is partially captured by `size_position(probability, volatility, nav)` in `backtest_trader.py`, which uses GBC `predict_proba` to scale position size. But there's no true two-stage training.

**Architecture needed (future work)**:
1. Train primary model (e.g., triple_barrier GBC) → generate predictions on training set
2. Create meta-labels: was primary prediction correct? (binary 0/1)
3. Train secondary model on same features + primary model's signal + primary confidence
4. At prediction time: secondary model filters/sizes primary model's bets

**Status**: Document-only for now. Full implementation requires architectural changes to `_build_training_data()` and `_predict_and_trade()`.

---

## Hybrid Strategy — Ensemble Approach

**Concept**: Use outputs from multiple strategies as features into an ensemble model. Each strategy's label becomes a feature for a meta-classifier. Reference: Dietterich 2000 "Ensemble Methods in Machine Learning" (stacking approach).

**Current optimizer approach**: `_CATEGORICAL_PARAMS["strategy"]` picks ONE strategy per run. The optimizer can switch strategies between experiments but never blends them.

**Proposed extension**: Add blend weights (e.g., `tb_weight`, `qm_weight`, `mr_weight`, `fm_weight`) as floats 0-1 in `_PARAM_BOUNDS`. Weighted vote of strategy labels: `final_label = sign(sum(weight_i × label_i))`. This allows the optimizer to discover that (e.g.) 0.6 TB + 0.3 FM + 0.1 QM outperforms any single strategy.

---

## Parameter Bounds Reference — Research Justification

| Param | Min | Max | Research Basis |
|-------|-----|-----|---------------|
| tp_pct | 2.0 | 30.0 | Lopez de Prado: volatility-scaled, typical 1-5x daily vol → 3-25% range |
| sl_pct | 2.0 | 30.0 | Asymmetric with TP. Academic consensus: SL < TP by 1.5-2x |
| holding_days | 30 | 252 | Momentum persistence 3-12M (J&T 1993). Factor 6-12M. Trading costs penalize shorter. |
| mr_holding_days | 5 | 30 | Mean reversion horizon (Lo & MacKinlay 1990). >30d = momentum territory. |
| frac_diff_d | 0.1 | 0.8 | AFML Ch. 5: d=0.4 balances stationarity vs memory. d<0.3 loses info, d>0.6 may be non-stationary. |
| n_estimators | 50 | 500 | GBC standard. >500 diminishing returns + overfitting risk. |
| max_depth | 2 | 8 | Shallow trees (4-6) generalize better. Deep trees (>8) overfit to training noise. |
| min_samples_leaf | 5 | 50 | Higher = more regularized. 20 is good default for ~5000 training samples. |
| learning_rate | 0.01 | 0.3 | Lower LR + more estimators = better generalization (Friedman 2001). 0.05-0.1 typical sweet spot. |
| target_vol | 0.05 | 0.30 | AQR inverse-vol sizing. 0.10-0.15 for moderate risk. >0.20 for aggressive. |
| max_positions | 5 | 40 | Diversification: 20-30 positions captures most diversification benefit (Elton & Gruber). |
| top_n_candidates | 20 | 100 | Larger universe = more diversification but slower. 50 is good balance. |
| momentum_weight | 0.0 | 1.0 | Candidate screening composite weight. Higher → momentum-tilted universe. |
| rsi_weight | 0.0 | 1.0 | Candidate screening. RSI penalizes extremes (MR component in screening). |
| volatility_weight | 0.0 | 1.0 | Candidate screening. Higher → prefer lower-vol stocks. |
| sma_weight | 0.0 | 1.0 | Candidate screening. Positive SMA distance = above trend = bullish. |

---

## Anti-Patterns for the Optimizer

1. **MR with long holds** — If strategy=mean_reversion and holding_days>60, the backtest is capturing momentum, not reversion. Propose holding_days=30-45 instead.
2. **Symmetric barriers** — tp_pct == sl_pct yields a 50/50 gamble. Always propose asymmetric (TP > SL by 1.5-2x) unless testing a specific hypothesis.
3. **Single-strategy fixation** — If the last 5+ experiments used the same strategy, propose a strategy switch. Cross-strategy comparison is more informative than fine-tuning within one strategy.
4. **Extreme learning rate** — learning_rate > 0.2 with n_estimators > 300 causes overfitting. If proposing high LR, reduce n_estimators proportionally.
5. **Tiny candidate universe** — top_n_candidates < 25 limits diversification and makes results noisy. Prefer 40-60.
6. **Ignoring DSR** — A high Sharpe with DSR < 0.95 means the result is not statistically significant (too few trials, or overfitting to specific windows). Propose more conservative params if DSR is low.

---

## Experiment Suggestions

When generating proposals, consider these high-value experiments in order:

1. **Asymmetric barriers**: `tp_pct = 15, sl_pct = 8` (risk-reward = 1.88:1)
2. **Strategy rotation**: If stuck on one strategy for 3+ experiments, switch to the least-tested strategy
3. **MR with short hold**: `strategy=mean_reversion, mr_holding_days=10, tp_pct=5, sl_pct=4`
4. **Factor model with low vol**: `strategy=factor_model, target_vol=0.10, max_positions=30`
5. **Aggressive momentum**: `strategy=quality_momentum, holding_days=120, tp_pct=20, sl_pct=10`
6. **Regularization sweep**: `max_depth=3, min_samples_leaf=30, learning_rate=0.05` (reduce overfitting)
7. **Frac diff sensitivity**: Try `frac_diff_d=0.3` (more memory) vs `frac_diff_d=0.6` (more stationary)
8. **Universe size**: `top_n_candidates=30` (concentrated) vs `top_n_candidates=80` (diversified)

---

## Feature Vector — 43 Features Available

| Category | Features | Used By |
|----------|----------|---------|
| Price | price_at_analysis | All labels (entry price) |
| Momentum | momentum_1m, 3m, 6m, 12m | Quality Momentum (6m), Factor (6m), Candidate screening |
| Technical | rsi_14, sma_50_distance, sma_200_distance | Mean Reversion (RSI + SMA50), trend confirmation |
| Volume | volume_ratio_20d | Liquidity filter |
| Risk | annualized_volatility, var_95/99_6m, expected_shortfall_6m | Position sizing, risk assessment |
| Anomaly | anomaly_count | Regime detection |
| Liquidity | amihud_illiquidity | Execution risk (unused in labels — data gap) |
| Fundamental | pe_ratio, pb_ratio, roe, profit_margin, debt_equity, market_cap | Factor Model, Quality score |
| Cash Flow | fcf_yield | Quality composite (unused — data gap) |
| Income | dividend_yield | Factor Model (yield), Quality (unused — data gap) |
| Growth | revenue_growth_yoy | Investment factor proxy (unused — data gap) |
| Balance Sheet | total_revenue, net_income, total_debt, total_equity, total_assets | Quality scoring inputs |
| Macro | fed_funds_rate, cpi_yoy, unemployment_rate, yield_curve_spread, consumer_sentiment, treasury_10y | Regime awareness |
| Categorical | sector, industry | Not used in ML (excluded from training) |
