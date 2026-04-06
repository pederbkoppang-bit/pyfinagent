# Phase 0 — Formula Validation Findings

## 1. Sharpe Ratio — `analytics.py:compute_sharpe()`

### Our Implementation
```python
excess = returns - risk_free_rate / periods_per_year
std = excess.std()
return (excess.mean() / std) * np.sqrt(periods_per_year)
```

### Academic Reference
**Sharpe (1994)** "The Sharpe Ratio", Journal of Portfolio Management:
- SR = mean(D) / std(D), where D = differential return (fund return - benchmark return)
- Time scaling: SR_T = SR_1 × √T (under IID assumption)
- "To maximize information content, it is usually desirable to measure risks and returns using fairly short (e.g. monthly) periods. For purposes of standardization it is then desirable to annualize the results."

**Lo (2002)** "The Statistics of Sharpe Ratios", Financial Analysts Journal:
- The √T annualization rule requires IID returns
- For non-IID returns (autocorrelated): SR_annual = SR_monthly × √12 × η, where η is a correction factor based on autocorrelation
- Key insight: "the annual Sharpe ratio of a strategy that has monthly returns with first-order serial correlation ρ is SR_annual = SR_monthly × √(12 + 2×11×ρ + 2×10×ρ² + ...)"

### Verdict: ✅ CORRECT (with documented limitations)
- Formula matches Sharpe (1994) ex-post definition
- risk_free_rate / periods_per_year correctly converts annual rate to per-period rate
- √252 annualization is correct for daily returns under IID assumption
- **Limitation (documented, not a bug)**: We don't adjust for serial autocorrelation per Lo (2002). This could inflate Sharpe for momentum strategies where returns are positively autocorrelated. Future improvement: add Newey-West or Lo(2002) correction.
- **Minor note**: `excess.std()` uses Bessel's correction (N-1 denominator) via numpy default — this matches Sharpe's ex-post formula which uses sample std.

### Source
- Sharpe, W.F. (1994). "The Sharpe Ratio." Journal of Portfolio Management, 21(1), 49-58. Available: https://web.stanford.edu/~wfsharpe/art/sr/SR.htm
- Lo, A.W. (2002). "The Statistics of Sharpe Ratios." Financial Analysts Journal, 58(4), 36-52.

---

## 2. Deflated Sharpe Ratio — `analytics.py:compute_deflated_sharpe()`

### Our Implementation
```python
# E[max(SR)] via Euler-Mascheroni approximation
e_max_sr = sqrt(V) * [(1-γ)*Φ^{-1}(1-1/N) + γ*Φ^{-1}(1-1/(N*e))]

# Standard error of SR (non-normality adjusted)
se_sr = sqrt((1 - skewness*SR + (kurtosis-1)/4 * SR²) / T)

# Test statistic
z = (observed_SR - e_max_sr) / se_sr
DSR = Φ(z)
```

### Academic Reference
**Bailey & López de Prado (2014)** "The Deflated Sharpe Ratio", Journal of Portfolio Management:
- DSR = PSR(SR*) where SR* = E[max(SR)] under the null hypothesis of no skill
- E[max(SR)] from the False Strategy Theorem (FST): 
  E[max(SR)] ≈ √(V(SR)) × [(1-γ)Φ⁻¹(1-1/N) + γΦ⁻¹(1-1/(Ne))]
  where γ ≈ 0.5772 (Euler-Mascheroni constant), N = number of trials
- SE(SR) = √((1 - γ₃×SR + (γ₄-1)/4 × SR²) / T)
  where γ₃ = skewness, γ₄ = kurtosis, T = sample length
- DSR = Φ((SR_observed - SR*) / SE(SR))
- DSR ≥ 0.95 means the observed SR is statistically significant

### Verdict: ✅ CORRECT
- E[max(SR)] formula matches the FST from Bailey & LdP (2014) exactly
- SE formula matches the non-normality adjustment from the paper
- Euler-Mascheroni constant γ = 0.5772 correctly used
- T < 10 guard is reasonable (too few observations for meaningful DSR)
- **Potential issue**: We use `variance_of_srs = 0.5` as default when we can't compute it from window Sharpes. The paper recommends estimating this from the actual trial variance. Our actual computation uses `sr_variance = float(np.var(window_sharpes))` when available, falling back to 0.5 — this is conservative and acceptable.
- **Potential issue**: `num_trials` should ideally account for correlated trials. The paper recommends clustering to estimate effective N. We use raw trial count, which is more conservative (higher E[max(SR)] threshold → harder to pass DSR). This is acceptable and actually safer than under-counting.

### Source
- Bailey, D.H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality." Journal of Portfolio Management, 40(5), 94-107.
- Wikipedia: https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio (confirmed formula matches)

---

## 3. Sample Weights (Average Uniqueness) — `backtest_engine.py:_compute_sample_weights()`

### Our Implementation
```python
for i in range(n):
    overlap_count = 0
    for j in range(n):
        if i == j: continue
        if entry_dates[j] < exit_dates[i] and exit_dates[j] > entry_dates[i]:
            overlap_count += 1
    weights[i] = 1.0 / (1.0 + overlap_count)
weights = weights * n / weights.sum()  # Normalize
```

### Academic Reference
**López de Prado (2018)** AFML Ch. 4 "Sample Weights":
- "The average uniqueness of a label is the average, over the label's lifespan, of the proportion of concurrent labels that are active at each time step"
- More precisely: for each observation i, compute the uniqueness at each time step t during its label's lifespan:
  u_i(t) = 1 / c_t, where c_t = number of concurrent labels at time t
  Then: w_i = mean(u_i(t)) over all t in [entry_i, exit_i]

### Verdict: ⚠️ APPROXIMATE (simplified but directionally correct)
- Our implementation counts the total number of overlapping labels and uses 1/(1+count) as the weight
- AFML's exact method computes a time-step-by-time-step average of inverse concurrency
- **Difference**: Our method gives equal weight to all overlapping labels regardless of how much they overlap. AFML's method accounts for partial overlap (a label that overlaps by 1 day gets less penalty than one that overlaps by 60 days)
- **Impact**: For our biweekly sampling with 90-day holding periods, most labels will have significant overlap. The simplified version will over-penalize labels with many short overlaps and under-penalize those with few long overlaps.
- **Recommendation**: Keep for now (directionally correct), flag for Phase 1 improvement. The O(n²) computation is also a concern — AFML uses an interval-based approach that's O(n × T) where T is the number of distinct time steps.
- **Performance**: With ~1200 samples per window, O(n²) ≈ 1.4M comparisons. Acceptable but will scale poorly.

### Source
- López de Prado, M. (2018). "Advances in Financial Machine Learning." Chapter 4: Sample Weights. Wiley.

---

## 4. Fractional Differentiation — `historical_data.py:fractional_diff()`

### Our Implementation
```python
weights = [1.0]
k = 1
while True:
    w = -weights[-1] * (d - k + 1) / k
    if abs(w) < threshold: break
    weights.append(w)
    k += 1
weights = np.array(weights[::-1])
# Apply via convolution
result[i] = np.dot(weights, series[i-width+1:i+1])
```

### Academic Reference
**López de Prado (2018)** AFML Ch. 5 "Fractionally Differentiated Features":
- The weights for fractional differentiation of order d are:
  w_0 = 1, w_k = -w_{k-1} × (d - k + 1) / k for k ≥ 1
- Fixed-width window: truncate weights where |w_k| < threshold (e.g., 1e-5)
- Apply as convolution: y_t = Σ_{k=0}^{K} w_k × x_{t-k}
- d = 0.4 is recommended as a starting point that balances stationarity and memory preservation

### Verdict: ✅ CORRECT
- Weight computation formula matches AFML Ch. 5 exactly
- Fixed-width window with threshold truncation is the recommended approach
- Convolution application via dot product is correct
- d = 0.4 default is the AFML recommendation
- **Note**: The weights are reversed before application (`weights[::-1]`) which is correct — w_0 (most recent) should align with the latest value in the window.
- **Improvement opportunity**: Could add ADF (Augmented Dickey-Fuller) test to dynamically find minimum d that achieves stationarity, as recommended in AFML. Currently d is a fixed parameter tuned by the optimizer.

### Source
- López de Prado, M. (2018). "Advances in Financial Machine Learning." Chapter 5: Fractionally Differentiated Features. Wiley.

---

## 5. Triple Barrier Labels — `backtest_engine.py:_compute_triple_barrier_label()`

### Our Implementation
```python
entry_price = prices["close"].iloc[0]
tp_price = entry_price * (1 + tp_pct / 100)
sl_price = entry_price * (1 - sl_pct / 100)
# Walk forward through prices
for idx in range(1, len(prices)):
    trading_days += 1
    price = prices["close"].iloc[idx]
    if price >= tp_price: return 1   # TP hit
    if price <= sl_price: return -1  # SL hit
    if trading_days >= holding_days: return 0  # Time expired
```

### Academic Reference
**López de Prado (2018)** AFML Ch. 3 "Labeling":
- Triple Barrier Method: set 3 barriers — upper (take-profit), lower (stop-loss), vertical (max holding period)
- Label = +1 if upper barrier hit first, -1 if lower, 0 if vertical (time expires)
- AFML recommends volatility-adjusted barriers: TP = SL = daily_vol × multiplier
- Also recommends event-driven sampling (CUSUM filter) rather than calendar sampling

### Verdict: ✅ CORRECT (basic implementation)
- Core logic matches AFML Ch. 3
- Walking forward through prices and checking barriers in order is correct
- `trading_days` counts actual price rows (trading days), not calendar days ✓
- Uses close prices (end-of-day) which is standard for daily backtesting

### Known Limitations (not bugs, but improvements for Phase 1):
1. **Fixed percentage barriers** — AFML recommends volatility-adjusted barriers. `tp_pct` and `sl_pct` are static across all stocks and time periods. A stock with 50% annual vol needs very different barriers than one with 15%.
2. **No transaction cost in labels** — the TP barrier doesn't account for spread/commission. A trade that "hits" TP at exactly the barrier price may actually lose money after costs.
3. **Calendar sampling** — we sample at biweekly intervals. AFML recommends CUSUM-filtered event-driven sampling to capture information arrivals more efficiently.
4. **Close-price execution** — assumes we can trade at the close price, which is approximately achievable with MOC (Market On Close) orders but not guaranteed.

### Source
- López de Prado, M. (2018). "Advances in Financial Machine Learning." Chapter 3: Labeling. Wiley.

---

## 6. Monte Carlo VaR — `historical_data.py:_compute_monte_carlo_var()`

### Our Implementation
```python
mu = daily_returns.mean()
sigma = daily_returns.std()
rng = np.random.default_rng(42)  # Deterministic
z = rng.standard_normal((1000, horizon_days))
daily_drift = mu - 0.5 * sigma**2
paths = current_price * np.exp(np.cumsum(daily_drift + sigma * z, axis=1))
```

### Academic Reference
**Geometric Brownian Motion (GBM)** — standard model:
- dS = μSdt + σSdW
- Discrete: S_{t+1} = S_t × exp((μ - σ²/2)Δt + σ√Δt × Z)
- For daily: Δt = 1/252, but when using daily μ and σ directly: S_{t+1} = S_t × exp((μ - σ²/2) + σZ)

### Verdict: ✅ CORRECT (with documented limitations)
- GBM formula correctly implemented with drift adjustment (-0.5σ²)
- Cumulative sum of log returns then exponentiation is the standard approach
- VaR at 95th/99th percentile and Expected Shortfall correctly computed
- **Deterministic seed (42)** — acceptable for backtesting (reproducibility), but means the same "random" paths are generated for every feature vector. This is fine because Monte Carlo VaR here is used as a FEATURE, not as a risk measure. The determinism ensures the same input always produces the same feature value.
- **Fat tail limitation** — GBM assumes log-normal returns. Real equity returns have fat tails (kurtosis > 3). This underestimates tail risk. Consider adding a note or supplementing with historical simulation.
- **1000 simulations** — adequate for feature computation (not for actual risk management where 10K+ is standard)

### Source
- Hull, J.C. "Options, Futures, and Other Derivatives" — GBM standard reference
- Glasserman, P. (2003). "Monte Carlo Methods in Financial Engineering." Springer.

---

## 7. Position Sizing (Inverse Volatility) — `backtest_trader.py:size_position()`

### Our Implementation
```python
vol_scale = min(target_vol / stock_vol, 3.0)
raw = probability × vol_scale × nav / max_positions
capped = min(raw, nav × max_single_pct)
```

### Academic Reference
- **Inverse volatility weighting** is used by AQR and described in Frazzini & Pedersen (2014) "Betting Against Beta"
- The idea: allocate more capital to low-volatility assets, less to high-volatility
- Kelly criterion connection: optimal bet size is proportional to edge / variance
- AQR formula: w_i = (1/σ_i) / Σ(1/σ_j), then scale to target portfolio volatility

### Verdict: ⚠️ APPROXIMATE (reasonable but non-standard formula)
- The `probability × vol_scale` component mixes two ideas: Kelly sizing (probability of win) and inverse-vol weighting (1/σ). This is a reasonable hybrid but not a standard formula from any single academic source.
- The `/ max_positions` divisor provides a simple diversification limit
- The 3x cap on vol_scale prevents extreme concentration in low-vol stocks ✓
- `max_single_pct` cap prevents excessive single-stock concentration ✓
- **Not a bug**, just a custom formula. Should be documented as "inspired by inverse-vol + Kelly" rather than attributed to any specific paper.
- **Improvement opportunity**: Implement proper fractional Kelly: f* = (p×b - q) / b, where p = probability, b = payoff ratio, q = 1-p. Then scale by inverse vol for portfolio construction.

### Source
- Frazzini, A. & Pedersen, L.H. (2014). "Betting Against Beta." Journal of Financial Economics, 111(1), 1-25.
- Kelly, J.L. (1956). "A New Interpretation of Information Rate." Bell System Technical Journal.

---

## 8. Scalar Metric — `perf_metrics.py:get_scalar_metric()`

### Our Implementation
```python
scalar = avg_return_pct × benchmark_beat_rate × (1 - min(0.3, turnover × tx_cost))
```

### Academic Reference
This is a **custom metric** — not from any single academic paper. It combines:
1. Average return (standard)
2. Benchmark beat rate (% of trades that beat SPY — non-standard as a multiplicative factor)
3. Transaction cost penalty (non-standard capping at 30%)

### Verdict: ⚠️ CUSTOM (functional but needs theoretical justification)
- The metric serves its purpose as a single optimization target
- **Concern**: `avg_return_pct × benchmark_beat_rate` creates a product of two related measures. If you have a 10% avg return with 50% beat rate, you get 5.0. But you'd get the same 5.0 with 5% avg return and 100% beat rate — these are very different strategies.
- **Concern**: The tx cost penalty caps at 30%, meaning a strategy with extremely high turnover only gets a 30% haircut. For high-frequency rebalancing, actual costs could eat 100%+ of returns.
- **Recommendation**: Consider using Sharpe ratio (already our primary backtest metric) as the unified scalar metric across all loops, since it's already risk-adjusted. The tx cost should be baked into returns directly (which the backtest trader already does via commission model) rather than as a multiplicative penalty.

### Source
- Custom metric. No single academic reference. Documented as "Karpathy-inspired single metric with PyFinAgent extensions."

---

## Summary of Findings

| Formula | Verdict | Action Needed |
|---------|---------|---------------|
| Sharpe Ratio | ✅ Correct | Document Lo(2002) autocorrelation limitation |
| Deflated Sharpe Ratio | ✅ Correct | None (conservative defaults are good) |
| Sample Weights | ⚠️ Approximate | Flag for Phase 1 (AFML exact method) |
| Fractional Differentiation | ✅ Correct | Consider adding ADF test |
| Triple Barrier Labels | ✅ Correct | Phase 1: add vol-adjusted barriers |
| Monte Carlo VaR | ✅ Correct | Document fat-tail limitation |
| Position Sizing | ⚠️ Approximate | Document as custom hybrid formula |
| Scalar Metric | ⚠️ Custom | Consider aligning with Sharpe as primary |

**Overall**: The core quantitative engine is sound. No critical bugs in the fundamental formulas. The issues are:
1. Simplifications vs full academic implementations (sample weights, position sizing)
2. Missing features that the literature recommends (vol-adjusted barriers, autocorrelation correction)
3. Custom formulas that work but need better documentation (scalar metric, position sizing)

The biggest **real** issues for making money are in Phase 0.2 (bug fixes): the incomplete quality score, hardcoded factor normalization, and non-functional meta-label strategy. These affect signal quality directly.
