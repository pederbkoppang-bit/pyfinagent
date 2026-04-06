# Optimizer Speed-Up Plan

**Date:** 2026-03-28  
**Context:** Optimizer runs at ~100% CPU, experiments take 4-35 minutes each.  
**Goal:** 5-10x speed improvement without sacrificing Sharpe/DSR quality.

Based on research from:
- [Karpathy autoresearch](https://github.com/karpathy/autoresearch) — autonomous experiment loop
- [Anthropic harness design](https://www.anthropic.com/engineering/harness-design-long-running-apps) — multi-agent architecture
- Walk-forward ML optimization best practices

---

## Current Bottleneck Analysis

Each experiment runs a **27-window walk-forward backtest**. Per window:
1. **Feature building** (~80% of time): 5000-9000 iterations (ticker × biweekly dates)
2. **ML training** (~5%): HistGradientBoosting on ~5000 samples — already fast
3. **MDA/Permutation importance** (~5%): sklearn permutation_importance
4. **Prediction + Trading** (~10%): scoring test candidates

**The feature building loop is the bottleneck.** Each call to `build_feature_vector()` 
does price/fundamental/macro lookups + ~30 feature calculations, **single-threaded**.

Feature caching already exists for ML-only param changes (n_estimators, max_depth, etc.)
but ~60% of experiments change data-affecting params (tp_pct, sl_pct, holding_days, 
frac_diff_d, top_n_candidates) which invalidate the cache.

---

## Speed-Up Plan (Ordered by Impact × Safety)

### Phase 1: Quick Wins (No Quality Risk) — Expected 3-5x

#### 1A. Switch to HistGradientBoosting for all models ✅ ALREADY DONE
Already using `HistGradientBoostingClassifier`. Good.

#### 1B. Cache Feature Vectors at Ticker+Date Level (NOT window level)
**Current:** Cache invalidated when ANY data param changes.  
**Proposed:** Cache raw feature vectors keyed by `(ticker, date)`. Only invalidate 
when `frac_diff_d` or `top_n_candidates` change. Label computation (which depends on 
tp_pct, sl_pct, holding_days) is separate from features.

**Why:** ~70% of optimizer experiments change tp_pct/sl_pct/holding_days — these change 
LABELS but not FEATURES. Currently we rebuild features anyway because the cache key 
includes all data params.

**Implementation:**
```python
# In _build_training_data: separate feature building from label computation
# Cache features at (ticker, date) → feature_vector level
# Only recompute labels when barrier params change
```

**Expected speedup:** 3-5x for majority of experiments (feature build → cache hit + label recompute only)

#### 1C. Early Stopping on Losing Experiments
**Current:** Early stopping exists (`best_known_sharpe`) but only kicks in after window 10.  
**Proposed:** Start checking after window 7 (25% of run). If running Sharpe < 70% of best, abort.

**Why Karpathy-safe:** autoresearch pattern is "try fast, discard fast." Our experiments 
that end up discarded waste 100% of compute. Catching losers early saves ~50% on discards.

#### 1D. Reduce MDA Computation Frequency
**Current:** Permutation importance computed for every window.  
**Proposed:** Compute MDA only on windows 5, 15, 25 (every 10th). Use MDI proxy elsewhere.

### Phase 2: Moderate Changes (Minimal Quality Risk) — Expected 2-3x additional

#### 2A. Vectorized Label Computation
**Current:** `_compute_triple_barrier_label()` called per-ticker per-date in a Python loop.  
**Proposed:** Batch label computation using vectorized pandas operations across all 
tickers simultaneously.

#### 2B. Monthly Sampling (configurable)
**Current:** Biweekly samples → ~26 samples/ticker/year.  
**Proposed:** Add `sample_frequency` param. Default monthly (~12/ticker/year) for 
optimization, biweekly for final validation only.

**Quality guard:** Final validation run always uses biweekly to confirm results hold.

#### 2C. Adaptive Window Count
**Current:** Always run all 27 windows.  
**Proposed:** During optimization, run 15 windows (every other). Full 27 for validation.

**Academic backing:** López de Prado recommends stratified sampling of walk-forward 
windows for computational efficiency during optimization.

### Phase 3: Architecture (Autoresearch Pattern) — Expected 2-5x additional

#### 3A. Fixed Time Budget per Experiment (Karpathy Core Insight)
**Current:** Each experiment runs to completion regardless of time.  
**Proposed:** Set a **fixed wall-clock budget** per experiment (e.g., 2 minutes).  
If the experiment doesn't finish, treat as timeout/discard.

**Why:** From Karpathy's autoresearch: *"Training always runs for exactly 5 minutes. 
This means experiments are directly comparable regardless of what the agent changes."*

For us: experiments that take longer (e.g., large top_n_candidates) are automatically 
penalized, which naturally selects for configurations that are efficient AND good.

#### 3B. Separate Generator & Evaluator (Anthropic Harness Pattern)
**Current:** Single optimizer loop proposes and evaluates.  
**Proposed:** Split into:
- **Proposer:** Generates experiment hypotheses (random or LLM-guided)  
- **Runner:** Executes backtest with time budget
- **Evaluator:** Grades results against criteria (Sharpe, DSR, feature stability, 
  drawdown, etc.) — NOT just Sharpe improvement

**Why:** From Anthropic's harness paper: *"Separating the agent doing the work from the 
agent judging it proves to be a strong lever."* Our current optimizer only checks 
Sharpe + DSR. A separate evaluator could also check:
- Feature importance stability (MDA drift)
- Regime robustness (performance across market regimes)
- Overfitting signals (train vs test gap)
- Parameter sensitivity (is this near a cliff?)

#### 3C. Never Stop / Autonomous Mode (Karpathy Pattern)
**Current:** Optimizer runs for N iterations then stops.  
**Proposed:** Run indefinitely until manually stopped or target achieved. Keep a 
`results.tsv` log of all experiments. Human reviews in the morning.

From Karpathy: *"You are autonomous. If you run out of ideas, think harder... 
The loop runs until the human interrupts you, period."*

---

## Implementation Priority

| Phase | Item | Effort | Expected Speedup | Risk |
|-------|------|--------|-------------------|------|
| 1B | Ticker-level feature cache | 2h | 3-5x | None |
| 1C | Earlier early stopping | 30min | ~1.5x on discards | Very low |
| 1D | Reduce MDA frequency | 15min | ~1.1x | None |
| 2A | Vectorized labels | 3h | ~2x | Low |
| 2B | Monthly sampling | 1h | ~2x | Low (validated) |
| 2C | Adaptive window count | 1h | ~1.8x | Low (validated) |
| 3A | Time budget per experiment | 1h | Variable | Low |
| 3B | Generator/Evaluator split | 4h | Quality improvement | None |
| 3C | Autonomous mode | 1h | Throughput | None |

**Combined realistic estimate:** Current ~15 min/experiment → ~90 seconds/experiment  
(~10x improvement, enabling ~40 experiments/hour vs current ~4/hour)

---

## What We're NOT Changing (Quality Guards)

- Walk-forward methodology (no look-ahead bias)
- Deflated Sharpe Ratio threshold (prevents overfitting)
- Sample weights (average uniqueness)
- Fractional differentiation on non-stationary features
- Full validation run at biweekly sampling before declaring victory
- Commission model and slippage assumptions
