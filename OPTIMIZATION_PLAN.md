# PyFinAgent Optimizer Speed & Architecture Plan

**Date:** 2026-03-27  
**Author:** Ford (AI Agent)  
**Status:** Draft — pending Peder's review  
**Goal:** 5-10x optimizer speedup without sacrificing quality  

---

## Current Bottleneck Analysis

**Current performance:** ~2h per experiment iteration (27 windows × 50 tickers)

| Phase | Time % | Bottleneck |
|-------|--------|------------|
| Feature building | ~60% | Sequential `build_feature_vector()` per ticker per date |
| ML training (GradientBoosting) | ~20% | Single-threaded sklearn |
| Data loading / BQ cache | ~10% | Already optimized with bulk preload |
| Label computation + trading | ~10% | Sequential per ticker |

**Root cause:** The inner loop in `_build_training_data()` iterates sequentially over `sample_dates × tickers` (e.g., 26 dates × 50 tickers = 1,300 calls per window × 27 windows = ~35,000 feature builds per experiment).

---

## Phase 1: Immediate Speedups (No Architecture Change)

### 1A. Vectorize Feature Building (~3-5x speedup)

**Current:** Calls `build_feature_vector()` one ticker at a time, each doing pandas operations on cached price data.

**Proposed:** Batch compute features across all tickers at once using vectorized pandas/numpy operations.

```python
# Instead of:
for sample_date in sample_dates:
    for ticker in tickers:
        fv = self.data_provider.build_feature_vector(ticker, sample_date)

# Do:
def build_features_batch(tickers, sample_dates, price_cache):
    """Vectorized feature computation across all tickers simultaneously."""
    # Compute momentum, RSI, volatility etc. as bulk array operations
    # One pass through price data per feature, not per ticker
```

**Impact:** Eliminates Python loop overhead, leverages numpy SIMD. Expected 3-5x on feature building phase alone.

### 1B. Parallel Window Processing with joblib (~2x speedup)

Windows are independent (by design — Bailey & López de Prado, 2014). Process them in parallel:

```python
from joblib import Parallel, delayed

# Instead of sequential:
for window in windows:
    wr = self._run_window(window, universe_tickers)

# Do parallel:
results = Parallel(n_jobs=4, prefer="threads")(
    delayed(self._run_window)(window, universe_tickers)
    for window in windows
)
```

**Caveat:** Trader state must be accumulated sequentially. Solution: run windows in parallel for feature+train+predict, then merge signals sequentially for trading.

### 1C. Feature Caching Across Experiments (~1.5x speedup)

Many features don't change between experiments (only ML hyperparams change, not the underlying data). Cache the feature matrix and only recompute when data-affecting params change.

```python
# Hash data-affecting params to create cache key
data_params = {k: v for k, v in params.items() 
               if k in ('frac_diff_d', 'top_n_candidates', 'holding_days', ...)}
cache_key = hashlib.md5(json.dumps(data_params, sort_keys=True).encode()).hexdigest()

# Reuse cached features when only ML params change
if cache_key == self._last_feature_cache_key:
    X, labels, weights = self._cached_features
else:
    X, labels, weights = self._build_training_data(...)
    self._cached_features = (X, labels, weights)
```

**Impact:** When optimizer only changes `n_estimators`, `max_depth`, `learning_rate`, etc., skip the entire feature building phase.

### 1D. Early Stopping for Experiments (~1.5x average speedup)

Inspired by Karpathy's autoresearch: if an experiment is clearly worse after N windows, abort early.

```python
# After processing first 10/27 windows, check interim Sharpe
if window.window_id >= 10 and interim_sharpe < self.best_sharpe * 0.85:
    logger.info(f"Early stopping: interim Sharpe {interim_sharpe:.4f} << best {self.best_sharpe:.4f}")
    break  # Don't waste time on remaining 17 windows
```

**Impact:** Bad experiments (which are most experiments) terminate in ~40% of the time.

---

## Phase 2: Karpathy Autoresearch Pattern (Architecture Upgrade)

### Core Principles from autoresearch

1. **Fixed time budget per experiment** — Currently each experiment takes variable time. Standardize to a fixed budget (e.g., 15 min) to make experiments comparable.

2. **Single file modification** — Our optimizer already follows this: only `_strategy_params` change between experiments.

3. **Autonomous loop with structured logging** — Our TSV log + JSON results match this pattern.

4. **Never stop** — The optimizer should run indefinitely, trying progressively more creative modifications.

### Improvements to Adopt

**A. Smarter Proposal Strategy (replace pure random)**

Current: Random perturbation of one parameter at a time.  
Proposed: Multi-armed bandit over parameter groups.

```python
# Track which params have historically yielded improvements
param_success_rates = {
    'learning_rate': 0.15,    # 15% of learning_rate changes improved
    'n_estimators': 0.05,     # only 5% helped
    'tp_pct': 0.20,           # promising
    ...
}
# Thompson sampling: preferentially try params with higher success rates
# Also: try multi-param changes (combinations) with decreasing probability
```

**B. Experiment Families (not just single-param changes)**

```python
# Level 1: Single param changes (current)
# Level 2: Correlated param pairs (tp_pct + sl_pct together)
# Level 3: Strategy-level changes (e.g., different feature subsets)
# Level 4: Architecture changes (different ML model types)
```

**C. Progress Checkpointing (from autoresearch git pattern)**

```python
# After each kept experiment, save full state:
# - Feature matrix (avoid recomputation)
# - Best model weights
# - Full experiment history for analysis
# This enables "rewind" if optimizer goes down a bad path
```

---

## Phase 3: Anthropic Harness Pattern (Multi-Agent Architecture)

### From "Harness Design for Long-Running Apps"

The key insights for our use case:

1. **Planner → Generator → Evaluator architecture** maps naturally to:
   - **Planner Agent:** Analyzes experiment history, proposes next experiments
   - **Generator Agent:** Runs the backtest with proposed parameters  
   - **Evaluator Agent:** Validates results, checks for overfitting, decides keep/discard

2. **Sprint contracts** — Before each experiment batch, the planner and evaluator agree on what "improvement" means (not just higher Sharpe — also DSR, max drawdown, consistency).

3. **Context resets** — For long optimization runs (100+ experiments), periodically reset the optimizer's "memory" to avoid confirmation bias. Fresh analysis of experiment history can reveal patterns the optimizer has become blind to.

### Implementation: LLM-Guided Research Mode

This is the `use_llm=True` mode that's already stubbed in `QuantStrategyOptimizer`:

```python
def _propose_llm(self, think_harder: bool = False) -> dict:
    """
    Use Gemini Flash to analyze experiment history and propose
    the next experiment. ~$0.01/proposal.
    
    The LLM sees:
    - Full experiment TSV history
    - Current best parameters
    - Which params have been tried and their outcomes
    - Feature importance rankings
    
    It proposes:
    - Which param(s) to change
    - Direction and magnitude
    - Reasoning (logged for audit)
    """
```

**Cost:** At ~$0.01/proposal and ~12 experiments/hour, this is $0.12/hour or ~$1/day. Negligible vs compute cost.

---

## Phase 4: Infrastructure & Parallelism

### 4A. Reduce Walk-Forward Windows

Current: 27 quarterly windows (2018-2025)  
Proposed: 18 windows (start from 2020, or use 4-month test periods)

```python
# Option A: Shorter history (2020-2025 instead of 2018-2025)
# Rationale: Pre-COVID regime may not be relevant
# Impact: 33% fewer windows → 33% faster

# Option B: Larger test windows (4 months instead of 3)
# Rationale: Fewer windows, more data per window
# Impact: ~25% fewer windows
```

### 4B. Tiered Evaluation

```python
# Quick eval (5 min): 10 windows, 30 tickers — rough estimate
# Full eval (20 min): All 27 windows, 50 tickers — confirmation
#
# Only promote to full eval if quick eval shows promise
# This is like autoresearch's 5-min training budget
```

### 4C. Mac Mini Specific: Use All CPU Cores

The Mac Mini has multiple performance cores. Currently the optimizer uses 1 core at 100%.

```python
# GradientBoosting: n_jobs=-1 (use all cores for training)
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(
    n_jobs=-1,  # Currently not set!
    ...
)

# Feature building: multiprocessing for ticker-level parallelism
from concurrent.futures import ProcessPoolExecutor
```

---

## Implementation Priority

| Priority | Change | Effort | Speedup | Risk |
|----------|--------|--------|---------|------|
| 🔴 P0 | Feature caching across ML-only experiments | 2h | 2-5x | None |
| 🔴 P0 | Early stopping for bad experiments | 1h | 1.5x avg | None |
| 🟡 P1 | Tiered evaluation (quick → full) | 4h | 3x | Low |
| 🟡 P1 | Parallel feature building (joblib) | 4h | 2x | Low |
| 🟡 P1 | Reduce windows (start 2020) | 30m | 1.5x | Medium* |
| 🟢 P2 | Vectorized feature computation | 8h | 3-5x | Medium |
| 🟢 P2 | Multi-armed bandit proposals | 4h | Indirect | Low |
| 🟢 P2 | LLM-guided proposals (Gemini Flash) | 4h | Indirect | Low |
| 🟢 P3 | Full multi-agent harness | 16h | Architecture | Medium |

*Medium risk: shorter history may miss important regime patterns

**Combined P0+P1 expected speedup: 5-10x** (from ~2h/experiment to ~15-25min/experiment)

---

## References

- **Karpathy autoresearch:** Fixed time budgets, autonomous loop, structured logging, simplicity criterion
- **Anthropic harness design:** Planner/Generator/Evaluator pattern, sprint contracts, context resets
- **Anthropic long-running agents:** Incremental progress, structured handoffs, feature lists
- **López de Prado (AFML):** Walk-forward independence, DSR guard, sample weights
- **Bailey & López de Prado (2014):** Deflated Sharpe Ratio for overfitting protection
