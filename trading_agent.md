# trading_agent.md — Living Memory for PyFinAgent Trading Optimization

> This file is the persistent instruction set and memory for the autonomous trading
> optimization system. It follows the Recursive Research Loop (RRL) pattern from
> karpathy/autoresearch, extended with hybrid elements for multi-loop financial optimization.

---

## 1. Agent Identity

**Mission**: Continuously improve PyFinAgent's trading recommendations by optimizing
three interacting feedback loops — quant strategy parameters, LLM agent prompts, and
API performance — while maintaining statistical rigor and avoiding overfitting.

**Constraints (NEVER violate)**:
- NEVER modify the fixed harness: orchestrator pipeline, output JSON schemas, BQ schema, evaluation formula, function signatures
- NEVER use LLM models for historical backtests (Lopez-Lira 2023 contamination risk)
- NEVER bypass the DSR ≥ 0.95 guard on quant experiments
- NEVER deploy prompt changes without outcome validation (7+ day window)
- NEVER exceed `MAX_ANALYSIS_COST_USD` or `PAPER_MAX_DAILY_COST_USD` budget caps
- ALWAYS keep simplicity criterion: ≥0.5% delta per 10 lines added to skills.md
- ALWAYS log every experiment to TSV (Karpathy rule: if it's not logged, it didn't happen)

**What IS modifiable** (the "train.py" equivalent):
- `## Prompt Template`, `## Skills & Techniques`, `## Anti-Patterns` sections of each `backend/agents/skills/*.md`
- 15 quant strategy parameters in `QuantStrategyOptimizer` (with bounds)
- API cache TTL values in `ENDPOINT_TTLS`

---

## 2. Research Summary

### What We've Learned from Backtesting

**~43 Feature Vector** built at any historical cutoff date:
- Price & Returns: `price_at_analysis`, momentum (1M/3M/6M/12M), volatility (1M/3M)
- Technical: RSI_14, SMA_50/200 distance, volume ratio
- Monte Carlo: VaR (95%/99%), expected shortfall, P(positive)
- Fundamentals: P/E, P/B, D/E, ROE, revenue growth, net margin, current ratio
- Macro: Fed Funds, CPI, unemployment, GDP growth, yield curve, consumer sentiment
- Advanced: Amihud illiquidity, turbulence index
- Fractionally differenced: price, market_cap, revenue, debt, equity

**MDA Feature Importance** (from walk-forward backtest, most→least predictive):
- *(To be populated after first backtest run — this is the bridge to SkillOpt)*

**Baseline Strategies** (from analytics.py):
1. Buy-and-hold SPY
2. Equal-weight top candidates
3. Momentum-only (top quartile by trailing 6M return)

**Research Foundations Powering the System**:

| Source | What It Gives Us |
|--------|-----------------|
| Harvard NBER | Focus on non-routine 29% — debate, anomalies, contradictions |
| Stanford Transformers | NLP sentiment via Vertex AI embeddings (not keyword) |
| Goldman Sachs 127-dim | Anomaly detection (Z-score) + Monte Carlo VaR (1K sims) |
| TradingAgents arXiv:2412.20138 | Bull/Bear/DA/Moderator debate + Risk Judge sizing |
| López de Prado AFML | Triple Barrier, sample weights, frac diff, walk-forward, MDA |
| Bailey & López de Prado 2014 | DSR overfitting guard (≥0.95 = statistically significant) |
| Lopez-Lira & Tang 2023 | Two-regime: quant-only historical, full LLM for live |
| FinRL arXiv:2011.09607 | Data→Agent→Analytics three-layer architecture |
| BlackRock domain LLMs | Skill-based prompt engineering on financial domain |
| VeNRA + CoVe | Fact ledger + chain-of-verification anti-hallucination |

---

## 3. Implementation Progress

### Phase 1: PerformanceSkill ✅
- [x] Created `backend/services/perf_metrics.py` with canonical formulas
- [x] `compute_position_pnl()` — position-level P&L
- [x] `compute_return_pct()` — simple return percentage
- [x] `compute_sharpe_from_snapshots()` — NAV→daily returns→canonical Sharpe
- [x] `compute_benchmark_return()` — geometric (not arithmetic) benchmark
- [x] `get_scalar_metric()` — THE unified metric with tx cost penalty
- [x] Deduplicated: paper_trading.py Sharpe → `compute_sharpe_from_snapshots`
- [x] Deduplicated: backtest_engine._sharpe → delegates to `analytics.compute_sharpe`
- [x] Deduplicated: outcome_tracker benchmark → geometric `compute_benchmark_return`
- [x] Deduplicated: portfolio.py P&L → `compute_position_pnl`
- [x] Wired: skill_optimizer.compute_metric → `get_scalar_metric_from_bq`

### Phase 2: trading_agent.md ✅
- [x] This file

### Phase 3: MetaCoordinator
- [x] Created `backend/agents/meta_coordinator.py`
- [x] Loop sequencing: QuantOpt → MDA extraction → SkillOpt targeting
- [x] Decision logic: low Sharpe → QuantOpt, low accuracy → SkillOpt, high latency → PerfOpt
- [ ] Wire into autonomous_loop.py (requires testing)

### Phase 4: Docs
- [ ] Update AGENTS.md with Phase 1-3 changes
- [ ] Update UX-AGENTS.md

---

## 4. Performance Skill API

**Module**: `backend/services/perf_metrics.py`

### Position-Level
```python
compute_position_pnl(quantity, current_price, cost_basis) → (pnl, pnl_pct)
compute_return_pct(current_price, entry_price) → float
```

### Portfolio-Level
```python
compute_portfolio_pnl(nav, starting_capital) → (pnl, pnl_pct)
compute_alpha(portfolio_pnl_pct, benchmark_pnl_pct) → float
compute_sharpe_from_snapshots(snapshots, nav_key, risk_free_rate) → float
```

### Benchmark
```python
compute_benchmark_return(holding_days, annual_rate=0.10) → float  # geometric
beat_benchmark(return_pct, holding_days, annual_rate=0.10) → bool
```

### Turnover & Cost
```python
compute_turnover_ratio(trades, avg_nav, period_days=365) → float
compute_tx_cost_drag(turnover_ratio, tx_cost_pct=0.001) → float
```

### Scalar Metric (THE metric all loops optimize)
```python
get_scalar_metric(ScalarMetricInputs) → float
# = risk_adjusted_return × (1 − min(0.3, turnover_ratio × tx_cost_pct))
# where risk_adjusted_return = avg_return_pct × benchmark_beat_rate

get_scalar_metric_from_bq(bq_client, trades=None) → float  # convenience
```

---

## 5. Iteration Loop (Karpathy-Adapted Hybrid)

### Core Loop Rules (from karpathy/autoresearch)
```
LOOP FOREVER:
  1. BASELINE FIRST — measure before modifying anything
  2. PROPOSE one modification (random or LLM-guided)
  3. APPLY the modification
  4. MEASURE the scalar metric
  5. KEEP if metric improved (with guards), DISCARD otherwise
  6. LOG to TSV — every experiment, kept or discarded
  7. NEVER STOP — the loop runs until externally stopped
```

### Three Loops, One Metric

| Loop | What It Modifies | Speed | Guard | Metric Source |
|------|-----------------|-------|-------|---------------|
| **QuantOpt** | 15 strategy params (bounds-constrained) | Minutes/cycle | DSR ≥ 0.95 | `analytics.compute_sharpe` on backtest |
| **SkillOpt** | Prompt Template / Skills / Anti-Patterns in skills.md | Days/cycle | Simplicity criterion | `get_scalar_metric_from_bq` (outcome_tracking) |
| **PerfOpt** | API cache TTL values | Minutes/cycle | ≥5% latency improvement | `perf_tracker.summarize` (p95 latency) |

### MetaCoordinator Sequencing
```
1. Check portfolio health signals
2. IF Sharpe < target         → run QuantOpt cycle
3. IF agent accuracy < target → run SkillOpt cycle
4. IF p95 latency > threshold → run PerfOpt cycle
5. AFTER QuantOpt keep → extract MDA features → map to responsible agents → queue SkillOpt
6. AFTER SkillOpt keep → queue 1-window backtest as fast validation proxy
```

### MDA→Agent Bridge (our unique innovation)
When QuantOpt discovers that certain features matter more:
- High MDA on `nlp_sentiment_score` → target NLP Sentiment Agent skill
- High MDA on `insider_signal` → target Insider Activity Agent skill
- High MDA on `yield_curve_spread` → target Enhanced Macro Agent skill

This closes the loop: quant insights → prompt improvements → better recommendations.

### What We DON'T Do (and why)
- **No RL (Reinforcement Learning)**: Data volume too low. Prompt search is more sample-efficient at our scale.
- **No fine-tuning**: Skills.md iteration is zero-cost vs fine-tuning GPU time. Correct choice.
- **No real-time streaming**: Daily cycle + 7-day outcome window makes batch the right approach.
- **No optimizer self-modification**: Optimizers modify agents, never themselves (Karpathy: "modify train.py, never program.md").

---

## 6. Strategy Research

### Research-Backed Strategy Registry

The backtest engine now supports multiple named strategies via `STRATEGY_REGISTRY`. The QuantOptimizer treats `strategy` as a categorical parameter, enabling autonomous strategy rotation.

| Strategy | Label Method | Key Features | Research Basis |
|----------|-------------|--------------|----------------|
| **triple_barrier** (default) | TP/SL/Time barriers | All 43 features | López de Prado Ch. 3 — original implementation |
| **quality_momentum** | Momentum + quality filter | momentum_12m, roe, profit_margin, debt_equity, quality_score | Asness et al. (2019) "Quality Minus Junk", Novy-Marx (2013) — momentum alpha persists when filtered by quality |
| **mean_reversion** | Z-score reversion bands | sma_50_distance, sma_200_distance, rsi_14, volume_ratio_20d, amihud_illiquidity | Poterba & Summers (1988), Lo & MacKinlay (1990) — mean-reversion strongest in liquid large-caps at 1-3 month horizon |
| **factor_model** | Multi-factor composite | pe_ratio, pb_ratio, dividend_yield, momentum_6m, annualized_volatility | Fama-French 5-factor (2015), Carhart (1997) — value + momentum + low-vol anomaly composite |
| **meta_label** | Triple Barrier meta-labeled | All features + primary model probability | López de Prado Ch. 3.6 — secondary model learns WHEN primary model is reliable |

### New Features Added (historical_data.py)

| Feature | Category | Computation | Used By Strategies |
|---------|----------|-------------|-------------------|
| `fcf_yield` | Fundamental | (operating_cash_flow − capex) / market_cap | factor_model, quality_momentum |
| `dividend_yield` | Fundamental | dividends_per_share / price | factor_model |
| `quality_score` | Composite | ROE × profit_margin × (1 − debt_equity_ratio_normalized) | quality_momentum |
| `pb_ratio` | Fundamental | price / (total_equity / shares) | factor_model |
| `volume_ratio_20d` | Technical | current_volume / 20d_avg_volume | mean_reversion |
| `revenue_growth_yoy` | Fundamental | (current_Q_revenue − year_ago_Q_revenue) / year_ago_Q_revenue | quality_momentum |

### Candidate Selector Weight Configurability

`CandidateSelector.screen_at_date()` now accepts `scoring_weights: dict` parameter, allowing QuantOptimizer to tune `momentum_weight`, `rsi_weight`, `volatility_weight`, `sma_weight` as part of the fast loop.

### Feature Drift Detection

After each QuantOpt keep, the top-5 MDA features are logged to `quant_results.tsv` as `top5_mda` column. When the top-5 differs from the previous keep's top-5, a WARNING is logged for MetaCoordinator to act on via the MDA→Agent bridge.

### Model Staleness Guards

- `model_trained_at: str` timestamp stored on `BacktestEngine` after each `_train_model()` call
- `get_model_trained_at()` public method for external staleness checks
- After each QuantOpt keep, the engine retrains automatically (new params → new model)
- Weekly staleness alert: if `model_trained_at` > 7 days old, QuantOptimizer logs a STALE_MODEL warning

### Auto-Ingest at Backtest Start

`run_backtest()` now checks BQ table row counts via `DataIngestionService.get_ingestion_status()` before the walk-forward loop. If `historical_prices` has 0 rows, auto-triggers `run_full_ingestion()` using the engine's BQ client and universe tickers. This removes the manual "ingest first" prerequisite — backtests self-bootstrap.

---

## 7. Phase 5 Implementation Plan

### Completed (Phase 5A — Strategy Research + Mock Testing)

| Step | Description | Status |
|------|-------------|--------|
| 6.1 | Strategy research documented in Section 6 | DONE |
| 6.2 | Strategy registry in `backtest_engine.py` (5 strategies) | DONE |
| 6.3 | New features in `historical_data.py` (6 features) | DONE |
| 6.4 | Configurable `candidate_selector.py` weights | DONE |
| 6.5 | `strategy` as 16th QuantOptimizer param + feature drift logging | DONE |
| 6.6 | Model staleness tracking | DONE |
| 6.7 | Auto-ingest check at backtest start | DONE |
| 6.8 | Mock-test script (`t_backtest_mock.py`) | DONE |

### Pending (Phase 5B — ML→Live Bridge)

| Step | Description | Status |
|------|-------------|--------|
| 7.1 | Create `backend/tools/quant_model.py` (12th enrichment signal) | DONE |
| 7.2 | Wire into `orchestrator.py` Step 6 (7 integration points) | DONE |
| 7.3 | Add `quant_model_agent.md` skill file | DONE |
| 7.4 | Wire into `prompts.py` + `signals.py` | DONE |
| 7.5 | Frontend: add QuantModel signal card to SignalCards.tsx | DONE |

### Pending (Phase 5C — SkillOpt Iteration)

| Step | Description | Status |
|------|-------------|--------|
| 8.1 | MetaCoordinator wired to `autonomous_loop.py` | DONE |
| 8.2 | MDA→Agent bridge live targeting | DONE |
| 8.3 | Proxy validation (1-window backtest for SkillOpt) | DONE |

### Phase 5D — Backtest BQ Client Type Fix + Schema Alignment (March 2026)

| Step | Description | Status |
|------|-------------|--------|
| 9.1 | Fix `bq_client=bq` → `bq_client=bq.client` in `backend/api/backtest.py` (2 call sites) | DONE |
| 9.2 | Defensive unwrap guard in `BacktestEngine.__init__()` | DONE |
| 9.3 | Upgrade `logger.debug` → `logger.warning` in `candidate_selector.py` | DONE |
| 9.4 | Backend `generate_report()` field alignment with frontend TS types (prior session) | DONE |
| 9.5 | Update `trading_agent.md` + `AGENTS.md` + `UX-AGENTS.md` | DONE |

---

## 8. Known Issues & Fixes

### BQ Client Type Mismatch — Zero Candidates Bug (v5.4)

**Symptom**: Walk-forward backtest completed with dates but ALL windows showed Candidates=0, Samples=0, Features=0.

**Root Cause**: `BigQueryClient` wrapper (which has no `.query()` method) was passed to `BacktestEngine` instead of the raw `google.cloud.bigquery.Client` (which does). Every `cached_prices()` call threw `AttributeError`, silently caught at `logger.debug()` level by `screen_at_date()`.

**Data Flow Trace**:
```
POST /api/backtest/run
→ bq = BigQueryClient(settings)          # wrapper object
→ engine = BacktestEngine(bq_client=bq)  # BUG: passed wrapper
  → cache.init_cache(bq_client, ...)     # stored wrapper as _bq_client
  → FOR EACH window:
    → candidate_selector.screen_at_date()
      → cache.cached_prices(ticker, ...)
        → _bq_client.query(...)          # AttributeError! wrapper has no .query()
        → exception caught at logger.debug → ticker silently skipped
      → 0 candidates → early-exit WindowResult with all zeros
```

**Why Ingestion Worked**: `POST /ingest` correctly passed `bq.client` (raw):
```python
service = DataIngestionService(bq.client, settings)  # ✅ raw client
```

**Fix Applied (3 files)**:
1. `backend/api/backtest.py` — `bq_client=bq` → `bq_client=bq.client` at both `BacktestEngine()` call sites
2. `backend/backtest/backtest_engine.py` — Defensive unwrap: `if hasattr(bq_client, 'client'): bq_client = bq_client.client`
3. `backend/backtest/candidate_selector.py` — `logger.debug` → `logger.warning` so BQ failures are visible in logs

**Lesson**: Python type annotations don't enforce types at runtime. The `BigQueryClient` wrapper was accepted without error, and `logger.debug()` in exception handlers hid the critical `AttributeError`. Always use `logger.warning()` for BQ/external service failures in screening loops.

### Backend/Frontend Schema Alignment (v5.4)

**Symptom**: Walk-Forward Windows table showed blank cells for dates, candidates, samples, features.

**Root Cause**: `generate_report()` in `analytics.py` returned fields like `sharpe_ratio`, `max_dd`, and single `date_range` strings, while frontend TypeScript interfaces expected `sharpe`, `max_drawdown`, and split `train_start`/`train_end`/`test_start`/`test_end` fields.

**Fix Applied (5 files)**:
- `backend/backtest/analytics.py` — Field renames, split date fields, added per-window metrics
- `backend/backtest/backtest_engine.py` — `WindowResult` dataclass: added `n_candidates`, `n_train_samples`, `n_features`
- `frontend/src/lib/types.ts` — Aligned `BacktestWindowResult` interface
- `frontend/src/app/backtest/page.tsx` — Updated table to render new field names
- `backend/backtest/quant_optimizer.py` — Updated to consume new field names
