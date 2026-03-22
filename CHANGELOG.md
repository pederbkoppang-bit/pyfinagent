# Changelog

All notable changes to PyFinAgent are documented here.
For architecture details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

### v5.6 — Backtest Vertical Timeline UI + Finalizing Phase Fix (March 2026)

Resolved three UX bugs in the Walk-Forward Backtest progress panel and replaced the flat progress card with a Jira-style vertical workflow timeline.

**Bug Fixes (backend)**:
1. **Window counter showed 0/8** — `_build_training_data()` called `_report_progress()` without the `window=` kwarg, so `kwargs.get("window", 0)` always returned 0. Fixed by tracking `self._current_window_id` in `__init__`, setting it at the top of `_run_window()`, and passing `window=self._current_window_id` in the throttled progress call.
2. **UI frozen at "Predicting Window 8/8"** — After `engine.run_backtest()` returned, `compute_baseline_strategies()` and `generate_report()` ran with no progress emitted (~30s gap). Fixed by emitting two `progress_cb` updates (step `"finalizing"`) in `_run_backtest_async()` — one before baseline computation, one before report generation. The `run_backtest()` method also emits a `"finalizing"` step before `cache.clear_cache()`.
3. **Elapsed timer only updated on 5s poll** — Added client-side `localElapsed` state with a 1s `setInterval` tick while running, server value overrides on each poll. Poll interval reduced from 5000 ms → 2000 ms.

**Vertical Timeline UI (page.tsx)**:
- Pipeline step definitions moved to module-level `STEP_ORDER` (8 steps) and `STEP_META` (icon + label per step, including new `finalizing` step).
- **Window rail**: Row of colored dots (emerald = done, sky-pulsing = active, outline = pending) + animated progress bar + `W/N` counter.
- **Vertical timeline**: 8 workflow steps connected by vertical pipes (emerald when done, slate when pending). Each node shows a filled emerald `CheckCircle` for done, a glowing sky-400 bubble for active, slate for pending. Active step shows `In Progress` badge + `step_detail` text + sample sub-progress bar for `building_features`.
- **Animated header**: Pulsing live dot + elapsed timer (client-side tick).
- **Cache footer**: Cache hit rate % (colour-coded emerald/amber/rose) + raw hit/miss counts.

**New `finalizing` step**: Covers baseline strategy comparison + report generation. Window rail shows all dots green when step is `"finalizing"`.

**Modified Files (3)**:
- `backend/backtest/backtest_engine.py` — Added `_current_window_id` field, sets it per window in `_run_window()`, passes it to throttled `_build_training_data` progress call; emits `"finalizing"` progress before `cache.clear_cache()`.
- `backend/api/backtest.py` — Added `import time`; emits two `"finalizing"` `progress_cb` updates for baseline computation and report generation in `_run_backtest_async()`.
- `frontend/src/app/backtest/page.tsx` — Added `CheckCircle` import; added `PipelineStepKey` type, `STEP_ORDER`, `STEP_META` module constants; added `localElapsed` state + two useEffects (client tick + server sync); poll interval 5000→2000; replaced flat status banner with full vertical timeline panel.

---

### v5.5 — Backtest Bulk Cache + Structured Progress UI (March 2026)

Fixed critical performance bottleneck where the backtest ran ~6,500 individual BQ queries per window (30+ min stuck on Window 1), and replaced the blind spinner UI with a detailed sub-step progress panel. Root cause: `cache.py` used exact `(ticker, start_date, end_date)` tuple keys — expanding walk-forward windows and per-sample label computations guaranteed cache misses on nearly every call.

**Performance Fix (cache.py — bulk preload)**:
1. **`preload_prices(tickers, start, end)`**: Single BQ query using `WHERE ticker IN UNNEST(@tickers)` loads all price data for all tickers across the full backtest date range. Stores per-ticker DataFrames in `_prices_full` dict (keyed by ticker only). `cached_prices()` now slices from preloaded data first — no BQ round-trip.
2. **`preload_fundamentals(tickers)`**: Same pattern for fundamentals. Single BQ query, grouped by ticker into `_fundamentals_full`.
3. **Cache hit/miss counters**: `_cache_stats` dict with `get_cache_stats()` accessor, surfaced in progress UI.
4. **Called at backtest start**: `run_backtest()` computes global date range (all windows + lookback + label forward) and calls both preload functions before the window loop. 2 BQ queries replace ~50,000.

**Structured Progress Reporting (backend)**:
1. **`_report_progress()` expanded**: Changed from `(message: str)` to `(step, detail, **kwargs)` — emits structured dict with `window`, `total_windows`, `step`, `step_detail`, `candidates_found`, `samples_built`, `samples_total`, `elapsed_seconds`, `cache_hits`, `cache_misses`.
2. **Sub-step calls**: Progress emitted at 7 points in `_run_window()`: screening, building_features (throttled every 200 samples), training, computing_mda, predicting, trading, plus preloading phase.
3. **API layer**: `_backtest_state["progress"]` changed from `str` to `dict`. `progress_cb` accepts dict.

**Frontend Progress Panel (page.tsx)**:
1. **Window progress bar**: Segmented bar showing "Window 2/8" with fill percentage.
2. **Step indicator**: Icon + label per step (MagnifyingGlass/Database/Brain/ChartBarHorizontal/TrendUp/ShoppingCart/CloudArrowDown).
3. **Detail line**: Sub-progress bar during `building_features` showing sample count.
4. **Elapsed time**: Formatted as "2m 34s elapsed".
5. **Cache stats**: "Cache: 1,200 hits / 5 misses" in footer row.
6. **Backward-compatible**: `BacktestProgress | string` type handles both old and new formats.

**Expected Performance Impact**:
- BQ queries per window: ~6,500 → ~10 (macro only)
- Window 1 wall time: 30+ min → ~1-2 min
- Full 8-window backtest: 4+ hours → ~10-15 min

**Modified Files (5)**:
*   `backend/backtest/cache.py` — Added `preload_prices()`, `preload_fundamentals()`, `_prices_full`, `_fundamentals_full`, `_cache_stats`, `get_cache_stats()`. Modified `cached_prices()`, `cached_fundamentals()`, `clear_cache()`
*   `backend/backtest/backtest_engine.py` — Added `import time`, preload calls in `run_backtest()`, expanded `_report_progress()` to structured dict, sub-step calls in `_run_window()` and `_build_training_data()` (throttled every 200 samples)
*   `backend/api/backtest.py` — `_backtest_state["progress"]` from `""` to `dict`, `progress_cb(msg: str)` → `progress_cb(data: dict)`
*   `frontend/src/lib/types.ts` — Added `BacktestProgress` interface, `BacktestStatus.progress` changed to `BacktestProgress | string`
*   `frontend/src/app/backtest/page.tsx` — Added 5 Phosphor icon imports, replaced spinner banner with detailed progress card (window bar, step icon, sample sub-progress, elapsed time, cache stats)

---

### v5.4 — Backtest BQ Client Type Fix + Schema Alignment (March 2026)

Fixed critical zero-candidates bug in walk-forward backtest: `BigQueryClient` wrapper (no `.query()` method) was passed to `BacktestEngine` instead of the raw `bigquery.Client`. Every `cached_prices()` call threw `AttributeError`, silently caught at `logger.debug()` level, causing ALL windows to return 0 candidates/samples/features. Also fixed backend/frontend schema mismatch that caused blank cells in the Walk-Forward Windows table. See `trading_agent.md` Section 8 for full data flow trace and root cause analysis.

**Bug Fixes (2)**:
1. **BQ client type mismatch (CRITICAL)**: `backend/api/backtest.py` passed `bq` (BigQueryClient wrapper) instead of `bq.client` (raw `bigquery.Client`) to `BacktestEngine()` at both call sites (`_run_backtest_async` and `_run_optimizer_async`). Ingestion endpoint correctly used `bq.client` — only backtest/optimizer were affected.
2. **Backend/frontend schema mismatch**: `generate_report()` in `analytics.py` returned `sharpe_ratio`/`max_dd`/`date_range` but frontend expected `sharpe`/`max_drawdown`/`train_start`/`test_start` etc. Walk-Forward Windows table rendered blank cells.

**Defensive Improvements (2)**:
1. `backend/backtest/backtest_engine.py` — Unwrap guard: `if hasattr(bq_client, 'client'): bq_client = bq_client.client` before `cache.init_cache()`
2. `backend/backtest/candidate_selector.py` — `logger.debug` → `logger.warning` for BQ query failures in screening loop

**Modified Files (8)**:
*   `backend/api/backtest.py` — `bq_client=bq` → `bq_client=bq.client` (2 call sites)
*   `backend/backtest/backtest_engine.py` — Defensive unwrap guard + `n_candidates`/`n_train_samples`/`n_features` in `WindowResult`
*   `backend/backtest/candidate_selector.py` — `logger.debug` → `logger.warning`
*   `backend/backtest/analytics.py` — Field renames (`sharpe_ratio` → `sharpe`, `max_dd` → `max_drawdown`), split date fields, per-window metrics
*   `backend/backtest/quant_optimizer.py` — Updated to consume new field names
*   `frontend/src/lib/types.ts` — Aligned `BacktestWindowResult` interface fields
*   `frontend/src/app/backtest/page.tsx` — Updated table to render new field names
*   `trading_agent.md` — Phase 5D checklist + Section 8 Known Issues & Fixes

### v5.3 — ML Training Pipeline Fixes + Backtest UX (July 2026)

Fixed 3 dead/degraded ML features in the walk-forward backtest pipeline, doubled training sample density, added auto-table creation, and improved the backtest page UX with ingestion feedback and cost visibility.

**Data Quality Fixes (3)**:
1. **`revenue_growth_yoy` was always NULL (CRITICAL)**: `cached_fundamentals()` in `cache.py` returned only 1 quarter (`LIMIT 1`). Changed to `LIMIT 5` to return 5 most recent quarters. Added `_compute_revenue_growth_yoy()` in `historical_data.py` to compare current quarter revenue vs same quarter 4 periods ago (Q vs Q-4). Returns `None` when <5 quarters available.
2. **`dividend_yield` never ingested**: `ingest_fundamentals()` in `data_ingestion.py` did not extract dividends from cash flow statements. Added `_compute_dividends_per_share()` method: extracts `Cash Dividends Paid` from quarterly cash flow, divides by shares outstanding. Added `dividends_per_share FLOAT64` column to `historical_fundamentals` BQ schema.
3. **Monthly sampling — low training density**: `_build_training_data()` in `backtest_engine.py` sampled at monthly intervals (`pd.DateOffset(months=1)`), producing only ~12 samples/ticker/window for 36 features. Changed to biweekly (`pd.DateOffset(weeks=2)`), roughly doubling training set to ~26 samples/ticker/window.

**Auto-Table Creation**:
*   Added `_ensure_tables_exist()` to `DataIngestionService`: imports schemas from `migrate_backtest_data.py`, creates missing tables before ingestion. No more silent failures when `migrate_backtest_data.py` hasn't been run.

**Frontend UX Improvements (Backtest Page)**:
*   **Ingestion banner**: `handleIngest()` now captures POST response and displays an emerald success banner (with row counts) or rose error banner. Dismissable by user.
*   **Cost info**: Inline text below row count metrics: "Data: yfinance + FRED (free) · BQ storage <$0.05 · Backtest: ML only, $0 LLM cost"
*   **Button tooltips**: `title` attributes on Ingest Data and Run Backtest buttons explaining data sources, cost, and duration.

**Modified Files (7)**:
*   `backend/backtest/cache.py` — `cached_fundamentals()`: returns `list[dict]` (up to 5 quarters) instead of single `dict`
*   `backend/backtest/historical_data.py` — `build_feature_vector()`: consumes list of fundamentals, computes `revenue_growth_yoy` from 5-quarter history. New `_compute_revenue_growth_yoy()` static method
*   `backend/backtest/data_ingestion.py` — Added `_ensure_tables_exist()`, `_compute_dividends_per_share()`, `dividends_per_share` field in ingestion rows
*   `backend/backtest/backtest_engine.py` — `_build_training_data()`: biweekly sampling
*   `migrate_backtest_data.py` — Added `dividends_per_share FLOAT64` to `historical_fundamentals` schema (14 cols)
*   `frontend/src/app/backtest/page.tsx` — Ingestion result banner, cost info section, button tooltips
*   `AGENTS.md` — This entry + schema docs updated

### v5.2 — BQ Schema Parity for Autoresearch Loops (July 2026)

Comprehensive BQ audit identified 20 missing columns required for the Karpathy autoresearch optimization loops (QuantOpt → MDA → MetaCoordinator → SkillOpt) to work end-to-end. Fixed critical Celery task bug where only 6 of 88 params were passed to `save_report()`. BQ schema expanded from 68 → 88 columns.

**Root Cause**: The `FEATURE_TO_AGENT` bridge in `MetaCoordinator` maps 27 MDA features to responsible agent skill files, but 5 of those features (`consumer_sentiment`, `revenue_growth_yoy`, `quality_score`, `momentum_6m`, `rsi_14`) had no corresponding BQ columns — they existed only in the backtest engine's `_NUMERIC_FEATURES` vector. Additionally, 15 enrichment/risk/debate/cost columns were missing parity coverage.

**Bug Fixes (3)**:
1. **Celery task `save_report()` (CRITICAL)**: `backend/tasks/analysis.py` was passing only 6 of 88 params (ticker, company_name, final_score, recommendation, summary, full_report). All 82+ ML-training columns were NULL when Celery workers ran. Fixed: full extraction logic mirrored from `api/analysis.py` (Phases 1–11, 85 kwargs).
2. **FEATURE_TO_AGENT bridge columns missing**: 5 features in `meta_coordinator.py` FEATURE_TO_AGENT had no BQ columns. Added as first-class FLOAT64 columns.
3. **Enrichment signal parity gap**: 15 enrichment tool signals, risk debate outputs, debate meta-features, and cost metrics had no BQ columns. Added for complete ML training coverage.

**BQ Schema Expansion (20 new columns across 7 categories)**:
*   **Autoresearch Bridge (+5)**: `consumer_sentiment`, `revenue_growth_yoy`, `quality_score`, `momentum_6m`, `rsi_14`
*   **Enrichment Signal Parity (+8)**: `alt_data_signal`, `alt_data_momentum_pct`, `anomaly_signal`, `monte_carlo_signal`, `quant_model_signal`, `quant_model_score`, `social_sentiment_velocity`, `nlp_sentiment_confidence`
*   **Risk Assessment Parity (+4)**: `risk_level`, `recommended_position_pct`, `neutral_analyst_confidence`, `risk_debate_rounds_count`
*   **Debate Parity (+2)**: `groupthink_flag`, `da_confidence_adjustment`
*   **Cost Parity (+1)**: `grounded_calls`

**Modified Files (4)**:
*   `migrate_bq_schema.py` — Added 20 columns in Phase 11 section
*   `backend/db/bigquery_client.py` — Added 20 new params + row dict entries to `save_report()`
*   `backend/api/analysis.py` — Added Phase 11 extraction variables + 20 new kwargs to `bq.save_report()` call
*   `backend/tasks/analysis.py` — Complete rewrite: 6-param call → 85-kwarg call with full Phase 1–11 extraction logic

### v5.1 — Phase 5B/5C: ML→Live Bridge + SkillOpt Iteration (July 2026)

Bridged the walk-forward backtest ML system into the live analysis pipeline via a new 12th enrichment signal (Quant Model), and wired the MetaCoordinator to orchestrate the three optimization loops with MDA→Agent targeting and proxy validation.

**Phase 5B — ML→Live Bridge (5 steps)**:

*   **New data tool**: `backend/tools/quant_model.py` — 12th enrichment signal. Reads MDA feature importance from backtest cache, builds 17 live features from yfinance, computes MDA-weighted factor score, classifies signal (STRONG_BULLISH→STRONG_BEARISH). Falls back to equal weights when no backtest MDA exists.
*   **MDA cache**: `backend/backtest/backtest_engine.py` — Added `_MDA_CACHE_PATH` (JSON), `get_latest_mda()`, `_save_mda_cache()`. Written after each backtest run.
*   **Orchestrator wiring**: `backend/agents/orchestrator.py` — 12th signal in Step 6 gather (11→12), new `fetch_quant_model()` + `run_quant_model_agent()` methods, wired into enrichment_raw, retry_funcs, _agent_list, enrichment_for_debate.
*   **Skill file**: `backend/agents/skills/quant_model_agent.md` — Factor decomposition, contradiction detection, MDA source awareness, extreme value flagging.
*   **Prompt + signals**: `backend/config/prompts.py` — `get_quant_model_prompt()`. `backend/api/signals.py` — 12th gather slot + `/{ticker}/quant-model` endpoint.
*   **Frontend**: `types.ts` (`quant_model: SignalSummary`), `icons.ts` (`ChartPieSlice as SignalQuantModel`), `SignalCards.tsx` (12th signal card).

**Phase 5C — SkillOpt Iteration (3 steps)**:

*   **MetaCoordinator wired to autonomous_loop**: `backend/services/autonomous_loop.py` — New Step 10 at end of daily cycle: `gather_health()` → `decide()` → logs coordinator decision (action, reason, target agents, health snapshot) to cycle summary. Module-level `_coordinator` instance exposed via `get_coordinator()`.
*   **MDA→Agent bridge live targeting**: `backend/agents/skill_optimizer.py` — `_run_one_iteration()` and `run_loop()` accept `target_agents: list[str]` parameter. When MetaCoordinator provides MDA-targeted agents, overrides the weakest-agent heuristic with cycle-through of targeted agents. `backend/backtest/quant_optimizer.py` — `run_loop()` accepts `on_mda_update` callback; invoked after each kept experiment with fresh MDA importances. `backend/api/backtest.py` — Wires `coordinator.update_mda_features` as callback. `backend/api/skills.py` — Reads MDA targets from coordinator before starting SkillOpt loop.
*   **Proxy validation**: `backend/agents/skill_optimizer.py` — New `_run_proxy_validation()` method delegates to `MetaCoordinator.run_proxy_validation()`. Called on pending experiments (delta==0) for fast quant-only feedback instead of waiting days for BQ outcome data.

**Updated registries**:
*   `meta_coordinator.py` — Added `quality_score`, `momentum_6m`, `rsi_14` → `quant_model_agent` in FEATURE_TO_AGENT (27 entries total)
*   `skill_optimizer.py` — Added `quant_model_agent` to OPTIMIZABLE_AGENTS (26 agents total)

### v5.0 — Phase 5A: Multi-Strategy Backtest Engine (July 2026)

Expanded the walk-forward backtesting engine from a single Triple Barrier strategy to 5 research-backed strategies, added 6 new ML features, configurable screener weights, feature drift detection, model staleness tracking, and auto-ingestion at backtest start. Feature vector expanded from ~43 → ~49 features. QuantStrategyOptimizer now rotates strategies as a categorical hyperparameter.

**Design Principles**:
- **Strategy Registry**: Dispatch pattern maps strategy name → label method, enabling QuantOptimizer to explore strategy space
- **Feature drift detection**: Log top-5 MDA changes between optimizer iterations to catch distribution shifts early
- **Model staleness**: Timestamp-based guard warns when trained model is >7 days old
- **Auto-ingest**: First backtest run auto-checks BQ tables and triggers ingestion if empty
- **No BQ for testing**: Full mock-test suite exercises all 5 strategies via monkey-patched cache

**Strategy Registry (5 strategies in `backtest_engine.py`)**:

| Strategy | Label Method | Research Basis | Key Signals |
|----------|-------------|----------------|-------------|
| `triple_barrier` | `_compute_triple_barrier_label()` | López de Prado Ch. 3 | TP/SL/time barriers on price path |
| `quality_momentum` | `_compute_quality_momentum_label()` | Asness et al. 2019 | 6M momentum × quality_score |
| `mean_reversion` | `_compute_mean_reversion_label()` | Lo & MacKinlay 1990 | SMA50 deviation + RSI oversold/overbought |
| `factor_model` | `_compute_factor_label()` | Fama-French 5-factor | Composite: value + momentum + low-vol + quality + yield |
| `meta_label` | `_compute_triple_barrier_label()` | López de Prado Ch. 3 | Same labels, secondary model layer (future) |

**New ML Features (6 in `historical_data.py`)**:

| Feature | Computation | Used By |
|---------|------------|--------|
| `volume_ratio_20d` | current_volume / 20d_avg | All strategies (liquidity signal) |
| `pb_ratio` | price / book_per_share | factor_model (value factor) |
| `fcf_yield` | annualized OCF / market_cap | factor_model (yield factor) |
| `dividend_yield` | dividends_per_share / price (from BQ fundamentals) | factor_model (yield factor) |
| `quality_score` | ROE × profit_margin × (1 − norm_D/E) | quality_momentum |
| `revenue_growth_yoy` | Q vs Q-4 revenue growth (from 5-quarter cache) | quality_momentum |

**Modified Backend Files**:

*   `backend/backtest/backtest_engine.py` — Added `STRATEGY_REGISTRY` (5 strategies), `strategy` param to `__init__`, `_compute_label()` dispatcher, 3 new label methods (`_compute_quality_momentum_label`, `_compute_mean_reversion_label`, `_compute_factor_label`), `_auto_ingest_if_needed()`, `model_trained_at` timestamp in `_train_model()`, configurable `scoring_weights` wired to candidate_selector, `_NUMERIC_FEATURES` expanded with 6 new features
*   `backend/backtest/historical_data.py` — Added 6 new feature computations: `volume_ratio_20d`, `pb_ratio`, `fcf_yield`, `dividend_yield`, `quality_score`, `revenue_growth_yoy`
*   `backend/backtest/candidate_selector.py` — Added `scoring_weights: dict | None = None` parameter to `screen_at_date()`, passes through to `_rank_candidates()`
*   `backend/backtest/quant_optimizer.py` — Added `AVAILABLE_STRATEGIES` list, `_CATEGORICAL_PARAMS` dict for strategy rotation, `_prev_top5_mda` for drift tracking. Updated `_propose_random()` to handle categorical `strategy` param (20% chance). Updated `_apply_params_to_engine()` to set `engine.strategy`. Updated `_log_experiment()` with `top5_mda` column. Added `_extract_top5_mda()`, `_detect_feature_drift()`, `_check_model_staleness()` helpers. Updated `run_loop()` with baseline MDA extraction, staleness checks every 10 iterations, drift detection on keep
*   `trading_agent.md` — Added Section 6 (Strategy Research) and Section 7 (Phase 5 Implementation Plan)

**New Test File**:
*   `t_backtest_mock.py` — Mock-test script: 20 synthetic tickers × 2y GBM prices, monkey-patched cache (no BQ), exercises all 5 strategies through full walk-forward pipeline, validates feature vectors, labels, candidate selection, optimizer helpers, and model staleness tracking. 6 test functions, all passing

**Updated Feature Vector (~49 features)**:
| Category | Features |
|----------|----------|
| Price & Returns | `price_at_analysis`, `momentum_1m`/`3m`/`6m`/`12m`, `annualized_volatility` |
| Technical | `rsi_14`, `sma_50_distance`, `sma_200_distance`, `volume_ratio_20d` |
| Monte Carlo | `var_95_6m`, `var_99_6m`, `expected_shortfall_6m`, `prob_positive_6m` |
| Anomaly | `anomaly_count` |
| Fundamentals | `pe_ratio`, `pb_ratio`, `debt_equity`, `roe`, `profit_margin`, `market_cap`, `total_revenue`, `net_income`, `total_debt`, `total_equity`, `total_assets`, `fcf_yield`, `dividend_yield`, `quality_score`, `revenue_growth_yoy` |
| Macro | `fed_funds_rate`, `cpi_yoy`, `unemployment_rate`, `yield_curve_spread`, `consumer_sentiment`, `treasury_10y` |
| Advanced | `amihud_illiquidity` |
| Fractionally Differenced | `frac_diff_price`, `frac_diff_market_cap`, `frac_diff_revenue`, `frac_diff_debt`, `frac_diff_equity` |

**QuantOptimizer Enhancements**:
- 16 tunable parameters (15 numeric + 1 categorical `strategy`)
- `_TSV_HEADER` now includes `top5_mda` column
- Feature drift: top-5 MDA features compared between iterations; WARNING logged when set changes
- Model staleness: checked every 10 iterations; WARNING if model >7 days old
- Strategy rotation: `_propose_random()` has ~6% chance (1/16 params) of proposing a strategy change

**Research Alignment**:

| Research Source | Implementation |
|----------------|----------------|
| **Asness et al. (2019)** — Quality minus Junk | `_compute_quality_momentum_label()`: quality_score × 6M momentum composite |
| **Lo & MacKinlay (1990)** — Mean Reversion | `_compute_mean_reversion_label()`: SMA50 deviation + RSI reversion bands |
| **Fama-French 5-factor** | `_compute_factor_label()`: value + momentum + low-vol + quality + yield composite |
| **López de Prado Ch. 3** | `_compute_triple_barrier_label()`: unchanged, still default strategy |
| **Karpathy autoresearch** | QuantOptimizer now explores strategy space as categorical param |

Unified P&L and portfolio metrics into a single `PerformanceSkill` module, eliminating formula duplication across 5 files. Created a `MetaCoordinator` to sequence the three optimization loops (QuantOpt, SkillOpt, PerfOpt) using portfolio health signals and MDA feature importance. Added `trading_agent.md` as a living memory file documenting the hybrid Karpathy/autoresearch approach, research foundations, and optimization architecture.

**Design Principles**:
- **Single source of truth**: All P&L, Sharpe, benchmark, and alpha formulas live in `perf_metrics.py`
- **Scalar metric**: One unified metric for all optimizers: `risk_adjusted_return × (1 − tx_cost_drag)`
- **MDA→Agent bridge**: QuantOpt's feature importance maps to responsible agents for targeted SkillOpt
- **Proxy validation**: Single-window backtest for fast SkillOpt feedback (vs waiting days for outcomes)
- **No circular imports**: `backtest_engine.py` delegates to `analytics.py` via lazy imports

**New Backend Module (1 in `backend/services/`)**:
*   `perf_metrics.py` — `PerformanceSkill`: canonical P&L, Sharpe, benchmark, alpha, turnover, and scalar metric. Functions: `compute_position_pnl()`, `compute_return_pct()`, `compute_portfolio_pnl()`, `compute_alpha()`, `compute_sharpe_from_snapshots()` (NAV→daily returns→canonical Sharpe), `compute_benchmark_return()` (geometric compounding), `beat_benchmark()`, `compute_turnover_ratio()`, `compute_tx_cost_drag()` (capped at 30%), `get_scalar_metric()` (THE unified metric), `get_scalar_metric_from_bq()`. `ScalarMetricInputs` dataclass for structured input.

**New Backend Module (1 in `backend/agents/`)**:
*   `meta_coordinator.py` — `MetaCoordinator`: cross-loop sequencing. `FEATURE_TO_AGENT` dict (20+ MDA features → agent skill files), `PortfolioHealth` dataclass (sharpe, accuracy, latency, data quality, days since last optimizations), `CoordinatorDecision` dataclass (action, reason, target_agents, priority). Methods: `decide()` (priority: latency→quant→skill→idle), `update_mda_features()`, `_get_mda_target_agents()` (MDA→Agent bridge — maps top 5 features to responsible agents), `gather_health()` (collects signals from BQ, PerfTracker, snapshots), `run_proxy_validation()` (single-window quant-only backtest for fast SkillOpt validation), `status()`.

**New Root File (1)**:
*   `trading_agent.md` — Living memory file for autonomous trading optimization. 5 sections: Agent Identity (mission, constraints, modifiable/fixed boundaries), Research Summary (43-feature vector, MDA, baselines, 10 research sources), Implementation Progress (phase checklist), Performance Skill API (function signatures), Iteration Loop (Karpathy-adapted hybrid rules, three-loop table, MetaCoordinator sequencing, MDA→Agent bridge).

**P&L Deduplication (5 files modified)**:
*   `backend/api/paper_trading.py` — Replaced 18-line inline Sharpe (no risk-free deduction, inflated by ~0.15-0.25) with 2-line `compute_sharpe_from_snapshots()` + `compute_alpha()` delegation
*   `backend/backtest/backtest_engine.py` — `_sharpe()` and `_max_drawdown()` now delegate to `analytics.py` via lazy imports (avoids circular dependency with `analytics.py` importing `BacktestResult`)
*   `backend/services/outcome_tracker.py` — `return_pct` uses `compute_return_pct()`, benchmark uses geometric `beat_benchmark()` (was arithmetic `10%/365 × days`), renamed variable to `beat_benchmark_flag` to avoid name collision
*   `backend/api/portfolio.py` — `_enrich_position()` uses `compute_position_pnl()` instead of inline formula
*   `backend/agents/skill_optimizer.py` — `compute_metric()` delegates to `perf_metrics.get_scalar_metric_from_bq()` (now includes transaction cost penalty)

**Scalar Metric Formula**:
```
scalar = risk_adjusted_return × (1 − tx_cost_drag)

where:
    risk_adjusted_return = avg(return_pct) × beat_benchmark_rate
    tx_cost_drag = min(0.3, turnover_ratio × tx_cost_pct)
```
Extends Karpathy's single-metric approach with transaction cost penalty that prevents discovery of "churn alpha" — strategies that look profitable on paper but lose to friction.

**MetaCoordinator Three-Loop Sequencing**:
```
MetaCoordinator.decide(health) → priority-based:
    Priority 3: p95 latency > 500ms      → PerfOpt  (fast, no cost)
    Priority 2: Sharpe < 0.5             → QuantOpt (fast, no LLM cost)
    Priority 1: Agent accuracy < 55%     → SkillOpt (slow, uses outcomes)
    Priority 0: All within targets       → idle

MDA→Agent Bridge (unique research contribution):
    QuantOpt backtest → MDA feature importance
    → FEATURE_TO_AGENT map (20+ features → agent skill files)
    → SkillOpt targets the agents responsible for top features
```

**Three-Loop Architecture (updated from v4.0)**:
```
Fast Loop: QuantStrategyOptimizer (minutes/cycle)
├── Propose parameter modification (random or LLM-guided)
├── Run full walk-forward backtest
├── Evaluate: Sharpe, DSR, alpha, hit rate
├── Keep if DSR ≥ 0.95 AND metric improves
├── Extract MDA feature importance → MetaCoordinator
└── Log to quant_results.tsv

Slow Loop: SkillOptimizer (days/cycle)
├── MetaCoordinator targets top agents via MDA→Agent bridge
├── Propose LLM-generated prompt modification
├── Run proxy validation (single backtest window, quant-only)
├── Evaluate via outcome_tracking table
└── Log to skill_results.tsv

Fast Loop: PerfOptimizer (minutes/cycle)
├── Propose TTL modification (random ±20% perturbation)
├── Measure p95 latency over 60s window
├── Keep if ≥5% improvement
└── Log to perf_results.tsv

Coordinator: MetaCoordinator (on-demand)
├── gather_health() — Sharpe from snapshots, accuracy from BQ, latency from PerfTracker
├── decide(health) — priority routing to the right optimizer
└── update_mda_features() — receives MDA from QuantOpt after each backtest
```

**Research Alignment**:

| Research Source | Implementation |
|----------------|----------------|
| **Karpathy autoresearch** | Single scalar metric, keep/discard loop, LOOP FOREVER — all 3 optimizers follow this |
| **FinRL three-layer** | Data→Agent→Analytics wired as MetaCoordinator feedback loop |
| **BlackRock regime-aware** | Market conditions (Sharpe, accuracy, latency) determine optimizer priority |
| **Lopez-Lira 2023** | Proxy validation is quant-only (no LLM contamination for historical data) |
| **López de Prado Ch. 8** | MDA→Agent bridge maps feature importance to responsible agents |
| **TradingAgents** | Multi-agent debate insights flow through debate features in MDA |
| **Goldman Sachs 127-dim** | Risk features (VaR, volatility, anomaly) trackable through FEATURE_TO_AGENT |

### v4.2 — Settings Sub-Navigation + Performance Dashboard (March 2026)

Restructured the Settings page from a flat 6-card layout to a 3-tab sub-navigation architecture. Added a new Performance tab exposing the v4.1 backend `/api/perf/*` endpoints in the frontend. Migrated all remaining emoji characters to Phosphor Icons per UX-AGENTS.md conventions. Fixed pre-existing reports page build failure.

**Settings Page Restructure**:
*   3-tab pill-style sub-navigation: **Models & Analysis** (Analysis Mode + Debate Depth + Model Config) | **Cost & Weights** (Cost Estimator + Cost Controls + Pillar Weights) | **Performance** (Cache Health + TTL Optimizer + API Latency)
*   Save button hidden on Performance tab (read-only monitoring)
*   Tab state managed via `useState<"models" | "cost" | "performance">`

**New Performance Tab (4 BentoCards)**:
*   **Cache Health** — Hit/miss counts, hit rate %, entry count, clear cache button with confirmation feedback
*   **TTL Optimizer** — Status indicator (running/idle), iteration count, kept/discarded experiments, start/stop controls
*   **Optimization Progress Chart** — Autoresearch-style scatter + step-line chart (inspired by karpathy/autoresearch): green dots for kept experiments, gray dots for discarded, green step line for running best p95 latency. Hover tooltip shows endpoint, TTL change, p95 change. Click expands full changelog detail panel below chart
*   **API Latency** — Overall p50/p95/p99 summary cards, per-endpoint latency table sorted by p95 descending, auto-refresh on tab switch

**New Component (1)**:
*   `PerfProgressChart.tsx` — Recharts `ComposedChart` with `Scatter` (kept/discarded dots) + `Line` (running best, stepAfter) + `LabelList` text annotations on kept dots (karpathy/autoresearch `analysis.ipynb` style). Custom `TooltipProps` with click-to-select. Detail panel shows endpoint, timestamp, TTL before→after, p95 before→after (color-coded improvement/regression), hit rate. Data derived from `getPerfOptimizerExperiments()` API

**New Frontend Types (5 interfaces in `types.ts`)**:
*   `EndpointLatency` — `endpoint`, `count`, `p50`, `p95`, `p99`
*   `PerfSummary` — `endpoints: EndpointLatency[]`, `overall: { p50, p95, p99 }`
*   `CacheStats` — `entries`, `hits`, `misses`, `hit_rate`
*   `PerfOptimizerStatus` — `running`, `iterations`, `kept`, `discarded`
*   `PerfExperiment` — `iteration`, `endpoint`, `old_ttl`, `new_ttl`, `p95_before`, `p95_after`, `decision`, `timestamp`

**New Frontend API Functions (8 in `api.ts`)**:
*   `getPerfSummary()`, `getCacheStats()`, `clearCache()`, `startPerfOptimizer()`, `stopPerfOptimizer()`, `getPerfOptimizerStatus()`, `getPerfOptimizerExperiments()`

**Emoji → Phosphor Icon Migration (2 files)**:
*   `frontend/src/lib/icons.ts` — 10 new Settings-prefixed icon aliases: `SettingsMode` (Lightning), `SettingsDebate` (ChatTeardropDots), `SettingsModel` (Brain), `SettingsCostControls` (ShieldCheck), `SettingsEstimator` (CurrencyDollar), `SettingsPillars` (ChartBar), `SettingsCache` (Database), `SettingsOptimizer` (GearSix), `SettingsLatency` (Timer), `SettingsRefresh` (ArrowClockwise)
*   `frontend/src/app/settings/page.tsx` — All 14 emoji/Unicode characters (⚡💰🧠🛡️🗣️📊🗄️⚙️⚠↻✓) replaced with Phosphor Icon components per UX conventions

**Bug Fix**:
*   `frontend/src/app/reports/page.tsx` — Wrapped `useSearchParams()` in `<Suspense>` boundary to fix Next.js static build failure

### v4.1 — API Performance Module + Autoresearch TTL Optimizer (March 2026)

Thread-safe in-memory response cache, per-endpoint latency tracking with percentile analytics, and an autoresearch-style TTL optimizer loop. Frontend fixes: parallelized sequential fetches, lightweight status-only polling, and skeleton loading states. Zero external dependencies added.

**New Backend Modules (3 in `backend/services/`)**:
*   `api_cache.py` — `APICache` class: thread-safe TTL cache with `CacheEntry` dataclass (value, expires_at, created_at, hits). Methods: `get()`, `set()`, `invalidate()` (glob pattern), `stats()`, `clear()`. Lazy eviction on read. Module-level singleton + `ENDPOINT_TTLS` dict with 14 endpoint-specific TTL configs (10s–3600s). Uses `time.monotonic()` for clock-immune TTLs.
*   `perf_tracker.py` — `PerfTracker` class: thread-safe per-endpoint latency recording with max 10K entries (FIFO eviction). Methods: `record()`, `summarize()` (p50/p95/p99 percentiles), `get_slow_endpoints()`, `export_tsv()`, `clear()`. Custom `_percentile()` implementation. Module-level singleton.
*   `perf_optimizer.py` — `PerfOptimizer` class: autoresearch-style optimization loop that tunes API cache TTL values. Proposes random ±20% perturbation per endpoint, measures p95 latency over 60s window, keeps if ≥5% improvement. `think_harder()` doubles TTL after 5 consecutive discards. TSV logging to `backend/services/experiments/perf_results.tsv`.

**New API Route (1)**:
*   `backend/api/performance_api.py` — 8 endpoints: `GET /api/perf/summary` (latency percentiles), `GET /api/perf/slow` (slow endpoints above threshold), `GET /api/perf/cache` (cache hit/miss stats), `POST /api/perf/cache/clear` (flush cache), `POST /api/perf/optimize` (start TTL optimizer), `POST /api/perf/optimize/stop` (stop optimizer), `GET /api/perf/optimize/status` (optimizer state), `GET /api/perf/optimize/experiments` (experiment history).

**Cache Wiring (4 API files modified)**:
*   `backend/api/reports.py` — Cached 4 endpoints (list, cost-summary, cost-history, ticker report). **Fixed double BQ query** in `get_latest_cost_summary()`: replaced two-call pattern with single `bq.get_latest_report_json()`.
*   `backend/api/paper_trading.py` — Cached 5 GET endpoints (status/portfolio/trades/snapshots/performance). Cache invalidation on `stop` and `run-now` write endpoints.
*   `backend/api/settings_api.py` — Cached `GET /` (300s) and `GET /models/available` (3600s). Invalidation on `PUT /`.
*   `backend/api/backtest.py` — Cached `GET /optimize/experiments` (10s) and `GET /optimize/best` (30s).

**Latency Tracking Middleware**:
*   `backend/main.py` — Wraps every request with `time.perf_counter()` timing, records to `PerfTracker` singleton, adds `X-Response-Time` header. Performance router registered.

**New BQ Method**:
*   `backend/db/bigquery_client.py` — `get_latest_report_json()`: single query replacing two-call pattern for cost summary.

**Frontend Fixes (4 files modified)**:
*   `frontend/src/app/backtest/page.tsx` — Parallelized sequential results/experiments/best fetches into `Promise.all()`. Created `refreshStatus()` for lightweight polling (status endpoints only). Full `refresh()` triggers on status transition to completed.
*   `frontend/src/app/paper-trading/page.tsx` — `handleRunNow()` polling now hits lightweight `getPaperTradingStatus()` only, with full `refresh()` after loop completes.
*   `frontend/src/app/reports/page.tsx` — Parallelized sequential report comparison fetches (`for...of await` → `Promise.all()`).
*   `frontend/src/app/settings/page.tsx` — Loading state uses `PageSkeleton` component.

**New Frontend Component (1)**:
*   `frontend/src/components/Skeleton.tsx` — Reusable loading skeleton components: `SkeletonPulse` (atomic), `SkeletonCard` (card-sized), `SkeletonGrid` (N-card grid), `PageSkeleton` (full page: metric grid + content area). Used in backtest, paper-trading, and settings pages.

**Performance Impact**:
| Optimization | Before | After | Improvement |
|-------------|--------|-------|-------------|
| Reports cost-summary | 2 BQ queries | 1 BQ query | ~50% latency reduction |
| Backtest data fetches | 3 sequential awaits | 1 `Promise.all()` | ~60% faster page load |
| Report comparison (5 tickers) | 5 sequential fetches | 5 parallel fetches | ~80% faster |
| Backtest polling (while running) | Full refresh (6 endpoints) | Status only (2 endpoints) | ~67% fewer requests |
| Paper trading polling | Full refresh (5 endpoints) | Status only (1 endpoint) | ~80% fewer requests |
| Cached endpoint (hit) | Full BQ/compute | In-memory return | ~95% latency reduction |

### v4.0 — Walk-Forward Backtesting Engine (March 2026)

Research-driven walk-forward backtesting system with Triple Barrier labeling, fractional differentiation, Deflated Sharpe Ratio guard, and an autoresearch-style quant strategy optimizer. Implements findings from López de Prado (*Advances in Financial Machine Learning*), TradingAgents (arXiv:2412.20138), FinRL (arXiv:2011.09607), and Lopez-Lira & Tang (arXiv:2304.07619). Two-regime architecture: quant-only ($0 LLM cost) for historical backtests, full 20-agent pipeline for live forward tests. Two-loop optimization: fast QuantStrategyOptimizer (minutes/cycle, quant params) + slow SkillOptimizer (days/cycle, LLM prompts), bridged by MDA feature importance.

**Design Principles**:
- **No future leakage**: Walk-forward expanding windows with 5-day embargo between train/test
- **No LLM contamination**: Historical regime uses only quant features + GradientBoosting (LLMs "know" historical outcomes)
- **Download once, replay forever**: BigQuery stores all historical data permanently (FinRL pattern)
- **Backtest overfitting guard**: Deflated Sharpe Ratio (Bailey & López de Prado 2014) penalizes multiple testing

**New Backend Modules (8 in `backend/backtest/`)**:
*   `data_ingestion.py` — `DataIngestionService`: bulk ingest yfinance OHLCV (batches of 50), quarterly financials, FRED 7-series macro into 3 BQ tables. `run_full_ingestion()`, `get_ingestion_status()`
*   `cache.py` — Module-level BQ query cache with `init_cache()`, `cached_prices()`, `cached_fundamentals()`, `cached_macro()`. Prevents redundant BQ reads during walk-forward windows
*   `historical_data.py` — `HistoricalDataProvider`: builds ~49-feature vectors at any historical cutoff date. Includes `fractional_diff(series, d=0.4)` (López de Prado Ch. 5), `compute_turbulence_index()` (Mahalanobis distance), `_compute_amihud_illiquidity()`, Monte Carlo VaR, RSI, anomaly count, `volume_ratio_20d`, `pb_ratio`, `fcf_yield`, `dividend_yield`, `quality_score`, `revenue_growth_yoy`
*   `candidate_selector.py` — `CandidateSelector`: S&P 500 screening at historical dates using composite score (momentum 40%, RSI 20%, volatility 20%, SMA distance 20%). 50-ticker fallback list for resilience
*   `walk_forward.py` — `WalkForwardScheduler`: generates expanding walk-forward windows with configurable train/test periods and embargo days. `WalkForwardWindow` dataclass
*   `backtest_engine.py` — `BacktestEngine`: central orchestrator. `run_backtest()` → per-window: screen candidates → build features → Triple Barrier labels (Ch. 3) → sample weights via average uniqueness (Ch. 4) → train `GradientBoostingClassifier(n_estimators=200, max_depth=4, min_samples_leaf=20)` → MDI + MDA feature importance (Ch. 8) → predict & trade. 31 numeric features, 5 non-stationary features get fractional differentiation
*   `backtest_trader.py` — `BacktestTrader`: in-memory portfolio simulator with inverse-volatility position sizing (target_vol=15%), probability-weighted allocation, transaction costs, sell-first-then-buy execution
*   `analytics.py` — `compute_sharpe()`, `compute_deflated_sharpe()` (Bailey & López de Prado 2014), `compute_max_drawdown()`, `compute_alpha()`, `compute_hit_rate()`, `compute_information_ratio()`, `compute_baseline_strategies()` (SPY + equal-weight + momentum), `generate_report()`

**New Backend Module (1 in `backend/backtest/`)**:
*   `quant_optimizer.py` — `QuantStrategyOptimizer`: autoresearch-style optimization loop. Proposes parameter modifications (random ±15% perturbation or LLM-guided via Gemini Flash), evaluates via full backtest, keeps improvements with DSR ≥ 0.95 guard. 15 tunable parameters with bounds. Logs to `quant_results.tsv` (8-column TSV). `think_harder()` widens exploration after 5 consecutive discards

**New API Route (1)**:
*   `backend/api/backtest.py` — 11 endpoints: `POST /run` (async backtest), `GET /status`, `GET /results`, `GET /results/{window_id}`, `POST /ingest` (BQ data ingestion), `GET /ingest/status`, `POST /optimize` (quant optimizer), `POST /optimize/stop`, `GET /optimize/status`, `GET /optimize/experiments`, `GET /optimize/best`

**BQ Schema (3 new tables)**:
*   `historical_prices` (8 cols) — OHLCV price history from yfinance
*   `historical_fundamentals` (14 cols) — Quarterly financials from yfinance `.quarterly_financials` + `.quarterly_balance_sheet` + `.quarterly_cashflow`
*   `historical_macro` (4 cols) — FRED 7-series macro indicators
*   Managed by `migrate_backtest_data.py` (idempotent)

**Modified Backend Files**:
*   `backend/config/settings.py` — 14 new backtest settings: `backtest_start_date` ("2023-01-01"), `backtest_end_date` ("2025-12-31"), `backtest_train_window_months` (12), `backtest_test_window_months` (3), `backtest_embargo_days` (5), `backtest_holding_days` (90), `backtest_tp_pct` (10.0), `backtest_sl_pct` (10.0), `backtest_frac_diff_d` (0.4), `backtest_target_vol` (0.15), `backtest_top_n_candidates` (50), `backtest_starting_capital` (100000.0), `backtest_max_positions` (20), `backtest_transaction_cost_pct` (0.1)
*   `backend/main.py` — Backtest router registered
*   `backend/requirements.txt` — Added `scikit-learn>=1.4.0`, `scipy>=1.12.0`

**Research Alignment**:

| Research Source | Implementation |
|----------------|----------------|
| **López de Prado** — Triple Barrier (Ch. 3) | `_compute_triple_barrier_label()` in `backtest_engine.py`: +1 (TP hit), -1 (SL hit), 0 (time expiry) |
| **López de Prado** — Sample Weights (Ch. 4) | `_compute_sample_weights()`: average uniqueness for overlapping 90-day labels → `sample_weight` in `GradientBoosting.fit()` |
| **López de Prado** — Fractional Differentiation (Ch. 5) | `fractional_diff(d=0.4)` in `historical_data.py`: applied to price, market_cap, revenue, debt, equity |
| **López de Prado** — Purged Walk-Forward CV (Ch. 7) | `WalkForwardScheduler` with 5-day embargo, expanding windows |
| **López de Prado** — MDI + MDA Feature Importance (Ch. 8) | `_compute_mda()` (permutation, primary) + MDI from `feature_importances_` (secondary) |
| **Bailey & López de Prado (2014)** — Deflated Sharpe Ratio | `compute_deflated_sharpe()` in `analytics.py`: penalizes multiple testing; DSR ≥ 0.95 gate in optimizer |
| **Lopez-Lira & Tang (2023)** — LLM Knowledge Contamination | Two-regime architecture: quant-only for historical, full LLM pipeline for live |
| **FinRL (2020)** — Three-layer Architecture | Data layer (BQ) → Agent layer (ML model) → Analytics layer (Sharpe, DSR, baselines) |
| **TradingAgents (2024)** — Multi-agent Debate | MDA feature importance bridges backtest insights → live agent prompt targeting |

**Two-Loop Architecture**:
```
Fast Loop: QuantStrategyOptimizer (minutes/cycle)
├── Propose parameter modification (random or LLM-guided)
├── Run full walk-forward backtest
├── Evaluate: Sharpe, DSR, alpha, hit rate
├── Keep if DSR ≥ 0.95 AND metric improves
└── Log to quant_results.tsv

Slow Loop: SkillOptimizer (days/cycle, existing v2.5)
├── MDA feature importance → identifies which features matter
├── Maps features → responsible agents
├── Targets prompt modifications at underperforming agents
└── Evaluates via outcome_tracking table

Bridge: MDA feature importance from fast loop informs slow loop targeting
```

**New Environment Variables (14)**:
```env
# Backtest
BACKTEST_START_DATE=2023-01-01
BACKTEST_END_DATE=2025-12-31
BACKTEST_TRAIN_WINDOW_MONTHS=12
BACKTEST_TEST_WINDOW_MONTHS=3
BACKTEST_EMBARGO_DAYS=5
BACKTEST_HOLDING_DAYS=90
BACKTEST_TP_PCT=10.0
BACKTEST_SL_PCT=10.0
BACKTEST_FRAC_DIFF_D=0.4
BACKTEST_TARGET_VOL=0.15
BACKTEST_TOP_N_CANDIDATES=50
BACKTEST_STARTING_CAPITAL=100000.0
BACKTEST_MAX_POSITIONS=20
BACKTEST_TRANSACTION_COST_PCT=0.1
```

**New Dependencies**: `scikit-learn>=1.4.0`, `scipy>=1.12.0`

**~49 Feature Vector** (built at any historical cutoff date):
| Category | Features |
|----------|----------|
| Price & Returns | `price_at_analysis`, `momentum_1m`/`3m`/`6m`/`12m`, `annualized_volatility` |
| Technical | `rsi_14`, `sma_50_distance`, `sma_200_distance`, `volume_ratio_20d` |
| Monte Carlo | `var_95_6m`, `var_99_6m`, `expected_shortfall_6m`, `prob_positive_6m` |
| Anomaly | `anomaly_count` |
| Fundamentals | `pe_ratio`, `pb_ratio`, `debt_equity`, `roe`, `profit_margin`, `market_cap`, `total_revenue`, `net_income`, `total_debt`, `total_equity`, `total_assets`, `fcf_yield`, `dividend_yield`, `quality_score`, `revenue_growth_yoy` |
| Macro | `fed_funds_rate`, `cpi_yoy`, `unemployment_rate`, `gdp_growth`, `yield_curve_spread`, `consumer_sentiment`, `treasury_10y` |
| Advanced | `amihud_illiquidity`, `turbulence_index` |
| Fractionally Differenced | `frac_diff_price`, `frac_diff_market_cap`, `frac_diff_revenue`, `frac_diff_debt`, `frac_diff_equity` |

### v3.4 — Multi-Provider LLM Support (March 2026)

Introduced a unified `LLMClient` abstraction layer supporting Gemini, GitHub Models (Copilot Pro), Anthropic Claude, and OpenAI GPT/o-series as drop-in LLM backends. Provider routing is transparent — the existing 2-slot model architecture (standard + deep-think) is preserved with zero pipeline changes.

**New File: `backend/agents/llm_client.py`** (280 lines):
- `UsageMeta(prompt_token_count, candidates_token_count, total_token_count)` — normalised usage dataclass compatible with `CostTracker.record()`
- `LLMResponse(text, thoughts, usage_metadata, grounding_metadata)` — normalised response
- `LLMClient` ABC with a single method `generate_content(prompt, generation_config) -> LLMResponse`
- `GeminiClient` — wraps Vertex AI `GenerativeModel`; extracts grounding metadata + `part.thinking` for extended thinking
- `OpenAIClient` — covers direct OpenAI AND GitHub Models (toggled by `base_url`); injects structured-output schema as system prompt for JSON mode
- `ClaudeClient` — Anthropic direct; maps `max_output_tokens` → `max_tokens`; parses thinking blocks
- `make_client(model_name, vertex_model, settings) -> LLMClient` — factory with priority routing:
  1. If `model_name` is in `GITHUB_MODELS_CATALOG` AND `GITHUB_TOKEN` is set → `OpenAIClient` via `https://models.inference.ai.azure.com`
  2. Elif starts with `claude-` AND `ANTHROPIC_API_KEY` is set → `ClaudeClient`
  3. Elif starts with `gpt-` / `o1` / `o3` / `o4` AND `OPENAI_API_KEY` is set → `OpenAIClient`
  4. Fallback → `GeminiClient` (existing Vertex AI path)
- `GITHUB_MODELS_CATALOG` — set of 25+ model names routable via GitHub Models: GPT-4o, GPT-4.1, o1/o3/o4-mini, Claude 3.5/3.7/Sonnet 4/Opus 4, Meta Llama 3.1, Phi-4, Mistral Large

**New Settings (`backend/config/settings.py`)**:
```env
ANTHROPIC_API_KEY=sk-ant-...   # Direct Anthropic access
OPENAI_API_KEY=sk-...           # Direct OpenAI access
GITHUB_TOKEN=ghp_...            # GitHub PAT (Copilot Pro) — primary testing path, ~150 req/day
```

**`backend/agents/orchestrator.py` Changes**:
- **Gemini Fallback**: `_resolve_gemini(model_name)` static method resolves all `GenerativeModel` instances to valid Gemini model names. When the user selects a non-Gemini model (Claude, GPT, etc.), RAG, grounding, and Vertex AI fallback models stay on `gemini-2.0-flash`. This prevents Vertex AI 404 errors when non-Gemini models are selected as standard/deep-think.
- Model instances replaced by `LLMClient` objects:
  - `self.general_client`, `self.deep_think_client`, `self.synthesis_client` → provider-routed via `make_client()`
  - `self.rag_client: GeminiClient` — always Gemini (Vertex AI Search is Google-only), uses `_resolve_gemini()` fallback
  - `self.grounded_client: GeminiClient` — always Gemini (Google Search Grounding is Google-only), uses `_resolve_gemini()` fallback
  - `self.supports_grounding: bool = isinstance(self.general_client, GeminiClient)` — when non-Gemini model selected, grounded agents fall back to `self.general_client` (text-only, no citations)
- `_generate_with_retry()`: thinking injection guarded by `isinstance(model, GeminiClient)` — Claude/OpenAI handle their own thinking natively; added generic `except Exception` retry for non-GCP transient errors (rate limit, overload, unavailable)
- `_extract_thoughts()` / `_extract_grounding_metadata()`: check `isinstance(response, LLMResponse)` first

**`backend/agents/debate.py` + `backend/agents/risk_debate.py` Changes**:
- `_generate_with_retry(model: LLMClient, ...)` — same thinking guard pattern
- `run_debate(model: LLMClient, ..., deep_think_model: LLMClient | None = None, ...)`
- `run_risk_debate()` same signature update

**`backend/agents/cost_tracker.py` Changes**:
- `MODEL_PRICING` expanded from 3 → 28 entries across all 4 providers:
  - Gemini (3): `gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-2.5-pro`
  - Anthropic (6): `claude-3-5-haiku-20241022`, `claude-3-5-sonnet-20241022`, `claude-3-7-sonnet-20250219`, `claude-sonnet-4-6`, `claude-sonnet-4`, `claude-opus-4`
  - OpenAI (9): `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`, `o1`, `o1-mini`, `o3`, `o3-mini`, `o4-mini`
  - Meta/Mistral/Github (5+): `meta-llama-3.1-405b-instruct`, `meta-llama-3.1-70b-instruct`, `meta-llama-3.1-8b-instruct`, `mistral-large-2407`, `mistral-nemo`

**`backend/api/settings_api.py` Changes**:
- `ModelPricing` class: added `provider: str = "Gemini"` field; added `copilot_multiplier: Optional[float] = None` — GitHub Copilot Pro premium quota multiplier (0.33x light / 1x standard / 3x premium) for GitHub Models entries
- `AVAILABLE_MODELS`: expanded to 24 entries grouped by provider (Gemini / GitHub Models / Anthropic). GitHub Models entries include `copilot_multiplier` values. o1/o1-mini/o3-mini reclassified from OpenAI direct to GitHub Models.
- `_VALID_MODELS` whitelist: 25 names
- `FullSettings`: added 3 read-only booleans: `anthropic_key_configured`, `openai_key_configured`, `github_token_configured` (populated from env without exposing the actual key values)

**Copilot Premium Quota Multipliers** (GitHub Models only — shown in UI instead of $/1M pricing):
| Multiplier | Models |
|-----------|--------|
| `0.33x` (light) | `claude-3-5-haiku-20241022`, `gpt-4o-mini`, `gpt-4.1-mini`, `o3-mini`, `o4-mini`, `meta-llama-3.1-70b-instruct`, `meta-llama-3.1-8b-instruct`, `phi-4`, `mistral-nemo` |
| `1x` (standard) | `claude-3-5-sonnet-20241022`, `claude-3-7-sonnet-20250219`, `claude-sonnet-4`, `gpt-4o`, `gpt-4.1`, `o1-mini`, `meta-llama-3.1-405b-instruct`, `mistral-large-2407` |
| `3x` (premium) | `o1`, `o3`, `claude-opus-4` |
| N/A (price-based) | Gemini, `claude-sonnet-4-6` (Anthropic direct) — show `$X.XX/1M` instead |

**Frontend Changes**:
- `frontend/src/lib/types.ts`: `ModelPricing.provider?: string`; `ModelPricing.copilot_multiplier?: number`; `FullSettings` gets 3 new optional booleans
- `frontend/src/app/settings/page.tsx`: Model Configuration BentoCard redesigned as VS Code GitHub Copilot-style model picker:
  - **`ModelPicker` component**: Searchable list replacing `<select>/<optgroup>`. Grouped by provider with a collapsible "Other Models" section for non-primary models. Selected model shown with checkmark.
  - **`CostBadge` component**: For GitHub Models when `github_token_configured`, shows `{multiplier}x` quota badge (green=0.33x, neutral=1x, amber=3x). For Gemini/Anthropic/OpenAI, shows `$X.XX/1M` in slate-500.
  - **`PRIMARY_MODEL_NAMES` set**: Controls which models appear in the main list vs "Other Models" collapsible.
  - **`MODEL_DISPLAY_NAMES`**: Human-readable names for all 24 models.
  - **Live Cost Estimator update**: When GitHub Models are selected, shows estimated premium request count alongside dollar cost: `~N Copilot premium requests`.

**Constraints**:
- `ENABLE_THINKING=true` still requires `DEEP_THINK_MODEL=gemini-2.5-flash` (or later) — thinking injection is silently skipped for non-Gemini deep-think models
- Google Search Grounding (Step 4/5/9/10 agents) and Vertex AI Search RAG (Step 3) always remain on Gemini regardless of model selection
- GitHub Models rate limit: ~150 requests/day on Copilot Pro — suitable for testing, not production analyses

**v3.4 Bug Fixes (post-release)**:

**(1) o-series `max_completion_tokens` fix (`backend/agents/llm_client.py`)**:
- OpenAI o1/o3/o4-series reasoning models reject `max_tokens` and `temperature` parameters
- Fixed with `_is_reasoning = self.model_name.startswith(("o1", "o3", "o4"))` flag in `OpenAIClient.generate_content()`
- Reasoning models use `max_completion_tokens` and omit `temperature`; all other models keep `max_tokens` + `temperature`

**(2) Live activity message now shows selected model name (`backend/agents/orchestrator.py`)**:
- Progress message was hardcoded as "Gemini analyzing {name}..." regardless of selected provider
- Fixed: `_model_label = self.settings.gemini_model` captured at start of `run_full_analysis()` and injected into the enrichment_analysis step message

**(3) Token limit guard for small-context models (`backend/agents/llm_client.py` + `backend/agents/orchestrator.py`)**:
- GitHub Models enforces a ~4,000 token (~14K char) input limit on `o3-mini`; `enrichment_for_debate` passes 10 sources × full `analysis` text (6–18K tokens total) — causing 413 `tokens_limit_reached` errors
- **`_MODEL_MAX_INPUT_CHARS` registry** added in `llm_client.py`: per-model character cap map (`o1-mini`/`o3-mini`: 13K, `o4-mini`: 56K, `gpt-4.1-mini`/`gpt-4o-mini`: 26K, small Phi/Mistral/Llama models: 14K)
- **`get_model_max_input_chars(model_name)`** public helper exported from `llm_client.py`
- **Safety-net** in `OpenAIClient.generate_content()`: before the API call, if total prompt chars exceed the model cap, truncates the last message to fit with a `[Context truncated — model input limit]` suffix and logs a warning
- **Proactive compaction** in `orchestrator.py` after `enrichment_for_debate` is built: when `general_client` model limit < 30K chars, drops the `analysis` field from each enrichment entry and applies tiered caps based on `lite` flag before passing to `run_debate()`:
  - Full Mode (non-lite): `summary` capped to 200 chars, `fact_ledger_json` to 1,500 chars
  - Lite Mode: `summary` capped to 100 chars, `fact_ledger_json` to 800 chars, **plus** ERROR/UNAVAILABLE/N/A signals stripped entirely from debate input (dead signals add noise without evidence value)
- Log message includes `lite=True/False` and final signal count for observability
- **`context_limited: bool` field** added to `ModelPricing` (`settings_api.py`) and `ModelPricing` TypeScript interface: marks `gpt-4.1-mini`, `gpt-4o-mini`, `o1-mini`, `o3-mini`, `meta-llama-3.1-8b-instruct`, `phi-4`, `mistral-nemo` as context-limited
- **`ModelPicker` UI warning**: context-limited models show an amber `ctx limit` chip in the dropdown; selecting a context-limited model as the Standard Model displays an amber info banner explaining debate compaction and recommending a full-context alternative

**(4) GitHub Models API endpoint migration (`backend/agents/llm_client.py` + `backend/api/settings_api.py`)**:
- GitHub Models migrated from `https://models.inference.ai.azure.com` (Azure-hosted) to `https://models.github.ai/inference` (new GitHub-native endpoint)
- New endpoint requires **namespaced model IDs** in `{publisher}/{model_name}` format (e.g. `openai/gpt-4.1`, `anthropic/claude-sonnet-4`) — confirmed by [GitHub REST API docs](https://docs.github.com/en/rest/models/inference)
- Newer models like `claude-sonnet-4` and `claude-opus-4` were added **only** to the new endpoint and returned `400 unknown_model` on the old Azure endpoint
- **`base_url`** in `make_client()` changed to `"https://models.github.ai/inference"`
- **`_GITHUB_MODELS_ID_MAP`** fully rewritten with namespaced IDs for all 29 models across 5 publishers: `openai/*`, `anthropic/*`, `meta/*`, `microsoft/*`, `mistral-ai/*`
- **`GITHUB_MODELS_CATALOG`** restored: `claude-sonnet-4` and `claude-opus-4` added back (they ARE on GitHub Models)
- **`settings_api.py`**: `claude-sonnet-4` and `claude-opus-4` reverted to `"provider": "GitHub Models"` with `copilot_multiplier` 1.0 and 3.0 respectively; `claude-sonnet-4-6` remains `"provider": "Anthropic"` (direct API only)

**(5) GitHub Models catalog refresh — new models added (June 2026)**:
Live catalog fetched from `GET https://models.github.ai/catalog/models`. Updated all three files: `llm_client.py`, `settings_api.py`, `cost_tracker.py`, and `frontend/.../settings/page.tsx`.

**Models added**:
| Model | Provider | Tier | Copilot Multiplier | Context Limited |
|-------|----------|------|-------------------|----------------|
| `gpt-4.1-nano` | OpenAI | low | 0.33x | ✓ |
| `gpt-5` | OpenAI | custom (8/day) | 3x | ✓ |
| `gpt-5-chat` | OpenAI | custom (12/day) | 1x | ✓ |
| `gpt-5-mini` | OpenAI | custom (12/day) | 1x | ✓ |
| `gpt-5-nano` | OpenAI | custom (12/day) | 1x | ✓ |
| `o1-preview` | OpenAI | custom (8/day) | 3x | ✓ |
| `deepseek-r1` | DeepSeek | custom (8/day) | 3x | ✓ |
| `deepseek-r1-0528` | DeepSeek | custom (8/day) | 3x | ✓ |
| `deepseek-v3-0324` | DeepSeek | high | 1x | |
| `grok-3` | xAI | custom (15/day) | 1x | ✓ |
| `grok-3-mini` | xAI | custom (30/day) | 0.33x | ✓ |
| `llama-3.3-70b-instruct` | Meta | high | 1x | |
| `llama-4-maverick` | Meta | high | 1x | |
| `llama-4-scout` | Meta | high | 1x | |
| `mai-ds-r1` | Microsoft | custom (8/day) | 3x | ✓ |
| `phi-4-mini-instruct` | Microsoft | low | 0.33x | ✓ |
| `phi-4-mini-reasoning` | Microsoft | low | 0.33x | ✓ |
| `phi-4-reasoning` | Microsoft | low | 0.33x | ✓ |
| `codestral-2501` | Mistral | low | 0.33x | ✓ |
| `mistral-medium-2505` | Mistral | low | 0.33x | ✓ |
| `mistral-small-2503` | Mistral | low | 0.33x | ✓ |

**Models removed** (no longer in live catalog):
- `meta-llama-3.1-70b-instruct` (superseded by `llama-3.3-70b-instruct`)
- `mistral-large-2407`, `mistral-nemo` (superseded by `mistral-medium-2505`, `ministral-3b`)
- `phi-3.5-moe-instruct`, `phi-3.5-mini-instruct`, `phi-3-medium-128k-instruct` (superseded by `phi-4*`)

**Primary model list in settings UI** updated to surface `gpt-5`, `deepseek-r1`, `llama-4-maverick`, `grok-3` by default; all other new models appear in the collapsible "Other Models" section.

**`_MODEL_MAX_INPUT_CHARS`** updated: all custom-tier models (gpt-5 family, o1-preview, deepseek-r1/r1-0528, grok-3/3-mini, mai-ds-r1) assigned 13,000-char limit to match 4,000-token GitHub Models cap; `gpt-4.1-nano` gets 26,000 chars (low tier, 8K limit); removed stale phi-3.x entries.

**(6) Structured compaction for debate and revision flows (`backend/agents/compaction.py` + `backend/agents/debate.py` + `backend/agents/orchestrator.py`)**:
- Live runs still hit `413 tokens_limit_reached` on GitHub Models `gpt-4.1` during the Moderator step because proactive compaction was keyed off canonical names like `gpt-4.1`, while GitHub requests use namespaced IDs like `openai/gpt-4.1`
- Added `_normalize_model_name()` in `llm_client.py` so `get_model_max_input_chars()` resolves namespaced GitHub model IDs back to canonical keys before applying size budgets
- Added explicit 26,000-char caps for standard 8K-input GitHub models such as `gpt-4.1` and `gpt-4o`; unresolved GitHub catalog entries now fall back to 26,000 chars instead of skipping compaction entirely
- New `backend/agents/compaction.py` module provides deterministic compact-state helpers instead of replaying large raw transcripts: `compact_text()`, `compact_argument()`, `compact_trace_summary()`, `build_compact_debate_history()`, `compact_da_result()`, `compact_quant_snapshot()`, and `compact_report_reference()`
- `debate.py` now switches to compact JSON for constrained models, trims trace payloads and fact ledgers, compacts Bull/Bear rebuttal carry-forward, compresses Devil's Advocate inputs, and gives the Moderator a bounded round summary instead of full prior arguments
- `orchestrator.py` now applies compact-mode section budgets to Synthesis/Critic, sends a reduced quant snapshot to the Critic, and passes a compact typed reference of the prior draft into Critic + synthesis revision rather than the full report body
- Design principle: use provider-agnostic structured compact state at stage boundaries, not "compacted conversation" transcripts; this preserves the highest-value evidence while keeping GitHub Models requests under hard limits

### v3.3 — Gemini 2.5 Flash + Extended Thinking: Phase 5 (March 2026)

Upgraded the deep-think model to `gemini-2.5-flash` and introduced opt-in extended thinking (chain-of-thought) for the four judge agents — Critic, Synthesis, Moderator, and Risk Judge. Tiered token budgets are set per agent. The feature is safe-defaulted to `false` so existing deployments on Gemini 2.0 are unaffected.

**New Settings (`backend/config/settings.py`)**:
*   `deep_think_model` default changed from `""` → `"gemini-2.5-flash"`
*   `enable_thinking: bool = False` — opt-in flag; must be `true` to activate thinking mode
*   `thinking_budget_critic: int = 8192` — token budget for Critic Agent reasoning
*   `thinking_budget_moderator: int = 8192` — token budget for Moderator Agent reasoning
*   `thinking_budget_risk_judge: int = 4096` — token budget for Risk Judge reasoning
*   `thinking_budget_synthesis: int = 4096` — token budget for Synthesis Agent reasoning

**New Environment Variables**:
```env
ENABLE_THINKING=false           # Set to true when using gemini-2.5-flash or later
THINKING_BUDGET_CRITIC=8192
THINKING_BUDGET_MODERATOR=8192
THINKING_BUDGET_RISK_JUDGE=4096
THINKING_BUDGET_SYNTHESIS=4096
```

**`backend/agents/orchestrator.py` Changes**:
*   Import: added `GenerationConfig` to vertexai imports
*   4 module-level thinking config dicts: `_THINKING_CRITIC_CONFIG`, `_THINKING_MODERATOR_CONFIG`, `_THINKING_RISK_JUDGE_CONFIG`, `_THINKING_SYNTHESIS_CONFIG` — each merges the existing structured output config with `{"thinking": {"type": "enabled", "budget_tokens": N}, "include_thoughts": True}`
*   `__init__`: stores `self.enable_thinking` and `self.thinking_budgets = {"Critic": N, "Moderator": N, "Risk Judge": N, "Synthesis": N}`
*   `_generate_with_retry()`: new thinking injection block — when `enable_thinking=True` and `is_deep_think=True` and `agent_name in self.thinking_budgets`, merges thinking config into `generation_config` before the API call
*   New `_extract_thoughts()` helper: safely reads `part.thinking` from Vertex AI response candidates; capped at 2000 chars for storage efficiency
*   `run_debate()` call site: added `enable_thinking=self.enable_thinking, thinking_budgets=self.thinking_budgets`
*   `run_risk_debate()` call site: added `enable_thinking=self.enable_thinking, thinking_budgets=self.thinking_budgets`

**`backend/agents/debate.py` Changes**:
*   `_generate_with_retry()`: added `thinking_budget: int = 0` parameter — when > 0, merges `{"thinking": {"type": "enabled", "budget_tokens": thinking_budget}, "include_thoughts": True}` into the config via dict spread
*   `run_debate()` signature: added `enable_thinking: bool = False, thinking_budgets: dict | None = None`
*   Moderator call: computes `_moderator_thinking_budget = (thinking_budgets or {}).get("Moderator", 0) if enable_thinking else 0` and passes it to `_generate_with_retry`

**`backend/agents/risk_debate.py` Changes**:
*   Same pattern as `debate.py`: `thinking_budget` in `_generate_with_retry`, new params on `run_risk_debate()`
*   Risk Judge call: computes `_judge_thinking_budget = (thinking_budgets or {}).get("Risk Judge", 0) if enable_thinking else 0`

**`backend/agents/trace.py` Change**:
*   `DecisionTrace` dataclass: added `thoughts: str = ""` field — stores extended thinking output for Glass Box audit trail rendering

**Tiered Thinking Budgets**:

| Agent | Budget Tokens | Rationale |
|-------|---------------|-----------|
| Critic | 8,192 | Deepest reasoning — Chain-of-Verification requires meticulous claim checking |
| Moderator | 8,192 | Must resolve Bull/Bear contradictions with full debate history |
| Risk Judge | 4,096 | Risk sizing verdict; moderate depth needed |
| Synthesis | 4,096 | Structured JSON output is constrained; budget limits runaway thinking |
| All others | 0 (disabled) | Enrichment agents use tight token limits; thinking cost outweighs benefit |

**Activation Requirement**: `ENABLE_THINKING=true` **requires** `DEEP_THINK_MODEL=gemini-2.5-flash` (or a later Gemini 2.5+ model). Do not enable on `gemini-2.0-flash` — the thinking config will be silently ignored or cause errors.

### v3.2 — Google Search Grounding: Phase 4 (March 2026)

Selective Google Search Grounding on 4 agents for live web fact-checking with source citations. Constraint: Schema + Grounding cannot combine on Gemini 2.0, so grounded agents use separate model instances and produce text responses (not structured output).

**New Model Instance**:
*   `orchestrator.py` — New `self.grounded_model = GenerativeModel(model, tools=[search_tool], generation_config=_gen_config)` using `Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())`

**Grounded Agents (4)**:
*   **Market Agent** (Step 4) — Breaking news supplement via Google Search
*   **Competitor Agent** (Step 5) — Real-time M&A, partnerships, competitive dynamics
*   **Enhanced Macro Agent** (Step 9) — Latest Fed speeches, policy context on top of FRED data
*   **Deep Dive Agent** (Step 10) — Verify claims against public record

**Grounding Metadata Extraction**:
*   New `_extract_grounding_metadata()` helper in `orchestrator.py` — Extracts `groundingChunks` (source URLs + titles) and `groundingSupports` (text→source mapping) from Vertex AI responses
*   All 4 grounded agent methods return `grounding_sources` in their output dict
*   Final report includes `grounding_sources` dict with per-agent metadata for Glass Box rendering

**DecisionTrace Extension (1 file)**:
*   `trace.py` — New `grounding_sources: list[dict]` field on `DecisionTrace` dataclass

**Cost Tracking Extension (1 file)**:
*   `cost_tracker.py` — New `is_grounded: bool` field on `AgentCostEntry`, tracked through `record()` and exposed in `summarize()` output

**Deep Dive Agent Return Type Change**:
*   `run_deep_dive_agent()` now returns `dict` (was `str`) with `text` and `grounding_sources` keys
*   Synthesis pipeline updated to extract `.text` from deep_dive dict

### v3.1 — Anti-Hallucination Stack: Phases 1-3 (March 2026)

Three-phase anti-hallucination system targeting temperature determinism, fact anchoring, and structured output enforcement. Research basis: VeNRA (Typed Fact Ledger), Chain-of-Verification (CoVe), OpenAI Structured Outputs, Google Gemini structured output.

**Phase 1: Temperature Determinism**:
*   All 8 generation configs set to `temperature=0.0, top_k=1` across `orchestrator.py`, `debate.py`, `risk_debate.py`
*   Preserved creative temperatures for `memory.py` (0.3) and `skill_optimizer.py` (0.7, 0.9)

**Phase 2: Fact Ledger + Prompt Hardening (11 files, 28 skills)**:
*   `orchestrator.py` — New `_build_fact_ledger()` builds a 26-field typed fact dict from `quant_data["yf_data"]` after Step 2, stored as `self._fact_ledger_json`
*   `prompts.py` — New `_build_fact_ledger_section()` helper; all 29 prompt functions accept `fact_ledger: str` and inject `{{fact_ledger_section}}` via `format_skill()`
*   28 `backend/agents/skills/*.md` — All updated with `{{fact_ledger_section}}` placeholder + 5 anti-hallucination rules (cite FACT_LEDGER, flag discrepancies, never fabricate, etc.)
*   `SKILL_TEMPLATE.md` — Updated with fact ledger section and anti-hallucination rules
*   All 31 call sites wired: orchestrator agent methods (via `self._fact_ledger_json`), `run_debate()`, `run_risk_debate()`, synthesis pipeline
*   `critic_agent.md` — Enhanced with Chain-of-Verification (CoVe) 3-step loop: extract claims → verify each against FACT_LEDGER → flag mismatches
*   `synthesis_agent.md` — Pillar anchoring: each pillar lists specific FACT_LEDGER fields it must cite; scoring guardrails (score >7 requires PEG <1.5 or P/E <25)

**Phase 3: Gemini Structured Output Enforcement (4 files)**:
*   NEW: `backend/agents/schemas.py` — 10 Pydantic models: `SynthesisReport` (+ `ScoringMatrix`, `Recommendation`), `CriticVerdict` (+ `CriticIssue`), `DevilsAdvocateResult`, `ModeratorConsensus` (+ `Contradiction`, `Dissent`), `RiskAnalystArgument`, `RiskJudgeVerdict` (+ `RiskLimits`)
*   Uses `Literal` type constraints on critical enum fields: `consensus` (5 values), `verdict` (PASS/REVISE), `decision` (4 values), `risk_level` (4 values)
*   `orchestrator.py` — `_generate_with_retry()` extended with `generation_config` parameter; 3 call sites (Synthesis draft, Synthesis revision, Critic) pass `_SYNTHESIS_STRUCTURED_CONFIG` / `_CRITIC_STRUCTURED_CONFIG` with `response_mime_type="application/json"` + `response_schema`
*   `debate.py` — New `_DA_STRUCTURED_CONFIG` and `_MODERATOR_STRUCTURED_CONFIG`; DA and Moderator calls use structured output schemas
*   `risk_debate.py` — New `_RISK_STRUCTURED_CONFIG` and `_JUDGE_STRUCTURED_CONFIG`; all 3 analyst calls + Judge call use structured output schemas
*   Existing `_clean_json_output` / `_parse_json_with_fallback` / `_clean_json` / `_parse_json` retained as safety fallbacks
*   SDK pattern: `{"response_mime_type": "application/json", "response_schema": PydanticModel}` (requires `google-cloud-aiplatform>=1.60.0`)
*   Constraint: Schema + Grounding cannot combine on Gemini 2.0 — structured output agents and grounded agents (Phase 4) use separate model instances

### v3.0 — Design System Overhaul: Geist + Phosphor Icons (March 2026)

Complete frontend design system migration: Geist self-hosted font replacing Google Fonts Inter, Phosphor Icons replacing all emoji icons across 20+ components and 9 pages, Motion animation library installed, and TradingView Lightweight Charts installed for future StockChart rewrite.

**Design System Foundation (4 config files)**:
*   `next.config.js` — Added `experimental.optimizePackageImports: ["@phosphor-icons/react"]` for tree-shaking
*   `layout.tsx` — Geist font loading via `geist/font/sans` and `geist/font/mono`, applied as CSS variable classes on `<html>`
*   `tailwind.config.js` — CSS variable font families (`var(--font-geist-sans/mono)`), navy color palette (`navy-500: "#243352"`, `navy-600: "#1a2744"`), shadow tokens (`card`, `card-hover`), border radius tokens (`card: 12px`, `button: 8px`, `badge: 6px`)
*   `globals.css` — `tabular-nums`, antialiasing, `skeleton` shimmer keyframe in `@layer base`

**Centralized Icon System (2 new files)**:
*   `frontend/src/lib/icons.ts` — ~110 aliased Phosphor re-exports organized by domain: Navigation (8), Pipeline Steps (16), Signals (12), Debate (7), Risk Team (4), Bias/Audit (12), Evaluation Pillars (5), Macro Indicators (8), GlassBox (4), Tabs (6), Utility (20+). All icons use `Icon` type from `@phosphor-icons/react` for TypeScript safety
*   `frontend/src/lib/motion.ts` — Shared motion variants (`fadeIn`, `slideUp`, `staggerContainer`, `staggerItem`) and spring presets (`springSnappy`, `springGentle`, `hoverTap`) for future animation integration

**Emoji → Phosphor Icon Conversion (15 components, 5 pages)**:
*   All emoji characters removed from entire frontend codebase (verified via grep: 0 matches)
*   Icon data types changed from `icon: string` to `icon: Icon` in all component interfaces (`TabDef`, `SIGNAL_META`, `PillarConfig`, `ProbabilityCard`, etc.)
*   Icon rendering changed from `{meta.icon}` string interpolation to `<meta.icon size={20} weight="duotone" />` JSX component rendering
*   Components converted: SignalCards, SignalDashboard, MacroDashboard, BiasReport, DecisionTraceView, DebateView, RiskDashboard, GlassBoxCards, EvaluationTable, CostDashboard, ValuationRange, PdfDownload, StockChart, ReportTabs, ResearchInvestigator
*   Pages converted: page.tsx, signals/page.tsx, compare/page.tsx, performance/page.tsx, reports/page.tsx
*   Previously converted (prior session): Sidebar.tsx, AnalysisProgress.tsx

**Package Changes**:
*   Added: `geist@1.7.0`, `@phosphor-icons/react@2.1.10`, `motion@12.38.0`, `lightweight-charts`
*   Removed: `lucide-react` (fully replaced by Phosphor Icons)

**Icon Naming Convention**: All icons are aliased with domain prefixes for discoverability:
*   Navigation: `Nav*` (NavDashboard, NavSignals, NavReports, …)
*   Pipeline: `Step*` (StepMarketIntel, StepIngestion, StepQuant, …)
*   Signals: `Signal*` (SignalInsider, SignalOptions, SignalSocial, …)
*   Debate: `Debate*` (DebateBull, DebateBear, DebateConsensus, …)
*   Risk: `Risk*` (RiskAggressive, RiskConservative, RiskNeutral, RiskJudge)
*   Settings: `Settings*` (SettingsMode, SettingsDebate, SettingsModel, SettingsCostControls, SettingsEstimator, SettingsPillars, SettingsCache, SettingsOptimizer, SettingsLatency, SettingsRefresh)
*   Tabs: `Tab*` (TabOverview, TabSignals, TabDebate, TabRisk, TabAudit, TabCost)
*   Utility: `Icon*` (IconWarning, IconSearch, IconDownload, IconChart, …)

### v2.9 — Autonomous Paper Trading System (March 2026)

Fully autonomous AI trading agent managing a virtual $10,000 portfolio. Daily cycle: quant screen S&P 500 → deep-analyze top candidates (lite mode) → execute virtual trades → track P&L vs SPY → learn from outcomes. Configurable via 11 new settings.

**New Backend Modules (3)**:
*   `backend/tools/screener.py` — S&P 500 quant screener: batch yfinance download, momentum/RSI/volatility/SMA filters, composite alpha score ranking. Zero LLM cost. Fallback 50-ticker list if Wikipedia scrape fails.
*   `backend/services/paper_trader.py` — `PaperTrader` class: virtual trade execution engine backed by BigQuery. `execute_buy()` (position averaging, cash check, max positions), `execute_sell()` (full/partial exit), `mark_to_market()` (live yfinance prices + SPY benchmark), `check_stop_losses()`, `save_daily_snapshot()`.
*   `backend/services/portfolio_manager.py` — `decide_trades()`: sell-first-then-buy logic. Sells on SELL/STRONG_SELL signal, signal downgrade (BUY→HOLD), or stop-loss hit. Buys sized by `min(risk_judge_pct * NAV, available_cash)`, sorted by final_score, respects max_positions and min_cash_reserve.
*   `backend/services/autonomous_loop.py` — `run_daily_cycle()`: 9-step async orchestrator (screen → filter → analyze candidates → re-evaluate holdings → MTM → decide trades → execute → snapshot → learn from closed trades). Module-level state tracking.

**New API Route (1)**:
*   `backend/api/paper_trading.py` — 8 endpoints: `POST /start` (init fund + scheduler), `POST /stop` (pause scheduler), `GET /status` (NAV + scheduler state), `GET /portfolio` (positions), `GET /trades` (history), `GET /snapshots` (daily NAV), `GET /performance` (Sharpe, win rate, alpha), `POST /run-now` (manual trigger). APScheduler cron job wired into FastAPI lifespan.

**BQ Schema (4 new tables)**:
*   `paper_portfolio` (8 cols) — Fund state (cash, NAV, benchmark)
*   `paper_positions` (14 cols) — Open positions with unrealized P&L
*   `paper_trades` (11 cols) — Trade history with reasons and realized P&L
*   `paper_portfolio_snapshots` (11 cols) — Daily NAV for charting and Sharpe calculation
*   Managed by `migrate_paper_trading.py` (idempotent)

**Frontend Dashboard (1 page)**:
*   `frontend/src/app/paper-trading/page.tsx` — Full dashboard: 6 summary cards (NAV, Cash, P&L, vs SPY, Sharpe, Positions), status banner, 3-tab view (Positions table, Trades table, NAV Chart with Recharts LineChart showing Portfolio/SPY/Alpha lines), Initialize Fund / Start / Pause / Run Now action buttons.
*   `frontend/src/lib/types.ts` — 6 new interfaces: `PaperPortfolio`, `PaperPosition`, `PaperTrade`, `PaperSnapshot`, `PaperTradingStatus`, `PaperPerformance`
*   `frontend/src/lib/api.ts` — 8 new functions for paper trading endpoints
*   `frontend/src/components/Sidebar.tsx` — "Paper Trading" (🤖) nav entry added

**Modified Backend Files**:
*   `backend/config/settings.py` — 11 new paper trading settings (capital, max positions, cash reserve, screen/analyze top N, trading hour, re-eval frequency, transaction cost, daily cost cap)
*   `backend/db/bigquery_client.py` — ~150 lines of paper trading CRUD methods (get/upsert portfolio, positions, trades, snapshots)
*   `backend/main.py` — Paper trading router registered, APScheduler init in lifespan startup

**Daily Cycle Cost Estimate**:
*   Screen: $0 (yfinance batch download)
*   Analyze 5 new candidates (lite mode): ~$0.50
*   Re-evaluate ~5 holdings (lite mode): ~$0.50
*   Daily total: ~$1.00/day, configurable via `PAPER_MAX_DAILY_COST_USD`

**New Environment Variables (11)**: `PAPER_TRADING_ENABLED`, `PAPER_STARTING_CAPITAL`, `PAPER_MAX_POSITIONS`, `PAPER_MIN_CASH_RESERVE_PCT`, `PAPER_SCREEN_TOP_N`, `PAPER_ANALYZE_TOP_N`, `PAPER_TRADING_HOUR`, `PAPER_REEVAL_FREQUENCY_DAYS`, `PAPER_TRANSACTION_COST_PCT`, `PAPER_MAX_DAILY_COST_USD`

**New Dependencies**: `APScheduler>=3.10.0` (already present from v2.8 Slack bot)

### v2.8 — Auth + Slack Bot + Deployment (March 2026)

End-to-end authentication (Google SSO + Passkey), Slack bot with slash commands and morning digest, Docker Compose overhaul, and OWASP security hardening.

**Frontend Authentication (12 files)**:
*   `prisma/schema.prisma` — SQLite database with 5 models: User, Account, Session, VerificationToken, Authenticator (WebAuthn)
*   `src/lib/auth.config.ts` — Edge-compatible auth config: Google + Passkey providers, JWT strategy (8h maxAge), email whitelist callback, `authorized` callback for middleware
*   `src/lib/auth.ts` — Full auth config extending auth.config with PrismaAdapter + `experimental: { enableWebAuthn: true }`
*   `src/lib/prisma.ts` — Singleton Prisma client with global hot-reload safety
*   `src/app/api/auth/[...nextauth]/route.ts` — NextAuth v5 catch-all route handler
*   `src/components/AuthProvider.tsx` — SessionProvider wrapper with 15-minute refetch interval
*   `src/app/login/page.tsx` — Login page: Google SSO button (SVG icon) + Passkey button (🔑), PyFinAgent branding, generic error messages, dark theme
*   `src/middleware.ts` — Route protection: imports from `auth.config` (Edge-safe), redirects unauthenticated → `/login`, skips `/api/auth`, `/_next`, `/favicon`
*   `src/lib/api.ts` — Added `getAuthToken()` (reads session cookie), Bearer token injection, 401 → redirect to `/login`, `Cache-Control: no-store`
*   `src/lib/types.ts` — Added `AuthUser` interface
*   `src/components/Sidebar.tsx` — Added user avatar/email display, passkey registration button, logout button
*   `.env.local` — Added `AUTH_SECRET`, `AUTH_GOOGLE_ID`, `AUTH_GOOGLE_SECRET`, `ALLOWED_EMAILS`

**Backend Authentication (3 files)**:
*   `api/auth.py` — HKDF key derivation + JWE A256CBC-HS512/dir decryption using `cryptography` library. `get_current_user()` validates Bearer token, checks email whitelist + token expiry
*   `main.py` — Auth middleware (skips public paths), OWASP security headers (6 headers), CORS updated for Tailscale IPs (`100.x.y.z`)
*   `config/settings.py` — Added `auth_secret`, `allowed_emails`, `slack_bot_token`, `slack_app_token`, `slack_channel_id`, `morning_digest_hour`

**Slack Bot Module (6 files in `backend/slack_bot/`)**:
*   `app.py` — Entry point: AsyncApp + AsyncSocketModeHandler, registers slash commands, starts scheduler
*   `commands.py` — 3 slash commands: `/analyze <TICKER>` (starts analysis, polls 5s intervals), `/portfolio` (P&L summary), `/report <TICKER>` (report card)
*   `scheduler.py` — APScheduler AsyncIOScheduler: morning digest cron job (configurable hour), `send_analysis_alert()` for proactive alerts after analysis completes
*   `formatters.py` — 4 Block Kit message builders: analysis result, portfolio summary, report card, morning digest
*   `Dockerfile` — python:3.11-slim, non-root appuser, Socket Mode (no inbound ports)

**Docker Compose Overhaul**:
*   3 services: `backend`, `frontend`, `slack-bot` (removed redis + celery-worker)
*   `slack-bot` service: depends on backend, Socket Mode (no ports exposed), `restart: unless-stopped`
*   `auth-db` named volume for SQLite persistence across container restarts
*   `restart: unless-stopped` on all services
*   Frontend gets `env_file: ./frontend/.env.local` and volume mount for Prisma

**New Dependencies**: `cryptography>=42.0.0`, `slack-bolt[async]>=1.18.0`, `slack-sdk>=3.27.0`, `APScheduler>=3.10.0`, `next-auth@5.0.0-beta.30`, `@prisma/client@6`, `@auth/prisma-adapter`, `@simplewebauthn/browser@9.0.1`, `@simplewebauthn/server@9.0.3`

**New Environment Variables**: `AUTH_SECRET`, `ALLOWED_EMAILS`, `AUTH_GOOGLE_ID`, `AUTH_GOOGLE_SECRET`, `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `SLACK_CHANNEL_ID`, `MORNING_DIGEST_HOUR`

### v2.7 — Cost Management + Settings UI Overhaul (March 2026)

Multi-layered LLM cost controls, per-agent output token limits, lite mode, budget warnings, prompt truncation, and a complete Settings page overhaul with live cost estimation using real token data from the most recent analysis.

**Cost Controls (Backend)**:
*   `orchestrator.py` — 4 GenerationConfig tiers: `_enrichment_config` (1024 tokens), `_deep_think_config` (2048), `_synthesis_config` (4096), `_gen_config` (base, no limit)
*   `orchestrator.py` — Lite mode: skips deep dive, devil's advocate, risk assessment; forces 1 debate round. Controlled by `LITE_MODE` env var
*   `orchestrator.py` — Prompt truncation: `_MAX_SECTION=1500` per enrichment output, `_MAX_CONTEXT=12000` total market context in synthesis
*   `orchestrator.py` — Configurable synthesis iterations via `MAX_SYNTHESIS_ITERATIONS` (1-3). New `self.synthesis_model` for synthesis calls (uses `_synthesis_config`)
*   `orchestrator.py` — Budget warning: `CostTracker.check_budget()` after analysis completes, adds `budget_warning` to final report
*   `cost_tracker.py` — New `total_cost` property (thread-safe sum) + `check_budget(max_cost_usd)` method
*   `debate.py` — `_DEBATE_GEN_CONFIG` (1536 tokens), `_MODERATOR_GEN_CONFIG` (2048 tokens), `gen_config` parameter on `_generate_with_retry()`, `skip_devils_advocate` parameter
*   `risk_debate.py` — `_RISK_GEN_CONFIG` (1024 tokens), `_JUDGE_GEN_CONFIG` (1536 tokens), `gen_config` parameter on `_generate_with_retry()`
*   `settings.py` — 3 new fields: `lite_mode` (bool), `max_analysis_cost_usd` (float, default 0.50), `max_synthesis_iterations` (int, 1-3, default 2)

**Settings API Overhaul**:
*   `settings_api.py` — Complete rewrite: `FullSettings` model (13 readable fields), `SettingsUpdate` model (all optional, validated)
*   `GET /api/settings/` returns all settings, `PUT /api/settings/` accepts partial updates with validation (model whitelist, pillar weight sum = 1.0)
*   `_FIELD_TO_ENV` mapping persists changes to `.env` file. Legacy `/models` endpoints preserved
*   `reports.py` — New `GET /api/reports/latest-cost-summary` returns per-agent cost breakdown from most recent analysis

**Frontend Settings Overhaul**:
*   `settings/page.tsx` — Complete rewrite with 6 BentoCards:
    1. **Analysis Mode**: Full vs Lite toggle with descriptions
    2. **Live Cost Estimator**: Uses real per-agent token counts from last analysis, scales by model pricing, debate rounds, synthesis iterations, lite mode. Shows $/analysis, total tokens, LLM calls
    3. **Model Configuration**: Standard + Deep Think model dropdowns
    4. **Cost Controls**: Budget slider ($0.05-$5.00), synthesis iterations (1-3), min data quality (0-100%)
    5. **Debate Depth**: Bull↔Bear rounds (1-5), risk rounds (1-3) with lite mode override warning
    6. **Pillar Weights**: 5 weight sliders (0-50%) with live total validation (must = 100%)
*   Single "Save All Settings" button sends only changed fields as diff
*   `types.ts` — New `FullSettings` + `LatestCostSummary` interfaces
*   `api.ts` — New `getFullSettings()`, `updateSettings()`, `getLatestCostSummary()` functions

**New Environment Variables**: `LITE_MODE`, `MAX_ANALYSIS_COST_USD`, `MAX_SYNTHESIS_ITERATIONS` (all optional, with defaults)

### v2.6 — Agent Design Pattern Optimization (March 2026)

Implements 4 design pattern enhancements derived from research analysis of Google Cloud, Andrew Ng, and Anthropic's AI agent design patterns: Reflection Loop (Evaluator-Optimizer), Session Memory, Quality Gates, and Sector Routing. BQ schema expanded from 67 → 68 columns.

**Reflection Loop (Evaluator-Optimizer Pattern)**:
*   `critic_agent.md` restructured: now returns structured `{verdict: "PASS"|"REVISE", issues: [...], corrected_report: {...}}` instead of just corrected JSON
*   `orchestrator.py` `run_synthesis_pipeline()` rewritten: Synthesis → Critic → if REVISE and iteration < 2, re-run Synthesis with Critic's issues → Critic again. Max 2 iterations
*   `prompts.py` — new `get_synthesis_revision_prompt()` function that prepends revision instruction block with Critic issues to the base synthesis template
*   `get_critic_prompt()` now accepts optional `critic_feedback` parameter for iteration context

**Session Memory (AnalysisContext)**:
*   `trace.py` — new `AnalysisContext` dataclass: accumulates `key_findings`, `contradictions`, `signal_consensus` during a run (capped at 20 findings, each ≤100 chars)
*   Populated after Steps 2 (quant), 4 (market), 6 (enrichment signals), 6b (info-gap), 8 (debate)
*   Injected into Synthesis prompt via `format_for_prompt()` → prepended to market_context

**Quality Gates (Conditional Pipeline)**:
*   Data quality gate after Step 6b: if `data_quality_score < data_quality_min` (default 0.5), Steps 8 (debate) and 12c (risk assessment) are skipped with placeholder results
*   Skipped debate returns `consensus: "HOLD"` with low confidence; skipped risk returns `decision: "REJECT"` with 0% position
*   `settings.py` — 3 new configurable thresholds: `data_quality_min` (0.5), `conflict_escalation_threshold` (5), `critic_major_issues_threshold` (3)

**Sector Routing (Tool Skipping)**:
*   `orchestrator.py` — new `SECTOR_SKIP_MAP` dict: Financial Services skips patents, Utilities/Real Estate skip patents + alt_data
*   Step 6 enrichment now conditionally runs `_skip_placeholder()` for sector-irrelevant tools
*   `info_gap.py` — new `SKIPPED` status: skipped tools excluded from data quality denominator

**BQ Schema Expansion (1 new column)**:
*   `("synthesis_iterations", "INT64")` — tracks reflection loop count per analysis

**New Environment Variables**: `DATA_QUALITY_MIN`, `CONFLICT_ESCALATION_THRESHOLD`, `CRITIC_MAJOR_ISSUES_THRESHOLD` (all optional, with defaults)

### v2.5 — Skills System + Autonomous Optimization Loop (March 2026)

Autoresearch-inspired skills architecture: each agent's prompt is defined in a skills.md file, loaded dynamically by a skill loader with mtime caching. An autonomous optimization loop (SkillOptimizer) proposes, tests, and keeps/discards prompt modifications using a single metric. Feedback loop wired: outcome evaluation now generates LLM reflections and persists them to BQ agent_memories table.

**New Backend Modules (2)**:
*   `backend/agents/skill_optimizer.py` — `SkillOptimizer` class: `establish_baseline()`, `analyze_agent_performance()`, `propose_skill_modification()`, `apply_modification()`, `revert_modification()`, `handle_crash()`, `think_harder()`, `run_loop()`. Uses git for experiment tracking. Simplicity criterion: ≥0.5% delta per 10 lines added.
*   `backend/agents/skills/experiments/analyze_experiments.py` — Experiment analysis: keep rate (overall + per-agent), delta chain, running best chart data, top hits table, summary stats. CLI + importable.

**New API Route (1)**:
*   `backend/api/skills.py` — 7 endpoints: POST optimize/stop, GET status/experiments/analysis/agents/{agent_name}

**Skills Architecture (28 files)**:
*   `backend/agents/skills/*.md` — 28 agent skills files following SKILL_TEMPLATE.md format
*   Each contains: Goal, Identity, CAN/CANNOT, Prompt Template (with `{{variable}}` placeholders), Experiment Log
*   `backend/config/prompts.py` refactored: 950 lines of inline prompts → ~380 lines of thin wrappers using `load_skill()` + `format_skill()`

**Feedback Loop Wired**:
*   `outcome_tracker.py` — After evaluating outcomes, generates LLM reflections via `generate_reflection()` for 4 agent types (bull, bear, moderator, risk_judge) and persists to `agent_memories` table via `save_agent_memory()`
*   Closes the learning loop: outcomes → reflections → BM25 memory → future agent prompts

### v2.4 — Dual LLM Strategy + Cost Analytics (March 2026)

Dual-model architecture with "deep think" model for judge agents (Moderator, Risk Judge, Synthesis, Critic), per-agent token/cost tracking across all LLM calls, Cost tab in dashboard, model configuration UI, and cost history on Performance page. BQ schema expanded from 64 → 67 columns.

**New Backend Module (1)**:
*   `backend/agents/cost_tracker.py` — `CostTracker` dataclass: thread-safe per-agent token recording from `response.usage_metadata`, `MODEL_PRICING` dict (Flash/2.5-Flash/2.5-Pro), `summarize()` produces JSON-serializable breakdown by model and agent

**New API Route (1)**:
*   `backend/api/settings_api.py` — `GET /api/settings/models` (current config), `GET /api/settings/models/available` (model list with pricing)

**Dual LLM Architecture**:
*   `settings.py` — New `deep_think_model` field (defaults to empty → falls back to `gemini_model`)
*   `orchestrator.py` — Creates `self.deep_think_model` GenerativeModel, passes to debate/risk_debate; Synthesis + Critic use deep_think_model with `is_deep_think=True`
*   `debate.py` — Moderator uses `deep_think_model or model`; Bull/Bear/DA use standard model
*   `risk_debate.py` — Risk Judge uses `deep_think_model or model`; Aggressive/Conservative/Neutral use standard model
*   All `_generate_with_retry` calls auto-record to `CostTracker` via `getattr(self, "_cost_tracker", None)`

**BQ Schema Expansion (3 new columns)**:
*   **Cost Metrics (+3)**: `total_tokens`, `total_cost_usd`, `deep_think_calls`

**Frontend Enhancements**:
*   `CostDashboard.tsx` — New component: 4 summary cards, token distribution bar, cost by model breakdown, per-agent table sorted by cost
*   `page.tsx` — 6th tab "Cost" (💰) with badge showing total cost
*   `settings/page.tsx` — "Model Configuration" BentoCard with Standard/Deep Think model dropdowns + pricing display
*   `performance/page.tsx` — Cost history section: total spend, avg cost/analysis, per-analysis table from BQ
*   `types.ts` — New interfaces: `AgentCostEntry`, `ModelBreakdown`, `CostSummary`, `CostHistoryEntry`, `ModelPricing`, `ModelConfig`
*   `api.ts` — New functions: `getModelConfig()`, `getAvailableModels()`, `getCostHistory()`

**New Environment Variable**: `DEEP_THINK_MODEL` (optional, e.g., `gemini-2.5-pro`)

### v2.3 — FinancialSituationMemory + Prompt Hardening (March 2026)

BM25-based agent memory that learns from past outcomes, anti-HOLD bias prompts, multi-round risk debate with cross-visibility, and configurable debate depth. Inspired by TradingAgents `FinancialSituationMemory` and prompt-hardening research.

**New Backend Module (1)**:
*   `backend/agents/memory.py` — `FinancialSituationMemory` class: BM25Okapi lexical similarity retrieval, 5 cold-start seed archetypes, `generate_reflection()` for LLM-based lesson extraction from outcome evaluations

**Prompt Hardening (7 prompts updated)**:
*   Anti-HOLD bias in Moderator: "Choose HOLD only if strongly justified by specific arguments — not as a fallback"
*   Anti-HOLD bias in Risk Judge: "Choose REJECT only if strongly justified by specific downside evidence"
*   Anti-hallucination guards in Aggressive/Conservative/Neutral: "do not hallucinate their arguments"
*   `past_memory` parameter added to Bull, Bear, Moderator, all 3 Risk Analysts, and Risk Judge prompts

**Risk Debate Rewrite**:
*   `risk_debate.py` rewritten: single-pass → multi-round with cross-visibility (each analyst sees the other two's prior arguments)
*   Risk Judge now receives full debate history when `max_risk_rounds > 1`
*   Output includes `risk_debate_rounds` and `total_risk_rounds` fields

**Configurable Debate Depth**:
*   `max_debate_rounds` (default 2, range 1-5) — Bull↔Bear iterative rebuttal exchanges
*   `max_risk_debate_rounds` (default 1, range 1-3) — Aggressive/Conservative/Neutral exchanges
*   Controlled via `MAX_DEBATE_ROUNDS` and `MAX_RISK_DEBATE_ROUNDS` env vars

**BQ Persistence**:
*   New `agent_memories` table: `agent_type`, `ticker`, `situation`, `lesson`, `created_at`
*   `migrate_agent_memories.py` — Idempotent table creation script
*   `bigquery_client.py` — Added `get_agent_memories()` and `save_agent_memory()` methods

**Frontend Enhancements**:
*   `settings/page.tsx` — New "Debate Depth" BentoCard with Bull↔Bear and Risk Debate round sliders
*   `types.ts` — Added `RiskDebateRound` interface + `risk_debate_rounds`/`total_risk_rounds` to `RiskAssessment`

**New Dependency**: `rank-bm25>=0.2.2`

### v2.2 — TradingAgents + AlphaQuanter Enhancement (March 2026)

Implemented multi-round adversarial debate, Devil's Advocate stress-testing, round-robin Risk Assessment Team, and AlphaQuanter-style info-gap detection with retry. BQ schema expanded from 55 → 64 columns. Pipeline expanded from 13 → 15 steps.

**New Backend Modules (2)**:
*   `backend/agents/risk_debate.py` — TradingAgents-style round-robin: Aggressive → Conservative → Neutral → Risk Judge
*   `backend/agents/info_gap.py` — AlphaQuanter ReAct loop: scan 11 sources, classify SUFFICIENT/PARTIAL/MISSING, retry critical gaps

**Enhanced Debate Framework**:
*   `debate.py` rewritten: single-round → multi-round iterative Bull↔Bear (each sees opponent's prior argument)
*   New Devil's Advocate agent: stress-tests both sides for hidden risks + groupthink detection
*   Moderator now receives full debate history + DA challenges
*   7 new prompts in `prompts.py`: Devil's Advocate, 4 Risk Analysts, Info-Gap

**New Pipeline Steps (2)**:
*   Step 6b: Info-Gap Detection (between enrichment + enrichment analysis)
*   Step 12c: Risk Assessment Team (after bias audit)

**Frontend Enhancements (4 components)**:
*   `DebateView.tsx` — DA section (violet theme), multi-round timeline
*   `RiskDashboard.tsx` — New `RiskAssessmentPanel` component (judge verdict + analyst cards + risk limits)
*   `AnalysisProgress.tsx` — 2 new steps (info_gap + risk_assessment)
*   `types.ts` — 7 new interfaces (DebateRound, DevilsAdvocate, RiskAssessment, InfoGap, etc.)

**BQ Schema Expansion (9 new columns across 3 categories)**:
*   **Debate Dynamics (+2)**: `debate_rounds_count`, `devils_advocate_challenges`
*   **Info-Gap Quality (+3)**: `info_gap_count`, `info_gap_resolved_count`, `data_quality_score`
*   **Risk Assessment (+4)**: `risk_judge_decision`, `risk_adjusted_confidence`, `aggressive_analyst_confidence`, `conservative_analyst_confidence`

### v2.1 — ML-Training-Ready BigQuery Schema (March 2026)

Expanded BigQuery `analysis_results` table from 18 → 55 columns for ML model training. Every quantitative feature that drives the recommendation is now stored as a first-class column.

**Schema Expansion (37 new columns across 6 categories)**:
*   **Financial Fundamentals (7)**: `price_at_analysis`, `market_cap`, `pe_ratio`, `peg_ratio`, `debt_equity`, `sector`, `industry`
*   **Risk Metrics (6)**: `annualized_volatility`, `var_95_6m`, `var_99_6m`, `expected_shortfall_6m`, `prob_positive_6m`, `anomaly_count`
*   **Debate Dynamics (8)**: `bull_confidence`, `bear_confidence`, `bull_thesis`, `bear_thesis`, `contradiction_count`, `dissent_count`, `recommendation_confidence`, `key_risks`
*   **Enrichment Signals (7)**: `insider_signal`, `options_signal`, `social_sentiment_score`, `nlp_sentiment_score`, `patent_signal`, `earnings_confidence`, `sector_signal`
*   **Bias & Conflict Audit (5)**: `bias_count`, `bias_adjusted_score`, `conflict_count`, `overall_reliability`, `decision_trace_count`
*   **Macro Context (4)**: `fed_funds_rate`, `cpi_yoy`, `unemployment_rate`, `yield_curve_spread`

**New Files**:
*   `migrate_bq_schema.py` — Idempotent schema migration script
*   `backend/db/bigquery_client.py` — Expanded `save_report()` (55-field row insert)
*   `backend/services/outcome_tracker.py` — Outcome evaluation with benchmark comparison

**LLM Temperature Fix**: Set `temperature=0.2` on all Gemini model calls (orchestrator + debate) to reduce score variance across runs.

### v2.0 — AI Research-Driven Upgrade (March 2026)

Based on comprehensive research analysis from Goldman Sachs, BlackRock, Morgan Stanley, Harvard, Stanford, Chicago Booth, and Wharton:

**New Backend Tools (3)**:
*   `nlp_sentiment.py` — Transformer-based sentiment via Vertex AI embeddings (Stanford ref 11)
*   `anomaly_detector.py` — Multi-dimensional anomaly detection using Z-score/IQR (Goldman ref 16)
*   `monte_carlo.py` — Monte Carlo VaR simulation engine with 1,000 GBM paths (Goldman ref 16)

**New Agent Framework (5 agents)**:
*   `debate.py` — Bull/Bear/Moderator 4-round adversarial debate (TradingAgents ref 32)
*   `trace.py` — Decision trace logger for Explainable AI (Goldman XAI ref 16)
*   `bias_detector.py` — LLM bias detection: tech favoritism, confirmation bias, recency bias (arXiv ref 33)
*   `conflict_detector.py` — Knowledge conflict detection: parametric vs real-time data (arXiv ref 33)
*   Enhanced Critic Agent — Now checks for bias patterns in addition to hallucination/logic

**New Frontend Components (4)**:
*   `DebateView.tsx` — Bull vs Bear adversarial debate visualization
*   `RiskDashboard.tsx` — Monte Carlo fan chart + VaR gauge + anomaly alerts
*   `SentimentDetail.tsx` — NLP sentiment deep-dive with keyword cloud
*   `BiasReport.tsx` — LLM bias flags + knowledge conflict table

**New Pages (1)**:
*   `/portfolio` — Position tracking, P&L, allocation, recommendation accuracy

**Pipeline Expansion**: 11 steps → 13 steps (added Agent Debate + Bias Audit)

### v1.0 — Initial Migration (February 2026)

Migrated from Streamlit to Next.js 15 + FastAPI architecture. Implemented 8 enrichment data tools, 8 enrichment LLM agents, signals API, Glass Box dashboard.
