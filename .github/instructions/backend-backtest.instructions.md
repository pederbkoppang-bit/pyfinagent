---
applyTo: "backend/backtest/**"
---

# Backtest Engine — Walk-Forward ML Conventions

## Architecture
- Research-driven walk-forward backtesting: López de Prado, TradingAgents, FinRL, Fama-French
- Two-regime: quant-only ($0 LLM cost) for historical, full 20-agent pipeline for live
- Download once, replay forever: BigQuery stores all historical data (3 tables)
- `GradientBoostingClassifier(n_estimators=200, max_depth=4, min_samples_leaf=20)`

## Key Modules
- `backtest_engine.py` — Central orchestrator, 5 strategies in STRATEGY_REGISTRY, MDA cache; daily mark-to-market loop (`pd.bdate_range`) produces ~500+ NAV snapshots per backtest for valid Sharpe annualization; calls `trader.full_reset()` before each run for optimizer iteration independence; `skip_cache_clear` param allows optimizer to keep warm BQ cache across iterations; `stop_check: Callable[[], bool] | None` attribute checked between windows in `run_backtest()` — enables mid-backtest interruption from optimizer or UI
- `walk_forward.py` — Expanding-window scheduler with configurable train/test/embargo
- `historical_data.py` — ~49-feature vector builder (point-in-time), fractional differentiation
- `candidate_selector.py` — S&P 500 screening at historical dates (composite score)
- `backtest_trader.py` — In-memory portfolio simulator (inverse-vol sizing); per-trade commission tracking (`flat_pct` + `per_share` models); `Trade` dataclass includes `commission` field; `reset()` clears positions only (between windows); `full_reset()` resets all state (cash/positions/trades/snapshots/commission) for optimizer iteration independence (Bailey & LdP 2014)
- `analytics.py` -- Sharpe (frequency-aware: `periods_per_year` param, default 252 for daily; Lo 2002 / Sharpe 1994), Deflated Sharpe Ratio (DSR), baselines (SPY/equal-weight/momentum with real Sharpe from daily returns), `compute_round_trips()` (FIFO BUY→SELL matching), `compute_trade_statistics()` (23-field: profit_factor/win_rate/expectancy/SQN/streaks/cost metrics)
- `quant_optimizer.py` -- Autoresearch-style strategy optimizer (17 tunable params including mr_holding_days); per-run UUID `run_id` tagging, step-level progress (`establishing_baseline`/`baseline_complete`/`running_experiment`/`evaluated`), passes `skip_cache_clear=True`, explicit `bq_cache.clear_cache()` at end of `run_loop()`; `_log_experiment()` serializes `trial_params` (not `best_params`) so each TSV row shows the actual tested params; logs `params_json` column to TSV; dual-source warm-start: (1) `optimizer_best.json`, (2) `result_store.load_latest()` fallback for standalone backtest results; LLM proposals load `quant_strategy.md` skill for research-backed guidance; wires `stop_check` to `engine.stop_check` at start of `run_loop()` + pre-baseline guard
- `result_store.py` — JSON-on-disk persistence for backtest results at `experiments/results/{timestamp}_{run_id}.json`; save/load/list/delete helpers; auto-loaded on API module init
- `cache.py` — BQ bulk-preload cache (`preload_prices` + `preload_fundamentals` = 2 queries for entire backtest), hit/miss stats via `get_cache_stats()`
- `data_ingestion.py` — Bulk ingest prices/fundamentals/macro from yfinance + FRED

## 5 Strategies
| Strategy | Research Basis |
|----------|---------------|
| `triple_barrier` | López de Prado Ch. 3 |
| `quality_momentum` | Asness et al. 2019 |
| `mean_reversion` | Lo & MacKinlay 1990 |
| `factor_model` | Fama-French 5-factor |
| `meta_label` | López de Prado Ch. 3 (future secondary model) |

## Conventions
- **No future leakage**: Walk-forward expanding windows with 5-day embargo
- **No LLM contamination**: Historical regime uses only quant features (LLMs "know" outcomes)
- **DSR guard**: Deflated Sharpe Ratio ≥ 0.95 required for optimizer to keep a result
- **MDA→Agent bridge**: Feature importance maps to responsible agents via `FEATURE_TO_AGENT`
- **Bulk preload cache**: `preload_prices(tickers + ["SPY"], start, end)` + `preload_fundamentals(tickers)` at backtest start -- SPY always preloaded alongside universe for baseline benchmarks
- **Baseline Sharpe**: `compute_baseline_strategies()` returns daily-return-based Sharpe for SPY, Equal Weight, and Momentum baselines using `compute_sharpe()`. `generate_report()` passes real Sharpe values through to the frontend.
- **Daily mark-to-market**: `_run_window()` trading step iterates every business day in the test window via `pd.bdate_range()`. Each day fetches close prices from cache and calls `trader.mark_to_market()`. This produces ~60-90 NAV snapshots per window (~500+ total), making sqrt(252) annualization statistically valid (Lo 2002, Sharpe 1994). Positions still close at test_end.
- **Frequency-aware Sharpe**: `compute_sharpe(returns, risk_free_rate, periods_per_year=252)` uses `sqrt(periods_per_year)` annualization. Default 252 (daily) is backward-compatible. Pass `periods_per_year=12` for monthly returns, `52` for weekly.
- **Trader dual-reset**: `reset()` clears positions only (between windows, preserves NAV history). `full_reset()` resets everything (cash, positions, trades, snapshots, commission) -- called at start of `run_backtest()` so each optimizer iteration is independent (Bailey & LdP 2014 IID requirement).
- **Trade pipeline**: `BacktestTrader.trades[]` -> `BacktestResult.all_trades` (capped at 500) -> `generate_report()` calls `compute_round_trips()` (FIFO matching) + `compute_trade_statistics()` -> report **always** includes `trades` + `trade_statistics` keys (empty list/dict when no round-trips). `WindowResult.num_trades` counts actual executed trades (BUY+SELL via trader), not ML signal count.
- **Commission models**: `flat_pct` (percentage of notional, default) or `per_share` ($0.005/share, $1.00 min). Configurable via `backtest_commission_model` and `backtest_commission_per_share` in settings. Commission recorded per-trade and surfaced in trade statistics (total, % of profit, avg cost/trade, break-even win rate).
- **`mr_holding_days`**: Separate holding period for mean reversion strategy (default 15, range 5-30). MR works at short horizons (Lo & MacKinlay 1990) -- shared `holding_days` (30-252) is too long.
- **Structured progress**: `_report_progress(step, detail, **kwargs)` emits dict with window/step/counts/elapsed/cache_stats — not plain strings. Always pass `window=self._current_window_id` (set at start of `_run_window()`).
- **8-step pipeline**: `preloading → screening → building_features → training → computing_mda → predicting → trading → finalizing`. The `finalizing` step spans baseline comparison + report generation in `_run_backtest_async()`. Must emit `progress_cb` for `finalizing` BEFORE and AFTER baseline computation so the UI never freezes.
- **`_current_window_id`**: Maintained as `self._current_window_id: int` in `BacktestEngine.__init__`, updated at the top of `_run_window()`. Required so throttled progress calls in `_build_training_data()` can include the correct window number.
- **`skip_cache_clear`**: `run_backtest(skip_cache_clear=False)` — when `True`, skips `cache.clear_cache()` at end of run. Optimizer uses `True` to keep warm BQ cache across iterations (drops per-experiment time from ~5-10min to <30s). Standalone backtests use default `False`.
- **Run tagging**: Each optimizer run generates a short UUID `run_id` (`self._run_id`). ALL `_log_experiment()` calls must pass `self._run_id` as the first argument — BASELINE rows and experiment rows share the same `run_id`. The API filters by `run_id` to show only the current run's results. Never pass literal `"BASELINE"` or `uuid.uuid4()[:8]` as run_id.
- **TSV write safety**: `_log_experiment()` is wrapped in try/except (logs at ERROR, does NOT re-raise). Always `f.flush()` after write. Do NOT have `quant_results.tsv` open in an editor while optimizer runs — editor buffer conflicts can overwrite appended rows.
- **Run grouping**: Each BASELINE row marks start of a new optimizer run. `GET /optimize/runs` returns per-run summaries (baseline_ts, baseline_sharpe, experiment_count, kept, discarded). `GET /optimize/experiments?run_index=0` returns latest run only. Index 0=latest, 1=second-latest, etc.
- **Clear history**: `DELETE /api/backtest/optimize/history` removes `quant_results.tsv`, `optimizer_best.json`, and all `experiments/results/*.json` files, then invalidates API cache (including `backtest:runs*`). Use after code changes that alter Sharpe/DSR calculation to avoid warm-starting from stale baselines.
- **Step progress**: Optimizer reports `current_step` + `current_detail` to `status_cb`. Engine's `progress_callback` is forwarded into optimizer state for sub-step visibility (e.g., "W3/8 training: Building training data").
- **Result persistence**: `result_store.py` saves backtest results as JSON to `experiments/results/`. API auto-loads latest on startup so previous results display immediately. `list_runs()` returns summary metadata for the run history selector.
- **Optimizer warm-start (dual-source)**: `_load_previous_best()` checks two sources in order: (1) `optimizer_best.json` (optimizer's own saved best), (2) `result_store.load_latest()` (most recent standalone backtest). Source 2 merges `strategy_params` into `best_params` and seeds `best_sharpe`/`best_dsr` from `analytics`. Sets `_warm_started = True` so `run_loop()` skips redundant baseline. TSV logs `params_json` column (full strategy params per experiment) for Insights slice plots.
- Feature drift detection: log top-5 MDA changes between iterations
- **ASCII-only logger calls**: Never use Unicode characters (arrows, em dashes, etc.) in `logger.*()` calls. On Windows, uvicorn handlers use cp1252 encoding which crashes on non-ASCII. Use `--` not `\u2014`, `->` not `\u2192`, `improved` not `\u2191`.
- Model staleness: warn when trained model >7 days old
