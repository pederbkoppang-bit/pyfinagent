---
paths:
  - "backend/backtest/**"
---

# Backtest Engine — Walk-Forward ML Conventions

## Architecture
- Research-driven walk-forward backtesting: López de Prado, TradingAgents, FinRL, Fama-French
- Two-regime: quant-only ($0 LLM cost) for historical, full 20-agent pipeline for live
- Download once, replay forever: BigQuery stores all historical data (3 tables)
- `GradientBoostingClassifier(n_estimators=200, max_depth=4, min_samples_leaf=20)`

## Key Modules
- `backtest_engine.py` — Central orchestrator, 5 strategies in STRATEGY_REGISTRY, MDA cache
- `walk_forward.py` — Expanding-window scheduler with configurable train/test/embargo
- `historical_data.py` — ~49-feature vector builder (point-in-time), fractional differentiation
- `candidate_selector.py` — S&P 500 screening at historical dates (composite score)
- `backtest_trader.py` — In-memory portfolio simulator (inverse-vol sizing)
- `analytics.py` — Sharpe, Deflated Sharpe Ratio (DSR), baselines (SPY/equal-weight/momentum)
- `quant_optimizer.py` — Autoresearch-style strategy optimizer (16 tunable params)
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
- **Bulk preload cache**: `preload_prices(tickers, start, end)` + `preload_fundamentals(tickers)` at backtest start — 2 BQ queries instead of ~50,000
- **Structured progress**: `_report_progress(step, detail, **kwargs)` emits dict with window/step/counts/elapsed/cache_stats — not plain strings. Always pass `window=self._current_window_id` (set at start of `_run_window()`).
- **8-step pipeline**: `preloading → screening → building_features → training → computing_mda → predicting → trading → finalizing`. Emit `progress_cb` for `finalizing` in `_run_backtest_async()` before AND after baseline computation so the UI never freezes after engine returns.
- **`_current_window_id`**: Field on `BacktestEngine`, updated at top of `_run_window()`. Required for correct window numbers in throttled `_build_training_data()` progress calls.
- Feature drift detection: log top-5 MDA changes between iterations
- Model staleness: warn when trained model >7 days old
