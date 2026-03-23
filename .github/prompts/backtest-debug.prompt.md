---
description: "Debug walk-forward backtest failures — reads TSV experiment logs, cache stats, window-level errors, optimizer state, and result_store JSON to diagnose issues"
agent: "agent"
---
Debug a walk-forward backtest or optimizer failure. Follow this diagnostic sequence:

## 1. Gather State
- Read `backend/backtest/experiments/quant_results.tsv` (last 20 lines) for recent experiment outcomes
- Check `backend/backtest/experiments/optimizer_best.json` for current best params
- Check `backend/backtest/experiments/results/` for persisted run JSONs
- Read any Python tracebacks from the terminal

## 2. Diagnose Common Failures

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| DSR always < 0.95 | Overfitting or too few candidates | Check `min_candidates`, increase train window |
| Cache miss rate > 50% | `skip_cache_clear=True` not passed | Verify optimizer passes it to engine |
| Window N stalls | Feature build OOM or BQ timeout | Check `_build_training_data()` logs |
| "No data for ticker" | BQ tables empty | Run `/api/backtest/ingest` first |
| Sharpe = 0.0 | No trades executed | Check `min_signal_strength` threshold |
| Module-level load fails | Corrupted JSON in results/ | Validate JSON, delete bad file |

## 3. Check Key Files
- [backend/backtest/backtest_engine.py](../../backend/backtest/backtest_engine.py) — 8-step pipeline, `_report_progress()`
- [backend/backtest/quant_optimizer.py](../../backend/backtest/quant_optimizer.py) — 16 tunable params, DSR guard
- [backend/backtest/result_store.py](../../backend/backtest/result_store.py) — JSON persistence
- [backend/backtest/cache.py](../../backend/backtest/cache.py) — BQ preload cache, `get_cache_stats()`
- [backend/api/backtest.py](../../backend/api/backtest.py) — API endpoints, `_backtest_state`, `_optimizer_state`

## 4. Output
Provide: root cause, affected file(s) with line numbers, and a concrete fix.
