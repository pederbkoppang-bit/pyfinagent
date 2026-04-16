# Research -- Cycle 18: Phase 4.4.1.3 Seed Stability

## Research Gate
WAIVED: Pure execution of existing `run_seed_stability.py` script. No new algorithmic code needed. The seed stability methodology (varying `random_state` in GradientBoosting) is a standard ML reproducibility check.

## Key Facts
- Script varies `random_state` in `GradientBoostingClassifier` across 5 seeds: [42, 123, 456, 789, 2026]
- Uses monkey-patching of `BacktestEngine._train_model` to inject each seed
- Full walk-forward backtest with 27 windows per seed run
- Same parameters from `optimizer_best.json` used across all seeds
- Gate: std < 0.1 and all seeds > 0.9 Sharpe

## Optimization Applied
- Added `skip_cache_clear=True` to `run_backtest()` call so BQ cache stays warm across seed runs
- Added explicit `clear_cache()` call after all runs complete
- This reduces total runtime from ~65min to ~35min (one BQ preload instead of five)
