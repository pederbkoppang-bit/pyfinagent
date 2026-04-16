# Contract -- Phase 4.4.1.3 Sharpe stable across 5 random seeds

**Cycle:** 18
**Date:** 2026-04-16
**Item:** 4.4.1.3

## Target
Verify that the backtest strategy is not dependent on a specific random initialization by running with 5 different seeds and checking Sharpe std < 0.1.

## Success Criteria
1. Run `scripts/harness/run_seed_stability.py` with seeds [42, 123, 456, 789, 2026]
2. All 5 seeds produce Sharpe > 0.9
3. Standard deviation of Sharpe values across 5 seeds < 0.1
4. Results saved to `handoff/seed_stability_results.json` and individual files in `experiments/results/`
5. Drill test at `scripts/go_live_drills/seed_stability_test.py` verifies results (stdlib-only, exit 0 on PASS)
6. Checklist item 4.4.1.3 flipped to [x] with evidence line

## Approach
- Optimize `run_seed_stability.py` to use `skip_cache_clear=True` for BQ cache efficiency
- Run full backtest with each seed (monkey-patches `_train_model` random_state)
- Write stdlib-only drill test verifying seed stability results
- Research gate WAIVED: pure execution of existing script, no new algorithmic code
