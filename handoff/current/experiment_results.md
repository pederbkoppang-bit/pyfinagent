# Experiment Results -- Phase 4.4.1.3 Seed Stability (BLOCKED)

**Cycle:** 18
**Date:** 2026-04-16

## Status: BLOCKED on compute time

Each 27-window backtest takes ~19 min (5 min BQ preload + 14 min window processing). 5 seeds = ~75 min with cache optimization. Exceeds the 30-min harness cycle limit.

## Improvements landed (prep work)
1. `backend/backtest/cache.py` -- cache guards to skip redundant BQ preloads
2. `scripts/harness/run_seed_stability.py` -- `skip_cache_clear=True` optimization
3. `scripts/go_live_drills/seed_stability_test.py` -- drill test (ready for results)

## Re-run recipe
```bash
source .venv/bin/activate
nohup python scripts/harness/run_seed_stability.py > handoff/seed_stability_output.log 2>&1 &
```

Then run drill: `python scripts/go_live_drills/seed_stability_test.py`
