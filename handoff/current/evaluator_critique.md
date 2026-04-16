# Evaluator Critique -- Phase 4.4.1.3 Seed Stability (BLOCKED)

**Cycle:** 18 (Ford, 2026-04-16)

## Verdict: BLOCKED -- compute time exceeds cycle limit

### Criteria Assessment

| Criterion | Score | Notes |
|---|---|---|
| Tractability | N/A | 5-seed test requires ~75 min; harness cycle allows ~30 min |
| Prep work | DONE | Cache guards, script optimization, drill test all landed |
| Item completion | BLOCKED | Cannot flip checklist `[ ]` to `[x]` without full results |

### Analysis

The seed stability test (`scripts/harness/run_seed_stability.py`) runs the full 27-window walk-forward backtest 5 times with different random seeds. Each run takes ~19 minutes (5 min BQ preload + 14 min window processing). Even with the BQ cache optimization (which eliminates redundant preloads for seeds 2-5), total runtime is ~75 minutes.

### Improvements Committed
1. **BQ cache guards** (`cache.py`): Skip redundant preload queries when data is already in memory. This is a general performance improvement that benefits all back-to-back backtest runs (optimizer iterations, seed tests, etc.).
2. **Script optimization**: `skip_cache_clear=True` prevents cache clearance between seeds.
3. **Drill test**: `seed_stability_test.py` ready to verify results once generated.

### Recommendation
Run the seed stability test as a standalone job outside the harness cycle:
```bash
source .venv/bin/activate
nohup python scripts/harness/run_seed_stability.py > handoff/seed_stability_output.log 2>&1 &
```
