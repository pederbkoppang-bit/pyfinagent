# Experiment Results — phase-8 / 8.3 (Ensemble blend + nested walk-forward CV)

**Step:** 8.3 **Date:** 2026-04-20 **Cycle:** 1.

## What was built

Two new files.

1. `backend/backtest/ensemble_blend.py` (~290 lines):
   - `EnsembleBlender(component_names=('mda','timesfm','chronos'), weighting_method='equal'|'correlation'|'shrinkage', lookback_days=252, purge_days=0, embargo_days=5, n_splits=5)`.
   - `_pearson` + `_compute_ic` — pure-Python Pearson IC.
   - `_ledoit_wolf_shrinkage(x)` — closed-form pure-Python Ledoit-Wolf shrinkage (Eq. 10-14).
   - `_minimum_variance_weights(cov)` — Gauss-Jordan inverse + simplex clamp.
   - `_walk_forward_splits(n)` — chronological purge+embargo folds.
   - `fit_weights(historical_signals, forward_returns)` — chooses equal / correlation / shrinkage.
   - `blend(signals_by_component)` — weighted average across components; unknown components dropped; missing-key-per-component handled with weight renormalization.
   - `cv_ic(...)` — nested walk-forward IC + IR.
   - Fail-open everywhere; pure Python (no numpy/sklearn/scipy).

2. `tests/models/test_ensemble_blend.py` — 15 tests covering init validation, Pearson math, correlation-weighted IC rewarding, equal-weight fallback, blend semantics (equal, missing key, unknown component), chronology of walk-forward, Ledoit-Wolf shape + bounds, shrinkage simplex weights, cv_ic shape, ASCII.

## Verification

```
$ python -c "import ast; ast.parse(open('backend/backtest/ensemble_blend.py').read()); print('SYNTAX OK')"
SYNTAX OK

$ python -m pytest tests/models/test_ensemble_blend.py -v
collected 15 items
... 15 passed in 0.02s

$ python scripts/harness/run_harness.py --dry-run --cycles 1
... HARNESS COMPLETE -- 1 cycles finished
... Final best: Sharpe=1.1705, DSR=0.9526
EXIT=0

$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped
```

## Criteria

| # | Criterion | Status |
|---|---|---|
| 1 | `ast.parse(ensemble_blend.py)` | PASS |
| 2 | `harness --dry-run --cycles 1` | PASS (exit 0) |

Supplementary pytest covers the blender in detail: 15/15 green.

## Design notes from the research brief honored

- **Nested walk-forward** with strict chronology (train < test always).
- **Purge + 5-day embargo** defaults (de Prado AFML Ch. 7).
- **Equal weight is the default** — safest with small n_splits.
- **Correlation-weighted**: `w_i = |IC_i| / sum(|IC_j|)` rewards high-IC components.
- **Shrinkage**: Ledoit-Wolf closed form toward identity-scaled target (sklearn formula, pure-Python port) + simplex-clamped minimum-variance weights.
- **Pure Python math** — no numpy/sklearn/scipy at module top.

## Caveats

1. **No real MDA/TimesFM/Chronos signals fed to the blender this cycle.** Tests use synthetic vectors. Wiring to the actual `backend/backtest/backtest_engine.py` predict_proba + `models/*_client.py::forecast_batch` outputs is phase-8.4 territory or beyond.
2. **Stacking reserved for later.** The brief flagged stacking as a phase-8.4 topic; this cycle ships equal/correlation/shrinkage only.
3. **`_minimum_variance_weights` uses simplex-clamp**, not true constrained MVO (quadprog). For a 3-component shadow blend this is fine; swap to a proper solver if N components grows past ~10.
4. **BQ write not added** — ensemble output is in-memory dict. Phase-8.4 decides if we persist.
5. **ASCII-only** — module + tests decode clean.

## Pre-Q/A self-check

- Both immutable criteria PASS.
- 15 new tests + existing 152 backend tests all green.
- `git status --short` shows only 2 new files + handoff trio.
- Harness dry-run unchanged (Sharpe 1.1705, DSR 0.9526 — baseline MDA stable).
