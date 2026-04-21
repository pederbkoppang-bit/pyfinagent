# Sprint Contract — phase-8 / 8.3 (Ensemble blend w/ nested walk-forward CV)

**Step id:** 8.3 **Cycle:** 1 **Date:** 2026-04-20 **Tier:** moderate

## Research-gate summary

6 sources in full (Cochrane nested CV, arXiv 2511.15350 stacking 2025, purged-CV de Prado, sklearn Ledoit-Wolf docs x2, AFML notes), 16 URLs, three-variant queries, recency scan. `gate_passed: true`. Brief at `handoff/current/phase-8.3-research-brief.md`.

Key findings:
- Nested walk-forward + strict chronological order; purge+embargo mandatory for financial labels.
- Equal-weight is the safest default with small n_splits; correlation-weighted + shrinkage as upgrade paths.
- Ledoit-Wolf has a closed-form implementable in pure Python.
- MDA is feature importance (not a per-date signal) — blender consumes GBM `predict_proba` scores reduced from the forecast horizon.
- Stacking reserved for phase-8.4.

## Hypothesis

Ship `backend/backtest/ensemble_blend.py` (~280 lines) with `EnsembleBlender` + 3 weighting modes (equal/correlation/shrinkage), nested walk-forward CV loop with purge+embargo, pure-Python Ledoit-Wolf shrinkage, pure-Python IC + ICIR. Pytest coverage in `tests/models/test_ensemble_blend.py`.

## Immutable criteria

- `python -c "import ast; ast.parse(open('backend/backtest/ensemble_blend.py').read())"`
- `python scripts/harness/run_harness.py --dry-run --cycles 1`

## Plan

1. Write `backend/backtest/ensemble_blend.py`:
   - `EnsembleBlender(component_names=('mda','timesfm','chronos'), weighting_method='equal', lookback_days=252, purge_days=0, embargo_days=5)`.
   - `blend(signals_by_component)` — weighted average.
   - `fit_weights(historical_signals, forward_returns)` — chooses between equal / correlation / shrinkage.
   - `_walk_forward_splits(n, n_splits=5, purge, embargo)` — yields (train_idx, test_idx).
   - `_compute_ic(signal, ret)` — Pearson.
   - `_ledoit_wolf_shrinkage(X)` — closed-form pure-Python.
   - Fail-open on empty / misshapen inputs.
2. Write `tests/models/test_ensemble_blend.py` (8-10 tests, similar shape to 8.1 / 8.2).
3. Run both immutable commands + regression.
4. Q/A, log, flip.

## Out of scope

- No real MDA/TimesFM/Chronos data — stub inputs in tests.
- No stacking (phase-8.4).
- No BQ write (ensemble output stays in memory for now).
- ASCII-only.

## References

- `handoff/current/phase-8.3-research-brief.md`
- `backend/backtest/backtest_engine.py:305-347` (MDA anchor)
- `backend/models/timesfm_client.py`, `backend/models/chronos_client.py`
- `.claude/masterplan.json` → phase-8 / 8.3
