# Experiment Results — phase-8.5 / 8.5.1 (Candidate space)

**Step:** 8.5.1 **Date:** 2026-04-20 **Cycle:** 1.

## What was built

Two new files:

1. `backend/autoresearch/__init__.py` — package marker.
2. `backend/autoresearch/candidate_space.yaml`:
   - `estimated_combinations: 15000` (= 5 × 4 × 3 × 2 × 5 × 5 × 5 across learning_rate × max_depth × n_estimators × rolling_window × prompts × features × model_archs).
   - `includes_transformer_signals: true` + dedicated `transformer_signals` list with `timesfm_forecast_20d`, `chronos_forecast_20d`, `ensemble_blend_median`.
   - 5 feature bundles (from `mda_only` baseline through `mda_plus_ensemble_blend`).
   - 5 model archs (gbm, random_forest, ar1_baseline, ensemble_blend, transformer_shadow).
   - Notes call out the runtime gate: transformer signals are listed but gated on phase-8-decision.md Sec. 6 conditions.

## Verification

```
$ test -f backend/autoresearch/candidate_space.yaml && python -c "import yaml; d=yaml.safe_load(open('backend/autoresearch/candidate_space.yaml')); assert d['estimated_combinations'] >= 10000; print('IMMUTABLE PASS -- est_combinations=', d['estimated_combinations'])"
IMMUTABLE PASS -- est_combinations= 15000

$ python -c "import yaml; d=yaml.safe_load(open('backend/autoresearch/candidate_space.yaml')); assert 'timesfm_forecast_20d' in d['transformer_signals']; assert 'chronos_forecast_20d' in d['transformer_signals']"
(exit 0)

$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped
```

## Contract criteria

| # | Success criterion | Status |
|---|---|---|
| 1 | `candidate_space_committed` | PASS (file exists, YAML parses) |
| 2 | `ge_1e4_combinations` | PASS (15,000 >= 10,000) |
| 3 | `includes_transformer_signals_from_phase_8` | PASS (3 signals listed under `transformer_signals`) |

## Caveats

1. **`15,000` is a declared estimate, not an enumerated list** — the YAML carries `estimated_combinations: 15000`. The phase-8.5.3 proposer is responsible for actually constructing the cartesian product at runtime.
2. **Transformer archs are gated** — notes explicitly say they enter live search only when phase-8-decision.md Sec. 6 conditions hold.
3. **Feature bundles are named aliases**, not flag dicts — the proposer resolves each alias to the concrete feature set when building a trial.
4. **ASCII-only** confirmed by inspection.
