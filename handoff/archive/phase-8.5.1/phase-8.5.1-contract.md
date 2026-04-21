# Sprint Contract — phase-8.5 / 8.5.1 (Define candidate space)

**Step id:** 8.5.1 **Cycle:** 1 **Date:** 2026-04-20 **Tier:** simple

## Research-gate summary

Closure-style brief at `handoff/current/phase-8.5.1-research-brief.md` (references 4 prior phase deliverables: MDA params, transformer scaffolds, ensemble blender, alt-data features). `gate_passed: true`.

## Hypothesis

Write `backend/autoresearch/candidate_space.yaml` with 5+ dimensions whose cartesian product is >= 10,000. Includes `timesfm_forecast_20d`, `chronos_forecast_20d`, `ensemble_blend_median` as feature toggles (satisfies `includes_transformer_signals_from_phase_8`).

Design: param_ranges (learning_rate x 5, max_depth x 4, n_estimators x 3, rolling_window x 2) * prompts (5) * features_set (5) * model_archs (5) = 5*4*3*2*5*5*5 = 15,000.

## Immutable criterion

- `test -f backend/autoresearch/candidate_space.yaml && python -c "import yaml; d=yaml.safe_load(open('backend/autoresearch/candidate_space.yaml')); assert d['estimated_combinations'] >= 10000"`

## Plan

1. Create `backend/autoresearch/__init__.py` (package marker).
2. Write `backend/autoresearch/candidate_space.yaml`:
   - Top-level: `estimated_combinations: 15000`, `params`, `prompts`, `features`, `model_archs`, `includes_transformer_signals: true`, `notes`.
3. Verify both immutable assertions + regression.
4. Q/A, log, flip.

## Out of scope

- No code — YAML only (plus a package marker).
- No optimizer wiring (phase-8.5.3+).
- ASCII-only.
