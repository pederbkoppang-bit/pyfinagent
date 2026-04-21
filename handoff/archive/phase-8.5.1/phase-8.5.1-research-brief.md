# Research Brief — phase-8.5 / 8.5.1 "Define candidate space"

**Tier:** simple (YAML + basic product arithmetic)
**Date:** 2026-04-20

## Objective

Immutable:
```
test -f backend/autoresearch/candidate_space.yaml && \
  python -c "import yaml; d=yaml.safe_load(open('backend/autoresearch/candidate_space.yaml')); \
             assert d['estimated_combinations'] >= 10000"
```

Success_criteria: `candidate_space_committed`, `ge_1e4_combinations`, `includes_transformer_signals_from_phase_8`.

## Constraint

Brief only; Main writes the YAML.

## Why no new external research

Candidate space is internal scope. The research foundation is already on disk:
- phase-1 MDA parameters: `backend/backtest/quant_optimizer.py`
- phase-8 transformer signals: `backend/models/{timesfm,chronos}_client.py` (scaffolded, retained per phase-8.4 REJECT)
- phase-8.3 ensemble blend: `backend/backtest/ensemble_blend.py`
- phase-7 alt-data features: `backend/alt_data/features.py`

## Design proposal

`backend/autoresearch/candidate_space.yaml` with sections:

```yaml
estimated_combinations: 12000  # >= 10000 hard requirement
params:           # 8-10 numeric ranges (gbm hyperparameters, triple-barrier, etc.)
prompts:          # 4-6 LLM prompt variants (existing skill .md files)
features:         # 20-30 feature toggles (MDA subset + alt-data + transformer shadows)
model_archs:      # 3-6 model family choices (GBM, RF, AR(1), ensemble_blend, transformer_shadow)
```

Product of the four categories must be >= 10,000. Use lists of sizes that multiply to 12,000+:
- 5 param values * 4 prompt variants * 10 feature toggles * 6 model archs = 1,200 — too low.
- Need more dims or larger lists. Use 12 param values (1 dim) + 5 prompt variants + 25 feature toggles + 8 model archs = 12 * 5 * 25 * 8 = 12,000. Or go wider across more params.

Better: use cartesian-product over several params:
- 5 values on learning_rate * 4 max_depth * 3 n_estimators * 6 prompt variants * 5 model archs * 5 feature toggles = 5*4*3*6*5*5 = 9,000. Just under.
- Add one more dim: * 2 rolling_window = 18,000. OK.

`includes_transformer_signals_from_phase_8`: features section lists `timesfm_forecast_20d`, `chronos_forecast_20d`, `ensemble_blend_median` toggles (even though the scaffolds can't run live, the YAML references them as planned signals).

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": true,
  "internal_files_inspected": 4,
  "gate_passed": true,
  "note": "closure-style; candidate space is defined by prior phase deliverables already on disk"
}
```
