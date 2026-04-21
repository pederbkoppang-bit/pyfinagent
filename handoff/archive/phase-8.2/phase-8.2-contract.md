# Sprint Contract — phase-8 / 8.2 (Chronos-Bolt shadow-logged feature pilot)

**Step id:** 8.2 **Cycle:** 1 **Date:** 2026-04-20 **Tier:** moderate

## Research-gate summary

6 sources in full (chronos-forecasting GitHub README + chronos_bolt.py source, amazon/chronos-bolt-small HF card, arXiv 2511.18578, AutoGluon tutorial, AWS blog), 14 URLs, three-variant queries, recency scan. `gate_passed: true`. Brief at `handoff/current/phase-8.2-research-brief.md`.

Key: install `chronos-forecasting` (not autogluon); API `BaseChronosPipeline.from_pretrained(...).predict(context=torch.Tensor, prediction_length=N)` → `[1, num_quantiles, N]` tensor; median point = `result[0, result.shape[1]//2, :]`.

## Hypothesis

Ship `backend/models/chronos_client.py` mirroring TimesFMClient pattern: lazy `chronos` + `torch` imports, fail-open, median-quantile extraction, same `shadow_log` table shape (model_name column distinguishes). 11 tests mirror 8.1 shape with a `StubPipeline` returning a numpy array of shape `(1, 9, horizon)`.

## Immutable criteria

- `python -c "import ast; ast.parse(open('backend/models/chronos_client.py').read())"`
- `python -m pytest tests/models/test_chronos_client.py -v`

## Plan

1. Write `backend/models/chronos_client.py` (~180 lines): `ChronosBoltClient(context_length=512, horizon_length=20)`; `_MODEL_NAME = "amazon/chronos-bolt-small"`; lazy `chronos` + `torch`; median-quantile extraction; `forecast`, `forecast_batch`, `shadow_log` (shares `ts_forecast_shadow_log` with TimesFM).
2. Write `tests/models/test_chronos_client.py` (11 tests matching 8.1 shape).
3. Run both immutable commands + regression.
4. Q/A. Log. Flip.

## Out of scope

- No live chronos inference (Python 3.14 + missing weights).
- No ensemble with TimesFM — that's phase-8.3.
- ASCII-only.

## References

- `handoff/current/phase-8.2-research-brief.md`
- `backend/models/timesfm_client.py` (pattern to mirror)
- `tests/models/test_timesfm_client.py` (test shape to mirror)
- `.claude/masterplan.json` → phase-8 / 8.2
