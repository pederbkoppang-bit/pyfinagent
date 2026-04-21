# Sprint Contract — phase-8 / 8.1 (TimesFM shadow-logged feature pilot)

**Step id:** 8.1 **Cycle:** 1 **Date:** 2026-04-20 **Tier:** moderate

## Research-gate summary

7 sources in full (TimesFM GitHub, HuggingFace model card, arXiv 2511.18578 Nov 2025, Preferred Networks blog, arXiv 2412.09880 Dec 2024, PyPI, PapersWithBacktest), 17 URLs, three-variant queries, recency scan. `gate_passed: true`. Brief at `handoff/current/phase-8.1-research-brief.md`.

Key findings influencing the contract:
1. **Python 3.14 in venv; TimesFM requires `<3.12`.** `timesfm` cannot be installed in this venv; scaffold must fail-open on `ImportError`.
2. **Model name:** `google/timesfm-2.5-200m-pytorch` (Sept 2025 release).
3. **Forecast API:** `model.forecast(horizon=N, inputs=[np.array, ...])` — list of arrays per ticker.
4. **Zero-shot equity IC is weak by design.** arXiv Nov 2025: zero-shot TimesFM underperforms AR(1). This is EXPECTED for a shadow-only pilot; phase-8.4 decides promotion vs rejection.
5. **Test collection:** `tests/` root; new `tests/models/__init__.py` required; mirror `sys.path.insert` pattern from `tests/test_retired_models.py`.

## Hypothesis

Ship a fail-open `TimesFMClient` scaffold + 5 pytest cases that exercise the API surface without loading a real model. Tests pass on a `<3.12` system (where timesfm is available) AND on Python 3.14 (where timesfm is absent) — discipline: never import timesfm at module top.

## Immutable criteria

- `python -c "import ast; ast.parse(open('backend/models/timesfm_client.py').read())"`
- `python -m pytest tests/models/test_timesfm_client.py -v`

## Plan

1. Create `backend/models/__init__.py` + `backend/models/timesfm_client.py` (~180 lines).
   - `_MODEL_NAME = "google/timesfm-2.5-200m-pytorch"`.
   - `TimesFMClient(context_length=512, horizon_length=20)`.
   - `forecast(ts, *, horizon=None)` → list[float]; fail-open `[]` on missing timesfm.
   - `forecast_batch(tickers, horizon=20)` → dict[str, list[float]]; fail-open.
   - `shadow_log(ticker, as_of_date, horizon, forecast_values, observed_values=None)` → fail-open BQ write to `pyfinagent_data.ts_forecast_shadow_log` (table creation NOT attempted by scaffold; logged warning on failure).
2. Create `tests/models/__init__.py` (empty) + `tests/models/test_timesfm_client.py` (5 tests).
3. Run both immutable commands.
4. Run full regression.
5. Write experiment-results, spawn Q/A, log-last, flip.

## Out of scope

- No live timesfm inference (Python version mismatch + weights download out of scope).
- No ts_forecast_shadow_log table creation (phase-8.3 smoketest or a separate migration).
- No fine-tuning / LoRA.
- ASCII-only.

## References

- `handoff/current/phase-8.1-research-brief.md`
- `tests/test_retired_models.py` (sys.path.insert pattern for tests/ collection)
- `.claude/masterplan.json` → phase-8 / 8.1
