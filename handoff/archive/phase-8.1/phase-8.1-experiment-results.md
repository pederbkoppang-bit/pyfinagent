# Experiment Results — phase-8 / 8.1 (TimesFM shadow-logged feature pilot)

**Step:** 8.1 **Date:** 2026-04-20 **Cycle:** 1.

## What was built

Four new files:

1. `backend/models/__init__.py` — package marker.
2. `backend/models/timesfm_client.py` (~180 lines): `TimesFMClient` with lazy `_get_model` (fail-open on ImportError or load failure), `forecast(ts, horizon)`, `forecast_batch(tickers, horizon)`, `shadow_log(ticker, as_of_date, horizon, forecast_values, observed_values)` writing to `pyfinagent_data.ts_forecast_shadow_log`. Model: `google/timesfm-2.5-200m-pytorch`. `numpy` also lazy-imported inside forecast methods.
3. `tests/models/__init__.py` — package marker.
4. `tests/models/test_timesfm_client.py` — 11 tests exercising: default + custom init, empty-series / zero-horizon fail-open, timesfm-absent fail-open, stub-model single + batch happy paths, batch-empty, batch-per-ticker fail-open when model is None, shadow_log fail-open on bad project, ASCII-only decode.

## Verification

```
$ python -c "import ast; ast.parse(open('backend/models/timesfm_client.py').read()); print('SYNTAX OK')"
SYNTAX OK

$ python -m pytest tests/models/test_timesfm_client.py -v
collected 11 items
tests/models/test_timesfm_client.py::test_client_init_defaults PASSED
tests/models/test_timesfm_client.py::test_client_init_custom PASSED
tests/models/test_timesfm_client.py::test_forecast_empty_series_returns_empty PASSED
tests/models/test_timesfm_client.py::test_forecast_zero_horizon_returns_empty PASSED
tests/models/test_timesfm_client.py::test_forecast_without_timesfm_installed_returns_empty PASSED
tests/models/test_timesfm_client.py::test_forecast_with_stub_model PASSED
tests/models/test_timesfm_client.py::test_forecast_batch_empty_input PASSED
tests/models/test_timesfm_client.py::test_forecast_batch_without_model_returns_empty_per_ticker PASSED
tests/models/test_timesfm_client.py::test_forecast_batch_with_stub_model PASSED
tests/models/test_timesfm_client.py::test_shadow_log_fail_open_no_bq PASSED
tests/models/test_timesfm_client.py::test_module_is_ascii_only PASSED
============================== 11 passed in 2.51s ==============================

$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped
```

## Criteria check

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `ast.parse(timesfm_client.py)` | PASS | SYNTAX OK |
| 2 | `pytest tests/models/test_timesfm_client.py -v` | PASS | 11 passed in 2.51s |

## Caveats

1. **Python version mismatch.** `.venv` is 3.14; `timesfm` requires <3.12. `_get_model` catches this as a normal ImportError + fail-opens. No live forecast path is exercised. Documented in the module docstring + research brief.
2. **Live model never loaded in CI.** Tests monkeypatch `_get_model` when a non-empty forecast is needed. The `from_pretrained(...).compile(...)` glue is covered by reading (not executing).
3. **Shadow-log BQ table NOT created.** `ts_forecast_shadow_log` DDL is deferred to phase-8.3 (ensemble blend) or a separate migration. `shadow_log` fails open when the table is missing.
4. **Zero-shot equity IC expected to be weak** (arXiv 2511.18578 Nov 2025). This is the reason the pilot is shadow-only; phase-8.4 decides promotion.
5. **Model-name constant** is `google/timesfm-2.5-200m-pytorch` (the Sept 2025 release), not the 2.0 variant mentioned in the spawn prompt.

## Pre-Q/A self-check

- Both immutable criteria PASS.
- Regression 152/1 unchanged.
- No top-level `import timesfm` / `import numpy` in `timesfm_client.py` (both lazy).
- Tests pass on Python 3.14 despite `timesfm` being uninstallable there (scaffold discipline).
- `git status --short` shows only the 4 new files + handoff trio.
