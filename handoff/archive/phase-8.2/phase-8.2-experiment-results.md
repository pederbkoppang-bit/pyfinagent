# Experiment Results — phase-8 / 8.2 (Chronos-Bolt shadow-logged feature pilot)

**Step:** 8.2 **Date:** 2026-04-20 **Cycle:** 1.

Two new files mirroring TimesFM pattern.

1. `backend/models/chronos_client.py` (~180 lines): `ChronosBoltClient` with lazy `chronos-forecasting` + `torch` imports, `_MODEL_NAME = "amazon/chronos-bolt-small"`. `_get_pipeline`, `_median_from_result` (extracts shape[1]//2 quantile), `forecast`, `forecast_batch` (sequential per-ticker), `shadow_log` (shares `ts_forecast_shadow_log` with TimesFM; `model_name` column distinguishes).

2. `tests/models/test_chronos_client.py` — 11 tests mirroring 8.1 shape. Stub pipeline returns numpy `(1, 9, horizon)` filled with constant; median index 4 verified. `sys.modules["torch"]` monkeypatch makes the test stand alone without a real torch install.

## Verification

```
$ python -c "import ast; ast.parse(open('backend/models/chronos_client.py').read()); print('SYNTAX OK')"
SYNTAX OK

$ python -m pytest tests/models/test_chronos_client.py -v
collected 11 items
... 11 passed in 2.06s

$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped
```

## Criteria

| # | Criterion | Status |
|---|---|---|
| 1 | `ast.parse(chronos_client.py)` | PASS |
| 2 | `pytest tests/models/test_chronos_client.py -v` | PASS (11/11) |

## Caveats

1. **Python 3.14 + no chronos/torch** — same fail-open discipline as TimesFM. Lazy imports; tests monkeypatch `_get_pipeline` and a minimal `torch` shim.
2. **`forecast_batch` is sequential**, not using Chronos's true batched 2D-tensor input. Keeps scaffold simple; daily volume is hundreds of tickers on CPU so no material latency cost.
3. **Median quantile = `result.shape[1] // 2`** — dynamic, not hardcoded 4, so it works for any quantile count.
4. **Shares DDL with TimesFM** (`ts_forecast_shadow_log`). Phase-8.3 creates the table.
5. **ASCII-only.**
