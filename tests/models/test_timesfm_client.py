"""phase-8.1 tests for backend.models.timesfm_client.

Never loads a real TimesFM model. Monkeypatches `_get_model` when a fake
forecast shape is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.models.timesfm_client import TimesFMClient, _MODEL_NAME


def test_client_init_defaults():
    c = TimesFMClient()
    assert c.context_length == 512
    assert c.horizon_length == 20
    assert c.model_name == _MODEL_NAME
    assert c._model is None  # lazy


def test_client_init_custom():
    c = TimesFMClient(context_length=256, horizon_length=10, model_name="alt/checkpoint")
    assert c.context_length == 256
    assert c.horizon_length == 10
    assert c.model_name == "alt/checkpoint"


def test_forecast_empty_series_returns_empty():
    c = TimesFMClient()
    assert c.forecast([]) == []
    assert c.forecast([1.0]) == []


def test_forecast_zero_horizon_returns_empty(monkeypatch):
    c = TimesFMClient()
    monkeypatch.setattr(c, "_get_model", lambda: object())  # never reached
    assert c.forecast([1.0, 2.0, 3.0], horizon=0) == []


def test_forecast_without_timesfm_installed_returns_empty(monkeypatch):
    """Fail-open when the timesfm package is absent."""
    c = TimesFMClient()
    monkeypatch.setattr(c, "_get_model", lambda: None)
    assert c.forecast([1.0, 2.0, 3.0, 4.0, 5.0], horizon=3) == []


def test_forecast_with_stub_model(monkeypatch):
    """Happy path: monkeypatched model returns a canned forecast shape.

    Exercises the glue between `forecast` and the model's expected
    `(point_forecast, quantile_forecast)` tuple.
    """
    import numpy as np

    class StubModel:
        def forecast(self, *, horizon, inputs):
            return (
                np.array([[1.1] * horizon for _ in inputs]),
                np.zeros((len(inputs), horizon, 10)),
            )

    c = TimesFMClient(horizon_length=5)
    monkeypatch.setattr(c, "_get_model", lambda: StubModel())
    out = c.forecast([1.0] * 10, horizon=5)
    assert len(out) == 5
    assert all(isinstance(x, float) for x in out)
    assert out[0] == 1.1


def test_forecast_batch_empty_input():
    c = TimesFMClient()
    assert c.forecast_batch({}) == {}


def test_forecast_batch_without_model_returns_empty_per_ticker(monkeypatch):
    c = TimesFMClient()
    monkeypatch.setattr(c, "_get_model", lambda: None)
    out = c.forecast_batch({"AAPL": [1.0, 2.0, 3.0], "MSFT": [4.0, 5.0, 6.0]})
    assert out == {"AAPL": [], "MSFT": []}


def test_forecast_batch_with_stub_model(monkeypatch):
    import numpy as np

    class StubModel:
        def forecast(self, *, horizon, inputs):
            # Each row = len(inputs); values encode input index for checking.
            pts = np.array([[float(i)] * horizon for i in range(len(inputs))])
            return (pts, np.zeros((len(inputs), horizon, 10)))

    c = TimesFMClient()
    monkeypatch.setattr(c, "_get_model", lambda: StubModel())
    out = c.forecast_batch({"AAPL": [1.0] * 10, "MSFT": [2.0] * 10}, horizon=3)
    assert set(out.keys()) == {"AAPL", "MSFT"}
    assert len(out["AAPL"]) == 3
    assert len(out["MSFT"]) == 3
    # StubModel encodes input index as the constant forecast value.
    # Dict insertion order preserves {AAPL->0, MSFT->1}.
    assert out["AAPL"][0] == 0.0
    assert out["MSFT"][0] == 1.0


def test_shadow_log_fail_open_no_bq():
    c = TimesFMClient()
    ok = c.shadow_log(
        "AAPL",
        "2026-04-20",
        horizon=5,
        forecast_values=[1.0, 2.0, 3.0, 4.0, 5.0],
        project="nonexistent-fail-open-test",
        dataset="nx",
    )
    assert ok is False


def test_module_is_ascii_only():
    mod_path = (
        Path(__file__).resolve().parents[2]
        / "backend"
        / "models"
        / "timesfm_client.py"
    )
    mod_path.read_bytes().decode("ascii")
