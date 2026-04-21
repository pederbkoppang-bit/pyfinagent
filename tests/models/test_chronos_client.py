"""phase-8.2 tests for backend.models.chronos_client.

Never loads a real chronos pipeline. StubPipeline returns a numpy array
shaped `(1, num_quantiles, horizon)`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.models.chronos_client import ChronosBoltClient, _MODEL_NAME


def test_client_init_defaults():
    c = ChronosBoltClient()
    assert c.context_length == 512
    assert c.horizon_length == 20
    assert c.model_name == _MODEL_NAME
    assert c._pipeline is None


def test_client_init_custom():
    c = ChronosBoltClient(context_length=256, horizon_length=10, model_name="alt/bolt")
    assert c.context_length == 256
    assert c.horizon_length == 10
    assert c.model_name == "alt/bolt"


def test_forecast_empty_series_returns_empty():
    c = ChronosBoltClient()
    assert c.forecast([]) == []
    assert c.forecast([1.0]) == []


def test_forecast_zero_horizon_returns_empty(monkeypatch):
    c = ChronosBoltClient()
    monkeypatch.setattr(c, "_get_pipeline", lambda: object())
    assert c.forecast([1.0, 2.0, 3.0], horizon=0) == []


def test_forecast_without_chronos_installed_returns_empty(monkeypatch):
    c = ChronosBoltClient()
    monkeypatch.setattr(c, "_get_pipeline", lambda: None)
    assert c.forecast([1.0, 2.0, 3.0, 4.0, 5.0], horizon=3) == []


def test_forecast_with_stub_pipeline(monkeypatch):
    """Stub returns a numpy array of shape (1, 9, horizon); median idx = 4."""
    import numpy as np

    class StubPipeline:
        def predict(self, *, context, prediction_length):
            return np.full((1, 9, prediction_length), 1.1)

    # stub out `import torch` inside forecast by monkeypatching sys.modules
    # with a minimal shim that has `.tensor(..., dtype=...)`.
    class _TorchShim:
        float32 = "float32"

        @staticmethod
        def tensor(data, dtype=None):
            # Chronos uses the tensor only as the `context` arg passed to
            # StubPipeline.predict, which ignores it. A list is sufficient.
            return list(data)

    monkeypatch.setitem(sys.modules, "torch", _TorchShim)
    c = ChronosBoltClient(horizon_length=5)
    monkeypatch.setattr(c, "_get_pipeline", lambda: StubPipeline())
    out = c.forecast([1.0] * 10, horizon=5)
    assert len(out) == 5
    assert all(isinstance(x, float) for x in out)
    assert out[0] == 1.1


def test_forecast_batch_empty_input():
    c = ChronosBoltClient()
    assert c.forecast_batch({}) == {}


def test_forecast_batch_without_pipeline_returns_empty_per_ticker(monkeypatch):
    c = ChronosBoltClient()
    monkeypatch.setattr(c, "_get_pipeline", lambda: None)
    out = c.forecast_batch({"AAPL": [1.0, 2.0, 3.0], "MSFT": [4.0, 5.0, 6.0]})
    assert out == {"AAPL": [], "MSFT": []}


def test_forecast_batch_with_stub_pipeline(monkeypatch):
    import numpy as np

    class _TorchShim:
        float32 = "float32"

        @staticmethod
        def tensor(data, dtype=None):
            return list(data)

    class StubPipeline:
        def predict(self, *, context, prediction_length):
            return np.full((1, 9, prediction_length), 2.2)

    monkeypatch.setitem(sys.modules, "torch", _TorchShim)
    c = ChronosBoltClient()
    monkeypatch.setattr(c, "_get_pipeline", lambda: StubPipeline())
    out = c.forecast_batch({"AAPL": [1.0] * 10, "MSFT": [1.5] * 10}, horizon=3)
    assert set(out.keys()) == {"AAPL", "MSFT"}
    assert out["AAPL"] == [2.2, 2.2, 2.2]
    assert out["MSFT"] == [2.2, 2.2, 2.2]


def test_shadow_log_fail_open_no_bq():
    c = ChronosBoltClient()
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
        / "chronos_client.py"
    )
    mod_path.read_bytes().decode("ascii")
