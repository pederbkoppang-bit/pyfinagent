"""phase-80.1 -- /api/signals/{ticker} must not 500 on a non-finite float.

Measured 2026-07-25: the endpoint returned HTTP 500 for EVERY ticker because
`backend/tools/sector_analysis.py:34` emitted `stock_returns["1mo"] = nan` and
Starlette renders with `json.dumps(allow_nan=False)`.

The tests reproduce the EXACT reported failure chain
(`dict item "1mo"` -> `"stock_returns"` -> `"sector"`) with the tool layer
monkeypatched, so they are deterministic and never touch the network.

THE THREE ASSERTIONS MATTER INDIVIDUALLY. A test that only asserts "no 500"
is satisfied by replacing NaN with 0.0 -- a silent-wrong-number regression,
not a fix -- and by dropping the key entirely, which hides a data outage
behind a missing field. So every endpoint test asserts:
    1. status 200
    2. all 12 signal keys present
    3. the period key is PRESENT *and* its value is None
Mutation-tested; see handoff/current/experiment_results_80.1.md for the matrix.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.api._json_safe import NaNSafeJSONResponse, sanitize_non_finite

SIGNAL_KEYS = (
    "insider",
    "options",
    "social_sentiment",
    "patent",
    "earnings_tone",
    "fred_macro",
    "alt_data",
    "sector",
    "nlp_sentiment",
    "anomalies",
    "monte_carlo",
    "quant_model",
)


# ---------------------------------------------------------------------------
# Unit level -- the sanitiser itself
# ---------------------------------------------------------------------------


def test_nan_becomes_none_and_the_key_survives():
    out = sanitize_non_finite({"stock_returns": {"1mo": float("nan")}})
    assert "1mo" in out["stock_returns"], "the key was DROPPED, not nulled"
    assert out["stock_returns"]["1mo"] is None


def test_infinities_become_none():
    out = sanitize_non_finite({"a": float("inf"), "b": float("-inf")})
    assert out == {"a": None, "b": None}


def test_numpy_float64_nan_is_caught():
    # np.float64 is a float subclass -- this is why a plain isinstance check
    # is sufficient and np.isfinite/pd.isna are not needed.
    out = sanitize_non_finite({"x": np.float64("nan")})
    assert out["x"] is None


def test_finite_values_are_untouched():
    payload = {"a": 1.5, "b": 0.0, "c": -3.25, "d": np.float64(2.5)}
    assert sanitize_non_finite(payload) == {"a": 1.5, "b": 0.0, "c": -3.25, "d": 2.5}


def test_zero_is_not_confused_with_missing():
    # A real 0.0 must survive as 0.0. If the sanitiser used truthiness
    # instead of isfinite, this would come back as None.
    assert sanitize_non_finite({"v": 0.0})["v"] == 0.0


def test_recurses_into_lists_and_nested_dicts():
    payload = {
        "features": {"momentum_1m": float("nan"), "rsi_14": 55.0},
        "top_factors": [{"name": "mom", "contribution": float("nan")}, {"name": "rsi", "contribution": 0.2}],
    }
    out = sanitize_non_finite(payload)
    assert out["features"]["momentum_1m"] is None
    assert out["features"]["rsi_14"] == 55.0
    assert out["top_factors"][0]["contribution"] is None
    assert out["top_factors"][1]["contribution"] == 0.2


def test_non_float_types_pass_through():
    payload = {"s": "nan", "i": 3, "b": True, "n": None, "l": ["a", 1]}
    assert sanitize_non_finite(payload) == payload


def test_booleans_are_not_treated_as_floats():
    # bool subclasses int, not float -- guard against a future isinstance
    # widening that would turn True into None.
    out = sanitize_non_finite({"flag": True, "off": False})
    assert out == {"flag": True, "off": False}


def test_response_class_renders_nan_as_json_null():
    body = NaNSafeJSONResponse(content={"a": float("nan")}).render({"a": float("nan")})
    assert body == b'{"a":null}', body


def test_plain_jsonresponse_still_raises_proving_the_defect_is_real():
    """Control: without the sanitiser this payload is a guaranteed 500."""
    import json

    import pytest as _pytest

    from starlette.responses import JSONResponse

    with _pytest.raises(ValueError):
        JSONResponse(content={"a": 1}).render({"a": float("nan")})
    # and the stdlib reason, for the record
    with _pytest.raises(ValueError):
        json.dumps({"a": float("nan")}, allow_nan=False)


# ---------------------------------------------------------------------------
# Endpoint level -- criteria 1, 2, 3
# ---------------------------------------------------------------------------


def _install_fake_tools(monkeypatch, sector_1mo=float("nan")):
    """Replace every tool the route gathers, so nothing hits the network.

    `sector` reproduces the exact reported payload shape; the rest return the
    documented tool contract from .claude/rules/backend-tools.md.
    """
    import yfinance as yf

    from backend.tools import (
        alphavantage,
        alt_data,
        anomaly_detector,
        earnings_tone,
        fred_data,
        monte_carlo,
        nlp_sentiment,
        options_flow,
        patent_tracker,
        quant_model,
        sec_insider,
        sector_analysis,
        social_sentiment,
    )

    def ok(name):
        return {"ticker": "AAPL", "signal": "NEUTRAL", "summary": f"{name} ok", "data": {}}

    # The exact names `backend/api/signals.py` gathers. Patched with
    # `raising=True` (monkeypatch's default) ON PURPOSE: if any of these is
    # ever renamed, this fixture must fail loudly rather than silently leave
    # the real network-calling tool in place, which would make every
    # assertion below meaningless.
    patches = [
        (sector_analysis, "get_sector_analysis", lambda *a, **k: {
            "ticker": "AAPL",
            "signal": "NEUTRAL",
            "summary": "sector",
            "stock_returns": {"1mo": sector_1mo, "3mo": 1.5},
        }),
        (quant_model, "get_quant_model_signal", lambda *a, **k: {
            "ticker": "AAPL",
            "signal": "NEUTRAL",
            "summary": "quant",
            "data": {"features": {"momentum_1m": float("nan")}},
        }),
        (sec_insider, "get_insider_trades", lambda *a, **k: ok("insider")),
        (options_flow, "get_options_flow", lambda *a, **k: ok("options")),
        (social_sentiment, "get_social_sentiment", lambda *a, **k: ok("social")),
        (patent_tracker, "get_patent_data", lambda *a, **k: ok("patent")),
        (earnings_tone, "get_earnings_tone", lambda *a, **k: ok("earnings")),
        (fred_data, "get_macro_indicators", lambda *a, **k: ok("fred")),
        (alt_data, "get_google_trends", lambda *a, **k: ok("alt_data")),
        (nlp_sentiment, "get_nlp_sentiment", lambda *a, **k: ok("nlp")),
        (anomaly_detector, "get_anomaly_scan", lambda *a, **k: ok("anomalies")),
        (monte_carlo, "get_monte_carlo_simulation", lambda *a, **k: ok("monte_carlo")),
        (alphavantage, "get_market_intel", lambda *a, **k: {"sentiment_summary": []}),
    ]
    for mod, fn, impl in patches:
        assert hasattr(mod, fn), (
            f"{mod.__name__}.{fn} no longer exists -- signals.py was refactored and "
            "this fixture would otherwise silently exercise the real tool"
        )
        monkeypatch.setattr(mod, fn, impl)

    class _FakeTicker:
        def __init__(self, *a, **k):
            pass

        @property
        def info(self):
            return {"longName": "Apple Inc."}

        @property
        def news(self):
            return []

        def history(self, *a, **k):
            import pandas as pd

            return pd.DataFrame()

    monkeypatch.setattr(yf, "Ticker", _FakeTicker)


def _client():
    from backend.main import app
    from backend.tests.auth_helper import authed_test_client

    return authed_test_client(app, raise_server_exceptions=False)


def test_signals_endpoint_returns_200_with_a_nan_sector_return(monkeypatch):
    """Criteria 1 + 2 + 3, on the exact reported failure."""
    _install_fake_tools(monkeypatch)
    r = _client().get("/api/signals/AAPL")

    assert r.status_code == 200, r.text

    body = r.json()
    missing = [k for k in SIGNAL_KEYS if k not in body]
    assert not missing, f"missing signal keys: {missing}"

    returns = body["sector"]["stock_returns"]
    # PRESENT ...
    assert "1mo" in returns, "the NaN key was dropped silently"
    # ... and NULL, not 0.0
    assert returns["1mo"] is None, f"expected null, got {returns['1mo']!r}"
    # a finite sibling is untouched
    assert returns["3mo"] == 1.5


def test_nested_quant_model_nan_is_also_nulled(monkeypatch):
    """The leak has TWO sites; the boundary fix must cover the nested one."""
    _install_fake_tools(monkeypatch)
    body = _client().get("/api/signals/AAPL").json()
    assert body["quant_model"]["data"]["features"]["momentum_1m"] is None


def test_control_finite_payload_is_unchanged(monkeypatch):
    """Green control for the mutation matrix: with no NaN anywhere, the
    endpoint returns the value verbatim -- so a red run under mutation is
    attributable to the mutation and not to a broken fixture."""
    _install_fake_tools(monkeypatch, sector_1mo=2.5)
    body = _client().get("/api/signals/AAPL").json()
    assert body["sector"]["stock_returns"]["1mo"] == 2.5


def test_the_fixture_default_is_actually_non_finite():
    """Fixture pin -- reads the default off `_install_fake_tools` ITSELF.

    The first version of this test asserted `not math.isfinite(float("nan"))`
    and `math.isfinite(2.5)`. Those are LIBRARY FACTS: true no matter what
    the fixture does. It therefore PASSED under the very fixture mutation
    (N4: `sector_1mo` default -> 2.5) it claimed to guard -- the
    "library-fact assertion posing as a fixture pin" shape recorded in
    feedback_mutation_test_guards_and_fixtures. Found by Q/A, not by me.

    This version binds to the actual subject, so it dies under N4.
    """
    import inspect

    default = inspect.signature(_install_fake_tools).parameters["sector_1mo"].default
    assert isinstance(default, float), f"fixture default is {default!r}, not a float"
    assert not math.isfinite(default), (
        f"fixture default is {default!r} -- a FINITE value, so every endpoint test "
        "above would pass against an implementation that does nothing"
    )
