"""GET /api/charts/{ticker} must not 500 on a non-finite float.

Measured 2026-08-18: `GET /api/charts/DELL?period=1y` returned HTTP 500.
backend.log's traceback (verbatim):

    ValueError('Out of range float values are not JSON compliant: nan')
    when serializing dict item 'Open'
    when serializing list item 250

`backend/tools/yfinance_tool.py::get_price_history` returns
`df.reset_index().to_dict(orient="records")` with no NaN sanitisation, and
`backend/api/charts.py`'s router had no `default_response_class`, so
Starlette's default `JSONResponse.render` (`json.dumps(..., allow_nan=False)`)
raises on the first non-finite float it meets -- a failure the route's own
try/except cannot catch, since it happens during ASGI rendering after the
route already returned its dict.

This is the identical defect class fixed for `GET /api/signals/{ticker}` in
phase-80.1 (`backend/api/_json_safe.py`, `NaNSafeJSONResponse`,
`test_phase_80_1_signals_nan_serialisation.py`) -- that fix was applied to
the `signals` router only; `charts.py` was never given the same treatment.
The fix here is the same one-line pattern: put `NaNSafeJSONResponse` on the
`charts` router's `default_response_class`.

THE ASSERTIONS MATTER INDIVIDUALLY, same rationale as the phase-80.1 suite: a
test that only checks "no 500" is satisfied by silently coercing NaN to 0.0
(a wrong-number regression) or by dropping the row/key entirely (hides a data
gap). So every endpoint test below asserts:
    1. status 200
    2. the NaN-bearing row survives -- both keys present
    3. the NaN field is specifically None (JSON null), not 0.0 or missing
    4. a finite sibling field/row is untouched
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.api._json_safe import NaNSafeJSONResponse

# ---------------------------------------------------------------------------
# Router wiring -- the actual fix
# ---------------------------------------------------------------------------


def test_charts_router_uses_nan_safe_response_class():
    from backend.api.charts import router

    assert router.default_response_class is NaNSafeJSONResponse, (
        f"charts router default_response_class is {router.default_response_class!r}, "
        "not NaNSafeJSONResponse -- a NaN in any charts payload will 500 again"
    )


def test_plain_jsonresponse_still_raises_proving_the_defect_is_real():
    """Control: without the sanitiser this exact payload is a guaranteed 500."""
    import pytest as _pytest
    from starlette.responses import JSONResponse

    payload = [{"Date": "2026-08-17", "Open": float("nan"), "Close": 123.4}]
    with _pytest.raises(ValueError):
        JSONResponse(content=payload).render(payload)


def test_nan_safe_response_renders_the_reported_shape():
    """Reproduces the exact reported failure ('Open' nan at a list tail row)."""
    payload = [
        {"Date": "2026-08-16", "Open": 120.1, "Close": 121.0},
        {"Date": "2026-08-17", "Open": float("nan"), "Close": 123.4},
    ]
    body = NaNSafeJSONResponse(content=payload).render(payload)
    import json

    decoded = json.loads(body)
    assert decoded[0]["Open"] == 120.1, "finite sibling row must be untouched"
    assert "Open" in decoded[1], "the NaN key was dropped, not nulled"
    assert decoded[1]["Open"] is None, f"expected null, got {decoded[1]['Open']!r}"
    assert decoded[1]["Close"] == 123.4, "finite sibling field must be untouched"


# ---------------------------------------------------------------------------
# Endpoint level -- criteria 1-4, through the real route + real middleware
# ---------------------------------------------------------------------------


def _client():
    from backend.main import app
    from backend.tests.auth_helper import authed_test_client

    return authed_test_client(app, raise_server_exceptions=False)


def _fake_rows(nan_open=True):
    rows = [
        {"Date": "2026-08-16", "Open": 120.1, "High": 122.0, "Low": 119.5, "Close": 121.0, "Volume": 1000},
        {"Date": "2026-08-17", "Open": (float("nan") if nan_open else 118.9), "High": 121.0, "Low": 117.0, "Close": 118.5, "Volume": 900},
    ]
    return rows


def test_charts_endpoint_returns_200_with_a_nan_open(monkeypatch):
    """Criteria 1 + 2 + 3, on the exact reported failure shape."""
    from backend.tools import yfinance_tool

    monkeypatch.setattr(yfinance_tool, "get_price_history", lambda *a, **k: _fake_rows(nan_open=True))

    r = _client().get("/api/charts/DELL?period=1y")
    assert r.status_code == 200, r.text

    body = r.json()
    assert len(body) == 2, "a row must not be dropped because one field is NaN"
    assert "Open" in body[1], "the NaN key was dropped silently"
    assert body[1]["Open"] is None, f"expected null, got {body[1]['Open']!r}"
    # finite sibling row is untouched
    assert body[0]["Open"] == 120.1


def test_control_finite_payload_is_unchanged(monkeypatch):
    """Green control for the mutation matrix: with no NaN anywhere, the
    endpoint returns the value verbatim -- so a red run under mutation is
    attributable to the mutation and not to a broken fixture."""
    from backend.tools import yfinance_tool

    monkeypatch.setattr(yfinance_tool, "get_price_history", lambda *a, **k: _fake_rows(nan_open=False))

    r = _client().get("/api/charts/DELL?period=1y")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body[1]["Open"] == 118.9


def test_the_fixture_default_is_actually_non_finite():
    """Fixture pin, per feedback_mutation_test_guards_and_fixtures: binds to
    the actual fixture output rather than restating a library fact, so it
    dies if the fixture default stops being non-finite."""
    rows = _fake_rows(nan_open=True)
    assert not math.isfinite(rows[1]["Open"]), (
        f"fixture's nan_open=True row is {rows[1]['Open']!r} -- a FINITE value, "
        "so test_charts_endpoint_returns_200_with_a_nan_open would pass against "
        "an implementation that does nothing"
    )
