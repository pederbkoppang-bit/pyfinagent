"""phase-80.2 -- the error-response contract.

An unhandled backend 500 used to escape every response-decorating layer:
no CORS header (so the browser blocked it and the UI claimed the backend
was DOWN), no OWASP headers, and no PerfTracker record (so the latency
dashboard reported healthy over a dead endpoint).

Every test here drives a REAL request through the REAL middleware stack
against a route that GENUINELY RAISES. That is deliberate: a source scan
(the shape at backend/tests/test_phase_75_deploy_surface.py:397-401)
cannot see middleware ordering, which is the entire bug, so it would be a
guard that cannot fail (feedback_mutation_test_guards_and_fixtures).

Mutation-tested -- see handoff/current/experiment_results.md for the
matrix showing each assertion FAILING when its guard is reverted.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.main import app
from backend.services.perf_tracker import get_perf_tracker
from backend.tests.auth_helper import authed_test_client

PROBE = "/api/__force_500_probe"
ALLOWED_ORIGIN = "http://localhost:3000"
DISALLOWED_ORIGIN = "https://evil.example"

# TestClient defaults to raise_server_exceptions=True (starlette
# testclient.py:204), which re-raises the endpoint exception instead of
# letting us inspect the 500 the client would actually receive.
client = authed_test_client(app, raise_server_exceptions=False)


def _hdr(response, name: str) -> str | None:
    return response.headers.get(name)


# ── criterion 1: CORS on a genuine 500 ──────────────────────────────


def test_probe_route_genuinely_raises_a_500():
    """The fixture must be a real unhandled exception, not a 404 or a 204.

    A 404 is an HTTPException handled by the innermost ExceptionMiddleware
    and therefore carried CORS headers even BEFORE this fix -- asserting
    against one would be a false pass.
    """
    r = client.get(PROBE, headers={"Origin": ALLOWED_ORIGIN})
    assert r.status_code == 500, (
        f"probe returned {r.status_code}; the CORS assertions below are only "
        "meaningful against a genuinely-raising route"
    )


def test_500_carries_cors_header_for_allowed_origin():
    r = client.get(PROBE, headers={"Origin": ALLOWED_ORIGIN})
    assert r.status_code == 500
    assert _hdr(r, "access-control-allow-origin") == ALLOWED_ORIGIN


def test_200_control_still_carries_cors_header():
    """The fix must not have been achieved by breaking the working path."""
    r = client.get("/api/health", headers={"Origin": ALLOWED_ORIGIN})
    assert r.status_code == 200
    assert _hdr(r, "access-control-allow-origin") == ALLOWED_ORIGIN


# ── criterion 4: the allow-list is UNCHANGED ────────────────────────


def test_500_omits_cors_header_for_disallowed_origin():
    """The fix must not be 'echo *'. A disallowed origin still gets nothing."""
    r = client.get(PROBE, headers={"Origin": DISALLOWED_ORIGIN})
    assert r.status_code == 500
    assert _hdr(r, "access-control-allow-origin") is None


def test_500_never_echoes_wildcard_origin():
    for origin in (ALLOWED_ORIGIN, DISALLOWED_ORIGIN):
        r = client.get(PROBE, headers={"Origin": origin})
        assert _hdr(r, "access-control-allow-origin") != "*"


def test_200_control_omits_cors_header_for_disallowed_origin():
    r = client.get("/api/health", headers={"Origin": DISALLOWED_ORIGIN})
    assert _hdr(r, "access-control-allow-origin") is None


# ── addendum (ii): OWASP headers on the 500 ─────────────────────────


def test_500_carries_owasp_headers():
    """`.claude/rules/security.md` requires these on ALL responses."""
    r = client.get(PROBE, headers={"Origin": ALLOWED_ORIGIN})
    assert _hdr(r, "x-content-type-options") == "nosniff"
    assert _hdr(r, "x-frame-options") == "DENY"
    assert _hdr(r, "referrer-policy") == "strict-origin-when-cross-origin"
    assert _hdr(r, "cache-control") == "no-store"
    assert _hdr(r, "permissions-policy") is not None
    assert _hdr(r, "x-response-time") is not None


# ── addendum (iii): the 500 reaches PerfTracker AND is visible ──────


def test_500_is_recorded_by_perf_tracker_with_status_500():
    tracker = get_perf_tracker()
    tracker.clear()
    client.get(PROBE, headers={"Origin": ALLOWED_ORIGIN})

    entries = [e for e in tracker._entries if e.endpoint == PROBE]
    assert entries, "the 500 never reached PerfTracker"
    assert any(e.status_code == 500 for e in entries), (
        f"recorded status codes were {[e.status_code for e in entries]}, "
        "expected a 500"
    )


def test_500_is_visible_as_an_error_in_observability_latency():
    """A bare count bump is not visibility -- the error must be derivable."""
    tracker = get_perf_tracker()
    tracker.clear()
    client.get(PROBE, headers={"Origin": ALLOWED_ORIGIN})

    r = client.get("/api/observability/latency?window=300")
    assert r.status_code == 200
    body = r.json()

    assert body["error_count"] >= 1, body
    assert body["error_rate_pct"] > 0, body
    assert body["per_endpoint"][PROBE]["error_count"] >= 1, body["per_endpoint"]


def test_successful_requests_do_not_count_as_errors():
    """Mutation-guard: an error counter that counts everything is useless."""
    tracker = get_perf_tracker()
    tracker.clear()
    client.get("/api/health", headers={"Origin": ALLOWED_ORIGIN})

    body = client.get("/api/observability/latency?window=300").json()
    assert body["total_requests"] >= 1
    assert body["error_count"] == 0, body
    assert body["error_rate_pct"] == 0, body


def test_observability_latency_keeps_its_pre_existing_keys():
    """The additive change must not break existing consumers."""
    body = client.get("/api/observability/latency").json()
    for key in (
        "p50",
        "p95",
        "p99",
        "total_requests",
        "window_seconds",
        "cache_hit_rate_pct",
        "per_endpoint",
    ):
        assert key in body, f"consumer-visible key {key!r} disappeared"


# ── do-no-harm (iii): the traceback is NOT swallowed ────────────────


def test_unhandled_exception_is_logged_with_traceback(caplog):
    """ServerErrorMiddleware used to re-raise purely so the server logged
    the traceback. Catching lower ends that, so the middleware must log it
    itself -- asserted, never assumed."""
    caplog.set_level(logging.ERROR, logger="backend.middleware.catch_all_errors")
    client.get(PROBE, headers={"Origin": ALLOWED_ORIGIN})

    records = [
        rec
        for rec in caplog.records
        if rec.name == "backend.middleware.catch_all_errors"
        and rec.levelno >= logging.ERROR
    ]
    assert records, "the unhandled exception was swallowed without a log line"
    assert any(rec.exc_info for rec in records), (
        "logged without exc_info -- backend.log would lose the traceback"
    )
    assert any(PROBE in rec.getMessage() for rec in records), (
        "the log line does not identify which path failed"
    )


def test_500_body_is_json_without_a_traceback():
    """phase-75.16: tracebacks must never leak to HTTP callers."""
    r = client.get(PROBE, headers={"Origin": ALLOWED_ORIGIN})
    body = r.json()
    assert body == {"detail": "Internal Server Error"}
    assert "Traceback" not in r.text
    assert "RuntimeError" not in r.text


# ── do-no-harm (i): the inverse trap -- client errors stay client errors ──


def test_404_still_returns_404_not_500():
    r = client.get("/api/definitely-not-a-route", headers={"Origin": ALLOWED_ORIGIN})
    assert r.status_code == 404


def test_401_is_unchanged_for_an_unauthenticated_caller():
    """A catch-all placed wrongly would turn every 401 into a 500."""
    from fastapi.testclient import TestClient

    anon = TestClient(app, raise_server_exceptions=False)
    r = anon.get("/api/paper-trading/portfolio", headers={"Origin": ALLOWED_ORIGIN})
    assert r.status_code in (200, 401, 403), (
        f"expected the auth path to be untouched, got {r.status_code}"
    )
    assert r.status_code != 500


def test_http_exception_is_rendered_with_its_own_status_code():
    """Directly exercise the middleware's defensive HTTPException branch.

    ExceptionMiddleware normally converts these before they reach the
    catch-all, so this drives the class in isolation -- otherwise the
    branch would be untested code asserting its own safety.
    """
    from starlette.applications import Starlette
    from starlette.exceptions import HTTPException
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient as RawTestClient

    from backend.middleware.catch_all_errors import CatchAllServerErrorMiddleware

    async def raise_401(request):  # pragma: no cover - trivial
        raise HTTPException(status_code=401, detail="nope")

    async def raise_value_error(request):  # pragma: no cover - trivial
        raise ValueError("boom")

    async def ok(request):  # pragma: no cover - trivial
        return PlainTextResponse("ok")

    inner = Starlette(
        routes=[
            Route("/401", raise_401),
            Route("/boom", raise_value_error),
            Route("/ok", ok),
        ]
    )
    # Wrap DIRECTLY, with no ExceptionMiddleware in between, so the
    # HTTPException actually reaches the catch-all.
    wrapped = CatchAllServerErrorMiddleware(inner.router)
    c = RawTestClient(wrapped, raise_server_exceptions=False)

    assert c.get("/401").status_code == 401
    assert c.get("/401").json() == {"detail": "nope"}
    assert c.get("/boom").status_code == 500
    assert c.get("/ok").status_code == 200


# ── the registration order the whole fix depends on ─────────────────


def test_catch_all_is_registered_inside_cors_middleware():
    """`add_middleware` inserts at index 0, so LATER registration = MORE
    OUTER. The catch-all must appear AFTER CORSMiddleware in
    `user_middleware` (i.e. be nested inside it), or CORS never decorates
    the 500 and the fix silently reverts."""
    names = [m.cls.__name__ for m in app.user_middleware]

    assert "CatchAllServerErrorMiddleware" in names, names
    assert "CORSMiddleware" in names, names
    assert names.index("CatchAllServerErrorMiddleware") > names.index(
        "CORSMiddleware"
    ), (
        f"middleware order is {names} (index 0 = outermost); the catch-all "
        "must be INSIDE CORSMiddleware"
    )


def test_no_catch_all_exception_handler_was_registered():
    """The intuitive fix is a trap and must not creep back in.

    starlette/applications.py:62-66 routes any handler keyed on 500 or
    Exception into ServerErrorMiddleware -- the OUTERMOST layer -- so such
    a handler produces a response outside CORS, outside the auth
    middleware, and outside PerfTracker. Measured: it changes only the
    body.
    """
    assert Exception not in app.exception_handlers
    assert 500 not in app.exception_handlers
