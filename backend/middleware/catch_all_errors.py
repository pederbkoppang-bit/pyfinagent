"""Catch-all server-error middleware (phase-80.2).

WHY THIS EXISTS
---------------
Starlette pins ``ServerErrorMiddleware`` as the OUTERMOST layer --
``build_middleware_stack`` builds
``[ServerErrorMiddleware] + user_middleware + [ExceptionMiddleware]``
(starlette 1.0.0 applications.py:57-77), and any handler registered
against ``500`` or ``Exception`` is routed into that outermost layer
(``:62-66``).

Two consequences bit this app, both measured on 2026-07-25:

1. An unhandled exception unwinds past ``CORSMiddleware``, so the 500 it
   produces carries NO ``Access-Control-Allow-Origin``. The browser then
   blocks the response and the frontend reports "Cannot reach backend" --
   sending the operator to restart a server that is running fine.
2. It also unwinds past ``auth_and_security_middleware``, whose whole tail
   (``main.py:548-569``) is therefore skipped: no ``PerfTracker.record``,
   no ``X-Response-Time``, and none of the six OWASP headers that
   ``.claude/rules/security.md`` requires on ALL responses. The
   observability tiles showed healthy latency over a dead endpoint
   because every failure was invisible to the only recorder.

``@app.exception_handler(Exception)`` does NOT fix either one -- it is
attached to that same outermost ``ServerErrorMiddleware`` and only changes
the body. Measured, not assumed (see the 3-variant probe recorded in
``handoff/current/contract.md`` section 4). Upstream considers this
by-design: Kludex/starlette#1175 is closed with no fix, and fastapi#13398
records "No plans" to even document it.

So the fix has to catch the exception at a point NESTED INSIDE
``CORSMiddleware``. Registration order is what puts it there -- see the
comment at the ``add_middleware`` call site in ``backend/main.py``.

DESIGN NOTES
------------
- Pure ASGI, deliberately NOT another ``BaseHTTPMiddleware``: each
  ``BaseHTTPMiddleware`` layer adds a task group and a contextvar copy per
  request, and Starlette 0.46/0.49 shipped fixes in that area.
- Mirrors ``starlette/middleware/errors.py:154-186``: wrap ``send``, track
  whether the response has started, and only substitute a response when it
  has not.
- ``logger.exception`` is MANDATORY here. ``ServerErrorMiddleware``
  re-raises (``errors.py:183-186``) purely so the server logs the
  traceback; catching lower ends that, and losing tracebacks from
  ``backend.log`` would be a strictly worse outcome than the bug being
  fixed. There is a caplog assertion for this.
- The body never carries a traceback (phase-75.16 established leaking
  tracebacks to HTTP callers as a defect class in this repo).
"""
from __future__ import annotations

import logging
from typing import Any

from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

__all__ = ["CatchAllServerErrorMiddleware"]

# Body returned for an unhandled exception. `api.ts` reads `body.detail`
# (frontend/src/lib/api.ts:151-157) before short-circuiting on status 500
# at :161, so this shape is what the existing frontend already expects.
_GENERIC_500_DETAIL = "Internal Server Error"


class CatchAllServerErrorMiddleware:
    """Convert an unhandled exception into a real ASGI response.

    Returning a *response* (rather than letting the exception escape to
    ``ServerErrorMiddleware``) is the entire point: a response travels back
    out through every enclosing layer, so ``CORSMiddleware`` decorates it
    and ``auth_and_security_middleware`` records and heads it.

    This middleware makes NO origin decision of its own. The allow-list
    stays entirely inside ``CORSMiddleware.is_allowed_origin``, so there is
    no second copy of ``_TAILSCALE_ORIGIN_RE`` to drift out of sync
    (the phase-75.1 security-04 invariant).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # WebSocket / lifespan scopes have no HTTP response to substitute.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def _send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, _send)
        except Exception as exc:
            method = scope.get("method", "?")
            path = scope.get("path", "?")

            # Replaces the traceback ServerErrorMiddleware's `raise exc`
            # used to feed to backend.log. Without this the fix would trade
            # one blind spot for another.
            logger.exception(
                "Unhandled exception serving %s %s: %r", method, path, exc
            )

            if response_started:
                # Headers (and possibly part of the body) are already on the
                # wire -- e.g. the SSE stream at backend/api/mas_events.py:36
                # failing mid-iteration. A partially-sent response cannot be
                # rewritten, so re-raise and let ServerErrorMiddleware handle
                # the teardown exactly as it does today. This case is NOT
                # covered by the CORS fix and this comment says so rather
                # than implying otherwise.
                raise

            response = self._build_response(exc)
            await response(scope, receive, send)

    @staticmethod
    def _build_response(exc: BaseException) -> JSONResponse:
        """Render `exc` without ever downgrading a client error to a 500.

        ``ExceptionMiddleware`` is nested INSIDE this middleware, so an
        ``HTTPException`` raised in a route is normally converted before it
        ever reaches here. This branch exists so that the "a catch-all
        swallowed my 401 and returned 500" trap is structurally impossible
        rather than merely unlikely -- if a future refactor moves this
        middleware outside ``ExceptionMiddleware``, auth responses still
        keep their own status code.
        """
        if isinstance(exc, StarletteHTTPException):
            detail: Any = exc.detail
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": detail},
                headers=getattr(exc, "headers", None),
            )
        return JSONResponse(
            status_code=500, content={"detail": _GENERIC_500_DETAIL}
        )
