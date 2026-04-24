"""phase-11.1 google-genai client factory (shim module).

Single import surface for the new Google Gen AI SDK (`google-genai`,
`from google import genai`). Every call site in phase-11.2 / 11.3
imports through this module so phase-11.4 (remove deprecated `vertexai`
top-level package) becomes a grep-verifiable operation.

Pattern: double-checked-lock singleton. `genai.Client` thread safety is
not documented in the SDK source; FastAPI / uvicorn worker threads will
call `get_genai_client()` concurrently, so a lock is required to avoid
building two clients on cold start.

Fail-open at every boundary:
- `google-genai` absent (SDK not installed) -> WARNING + return None.
- `genai.Client(...)` raises (bad creds / bad project / network) ->
  WARNING + return None.
Callers MUST handle a None return. The factory never raises.

See `docs/VERTEX_AI_GENAI_MIGRATION.md` for the phase-11 migration plan.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_client_lock = threading.Lock()
_client: Any = None


def _build_client() -> Any:
    """Construct a fresh `genai.Client` via Vertex. Never raises."""
    try:
        from google import genai  # type: ignore[attr-defined]
    except Exception as exc:
        logger.warning(
            "_genai_client: google-genai not installed (%r); returning None. "
            "Install with: pip install google-genai==1.73.1",
            exc,
        )
        return None

    try:
        from backend.config.settings import get_settings

        s = get_settings()
        project = s.gcp_project_id
        location = s.gcp_location
        credentials_json = getattr(s, "gcp_credentials_json", "") or ""
    except Exception as exc:
        logger.warning(
            "_genai_client: settings load failed (%r); returning None", exc
        )
        return None

    credentials = None
    if credentials_json:
        try:
            import json

            from google.oauth2 import service_account  # type: ignore

            info = json.loads(credentials_json)
            credentials = service_account.Credentials.from_service_account_info(
                info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        except Exception as exc:
            logger.warning(
                "_genai_client: explicit credentials parse failed (%r); "
                "falling back to ADC",
                exc,
            )
            credentials = None

    try:
        kwargs: dict[str, Any] = {"vertexai": True, "project": project, "location": location}
        if credentials is not None:
            kwargs["credentials"] = credentials
        client = genai.Client(**kwargs)
        logger.debug(
            "_genai_client: built genai.Client project=%s location=%s creds=%s",
            project,
            location,
            "explicit" if credentials is not None else "ADC",
        )
        return client
    except Exception as exc:
        logger.warning(
            "_genai_client: genai.Client(...) init failed (%r); returning None",
            exc,
        )
        return None


def get_genai_client() -> Any:
    """Return the process-singleton `genai.Client`. Never raises.

    May return `None` on fail-open (SDK absent, bad creds, etc.).
    Callers must guard: `if client is None: ...`.

    Defense-in-depth: `_build_client` is constructed to never raise, but
    the outer path also wraps the call in a catch-all so no hypothetical
    bug downstream can make this function throw.
    """
    global _client
    # Fast path (no lock)
    if _client is not None:
        return _client
    # Slow path (double-checked lock)
    with _client_lock:
        if _client is None:
            try:
                _client = _build_client()
            except Exception as exc:  # pragma: no cover -- defense-in-depth
                logger.warning(
                    "_genai_client: unexpected exception from _build_client "
                    "(%r); staying None",
                    exc,
                )
                _client = None
        return _client


def close_genai_client() -> None:
    """Drop the singleton so the next `get_genai_client()` rebuilds.

    Safe to call when no client has been built yet. Safe to call from
    multiple threads. Does not raise.

    Intended for FastAPI lifespan shutdown and for test isolation.
    """
    global _client
    with _client_lock:
        existing = _client
        _client = None
    if existing is not None and hasattr(existing, "close"):
        try:
            existing.close()
        except Exception as exc:  # pragma: no cover -- defensive
            logger.debug("_genai_client: close() failed (%r)", exc)


def reset_for_test() -> None:
    """Explicit-named alias for `close_genai_client` used in tests."""
    close_genai_client()


__all__ = ["get_genai_client", "close_genai_client", "reset_for_test"]
