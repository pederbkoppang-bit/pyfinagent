"""phase-23.5.3.1: URL-pinning regression guards for the digest handlers.

The morning + evening digests were unconditionally hitting the
Docker-alias `_BACKEND_URL = "http://backend:8000"`, which doesn't
resolve from the Mac host process. After phase-23.5.3.1 they MUST
hit `127.0.0.1:8000` (via `_LOCAL_BACKEND_URL`) so the Slack post
actually goes out.

Helper shapes (`_FakeAsyncClient`, `_fake_response`, `_fake_app`)
are inlined to avoid a cross-test import path -- the watchdog test
module uses identical helpers.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.slack_bot import scheduler as scheduler_mod


def _fake_app():
    app = MagicMock()
    app.client = MagicMock()
    app.client.chat_postMessage = AsyncMock()
    return app


def _fake_response(status_code: int, body):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=body if body is not None else {})
    return resp


class _FakeAsyncClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url):
        self.calls.append(url)
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _patch_client_with(responses):
    fake = _FakeAsyncClient(responses)
    return patch.object(scheduler_mod.httpx, "AsyncClient", return_value=fake), fake


@pytest.fixture(autouse=True)
def _settings_stub():
    """Inject a fake settings so digest handlers don't bail on missing channel."""
    with patch.object(
        scheduler_mod,
        "get_settings",
        return_value=SimpleNamespace(slack_channel_id="C_TEST"),
    ):
        yield


def _run(coro):
    return asyncio.run(coro)


def test_morning_digest_uses_localhost_not_docker_alias():
    """Both httpx GET calls in _send_morning_digest must hit 127.0.0.1:8000."""
    app = _fake_app()
    portfolio_resp = _fake_response(200, {"total_pnl": 0, "positions": []})
    reports_resp = _fake_response(200, [])
    cm, fake = _patch_client_with([portfolio_resp, reports_resp])
    with cm:
        _run(scheduler_mod._send_morning_digest(app))
    assert len(fake.calls) == 2, f"expected 2 GETs, saw {len(fake.calls)}: {fake.calls}"
    for url in fake.calls:
        assert "127.0.0.1:8000" in url or "localhost:8000" in url, (
            f"digest URL not localhost-pinned: {url!r}"
        )
        assert "://backend:8000" not in url, (
            f"docker alias hostname leaked back in: {url!r}"
        )


def test_morning_digest_posts_to_slack_on_success():
    """On healthy responses, _send_morning_digest must call chat_postMessage once."""
    app = _fake_app()
    portfolio_resp = _fake_response(200, {"total_pnl": 0, "positions": []})
    reports_resp = _fake_response(200, [])
    cm, _ = _patch_client_with([portfolio_resp, reports_resp])
    with cm:
        _run(scheduler_mod._send_morning_digest(app))
    assert app.client.chat_postMessage.await_count == 1


def test_evening_digest_uses_localhost_not_docker_alias():
    """Both httpx GET calls in _send_evening_digest must hit 127.0.0.1:8000.

    phase-23.5.7.1: trades_resp is the realistic dict-envelope shape
    `{"trades": [...], "count": N}` (per backend/api/paper_trading.py:226),
    NOT a bare list. The boundary coerce in _send_evening_digest must
    unwrap it before format_evening_digest's slice access.
    """
    app = _fake_app()
    portfolio_resp = _fake_response(200, {"total_pnl": 0, "positions": []})
    trades_resp = _fake_response(200, {"trades": [], "count": 0})
    cm, fake = _patch_client_with([portfolio_resp, trades_resp])
    with cm:
        _run(scheduler_mod._send_evening_digest(app))
    assert len(fake.calls) == 2, f"expected 2 GETs, saw {len(fake.calls)}: {fake.calls}"
    for url in fake.calls:
        assert "127.0.0.1:8000" in url or "localhost:8000" in url, (
            f"digest URL not localhost-pinned: {url!r}"
        )
        assert "://backend:8000" not in url, (
            f"docker alias hostname leaked back in: {url!r}"
        )


def test_evening_digest_posts_to_slack_on_success():
    """On healthy responses, _send_evening_digest must call chat_postMessage once.

    phase-23.5.7.1: trades_resp uses the realistic dict-envelope shape so
    the boundary coerce is exercised on the success path.
    """
    app = _fake_app()
    portfolio_resp = _fake_response(200, {"total_pnl": 0, "positions": []})
    trades_resp = _fake_response(200, {"trades": [], "count": 0})
    cm, _ = _patch_client_with([portfolio_resp, trades_resp])
    with cm:
        _run(scheduler_mod._send_evening_digest(app))
    assert app.client.chat_postMessage.await_count == 1
