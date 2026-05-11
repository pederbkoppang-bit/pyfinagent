"""phase-23.5.2.6: state-transition alert semantics for _watchdog_health_check.

The watchdog must:
  - Post on first failure (None -> False or True -> False)
  - Post on recovery (False -> True)
  - NOT post on steady-healthy (None -> True, True -> True)
  - NOT post on steady-unhealthy (False -> False) -- the spam pre-fix
  - Reset cleanly per test via the module-level state variable

Each test resets `scheduler._watchdog_last_was_healthy = None` to isolate.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.slack_bot import scheduler as scheduler_mod


def _fake_app():
    """Build a minimal Bolt-app-shaped fake with an awaitable chat_postMessage."""
    app = MagicMock()
    app.client = MagicMock()
    app.client.chat_postMessage = AsyncMock()
    return app


def _fake_response(status_code: int, body: dict | None):
    """Httpx-shaped response for AsyncClient.get monkeypatch."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=body if body is not None else {})
    return resp


class _FakeAsyncClient:
    """Async-context-manager-shaped httpx client that returns a queued response."""

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


@pytest.fixture(autouse=True)
def _reset_watchdog_state():
    """Each test starts with `_watchdog_last_was_healthy = None`."""
    scheduler_mod._watchdog_last_was_healthy = None
    # Inject a fake slack_channel_id so the function doesn't bail on settings.
    with patch.object(scheduler_mod, "get_settings", return_value=SimpleNamespace(slack_channel_id="C_TEST")):
        yield
    scheduler_mod._watchdog_last_was_healthy = None


def _run(coro):
    return asyncio.run(coro)


def _patch_client_with(responses):
    """Patch `httpx.AsyncClient(...)` to yield a fake client returning `responses` in order."""
    fake = _FakeAsyncClient(responses)
    return patch.object(scheduler_mod.httpx, "AsyncClient", return_value=fake), fake


def test_steady_healthy_after_clean_start_no_post():
    """None -> True -> True -> True : NO Slack posts ever."""
    app = _fake_app()
    healthy = _fake_response(200, {"status": "ok"})
    cm, _ = _patch_client_with([healthy, healthy, healthy])
    with cm:
        for _ in range(3):
            _run(scheduler_mod._watchdog_health_check(app))
    assert app.client.chat_postMessage.await_count == 0
    assert scheduler_mod._watchdog_last_was_healthy is True


def test_first_failure_after_clean_start_posts_alert():
    """None -> False : POST one alert (post-restart already-broken case)."""
    app = _fake_app()
    cm, _ = _patch_client_with([ConnectionError("dns failure")])
    with cm:
        _run(scheduler_mod._watchdog_health_check(app))
    assert app.client.chat_postMessage.await_count == 1
    text_arg = app.client.chat_postMessage.await_args.kwargs.get("text", "")
    assert "unreachable" in text_arg.lower()
    assert scheduler_mod._watchdog_last_was_healthy is False


def test_consecutive_failures_no_repost():
    """None -> False -> False -> False : ONE post (the first-failure), no repeats."""
    app = _fake_app()
    cm, _ = _patch_client_with([ConnectionError("x"), ConnectionError("y"), ConnectionError("z")])
    with cm:
        for _ in range(3):
            _run(scheduler_mod._watchdog_health_check(app))
    # Pre-fix this would be 3 posts (the spam). Post-fix: exactly 1.
    assert app.client.chat_postMessage.await_count == 1
    assert scheduler_mod._watchdog_last_was_healthy is False


def test_recovery_after_failure_posts_recovery():
    """None -> False -> True : 2 posts (alert + recovery)."""
    app = _fake_app()
    healthy = _fake_response(200, {"status": "ok"})
    cm, _ = _patch_client_with([ConnectionError("x"), healthy])
    with cm:
        _run(scheduler_mod._watchdog_health_check(app))  # alert
        _run(scheduler_mod._watchdog_health_check(app))  # recovery
    assert app.client.chat_postMessage.await_count == 2
    second = app.client.chat_postMessage.await_args_list[-1].kwargs.get("text", "")
    assert "recovery" in second.lower() or "reachable" in second.lower()
    assert scheduler_mod._watchdog_last_was_healthy is True


def test_steady_healthy_after_recovery_no_more_posts():
    """None -> False -> True -> True -> True : 2 total posts (alert + recovery)."""
    app = _fake_app()
    healthy = _fake_response(200, {"status": "ok"})
    cm, _ = _patch_client_with([ConnectionError("x"), healthy, healthy, healthy])
    with cm:
        for _ in range(4):
            _run(scheduler_mod._watchdog_health_check(app))
    assert app.client.chat_postMessage.await_count == 2
    assert scheduler_mod._watchdog_last_was_healthy is True


def test_uses_localhost_probe_url_not_docker_alias():
    """Regression guard: the probe MUST hit 127.0.0.1, not the Docker alias."""
    app = _fake_app()
    healthy = _fake_response(200, {"status": "ok"})
    cm, fake = _patch_client_with([healthy])
    with cm:
        _run(scheduler_mod._watchdog_health_check(app))
    assert len(fake.calls) == 1
    url = fake.calls[0]
    assert "127.0.0.1:8000" in url or "localhost:8000" in url, f"probe URL regressed: {url!r}"
    assert "://backend:8000" not in url, f"docker alias hostname leaked back in: {url!r}"
