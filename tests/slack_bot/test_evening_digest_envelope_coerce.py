"""phase-23.5.7.1: regression guard for the dict-envelope coerce in
_send_evening_digest.

The /api/paper-trading/trades?limit=10 endpoint returns a dict envelope
`{"trades": [...], "count": N}` (paper_trading.py:226). Before
phase-23.5.7.1, _send_evening_digest passed the dict directly to
format_evening_digest, which tried `trades_today[:10]` and raised
`KeyError: slice(None, 10, None)` because dicts don't support slicing.

Post-fix, _send_evening_digest unwraps the envelope at the HTTP boundary
via `if isinstance(_raw, dict): trades_data = _raw.get("trades", [])`.
This file pins that behavior across four shapes:

  1. Typical dict envelope -> unwraps to the inner list.
  2. Empty dict envelope -> unwraps to [].
  3. Bare list (legacy fallback) -> passthrough unchanged.
  4. Status != 200 -> empty list (existing behavior preserved).
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
    with patch.object(
        scheduler_mod,
        "get_settings",
        return_value=SimpleNamespace(slack_channel_id="C_TEST"),
    ):
        yield


def _run(coro):
    return asyncio.run(coro)


def _capture_format_input(monkeypatch):
    """Replace format_evening_digest with a recorder that returns a minimal
    valid Block-Kit list and captures (portfolio, trades) it received."""
    captured: dict = {}

    def _recorder(portfolio_data, trades_data):
        captured["portfolio"] = portfolio_data
        captured["trades"] = trades_data
        return [{"type": "section", "text": {"type": "mrkdwn", "text": "ok"}}]

    monkeypatch.setattr(scheduler_mod, "format_evening_digest", _recorder)
    return captured


def test_dict_envelope_typical_unwraps_to_inner_list(monkeypatch: pytest.MonkeyPatch):
    """{"trades": [...], "count": N} -> list reaches the formatter."""
    captured = _capture_format_input(monkeypatch)
    app = _fake_app()
    portfolio_resp = _fake_response(200, {"total_pnl": 0, "positions": []})
    trades_resp = _fake_response(
        200,
        {"trades": [{"ticker": "AAPL", "action": "BUY"}], "count": 1},
    )
    cm, _ = _patch_client_with([portfolio_resp, trades_resp])
    with cm:
        _run(scheduler_mod._send_evening_digest(app))
    assert isinstance(captured["trades"], list), (
        f"formatter received non-list trades: {type(captured['trades'])}"
    )
    assert captured["trades"] == [{"ticker": "AAPL", "action": "BUY"}]
    assert app.client.chat_postMessage.await_count == 1


def test_dict_envelope_empty_unwraps_to_empty_list(monkeypatch: pytest.MonkeyPatch):
    """{"trades": [], "count": 0} -> [] reaches the formatter."""
    captured = _capture_format_input(monkeypatch)
    app = _fake_app()
    portfolio_resp = _fake_response(200, {})
    trades_resp = _fake_response(200, {"trades": [], "count": 0})
    cm, _ = _patch_client_with([portfolio_resp, trades_resp])
    with cm:
        _run(scheduler_mod._send_evening_digest(app))
    assert captured["trades"] == []
    assert app.client.chat_postMessage.await_count == 1


def test_bare_list_passthrough_unchanged(monkeypatch: pytest.MonkeyPatch):
    """Legacy bare-list response -> reaches the formatter unchanged."""
    captured = _capture_format_input(monkeypatch)
    app = _fake_app()
    portfolio_resp = _fake_response(200, {})
    trades_resp = _fake_response(200, [{"ticker": "MSFT"}, {"ticker": "TSLA"}])
    cm, _ = _patch_client_with([portfolio_resp, trades_resp])
    with cm:
        _run(scheduler_mod._send_evening_digest(app))
    assert captured["trades"] == [{"ticker": "MSFT"}, {"ticker": "TSLA"}]
    assert app.client.chat_postMessage.await_count == 1


def test_status_non_200_yields_empty_list(monkeypatch: pytest.MonkeyPatch):
    """status_code 500 -> trades_data falls back to [] (existing behavior)."""
    captured = _capture_format_input(monkeypatch)
    app = _fake_app()
    portfolio_resp = _fake_response(200, {})
    trades_resp = _fake_response(500, None)
    cm, _ = _patch_client_with([portfolio_resp, trades_resp])
    with cm:
        _run(scheduler_mod._send_evening_digest(app))
    assert captured["trades"] == []
    assert app.client.chat_postMessage.await_count == 1
