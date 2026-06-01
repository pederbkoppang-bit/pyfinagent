"""phase-51.3: Slack morning/evening digests skip non-trading days.

The digests fired 7 days/week and re-sent the prior trading day's data on
weekends/holidays. The guard early-returns BEFORE any HTTP fetch / chat_postMessage
when today (ET) is not a US trading session. These tests invoke the real async
digest functions with a probe that flags whether execution reached the httpx body,
so a SKIP is proven by the body NEVER running (not merely by "no post").

$0, no network, no Slack: httpx + the trading-day helper are monkeypatched.
"""
import asyncio
import types

import pytest

import backend.slack_bot.scheduler as sched


class _Reached(Exception):
    pass


def _probe():
    """A fake httpx.AsyncClient that records instantiation then short-circuits."""
    flag = {"reached": False}

    class _Probe:
        def __init__(self, *a, **k):
            flag["reached"] = True
            raise _Reached()  # caught by the digest's try/except

    return _Probe, flag


def _fake_app():
    posted = []

    class _Client:
        async def chat_postMessage(self, **kw):
            posted.append(kw)

    return types.SimpleNamespace(client=_Client()), posted


def test_is_us_trading_day_now_delegates_to_is_trading_day(monkeypatch):
    monkeypatch.setattr("backend.backtest.markets.is_trading_day", lambda d, market="US": False)
    assert sched._is_us_trading_day_now() is False
    monkeypatch.setattr("backend.backtest.markets.is_trading_day", lambda d, market="US": True)
    assert sched._is_us_trading_day_now() is True


@pytest.mark.parametrize("fn_name", ["_send_morning_digest", "_send_evening_digest"])
def test_digest_skips_on_non_trading_day(monkeypatch, fn_name):
    probe, flag = _probe()
    monkeypatch.setattr(sched, "_is_us_trading_day_now", lambda: False)
    monkeypatch.setattr(sched.httpx, "AsyncClient", probe)
    monkeypatch.setattr(sched, "_route_exception_to_p1", lambda *a, **k: None)
    app, posted = _fake_app()
    asyncio.run(getattr(sched, fn_name)(app))
    assert flag["reached"] is False, "guard did NOT early-return -- the digest body ran"
    assert posted == [], "digest posted to Slack on a non-trading day"


@pytest.mark.parametrize("fn_name", ["_send_morning_digest", "_send_evening_digest"])
def test_digest_proceeds_on_trading_day(monkeypatch, fn_name):
    probe, flag = _probe()
    monkeypatch.setattr(sched, "_is_us_trading_day_now", lambda: True)
    monkeypatch.setattr(sched.httpx, "AsyncClient", probe)
    monkeypatch.setattr(sched, "_route_exception_to_p1", lambda *a, **k: None)
    app, _ = _fake_app()
    asyncio.run(getattr(sched, fn_name)(app))
    assert flag["reached"] is True, "guard wrongly skipped a US trading day"
