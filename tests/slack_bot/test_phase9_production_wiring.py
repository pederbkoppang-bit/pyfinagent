"""phase-23.6.1: regression guards for the phase-9 production-fn wiring.

Tests assert:
  1. `register_phase9_jobs()` accepts `app` + `loop` kwargs.
  2. When `app` and `loop` are passed, the registered job is a
     `functools.partial` wrapping `run` with the expected `prod_fns` keys.
  3. When `app` or `loop` is None, the bare `run` is registered (preserves
     unit-test injection paths for the existing per-job tests). EXCEPTION
     (phase-47.1): `daily_price_refresh` always resolves to the module-level
     `run_production` (full-universe OHLCV ingest), never the bare 5-ticker
     `run` and never a prod-fn partial.
  4. Each factory in `_production_fns` returns a callable AND defers the
     external import (yfinance / fredapi / google.cloud.bigquery) until
     call-time — so this test module loads even without those deps.
"""
from __future__ import annotations

import asyncio
import functools
from unittest.mock import MagicMock

import pytest

from backend.slack_bot import scheduler as scheduler_mod
from backend.slack_bot.jobs import _production_fns as pf


class _StubScheduler:
    """Records add_job calls without executing anything."""

    def __init__(self) -> None:
        self.added: list[dict] = []

    def add_job(self, func, *, trigger=None, id=None, replace_existing=False, **kwargs):
        self.added.append({
            "func": func,
            "trigger": trigger,
            "id": id,
            "replace_existing": replace_existing,
            "kwargs": kwargs,
        })


# ── register_phase9_jobs signature + behaviour ────────────────────


def test_register_accepts_app_and_loop_kwargs():
    """phase-23.6.1: the new app+loop kwargs must be present and have None defaults."""
    import inspect
    sig = inspect.signature(scheduler_mod.register_phase9_jobs)
    assert "app" in sig.parameters
    assert "loop" in sig.parameters
    assert sig.parameters["app"].default is None
    assert sig.parameters["loop"].default is None


def test_register_without_app_returns_bare_run(monkeypatch: pytest.MonkeyPatch):
    """When app+loop omitted, prod-fn jobs register their bare module `run` (no partial).
    phase-47.1 exception: daily_price_refresh always resolves to run_production."""
    sched = _StubScheduler()
    registered = scheduler_mod.register_phase9_jobs(sched, replace_existing=True)
    assert "daily_price_refresh" in registered

    # weekly_fred_refresh still follows the bare-run pattern when app is omitted.
    fred = next(e for e in sched.added if e["id"] == "weekly_fred_refresh")
    assert not isinstance(fred["func"], functools.partial)
    from backend.slack_bot.jobs import weekly_fred_refresh as wfr
    assert fred["func"] is wfr.run

    # phase-47.1: daily_price_refresh resolves to the module-level run_production
    # (full-universe ingest -> historical_prices), NOT the bare 5-ticker run.
    daily = next(e for e in sched.added if e["id"] == "daily_price_refresh")
    assert not isinstance(daily["func"], functools.partial)
    from backend.slack_bot.jobs import daily_price_refresh as dpr
    assert daily["func"] is dpr.run_production


def test_register_with_app_uses_functools_partial(monkeypatch: pytest.MonkeyPatch):
    """When app+loop provided, registered func is a partial(run, **prod_fns)."""
    sched = _StubScheduler()
    fake_app = MagicMock()
    fake_app.client = MagicMock()
    fake_loop = MagicMock(spec=asyncio.AbstractEventLoop)

    # Stub get_settings to avoid Slack env requirement
    fake_settings = MagicMock()
    fake_settings.slack_channel_id = "C_TEST"
    monkeypatch.setattr(scheduler_mod, "get_settings", lambda: fake_settings)

    registered = scheduler_mod.register_phase9_jobs(
        sched, replace_existing=True, app=fake_app, loop=fake_loop,
    )
    assert "daily_price_refresh" in registered

    # Each prod-wired job should be a partial wrapping its module's run + the
    # expected prod_fns keys. phase-47.1: daily_price_refresh is NO LONGER in
    # this set -- it resolves to run_production (asserted separately below).
    expected = {
        "weekly_fred_refresh":      {"fetch_fn", "write_fn"},
        "nightly_outcome_rebuild":  {"ledger_fetch_fn", "outcome_write_fn"},
        "cost_budget_watcher":      {"alert_fn"},
        "weekly_data_integrity":    {"alert_fn"},
    }
    for entry in sched.added:
        jid = entry["id"]
        if jid in expected:
            assert isinstance(entry["func"], functools.partial), (
                f"{jid} expected functools.partial, got {type(entry['func'])}"
            )
            assert set(entry["func"].keywords.keys()) == expected[jid], (
                f"{jid} prod_fns keys mismatch: got {set(entry['func'].keywords.keys())} expected {expected[jid]}"
            )

    # phase-47.1: daily_price_refresh is NOT prod-fn-wrapped even with app+loop;
    # it resolves to the module-level run_production (full-universe ingest_prices).
    from backend.slack_bot.jobs import daily_price_refresh as dpr
    daily = next(e for e in sched.added if e["id"] == "daily_price_refresh")
    assert not isinstance(daily["func"], functools.partial), \
        "daily_price_refresh must NOT be prod-fn-wrapped after phase-47.1"
    assert daily["func"] is dpr.run_production


def test_register_with_app_but_none_loop_returns_bare_run(monkeypatch: pytest.MonkeyPatch):
    """If only one of app/loop is provided, no partial-application happens
    (defensive: both must be present to safely build the alert_fn closures).
    """
    sched = _StubScheduler()
    fake_app = MagicMock()
    registered = scheduler_mod.register_phase9_jobs(
        sched, replace_existing=True, app=fake_app, loop=None,
    )
    assert "daily_price_refresh" in registered
    daily = next(e for e in sched.added if e["id"] == "daily_price_refresh")
    assert not isinstance(daily["func"], functools.partial)


# ── Factory shape + lazy-import guarantees ─────────────────────────


def test_make_price_fetch_fn_returns_callable():
    fn = pf.make_price_fetch_fn()
    assert callable(fn)
    # Lazy-import test: even without yfinance installed (or with import error
    # patched), constructing the closure must NOT raise.


def test_make_price_write_fn_returns_callable():
    assert callable(pf.make_price_write_fn())


def test_make_fred_fetch_fn_returns_callable():
    assert callable(pf.make_fred_fetch_fn())


def test_make_fred_write_fn_returns_callable():
    assert callable(pf.make_fred_write_fn())


def test_make_ledger_fetch_fn_returns_callable():
    assert callable(pf.make_ledger_fetch_fn())


def test_make_outcome_write_fn_returns_callable():
    assert callable(pf.make_outcome_write_fn())


def test_make_alert_fn_for_budget_returns_callable():
    fake_app = MagicMock()
    fake_loop = MagicMock(spec=asyncio.AbstractEventLoop)
    fn = pf.make_alert_fn_for_budget(fake_app, fake_loop, "C_TEST")
    assert callable(fn)


def test_make_alert_fn_for_integrity_returns_callable():
    fake_app = MagicMock()
    fake_loop = MagicMock(spec=asyncio.AbstractEventLoop)
    fn = pf.make_alert_fn_for_integrity(fake_app, fake_loop, "C_TEST")
    assert callable(fn)


# ── alert_fn fail-open semantics ──────────────────────────────────


def test_alert_fn_for_budget_swallows_post_failure(monkeypatch: pytest.MonkeyPatch):
    """A Slack-post exception inside the closure must NOT raise out of alert_fn."""
    fake_app = MagicMock()
    fake_loop = MagicMock(spec=asyncio.AbstractEventLoop)

    def _boom(coro, loop):
        raise RuntimeError("simulated rate limit")

    monkeypatch.setattr(pf.asyncio, "run_coroutine_threadsafe", _boom)
    fn = pf.make_alert_fn_for_budget(fake_app, fake_loop, "C_TEST")
    # Should not raise.
    fn("daily", {"reason": "test", "state": "tripped"})


def test_alert_fn_for_integrity_handles_empty_drifts(monkeypatch: pytest.MonkeyPatch):
    """Empty drifts list is a no-op (do not post a Slack message)."""
    fake_app = MagicMock()
    fake_loop = MagicMock(spec=asyncio.AbstractEventLoop)

    posts: list = []
    def _record(coro, loop):
        posts.append(coro)
        # Return an object with .result() that does nothing
        m = MagicMock()
        m.result = MagicMock(return_value=None)
        return m

    monkeypatch.setattr(pf.asyncio, "run_coroutine_threadsafe", _record)
    fn = pf.make_alert_fn_for_integrity(fake_app, fake_loop, "C_TEST")
    fn([])
    assert posts == [], "empty drifts should not trigger a Slack post"
