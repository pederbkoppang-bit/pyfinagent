"""phase-23.3.3: regression guard for phase-9 job activation.

Pre-fix: register_phase9_jobs was defined but never called. The 7
phase-9 jobs (daily_price_refresh, ..., cost_budget_watcher) were
dormant since the file was added.

Post-fix: backend/slack_bot/scheduler.py::start_scheduler now invokes
register_phase9_jobs(_scheduler) after _scheduler.start(). The
mapping passes misfire_grace_time + coalesce=True per researcher's
brief.

These tests assert:
- start_scheduler source contains the register_phase9_jobs call.
- Calling register_phase9_jobs against a fake scheduler returns 7 ids
  (provided each module imports cleanly) and passes the safety
  kwargs to add_job for each.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.slack_bot import scheduler as slack_scheduler


def test_start_scheduler_source_calls_register_phase9_jobs():
    """Source-level guard: start_scheduler must call register_phase9_jobs."""
    from pathlib import Path
    src = Path(slack_scheduler.__file__).read_text()
    # Find the start_scheduler function body
    fn_start = src.find("def start_scheduler")
    assert fn_start > 0, "start_scheduler not found"
    # Find the next top-level def after start_scheduler
    fn_end = src.find("\nasync def ", fn_start + 1)
    if fn_end == -1:
        fn_end = src.find("\ndef ", fn_start + 1)
    body = src[fn_start:fn_end] if fn_end > 0 else src[fn_start:]
    assert "register_phase9_jobs(_scheduler)" in body, (
        "start_scheduler must call register_phase9_jobs(_scheduler) -- "
        "phase-23.3.3 fix to activate the 7 dormant phase-9 jobs"
    )


def test_register_phase9_passes_safety_kwargs():
    """Each add_job call must include misfire_grace_time + coalesce."""
    captured: list[dict] = []

    class _FakeScheduler:
        def add_job(self, func, **kwargs):
            captured.append(kwargs)

    # Returns the ids of jobs whose modules imported cleanly
    registered = slack_scheduler.register_phase9_jobs(
        _FakeScheduler(), replace_existing=True,
    )

    # All 7 jobs should register (modules exist on disk)
    assert len(registered) == 7, \
        f"expected 7 phase-9 jobs registered, got {len(registered)}: {registered}"
    expected_ids = {
        "daily_price_refresh", "weekly_fred_refresh", "nightly_mda_retrain",
        "hourly_signal_warmup", "nightly_outcome_rebuild",
        "weekly_data_integrity", "cost_budget_watcher",
    }
    assert set(registered) == expected_ids

    # Every captured add_job kwargs must include misfire_grace_time + coalesce
    assert len(captured) == 7
    for kwargs in captured:
        assert "misfire_grace_time" in kwargs, \
            f"job missing misfire_grace_time: {kwargs}"
        assert kwargs.get("coalesce") is True, \
            f"job missing coalesce=True: {kwargs}"
        assert kwargs.get("replace_existing") is True
        assert "id" in kwargs


def test_register_phase9_grace_times_per_tier():
    """Daily=3600, weekly=7200, hourly=600."""
    captured_by_id: dict[str, dict] = {}

    class _FakeScheduler:
        def add_job(self, func, **kwargs):
            captured_by_id[kwargs.get("id", "?")] = kwargs

    slack_scheduler.register_phase9_jobs(_FakeScheduler())

    daily = ("daily_price_refresh", "nightly_mda_retrain",
             "nightly_outcome_rebuild", "cost_budget_watcher")
    for jid in daily:
        assert captured_by_id[jid].get("misfire_grace_time") == 3600, \
            f"{jid} grace should be 3600 (daily)"

    weekly = ("weekly_fred_refresh", "weekly_data_integrity")
    for jid in weekly:
        assert captured_by_id[jid].get("misfire_grace_time") == 7200, \
            f"{jid} grace should be 7200 (weekly)"

    hourly = ("hourly_signal_warmup",)
    for jid in hourly:
        assert captured_by_id[jid].get("misfire_grace_time") == 600, \
            f"{jid} grace should be 600 (hourly)"


def test_register_phase9_fail_open_on_module_import_error(monkeypatch):
    """If a phase-9 module's import fails, the others must still register."""
    import importlib

    real_import = importlib.import_module

    def _flaky_import(name, *args, **kwargs):
        if name == "backend.slack_bot.jobs.cost_budget_watcher":
            raise ImportError("simulated module failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _flaky_import)

    captured = []

    class _FakeScheduler:
        def add_job(self, func, **kwargs):
            captured.append(kwargs.get("id"))

    registered = slack_scheduler.register_phase9_jobs(_FakeScheduler())
    # cost_budget_watcher missing; the other 6 still present
    assert "cost_budget_watcher" not in registered
    assert len(registered) == 6
