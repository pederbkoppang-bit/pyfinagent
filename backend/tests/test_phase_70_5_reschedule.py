"""phase-70.5 (P3): deposit-aware Starting-capital display + cron reschedule-on-save.

Deterministic (network-free) proofs:
  - reschedule_paper_job re-adds the daily cron with the new hour via add_job(replace_
    existing=True) when a live job exists, and is a guarded no-op when paper trading is
    off (never CREATES a job on a settings PUT).
  - paper_trading_hour is now writable (SettingsUpdate + _FIELD_TO_ENV + FullSettings).
(The Starting-capital display fix is UI -- proven by the live Playwright capture in
live_check_70.5.md.)
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from backend.api import paper_trading as ptapi


def _settings(hour=14):
    return SimpleNamespace(paper_trading_hour=hour, paper_trading_enabled=True)


def test_reschedule_noop_when_no_live_job():
    """Guard: a settings PUT must NEVER create the cron when paper trading is off."""
    orig = ptapi._scheduler
    try:
        ptapi._scheduler = MagicMock()
        ptapi._scheduler.get_job.return_value = None      # no live job
        assert ptapi.reschedule_paper_job(_settings(9)) is False
        ptapi._scheduler.add_job.assert_not_called()
    finally:
        ptapi._scheduler = orig


def test_reschedule_readds_job_with_new_hour():
    orig = ptapi._scheduler
    try:
        sched = MagicMock()
        sched.get_job.return_value = SimpleNamespace(next_run_time="2026-07-18T18:00:00Z")
        ptapi._scheduler = sched
        assert ptapi.reschedule_paper_job(_settings(hour=18)) is True
        # _add_scheduler_job -> add_job called with the new hour + replace_existing
        _, kw = sched.add_job.call_args
        assert kw["hour"] == 18 and kw["replace_existing"] is True
        assert kw["id"] == ptapi._scheduler_job_id
    finally:
        ptapi._scheduler = orig


def test_reschedule_fail_open_on_error():
    orig = ptapi._scheduler
    try:
        sched = MagicMock()
        sched.get_job.return_value = SimpleNamespace(next_run_time=None)
        sched.add_job.side_effect = RuntimeError("boom")
        ptapi._scheduler = sched
        assert ptapi.reschedule_paper_job(_settings(10)) is False   # swallowed, never raises
    finally:
        ptapi._scheduler = orig


def test_paper_trading_hour_is_writable():
    from backend.api.settings_api import SettingsUpdate, _FIELD_TO_ENV, FullSettings
    assert "paper_trading_hour" in SettingsUpdate.model_fields
    assert _FIELD_TO_ENV.get("paper_trading_hour") == "PAPER_TRADING_HOUR"
    assert "paper_trading_hour" in FullSettings.model_fields
    # bounds 0-23
    f = SettingsUpdate.model_fields["paper_trading_hour"]
    md = {m.__class__.__name__: getattr(m, "ge", None) or getattr(m, "le", None) for m in f.metadata}
    assert any(getattr(m, "ge", None) == 0 for m in f.metadata)
    assert any(getattr(m, "le", None) == 23 for m in f.metadata)
