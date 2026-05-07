"""phase-23.3.2: regression guard for slack-bot heartbeat-push wiring.

Pre-fix: slack-bot APScheduler jobs fired but the events were sunk into
logger.info instead of POSTed to /api/jobs/heartbeat. /api/jobs/status
returned 'never_run' for every job for a month of slack-bot uptime.

Post-fix: backend/slack_bot/scheduler.py defines _aps_to_heartbeat
(an APScheduler event listener) that POSTs each terminal job event to
http://127.0.0.1:8000/api/jobs/heartbeat.

These tests assert:
- The listener POSTs the right payload shape.
- The listener swallows httpx exceptions (fail-open).
- backend/api/job_status_api.py::_JOB_NAMES now includes the 4 core
  slack-bot job ids and record_heartbeat updates a core job from
  never_run to ok.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.slack_bot import scheduler as slack_scheduler
from backend.api import job_status_api


def _fake_event(job_id: str, exception=None):
    """APScheduler-shaped event SimpleNamespace."""
    return SimpleNamespace(job_id=job_id, exception=exception)


def test_listener_posts_to_heartbeat_endpoint():
    """Happy path: a successful job event posts ok status."""
    captured = {}

    class _FakeClient:
        def __init__(self, *a, **kw):
            captured["init_kwargs"] = kw

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None):
            captured["url"] = url
            captured["payload"] = json
            return SimpleNamespace(status_code=200)

    with patch.object(slack_scheduler.httpx, "Client", _FakeClient):
        slack_scheduler._aps_to_heartbeat(_fake_event("morning_digest"))

    assert captured["url"] == "http://127.0.0.1:8000/api/jobs/heartbeat"
    assert captured["payload"]["job"] == "morning_digest"
    assert captured["payload"]["status"] == "ok"
    assert captured["payload"]["error"] is None
    assert "finished_at" in captured["payload"]


def test_listener_marks_failed_when_exception_set():
    captured = {}

    class _FakeClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, json=None):
            captured["payload"] = json

    with patch.object(slack_scheduler.httpx, "Client", _FakeClient):
        slack_scheduler._aps_to_heartbeat(
            _fake_event("watchdog_health_check", exception=RuntimeError("boom")),
        )

    assert captured["payload"]["status"] == "failed"
    assert "boom" in captured["payload"]["error"]


def test_listener_fails_open_when_backend_unreachable(caplog):
    """Listener must NEVER raise -- the slack-bot scheduler must keep going."""
    import logging

    class _BoomClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, *a, **kw):
            raise ConnectionError("simulated backend down")

    with patch.object(slack_scheduler.httpx, "Client", _BoomClient):
        with caplog.at_level(logging.WARNING, logger="backend.slack_bot.scheduler"):
            # MUST NOT raise
            slack_scheduler._aps_to_heartbeat(_fake_event("morning_digest"))

    warns = [r for r in caplog.records if "fail-open" in r.getMessage()]
    assert len(warns) == 1, "fail-open warn must be logged"


def test_job_names_includes_4_core_slack_bot_jobs():
    """Pre-seed registry must include the 4 core ids so /api/jobs/status
    returns 11 rows after this phase."""
    assert "morning_digest" in job_status_api._JOB_NAMES
    assert "evening_digest" in job_status_api._JOB_NAMES
    assert "watchdog_health_check" in job_status_api._JOB_NAMES
    assert "prompt_leak_redteam" in job_status_api._JOB_NAMES
    # And the original 7 still present
    assert "cost_budget_watcher" in job_status_api._JOB_NAMES
    assert len(job_status_api._JOB_NAMES) == 11


def test_record_heartbeat_updates_core_job_from_never_run():
    """End-to-end: a core-job heartbeat POST should flip the registry row."""
    from datetime import datetime, timezone

    # Reset state for the morning_digest row to never_run
    with job_status_api._lock:
        job_status_api._registry["morning_digest"] = {"name": "morning_digest"}

    pre = job_status_api._registry["morning_digest"].copy()
    assert pre.get("status") in (None, "never_run")

    job_status_api.record_heartbeat({
        "job": "morning_digest",
        "status": "ok",
        "started_at": "2026-05-07T13:00:00+00:00",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": 1.23,
    })

    post = job_status_api._registry["morning_digest"]
    assert post.get("status") == "ok", \
        f"expected status ok after heartbeat; got {post}"
    assert post.get("last_run_at") is not None
