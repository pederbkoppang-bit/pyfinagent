"""phase-23.2.23: regression guard for /api/jobs/all and /api/logs/tail.

Tests the cron-dashboard endpoints in isolation (FastAPI TestClient).
Auth middleware is bypassed in TestClient because no token is required at
the route layer -- middleware reads the cookie / header and would 401 in
a real session. The tests below exercise the route logic, NOT the auth
middleware (auth coverage lives in tests/api/test_auth_middleware.py).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.api import cron_dashboard_api as cda


# ── /api/jobs/all ─────────────────────────────────────────────────


def _fake_job(job_id: str, schedule: str = "cron mon-fri 14:00 ET", paused: bool = False):
    """Build an APScheduler-shaped fake job for introspection."""
    return SimpleNamespace(
        id=job_id,
        name=job_id,
        trigger=schedule,
        next_run_time=None if paused else SimpleNamespace(isoformat=lambda: "2026-05-08T18:00:00+00:00"),
    )


def test_jobs_all_returns_envelope_shape():
    fake_scheduler = SimpleNamespace(get_jobs=lambda: [_fake_job("paper_trading_daily")])
    cda._RUNNING_SCHEDULERS.clear()
    cda.register_scheduler("main", fake_scheduler)

    import asyncio
    body = asyncio.run(cda.get_all_jobs())

    assert "jobs" in body
    assert "generated_at" in body
    assert "n_total" in body
    assert body["n_total"] == len(body["jobs"])

    # Every job has the documented keys
    for j in body["jobs"]:
        for key in ("id", "source", "schedule", "next_run", "last_run", "status", "description"):
            assert key in j, f"missing key {key!r} in job {j}"


def test_jobs_all_includes_live_apscheduler_jobs():
    fake_scheduler = SimpleNamespace(get_jobs=lambda: [
        _fake_job("paper_trading_daily"),
        _fake_job("queue_processor", schedule="interval 5s"),
    ])
    cda._RUNNING_SCHEDULERS.clear()
    cda.register_scheduler("main", fake_scheduler)

    import asyncio
    body = asyncio.run(cda.get_all_jobs())

    ids = [j["id"] for j in body["jobs"]]
    assert "paper_trading_daily" in ids
    assert "queue_processor" in ids

    # Live jobs come from main_apscheduler source
    main_jobs = [j for j in body["jobs"] if j["source"] == "main_apscheduler"]
    assert len(main_jobs) == 2


def test_jobs_all_includes_static_slack_bot_manifest():
    cda._RUNNING_SCHEDULERS.clear()
    import asyncio
    body = asyncio.run(cda.get_all_jobs())
    slack_jobs = [j for j in body["jobs"] if j["source"] == "slack_bot"]
    assert len(slack_jobs) == len(cda._SLACK_BOT_JOBS)
    ids = {j["id"] for j in slack_jobs}
    assert "morning_digest" in ids
    assert "cost_budget_watcher" in ids


def test_jobs_all_includes_static_launchd_manifest():
    cda._RUNNING_SCHEDULERS.clear()
    import asyncio
    body = asyncio.run(cda.get_all_jobs())
    launchd_jobs = [j for j in body["jobs"] if j["source"] == "launchd"]
    assert len(launchd_jobs) == len(cda._LAUNCHD_JOBS)
    assert launchd_jobs[0]["id"] == "com.pyfinagent.backend-watchdog"


def test_jobs_all_handles_introspection_failure_gracefully():
    """If a registered scheduler raises on get_jobs(), we still return manifests."""
    class _BoomScheduler:
        def get_jobs(self):
            raise RuntimeError("simulated scheduler failure")

    cda._RUNNING_SCHEDULERS.clear()
    cda.register_scheduler("main", _BoomScheduler())

    import asyncio
    body = asyncio.run(cda.get_all_jobs())
    # No live jobs, but slack_bot + launchd manifests still present
    main_jobs = [j for j in body["jobs"] if j["source"] == "main_apscheduler"]
    assert main_jobs == []
    assert len(body["jobs"]) >= len(cda._SLACK_BOT_JOBS) + len(cda._LAUNCHD_JOBS)


# ── /api/logs/tail ─────────────────────────────────────────────────


def test_logs_tail_rejects_unknown_log_key():
    """Path traversal is impossible: client passes a KEY, server resolves Path."""
    from fastapi import HTTPException
    import asyncio
    with pytest.raises(HTTPException) as exc:
        asyncio.run(cda.get_log_tail(log="etc/passwd", lines=20))
    assert exc.value.status_code == 400


def test_logs_tail_rejects_traversal_attempt():
    """Even a key shaped like a real path is rejected (not in allowlist)."""
    from fastapi import HTTPException
    import asyncio
    for evil in ("../../../etc/passwd", "/etc/passwd", "backend.log/../etc/passwd"):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(cda.get_log_tail(log=evil, lines=20))
        assert exc.value.status_code == 400, f"unexpected status for {evil!r}"


def test_logs_tail_returns_last_n_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Happy path: write a fake log, tail-read, assert the LAST n lines."""
    fake = tmp_path / "fake_backend.log"
    fake.write_text("\n".join(f"line {i}" for i in range(1, 51)) + "\n", encoding="utf-8")

    monkeypatch.setattr(cda, "_log_paths", lambda: {"backend": fake})

    import asyncio
    body = asyncio.run(cda.get_log_tail(log="backend", lines=10))

    assert body["log"] == "backend"
    assert body["n_returned"] == 10
    assert body["exists"] is True
    assert body["lines"][0] == "line 41"
    assert body["lines"][-1] == "line 50"


def test_logs_tail_clamps_lines_to_max(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """lines=50000 is clamped to 1000 ceiling."""
    fake = tmp_path / "big.log"
    fake.write_text("\n".join(f"row {i}" for i in range(1, 1500)) + "\n", encoding="utf-8")
    monkeypatch.setattr(cda, "_log_paths", lambda: {"backend": fake})

    import asyncio
    body = asyncio.run(cda.get_log_tail(log="backend", lines=5000))
    assert body["n_returned"] == 1000


def test_logs_tail_clamps_lines_to_min(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """lines=1 is clamped to the 10 floor."""
    fake = tmp_path / "tiny.log"
    fake.write_text("\n".join(f"r{i}" for i in range(1, 30)) + "\n", encoding="utf-8")
    monkeypatch.setattr(cda, "_log_paths", lambda: {"backend": fake})

    import asyncio
    body = asyncio.run(cda.get_log_tail(log="backend", lines=1))
    # Query(..., ge=1) accepts 1; the server clamps it up to _LINES_MIN=10.
    assert body["n_returned"] == 10


def test_logs_tail_returns_empty_when_log_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A registered key whose file doesn't exist yet returns empty + exists=False."""
    monkeypatch.setattr(cda, "_log_paths", lambda: {"backend": tmp_path / "does_not_exist.log"})

    import asyncio
    body = asyncio.run(cda.get_log_tail(log="backend", lines=20))
    assert body["lines"] == []
    assert body["n_returned"] == 0
    assert body["exists"] is False
    assert body["total_size_bytes"] == 0
