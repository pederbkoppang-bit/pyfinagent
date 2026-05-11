"""phase-23.5.13.2: launchctl-print bridge for /api/jobs/all launchd block.

The bridge replaces the prior hardcoded `status="manifest"` placeholder
for launchd entries with a live `launchctl print gui/<uid>/<label>` probe.
This file pins the state-to-status mapping + cache behavior so future
edits don't silently regress the dashboard.
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.api import cron_dashboard_api as cda


# ── _classify_launchctl_state ─────────────────────────────────────


def test_classify_running_returns_running():
    assert cda._classify_launchctl_state("running", None) == "running"


def test_classify_running_with_exit_still_running():
    # `last exit code` may persist across restarts; state=running takes priority.
    assert cda._classify_launchctl_state("running", 0) == "running"


def test_classify_not_running_no_exit_returns_ok():
    """Never-fired job or pre-exit-record state -> ok (clean)."""
    assert cda._classify_launchctl_state("not running", None) == "ok"


def test_classify_not_running_clean_exit_returns_ok():
    assert cda._classify_launchctl_state("not running", 0) == "ok"


def test_classify_not_running_sigterm_returns_ok():
    """SIGTERM = -15 = normal KeepAlive cycle, not a failure."""
    assert cda._classify_launchctl_state("not running", -15) == "ok"


def test_classify_not_running_failed_exit_returns_failed():
    assert cda._classify_launchctl_state("not running", 1) == "failed"
    assert cda._classify_launchctl_state("not running", 127) == "failed"


def test_classify_unknown_state_returns_unknown():
    assert cda._classify_launchctl_state(None, None) == "unknown"
    assert cda._classify_launchctl_state("weird-state", 0) == "unknown"


# ── _probe_launchctl ──────────────────────────────────────────────


def _fake_completed(returncode: int, stdout: str = "", stderr: str = ""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_probe_running_state(monkeypatch: pytest.MonkeyPatch):
    out = """
    com.pyfinagent.backend = {
        active count = 1
        path = /Users/u/Library/LaunchAgents/com.pyfinagent.backend.plist
        state = running
        pid = 12345
        runs = 11
    };
    """
    monkeypatch.setattr(cda.subprocess, "run", lambda *a, **k: _fake_completed(0, out))
    r = cda._probe_launchctl("com.pyfinagent.backend")
    assert r["status"] == "running"
    assert r["pid"] == 12345
    assert r["runs"] == 11
    assert r["last_exit_code"] is None


def test_probe_failed_with_nonzero_exit(monkeypatch: pytest.MonkeyPatch):
    out = """
    com.pyfinagent.autoresearch = {
        state = not running
        last exit code = 1
        runs = 4
    };
    """
    monkeypatch.setattr(cda.subprocess, "run", lambda *a, **k: _fake_completed(0, out))
    r = cda._probe_launchctl("com.pyfinagent.autoresearch")
    assert r["status"] == "failed"
    assert r["last_exit_code"] == 1


def test_probe_sigterm_clean(monkeypatch: pytest.MonkeyPatch):
    out = """
    com.pyfinagent.frontend = {
        state = not running
        last exit code = -15
    };
    """
    monkeypatch.setattr(cda.subprocess, "run", lambda *a, **k: _fake_completed(0, out))
    r = cda._probe_launchctl("com.pyfinagent.frontend")
    assert r["status"] == "ok"
    assert r["last_exit_code"] == -15


def test_probe_not_loaded_returns_status(monkeypatch: pytest.MonkeyPatch):
    """returncode != 0 means launchctl couldn't find the label (booted out)."""
    monkeypatch.setattr(
        cda.subprocess, "run",
        lambda *a, **k: _fake_completed(113, "", "Could not find specified service"),
    )
    r = cda._probe_launchctl("com.pyfinagent.mas-harness")
    assert r["status"] == "not_loaded"


def test_probe_timeout_returns_unknown(monkeypatch: pytest.MonkeyPatch):
    def _boom(*a, **k):
        raise cda.subprocess.TimeoutExpired(cmd="launchctl", timeout=5)

    monkeypatch.setattr(cda.subprocess, "run", _boom)
    r = cda._probe_launchctl("any.label")
    assert r["status"] == "unknown"


def test_probe_oserror_returns_unknown(monkeypatch: pytest.MonkeyPatch):
    def _boom(*a, **k):
        raise OSError("launchctl missing")

    monkeypatch.setattr(cda.subprocess, "run", _boom)
    r = cda._probe_launchctl("any.label")
    assert r["status"] == "unknown"


# ── _launchctl_state cache ────────────────────────────────────────


def test_cache_hit_returns_same_dict_without_reinvoking(monkeypatch: pytest.MonkeyPatch):
    """A second call within TTL must NOT invoke subprocess.run again."""
    cda._LAUNCHCTL_CACHE.clear()
    call_count = {"n": 0}

    def _counting_run(*a, **k):
        call_count["n"] += 1
        return _fake_completed(0, "state = running\npid = 1\nruns = 1\n")

    monkeypatch.setattr(cda.subprocess, "run", _counting_run)
    r1 = cda._launchctl_state("test.label")
    r2 = cda._launchctl_state("test.label")
    assert r1 is r2 or r1 == r2
    assert call_count["n"] == 1, f"expected 1 subprocess call (cached), got {call_count['n']}"


def test_cache_miss_after_ttl_re_probes(monkeypatch: pytest.MonkeyPatch):
    """After TTL elapses, the cache must re-invoke subprocess.run."""
    cda._LAUNCHCTL_CACHE.clear()
    call_count = {"n": 0}

    def _counting_run(*a, **k):
        call_count["n"] += 1
        return _fake_completed(0, "state = running\npid = 1\nruns = 1\n")

    monkeypatch.setattr(cda.subprocess, "run", _counting_run)
    cda._launchctl_state("test.label")
    # Force the cache entry to be older than TTL
    cached_dict, _ = cda._LAUNCHCTL_CACHE["test.label"]
    cda._LAUNCHCTL_CACHE["test.label"] = (cached_dict, time.monotonic() - cda._LAUNCHCTL_TTL_SECONDS - 1.0)
    cda._launchctl_state("test.label")
    assert call_count["n"] == 2, f"expected 2 subprocess calls (one fresh, one re-probe), got {call_count['n']}"


# ── End-to-end /api/jobs/all integration ──────────────────────────


def test_jobs_all_launchd_block_uses_bridge(monkeypatch: pytest.MonkeyPatch):
    """All 6 launchd entries must surface their probed status, never 'manifest'."""
    cda._RUNNING_SCHEDULERS.clear()
    monkeypatch.setattr(cda.job_status_api, "get_registry_snapshot", lambda: {})

    fake_running = {
        "status": "running",
        "last_exit_code": None,
        "pid": 1,
        "runs": 1,
        "next_run": None,
        "last_run": None,
    }
    monkeypatch.setattr(cda, "_launchctl_state", lambda label: fake_running)

    import asyncio
    body = asyncio.run(cda.get_all_jobs())
    ld = [j for j in body["jobs"] if j["source"] == "launchd"]
    assert len(ld) == 6
    for j in ld:
        assert j["status"] == "running"
        assert j["status"] != "manifest"
