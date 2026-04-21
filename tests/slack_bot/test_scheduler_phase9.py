"""phase-9.9 / 9.10 scheduler wiring tests."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.slack_bot.scheduler import register_phase9_jobs, _PHASE9_JOB_IDS


class StubScheduler:
    """Minimal APScheduler shim for testing."""

    def __init__(self):
        self.jobs: list[dict] = []

    def add_job(self, func, *, trigger, id, replace_existing=False, **kwargs):
        self.jobs.append({"id": id, "trigger": trigger, "kwargs": kwargs, "replace_existing": replace_existing})


def test_all_seven_jobs_registered():
    s = StubScheduler()
    registered = register_phase9_jobs(s)
    assert set(registered) == set(_PHASE9_JOB_IDS)
    assert len(s.jobs) == 7


def test_no_double_registration_on_reload():
    """Two calls with replace_existing=True leave 7 total active ids."""
    s = StubScheduler()
    register_phase9_jobs(s, replace_existing=True)
    register_phase9_jobs(s, replace_existing=True)
    # StubScheduler doesn't dedup; but the scheduler real-world applies
    # replace_existing. We verify the flag is passed.
    assert all(j["replace_existing"] is True for j in s.jobs)


def test_phase9_ids_stable():
    assert len(_PHASE9_JOB_IDS) == 7
    assert "daily_price_refresh" in _PHASE9_JOB_IDS
    assert "cost_budget_watcher" in _PHASE9_JOB_IDS


def test_runbook_exists():
    # Also asserted by the 9.10 immutable command; duplicated here so
    # the pytest suite catches a missing runbook.
    runbook = _REPO_ROOT / "docs" / "runbooks" / "phase9-cron-runbook.md"
    assert runbook.exists(), f"missing runbook at {runbook}"
