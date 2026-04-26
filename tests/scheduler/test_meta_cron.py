"""phase-10.7.6 unit tests for the weekly meta-evolution cron wiring.

Mirrors the StubScheduler pattern from
tests/slack_bot/test_scheduler_phase9.py:14-21 -- a minimal class with
.add_job() that records calls. No live scheduler thread, no DST timer,
no asyncio loop. Trigger semantics are verified by instantiating
APScheduler 3.x's CronTrigger directly + get_next_fire_time(None, ref).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apscheduler.triggers.cron import CronTrigger  # noqa: E402

from backend.meta_evolution.cron import (  # noqa: E402
    JOB_ID,
    TIMEZONE,
    register_meta_evolution_cron,
    run_meta_evolution_cycle,
)


class StubScheduler:
    """Records add_job calls without launching anything."""

    def __init__(self):
        self.jobs: list[dict] = []
        self.raise_on_add = False

    def add_job(self, func, *, trigger, id, replace_existing=False, **kwargs):
        if self.raise_on_add:
            raise RuntimeError("simulated add_job failure")
        self.jobs.append(
            {
                "func": func,
                "trigger": trigger,
                "id": id,
                "replace_existing": replace_existing,
                "kwargs": kwargs,
            }
        )


# ----------------------
# Registration tests
# ----------------------

def test_register_adds_job_to_scheduler():
    s = StubScheduler()
    job_id = register_meta_evolution_cron(s)
    assert job_id == JOB_ID
    assert len(s.jobs) == 1
    assert s.jobs[0]["id"] == JOB_ID


def test_job_id_is_meta_evolution_weekly():
    """Constant + registered id must match the canonical name."""
    assert JOB_ID == "meta_evolution_weekly"
    s = StubScheduler()
    register_meta_evolution_cron(s)
    assert s.jobs[0]["id"] == "meta_evolution_weekly"


def test_register_passes_replace_existing_true():
    """Idempotent reload requires replace_existing=True."""
    s = StubScheduler()
    register_meta_evolution_cron(s, replace_existing=True)
    register_meta_evolution_cron(s, replace_existing=True)
    assert all(j["replace_existing"] is True for j in s.jobs)


def test_register_uses_cron_trigger_with_sunday_2am_kwargs():
    s = StubScheduler()
    register_meta_evolution_cron(s)
    job = s.jobs[0]
    assert job["trigger"] == "cron"
    assert job["kwargs"]["day_of_week"] == "sun"
    assert job["kwargs"]["hour"] == 2
    assert job["kwargs"]["minute"] == 0


def test_timezone_is_explicitly_new_york():
    """Timezone kwarg must be ZoneInfo('America/New_York'), not a bare string."""
    s = StubScheduler()
    register_meta_evolution_cron(s)
    tz = s.jobs[0]["kwargs"]["timezone"]
    assert isinstance(tz, ZoneInfo)
    assert tz.key == "America/New_York"
    assert TIMEZONE.key == "America/New_York"


def test_register_fail_open_returns_none():
    """If add_job raises, register returns None instead of propagating."""
    s = StubScheduler()
    s.raise_on_add = True
    assert register_meta_evolution_cron(s) is None


# ----------------------
# Trigger semantics (direct CronTrigger instantiation)
# ----------------------

def test_trigger_fires_sunday_2am_et():
    """A CronTrigger configured the same way must next-fire on a Sunday at 02:00 ET."""
    trigger = CronTrigger(
        day_of_week="sun",
        hour=2,
        minute=0,
        timezone=ZoneInfo("America/New_York"),
    )
    # Reference: Monday 2026-01-05 12:00 ET -> next Sunday 2026-01-11 02:00 ET
    monday = datetime(2026, 1, 5, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    nxt = trigger.get_next_fire_time(None, monday)
    assert nxt is not None
    assert nxt.weekday() == 6  # Sunday
    assert nxt.hour == 2
    assert nxt.minute == 0
    assert nxt.tzinfo is not None
    # Must be the next Sunday after the Monday reference
    assert nxt.date() == datetime(2026, 1, 11).date()


# ----------------------
# Cycle execution tests
# ----------------------

def test_run_cycle_calls_cron_allocator(monkeypatch, tmp_path):
    """run_meta_evolution_cycle must invoke cron_allocator.allocate."""
    calls = {"n": 0}

    def fake_allocate(yaml_path):
        calls["n"] += 1
        return {"slot_x": 1}

    monkeypatch.setattr(
        "backend.meta_evolution.cron_allocator.allocate", fake_allocate
    )
    out = run_meta_evolution_cycle(
        cron_budget_yaml=tmp_path / "fake.yaml",
        provider_budget_yaml=tmp_path / "fake_p.yaml",
    )
    assert calls["n"] == 1
    assert out["cron_allocations"] == {"slot_x": 1}


def test_run_cycle_calls_provider_rebalancer(monkeypatch, tmp_path):
    def fake_p(yaml_path):
        return {"anthropic": 4.0}

    monkeypatch.setattr(
        "backend.meta_evolution.provider_rebalancer.allocate", fake_p
    )
    out = run_meta_evolution_cycle(
        cron_budget_yaml=tmp_path / "missing_cron.yaml",
        provider_budget_yaml=tmp_path / "any.yaml",
    )
    assert out["provider_allocations"] == {"anthropic": 4.0}


def test_run_cycle_handles_sub_failures_fail_open(monkeypatch, tmp_path):
    """If a sub-call raises, the cycle still returns and records the error."""

    def boom(yaml_path):
        raise RuntimeError("yaml gone")

    monkeypatch.setattr(
        "backend.meta_evolution.cron_allocator.allocate", boom
    )
    monkeypatch.setattr(
        "backend.meta_evolution.provider_rebalancer.allocate", boom
    )
    out = run_meta_evolution_cycle(
        cron_budget_yaml=tmp_path / "x.yaml",
        provider_budget_yaml=tmp_path / "y.yaml",
    )
    # Errors recorded but cycle did not raise
    assert any(e["step"] == "cron_allocator" for e in out["errors"])
    assert any(e["step"] == "provider_rebalancer" for e in out["errors"])
    # archetype_library is pure import + len() so it should still succeed
    assert isinstance(out["archetype_count"], int)
    assert out["archetype_count"] > 0
    assert "started_at" in out and "finished_at" in out


def test_run_cycle_returns_well_formed_dict(monkeypatch, tmp_path):
    """Top-level keys present even when sub-calls succeed."""
    monkeypatch.setattr(
        "backend.meta_evolution.cron_allocator.allocate", lambda p: {"a": 1}
    )
    monkeypatch.setattr(
        "backend.meta_evolution.provider_rebalancer.allocate",
        lambda p: {"anthropic": 2.0},
    )
    out = run_meta_evolution_cycle(
        cron_budget_yaml=tmp_path / "c.yaml",
        provider_budget_yaml=tmp_path / "p.yaml",
    )
    for key in (
        "started_at",
        "finished_at",
        "duration_seconds",
        "cron_allocations",
        "provider_allocations",
        "archetype_count",
        "errors",
    ):
        assert key in out, f"missing key {key} in cycle result"
    assert out["errors"] == []
    assert out["duration_seconds"] >= 0
