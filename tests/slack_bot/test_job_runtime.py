"""phase-9.1 tests for backend.slack_bot.job_runtime."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.slack_bot.job_runtime import (
    IdempotencyStore,
    IdempotencyKey,
    heartbeat,
)


def test_idempotency_store_seen_mark():
    s = IdempotencyStore()
    assert not s.seen("k")
    s.mark("k")
    assert s.seen("k")


def test_daily_key_contains_date():
    k = IdempotencyKey.daily("job_a", day="2026-04-20")
    assert k == "job_a:2026-04-20"


def test_weekly_key_contains_iso_week():
    k = IdempotencyKey.weekly("job_b", iso_year_week="2026-W17")
    assert k == "job_b:2026-W17"


def test_hourly_key_contains_hour():
    k = IdempotencyKey.hourly("job_c", iso_hour="2026-04-20T02")
    assert k == "job_c:2026-04-20T02"


def test_heartbeat_success_path():
    events: list[dict] = []
    store = IdempotencyStore()
    with heartbeat("j1", sink=events.append, store=store) as state:
        state["ran"] = True
    assert events[0]["status"] == "started"
    assert events[-1]["status"] == "ok"
    assert events[-1]["duration_s"] >= 0.0


def test_heartbeat_failure_path():
    events: list[dict] = []
    store = IdempotencyStore()
    with heartbeat("j2", sink=events.append, store=store):
        raise RuntimeError("boom")
    assert events[-1]["status"] == "failed"
    assert "boom" in events[-1]["error"]


def test_heartbeat_idempotency_dedup():
    """Second run with the same idempotency_key is skipped."""
    events: list[dict] = []
    store = IdempotencyStore()
    key = "daily:2026-04-20"
    # First run: ok
    with heartbeat("daily", idempotency_key=key, sink=events.append, store=store):
        pass
    assert store.seen(key)
    # Second run: skipped
    events2: list[dict] = []
    with heartbeat("daily", idempotency_key=key, sink=events2.append, store=store) as state:
        assert state["skipped"] is True
    assert events2[0]["status"] == "skipped_idempotent"


def test_failed_run_does_not_mark_idempotent():
    """A failed run should NOT mark the key -- allows retry."""
    events: list[dict] = []
    store = IdempotencyStore()
    key = "x:2026-04-20"
    with heartbeat("x", idempotency_key=key, sink=events.append, store=store):
        raise RuntimeError("fail")
    # Key should not be marked -> retry path open
    assert not store.seen(key)


def test_module_ascii():
    mod = (_REPO_ROOT / "backend" / "slack_bot" / "job_runtime.py").read_bytes()
    mod.decode("ascii")
