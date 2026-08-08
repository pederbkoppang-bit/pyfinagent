"""phase-85.4 criterion 4 -- "no completed cycle in N days" must be pageable.

Criterion 4 (verbatim from .claude/masterplan.json):

    the last-completed-cycle age is exposed as a health signal the existing
    watchdog can page on, so 'no completed cycle in N days' cannot again be
    discoverable only by hand-reading a jsonl

The defect these tests pin down is subtle and was live for 8 days:
`cycle_heartbeat_alarm` measured the age of the last row of ANY terminal
status, and `record_cycle_end` stamps `completed_at` on timeout rows too. So a
cycle that timed out every single weekday reset the alarm's clock every single
weekday, and the alarm stayed green while the book had not traded since
2026-07-31.

The tests therefore all share one shape: build a ledger where the OLD clock is
green and assert the NEW clock is red.
"""

from __future__ import annotations

import json
import types
from datetime import datetime, timedelta, timezone

import pytest

from backend.services import cycle_health


@pytest.fixture(autouse=True)
def _isolate_history(monkeypatch, tmp_path):
    history = tmp_path / "cycle_history.jsonl"
    monkeypatch.setattr(cycle_health, "_HISTORY_PATH", history)
    monkeypatch.setattr(cycle_health, "_HEARTBEAT_PATH", tmp_path / ".cycle_heartbeat.json")
    return history


# A Wednesday, so `is_weekday_et` is True and the weekday gate never masks a
# result. Pinned rather than "now" so the suite cannot go green/red by calendar.
_WED = datetime(2026, 8, 5, 18, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def frozen_now(monkeypatch):
    """Freeze cycle_health's clock at a known weekday."""
    monkeypatch.setattr(cycle_health, "_now_utc", lambda: _WED)
    return _WED


def _write(history, rows: list[dict]) -> None:
    history.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


def _row(status: str, ended: datetime, cycle_id: str = "c") -> dict:
    return {
        "cycle_id": cycle_id,
        "started_at": (ended - timedelta(hours=2)).isoformat(),
        "completed_at": ended.isoformat(),
        "duration_ms": 7_200_000,
        "status": status,
        "n_trades": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# The production regression, reproduced
# ─────────────────────────────────────────────────────────────────────────────


def test_daily_timeouts_keep_the_old_clock_green_and_the_new_clock_red(
    _isolate_history, frozen_now
):
    """THE 2026-07-31 -> 2026-08-08 SHAPE.

    One completed cycle 7 days back, then a fresh `timeout` row every weekday.
    The heartbeat clock reads ~24h (green). The completed clock reads ~7d (red).
    """
    history = _isolate_history
    rows = [_row("completed", frozen_now - timedelta(days=7), "old-good")]
    for d in (5, 4, 3, 2, 1):
        rows.append(_row("timeout", frozen_now - timedelta(days=d), f"to-{d}"))
    _write(history, rows)

    v = cycle_health.cycle_heartbeat_alarm()

    # Old clock: last terminal row is 1 day old, threshold 26h -> NOT stale.
    assert v["stale"] is False, v
    assert v["should_alarm"] is False, v

    # New clock: last COMPLETED row is 7 days old, threshold 4 days -> stale.
    assert v["last_terminal_status"] == "timeout"
    assert v["success_stale"] is True, v
    assert v["should_alarm_success"] is True, v
    assert v["success_age_sec"] == pytest.approx(7 * 86400, abs=1)
    assert v["last_success_at"].startswith("2026-07-29")


def test_fresh_completion_clears_the_new_clock(_isolate_history, frozen_now):
    """Mutation guard: the alarm must actually depend on the ledger, not fire
    unconditionally. A recent `completed` row makes it green."""
    _write(
        _isolate_history,
        [
            _row("completed", frozen_now - timedelta(days=7), "old"),
            _row("timeout", frozen_now - timedelta(days=2), "to"),
            _row("completed", frozen_now - timedelta(hours=20), "fresh"),
        ],
    )
    v = cycle_health.cycle_heartbeat_alarm()
    assert v["success_stale"] is False, v
    assert v["should_alarm_success"] is False, v
    assert v["success_age_sec"] == pytest.approx(20 * 3600, abs=1)


def test_friday_to_monday_weekend_gap_does_not_page(_isolate_history, monkeypatch):
    """72h Fri->Mon is the largest LEGITIMATE gap between weekday cycles and
    must stay under the threshold, or the alarm cries wolf every Monday."""
    monday = datetime(2026, 8, 10, 18, 0, 0, tzinfo=timezone.utc)  # a Monday
    monkeypatch.setattr(cycle_health, "_now_utc", lambda: monday)
    _write(_isolate_history, [_row("completed", monday - timedelta(hours=72), "fri")])

    v = cycle_health.cycle_heartbeat_alarm()
    assert v["is_weekday_et"] is True
    assert v["success_stale"] is False, (
        "a normal Fri->Mon weekend must not trip the completed-age alarm"
    )


def test_one_extra_failed_weekday_past_the_weekend_does_page(
    _isolate_history, monkeypatch
):
    """...but Fri completed + Mon failed + Tue check = 96h+ MUST page. This is
    the boundary the 96h threshold was chosen to sit on."""
    tuesday = datetime(2026, 8, 11, 18, 0, 0, tzinfo=timezone.utc)  # a Tuesday
    monkeypatch.setattr(cycle_health, "_now_utc", lambda: tuesday)
    _write(
        _isolate_history,
        [
            _row("completed", tuesday - timedelta(hours=96, minutes=1), "fri"),
            _row("timeout", tuesday - timedelta(hours=24), "mon"),
        ],
    )
    v = cycle_health.cycle_heartbeat_alarm()
    assert v["success_stale"] is True, v
    assert v["should_alarm_success"] is True, v


def test_a_ledger_with_no_completion_at_all_pages(_isolate_history, frozen_now):
    """Worst case, and the easiest one to get wrong: if NOTHING ever completed
    there is no timestamp to age, and a naive implementation returns the benign
    first-boot sentinel. That would be silence in the exact scenario the alarm
    exists for."""
    _write(
        _isolate_history,
        [_row("timeout", frozen_now - timedelta(hours=h), f"to-{h}") for h in (72, 48, 24)],
    )
    v = cycle_health.cycle_heartbeat_alarm()
    assert v["last_success_at"] is None
    assert v["success_age_sec"] is None
    assert v["success_stale"] is True, v
    assert v["should_alarm_success"] is True, v


def test_empty_ledger_is_first_boot_and_stays_quiet(_isolate_history, frozen_now):
    """The genuine first-boot case must NOT page -- otherwise every fresh
    checkout raises a P1."""
    _isolate_history.write_text("", encoding="utf-8")
    v = cycle_health.cycle_heartbeat_alarm()
    assert v["should_alarm_success"] is False, v
    assert v["success_stale"] is False, v


def test_halted_kill_switch_rows_are_not_completions(_isolate_history, frozen_now):
    """phase-85.4's new terminal status must count as a NON-completion --
    a paused book that halts cleanly every day is still a book that is not
    trading."""
    _write(
        _isolate_history,
        [
            _row("completed", frozen_now - timedelta(days=9), "old"),
            _row("halted_kill_switch", frozen_now - timedelta(hours=24), "halt"),
        ],
    )
    v = cycle_health.cycle_heartbeat_alarm()
    assert v["last_terminal_status"] == "halted_kill_switch"
    assert v["success_stale"] is True, v
    assert v["success_age_sec"] == pytest.approx(9 * 86400, abs=1)


def test_started_rows_are_skipped_by_both_clocks(_isolate_history, frozen_now):
    """Regression guard on the phase-38.2 behaviour the new leg reuses."""
    _write(
        _isolate_history,
        [
            _row("completed", frozen_now - timedelta(hours=20), "good"),
            {"cycle_id": "inflight", "started_at": frozen_now.isoformat(),
             "completed_at": None, "status": "started"},
        ],
    )
    v = cycle_health.cycle_heartbeat_alarm()
    assert v["last_terminal_status"] == "completed"
    assert v["success_stale"] is False, v


def test_weekend_suppresses_the_page_like_the_heartbeat_leg(
    _isolate_history, monkeypatch
):
    """Saturday: stale but not pageable, matching the existing heartbeat gate."""
    saturday = datetime(2026, 8, 8, 18, 0, 0, tzinfo=timezone.utc)  # a Saturday
    monkeypatch.setattr(cycle_health, "_now_utc", lambda: saturday)
    _write(_isolate_history, [_row("completed", saturday - timedelta(days=10), "old")])
    v = cycle_health.cycle_heartbeat_alarm()
    assert v["is_weekday_et"] is False
    assert v["success_stale"] is True
    assert v["should_alarm_success"] is False, v


# ─────────────────────────────────────────────────────────────────────────────
# The watchdog actually pages on it (criterion 4 says "the existing watchdog
# can page on" it -- a verdict field nobody reads is not a health signal)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def watchdog(monkeypatch):
    """Drive the REAL `scheduler.check_cycle_health_alarms`, with only the two
    Slack dispatchers replaced by recorders.

    Deliberately NOT a source scan. phase-36.12 already learned that a source
    scan cannot distinguish a live branch from a neutered one -- `if False and
    ...` keeps every symbol a grep looks for. The mutation matrix
    (scripts/qa/mutation_matrix_85_4.py M8) neuters exactly that branch, and
    this fixture is what catches it.
    """
    import backend.slack_bot.scheduler as sched

    fired_success: list[dict] = []
    fired_heartbeat: list[dict] = []
    monkeypatch.setattr(
        cycle_health, "fire_cycle_completed_stale_alarm", lambda v: fired_success.append(v)
    )
    monkeypatch.setattr(
        cycle_health, "fire_cycle_heartbeat_alarm", lambda v: fired_heartbeat.append(v)
    )
    monkeypatch.setattr(sched, "_cycle_completed_stale_last_was_stale", None, raising=False)
    monkeypatch.setattr(sched, "_cycle_heartbeat_last_was_stale", None, raising=False)
    return types.SimpleNamespace(
        run=sched.check_cycle_health_alarms,
        success=fired_success,
        heartbeat=fired_heartbeat,
    )


def test_watchdog_fires_the_completed_stale_p1_exactly_once_per_transition(
    _isolate_history, frozen_now, watchdog
):
    rows = [_row("completed", frozen_now - timedelta(days=7), "old")]
    rows += [_row("timeout", frozen_now - timedelta(days=d), f"to-{d}") for d in (2, 1)]
    _write(_isolate_history, rows)

    v = watchdog.run()
    assert v.get("should_alarm_success") is True, (
        "PRECONDITION: the ledger under test must be completed-stale"
    )
    watchdog.run()
    watchdog.run()
    assert len(watchdog.success) == 1, (
        f"expected one P1 per transition, got {len(watchdog.success)}"
    )
    # The heartbeat leg must stay QUIET on this ledger -- that contrast is the
    # whole defect: fresh timeouts, silent heartbeat, stale book.
    assert watchdog.heartbeat == [], "the heartbeat alarm should not have fired"

    # Recovery, then a re-break, must page again.
    _write(_isolate_history, rows + [_row("completed", frozen_now - timedelta(hours=3), "ok")])
    watchdog.run()
    assert len(watchdog.success) == 1
    _write(_isolate_history, rows)
    watchdog.run()
    assert len(watchdog.success) == 2, "a re-break after recovery must page again"


def test_watchdog_stays_silent_when_a_recent_cycle_completed(
    _isolate_history, frozen_now, watchdog
):
    """Mutation guard on the watchdog seam itself: it must be capable of NOT
    firing, or the transition test above proves only that it fires always."""
    _write(_isolate_history, [_row("completed", frozen_now - timedelta(hours=20), "ok")])
    watchdog.run()
    assert watchdog.success == []
    assert watchdog.heartbeat == []


def test_watchdog_p1_payload_names_the_last_completion_and_the_last_status(
    _isolate_history, frozen_now, monkeypatch
):
    """The operator must be able to act on the page without opening the JSONL:
    it has to carry WHEN the book last worked and WHAT it has been doing since."""
    import backend.slack_bot.scheduler as sched

    raised: list[dict] = []
    import backend.services.observability.alerting as alerting

    monkeypatch.setattr(
        alerting,
        "raise_cron_alert_sync",
        lambda **kw: raised.append(kw) or True,
    )
    monkeypatch.setattr(cycle_health, "fire_cycle_heartbeat_alarm", lambda v: None)
    monkeypatch.setattr(sched, "_cycle_completed_stale_last_was_stale", None, raising=False)
    monkeypatch.setattr(sched, "_cycle_heartbeat_last_was_stale", None, raising=False)

    _write(
        _isolate_history,
        [
            _row("completed", frozen_now - timedelta(days=7), "old"),
            _row("timeout", frozen_now - timedelta(hours=24), "to"),
        ],
    )
    sched.check_cycle_health_alarms()

    assert raised, "no alert reached the alerting layer"
    a = raised[-1]
    assert a["severity"] == "P1"
    assert a["error_type"] == "cycle_completed_stale_weekday"
    assert a["details"]["last_completed_cycle_at"].startswith("2026-07-29")
    assert a["details"]["last_terminal_status"] == "timeout"
    assert "7.0d" in a["details"]["age_since_last_completion"]
