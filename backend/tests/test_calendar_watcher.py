"""phase-6.6 tests for the FOMC + earnings calendar watcher.

Coverage:
 1. event_id determinism across re-computation.
 2. event_id stability when ticker casing differs.
 3. Finnhub hour -> window mapping (bmo / amc / dmh / missing).
 4. Blackout computation for Tue-Wed FOMC (e.g., Mar 18-19, 2025).
 5. Blackout edge case for Monday-only meeting.
 6. Registry add / get / clear + duplicate-name replacement.
 7. watcher.run_once dedups by event_id across multiple sources.
 8. watcher.run_once fail-opens when one source raises.
 9. normalize_event defaults propagate correctly.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Iterable

import pytest

from backend.calendar.blackout import compute_fomc_blackout
from backend.calendar.normalize import compute_event_id, normalize_window
from backend.calendar.registry import (
    clear_registry,
    get_sources,
    register,
)
from backend.calendar.watcher import (
    CalendarFetchReport,
    normalize_event,
    run_once,
)


# ---------- 1 + 2. event_id determinism ----------


def test_event_id_deterministic_across_recompute():
    e1 = compute_event_id(
        event_type="earnings",
        ticker="AAPL",
        fiscal_period_end="2025-12-31",
        scheduled_at="2026-01-30T13:30:00+00:00",
    )
    e2 = compute_event_id(
        event_type="earnings",
        ticker="AAPL",
        fiscal_period_end="2025-12-31",
        scheduled_at="2026-01-30T13:30:00+00:00",
    )
    assert e1 == e2


def test_event_id_stable_across_ticker_casing():
    e1 = compute_event_id(
        event_type="earnings",
        ticker="aapl",
        fiscal_period_end="2025-12-31",
        scheduled_at="2026-01-30",
    )
    e2 = compute_event_id(
        event_type="EARNINGS",
        ticker="AAPL",
        fiscal_period_end="2025-12-31",
        scheduled_at="2026-01-30",
    )
    assert e1 == e2


def test_event_id_fomc_uses_scheduled_date_when_no_fiscal_period():
    e1 = compute_event_id(
        event_type="fomc_meeting",
        ticker=None,
        fiscal_period_end=None,
        scheduled_at="2025-03-18T14:00:00+00:00",
    )
    e2 = compute_event_id(
        event_type="fomc_meeting",
        ticker=None,
        fiscal_period_end=None,
        scheduled_at="2025-03-18T20:00:00+00:00",  # different time, same date
    )
    assert e1 == e2


# ---------- 3. Window mapping ----------


def test_normalize_window_finnhub_hour_mapping():
    assert normalize_window("bmo") == "pre_open"
    assert normalize_window("amc") == "post_close"
    assert normalize_window("dmh") == "intraday"
    assert normalize_window("") == "all_day"
    assert normalize_window(None) == "all_day"
    assert normalize_window("unknown_code") == "all_day"
    # Passthrough of already-canonical values
    assert normalize_window("pre_open") == "pre_open"
    assert normalize_window("post_close") == "post_close"
    assert normalize_window("intraday") == "intraday"
    assert normalize_window("all_day") == "all_day"


# ---------- 4 + 5. Blackout window ----------


def test_blackout_tue_wed_fomc_march_2025():
    """Mar 18-19 2025 FOMC (Tue-Wed). Meeting day-1 = Mon Mar 17.

    First Saturday before Mon Mar 17 = Sat Mar 15. Second Saturday = Sat Mar 8.
    Blackout end = midnight on the day after Mar 19 = Mar 20 00:00.
    """
    start = datetime(2025, 3, 18, 19, 0, tzinfo=timezone.utc)
    end = datetime(2025, 3, 19, 19, 0, tzinfo=timezone.utc)
    bs, be = compute_fomc_blackout(start, end)
    assert bs.date() == date(2025, 3, 8)
    assert bs.weekday() == 5  # Saturday
    assert bs.hour == 0 and bs.minute == 0
    assert be.date() == date(2025, 3, 20)
    assert be.hour == 0 and be.minute == 0


def test_blackout_monday_only_meeting():
    """Hypothetical Mon meeting. Meeting day-1 = Sun.

    First Saturday before Sun is the previous day (Sat). Second Saturday is
    minus 7 more days = 8 days before meeting day-1 total.
    """
    start = datetime(2025, 6, 9, 19, 0, tzinfo=timezone.utc)  # Mon Jun 9
    end = start
    bs, be = compute_fomc_blackout(start, end)
    # meeting day-1 = Sun Jun 8. First Sat before = Jun 7. Second Sat = May 31.
    assert bs.date() == date(2025, 5, 31)
    assert bs.weekday() == 5
    assert be.date() == date(2025, 6, 10)


# ---------- 6. Registry ----------


class _FakeSource:
    def __init__(self, name: str, events: list[dict[str, Any]]) -> None:
        self.name = name
        self._events = events

    def fetch(self, from_date: date, to_date: date) -> Iterable[dict[str, Any]]:
        return iter(self._events)


def test_registry_add_get_clear_and_dedup_by_name():
    clear_registry()
    s1 = _FakeSource("dup_name", [{"a": 1}])
    s2 = _FakeSource("dup_name", [{"b": 2}])
    register(s1)
    register(s2)
    srcs = get_sources()
    assert len(srcs) == 1  # second register replaced the first
    assert srcs[0] is s2
    clear_registry()
    assert get_sources() == []


# ---------- 7. run_once dedup ----------


def test_run_once_dedups_by_event_id_across_sources(monkeypatch):
    clear_registry()
    fiscal = "2025-12-31"
    common = dict(
        event_type="earnings",
        ticker="AAPL",
        scheduled_at="2026-01-30T13:30:00+00:00",
        fiscal_period_end=fiscal,
    )
    src_a = _FakeSource(
        "src_a", [{**common, "source": "finnhub", "metadata": {"k": "a"}}]
    )
    src_b = _FakeSource(
        "src_b", [{**common, "source": "alphavantage", "metadata": {"k": "b"}}]
    )
    register(src_a)
    register(src_b)
    report = run_once(days_forward=365)
    assert report.n_events == 1
    ev = report.events[0]
    # metadata merged + merged_from recorded
    assert ev["metadata"].get("k") in ("a", "b")
    assert "merged_from" in ev["metadata"]
    clear_registry()


# ---------- 8. Fail-open on one source raising ----------


def test_run_once_fail_open_on_source_exception():
    clear_registry()

    class _BrokenSource:
        name = "broken"

        def fetch(self, from_date: date, to_date: date) -> Iterable[dict[str, Any]]:
            raise RuntimeError("boom")

    working = _FakeSource(
        "ok_src",
        [
            {
                "event_type": "earnings",
                "ticker": "MSFT",
                "scheduled_at": "2026-02-15T21:00:00+00:00",
                "fiscal_period_end": "2025-12-31",
                "source": "ok_src",
                "confidence": "confirmed",
            }
        ],
    )
    register(_BrokenSource())
    register(working)
    report = run_once(days_forward=365)
    assert report.n_events == 1
    assert any(e["source"] == "broken" for e in report.errors)
    assert report.events[0]["ticker"] == "MSFT"
    clear_registry()


# ---------- 9. normalize_event defaults ----------


def test_normalize_event_fills_defaults_and_computes_event_id():
    ev = normalize_event(
        {
            "event_type": "earnings",
            "ticker": "NVDA",
            "scheduled_at": "2026-02-20T21:00:00+00:00",
            "fiscal_period_end": "2026-01-31",
            "source": "finnhub",
            "hour": "amc",
            "eps_estimate": 0.88,
        }
    )
    assert ev["window"] == "post_close"  # amc mapped
    assert ev["event_id"] != ""
    assert ev["source"] == "finnhub"
    assert ev["confidence"] == "estimated"  # default when not provided
    assert ev["metadata"] == {}
    assert isinstance(ev["fetched_at"], str)
    assert ev["eps_estimate"] == 0.88
