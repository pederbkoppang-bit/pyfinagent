"""phase-16.30 regression coverage for outcome_tracker.

#35 (16.26 follow-up) -- the BQ Python SDK returns TIMESTAMP columns as
native `datetime.datetime` objects in row dicts. `OutcomeTracker.evaluate_all_pending`
previously called `datetime.fromisoformat(report["analysis_date"])` which
raises `TypeError: fromisoformat: argument must be str` on datetime input.
The fix at outcome_tracker.py:94 adds an isinstance guard.

These tests cover both shapes (datetime + iso str) plus the tz-aware path
that BQ also produces (TIMESTAMP fields can come back tz-aware UTC).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.outcome_tracker import OutcomeTracker  # noqa: E402


class _FakeBQ:
    """Minimal stub for the BigQueryClient surface OutcomeTracker uses."""

    def __init__(self, recent_reports: list[dict]):
        self._recent = recent_reports
        self.fetched_for: list[tuple[str, object]] = []

    def get_recent_reports(self, limit: int = 100):
        return self._recent[:limit]

    def get_report(self, ticker: str, analysis_date):
        # Return None so evaluate_all_pending's "if not stored" branch
        # short-circuits cleanly. The point of these tests is the date
        # parsing, not the per-report eval.
        self.fetched_for.append((ticker, analysis_date))
        return None

    def insert_outcome(self, *args, **kwargs):
        return None

    def get_performance_stats(self):
        return {}


def _settings_stub():
    """Minimal settings stub so OutcomeTracker init doesn't drag in
    the full Settings() (which requires GCP env vars + model keys).
    """
    s = MagicMock()
    s.benchmark_ticker = "SPY"
    return s


def _tracker_with_reports(reports: list[dict]) -> OutcomeTracker:
    settings = _settings_stub()
    bq = _FakeBQ(reports)
    # Bypass __init__ if needed -- the real __init__ may construct a
    # BQ client. We patch the bq attribute directly post-init via a
    # simple subclass dance.
    tracker = OutcomeTracker.__new__(OutcomeTracker)
    tracker.settings = settings
    tracker.bq = bq
    return tracker


def test_evaluate_all_pending_handles_native_datetime():
    """phase-16.30 regression: BQ TIMESTAMP -> native datetime must NOT raise."""
    reports = [{
        "ticker": "AAPL",
        "analysis_date": datetime(2026, 1, 1, 12, 0, 0),  # naive datetime
        "recommendation": "BUY",
        "score": 0.7,
    }]
    tracker = _tracker_with_reports(reports)

    # Must not raise TypeError("fromisoformat: argument must be str")
    out = tracker.evaluate_all_pending()
    assert isinstance(out, list)
    # Empty because get_report() returns None in our stub; the point is
    # we got past the date parsing without TypeError.
    assert out == []


def test_evaluate_all_pending_handles_tz_aware_datetime():
    """BQ TIMESTAMP can be tz-aware UTC; subtraction with utcnow() must not raise."""
    reports = [{
        "ticker": "MSFT",
        "analysis_date": datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "recommendation": "HOLD",
        "score": 0.5,
    }]
    tracker = _tracker_with_reports(reports)

    out = tracker.evaluate_all_pending()
    assert isinstance(out, list)
    assert out == []


def test_evaluate_all_pending_still_handles_iso_string():
    """ISO-string analysis_date (legacy / hand-written) still works after the fix."""
    reports = [{
        "ticker": "NVDA",
        "analysis_date": "2026-01-01T12:00:00",
        "recommendation": "BUY",
        "score": 0.8,
    }]
    tracker = _tracker_with_reports(reports)

    out = tracker.evaluate_all_pending()
    assert isinstance(out, list)
    assert out == []


def test_evaluate_all_pending_skips_too_recent_reports():
    """analysis_date < 7 days ago -> skipped (no error path probe)."""
    very_recent = datetime.now(timezone.utc).replace(tzinfo=None)
    reports = [{
        "ticker": "GOOGL",
        "analysis_date": very_recent,
        "recommendation": "BUY",
        "score": 0.6,
    }]
    tracker = _tracker_with_reports(reports)

    out = tracker.evaluate_all_pending()
    assert out == []
    # And the bq.get_report shouldn't have been queried (skip-recent path)
    assert tracker.bq.fetched_for == []


def test_module_level_evaluate_recent_returns_dict_or_list():
    """phase-16.21 wrapper still returns either list or graceful-empty dict."""
    # We import here so the wrapper is exercised, not the class.
    from backend.services.outcome_tracker import evaluate_recent
    out = evaluate_recent(limit=5)
    # If BQ is reachable -> list; if not -> graceful dict per phase-16.26
    assert isinstance(out, (list, dict))
    if isinstance(out, dict):
        # Graceful-empty shape from the 16.26 wrapper
        assert "status" in out and "outcomes" in out
