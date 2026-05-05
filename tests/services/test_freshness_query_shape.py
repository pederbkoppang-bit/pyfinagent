"""phase-23.2.20: regression guard against the freshness query type-coercion bug.

Pre-fix: cycle_health._bq_max_event_age built
  SELECT TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(col), SECOND) AS age FROM table
which BigQuery rejected with `Unable to coerce type STRING to expected type
TIMESTAMP` because paper_trades.created_at and paper_portfolio_snapshots.snapshot_date
are STRING. The function swallowed the exception at logger.debug level (silent at
default INFO) and returned None, so the OpsStatusBar showed two perpetually-gray
'unknown' circles for paper_trades / paper_snapshots.

These tests assert:
- The SQL string includes SAFE.TIMESTAMP(MAX(col)) so STRING columns parse.
- A successful query returns the parsed float age.
- A failed query logs at WARNING (not DEBUG) so future regressions surface.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.services.cycle_health import _bq_max_event_age


def _make_fake_bq(captured_sql: list[str], rows_or_exc):
    """Fake BQ client. `rows_or_exc` is either a list of row dicts or an Exception."""

    class _FakeQueryJob:
        def __init__(self, rows):
            self._rows = rows

        def result(self):
            return iter(self._rows)

    class _FakeClient:
        def query(self, sql: str):
            captured_sql.append(sql)
            if isinstance(rows_or_exc, Exception):
                raise rows_or_exc
            return _FakeQueryJob(rows_or_exc)

    bq = SimpleNamespace(
        client=_FakeClient(),
        _pt_table=lambda name: f"proj.dataset.{name}",
    )
    return bq


def test_sql_uses_safe_timestamp_wrapper():
    """The SQL must wrap MAX(col) in SAFE.TIMESTAMP() to coerce STRING columns."""
    captured: list[str] = []
    bq = _make_fake_bq(captured, [{"age": 123.0}])
    age = _bq_max_event_age(bq, "paper_trades", "created_at")
    assert age == 123.0
    assert len(captured) == 1
    assert "SAFE.TIMESTAMP(MAX(created_at))" in captured[0], \
        f"SAFE.TIMESTAMP wrapper missing from SQL: {captured[0]}"
    assert "TIMESTAMP_DIFF" in captured[0]


def test_returns_age_on_successful_query():
    """When BQ returns a row with a numeric age, function returns it as float."""
    captured: list[str] = []
    bq = _make_fake_bq(captured, [{"age": 349672}])
    age = _bq_max_event_age(bq, "paper_trades", "created_at")
    assert age == 349672.0


def test_returns_none_on_empty_result():
    captured: list[str] = []
    bq = _make_fake_bq(captured, [])
    age = _bq_max_event_age(bq, "paper_trades", "created_at")
    assert age is None


def test_returns_none_when_age_is_null():
    """SAFE.TIMESTAMP returns NULL for unparseable input -> TIMESTAMP_DIFF returns NULL -> None."""
    captured: list[str] = []
    bq = _make_fake_bq(captured, [{"age": None}])
    age = _bq_max_event_age(bq, "paper_trades", "created_at")
    assert age is None


def test_failed_query_logs_at_warning_not_debug(caplog):
    """The pre-fix code logged at DEBUG (silent at INFO). Post-fix must log at WARNING."""
    captured: list[str] = []
    bq = _make_fake_bq(captured, RuntimeError("simulated BQ rejection"))
    with caplog.at_level(logging.WARNING, logger="backend.services.cycle_health"):
        age = _bq_max_event_age(bq, "paper_trades", "created_at")
    assert age is None
    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warning_records) == 1, \
        f"expected exactly 1 WARNING record, got {len(warning_records)}"
    msg = warning_records[0].getMessage()
    assert "paper_trades" in msg
    assert "created_at" in msg
    assert "simulated BQ rejection" in msg
