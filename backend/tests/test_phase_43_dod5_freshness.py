"""phase-43.0 cycle-19 -- DoD-5 freshness type-branch tests.

Covers the cycle-14 fix at backend/services/cycle_health.py:420-462:
the `_STRING_DATE_TIMESTAMP_COLS` set-membership branch inside
`_bq_max_event_age` that selects between `SAFE.TIMESTAMP(MAX(col))`
(STRING/DATE columns) and bare `MAX(col)` (native TIMESTAMP columns).

Pattern mirrors test_dod4_tier1_coverage_investment.py:417-424
(existing canonical chain mock) + test_phase_43_dod2_window.py
(in-test MagicMock construction, no shared conftest).

Bug context: applying SAFE.TIMESTAMP to a native-TIMESTAMP MAX result
raises BQ 400 `SAFE with function timestamp is not supported`; the
broad `except` swallowed it; 4 sources stayed band=unknown (DoD-5
cycle-12 FAIL). The cycle-14 fix branched on column type.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.services import cycle_health


def _fresh_bq(age_seconds: float = 12345.0) -> MagicMock:
    """Build a MagicMock BQ client that returns a single row with `age`."""
    bq = MagicMock()
    row = MagicMock()
    row.get.return_value = age_seconds
    bq.client.query.return_value.result.return_value = [row]
    return bq


# ---- STRING/DATE columns: must use SAFE.TIMESTAMP(MAX(col)) ----

@pytest.mark.parametrize(
    "table,col",
    [
        ("paper_trades", "created_at"),
        ("paper_portfolio_snapshots", "snapshot_date"),
    ],
)
def test_bq_max_event_age_string_columns_use_safe_timestamp(table, col):
    """STRING/DATE columns must be coerced via SAFE.TIMESTAMP(MAX(col))."""
    bq = _fresh_bq()
    age = cycle_health._bq_max_event_age(bq, table, col)
    assert age == 12345.0
    sql = bq.client.query.call_args[0][0]
    assert f"SAFE.TIMESTAMP(MAX({col}))" in sql, (
        f"expected SAFE.TIMESTAMP wrapper for {table}.{col}, got: {sql}"
    )


# ---- Native TIMESTAMP columns: must use bare MAX(col), no SAFE wrapper ----

@pytest.mark.parametrize(
    "table,col",
    [
        ("historical_prices", "ingested_at"),
        ("signals_log", "recorded_at"),
    ],
)
def test_bq_max_event_age_timestamp_columns_use_bare_max(table, col):
    """Native TIMESTAMP columns must use bare MAX(col); the SAFE.TIMESTAMP
    wrapper triggers BQ 400 on TIMESTAMP-typed inputs (cycle-12 root cause)."""
    bq = _fresh_bq()
    age = cycle_health._bq_max_event_age(bq, table, col)
    assert age == 12345.0
    sql = bq.client.query.call_args[0][0]
    assert f"MAX({col})" in sql
    assert "SAFE.TIMESTAMP" not in sql, (
        f"unexpected SAFE.TIMESTAMP wrapper for native TIMESTAMP "
        f"{table}.{col}, got: {sql}"
    )
