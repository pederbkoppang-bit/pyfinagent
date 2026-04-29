"""phase-23.1.18: paper_portfolio_snapshots MERGE upsert + red-line MAX query."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _bq_instance() -> "MagicMock":  # type: ignore[name-defined]
    from backend.db.bigquery_client import BigQueryClient
    settings = SimpleNamespace(gcp_project_id="test-proj", bq_dataset_reports="test_ds")
    bq = BigQueryClient.__new__(BigQueryClient)
    bq.settings = settings
    bq.client = MagicMock()
    bq.client.query.return_value.result.return_value = None
    return bq


def test_save_paper_snapshot_uses_merge():
    """Smoke test: save_paper_snapshot issues a MERGE statement keyed on
    snapshot_date, not plain INSERT."""
    bq = _bq_instance()
    bq.save_paper_snapshot({
        "snapshot_date": "2026-04-29",
        "total_nav": 15647.74,
        "cash": 2146.39,
        "positions_value": 13501.35,
    })
    sent_sql = bq.client.query.call_args[0][0]
    assert "MERGE" in sent_sql, f"expected MERGE; got: {sent_sql[:200]}"
    assert "ON T.snapshot_date = S.snapshot_date" in sent_sql
    assert "WHEN MATCHED" in sent_sql and "WHEN NOT MATCHED" in sent_sql


def test_save_paper_snapshot_rejects_missing_snapshot_date():
    """MERGE requires the snapshot_date key for the merge predicate."""
    bq = _bq_instance()
    with pytest.raises(ValueError, match="requires 'snapshot_date' field"):
        bq.save_paper_snapshot({"total_nav": 1.0, "cash": 1.0})


def test_red_line_query_uses_max_total_nav():
    """phase-23.1.18: defense-in-depth — _fetch_snapshots uses MAX(total_nav)
    instead of ANY_VALUE so legacy duplicate rows pick the post-repair value."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[2] / "backend/api/sovereign_api.py"
    text = src.read_text(encoding="utf-8")
    assert "MAX(total_nav) AS nav" in text, \
        "_fetch_snapshots must aggregate with MAX(total_nav), not ANY_VALUE"
    assert "ANY_VALUE(total_nav)" not in text, \
        "remove the ANY_VALUE legacy aggregation"
