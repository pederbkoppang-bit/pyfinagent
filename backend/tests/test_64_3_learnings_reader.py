"""phase-64.3: learnings-reader error-vs-empty gap test (pure; no live BQ).

get_paper_trades_in_window is the clean reader seam: it RAISES on a genuine
query error (so the caller can surface it) and returns [] on an empty result
-- error != empty. (The downstream aggregator _compute_learnings swallows
errors to []; that is a known gap, out of scope for this test-only step.)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.db.bigquery_client import BigQueryClient
from backend.services.paper_round_trips import pair_round_trips


def _reader_with_mock_client():
    """A BigQueryClient with a mocked .client (skips __init__/ADC)."""
    bq = BigQueryClient.__new__(BigQueryClient)
    bq.client = MagicMock()
    bq._pt_table = lambda t: "proj.dataset.table"
    return bq


def test_64_3_learnings_reader_error_surfaces():
    bq = _reader_with_mock_client()
    bq.client.query.side_effect = RuntimeError("BQ 400: bad query")
    with pytest.raises(RuntimeError):
        bq.get_paper_trades_in_window(window_days=7)


def test_64_3_learnings_reader_empty_returns_empty_list():
    bq = _reader_with_mock_client()
    bq.client.query.return_value.result.return_value = []
    out = bq.get_paper_trades_in_window(window_days=7)
    assert out == []


def test_64_3_learnings_reader_error_is_not_empty():
    """The whole point: a genuine error must NOT be indistinguishable from an
    empty window (no silent swallow at the reader layer)."""
    bq = _reader_with_mock_client()
    # empty -> []
    bq.client.query.return_value.result.return_value = []
    assert bq.get_paper_trades_in_window(window_days=7) == []
    # error -> raises (distinct outcome)
    bq.client.query.side_effect = RuntimeError("transient")
    with pytest.raises(RuntimeError):
        bq.get_paper_trades_in_window(window_days=7)


def test_64_3_learnings_reader_pair_round_trips_empty():
    assert pair_round_trips([]) == []
