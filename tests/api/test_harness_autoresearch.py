"""phase-10.11 pytest companion to phase10_integration_test.py."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.api.harness_autoresearch import (
    HarnessSprintWeekState,
    _build_sql,
    _current_week_iso,
    fetch_sprint_state,
)


def _rows():
    return [
        {
            "slot_id": "thu_batch",
            "result_json": '{"batch_id": "b9686bc5", "candidates_kicked": 128}',
            "logged_at": "2026-04-23T21:00:00Z",
        },
        {
            "slot_id": "fri_promotion",
            "result_json": '{"promoted_ids": ["g1"], "rejected_ids": []}',
            "logged_at": "2026-04-24T21:00:00Z",
        },
        {
            "slot_id": "monthly_gate",
            "result_json": '{"sortino_delta": 0.35, "approval_pending": true, "approved": false}',
            "logged_at": "2026-04-24T21:05:00Z",
        },
    ]


def test_fetch_projects_all_three_slots_into_state():
    s = fetch_sprint_state(week_iso="2026-W17", bq_query_fn=lambda sql, p: _rows())
    assert isinstance(s, HarnessSprintWeekState)
    assert s.weekIso == "2026-W17"
    assert s.thu is not None and s.thu.candidatesKicked == 128
    assert s.fri is not None and s.fri.promotedIds == ["g1"]
    assert s.monthly is not None and s.monthly.approvalPending is True
    assert s.monthly.approved is False


def test_empty_bq_result_returns_none():
    s = fetch_sprint_state(week_iso="2026-W17", bq_query_fn=lambda sql, p: [])
    assert s is None


def test_partial_data_returns_partial_state():
    only_thursday = [
        {
            "slot_id": "thu_batch",
            "result_json": '{"batch_id": "b1", "candidates_kicked": 100}',
            "logged_at": "2026-04-23T21:00:00Z",
        }
    ]
    s = fetch_sprint_state(week_iso="2026-W17", bq_query_fn=lambda sql, p: only_thursday)
    assert s is not None
    assert s.thu is not None
    assert s.fri is None
    assert s.monthly is None


def test_sql_references_harness_learning_log_and_binds_week_iso():
    captured = {}

    def capture(sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return []

    fetch_sprint_state(week_iso="2026-W25", bq_query_fn=capture)
    assert "harness_learning_log" in captured["sql"]
    assert "@week_iso" in captured["sql"]
    assert captured["params"] == {"week_iso": "2026-W25"}


def test_default_week_iso_uses_isocalendar():
    wk = _current_week_iso()
    # format: YYYY-Www with zero-padded 2-digit week
    import re
    assert re.match(r"^\d{4}-W\d{2}$", wk)


def test_latest_row_wins_per_slot_id():
    # Two rows for thu_batch — SQL orders DESC so the FIRST in the input list wins.
    rows = [
        {
            "slot_id": "thu_batch",
            "result_json": '{"batch_id": "new", "candidates_kicked": 200}',
            "logged_at": "2026-04-24T10:00:00Z",
        },
        {
            "slot_id": "thu_batch",
            "result_json": '{"batch_id": "old", "candidates_kicked": 100}',
            "logged_at": "2026-04-23T10:00:00Z",
        },
    ]
    s = fetch_sprint_state(week_iso="2026-W17", bq_query_fn=lambda sql, p: rows)
    assert s.thu.batchId == "new"
    assert s.thu.candidatesKicked == 200


def test_router_prefix_is_api_harness():
    """qa_1011_v1 M1 gap-closer: pin the router prefix so /api/other mutation fails."""
    from backend.api.harness_autoresearch import router
    assert router.prefix == "/api/harness"


def test_malformed_result_json_fails_open():
    rows = [
        {
            "slot_id": "thu_batch",
            "result_json": "{not valid json",
            "logged_at": "2026-04-24T10:00:00Z",
        }
    ]
    # Does not raise; just doesn't populate thu.
    s = fetch_sprint_state(week_iso="2026-W17", bq_query_fn=lambda sql, p: rows)
    # rows_by_slot is non-empty (slot id captured), so we get a state object,
    # but thu.batchId is "" so the projection skips it.
    assert s is not None
    assert s.thu is None
