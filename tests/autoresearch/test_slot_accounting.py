"""phase-10.8 pytest companion to phase10_slot_accounting_test.py."""
from __future__ import annotations

import json as _json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.autoresearch.slot_accounting import (
    log_slot_usage,
    verify_weekly_invariant,
)


T0 = datetime(2026, 4, 24, 21, 0, tzinfo=timezone.utc)


def _capture_factory():
    store: list = []

    def capture(table, rows):
        store.append((table, rows))
        return True

    return store, capture


def test_default_table_is_pyfinagent_data_harness_learning_log():
    store, capture = _capture_factory()
    r = log_slot_usage(
        week_iso="2026-W17",
        slot_id="thu_batch",
        routine="trigger_thursday_batch",
        result={"ok": True},
        bq_insert_fn=capture,
    )
    assert r["table"] == "pyfinagent_data.harness_learning_log"
    assert store[0][0] == "pyfinagent_data.harness_learning_log"


def test_phase_label_applied():
    store, capture = _capture_factory()
    log_slot_usage(
        week_iso="2026-W17",
        slot_id="thu_batch",
        routine="r",
        result={"ok": True},
        bq_insert_fn=capture,
    )
    row = store[0][1][0]
    assert row["phase"] == "phase-10"


def test_invalid_slot_id_raises():
    _, capture = _capture_factory()
    with pytest.raises(ValueError, match="slot_id"):
        log_slot_usage(
            week_iso="2026-W17",
            slot_id="bogus_slot",
            routine="r",
            result={"ok": True},
            bq_insert_fn=capture,
        )


def test_result_json_serialized():
    store, capture = _capture_factory()
    log_slot_usage(
        week_iso="2026-W17",
        slot_id="fri_promotion",
        routine="run_friday_promotion",
        result={"promoted_ids": ["g1", "g2"], "allocations": [0.05, 0.05]},
        bq_insert_fn=capture,
    )
    row = store[0][1][0]
    parsed = _json.loads(row["result_json"])
    assert parsed["promoted_ids"] == ["g1", "g2"]


def test_bq_insert_failure_returns_inserted_false():
    def boom(table, rows):
        raise RuntimeError("BQ down")
    r = log_slot_usage(
        week_iso="2026-W17",
        slot_id="thu_batch",
        routine="r",
        result={"ok": True},
        bq_insert_fn=boom,
    )
    assert r["inserted"] is False
    assert r["row"]["slot_id"] == "thu_batch"


def test_invariant_sql_pins_slot_id_filter():
    """qa_108_v1 cycle-2 fix: pin the SQL text so widening the IN clause
    (e.g., accidentally including monthly_gate) is caught by the test suite.

    Prior tests stubbed bq_query_fn with a Python closure that duplicated the
    filter — SQL drift in the production module was invisible. This test
    inspects the SQL string itself.
    """
    captured: dict = {}

    def capture_sql(sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return 2

    r = verify_weekly_invariant("2026-W17", bq_query_fn=capture_sql)
    assert "IN ('thu_batch', 'fri_promotion')" in captured["sql"]
    assert "monthly_gate" not in captured["sql"]
    assert "rollback" not in captured["sql"]
    assert "phase = 'phase-10'" in captured["sql"]
    assert captured["params"] == {"week_iso": "2026-W17"}
    assert r["sum"] == 2
    assert r["satisfied"] is True


def test_weekly_invariant_sum_equals_2():
    store = []

    def capture(table, rows):
        store.extend(rows)
        return True

    def query_count(sql, params):
        wk = params["week_iso"]
        return sum(
            1 for r in store
            if r["week_iso"] == wk
            and r["phase"] == "phase-10"
            and r["slot_id"] in ("thu_batch", "fri_promotion")
        )

    log_slot_usage(week_iso="2026-W17", slot_id="thu_batch", routine="r", result={"ok": True}, bq_insert_fn=capture)
    log_slot_usage(week_iso="2026-W17", slot_id="fri_promotion", routine="r", result={"ok": True}, bq_insert_fn=capture)
    log_slot_usage(week_iso="2026-W17", slot_id="monthly_gate", routine="r", result={"ok": True}, bq_insert_fn=capture)
    log_slot_usage(week_iso="2026-W17", slot_id="rollback", routine="r", result={"ok": True}, bq_insert_fn=capture)

    r = verify_weekly_invariant("2026-W17", bq_query_fn=query_count)
    assert r["sum"] == 2
    assert r["satisfied"] is True


def test_weekly_invariant_unsatisfied_when_missing_slot():
    store = []

    def capture(table, rows):
        store.extend(rows)
        return True

    def query_count(sql, params):
        wk = params["week_iso"]
        return sum(
            1 for r in store
            if r["week_iso"] == wk
            and r["phase"] == "phase-10"
            and r["slot_id"] in ("thu_batch", "fri_promotion")
        )

    # Only log Thursday; Friday missing -> sum==1, satisfied=False
    log_slot_usage(week_iso="2026-W18", slot_id="thu_batch", routine="r", result={"ok": True}, bq_insert_fn=capture)
    r = verify_weekly_invariant("2026-W18", bq_query_fn=query_count)
    assert r["sum"] == 1
    assert r["satisfied"] is False


def test_row_id_is_unique_uuid():
    store, capture = _capture_factory()
    for _ in range(5):
        log_slot_usage(
            week_iso="2026-W17", slot_id="thu_batch", routine="r",
            result={"ok": True}, bq_insert_fn=capture,
        )
    row_ids = [rows[0]["row_id"] for _, rows in store]
    assert len(set(row_ids)) == 5


def test_error_msg_populated_when_result_has_error():
    store, capture = _capture_factory()
    log_slot_usage(
        week_iso="2026-W17",
        slot_id="rollback",
        routine="auto_demote_on_dd_breach",
        result={"demoted": False, "error": "no_thursday_batch_on_ledger"},
        bq_insert_fn=capture,
    )
    row = store[0][1][0]
    assert row["error_msg"] == "no_thursday_batch_on_ledger"
