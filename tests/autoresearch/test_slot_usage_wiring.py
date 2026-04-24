"""phase-10.8.1 -- log_slot_usage wiring across the 4 autoresearch routines.

Covers the masterplan verification criteria:
  - trigger_thursday_batch() logs slot_id='thu_batch' (post-ledger-write)
  - run_friday_promotion() logs slot_id='fri_promotion' (success path)
  - run_monthly_sortino_gate() + auto_demote_on_dd_breach() each log with
    their slot_id and result dict
  - After stub runs, captured log_slot_usage calls include all 4 slot_ids
    with week_iso set correctly.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.autoresearch.thursday_batch import trigger_thursday_batch
from backend.autoresearch.friday_promotion import run_friday_promotion
from backend.autoresearch.monthly_champion_challenger import run_monthly_sortino_gate
from backend.autoresearch.rollback import auto_demote_on_dd_breach


def _capture_factory():
    """Return (store, capture_fn). capture_fn mirrors log_slot_usage kwargs."""
    store: list[dict] = []

    def capture(**kwargs):
        store.append(kwargs)
        return {"inserted": True, "row_id": "test", "table": "test", "row": {}}

    return store, capture


def _fake_gate_pass_candidates():
    """Candidates that the phase-8.5.5 PromotionGate will pass."""
    return [
        {
            "trial_id": f"trial_{i}",
            "dsr": 0.8,
            "pbo": 0.05,
            "sortino_12m": 2.0,
            "max_dd": 0.08,
            "trades": 500,
            "months_of_oos": 18,
            "rolling_sharpe_12m_slope": 0.01,
            "trials_in_family": 5,
        }
        for i in range(3)
    ]


def test_thursday_batch_logs_on_fired(tmp_path):
    lpath = tmp_path / "ledger.tsv"
    store, capture = _capture_factory()
    r = trigger_thursday_batch("2026-W17", ledger_path=lpath, log_fn=capture)
    assert r["already_fired"] is False
    assert len(store) == 1
    call = store[0]
    assert call["slot_id"] == "thu_batch"
    assert call["phase"] == "phase-10"
    assert call["week_iso"] == "2026-W17"
    assert call["routine"] == "trigger_thursday_batch"
    assert call["result"]["batch_id"] == r["batch_id"]
    assert call["result"]["candidates_kicked"] == r["candidates_kicked"]
    assert call["result"]["status"] == "fired"


def test_thursday_batch_logs_on_already_fired(tmp_path):
    lpath = tmp_path / "ledger.tsv"
    store, capture = _capture_factory()
    trigger_thursday_batch("2026-W18", ledger_path=lpath, log_fn=capture)
    trigger_thursday_batch("2026-W18", ledger_path=lpath, log_fn=capture)
    # two log calls: one 'fired', one 'already_fired'
    assert len(store) == 2
    assert store[0]["result"]["status"] == "fired"
    assert store[1]["result"]["status"] == "already_fired"
    assert store[1]["week_iso"] == "2026-W18"


def test_friday_promotion_logs_on_success(tmp_path):
    lpath = tmp_path / "ledger.tsv"
    store_thu, cap_thu = _capture_factory()
    store_fri, cap_fri = _capture_factory()
    trigger_thursday_batch("2026-W17", ledger_path=lpath, log_fn=cap_thu)
    r = run_friday_promotion(
        "2026-W17",
        candidates=_fake_gate_pass_candidates(),
        ledger_path=lpath,
        log_fn=cap_fri,
    )
    assert r["error"] is None
    assert len(store_fri) == 1
    call = store_fri[0]
    assert call["slot_id"] == "fri_promotion"
    assert call["phase"] == "phase-10"
    assert call["week_iso"] == "2026-W17"
    assert call["routine"] == "run_friday_promotion"
    assert call["result"]["status"] == "promoted"
    assert "promoted_ids" in call["result"]


def test_friday_promotion_logs_on_no_thursday_batch(tmp_path):
    lpath = tmp_path / "ledger.tsv"
    store, capture = _capture_factory()
    r = run_friday_promotion(
        "2026-W30",
        candidates=[],
        ledger_path=lpath,
        log_fn=capture,
    )
    assert r["error"] == "no_thursday_batch_on_ledger"
    assert len(store) == 1
    assert store[0]["slot_id"] == "fri_promotion"
    assert store[0]["week_iso"] == "2026-W30"
    assert store[0]["result"]["status"] == "no_thursday_batch_on_ledger"


def test_monthly_sortino_gate_logs_on_fail_closed_branches(tmp_path):
    """eval_date is NOT last trading Friday -> no log emitted."""
    store, capture = _capture_factory()
    spath = tmp_path / "state.json"
    # 2026-04-21 is a Tuesday, not last-Friday.
    r = run_monthly_sortino_gate(
        eval_date=date(2026, 4, 21),
        champion_returns=[0.01] * 30,
        challenger_returns=[0.01] * 30,
        champion_max_dd=-0.05,
        challenger_max_dd=-0.05,
        challenger_pbo=0.1,
        state_path=spath,
        log_fn=capture,
    )
    assert r["fired"] is False
    assert r["reason"] == "not_last_trading_friday"
    # Non-fired day -> no log (slot was never consumed).
    assert store == []


def test_monthly_sortino_gate_logs_on_fired_path(tmp_path):
    """eval_date is last Friday of Apr 2026 (2026-04-24) -> log emitted."""
    store, capture = _capture_factory()
    spath = tmp_path / "state.json"
    r = run_monthly_sortino_gate(
        eval_date=date(2026, 4, 24),
        champion_returns=[0.001] * 30,
        challenger_returns=[0.001] * 30,
        champion_max_dd=-0.05,
        challenger_max_dd=-0.05,
        challenger_pbo=0.1,
        state_path=spath,
        log_fn=capture,
    )
    # Whether it passed or failed quality gates, the routine was called on the
    # right day so telemetry must fire.
    assert r["fired"] is True
    assert len(store) == 1
    call = store[0]
    assert call["slot_id"] == "monthly_gate"
    assert call["phase"] == "phase-10"
    assert call["routine"] == "run_monthly_sortino_gate"
    # week_iso derived from eval_date.isocalendar(). 2026-04-24 -> 2026-W17.
    year, week, _ = date(2026, 4, 24).isocalendar()
    assert call["week_iso"] == f"{year:04d}-W{week:02d}"


def test_rollback_logs_on_no_breach(tmp_path):
    store, capture = _capture_factory()
    spath = tmp_path / "state.json"
    apath = tmp_path / "audit.jsonl"
    r = auto_demote_on_dd_breach(
        challenger_id="c1",
        challenger_current_dd=-0.05,  # within threshold
        state_path=spath,
        audit_path=apath,
        log_fn=capture,
    )
    assert r["decision"] == "no_breach"
    assert len(store) == 1
    call = store[0]
    assert call["slot_id"] == "rollback"
    assert call["phase"] == "phase-10"
    assert call["routine"] == "auto_demote_on_dd_breach"
    assert call["week_iso"] == "unknown"
    assert call["result"]["decision"] == "no_breach"
    assert call["result"]["challenger_id"] == "c1"


def test_rollback_logs_on_auto_demoted_with_week_iso(tmp_path):
    store, capture = _capture_factory()
    spath = tmp_path / "state.json"
    apath = tmp_path / "audit.jsonl"
    lpath = tmp_path / "ledger.tsv"
    r = auto_demote_on_dd_breach(
        challenger_id="c2",
        challenger_current_dd=-0.50,  # breach
        state_path=spath,
        audit_path=apath,
        ledger_path=lpath,
        week_iso="2026-W17",
        log_fn=capture,
    )
    assert r["decision"] == "auto_demoted"
    assert len(store) == 1
    assert store[0]["slot_id"] == "rollback"
    assert store[0]["week_iso"] == "2026-W17"
    assert store[0]["result"]["decision"] == "auto_demoted"


def test_all_four_slot_ids_captured_in_one_run(tmp_path):
    """Single-test combined assertion: after stubbing all four routines, the
    captured log_slot_usage calls include thu_batch + fri_promotion +
    monthly_gate + rollback, each with week_iso set correctly."""
    store, capture = _capture_factory()
    lpath = tmp_path / "ledger.tsv"
    spath = tmp_path / "state.json"
    apath = tmp_path / "audit.jsonl"

    trigger_thursday_batch("2026-W17", ledger_path=lpath, log_fn=capture)
    run_friday_promotion(
        "2026-W17",
        candidates=_fake_gate_pass_candidates(),
        ledger_path=lpath,
        log_fn=capture,
    )
    run_monthly_sortino_gate(
        eval_date=date(2026, 4, 24),
        champion_returns=[0.001] * 30,
        challenger_returns=[0.001] * 30,
        champion_max_dd=-0.05,
        challenger_max_dd=-0.05,
        challenger_pbo=0.1,
        state_path=spath,
        log_fn=capture,
    )
    auto_demote_on_dd_breach(
        challenger_id="c",
        challenger_current_dd=-0.50,
        state_path=spath,
        audit_path=apath,
        ledger_path=lpath,
        week_iso="2026-W17",
        log_fn=capture,
    )

    slot_ids = {c["slot_id"] for c in store}
    assert slot_ids == {"thu_batch", "fri_promotion", "monthly_gate", "rollback"}
    # Every call has a week_iso string set, not None/empty.
    for c in store:
        assert isinstance(c["week_iso"], str) and c["week_iso"]
    # Every call carries phase='phase-10'.
    for c in store:
        assert c["phase"] == "phase-10"
