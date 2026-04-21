"""phase-10.7 pytest companion to phase10_rollback_test.py."""
from __future__ import annotations

import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.autoresearch import weekly_ledger
from backend.autoresearch.rollback import auto_demote_on_dd_breach

T0 = datetime(2026, 4, 24, 21, 0, tzinfo=timezone.utc)


def test_challenger_dd_breach_auto_demotes(tmp_path):
    r = auto_demote_on_dd_breach(
        challenger_id="s1",
        challenger_current_dd=-0.11,
        state_path=tmp_path / "state.json",
        audit_path=tmp_path / "audit.jsonl",
        now=T0,
    )
    assert r["demoted"] is True
    assert r["decision"] == "auto_demoted"


def test_sub_threshold_no_demote(tmp_path):
    r = auto_demote_on_dd_breach(
        challenger_id="s2",
        challenger_current_dd=-0.05,
        state_path=tmp_path / "state.json",
        audit_path=tmp_path / "audit.jsonl",
        now=T0,
    )
    assert r["demoted"] is False
    assert r["decision"] == "no_breach"


def test_demotion_logged_with_auto_demoted_decision(tmp_path):
    apath = tmp_path / "audit.jsonl"
    auto_demote_on_dd_breach(
        challenger_id="s3",
        challenger_current_dd=-0.15,
        state_path=tmp_path / "state.json",
        audit_path=apath,
        now=T0,
    )
    records = [json.loads(l) for l in apath.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["decision"] == "auto_demoted"
    assert records[0]["challenger_id"] == "s3"
    assert records[0]["event"] == "auto_demoted"


def test_no_human_approval_required_for_demotion():
    sig = inspect.signature(auto_demote_on_dd_breach)
    params = sig.parameters
    # Signature must have no slack_fn / approver kwargs.
    assert not any("slack" in name.lower() for name in params)
    assert not any("approv" in name.lower() for name in params)


def test_idempotent_second_call_no_op(tmp_path):
    spath = tmp_path / "state.json"
    apath = tmp_path / "audit.jsonl"
    auto_demote_on_dd_breach(
        challenger_id="s4",
        challenger_current_dd=-0.11,
        state_path=spath,
        audit_path=apath,
        now=T0,
    )
    r2 = auto_demote_on_dd_breach(
        challenger_id="s4",
        challenger_current_dd=-0.11,
        state_path=spath,
        audit_path=apath,
        now=T0,
    )
    assert r2["demoted"] is True
    assert r2["decision"] == "already_demoted"
    # Audit file should have only 1 record (no duplicate from re-fire).
    records = [json.loads(l) for l in apath.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1


def test_jsonl_appends_not_overwrites(tmp_path):
    """Multiple different challengers write multiple JSONL records."""
    apath = tmp_path / "audit.jsonl"
    spath = tmp_path / "state.json"
    for tid, dd in [("a", -0.11), ("b", -0.12), ("c", -0.13)]:
        auto_demote_on_dd_breach(
            challenger_id=tid,
            challenger_current_dd=dd,
            state_path=spath,
            audit_path=apath,
            now=T0,
        )
    records = [json.loads(l) for l in apath.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    assert {r["challenger_id"] for r in records} == {"a", "b", "c"}


def test_ledger_notes_preserved_on_demotion(tmp_path):
    lpath = tmp_path / "ledger.tsv"
    # Seed a Friday row.
    weekly_ledger.append_row(
        week_iso="2026-W17",
        thu_batch_id="uuid-stub",
        thu_candidates_kicked=128,
        notes="kicked_off; starting_alloc=0.05",
        path=lpath,
    )
    auto_demote_on_dd_breach(
        challenger_id="s5",
        challenger_current_dd=-0.11,
        state_path=tmp_path / "state.json",
        audit_path=tmp_path / "audit.jsonl",
        ledger_path=lpath,
        week_iso="2026-W17",
        now=T0,
    )
    rows = weekly_ledger.read_rows(path=lpath)
    assert len(rows) == 1
    notes = rows[0]["notes"]
    assert "kicked_off" in notes
    assert "starting_alloc=0.05" in notes
    assert "auto_demoted:s5" in notes


def test_imports_dd_trigger_from_promoter():
    """DD_TRIGGER must be the same as promoter.DD_TRIGGER (single source of truth)."""
    from backend.autoresearch.promoter import DD_TRIGGER as PROMOTER_TRIGGER
    from backend.autoresearch.rollback import DD_TRIGGER as ROLLBACK_TRIGGER
    assert PROMOTER_TRIGGER == ROLLBACK_TRIGGER


def test_exact_boundary_dd_equals_threshold_no_breach(tmp_path):
    """qa_107 boundary-coverage gap (mutation M3): dd==threshold must NOT demote.

    Docstring says 'exceeds DD_TRIGGER' -- i.e., abs(dd) > threshold triggers.
    At abs(dd) == threshold, the decision must be `no_breach`. Without this
    test, flipping `<=` to `<` in the no-breach guard is undetected.
    """
    r = auto_demote_on_dd_breach(
        challenger_id="boundary",
        challenger_current_dd=-0.10,
        state_path=tmp_path / "state.json",
        audit_path=tmp_path / "audit.jsonl",
        now=T0,
    )
    assert r["demoted"] is False
    assert r["decision"] == "no_breach"


def test_return_dict_shape(tmp_path):
    r = auto_demote_on_dd_breach(
        challenger_id="s6",
        challenger_current_dd=-0.11,
        state_path=tmp_path / "state.json",
        audit_path=tmp_path / "audit.jsonl",
        now=T0,
    )
    for key in ("demoted", "decision", "challenger_id", "dd", "threshold", "ts"):
        assert key in r
