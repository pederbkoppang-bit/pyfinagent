"""phase-10.4 pytest companion to phase10_friday_promotion_test.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.autoresearch import weekly_ledger
from backend.autoresearch.friday_promotion import run_friday_promotion
from backend.autoresearch.thursday_batch import trigger_thursday_batch


def _good(tid: str, dsr: float = 0.99, pbo: float = 0.10) -> dict:
    return {"trial_id": tid, "dsr": dsr, "pbo": pbo}


def _bad(tid: str) -> dict:
    return {"trial_id": tid, "dsr": 0.90, "pbo": 0.10}


def _seed(lpath: Path, wk: str) -> None:
    trigger_thursday_batch(wk, ledger_path=lpath)


# --- 4 CLI cases ---------------------------------------------------------


def test_consumes_exactly_one_slot(tmp_path):
    lpath = tmp_path / "ledger.tsv"
    _seed(lpath, "2026-W17")
    cands = [_good("t1"), _good("t2", dsr=0.98)]
    r1 = run_friday_promotion("2026-W17", candidates=cands, ledger_path=lpath)
    r2 = run_friday_promotion("2026-W17", candidates=cands, ledger_path=lpath)
    rows = weekly_ledger.read_rows(path=lpath)
    assert r1["already_fired"] is False
    assert r2["already_fired"] is True
    assert len(rows) == 1
    assert r1["promoted_ids"] == r2["promoted_ids"]


def test_reuses_phase_8_5_5_dsr_pbo_gate(tmp_path):
    lpath = tmp_path / "ledger.tsv"
    _seed(lpath, "2026-W18")
    cands = [_good("g1"), _bad("b1"), _bad("b2")]
    r = run_friday_promotion("2026-W18", candidates=cands, top_n=5, ledger_path=lpath)
    assert "b1" in r["rejected_ids"]
    assert "b2" in r["rejected_ids"]
    assert "g1" in r["promoted_ids"]


def test_promotion_at_5pct_starting_allocation(tmp_path):
    lpath = tmp_path / "ledger.tsv"
    _seed(lpath, "2026-W19")
    r = run_friday_promotion("2026-W19", candidates=[_good("g1")], ledger_path=lpath)
    assert r["allocations"] == [0.05]
    rows = weekly_ledger.read_rows(path=lpath)
    assert "starting_alloc=0.05" in rows[0]["notes"]


def test_top_n_default_1_max_3(tmp_path):
    five_good = [
        _good("g1", dsr=0.99),
        _good("g2", dsr=0.98),
        _good("g3", dsr=0.97),
        _good("g4", dsr=0.96),
        _good("g5", dsr=0.955),
    ]
    lpath = tmp_path / "a.tsv"
    _seed(lpath, "2026-W20")
    r_default = run_friday_promotion("2026-W20", candidates=five_good, ledger_path=lpath)
    assert len(r_default["promoted_ids"]) == 1
    assert r_default["promoted_ids"] == ["g1"]

    lpath_b = tmp_path / "b.tsv"
    _seed(lpath_b, "2026-W20")
    r_three = run_friday_promotion(
        "2026-W20", candidates=five_good, top_n=3, ledger_path=lpath_b
    )
    assert len(r_three["promoted_ids"]) == 3
    assert r_three["promoted_ids"] == ["g1", "g2", "g3"]

    lpath_c = tmp_path / "c.tsv"
    _seed(lpath_c, "2026-W20")
    r_cap = run_friday_promotion(
        "2026-W20", candidates=five_good, top_n=5, max_n=3, ledger_path=lpath_c
    )
    assert len(r_cap["promoted_ids"]) == 3


# --- edge cases beyond the 4 --------------------------------------------


def test_fail_closed_when_thursday_row_missing(tmp_path):
    lpath = tmp_path / "ledger.tsv"
    # No _seed(lpath) — ledger is empty
    r = run_friday_promotion("2026-W21", candidates=[_good("g1")], ledger_path=lpath)
    assert r["error"] == "no_thursday_batch_on_ledger"
    assert r["promoted_ids"] == []
    assert r["already_fired"] is False


def test_fail_closed_when_thu_batch_id_empty(tmp_path):
    lpath = tmp_path / "ledger.tsv"
    # Seed a ledger row with empty thu_batch_id
    weekly_ledger.append_row(
        week_iso="2026-W22",
        thu_batch_id="",
        thu_candidates_kicked=0,
        path=lpath,
    )
    r = run_friday_promotion("2026-W22", candidates=[_good("g1")], ledger_path=lpath)
    assert r["error"] == "no_thursday_batch_on_ledger"


def test_empty_candidates_does_not_raise(tmp_path):
    lpath = tmp_path / "ledger.tsv"
    _seed(lpath, "2026-W23")
    r = run_friday_promotion("2026-W23", candidates=[], ledger_path=lpath)
    assert r["promoted_ids"] == []
    assert r["rejected_ids"] == []
    assert r["error"] is None


def test_preserves_thursday_notes_kicked_off(tmp_path):
    lpath = tmp_path / "ledger.tsv"
    _seed(lpath, "2026-W24")
    run_friday_promotion("2026-W24", candidates=[_good("g1")], ledger_path=lpath)
    rows = weekly_ledger.read_rows(path=lpath)
    notes = rows[0]["notes"]
    # Both Thursday and Friday markers must be present
    assert "kicked_off" in notes
    assert "starting_alloc=0.05" in notes


def test_ranks_by_dsr_desc_then_pbo_asc(tmp_path):
    lpath = tmp_path / "ledger.tsv"
    _seed(lpath, "2026-W25")
    cands = [
        {"trial_id": "high_dsr_high_pbo", "dsr": 0.99, "pbo": 0.18},
        {"trial_id": "high_dsr_low_pbo", "dsr": 0.99, "pbo": 0.05},  # should win tie-break
        {"trial_id": "mid_dsr", "dsr": 0.97, "pbo": 0.05},
    ]
    r = run_friday_promotion(
        "2026-W25", candidates=cands, top_n=1, ledger_path=lpath
    )
    assert r["promoted_ids"] == ["high_dsr_low_pbo"]
