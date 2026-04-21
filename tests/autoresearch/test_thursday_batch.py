"""phase-10.3 pytest companion to scripts/harness/phase10_thursday_batch_test.py."""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.autoresearch import weekly_ledger
from backend.autoresearch.thursday_batch import trigger_thursday_batch


def test_consumes_exactly_one_slot(tmp_path):
    lpath = tmp_path / "ledger.tsv"
    r1 = trigger_thursday_batch("2026-W17", ledger_path=lpath)
    r2 = trigger_thursday_batch("2026-W17", ledger_path=lpath)
    rows = weekly_ledger.read_rows(path=lpath)

    assert r1["already_fired"] is False
    assert r2["already_fired"] is True
    assert r1["batch_id"] == r2["batch_id"]
    assert len(rows) == 1


def test_kicks_ge_100_candidates(tmp_path):
    lpath = tmp_path / "ledger.tsv"
    r = trigger_thursday_batch("2026-W18", ledger_path=lpath)
    assert r["candidates_kicked"] >= 100
    rows = weekly_ledger.read_rows(path=lpath)
    assert int(rows[0]["thu_candidates_kicked"]) >= 100


def test_batch_id_is_valid_uuid(tmp_path):
    lpath = tmp_path / "ledger.tsv"
    r = trigger_thursday_batch("2026-W19", ledger_path=lpath)
    # Must parse as a UUID
    uuid.UUID(r["batch_id"])
    # Persisted row echoes the same id
    rows = weekly_ledger.read_rows(path=lpath)
    assert rows[0]["thu_batch_id"] == r["batch_id"]


def test_batch_id_is_deterministic_per_week(tmp_path):
    """Second fire in a fresh ledger for same week_iso must produce same batch_id."""
    lpath_a = tmp_path / "a.tsv"
    lpath_b = tmp_path / "b.tsv"
    r_a = trigger_thursday_batch("2026-W20", ledger_path=lpath_a)
    r_b = trigger_thursday_batch("2026-W20", ledger_path=lpath_b)
    assert r_a["batch_id"] == r_b["batch_id"]


def test_different_weeks_produce_different_batch_ids(tmp_path):
    lpath = tmp_path / "ledger.tsv"
    a = trigger_thursday_batch("2026-W17", ledger_path=lpath)
    b = trigger_thursday_batch("2026-W18", ledger_path=lpath)
    assert a["batch_id"] != b["batch_id"]


def test_n_below_floor_raises(tmp_path):
    with pytest.raises(ValueError, match="below floor"):
        trigger_thursday_batch(
            "2026-W21", n_candidates=50, ledger_path=tmp_path / "ledger.tsv"
        )


def test_ledger_row_notes_kicked_off(tmp_path):
    lpath = tmp_path / "ledger.tsv"
    trigger_thursday_batch("2026-W22", ledger_path=lpath)
    rows = weekly_ledger.read_rows(path=lpath)
    assert rows[0]["notes"] == "kicked_off"
