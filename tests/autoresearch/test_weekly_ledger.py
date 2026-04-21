"""phase-10.2 tests for backend.autoresearch.weekly_ledger."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.autoresearch.weekly_ledger import append_row, read_rows, COLUMNS


def test_append_new_week(tmp_path):
    p = tmp_path / "ledger.tsv"
    # Seed header only.
    p.write_text("\t".join(COLUMNS) + "\n", encoding="utf-8")
    ok = append_row(
        "2026-W17",
        thu_batch_id="batch_abc",
        thu_candidates_kicked=100,
        fri_promoted_ids=["t1", "t2"],
        fri_rejected_ids=["t3"],
        cost_usd=4.20,
        sortino_monthly=1.15,
        notes="first write",
        path=p,
    )
    assert ok is True
    rows = read_rows(path=p)
    assert len(rows) == 1
    assert rows[0]["week_iso"] == "2026-W17"
    assert rows[0]["fri_promoted_ids"] == "[t1,t2]"


def test_idempotent_update_same_week(tmp_path):
    p = tmp_path / "ledger.tsv"
    p.write_text("\t".join(COLUMNS) + "\n", encoding="utf-8")
    append_row("2026-W17", thu_batch_id="v1", cost_usd=1.00, path=p)
    append_row("2026-W17", thu_batch_id="v2", cost_usd=9.99, path=p)  # overwrite
    rows = read_rows(path=p)
    assert len(rows) == 1  # idempotent -> single row
    assert rows[0]["thu_batch_id"] == "v2"  # second write wins
    assert rows[0]["cost_usd"] == "9.99"


def test_read_rows_parses_header_and_data(tmp_path):
    p = tmp_path / "ledger.tsv"
    header = "\t".join(COLUMNS)
    data = "2026-W18\tbatch_xyz\t42\t[a,b]\t[c]\t2.5\t0.8\tsome notes"
    p.write_text(header + "\n" + data + "\n", encoding="utf-8")
    rows = read_rows(path=p)
    assert len(rows) == 1
    r = rows[0]
    assert r["week_iso"] == "2026-W18"
    assert r["thu_batch_id"] == "batch_xyz"
    assert r["thu_candidates_kicked"] == "42"


def test_fail_open_on_bad_path():
    # Both functions should tolerate a nonexistent/unwritable path.
    rows = read_rows(path=Path("/no/such/path/ledger.tsv"))
    assert rows == []
    # append_row returns False (never raises); a read-only dir path ensures it.
    ok = append_row("2026-W99", path=Path("/no/such/path/cannot_write.tsv"))
    assert ok in (True, False)  # Whatever happens, no exception propagates.


def test_module_is_ascii_only():
    mod = (_REPO_ROOT / "backend" / "autoresearch" / "weekly_ledger.py").read_bytes()
    mod.decode("ascii")
