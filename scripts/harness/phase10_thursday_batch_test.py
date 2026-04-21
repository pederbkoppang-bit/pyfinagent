"""phase-10.3 verification CLI: Thursday batch trigger.

Three test cases mapping to the masterplan success criteria:
  1. routine_consumes_exactly_1_slot
  2. ge_100_candidates_kicked_off
  3. batch_id_persisted_to_weekly_ledger

Each case runs inside a tempfile.TemporaryDirectory so the real
weekly_ledger.tsv is never touched. Prints PASS/FAIL per case;
exits 0 iff all three pass.
"""
from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.autoresearch import weekly_ledger
from backend.autoresearch.thursday_batch import trigger_thursday_batch


def _is_valid_uuid(s: str) -> bool:
    try:
        uuid.UUID(s)
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def case_consumes_exactly_one_slot() -> bool:
    with tempfile.TemporaryDirectory() as td:
        lpath = Path(td) / "ledger.tsv"
        r1 = trigger_thursday_batch("2026-W17", ledger_path=lpath)
        r2 = trigger_thursday_batch("2026-W17", ledger_path=lpath)
        rows = weekly_ledger.read_rows(path=lpath)

        ok = (
            r1["already_fired"] is False
            and r2["already_fired"] is True
            and len(rows) == 1
            and r1["batch_id"] == r2["batch_id"]
        )
        print(
            f"[{'PASS' if ok else 'FAIL'}] routine_consumes_exactly_1_slot  "
            f"(r1.already_fired={r1['already_fired']}, "
            f"r2.already_fired={r2['already_fired']}, rows={len(rows)})"
        )
        return ok


def case_kicks_ge_100_candidates() -> bool:
    with tempfile.TemporaryDirectory() as td:
        lpath = Path(td) / "ledger.tsv"
        r = trigger_thursday_batch("2026-W18", ledger_path=lpath)
        rows = weekly_ledger.read_rows(path=lpath)
        persisted = int(rows[0]["thu_candidates_kicked"]) if rows else 0

        ok = (
            r["candidates_kicked"] >= 100
            and persisted >= 100
            and persisted == r["candidates_kicked"]
        )
        print(
            f"[{'PASS' if ok else 'FAIL'}] ge_100_candidates_kicked_off  "
            f"(returned={r['candidates_kicked']}, persisted={persisted})"
        )
        return ok


def case_batch_id_persisted() -> bool:
    with tempfile.TemporaryDirectory() as td:
        lpath = Path(td) / "ledger.tsv"
        r = trigger_thursday_batch("2026-W19", ledger_path=lpath)
        rows = weekly_ledger.read_rows(path=lpath)
        persisted_id = rows[0]["thu_batch_id"] if rows else ""

        ok = (
            _is_valid_uuid(r["batch_id"])
            and persisted_id == r["batch_id"]
            and persisted_id != ""
        )
        print(
            f"[{'PASS' if ok else 'FAIL'}] batch_id_persisted_to_weekly_ledger  "
            f"(batch_id={r['batch_id'][:8]}..., persisted={persisted_id[:8]}...)"
        )
        return ok


def main() -> int:
    results = [
        case_consumes_exactly_one_slot(),
        case_kicks_ge_100_candidates(),
        case_batch_id_persisted(),
    ]
    ok = all(results)
    print(f"\n{'ALL PASS' if ok else 'FAILED'}  ({sum(results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
