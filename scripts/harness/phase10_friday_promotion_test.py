"""phase-10.4 verification CLI: Friday promotion gate.

Four cases mapping to the masterplan success_criteria:
  1. routine_consumes_exactly_1_slot
  2. reuses_phase_8_5_5_dsr_pbo_gate
  3. promotion_at_5pct_starting_allocation
  4. top_n_default_1_max_3

Each case runs inside a tempfile.TemporaryDirectory; no persistent ledger
touched. Thursday's row is seeded via `trigger_thursday_batch` so the
idempotency guard has a real batch_id to anchor against.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.autoresearch import weekly_ledger
from backend.autoresearch.friday_promotion import run_friday_promotion
from backend.autoresearch.thursday_batch import trigger_thursday_batch


# Reusable fixtures (mirror scripts/harness/autoresearch_gate_test.py).
def _good(trial_id: str, dsr: float = 0.99, pbo: float = 0.10) -> dict:
    return {"trial_id": trial_id, "dsr": dsr, "pbo": pbo}


def _bad(trial_id: str) -> dict:
    # dsr below min_dsr=0.95 -> fails gate
    return {"trial_id": trial_id, "dsr": 0.90, "pbo": 0.10}


def _seed_thursday(lpath: Path, week_iso: str) -> None:
    trigger_thursday_batch(week_iso, ledger_path=lpath)


def case_consumes_exactly_one_slot() -> bool:
    with tempfile.TemporaryDirectory() as td:
        lpath = Path(td) / "ledger.tsv"
        _seed_thursday(lpath, "2026-W17")
        cands = [_good("t1"), _good("t2", dsr=0.98)]
        r1 = run_friday_promotion("2026-W17", candidates=cands, ledger_path=lpath)
        r2 = run_friday_promotion("2026-W17", candidates=cands, ledger_path=lpath)
        rows = weekly_ledger.read_rows(path=lpath)
        ok = (
            r1["already_fired"] is False
            and r2["already_fired"] is True
            and len(rows) == 1
            and r1["promoted_ids"] == r2["promoted_ids"]
        )
        print(
            f"[{'PASS' if ok else 'FAIL'}] routine_consumes_exactly_1_slot  "
            f"(r1.af={r1['already_fired']}, r2.af={r2['already_fired']}, rows={len(rows)})"
        )
        return ok


def case_reuses_phase_8_5_5_dsr_pbo_gate() -> bool:
    with tempfile.TemporaryDirectory() as td:
        lpath = Path(td) / "ledger.tsv"
        _seed_thursday(lpath, "2026-W18")
        # Two good + two bad; top_n=5 so all passers promote, all failures reject.
        cands = [_good("g1"), _good("g2"), _bad("b1"), _bad("b2")]
        r = run_friday_promotion(
            "2026-W18", candidates=cands, top_n=5, ledger_path=lpath
        )
        ok = (
            "b1" in r["rejected_ids"]
            and "b2" in r["rejected_ids"]
            and ("g1" in r["promoted_ids"] or "g2" in r["promoted_ids"])
        )
        print(
            f"[{'PASS' if ok else 'FAIL'}] reuses_phase_8_5_5_dsr_pbo_gate  "
            f"(promoted={r['promoted_ids']}, rejected={r['rejected_ids']})"
        )
        return ok


def case_promotion_at_5pct_starting_allocation() -> bool:
    with tempfile.TemporaryDirectory() as td:
        lpath = Path(td) / "ledger.tsv"
        _seed_thursday(lpath, "2026-W19")
        r = run_friday_promotion(
            "2026-W19", candidates=[_good("g1")], ledger_path=lpath
        )
        rows = weekly_ledger.read_rows(path=lpath)
        notes = rows[0].get("notes", "") if rows else ""
        ok = (
            "starting_alloc=0.05" in notes
            and r["allocations"] == [0.05]
            and r["promoted_ids"] == ["g1"]
        )
        print(
            f"[{'PASS' if ok else 'FAIL'}] promotion_at_5pct_starting_allocation  "
            f"(notes={notes!r}, allocations={r['allocations']})"
        )
        return ok


def case_top_n_default_1_max_3() -> bool:
    with tempfile.TemporaryDirectory() as td:
        lpath_a = Path(td) / "a.tsv"
        lpath_b = Path(td) / "b.tsv"
        lpath_c = Path(td) / "c.tsv"
        _seed_thursday(lpath_a, "2026-W20")
        _seed_thursday(lpath_b, "2026-W20")
        _seed_thursday(lpath_c, "2026-W20")

        five_good = [
            _good("g1", dsr=0.99),
            _good("g2", dsr=0.98),
            _good("g3", dsr=0.97),
            _good("g4", dsr=0.96),
            _good("g5", dsr=0.955),
        ]
        r_default = run_friday_promotion(
            "2026-W20", candidates=five_good, ledger_path=lpath_a
        )
        r_three = run_friday_promotion(
            "2026-W20", candidates=five_good, top_n=3, ledger_path=lpath_b
        )
        r_cap = run_friday_promotion(
            "2026-W20", candidates=five_good, top_n=5, max_n=3, ledger_path=lpath_c
        )

        ok = (
            len(r_default["promoted_ids"]) == 1
            and r_default["promoted_ids"] == ["g1"]  # highest DSR
            and len(r_three["promoted_ids"]) == 3
            and r_three["promoted_ids"] == ["g1", "g2", "g3"]
            and len(r_cap["promoted_ids"]) == 3  # capped at max_n
        )
        print(
            f"[{'PASS' if ok else 'FAIL'}] top_n_default_1_max_3  "
            f"(default={len(r_default['promoted_ids'])}, "
            f"three={len(r_three['promoted_ids'])}, capped={len(r_cap['promoted_ids'])})"
        )
        return ok


def main() -> int:
    results = [
        case_consumes_exactly_one_slot(),
        case_reuses_phase_8_5_5_dsr_pbo_gate(),
        case_promotion_at_5pct_starting_allocation(),
        case_top_n_default_1_max_3(),
    ]
    ok = all(results)
    print(f"\n{'ALL PASS' if ok else 'FAILED'}  ({sum(results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
