"""phase-10.8 verification CLI: slot accounting.

Four cases matching the masterplan success_criteria:
  1. every_phase10_routine_logged
  2. label_phase_10_applied
  3. weekly_invariant_sum_equals_2
  4. bq_writes_go_to_pyfinagent_data_harness_learning_log
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.autoresearch.slot_accounting import (
    log_slot_usage,
    verify_weekly_invariant,
)


def case_every_phase10_routine_logged() -> bool:
    captured: list[tuple[str, list[dict]]] = []

    def capture(table, rows):
        captured.append((table, rows))
        return True

    routines = [
        ("thu_batch", "trigger_thursday_batch", {"batch_id": "uuid-1", "candidates_kicked": 128}),
        ("fri_promotion", "run_friday_promotion", {"promoted_ids": ["g1"], "allocations": [0.05]}),
        ("monthly_gate", "run_monthly_sortino_gate", {"gate_pass": True, "approval_pending": True}),
        ("rollback", "auto_demote_on_dd_breach", {"demoted": True, "decision": "auto_demoted"}),
    ]
    for slot, routine, result in routines:
        log_slot_usage(
            week_iso="2026-W17",
            slot_id=slot,
            routine=routine,
            result=result,
            bq_insert_fn=capture,
            now=datetime(2026, 4, 24, 21, 0, tzinfo=timezone.utc),
        )

    ok = (
        len(captured) == 4
        and all(len(rows) == 1 for _, rows in captured)
        and {rows[0]["slot_id"] for _, rows in captured}
            == {"thu_batch", "fri_promotion", "monthly_gate", "rollback"}
        and {rows[0]["routine"] for _, rows in captured}
            == {"trigger_thursday_batch", "run_friday_promotion",
                "run_monthly_sortino_gate", "auto_demote_on_dd_breach"}
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] every_phase10_routine_logged  "
        f"(captured={len(captured)}, slot_ids={{{','.join(sorted({r[1][0]['slot_id'] for r in captured}))}}})"
    )
    return ok


def case_label_phase_10_applied() -> bool:
    captured: list[dict] = []

    def capture(table, rows):
        captured.extend(rows)
        return True

    for slot in ("thu_batch", "fri_promotion", "monthly_gate", "rollback"):
        log_slot_usage(
            week_iso="2026-W18",
            slot_id=slot,
            routine="r",
            result={"ok": True},
            bq_insert_fn=capture,
        )

    ok = len(captured) == 4 and all(r["phase"] == "phase-10" for r in captured)
    print(
        f"[{'PASS' if ok else 'FAIL'}] label_phase_10_applied  "
        f"(all_phase10={all(r['phase'] == 'phase-10' for r in captured)}, count={len(captured)})"
    )
    return ok


def case_weekly_invariant_sum_equals_2() -> bool:
    # Stub ledger: fake BQ inserts into an in-memory list.
    store: list[dict] = []

    def capture(table, rows):
        store.extend(rows)
        return True

    def query_count(sql, params):
        # Emulate the WHERE clause in slot_accounting SQL.
        wk = params["week_iso"]
        return sum(
            1 for r in store
            if r["week_iso"] == wk
            and r["phase"] == "phase-10"
            and r["slot_id"] in ("thu_batch", "fri_promotion")
        )

    # Log Thursday + Friday for W17.
    log_slot_usage(week_iso="2026-W17", slot_id="thu_batch",
                   routine="trigger_thursday_batch", result={"ok": True},
                   bq_insert_fn=capture)
    log_slot_usage(week_iso="2026-W17", slot_id="fri_promotion",
                   routine="run_friday_promotion", result={"ok": True},
                   bq_insert_fn=capture)
    # Log monthly + rollback -- must NOT count toward the invariant.
    log_slot_usage(week_iso="2026-W17", slot_id="monthly_gate",
                   routine="run_monthly_sortino_gate", result={"ok": True},
                   bq_insert_fn=capture)
    log_slot_usage(week_iso="2026-W17", slot_id="rollback",
                   routine="auto_demote_on_dd_breach", result={"ok": True},
                   bq_insert_fn=capture)

    r = verify_weekly_invariant("2026-W17", bq_query_fn=query_count)
    ok = r["sum"] == 2 and r["satisfied"] is True
    print(
        f"[{'PASS' if ok else 'FAIL'}] weekly_invariant_sum_equals_2  "
        f"(sum={r['sum']}, satisfied={r['satisfied']})"
    )
    return ok


def case_bq_writes_go_to_pyfinagent_data_harness_learning_log() -> bool:
    captured_tables: list[str] = []

    def capture(table, rows):
        captured_tables.append(table)
        return True

    r = log_slot_usage(
        week_iso="2026-W17",
        slot_id="thu_batch",
        routine="trigger_thursday_batch",
        result={"ok": True},
        bq_insert_fn=capture,
    )

    ok = (
        captured_tables == ["pyfinagent_data.harness_learning_log"]
        and r["table"] == "pyfinagent_data.harness_learning_log"
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] bq_writes_go_to_pyfinagent_data_harness_learning_log  "
        f"(table={r['table']!r})"
    )
    return ok


def main() -> int:
    results = [
        case_every_phase10_routine_logged(),
        case_label_phase_10_applied(),
        case_weekly_invariant_sum_equals_2(),
        case_bq_writes_go_to_pyfinagent_data_harness_learning_log(),
    ]
    ok = all(results)
    print(f"\n{'ALL PASS' if ok else 'FAILED'}  ({sum(results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
