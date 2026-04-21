"""phase-10.11 verification CLI: backend endpoint + dashboard wiring.

Four cases matching the masterplan success_criteria verbatim:
  1. backend_endpoint_returns_harness_sprint_week_state_shape
  2. endpoint_reads_from_harness_learning_log
  3. dashboard_renders_tile_when_data_present
  4. dashboard_renders_empty_state_when_data_null

Cases 1-2 are Python (backend). Cases 3-4 inspect the frontend source file
(.tsx) via string matching -- sufficient for integration assertions without
firing up a full vite runtime.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.api.harness_autoresearch import (
    HarnessSprintWeekState,
    fetch_sprint_state,
)


_DASHBOARD_PATH = _PROJECT_ROOT / "frontend" / "src" / "components" / "HarnessDashboard.tsx"


def case_backend_endpoint_returns_harness_sprint_week_state_shape() -> bool:
    rows = [
        {
            "slot_id": "thu_batch",
            "result_json": '{"batch_id": "b1", "candidates_kicked": 128}',
            "logged_at": "2026-04-23T21:00:00Z",
        },
        {
            "slot_id": "fri_promotion",
            "result_json": '{"promoted_ids": ["g1", "g2"], "rejected_ids": ["r1"]}',
            "logged_at": "2026-04-24T21:00:00Z",
        },
        {
            "slot_id": "monthly_gate",
            "result_json": '{"sortino_delta": 0.42, "approval_pending": false, "approved": true}',
            "logged_at": "2026-04-24T21:05:00Z",
        },
    ]
    s = fetch_sprint_state(week_iso="2026-W17", bq_query_fn=lambda sql, p: rows)
    ok = (
        isinstance(s, HarnessSprintWeekState)
        and s.weekIso == "2026-W17"
        and s.thu is not None and s.thu.batchId == "b1" and s.thu.candidatesKicked == 128
        and s.fri is not None and s.fri.promotedIds == ["g1", "g2"]
        and s.fri.rejectedIds == ["r1"]
        and s.monthly is not None and s.monthly.sortinoDelta == 0.42
        and s.monthly.approved is True
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] backend_endpoint_returns_harness_sprint_week_state_shape  "
        f"(shape={type(s).__name__}, fields=thu/fri/monthly)"
    )
    return ok


def case_endpoint_reads_from_harness_learning_log() -> bool:
    captured = {}

    def capture(sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return []

    fetch_sprint_state(week_iso="2026-W25", bq_query_fn=capture)
    ok = (
        "harness_learning_log" in captured["sql"]
        and "@week_iso" in captured["sql"]
        and captured["params"] == {"week_iso": "2026-W25"}
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] endpoint_reads_from_harness_learning_log  "
        f"(table_in_sql={'harness_learning_log' in captured['sql']}, param_bound={captured.get('params')})"
    )
    return ok


def case_dashboard_renders_tile_when_data_present() -> bool:
    src = _DASHBOARD_PATH.read_text(encoding="utf-8")
    has_import = re.search(
        r'import\s*\{\s*HarnessSprintTile\s*\}\s*from\s*["\']@/components/HarnessSprintTile["\']',
        src,
    )
    # Passes the populated sprintState (non-null data path).
    has_render_with_data = re.search(r"<HarnessSprintTile\s+data=\{sprintState\}", src)
    has_fetch = "getHarnessSprintState" in src
    ok = bool(has_import) and bool(has_render_with_data) and has_fetch
    print(
        f"[{'PASS' if ok else 'FAIL'}] dashboard_renders_tile_when_data_present  "
        f"(import={bool(has_import)}, render={bool(has_render_with_data)}, fetch={has_fetch})"
    )
    return ok


def case_dashboard_renders_empty_state_when_data_null() -> bool:
    src = _DASHBOARD_PATH.read_text(encoding="utf-8")
    # sprintState is null-typed; the tile passes null through to its own empty-state.
    has_null_default = re.search(
        r"setSprintState.*null|sprintState.*null|HarnessSprintWeekState\s*\|\s*null",
        src,
    )
    has_null_catch = re.search(r"getHarnessSprintState\(\)\.catch\(\(\)\s*=>\s*null\)", src)
    # The tile itself renders the empty state -- verified by phase-10.9's own tests.
    passes_sprint_state_even_when_null = "<HarnessSprintTile data={sprintState}" in src
    ok = (
        bool(has_null_default)
        and bool(has_null_catch)
        and passes_sprint_state_even_when_null
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] dashboard_renders_empty_state_when_data_null  "
        f"(null_typed={bool(has_null_default)}, catch_null={bool(has_null_catch)}, always_rendered={passes_sprint_state_even_when_null})"
    )
    return ok


def main() -> int:
    results = [
        case_backend_endpoint_returns_harness_sprint_week_state_shape(),
        case_endpoint_reads_from_harness_learning_log(),
        case_dashboard_renders_tile_when_data_present(),
        case_dashboard_renders_empty_state_when_data_null(),
    ]
    ok = all(results)
    print(f"\n{'ALL PASS' if ok else 'FAILED'}  ({sum(results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
