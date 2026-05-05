"""phase-23.2.19: immutable verification.

Asserts:
1. backend/services/kill_switch.py: _sod_date field, update_sod_nav accepts
   date kwarg, snapshot includes sod_date, boot replay handles legacy rows
   via ts fallback.
2. backend/services/paper_trader.py: daily-roll guard compares snap["sod_date"]
   against today.
3. backend/api/paper_trading.py: kill-switch endpoint exposes sod_date.
4. frontend/src/components/OpsStatusBar.tsx: GateSegment tooltip references
   each of the 5 boolean keys + builds a multi-line title.
5. tests/services/test_sod_daily_roll.py exists with the expected test names.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def check_kill_switch():
    rel = "backend/services/kill_switch.py"
    text = _read(rel)
    ast.parse(text)
    assert "_sod_date" in text, "_sod_date field missing"
    assert "def update_sod_nav(self, nav: float, date:" in text, \
        "update_sod_nav must accept optional `date` kwarg"
    assert '"sod_date": self._sod_date' in text, \
        "snapshot payload must include sod_date"
    assert "row.get(\"date\")" in text or "row.get('date')" in text, \
        "boot replay must read explicit `date` field from audit row"
    assert "fromisoformat" in text, "boot replay must fall back to ts parse for legacy rows"
    return f"OK {rel}"


def check_paper_trader():
    rel = "backend/services/paper_trader.py"
    text = _read(rel)
    ast.parse(text)
    assert 'snap.get("sod_date")' in text, \
        "daily-roll guard must compare snap['sod_date'] against today"
    assert 'state.update_sod_nav(nav, date=today)' in text, \
        "must call update_sod_nav with date=today on roll"
    return f"OK {rel}"


def check_paper_trading_api():
    rel = "backend/api/paper_trading.py"
    text = _read(rel)
    ast.parse(text)
    assert 'state.get("sod_date")' in text or '"sod_date":' in text, \
        "kill-switch endpoint must expose sod_date"
    return f"OK {rel}"


def check_ops_status_bar():
    rel = "frontend/src/components/OpsStatusBar.tsx"
    text = _read(rel)
    # all 5 boolean keys must be referenced in the new tooltip
    for key in (
        "trades_ge_100",
        "psr_ge_95_sustained_30d",
        "dsr_ge_95",
        "sr_gap_le_30pct",
        "max_dd_within_tolerance",
    ):
        assert key in text, f"GateSegment tooltip must reference boolean `{key}`"
    assert "tooltipLines" in text or "tooltip" in text, \
        "GateSegment must build a multi-line tooltip variable"
    assert 'title={tooltip}' in text or 'title={tooltipLines.join' in text, \
        "GateSegment must wire the multi-line tooltip into title="
    return f"OK {rel}"


def check_test_exists():
    rel = "tests/services/test_sod_daily_roll.py"
    text = _read(rel)
    ast.parse(text)
    for fn in (
        "test_snapshot_now_includes_sod_date",
        "test_update_sod_nav_stamps_explicit_date_in_audit_row",
        "test_paper_trader_rolls_sod_on_new_day",
        "test_paper_trader_does_not_roll_same_day",
        "test_boot_replay_falls_back_to_ts_for_legacy_rows",
        "test_legacy_row_then_new_day_rolls_correctly",
    ):
        assert fn in text, f"missing test: {fn}"
    return f"OK {rel}"


def main() -> int:
    checks = [
        check_kill_switch,
        check_paper_trader,
        check_paper_trading_api,
        check_ops_status_bar,
        check_test_exists,
    ]
    failed = 0
    for fn in checks:
        try:
            print(fn())
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {fn.__name__}: {e!r}")
            failed += 1
    if failed:
        print(f"\n{failed} verification(s) failed")
        return 1
    print("\nphase-23.2.19 verification: ALL PASS (5/5)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
