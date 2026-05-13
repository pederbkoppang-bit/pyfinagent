"""verify_phase_25_L -- Drawdown alarm with tiered thresholds.

Verifies:
  1. `drawdown_alarm` module exists with DRAWDOWN_TIERS including 3%/5%/10%.
  2. `check_drawdown_alarms([healthy snapshots]) == []` (no breach).
  3. `check_drawdown_alarms([-3.5% drawdown snapshots])` returns exactly 1 entry
     (warn_3pct).
  4. `check_drawdown_alarms([-12% drawdown snapshots])` returns exactly 3 entries
     (all 3 tiers; both warn_5pct and critical_10pct should be P1).
  5. Behavioral round-trip -- patch `raise_cron_alert_sync`, call
     `emit_drawdown_alarms` with a -6% snapshot series; assert P1 dedup keys
     `drawdown_warn_3pct` + `drawdown_warn_5pct` were dispatched (critical_10pct NOT fired).
  6. `autonomous_loop.py` imports/uses `emit_drawdown_alarms` (covers
     `drawdown_threshold_check_in_morning_digest_or_cycle` criterion).

Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

results: list[tuple[str, bool, str]] = []


def claim(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))


# ── Claim 1: module + DRAWDOWN_TIERS ───────────────────────────────────
mod_path = REPO / "backend/services/drawdown_alarm.py"
mod_exists = mod_path.exists()
mod_src = mod_path.read_text(encoding="utf-8") if mod_exists else ""
has_3pct = '"warn_3pct"' in mod_src and "-0.03" in mod_src
has_5pct = '"warn_5pct"' in mod_src and "-0.05" in mod_src
has_10pct = '"critical_10pct"' in mod_src and "-0.10" in mod_src
claim(
    "1. p1_slack_alert_at_3pct_5pct_10pct_drawdown_tiers",
    mod_exists and has_3pct and has_5pct and has_10pct,
    f"exists={mod_exists} 3%={has_3pct} 5%={has_5pct} 10%={has_10pct}",
)


# Helpers for tier-check claims
def _series(navs: list[float]) -> list[dict]:
    return [{"total_nav": n} for n in navs]


# ── Claim 2: healthy = no breach ───────────────────────────────────────
from backend.services.drawdown_alarm import (  # noqa: E402
    check_drawdown_alarms,
    compute_drawdown_from_snapshots,
    emit_drawdown_alarms,
    DRAWDOWN_TIERS,
)

healthy = _series([10000.0, 10050.0, 10100.0, 10075.0, 10110.0])  # peak 10110, current 10110
breached_healthy = check_drawdown_alarms(healthy)
claim(
    "2. check_drawdown_alarms_returns_empty_on_healthy",
    breached_healthy == [],
    f"got {breached_healthy}",
)


# ── Claim 3: -3.5% drawdown -> warn_3pct only ──────────────────────────
mild = _series([10000.0, 10100.0, 9750.0])  # peak 10100, current 9750 -> -3.46%
breached_mild = check_drawdown_alarms(mild)
mild_names = {t[0] for t in breached_mild}
claim(
    "3. check_drawdown_alarms_returns_one_tier_at_minus_3_5pct",
    mild_names == {"warn_3pct"},
    f"breached_tiers={mild_names} (expected {{warn_3pct}})",
)


# ── Claim 4: -12% drawdown -> all 3 tiers ──────────────────────────────
severe = _series([10000.0, 10100.0, 8888.0])  # peak 10100, current 8888 -> -12%
breached_severe = check_drawdown_alarms(severe)
severe_names = {t[0] for t in breached_severe}
claim(
    "4. check_drawdown_alarms_returns_all_three_tiers_at_minus_12pct",
    severe_names == {"warn_3pct", "warn_5pct", "critical_10pct"},
    f"breached_tiers={severe_names}",
)


# ── Claim 5: behavioral round-trip ─────────────────────────────────────
try:
    with patch(
        "backend.services.observability.alerting.raise_cron_alert_sync"
    ) as mock_raise:
        mock_raise.return_value = True
        moderate = _series([10000.0, 10100.0, 9494.0])  # -6%
        fired = emit_drawdown_alarms(moderate, source="test_source")
        # Should fire 2 alerts: drawdown_warn_3pct + drawdown_warn_5pct (not critical_10pct)
        error_types = [c.kwargs.get("error_type") for c in mock_raise.call_args_list]
        severities = [c.kwargs.get("severity") for c in mock_raise.call_args_list]
        rt_ok = (
            "drawdown_warn_3pct" in error_types
            and "drawdown_warn_5pct" in error_types
            and "drawdown_critical_10pct" not in error_types
            and "P1" in severities
        )
        rt_detail = f"fired={fired} error_types={error_types} severities={severities}"
except Exception as e:
    rt_ok = False
    rt_detail = f"Exception: {type(e).__name__}: {e}"

claim(
    "5. behavioral_round_trip_fires_p1_at_minus_6pct",
    rt_ok,
    rt_detail,
)


# ── Claim 6: autonomous_loop wires emit_drawdown_alarms ────────────────
loop_src = (REPO / "backend/services/autonomous_loop.py").read_text(encoding="utf-8")
has_import = "from backend.services.drawdown_alarm import emit_drawdown_alarms" in loop_src
has_call = bool(re.search(r"emit_drawdown_alarms\(", loop_src))
claim(
    "6. drawdown_threshold_check_in_morning_digest_or_cycle",
    has_import and has_call,
    f"import={has_import} call_site={has_call}",
)


# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== phase-25.L verification ===\n")
all_pass = True
for name, ok, detail in results:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    if detail:
        print(f"        -> {detail}")
    if not ok:
        all_pass = False

print()
if all_pass:
    print(f"ALL {len(results)} CLAIMS PASS")
    sys.exit(0)
else:
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"{failed} of {len(results)} claims FAILED")
    sys.exit(1)
