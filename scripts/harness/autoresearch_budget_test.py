"""phase-8.5.2 budget-enforcer verification script.

Exercises the three success criteria:
  1. wallclock_budget_termination_deterministic
  2. usd_budget_termination_deterministic
  3. budget_exceeded_alerts_to_slack

Prints per-case PASS/FAIL + aggregate PASS/FAIL. Exits 0 iff all three pass.

Run:
    python scripts/harness/autoresearch_budget_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.autoresearch.budget import BudgetEnforcer


def case_wallclock() -> tuple[bool, str]:
    """Wall-clock cap 0.2s; sleep 0.3s between ticks; expect termination."""
    e = BudgetEnforcer(wallclock_seconds=0.2, usd_budget=100.0)
    s = e.tick(0.0)
    if s["terminated"]:
        return False, "wallclock: terminated prematurely on tick 0"
    time.sleep(0.3)
    s = e.tick(0.0)
    if not s["terminated"]:
        return False, "wallclock: did NOT terminate after 0.3s vs 0.2s cap"
    if s["reason"] != "wallclock":
        return False, f"wallclock: terminated but reason={s['reason']!r}"
    return True, "wallclock termination deterministic"


def case_usd() -> tuple[bool, str]:
    """USD cap $5; tick $3 then $3; expect termination on second tick."""
    e = BudgetEnforcer(wallclock_seconds=3600.0, usd_budget=5.0)
    s = e.tick(3.0)
    if s["terminated"]:
        return False, "usd: terminated prematurely on tick 1 (spent=3 cap=5)"
    s = e.tick(3.0)
    if not s["terminated"]:
        return False, "usd: did NOT terminate after $6 vs $5 cap"
    if s["reason"] != "usd":
        return False, f"usd: terminated but reason={s['reason']!r}"
    return True, "usd termination deterministic"


def case_alert() -> tuple[bool, str]:
    """Injectable alert_fn called exactly once on first breach."""
    captive: list[tuple[str, dict]] = []

    def alert(reason: str, state: dict) -> None:
        captive.append((reason, state))

    e = BudgetEnforcer(wallclock_seconds=3600.0, usd_budget=1.0, alert_fn=alert)
    e.tick(2.0)  # breaches on first tick
    if len(captive) != 1:
        return False, f"alert: expected 1 alert, got {len(captive)}"
    reason, state = captive[0]
    if reason != "usd":
        return False, f"alert: expected reason='usd', got {reason!r}"
    # Subsequent tick should NOT re-alert
    e.tick(10.0)
    if len(captive) != 1:
        return False, f"alert: re-alerted on subsequent tick; got {len(captive)} total"
    return True, "alert_fn called exactly once on first breach"


def main() -> int:
    cases = [
        ("wallclock_budget_termination_deterministic", case_wallclock),
        ("usd_budget_termination_deterministic", case_usd),
        ("budget_exceeded_alerts_to_slack", case_alert),
    ]
    all_pass = True
    for name, fn in cases:
        try:
            ok, msg = fn()
        except Exception as exc:
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {name} -- {msg}")
        if not ok:
            all_pass = False
    print("---")
    print("PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
