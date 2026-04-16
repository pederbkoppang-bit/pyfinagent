"""
Go-Live drill test: escalation path defined (Phase 4.4.5.2).

Standalone, stdlib-only drill. Verifies that the incident runbook exists with
all required sections and that the escalation code paths are wired in the
backend (SLA monitor, stuck-task reaper, watchdog, iMessage bridge).

The checklist item (4.4.5.2) requires:
  - Documented escalation ladder: Ford alerts -> iMessage -> manual intervention
  - `backend/slack_bot/app.py` has escalation message helpers
  - iMessage bridge (or equivalent mobile push) is wired

Run from the repo root:

    python scripts/go_live_drills/escalation_path_test.py

Exit code 0 on PASS, exit 1 on any failure.
"""

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNBOOK_PATH = REPO_ROOT / "docs" / "INCIDENT_RUNBOOK.md"
SLA_MONITOR_PATH = REPO_ROOT / "backend" / "services" / "sla_monitor.py"
STUCK_REAPER_PATH = REPO_ROOT / "backend" / "services" / "stuck_task_reaper.py"
QUEUE_NOTIF_PATH = REPO_ROOT / "backend" / "services" / "queue_notification.py"
SLACK_APP_PATH = REPO_ROOT / "backend" / "slack_bot" / "app.py"
SCHEDULER_PATH = REPO_ROOT / "backend" / "slack_bot" / "scheduler.py"

PASS_COUNT = 0
FAIL_COUNT = 0


def _report(name: str, passed: bool, detail: str = ""):
    global PASS_COUNT, FAIL_COUNT
    if passed:
        PASS_COUNT += 1
        print(f"  PASS  {name}" + (f" -- {detail}" if detail else ""))
    else:
        FAIL_COUNT += 1
        print(f"  FAIL  {name}" + (f" -- {detail}" if detail else ""))


def run_drill():
    # ========== A. RUNBOOK EXISTENCE AND STRUCTURE ==========

    _report(
        "S0: INCIDENT_RUNBOOK.md exists",
        RUNBOOK_PATH.is_file(),
        str(RUNBOOK_PATH),
    )
    if not RUNBOOK_PATH.is_file():
        print("FATAL: INCIDENT_RUNBOOK.md not found, cannot proceed")
        return

    runbook = RUNBOOK_PATH.read_text(encoding="utf-8")

    required_sections = [
        "Escalation Ladder",
        "Priority Definitions",
        "Automatic Escalation Services",
        "Incident Response Procedures",
        "iMessage Bridge",
        "Contact Information",
        "Post-Incident Review",
    ]
    found_sections = []
    for section in required_sections:
        found = section.lower() in runbook.lower()
        found_sections.append(found)

    _report(
        "S1: runbook has all 7 required sections",
        all(found_sections),
        f"{sum(found_sections)}/7 found"
        + (
            "; missing: "
            + ", ".join(
                s for s, f in zip(required_sections, found_sections) if not f
            )
            if not all(found_sections)
            else ""
        ),
    )

    _report(
        "S2: escalation ladder documents Ford -> iMessage -> Peder path",
        "imessage" in runbook.lower()
        and "peder" in runbook.lower()
        and "ford" in runbook.lower(),
        "all three actors present in runbook",
    )

    _report(
        "S3: phone number documented",
        "+4794810537" in runbook,
        "Peder's escalation phone present",
    )

    _report(
        "S4: P0-P3 priority levels documented",
        all(f"P{i}" in runbook for i in range(4)),
        "all four priority levels found",
    )

    _report(
        "S5: SLA thresholds documented (5 min / 30 min for P0)",
        "5 min" in runbook and "30 min" in runbook,
        "P0 response and resolution SLAs present",
    )

    # ========== B. SLA MONITOR ESCALATION CODE ==========

    _report(
        "S6: sla_monitor.py exists",
        SLA_MONITOR_PATH.is_file(),
        str(SLA_MONITOR_PATH),
    )
    if SLA_MONITOR_PATH.is_file():
        sla_src = SLA_MONITOR_PATH.read_text(encoding="utf-8")
        sla_tree = ast.parse(sla_src)

        class_names = [
            n.name for n in ast.walk(sla_tree) if isinstance(n, ast.ClassDef)
        ]
        _report(
            "S7: SLAMonitoringService class defined",
            "SLAMonitoringService" in class_names,
        )

        method_names = [
            n.name
            for n in ast.walk(sla_tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        _report(
            "S8: send_escalation_alert method defined",
            "send_escalation_alert" in method_names,
        )

        _report(
            "S9: iMessage CLI invoked in sla_monitor",
            "imsg" in sla_src and "send" in sla_src,
            "imsg send command found",
        )

        _report(
            "S10: escalation phone hardcoded in sla_monitor",
            "+4794810537" in sla_src,
            "phone number matches runbook",
        )

        _report(
            "S11: P0 resolution breach triggers escalation",
            "'P0'" in sla_src and "resolution" in sla_src,
            "P0 resolution SLA breach path present",
        )

    # ========== C. BACKGROUND SERVICES WIRED IN APP.PY ==========

    _report(
        "S12: app.py exists",
        SLACK_APP_PATH.is_file(),
    )
    if SLACK_APP_PATH.is_file():
        app_src = SLACK_APP_PATH.read_text(encoding="utf-8")

        _report(
            "S13: start_sla_monitoring imported and called",
            "start_sla_monitoring" in app_src,
            "SLA monitor wired as background task",
        )

        _report(
            "S14: start_stuck_task_reaper imported and called",
            "start_stuck_task_reaper" in app_src,
            "stuck-task reaper wired as background task",
        )

        _report(
            "S15: start_scheduler imported and called",
            "start_scheduler" in app_src,
            "scheduler (watchdog) wired",
        )

    # ========== D. STUCK-TASK REAPER ==========

    _report(
        "S16: stuck_task_reaper.py exists",
        STUCK_REAPER_PATH.is_file(),
    )
    if STUCK_REAPER_PATH.is_file():
        reaper_src = STUCK_REAPER_PATH.read_text(encoding="utf-8")
        _report(
            "S17: StuckTaskReaper class defined",
            "class StuckTaskReaper" in reaper_src,
        )

    # ========== E. QUEUE FAILOVER NOTIFICATIONS ==========

    _report(
        "S18: queue_notification.py exists",
        QUEUE_NOTIF_PATH.is_file(),
    )
    if QUEUE_NOTIF_PATH.is_file():
        qn_src = QUEUE_NOTIF_PATH.read_text(encoding="utf-8")
        _report(
            "S19: send_failover_notification method defined",
            "send_failover_notification" in qn_src,
        )

    # ========== F. WATCHDOG HEALTH CHECK ==========

    _report(
        "S20: scheduler.py exists",
        SCHEDULER_PATH.is_file(),
    )
    if SCHEDULER_PATH.is_file():
        sched_src = SCHEDULER_PATH.read_text(encoding="utf-8")
        _report(
            "S21: watchdog_health_check job registered",
            "watchdog_health_check" in sched_src,
            "interval-based health probe present",
        )

    # ========== G. RUNBOOK CROSS-REFERENCES ==========

    _report(
        "S22: runbook references sla_monitor.py",
        "sla_monitor" in runbook,
        "code path traceable from runbook",
    )

    _report(
        "S23: runbook references ROLLBACK_PLAN.md",
        "ROLLBACK_PLAN" in runbook,
        "rollback cross-reference present",
    )

    _report(
        "S24: runbook references known-blockers.md for post-incident",
        "known-blockers" in runbook,
        "post-incident review references incident log",
    )


def main():
    print(f"Phase 4.4.5.2 Escalation Path Drill")
    print(f"{'=' * 50}")
    run_drill()
    print(f"{'=' * 50}")

    total = PASS_COUNT + FAIL_COUNT
    if FAIL_COUNT == 0:
        print(f"DRILL PASS: {PASS_COUNT}/{total} escalation path checks verified")
        sys.exit(0)
    else:
        print(f"DRILL FAIL: {FAIL_COUNT}/{total} checks failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
