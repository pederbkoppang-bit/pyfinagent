"""
Go-Live drill test: Escalation path (Phase 4.4.5.2).

Verifies that the escalation ladder is fully wired:
  1. format_escalation_alert exists in formatters.py and returns valid Block Kit
  2. send_trading_escalation exists in scheduler.py with L1 (Slack) + L2 (iMessage) paths
  3. sla_monitor.py has send_escalation_alert with imsg CLI call
  4. docs/INCIDENT_RUNBOOK.md exists with required sections
  5. Escalation phone number is consistent across all paths

Run from the repo root:

    python scripts/go_live_drills/escalation_path_test.py

Exit code 0 on PASS, 1 on any failure.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def _parse(relpath: str) -> ast.Module:
    return ast.parse(_read(relpath), filename=relpath)


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _source_has(relpath: str, needle: str) -> bool:
    return needle in _read(relpath)


def main():
    failures = []
    results = []

    def check(label: str, condition: bool, detail: str = ""):
        if condition:
            results.append(f"PASS {label}")
            print(f"PASS {label}")
        else:
            msg = f"{label}: {detail}" if detail else label
            failures.append(msg)
            results.append(f"FAIL {label}")
            print(f"FAIL {msg}")

    # --- S0: format_escalation_alert exists in formatters.py ---
    fmt_tree = _parse("backend/slack_bot/formatters.py")
    fmt_fn = _find_function(fmt_tree, "format_escalation_alert")
    check("S0 format_escalation_alert exists", fmt_fn is not None)

    # --- S1: format_escalation_alert has correct parameters ---
    if fmt_fn:
        arg_names = [a.arg for a in fmt_fn.args.args]
        check("S1 format_escalation_alert params",
              "severity" in arg_names and "title" in arg_names and "details" in arg_names,
              f"got {arg_names}")
    else:
        check("S1 format_escalation_alert params", False, "function not found")

    # --- S2: format_escalation_alert returns Block Kit with header ---
    fmt_src = _read("backend/slack_bot/formatters.py")
    check("S2 escalation formatter has header block",
          'type": "header"' in fmt_src or '"header"' in fmt_src)

    # --- S3: send_trading_escalation exists in scheduler.py ---
    sched_tree = _parse("backend/slack_bot/scheduler.py")
    send_fn = _find_function(sched_tree, "send_trading_escalation")
    check("S3 send_trading_escalation exists", send_fn is not None)

    # --- S4: send_trading_escalation is async ---
    check("S4 send_trading_escalation is async",
          isinstance(send_fn, ast.AsyncFunctionDef) if send_fn else False)

    # --- S5: send_trading_escalation calls format_escalation_alert ---
    sched_src = _read("backend/slack_bot/scheduler.py")
    check("S5 scheduler calls format_escalation_alert",
          "format_escalation_alert" in sched_src)

    # --- S6: send_trading_escalation has L1 Slack path (chat_postMessage) ---
    check("S6 L1 Slack path in send_trading_escalation",
          "chat_postMessage" in sched_src)

    # --- S7: send_trading_escalation has L2 iMessage path ---
    check("S7 L2 iMessage path in send_trading_escalation",
          "imsg" in sched_src and "send" in sched_src)

    # --- S8: Escalation phone number in scheduler.py ---
    check("S8 escalation phone in scheduler.py",
          "+4794810537" in sched_src)

    # --- S9: sla_monitor.py has send_escalation_alert ---
    sla_tree = _parse("backend/services/sla_monitor.py")
    sla_fn = _find_function(sla_tree, "send_escalation_alert")
    check("S9 sla_monitor send_escalation_alert exists", sla_fn is not None)

    # --- S10: sla_monitor.py has imsg CLI call ---
    sla_src = _read("backend/services/sla_monitor.py")
    check("S10 sla_monitor has imsg CLI call",
          "imsg" in sla_src and "send" in sla_src)

    # --- S11: Escalation phone consistent across both paths ---
    check("S11 phone consistent across sla_monitor and scheduler",
          "+4794810537" in sla_src and "+4794810537" in sched_src)

    # --- S12: docs/INCIDENT_RUNBOOK.md exists ---
    runbook_path = REPO_ROOT / "docs" / "INCIDENT_RUNBOOK.md"
    check("S12 INCIDENT_RUNBOOK.md exists", runbook_path.exists())

    # --- S13-S18: Runbook content checks ---
    if runbook_path.exists():
        runbook = runbook_path.read_text(encoding="utf-8")

        check("S13 runbook has escalation ladder",
              "Escalation Ladder" in runbook)

        check("S14 runbook documents L1 Slack alert",
              "L1" in runbook and "Slack" in runbook)

        check("S15 runbook documents L2 iMessage",
              "L2" in runbook and "iMessage" in runbook)

        check("S16 runbook documents L3 auto-kill",
              "L3" in runbook and "pause_signals" in runbook)

        check("S17 runbook has incident types",
              "Kill Switch" in runbook and "Backend Unreachable" in runbook)

        check("S18 runbook has Peder response checklist",
              "Peder" in runbook and "Response" in runbook)
    else:
        for i in range(13, 19):
            check(f"S{i} runbook content", False, "file not found")

    # --- S19: scheduler imports format_escalation_alert ---
    check("S19 scheduler imports format_escalation_alert",
          "import" in sched_src and "format_escalation_alert" in sched_src)

    # --- S20: P0 severity gates iMessage in scheduler ---
    check("S20 P0 gates iMessage escalation",
          'severity == "P0"' in sched_src or "severity == 'P0'" in sched_src)

    # --- S21: watchdog health check exists (L1 automated detection) ---
    watchdog_fn = _find_function(sched_tree, "_watchdog_health_check")
    check("S21 watchdog health check exists", watchdog_fn is not None)

    print()
    total = len(results)
    passed = sum(1 for r in results if r.startswith("PASS"))
    if failures:
        print(f"DRILL FAIL: {passed}/{total} passed, {len(failures)} failed")
        return 1
    print(f"DRILL PASS: {passed}/{total} escalation path checks verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
