"""phase-25.K verifier — kill-switch state changes to Slack.

Closes phase-24.5 audit F-5(b) + phase-24.8 F-2 (pause_signals at
scheduler.py:353-366 only logged INFO; no Slack escalation).

Run: source .venv/bin/activate && python3 tests/verify_phase_25_K.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCHEDULER = REPO / "backend" / "slack_bot" / "scheduler.py"


def main() -> int:
    if not SCHEDULER.exists():
        print(f"FAIL: {SCHEDULER} not found")
        return 1
    text = SCHEDULER.read_text(encoding="utf-8")
    results: list[tuple[str, str, str]] = []

    # Claim 1: pause_signals now accepts an optional app param
    sig = re.search(r'def pause_signals\s*\(\s*app\s*:\s*["\']?AsyncApp\s*\|\s*None["\']?\s*=\s*None\s*\)', text)
    results.append(("PASS" if sig else "FAIL",
                    "pause_signals_accepts_optional_app_param",
                    "pause_signals signature must accept app: AsyncApp | None = None"))

    # Claim 2: pause_signals schedules notify_kill_switch_activated via asyncio.create_task
    create_task_pattern = re.search(
        r'asyncio\.create_task\s*\(\s*notify_kill_switch_activated\s*\(',
        text,
    )
    results.append(("PASS" if create_task_pattern else "FAIL",
                    "pause_signals_calls_send_trading_escalation_before_shutdown",
                    "pause_signals must asyncio.create_task(notify_kill_switch_activated(...)) before shutdown"))

    # Claim 3: notify_kill_switch_activated function exists with P0 severity
    activate_fn = re.search(
        r'async def notify_kill_switch_activated\s*\([^)]+\).*?send_trading_escalation\s*\(.*?severity\s*=\s*["\']P0["\']',
        text,
        re.DOTALL,
    )
    results.append(("PASS" if activate_fn else "FAIL",
                    "kill_switch_activate_emits_p0_slack_escalation",
                    "async notify_kill_switch_activated must exist and call send_trading_escalation with severity=P0"))

    # Claim 4: notify_kill_switch_deactivated function exists with P1 severity
    deactivate_fn = re.search(
        r'async def notify_kill_switch_deactivated\s*\([^)]+\).*?send_trading_escalation\s*\(.*?severity\s*=\s*["\']P1["\']',
        text,
        re.DOTALL,
    )
    results.append(("PASS" if deactivate_fn else "FAIL",
                    "kill_switch_deactivate_emits_p1_slack_escalation",
                    "async notify_kill_switch_deactivated must exist and call send_trading_escalation with severity=P1"))

    # Claim 5: phase-25.K attribution
    results.append(("PASS" if "phase-25.K" in text else "FAIL",
                    "phase_25_K_attribution_comment_present",
                    "Comment must reference phase-25.K closure of phase-24.5 F-5(b)"))

    # Claim 6: AST clean
    try:
        ast.parse(text)
        ast_ok, ast_detail = True, ""
    except SyntaxError as e:
        ast_ok, ast_detail = False, f"SyntaxError: {e}"
    results.append(("PASS" if ast_ok else "FAIL",
                    "scheduler_py_syntax_clean", ast_detail))

    # Claim 7: notify_kill_switch_activated uses "Kill Switch Activated" title (operator-recognizable)
    title_pattern = re.search(r'title\s*=\s*["\']Kill Switch Activated["\']', text)
    results.append(("PASS" if title_pattern else "FAIL",
                    "kill_switch_activated_uses_recognizable_title",
                    "notify_kill_switch_activated must use 'Kill Switch Activated' title"))

    # --- Output ---
    print("=== phase-25.K (kill-switch Slack wiring) verifier ===")
    fail = 0
    for flag, name, detail in results:
        prefix = "[PASS]" if flag == "PASS" else "[FAIL]"
        print(f"  {prefix} {name}")
        if flag == "FAIL" and detail:
            print(f"         -> {detail}")
            fail += 1
    total = len(results)
    passed = total - fail
    verdict = "PASS" if fail == 0 else "FAIL"
    print(f"{verdict} ({passed}/{total}) EXIT={0 if fail == 0 else 1}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
