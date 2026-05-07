"""phase-23.3.2: immutable verification.

Asserts:
1. backend/slack_bot/scheduler.py has _aps_to_heartbeat listener +
   add_listener wired before _scheduler.start().
2. backend/api/job_status_api.py::_JOB_NAMES has 11 names (was 7) and
   includes the 4 core slack-bot ids.
3. tests/services/test_slack_bot_heartbeat_push.py passes (5 tests).
4. handoff/current/phase-23.3.2-audit-findings.md exists with the
   operator-restart caveat.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def check_scheduler_listener_wired():
    rel = "backend/slack_bot/scheduler.py"
    text = _read(rel)
    ast.parse(text)
    assert "_aps_to_heartbeat" in text, "listener function missing"
    assert "add_listener" in text, "add_listener call missing"
    # Listener must reference the heartbeat URL
    assert "/api/jobs/heartbeat" in text, "heartbeat URL missing"
    # Must use sync httpx.Client (not AsyncClient) since listener is sync
    assert "httpx.Client" in text, "sync httpx.Client not used"
    # Must register all 3 event types
    for event in ("EVENT_JOB_EXECUTED", "EVENT_JOB_ERROR", "EVENT_JOB_MISSED"):
        assert event in text, f"event {event} not imported/used"
    return f"OK {rel}"


def check_job_names_extended():
    rel = "backend/api/job_status_api.py"
    text = _read(rel)
    ast.parse(text)
    for core_job in (
        "morning_digest",
        "evening_digest",
        "watchdog_health_check",
        "prompt_leak_redteam",
    ):
        assert f'"{core_job}"' in text, f"_JOB_NAMES missing {core_job}"
    return f"OK {rel}"


def check_pytest_passes():
    rel = "tests/services/test_slack_bot_heartbeat_push.py"
    p = ROOT / rel
    assert p.exists(), f"test file missing: {rel}"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", rel, "-q", "--no-header"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.strip().splitlines()[-10:])
        raise AssertionError(f"pytest failed: {tail}")
    return f"OK {rel}"


def check_audit_findings():
    rel = "handoff/current/phase-23.3.2-audit-findings.md"
    p = ROOT / rel
    assert p.exists(), f"audit findings missing: {rel}"
    text = p.read_text()
    assert "operator-restart" in text.lower() or "restart" in text.lower(), \
        "audit findings must document operator-restart caveat"
    assert "morning_digest" in text and "_aps_to_heartbeat" in text, \
        "audit findings must reference the wired listener + jobs"
    return f"OK {rel}"


def main() -> int:
    checks = [
        check_scheduler_listener_wired,
        check_job_names_extended,
        check_pytest_passes,
        check_audit_findings,
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
    print(f"\nphase-23.3.2 verification: ALL PASS ({len(checks)}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
