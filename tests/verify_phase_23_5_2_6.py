"""phase-23.5.2.6 verifier — watchdog_health_check spam fix.

Replays the immutable verification from `.claude/masterplan.json::23.5.2.6`:

  1. Watchdog source must NOT use the Docker alias hostname (regression
     guard against the every-15-min spam bug).
  2. Probe URL must reach localhost / 127.0.0.1.
  3. Module exposes the state-machine symbol `_watchdog_last_was_healthy`.
  4. The 6 unit tests in tests/slack_bot/test_watchdog_alert_semantics.py
     all pass.

Exit 0 only when all 4 checks pass.

Run via:
    python tests/verify_phase_23_5_2_6.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCHEDULER_PY = REPO / "backend" / "slack_bot" / "scheduler.py"
TEST_FILE = REPO / "tests" / "slack_bot" / "test_watchdog_alert_semantics.py"


def check_no_docker_alias_in_watchdog() -> tuple[bool, str]:
    src = SCHEDULER_PY.read_text(encoding="utf-8")
    # The Docker alias _BACKEND_URL constant may still be defined for other
    # call sites, but the watchdog probe MUST NOT reach the alias hostname.
    # Locate the watchdog function body and grep within it.
    m = re.search(
        r"async def _watchdog_health_check\(.*?\):\n(.*?)\n(?:async def|def [a-z]|class )",
        src,
        flags=re.DOTALL,
    )
    if not m:
        return False, "could not locate _watchdog_health_check function body"
    body = m.group(1)
    if "://backend:8000" in body or "{_BACKEND_URL}/api/health" in body:
        return False, "watchdog body still references Docker alias"
    return True, "watchdog body free of Docker alias"


def check_probe_url_localhost() -> tuple[bool, str]:
    src = SCHEDULER_PY.read_text(encoding="utf-8")
    if "_HEALTH_PROBE_URL" not in src:
        return False, "_HEALTH_PROBE_URL constant missing"
    # The constant must point to localhost / 127.0.0.1.
    m = re.search(r'_HEALTH_PROBE_URL\s*=\s*"([^"]+)"', src)
    if not m:
        return False, "_HEALTH_PROBE_URL definition not parseable"
    url = m.group(1)
    if "127.0.0.1" not in url and "localhost" not in url:
        return False, f"_HEALTH_PROBE_URL not localhost-pinned: {url!r}"
    return True, f"_HEALTH_PROBE_URL = {url!r}"


def check_state_symbol_present() -> tuple[bool, str]:
    src = SCHEDULER_PY.read_text(encoding="utf-8")
    if "_watchdog_last_was_healthy" not in src:
        return False, "state-machine symbol _watchdog_last_was_healthy missing"
    return True, "state-machine symbol present"


def check_tests_pass() -> tuple[bool, str]:
    if not TEST_FILE.exists():
        return False, f"test file missing: {TEST_FILE}"
    pytest_bin = REPO / ".venv" / "bin" / "python"
    bin_str = str(pytest_bin) if pytest_bin.exists() else sys.executable
    p = subprocess.run(
        [bin_str, "-m", "pytest", str(TEST_FILE), "-q"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO,
    )
    if p.returncode != 0:
        return False, f"pytest failed (exit {p.returncode})\nstdout:\n{p.stdout}\nstderr:\n{p.stderr[-500:]}"
    return True, p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "pytest OK"


def main() -> int:
    checks = [
        ("no docker alias in watchdog", check_no_docker_alias_in_watchdog),
        ("probe URL is localhost",      check_probe_url_localhost),
        ("state symbol present",        check_state_symbol_present),
        ("unit tests pass",             check_tests_pass),
    ]
    print("=== phase-23.5.2.6 verifier ===")
    failed = []
    for label, fn in checks:
        ok, info = fn()
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {label}: {info}")
        if not ok:
            failed.append(label)
    print()
    if failed:
        print(f"FAIL ({len(failed)}/{len(checks)}): {failed}")
        return 1
    print(f"PASS ({len(checks)}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
