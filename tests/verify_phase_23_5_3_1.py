"""phase-23.5.3.1 verifier — Docker-alias removal from digest handlers.

  1. _send_morning_digest body must NOT reference _BACKEND_URL or
     `://backend:8000`.
  2. _send_evening_digest body must NOT reference _BACKEND_URL or
     `://backend:8000`.
  3. Both functions must reference _LOCAL_BACKEND_URL (or 127.0.0.1).
  4. The 4 unit tests in tests/slack_bot/test_digest_url_semantics.py
     all pass.

Exit 0 only when all 4 checks pass.

Run via:
    python tests/verify_phase_23_5_3_1.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCHEDULER_PY = REPO / "backend" / "slack_bot" / "scheduler.py"
TEST_FILE = REPO / "tests" / "slack_bot" / "test_digest_url_semantics.py"


def _extract_function_body(src: str, name: str) -> str | None:
    pattern = rf"async def {re.escape(name)}\(.*?\):\n(.*?)\n(?:async def|def [a-z]|class )"
    m = re.search(pattern, src, flags=re.DOTALL)
    return m.group(1) if m else None


_BARE_BACKEND_URL_RE = re.compile(r"(?<!_LOCAL)_BACKEND_URL\b")


def check_morning_digest_clean() -> tuple[bool, str]:
    body = _extract_function_body(SCHEDULER_PY.read_text(encoding="utf-8"), "_send_morning_digest")
    if body is None:
        return False, "could not locate _send_morning_digest body"
    if _BARE_BACKEND_URL_RE.search(body) or "://backend:8000" in body:
        return False, "morning digest body still references docker alias"
    if "_LOCAL_BACKEND_URL" not in body and "127.0.0.1" not in body:
        return False, "morning digest body does not reference _LOCAL_BACKEND_URL or 127.0.0.1"
    return True, "morning digest uses _LOCAL_BACKEND_URL"


def check_evening_digest_clean() -> tuple[bool, str]:
    body = _extract_function_body(SCHEDULER_PY.read_text(encoding="utf-8"), "_send_evening_digest")
    if body is None:
        return False, "could not locate _send_evening_digest body"
    if _BARE_BACKEND_URL_RE.search(body) or "://backend:8000" in body:
        return False, "evening digest body still references docker alias"
    if "_LOCAL_BACKEND_URL" not in body and "127.0.0.1" not in body:
        return False, "evening digest body does not reference _LOCAL_BACKEND_URL or 127.0.0.1"
    return True, "evening digest uses _LOCAL_BACKEND_URL"


def check_constant_defined() -> tuple[bool, str]:
    src = SCHEDULER_PY.read_text(encoding="utf-8")
    m = re.search(r'_LOCAL_BACKEND_URL\s*=\s*"([^"]+)"', src)
    if not m:
        return False, "_LOCAL_BACKEND_URL constant missing"
    url = m.group(1)
    if "127.0.0.1" not in url and "localhost" not in url:
        return False, f"_LOCAL_BACKEND_URL not localhost-pinned: {url!r}"
    return True, f"_LOCAL_BACKEND_URL = {url!r}"


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
        ("morning digest clean", check_morning_digest_clean),
        ("evening digest clean", check_evening_digest_clean),
        ("constant defined",     check_constant_defined),
        ("unit tests pass",      check_tests_pass),
    ]
    print("=== phase-23.5.3.1 verifier ===")
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
