"""phase-23.4.0 verifier.

Replays the immutable verification from `.claude/masterplan.json::23.4.0`:

  1. curl -s -o /dev/null -w %{http_code} http://localhost:3000/login -> 200
  2. curl -sL http://localhost:3000/  body contains <html or <title>
  3. cd frontend && npx --no-install tsc --noEmit -> exit 0
  4. cd frontend && npx --no-install eslint . --quiet -> exit 0

Exit 0 only when all 4 checks pass. Otherwise non-zero with a label
identifying which check failed (Q/A's deterministic leg). Stdout
prints a one-line PASS/FAIL summary per check.

Run via:
    cd /Users/ford/.openclaw/workspace/pyfinagent
    python tests/verify_phase_23_4_0.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FRONTEND = REPO / "frontend"
LOGIN_URL = "http://localhost:3000/login"
ROOT_URL = "http://localhost:3000/"


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=180, **kw)


def check_login_200() -> tuple[bool, str]:
    if shutil.which("curl") is None:
        return False, "curl not on PATH"
    p = _run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
             "--max-time", "10", LOGIN_URL])
    code = (p.stdout or "").strip()
    return code == "200", f"http_code={code or 'empty'}"


def check_root_html_marker() -> tuple[bool, str]:
    p = _run(["curl", "-sL", "--max-time", "10", ROOT_URL])
    body = p.stdout or ""
    if not body:
        return False, "empty response body"
    has_marker = ("<html" in body) or ("<title" in body)
    return has_marker, f"len={len(body)} has_html_or_title={has_marker}"


def check_tsc_clean() -> tuple[bool, str]:
    p = _run(["npx", "--no-install", "tsc", "--noEmit"], cwd=FRONTEND)
    return p.returncode == 0, f"tsc_exit={p.returncode}"


def check_eslint_quiet() -> tuple[bool, str]:
    p = _run(["npx", "--no-install", "eslint", ".", "--quiet"], cwd=FRONTEND)
    return p.returncode == 0, f"eslint_exit={p.returncode}"


def main() -> int:
    checks = [
        ("login 200",      check_login_200),
        ("root has html",  check_root_html_marker),
        ("tsc --noEmit",   check_tsc_clean),
        ("eslint quiet",   check_eslint_quiet),
    ]
    print("=== phase-23.4.0 verifier ===")
    failed = []
    for label, fn in checks:
        ok, info = fn()
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {label}: {info}")
        if not ok:
            failed.append(label)
    print()
    if failed:
        print(f"FAIL ({len(failed)}/{len(checks)} failed): {failed}")
        return 1
    print(f"PASS ({len(checks)}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
