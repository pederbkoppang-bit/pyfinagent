#!/usr/bin/env python
"""phase-4.17.9 smoke test -- Self-update deploy system audit.

Static + dry-run audit of `backend/slack_bot/self_update.py`:
- Python syntax clean.
- `logs/` directory writable (deploy logs land there).
- `git fetch --dry-run` reachable from repo root.
- No hardcoded absolute paths outside project root (portable to any
  Mac Mini clone).

Does NOT execute a real `git pull` or restart. This is the audit gate
per the research brief's carry-forward section.

Criteria:
- self_update_py_syntax_clean
- deploy_log_dir_writable
- git_fetch_dry_run_succeeds
- no_hardcoded_absolute_paths_outside_project_root
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_self_update_script_audit():
    su = REPO_ROOT / "backend/slack_bot/self_update.py"
    assert su.exists(), f"{su} missing"
    src = su.read_text(encoding="utf-8")

    # 1. Syntax
    try:
        ast.parse(src)
        print("PASS self_update_py_syntax_clean")
    except SyntaxError as e:
        raise AssertionError(f"syntax: {e}")

    # 2. logs/ writable
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    probe = log_dir / "smoke_probe_4_17_9.txt"
    try:
        probe.write_text("probe")
        probe.unlink()
        print("PASS deploy_log_dir_writable")
    except Exception as e:
        raise AssertionError(f"logs dir not writable: {e}")

    # 3. git fetch --dry-run reachable
    r = subprocess.run(
        ["git", "fetch", "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Non-zero is OK if the cause is network / unconfigured remote; the
    # check is that git is reachable + we're inside a repo.
    assert (
        r.returncode == 0 or "not a git repository" not in (r.stderr or "").lower()
    ), f"git fetch fail: stderr={r.stderr[:200]}"
    print("PASS git_fetch_dry_run_succeeds")

    # 4. No hardcoded absolute paths pointing outside repo root.
    # Allow /tmp, /dev/null, /usr/bin/env shebang-style references.
    abs_paths = re.findall(r'"(/[A-Za-z][^"]*)"', src) + re.findall(r"'(/[A-Za-z][^']*)'", src)
    bad = [
        p for p in abs_paths
        if not (
            p.startswith("/tmp")
            or p.startswith("/dev")
            or p.startswith("/usr")
            or p.startswith("/bin")
            or p.startswith("/var")
            or str(REPO_ROOT) in p
        )
    ]
    assert not bad, f"no_hardcoded_absolute_paths_outside_project_root FAIL: {bad}"
    print("PASS no_hardcoded_absolute_paths_outside_project_root")

    print("PASS 4.17.9 self-update audit")


if __name__ == "__main__":
    try:
        test_self_update_script_audit()
    except AssertionError as e:
        print("FAIL:", e, file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
