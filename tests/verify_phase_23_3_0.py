"""phase-23.3.0: immutable verification.

Asserts:
1. .claude/agents/qa.md has the literal '### 1b. Frontend lint' header
   AND a literal `npx eslint .` AND `tsc --noEmit` command.
2. The phase-23.2.24 commit is on origin/main (next session pulling
   main will have the new rubric).
3. scripts/qa/verify_qa_roster_live.sh exists, is executable, and has
   the expected operator-prompt block.
4. CLAUDE.md cross-references the smoke script + retry-on-FAIL section.
5. bash -n on the smoke script exits 0.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def check_qa_md_section_and_commands():
    rel = ".claude/agents/qa.md"
    text = _read(rel)
    assert "### 1b. Frontend lint" in text, \
        "qa.md missing '### 1b. Frontend lint' section header"
    assert "npx eslint ." in text, \
        "qa.md must include `npx eslint .` literal command"
    assert "tsc --noEmit" in text, \
        "qa.md must include `tsc --noEmit` literal command"
    return f"OK {rel}"


def check_phase_23_2_24_commit_on_origin_main():
    """Verify the commit that introduced section 1b has been pushed."""
    proc = subprocess.run(
        ["git", "log", "--grep", "phase-23.2.24:", "--format=%H", "-n", "1"],
        cwd=ROOT, capture_output=True, text=True, timeout=20,
    )
    assert proc.returncode == 0, f"git log failed: {proc.stderr}"
    commit = proc.stdout.strip()
    assert commit, "phase-23.2.24 commit not found in local history"

    # Check the commit is reachable from origin/main
    proc = subprocess.run(
        ["git", "branch", "-r", "--contains", commit],
        cwd=ROOT, capture_output=True, text=True, timeout=20,
    )
    assert proc.returncode == 0, f"git branch -r failed: {proc.stderr}"
    branches = proc.stdout
    assert "origin/main" in branches, (
        f"phase-23.2.24 commit {commit} not on origin/main; "
        f"branches containing it: {branches.strip() or '(none)'}"
    )
    return f"OK origin/main has commit {commit[:8]}"


def check_smoke_script():
    rel = "scripts/qa/verify_qa_roster_live.sh"
    p = ROOT / rel
    assert p.exists(), f"smoke script missing: {rel}"
    # Executable bit
    mode = p.stat().st_mode
    assert mode & 0o111, f"smoke script not executable: mode={oct(mode)}"
    text = p.read_text()
    assert "1b. Frontend lint" in text, \
        "smoke script must reference the section header"
    assert "Self-disclosure" in text, \
        "smoke script must embed the operator self-disclosure prompt"
    assert "verify_qa_roster_live" in text or "phase-23.3.0" in text, \
        "smoke script must self-identify"
    # bash -n syntax check
    proc = subprocess.run(
        ["bash", "-n", str(p)],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0, f"bash -n failed: {proc.stderr}"
    return f"OK {rel}"


def check_claude_md_cross_reference():
    rel = "CLAUDE.md"
    text = _read(rel)
    assert "verify_qa_roster_live.sh" in text, \
        "CLAUDE.md must cross-reference the smoke script"
    assert "Retry-on-FAIL" in text or "retry-on-FAIL" in text, \
        "CLAUDE.md must cross-reference the per-step-protocol retry section"
    return f"OK {rel}"


def main() -> int:
    checks = [
        check_qa_md_section_and_commands,
        check_phase_23_2_24_commit_on_origin_main,
        check_smoke_script,
        check_claude_md_cross_reference,
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
    print(f"\nphase-23.3.0 verification: ALL PASS ({len(checks)}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
