"""phase-23.2.24: immutable verification.

Asserts:
1. frontend/src/app/cron/page.tsx::JobsTab calls useMemo BEFORE any
   conditional return (Rules-of-Hooks compliant).
2. cd frontend && npx eslint . exits 0 (errors-only). The hook-order
   bug class is set to `"error"` severity so any future violation
   fails this step. Pre-existing WARNINGS are tracked separately as
   a phase-2 deferral and do not block this gate.
3. cd frontend && npx tsc --noEmit exits 0.
4. .claude/agents/qa.md has the new "1b. Frontend lint + typecheck"
   section with the literal commands.
5. docs/runbooks/per-step-protocol.md has the formalised
   Retry-on-FAIL loop section with file-based-handoff semantics.
6. frontend/package.json lint script is `eslint .` (errors-only,
   per criterion 6 of the contract).
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def check_jobs_tab_hook_order():
    rel = "frontend/src/app/cron/page.tsx"
    text = _read(rel)
    # Slice the JobsTab function body so we can do an order check.
    fn_start = text.find("function JobsTab")
    fn_end = text.find("\nfunction ", fn_start + 1)
    if fn_end == -1:
        fn_end = len(text)
    body = text[fn_start:fn_end]
    # Find positions of every hook call and every early return inside
    # the JobsTab body. useMemo MUST appear before any `return` statement
    # at the function-top level.
    use_memo_pos = body.find("useMemo(")
    assert use_memo_pos > 0, "useMemo missing from JobsTab"
    # The first early return after the hooks block. We search for "  return ("
    # at indent depth 2 (function-top-level returns).
    early_return_match = re.search(r"\n  if \([^)]*\) \{\n    return", body)
    assert early_return_match, "expected an `if (...) { return` early-return pattern"
    early_return_pos = early_return_match.start()
    assert use_memo_pos < early_return_pos, (
        f"useMemo at byte {use_memo_pos} appears AFTER the first early return at "
        f"byte {early_return_pos} -- Rules-of-Hooks violation. useMemo MUST be "
        f"called on every render, before any conditional return."
    )
    return f"OK {rel} -- useMemo before early returns"


def check_eslint_exits_zero():
    """Run eslint . and assert exit code 0 (errors-only)."""
    proc = subprocess.run(
        ["npx", "eslint", "."],
        cwd=FRONTEND,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        stdout_tail = "\n".join(proc.stdout.strip().splitlines()[-15:])
        raise AssertionError(
            f"eslint exited {proc.returncode}\nstdout tail:\n{stdout_tail}\n"
            f"stderr tail:\n{stderr_tail}"
        )
    return "OK frontend npx eslint ."


def check_tsc_exits_zero():
    """Run tsc --noEmit and assert exit code 0."""
    proc = subprocess.run(
        ["npx", "tsc", "--noEmit"],
        cwd=FRONTEND,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        out = (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-15:]
        raise AssertionError(f"tsc exited {proc.returncode}:\n" + "\n".join(out))
    return "OK frontend npx tsc --noEmit"


def check_qa_md_section():
    rel = ".claude/agents/qa.md"
    text = _read(rel)
    assert "### 1b. Frontend lint" in text, \
        "qa.md missing `### 1b. Frontend lint` section header"
    assert "react-hooks/rules-of-hooks" in text, \
        "qa.md must explain the canonical guard rule"
    assert "tsc --noEmit` does NOT catch" in text or \
           "tsc --noEmit does NOT catch" in text, \
        "qa.md must state TypeScript does NOT catch hook-order violations"
    assert "npx eslint ." in text, "qa.md must include the literal eslint command"
    assert "frontend/**" in text, \
        "qa.md must say the section is required when diff touches `frontend/**`"
    return f"OK {rel}"


def check_runbook_retry_section():
    rel = "docs/runbooks/per-step-protocol.md"
    text = _read(rel)
    assert "Retry-on-FAIL loop" in text, \
        "per-step-protocol.md missing `Retry-on-FAIL loop` section"
    assert "fresh Q/A" in text, "must specify fresh Q/A respawn (not SendMessage)"
    assert "max 3 retries" in text or "max-3" in text or "3 retries" in text, \
        "must specify the max-3 retry ceiling"
    assert "second-opinion-shop" in text, \
        "must distinguish legitimate retry from second-opinion-shopping"
    return f"OK {rel}"


def check_package_json_lint_script():
    rel = "frontend/package.json"
    pkg = json.loads(_read(rel))
    lint_script = pkg.get("scripts", {}).get("lint", "")
    assert lint_script == "eslint .", \
        f"package.json lint script should be 'eslint .' (errors-only); got {lint_script!r}"
    return f"OK {rel}"


def main() -> int:
    checks = [
        check_jobs_tab_hook_order,
        check_qa_md_section,
        check_runbook_retry_section,
        check_package_json_lint_script,
        check_eslint_exits_zero,   # slowest: ~10-15s
        check_tsc_exits_zero,      # slowest: ~15-25s
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
    print(f"\nphase-23.2.24 verification: ALL PASS ({len(checks)}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
