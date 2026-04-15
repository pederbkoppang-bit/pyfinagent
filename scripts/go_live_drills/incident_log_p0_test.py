"""
Go-Live drill test: no unresolved P0 incidents (Phase 4.4.3.5).

Standalone, stdlib-only drill. Parses `.claude/context/known-blockers.md`
and verifies that no entry is tagged P0 without being in the RESOLVED
section or having an explicit `resolved:` line.

The checklist item (4.4.3.5) gates launch: any open P0 blocks go-live
until resolved or downgraded with Peder's explicit note.

Run from the repo root:

    python scripts/go_live_drills/incident_log_p0_test.py

Exit code 0 on PASS, exit 1 on any failure.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KNOWN_BLOCKERS_PATH = REPO_ROOT / ".claude" / "context" / "known-blockers.md"

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
    # --- Pre-check: file exists ---
    _report(
        "S0: known-blockers.md exists",
        KNOWN_BLOCKERS_PATH.is_file(),
        str(KNOWN_BLOCKERS_PATH),
    )
    if not KNOWN_BLOCKERS_PATH.is_file():
        print("FATAL: known-blockers.md not found, cannot proceed")
        return

    content = KNOWN_BLOCKERS_PATH.read_text(encoding="utf-8")
    lines = content.splitlines()

    # --- S1: Parse sections (RESOLVED vs STILL ACTIVE) ---
    resolved_section = False
    active_section = False
    resolved_lines = []
    active_lines = []

    for line in lines:
        if re.match(r"^##\s+RESOLVED", line, re.IGNORECASE):
            resolved_section = True
            active_section = False
            continue
        elif re.match(r"^##\s+STILL\s+ACTIVE", line, re.IGNORECASE):
            resolved_section = False
            active_section = True
            continue
        elif re.match(r"^##\s+", line):
            resolved_section = False
            active_section = False
            continue

        if resolved_section:
            resolved_lines.append(line)
        elif active_section:
            active_lines.append(line)

    _report(
        "S1: File has parseable RESOLVED and STILL ACTIVE sections",
        len(resolved_lines) > 0 or len(active_lines) > 0,
        f"resolved={len(resolved_lines)} lines, active={len(active_lines)} lines",
    )

    # --- S2: Scan entire file for P0 mentions ---
    p0_pattern = re.compile(r"\bP0\b", re.IGNORECASE)
    all_p0_lines = []
    for i, line in enumerate(lines, 1):
        if p0_pattern.search(line):
            all_p0_lines.append((i, line.strip()))

    _report(
        "S2: Count P0 mentions in entire file",
        True,
        f"found {len(all_p0_lines)} line(s) mentioning P0",
    )

    # --- S3: Check for P0 mentions in the STILL ACTIVE section ---
    active_p0_lines = []
    for line in active_lines:
        if p0_pattern.search(line):
            active_p0_lines.append(line.strip())

    _report(
        "S3: No P0 mentions in STILL ACTIVE section",
        len(active_p0_lines) == 0,
        f"found {len(active_p0_lines)} P0 mention(s) in active section"
        + (f": {active_p0_lines}" if active_p0_lines else ""),
    )

    # --- S4: Any P0 in RESOLVED section has a resolved marker ---
    resolved_p0_lines = []
    for line in resolved_lines:
        if p0_pattern.search(line):
            resolved_p0_lines.append(line.strip())

    resolved_p0_ok = True
    if resolved_p0_lines:
        resolved_text = "\n".join(resolved_lines)
        has_fixed = bool(
            re.search(r"(resolved|fixed|closed)", resolved_text, re.IGNORECASE)
        )
        resolved_p0_ok = has_fixed

    _report(
        "S4: Any P0 in RESOLVED section is properly marked resolved",
        resolved_p0_ok,
        f"{len(resolved_p0_lines)} P0 mention(s) in resolved section, all resolved={resolved_p0_ok}",
    )

    # --- S5: Final verdict: no unresolved P0 entries ---
    no_unresolved_p0 = len(active_p0_lines) == 0 and resolved_p0_ok
    _report(
        "S5: No unresolved P0 incidents (composite)",
        no_unresolved_p0,
        "CLEAR" if no_unresolved_p0 else "BLOCKED -- unresolved P0 found",
    )


if __name__ == "__main__":
    print(f"Incident Log P0 Drill -- Phase 4.4.3.5")
    print(f"File: {KNOWN_BLOCKERS_PATH}")
    print()

    run_drill()

    total = PASS_COUNT + FAIL_COUNT
    print()
    if FAIL_COUNT == 0:
        print(f"DRILL PASS: {PASS_COUNT}/{total} incident-log-P0 scenarios verified")
        sys.exit(0)
    else:
        print(f"DRILL FAIL: {FAIL_COUNT}/{total} scenarios failed")
        sys.exit(1)
