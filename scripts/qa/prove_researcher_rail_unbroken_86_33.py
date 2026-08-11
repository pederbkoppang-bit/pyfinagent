#!/usr/bin/env python3
"""phase-86.33 criterion 3 -- the researcher rail still writes, for EVERY spelling.

The criterion: *"drive the guard with the real researcher identities from the log
(researcher, research-*, res-*) and show every one still writes."*

WHY THIS IS DRIVEN AND NOT REASONED
-----------------------------------
The researcher rail is the mechanism that preserved a 76KB brief and three Q/A
evaluations from rail drops on 2026-08-11. A fail-closed mistake in this guard
breaks write-first, and write-first is the only reason a dropped agent leaves
anything behind. So this is checked by running the real hook against every identity
the log has ever seen, not by reading the match rule and agreeing with it.

THE POPULATION IS DERIVED, WITH ITS RULE STATED
-----------------------------------------------
Spellings come from `handoff/logs/qa_write_guard.log`, not from a hand-typed list.
Two rules are reported because they disagree and a rate without its rule is how
three measurements went wrong in this step already:

    startswith("research")            -> the narrower set
    startswith(("research", "res-"))  -> adds the res-NN-N abbreviations

The WIDER rule is used for the assertion. If a spelling is a researcher and the
guard blocks it, that is a broken rail regardless of which prefix convention it
happens to follow.

ANTI-VACUITY
------------
A test that only drives ALLOW cases cannot fail if the guard is replaced by
`exit 0`. This script therefore also drives a control that MUST be blocked, and
refuses to report success unless BOTH directions behaved.

Run:  python scripts/qa/prove_researcher_rail_unbroken_86_33.py
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
GUARD = REPO / ".claude" / "hooks" / "qa-write-guard.sh"
LOG = REPO / "handoff" / "logs" / "qa_write_guard.log"

# A path the researcher legitimately writes: its own brief. Write-first depends on
# exactly this working.
RESEARCHER_TARGET = "handoff/current/research_brief_86.33.md"

# The control. A qa-role identity writing outside its verdict dir MUST be blocked;
# if this comes back ALLOW the guard is not enforcing anything and every ALLOW
# above it is meaningless.
CONTROL = ("qa", "backend/main.py")


def derive_spellings() -> tuple[list[str], list[str]]:
    seen: set[str] = set()
    for line in LOG.read_text(errors="replace").splitlines():
        i = line.find("{")
        if i < 0:
            continue
        try:
            seen.add(json.loads(line[i:]).get("agent_type") or "")
        except Exception:  # noqa: BLE001
            continue
    narrow = sorted(t for t in seen if t.startswith("research"))
    wide = sorted(t for t in seen if t.startswith(("research", "res-")))
    return narrow, wide


def drive(agent_type: str, file_path: str) -> int:
    payload = {
        "agent_type": agent_type,
        "tool_name": "Write",
        "tool_input": {"file_path": file_path},
        "hook_event_name": "PreToolUse",
    }
    p = subprocess.run(
        ["bash", str(GUARD)], input=json.dumps(payload),
        capture_output=True, text=True, cwd=REPO,
    )
    return p.returncode


def main() -> int:
    if not GUARD.exists():
        print(f"  GUARD NOT FOUND: {GUARD}")
        return 2
    narrow, wide = derive_spellings()
    print("=" * 82)
    print("phase-86.33 criterion 3 -- researcher rail, every spelling, driven")
    print("=" * 82)
    print(f"\n  population rule startswith('research')          -> {len(narrow)} spellings")
    print(f"  population rule startswith('research'|'res-')  -> {len(wide)} spellings  [USED]")
    if not wide:
        print("\n  ABORT: zero spellings derived -- the log is empty or the parser is broken.")
        print("  A pass over an empty population proves nothing.")
        return 1

    print(f"\n  driving {len(wide)} identities against {RESEARCHER_TARGET}\n")
    blocked = []
    for t in wide:
        rc = drive(t, RESEARCHER_TARGET)
        mark = "ALLOW" if rc == 0 else f"BLOCK(rc={rc})"
        if rc != 0:
            blocked.append(t)
        print(f"    {mark:<12} {t}")

    ctrl_rc = drive(*CONTROL)
    print(f"\n  CONTROL  {CONTROL[0]} -> {CONTROL[1]}: "
          f"{'BLOCK' if ctrl_rc != 0 else 'ALLOW'} (rc={ctrl_rc})")

    print("\n" + "=" * 82)
    ok = True
    if blocked:
        ok = False
        print(f"  FAIL: {len(blocked)} researcher spelling(s) BLOCKED -- write-first is broken:")
        for t in blocked:
            print(f"    {t}")
    if ctrl_rc == 0:
        ok = False
        print("  FAIL: the control was ALLOWED. The guard is not enforcing, so every")
        print("        ALLOW above is vacuous and this run proves nothing.")
    if ok:
        print(f"  OK -- all {len(wide)} researcher spellings still write, and the control")
        print("       is still blocked, so both directions are exercised.")
    print("=" * 82)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
