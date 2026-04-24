#!/usr/bin/env python
"""phase-4.17.10 aggregate-gate finalizer.

Second leg of the step's verification (after `pytest scripts/go_live_drills/`):

1. Every smoke_test_4_17_*.py must exist (1..9 + 11 + 12).
2. masterplan.json has all of 4.17.1..4.17.12 EXCEPT 4.17.10 marked `done`.
3. harness_log.md tail (last 50 lines) carries no new CRITICAL / HARNESS HALT markers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ok = True

    drills_dir = REPO_ROOT / "scripts" / "go_live_drills"
    required = [f"smoke_test_4_17_{i}.py" for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12)]
    missing = [r for r in required if not (drills_dir / r).exists()]
    if missing:
        print(f"FAIL: missing drill files: {missing}")
        ok = False
    else:
        print(f"PASS: all 11 drill files present")

    # masterplan flips
    mp = json.loads((REPO_ROOT / ".claude/masterplan.json").read_text(encoding="utf-8"))
    phase = next((p for p in mp["phases"] if p["id"] == "phase-4.17"), None)
    if phase is None:
        print("FAIL: phase-4.17 missing from masterplan")
        return 1
    statuses = {s["id"]: s.get("status") for s in phase["steps"]}
    for sid in (f"4.17.{i}" for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12)):
        if statuses.get(sid) != "done":
            print(f"FAIL: step {sid} not done (status={statuses.get(sid)})")
            ok = False
    if ok:
        print("PASS: subtasks 4.17.1..4.17.9 + 4.17.11..4.17.12 all done")

    # harness_log tail critical markers check -- look for actual incident
    # HEADERS (lines starting with `## CRITICAL` or `## HARNESS HALT`), NOT
    # descriptive prose that happens to quote those words.
    log = (REPO_ROOT / "handoff/harness_log.md").read_text(encoding="utf-8")
    tail_lines = log.splitlines()[-50:]
    critical_headers = [
        l for l in tail_lines
        if l.startswith("##") and ("CRITICAL" in l.upper() or "HARNESS HALT" in l.upper())
    ]
    if critical_headers:
        print(f"FAIL: critical/halt headers in tail: {critical_headers}")
        ok = False
    else:
        print("PASS: no critical/halt headers in harness_log tail 50")

    if ok:
        print("PASS 4.17.10 aggregate gate")
        return 0
    print("FAIL 4.17.10 aggregate gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
