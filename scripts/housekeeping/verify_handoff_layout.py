"""phase-4.16.2 handoff layout verifier.

Asserts:
1. `handoff/current/` contains only `_templates/*`, rolling files,
   and in-progress-step files (no done-step files).
2. `handoff/` root contains no `*_audit.json*` files (they belong in
   `handoff/audit/`).
3. `handoff/` root contains no `*.log` files (they belong in
   `handoff/logs/`).

Exit 0 when clean; 1 when any invariant is violated, with a
diff list printed to stdout so the caller sees exactly what to move.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HANDOFF = REPO / "handoff"
CURRENT = HANDOFF / "current"
MASTERPLAN = REPO / ".claude" / "masterplan.json"

ROLLING_KEEP = {
    "contract.md",
    "experiment_results.md",
    "evaluator_critique.md",
    "research_brief.md",
    "research.md",
    "research_plan.md",
    "harness_log.md",
}

STEP_ID_RE = re.compile(r"^(?:phase-)?([0-9]+(?:\.[0-9]+)*)[-.].*\.md$")


def _statuses() -> dict[str, str]:
    with MASTERPLAN.open() as f:
        mp = json.load(f)
    out: dict[str, str] = {}
    for p in mp.get("phases", []):
        for s in p.get("steps", []):
            sid = s.get("id")
            if sid:
                out[str(sid)] = str(s.get("status") or "pending")
    return out


def main() -> int:
    failures: list[str] = []

    if CURRENT.exists():
        statuses = _statuses()
        for p in CURRENT.iterdir():
            if p.is_dir():
                continue
            if p.name in ROLLING_KEEP or p.name.startswith("."):
                continue
            m = STEP_ID_RE.match(p.name)
            if not m:
                failures.append(
                    f"current/{p.name} has no step-id prefix; "
                    "move to handoff/archive/misc/"
                )
                continue
            sid = m.group(1)
            # Masterplan step ids are inconsistent across buckets: some are
            # bare `4.14.1`, others `phase-6.1`. Try both forms.
            status = statuses.get(sid) or statuses.get(f"phase-{sid}")
            if status == "done":
                failures.append(
                    f"current/{p.name} belongs to done step {sid}; "
                    f"move to handoff/archive/phase-{sid}/"
                )

    if HANDOFF.exists():
        for p in HANDOFF.iterdir():
            if p.is_dir():
                continue
            if p.name.endswith(".log"):
                failures.append(
                    f"handoff/{p.name} is a log; move to handoff/logs/"
                )
            if p.name.endswith("_audit.json") or p.name.endswith("_audit.jsonl"):
                failures.append(
                    f"handoff/{p.name} is audit output; move to handoff/audit/"
                )

    if failures:
        print(f"handoff layout FAIL -- {len(failures)} invariant violation(s):")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("handoff layout OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
