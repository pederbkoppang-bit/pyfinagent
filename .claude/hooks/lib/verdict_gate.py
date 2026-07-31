#!/usr/bin/env python3
"""Machine-readable verdict gate for `.claude/hooks/auto-commit-and-push.sh`.

phase-71.3 -- the Q/A now emits a machine-readable verdict object; Main
persists it to `handoff/current/evaluator_critique.json`
({step_id, cycle_num, ok, verdict PASS|CONDITIONAL|FAIL, violated_criteria,
checks_run, ...}). When a masterplan step is flipped to `done`, the hook
calls this helper so the status-flip gate reads the VERDICT (JSON), not prose.

Decision (printed to stdout, one of):
  no_input -- the evaluator JSON does not exist at all. FAIL-OPEN like
              'proceed' (the hook continues), but distinguishable in the log
              so that "no gate ever ran" stops looking identical to
              "the gate ran and passed". See phase-81.0 below.
  proceed  -- the file exists but cannot gate: unreadable, not a dict, its
              step_id does not match the flipped step, or it carries no
              verdict field. Hook continues as today (FAIL-OPEN).
  passed   -- evaluator_critique.json matches this step AND verdict=="PASS"
              AND ok is true. Hook logs INFO and proceeds.
  hold     -- evaluator_critique.json matches this step AND the verdict is
              explicitly NOT a clean PASS (CONDITIONAL/FAIL or ok false).
              Hook logs WARN and holds the push (exit 0; never blocks the
              masterplan Write).

phase-81.0 (2026-07-31) -- WHY 'no_input' EXISTS. This gate fired correctly
24 times through 2026-07-20 and then went dark for 13 consecutive step
closes with no signal, because a missing input and a clean PASS were both
silent. Splitting the file-absent case out is the whole fix: the caller
logs a distinct WARN so a cold session can grep auto-push.log and tell
"a gate checked this step" from "nothing checked this step".

CONSUMER-CONTRACT NOTE: this widens the return set from three tokens to
four. The shell caller dispatches with `case ... in proceed|*)`, so an
unhandled token would fail open SILENTLY -- which is the very defect this
change exists to remove. auto-commit-and-push.sh therefore MUST carry a
`no_input)` arm BEFORE its catch-all; the two changes ship together.
Pending step 75.5.9 (claims_ledger.py, "contract identical to
verdict_gate.py") INHERITS this 4-token contract -- operator decision
recorded 2026-07-31; no immutable criterion was edited.

Like live_check_gate.py, this helper NEVER raises -- any argument / parse /
I/O error fails open to "proceed", matching the surrounding hook's
discipline of never breaking the masterplan Write that triggered it. It only
ever HOLDS on an explicit, step-matched, non-PASS verdict -- so steps that
predate the JSON emission (or emit a PASS) are unaffected.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def gate_decision(evaluator_json_path: str, step_id: str) -> str:
    """Return one of: 'no_input', 'proceed', 'passed', 'hold'. Never raises."""
    try:
        p = Path(evaluator_json_path)
        if not p.exists():
            # phase-81.0: distinct from 'proceed'. Still fail-open at the
            # caller, but no longer indistinguishable from a verified PASS.
            return "no_input"
        data: Any = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return "proceed"
    if not isinstance(data, dict):
        return "proceed"
    # Only gate when the JSON is unambiguously about THIS step. A missing or
    # mismatched step_id fails open (do not block on a stale/ambiguous file).
    json_step = str(data.get("step_id", "")).strip()
    if json_step and json_step != str(step_id).strip():
        return "proceed"
    verdict = data.get("verdict")
    if verdict is None:
        return "proceed"
    ok = data.get("ok", True)
    if str(verdict).strip().upper() == "PASS" and bool(ok):
        return "passed"
    return "hold"


# --------------------------------------------------------------------------
# phase-81.2: ORDER-INDEPENDENT RESOLUTION.
#
# Until 81.2 this helper resolved exactly ONE literal path, so anything that
# moved that file disarmed the gate -- silently, because a miss fails open.
# That is not hypothetical: STEP_ID_RE in scripts/housekeeping/*.py matches 0
# of the 127 .md files in handoff/current/, so backfill_handoff_archive.py
# classifies every step artifact as unclassifiable and moves it to
# handoff/archive/misc/. Commit fa9aaf8e (2026-07-24) swept 315 files that way
# and took handoff/current/evaluator_critique.json with them; the gate then ran
# dark for 13 consecutive step closes with no signal.
#
# The durable fix is NOT to argue about where the file belongs -- archive/misc
# grew +60 files in the six days to 2026-07-31, so the sweep is ongoing. It is
# to stop depending on one location. We search an ordered chain and REPORT WHICH
# SOURCE ANSWERED, so "gated on a fresh verdict in current/" and "gated on an
# archived verdict" are distinguishable in the log instead of both reading as a
# clean pass.
#
# This function is READ-ONLY by construction: it only ever calls .exists() and
# .read_text(). It must never create, move, or delete anything -- a gate that
# mutates the tree it inspects is how the last defect was introduced.
# --------------------------------------------------------------------------

def resolve_verdict_source(step_id: str, handoff_root: str):
    """Return (Path|None, source_label) for the first location that has a file.

    Order is deliberate and is asserted by the tests: the freshest, most
    specific location wins, and the archive is consulted only as a last resort.
    """
    sid = str(step_id).strip()
    root = Path(handoff_root)
    candidates = [
        (root / "current" / f"evaluator_critique_{sid}.json", "current:per-step"),
        (root / "current" / "evaluator_critique.json", "current:rolling"),
        (root / "archive" / f"phase-{sid}" / f"evaluator_critique_{sid}.json", "archive:step"),
        (root / "archive" / f"phase-{sid}" / "evaluator_critique.json", "archive:step-rolling"),
        (root / "archive" / "misc" / f"evaluator_critique_{sid}.json", "archive:misc"),
    ]
    for path, label in candidates:
        try:
            if path.exists():
                return path, label
        except Exception:
            # A single unreadable candidate must not abort the whole chain --
            # keep looking rather than failing open early.
            continue
    return None, "none"


def gate_decision_with_source(step_id: str, handoff_root: str):
    """Resolve the input, then gate on it. Returns (decision, source_label)."""
    path, label = resolve_verdict_source(step_id, handoff_root)
    if path is None:
        return "no_input", label
    return gate_decision(str(path), step_id), label


def main() -> int:
    # Legacy 2-arg mode: byte-identical behaviour to phase-71.3. Preserved so
    # every existing caller and the phase-71.3 assertions keep working -- the
    # resolution capability is strictly ADDITIVE.
    if len(sys.argv) == 3:
        evaluator_json_path, step_id = sys.argv[1:3]
        print(gate_decision(evaluator_json_path, step_id))
        return 0
    # phase-81.2 resolution mode: <step_id> <handoff_root>.
    # Line 1 is the decision token (so a shell `case` reading the first line is
    # unchanged); line 2 is the source label for logging.
    if len(sys.argv) == 4 and sys.argv[1] == "--resolve":
        decision, source = gate_decision_with_source(sys.argv[2], sys.argv[3])
        print(decision)
        print(source)
        return 0
    # Wrong arg count -> fail-open. Bash callers expect a single token.
    print("proceed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
