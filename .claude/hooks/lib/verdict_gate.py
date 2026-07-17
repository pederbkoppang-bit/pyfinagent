#!/usr/bin/env python3
"""Machine-readable verdict gate for `.claude/hooks/auto-commit-and-push.sh`.

phase-71.3 -- the Q/A now emits a machine-readable verdict object; Main
persists it to `handoff/current/evaluator_critique.json`
({step_id, cycle_num, ok, verdict PASS|CONDITIONAL|FAIL, violated_criteria,
checks_run, ...}). When a masterplan step is flipped to `done`, the hook
calls this helper so the status-flip gate reads the VERDICT (JSON), not prose.

Decision (printed to stdout, one of):
  proceed  -- no evaluator_critique.json, or it is unreadable, or its
              step_id does not match the flipped step, or it carries no
              verdict field. Hook continues as today (FAIL-OPEN).
  passed   -- evaluator_critique.json matches this step AND verdict=="PASS"
              AND ok is true. Hook logs INFO and proceeds.
  hold     -- evaluator_critique.json matches this step AND the verdict is
              explicitly NOT a clean PASS (CONDITIONAL/FAIL or ok false).
              Hook logs WARN and holds the push (exit 0; never blocks the
              masterplan Write).

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
    """Return one of: 'proceed', 'passed', 'hold'. Never raises."""
    try:
        p = Path(evaluator_json_path)
        if not p.exists():
            return "proceed"
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


def main() -> int:
    if len(sys.argv) != 3:
        # Wrong arg count -> fail-open. Bash callers expect a single token.
        print("proceed")
        return 0
    evaluator_json_path, step_id = sys.argv[1:3]
    print(gate_decision(evaluator_json_path, step_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
