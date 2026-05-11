#!/usr/bin/env python3
"""live_check gate logic for `.claude/hooks/auto-commit-and-push.sh`.

phase-23.8.1 (R-1) — adds a per-step `verification.live_check` field
to `.claude/masterplan.json`. When a step has a non-empty `live_check`
value AND the auto-commit hook would push it, the hook calls this
helper to decide whether to proceed.

Decision (printed to stdout, one of):
  proceed  -- step has no live_check field, or step not found, or
              masterplan unreadable. Hook continues as today.
  passed   -- live_check set AND handoff/current/live_check_<id>.md exists.
              Hook logs INFO and proceeds.
  skip     -- live_check set AND artifact MISSING. Hook logs WARN and
              skips the push (exit 0, never blocks the masterplan Write).

The helper itself NEVER raises -- argument / parse errors fail-open to
"proceed", consistent with the surrounding hook's failure discipline
of never breaking the masterplan Write that triggered it.

Audit basis: docs/audits/dev-mas-2026-05-11/04-remediation.md R-1
Researcher findings: hook exit-code semantics in
`https://code.claude.com/docs/en/hooks` -- exit 0 with WARN is the
graceful-skip pattern for PostToolUse hooks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional


def find_step(node: Any, target_id: str) -> Optional[dict]:
    """Walk the masterplan tree and return the dict node whose id == target_id."""
    if isinstance(node, dict):
        if str(node.get("id", "")) == target_id:
            return node
        for v in node.values():
            r = find_step(v, target_id)
            if r is not None:
                return r
    elif isinstance(node, list):
        for v in node:
            r = find_step(v, target_id)
            if r is not None:
                return r
    return None


def gate_decision(masterplan_path: str, step_id: str, handoff_current_dir: str) -> str:
    """Return one of: 'proceed', 'passed', 'skip'.

    Never raises. Fail-open to 'proceed' on any read / parse / I/O error,
    matching the hook's existing fail-open discipline.
    """
    try:
        data = json.loads(Path(masterplan_path).read_text(encoding="utf-8"))
    except Exception:
        return "proceed"
    step = find_step(data, step_id)
    if not isinstance(step, dict):
        return "proceed"
    verification = step.get("verification")
    if not isinstance(verification, dict):
        return "proceed"
    live_check = verification.get("live_check")
    # Treat empty string / None / explicit false-y values as "no gate".
    if not live_check:
        return "proceed"
    artifact = Path(handoff_current_dir) / f"live_check_{step_id}.md"
    return "passed" if artifact.exists() else "skip"


def main() -> int:
    if len(sys.argv) != 4:
        # Wrong arg count -> fail-open. Bash callers expect single token on stdout.
        print("proceed")
        return 0
    masterplan_path, step_id, handoff_current_dir = sys.argv[1:4]
    print(gate_decision(masterplan_path, step_id, handoff_current_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
