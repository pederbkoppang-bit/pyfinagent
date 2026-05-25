#!/usr/bin/env python3
"""harness_log gate logic for `.claude/hooks/auto-commit-and-push.sh`.

phase-38.4 (OPEN-13) -- adds a hook gate that REQUIRES `handoff/harness_log.md`
to contain an entry for the step-id being closed BEFORE the auto-commit
push fires. Closes the failure mode where Main flips a step to `done`
without first appending the cycle block (phase-34 cycle 9 retro
identified this as a recurring process slip).

Mirrors `live_check_gate.py` (phase-23.8.1 / audit R-1) exactly:
- Helper NEVER raises -- argument / parse errors fail-open to "proceed",
  consistent with the surrounding hook's discipline of never breaking
  the masterplan Write that triggered it.
- Decision printed to stdout, one of: proceed / passed / skip.

Default-OFF: the hook reads HARNESS_LOG_GATE_ENABLED env var; if not
"true" the gate returns "proceed" without checking. Operator opts in
once they're satisfied the doctrine is sound. This matches the operator-
approval criterion in masterplan 38.4.verification.

Detection: looks for the step-id in a Cycle-block header line of the form
`## Cycle N -- YYYY-MM-DD -- phase=<step_id> result=...` OR a less-strict
`phase=<step_id>` token anywhere in the tail of the log. Tail of last
~200 lines is sufficient -- avoids reading multi-MB log files.

Audit basis: research_brief Section B OPEN-13 (OPS-F7). Precedent:
live_check_gate.py fail-open pattern.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def gate_decision(harness_log_path: str, step_id: str, enabled: bool) -> str:
    """Return one of: 'proceed', 'passed', 'skip'.

    Fail-open to 'proceed' on any read / parse / I/O error, matching the
    hook's existing fail-open discipline.
    """
    if not enabled:
        return "proceed"
    if not step_id:
        return "proceed"
    try:
        path = Path(harness_log_path)
        if not path.exists():
            return "proceed"  # no log file yet -> first cycle; don't block
        # Tail-read last ~200 lines (sufficient for any recent cycle).
        text = path.read_text(encoding="utf-8")
        tail = "\n".join(text.splitlines()[-200:])
    except Exception:
        return "proceed"
    # Match `phase=<step_id>` as a whole token (avoid 38.6 matching 38.6.1).
    # Step-id must be followed by whitespace or end-of-line -- NOT by a
    # digit or dot (which would extend the id, e.g. phase=38.6.1).
    pattern = re.compile(rf"phase={re.escape(step_id)}(?=\s|$)", re.MULTILINE)
    if pattern.search(tail):
        return "passed"
    return "skip"


def main() -> int:
    # Usage: harness_log_gate.py <harness_log_path> <step_id>
    # ENV: HARNESS_LOG_GATE_ENABLED=true to actually gate (default OFF).
    if len(sys.argv) != 3:
        print("proceed")
        return 0
    harness_log_path, step_id = sys.argv[1:3]
    enabled = os.environ.get("HARNESS_LOG_GATE_ENABLED", "").lower() == "true"
    print(gate_decision(harness_log_path, step_id, enabled))
    return 0


if __name__ == "__main__":
    sys.exit(main())
