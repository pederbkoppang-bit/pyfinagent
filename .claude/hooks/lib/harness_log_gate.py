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


# phase-81.0: how many trailing lines of the log to search. Measured
# 2026-07-31: handoff/harness_log.md is 29,823 lines / 2.5 MB and its last 200
# lines contain only TWO `## Cycle` headers (~84 lines per cycle block). Any
# flow that appends the log and then appends more before the status flip pushed
# the step's `phase=` token out of a 200-line window, producing a FALSE 'skip'
# -- and a 'skip' exits before `git add -A`, so it would have skipped the
# commit, the changelog AND the push. ~24 cycle blocks of headroom instead.
TAIL_LINES = 2000


def gate_decision(
    harness_log_path: str,
    step_id: str,
    enabled: bool,
    warn_only: bool = False,
) -> str:
    """Return one of: 'proceed', 'passed', 'warn', 'skip'.

    Fail-open to 'proceed' on any read / parse / I/O error, matching the
    hook's existing fail-open discipline.

    phase-81.0 -- `warn_only` is the missing middle state. Before
    81.0 this helper had no way to say "the log append is missing, and you
    should know, but I am not going to hold your push": enabling the gate at all
    jumped straight to 'skip', which blocks commit + changelog + push. That made
    the gate too risky to ever turn on, which is why it sat built, wired,
    unit-tested and permanently OFF. Warn-first lets the doctrine be observed
    for real cycles before anyone grants it teeth.

    `warn_only` defaults to **False** so that the three-positional-argument call
    `gate_decision(path, step_id, enabled=True)` keeps returning 'skip' exactly
    as masterplan step 38.4 pinned it -- that step is `done` and its immutable
    verification command runs backend/tests/test_phase_38_4_hook_gate.py. The
    new token is strictly ADDITIVE: no existing caller's behaviour changes.
    The safe default lives one layer up, in main()'s env handling, where it
    affects the shipped CLI without redefining a pinned contract.

    NOTE the gate remains DISABLED overall: HARNESS_LOG_GATE_ENABLED is not set
    anywhere in the repo or in settings.json `env`. Enabling it is gated on the
    operator approval required by masterplan step 38.4's immutable criteria and
    is deliberately NOT part of phase-81.0.
    """
    if not enabled:
        return "proceed"
    if not step_id:
        return "proceed"
    try:
        path = Path(harness_log_path)
        if not path.exists():
            return "proceed"  # no log file yet -> first cycle; don't block
        text = path.read_text(encoding="utf-8")
        tail = "\n".join(text.splitlines()[-TAIL_LINES:])
    except Exception:
        return "proceed"
    # Match `phase=<step_id>` as a whole token (avoid 38.6 matching 38.6.1).
    # Step-id must be followed by whitespace or end-of-line -- NOT by a
    # digit or dot (which would extend the id, e.g. phase=38.6.1).
    pattern = re.compile(rf"phase={re.escape(step_id)}(?=\s|$)", re.MULTILINE)
    if pattern.search(tail):
        return "passed"
    return "warn" if warn_only else "skip"


def main() -> int:
    # Usage: harness_log_gate.py <harness_log_path> <step_id>
    # ENV: HARNESS_LOG_GATE_ENABLED=true to actually gate (default OFF).
    # ENV: HARNESS_LOG_GATE_MODE=block to hold the push on a missing append.
    #      Default is "warn" -- phase-81.0 deliberately makes the SAFE mode the
    #      default, so that a future operator who sets ENABLED=true without
    #      reading this file gets an audible warning rather than a held
    #      commit+changelog+push on the first false positive.
    if len(sys.argv) != 3:
        print("proceed")
        return 0
    harness_log_path, step_id = sys.argv[1:3]
    enabled = os.environ.get("HARNESS_LOG_GATE_ENABLED", "").lower() == "true"
    warn_only = os.environ.get("HARNESS_LOG_GATE_MODE", "warn").lower() != "block"
    print(gate_decision(harness_log_path, step_id, enabled, warn_only))
    return 0


if __name__ == "__main__":
    sys.exit(main())
