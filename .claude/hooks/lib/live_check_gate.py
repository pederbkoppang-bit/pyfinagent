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


# ── phase-86.19: SCOPED, AND IT REFUSES TO GUESS ───────────────────────────
# `masterplan.json` carries FOUR duplicated ids across 1348 id-bearing nodes,
# in two distinct classes:
#
#   Class B  5.1 / 5.2 / 5.3 -- each appears once in `archived_legacy_steps[]`
#            and once in the LIVE `steps[]` of the same phase.
#   Class A  `phase-6.5` -- a PHASE at phases[13] and a STEP at
#            phases[12].steps[4]. A CROSS-TYPE collision, which no
#            archive-exclusion can fix.
#
# The old walk was depth-first first-match over `node.values()`, so JSON
# KEY-INSERTION ORDER decided the winner. `archived_legacy_steps` is serialised
# before `steps`, so the ARCHIVED twin (status pending) beat the LIVE done step.
# Nothing chose that behaviour; it fell out of serialisation order.
#
# "First match wins" is the one behaviour no standard endorses: RFC 8259 calls
# duplicate names "unpredictable" and recognises last-wins, error, or
# collect-all -- not first-wins. W3C XML makes id uniqueness document-wide.
# PEP 20: "In the face of ambiguity, refuse the temptation to guess." JSON
# Schema SHOULD raise; C# CS0121 refuses an ambiguous overload rather than
# picking one. So the remedy is not to pick better -- it is to STOP PICKING.
#
# Saltzer & Schroeder is the reason it matters here: an exclusion-shaped
# mechanism fails by PERMITTING, unnoticed. A resolver that silently returns the
# wrong node makes this gate emit a confident `proceed` about a subject nobody
# asked it about.
#
# NOTE ON SCOPE OF IMPACT, stated honestly: the defect is LATENT AND ARMED, not
# firing. No colliding id currently declares a `live_check`, so all four resolve
# to `proceed` either way today.

#: Containers whose members are LIVE steps. A positive allowlist, not a denylist
#: of archives -- a denylist has to be updated every time a new archive
#: container is invented, and the allowlist does not.
#:
#: DELIBERATELY DUPLICATED from `scripts/meta/preflight_verify_masterplan.py:90`
#: rather than imported. This module is a HOOK library on the auto-commit path
#: whose contract is to never raise; reaching outside `.claude/` for an import
#: would add a failure mode to a component whose whole job is to not have one.
#: The drift risk that duplication creates is closed by an ASSERTION in the
#: step's test that the two definitions are equal -- a guard, not a coupling.
LIVE_STEP_CONTAINERS = frozenset({"steps", "subphases"})

#: Distinct from None. None means "no such id"; AMBIGUOUS means "more than one,
#: and choosing between them would be a guess". They must not be conflated,
#: because they warrant different actions: a missing id is a caller bug, an
#: ambiguous id is a data defect.
AMBIGUOUS = "ambiguous"


def collect_steps(node: Any, target_id: str, in_live: bool = False) -> list:
    """Every LIVE step node whose id == target_id. Archives are out of scope.

    `in_live` tracks whether the current node sits inside a live-step container.
    A top-level phase is reached via the `phases` key, which is NOT a live-step
    container, so a phase can never match -- that is what fixes Class A without
    a special case for it.
    """
    found: list = []
    if isinstance(node, dict):
        if in_live and str(node.get("id", "")) == target_id:
            found.append(node)
        for k, v in node.items():
            found.extend(collect_steps(v, target_id, k in LIVE_STEP_CONTAINERS))
    elif isinstance(node, list):
        for v in node:
            found.extend(collect_steps(v, target_id, in_live))
    return found


def resolve_step(node: Any, target_id: str):
    """(step | None | AMBIGUOUS). The caller decides what ambiguity costs."""
    hits = collect_steps(node, target_id)
    if not hits:
        return None
    if len(hits) > 1:
        return AMBIGUOUS
    return hits[0]


def find_step(node: Any, target_id: str) -> Optional[dict]:
    """The LIVE step whose id == target_id, or None.

    Signature preserved so the step's immutable verification command keeps
    working. Ambiguity collapses to None here -- callers that must DISTINGUISH
    "absent" from "ambiguous" use `resolve_step`, and `gate_decision` does.
    """
    r = resolve_step(node, target_id)
    return r if isinstance(r, dict) else None


def gate_decision(masterplan_path: str, step_id: str, handoff_current_dir: str) -> str:
    """Return one of: 'proceed', 'passed', 'skip'.

    Never raises. Fail-open to 'proceed' on any read / parse / I/O error,
    matching the hook's existing fail-open discipline.
    """
    try:
        data = json.loads(Path(masterplan_path).read_text(encoding="utf-8"))
    except Exception:
        return "proceed"
    step = resolve_step(data, step_id)
    if step is AMBIGUOUS:
        # phase-86.19: HOLD. Two live nodes share this id, so any answer about
        # "the" step would be about an arbitrary one of them. This is the same
        # side as a missing artifact, deliberately: a gate that cannot identify
        # its subject must not wave the commit through. It is reachable only
        # when two LIVE nodes collide -- zero cases today (per-type uniqueness
        # is green at 1230 steps / 114 phases), so this holds nothing that
        # currently exists.
        return "skip"
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
