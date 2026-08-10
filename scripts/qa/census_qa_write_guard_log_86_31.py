#!/usr/bin/env python3
"""phase-86.31 -- census of `handoff/logs/qa_write_guard.log` by agent_type.

This is the derivation behind the cycle-1 Q/A's principal finding: the hook's
predicate was an EXACT match on `agent_type == "qa"`, while `agent_type` in fact
carries the SPAWN NAME, so every cycle-numbered Q/A walked straight past it.

THE PROBE CONTAMINATES ITS OWN EVIDENCE, AND THAT IS HANDLED, NOT IGNORED.
`verify_qa_write_first_86_31.py` drives the real hook with SYNTHETIC identities
(`qa-80-2-c2`, `QA-80-2`, `qa_86_31`, ...) chosen precisely because they mimic
the real ones -- and the hook logs every invocation. So running the checker
INFLATES this census: measured 2026-08-10, the log went from 3,110 rows / 27
qa-* identities to 3,679 rows / 30 after a handful of checker runs, and three of
the "new" identities were the checker's own fixtures. A census taken after the
checker has run is measuring the instrument.

`--before` is therefore not a convenience. Pass the UTC timestamp at which THIS
step's tooling first ran; rows at or after it are excluded and the count of
excluded rows is reported, so a reader can see the size of the contamination
rather than having to trust that it was handled.

    $ python scripts/qa/census_qa_write_guard_log_86_31.py --before 2026-08-10T09:30:00Z
    $ python scripts/qa/census_qa_write_guard_log_86_31.py            # UNFILTERED (contaminated)
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
LOG = REPO / "handoff" / "logs" / "qa_write_guard.log"
MEMORY_DIR = ".claude/agent-memory/qa/"

#: The class of event that is the no-self-eval guarantee failing outright,
#: rather than merely a guard being wider than documented.
#:
#: CYCLE-2 CORRECTION (finding C). This was a hardcoded 2-tuple --
#: ("frontend/src/lib/api.ts", "handoff/current/evaluator_critique.md") -- and
#: the report built on it said "the breach is SIX events, not two". Both
#: numbers were counts of a hand-picked path list, not of the CLASS the
#: sentence named. The cycle-2 Q/A derived the real class and found 20 events
#: across 8 identities. Undercounting in the safe direction is still the
#: measure-don't-assert failure, so the list is now a PREDICATE over the class:
#: an evaluator writing (a) ANY evaluator_critique artifact -- the file Main is
#: contractually the verbatim scribe for -- or (b) ANY production source.
PRODUCTION_PREFIXES = ("backend/", "frontend/src/", "scripts/", "docs/")


def is_self_eval_breach(path: str) -> bool:
    """True when this write is the no-self-eval guarantee failing, not merely
    a guard being wider than its docstring."""
    p = (path or "").replace("\\", "/")
    tail = p.split("/pyfinagent/", 1)[-1]
    if "evaluator_critique" in tail:
        return True
    return any(tail.startswith(pre) for pre in PRODUCTION_PREFIXES)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default=None, metavar="ISO8601",
                    help="exclude rows at or after this UTC timestamp (see the "
                         "self-contamination note in the module docstring)")
    ap.add_argument("--log", type=pathlib.Path, default=LOG)
    ns = ap.parse_args()

    if not ns.log.exists():
        print(f"NO LOG at {ns.log}")
        return 1

    rows, excluded, unparsed = [], 0, 0
    for line in ns.log.read_text(errors="replace").splitlines():
        if not line.startswith("LOG "):
            continue
        try:
            r = json.loads(line[4:])
        except Exception:
            unparsed += 1
            continue
        if ns.before and str(r.get("ts", "")) >= ns.before:
            excluded += 1
            continue
        rows.append(r)

    print(f"log            : {ns.log}")
    print(f"cutoff         : {ns.before or 'NONE -- this census INCLUDES the checker''s own probe rows'}")
    print(f"rows counted   : {len(rows)}")
    print(f"rows excluded  : {excluded}   (at or after the cutoff)")
    print(f"rows unparsed  : {unparsed}")

    qa_like = [r for r in rows
               if str(r.get("agent_type", "")).lower().startswith("qa")
               and r.get("agent_type") != "qa"]
    names = sorted({r["agent_type"] for r in qa_like})
    print(f"\nqa-ROLE identities that are NOT exactly 'qa': {len(names)}")
    for n in names:
        print(f"    {n}")

    we = [r for r in qa_like if r.get("tool_name") in ("Write", "Edit")]
    outside = [r for r in we if MEMORY_DIR not in (r.get("file_path") or "")]
    print(f"\nWrite/Edit events from those identities            : {len(we)}")
    print(f"  ... targeting paths OUTSIDE {MEMORY_DIR}: {len(outside)}   -> ALLOWED by the old predicate")

    breaches = [r for r in outside if is_self_eval_breach(r.get("file_path", ""))]
    b_ids = sorted({r["agent_type"] for r in breaches})
    print(f"\nNO-SELF-EVAL BREACHES -- DERIVED over the class, not a hand-picked path list:")
    print(f"  {len(breaches)} events across {len(b_ids)} identities")
    for r in breaches:
        tail = r.get("file_path", "").split("/pyfinagent/", 1)[-1]
        print(f"    {r['agent_type']:16s} {r['tool_name']:5s} {tail}")
    if not breaches:
        print("    (none in this slice)")
    print("\n  The class rule, stated so it can be re-checked: an evaluator write is a")
    print("  breach when the path contains 'evaluator_critique' (Main is its verbatim")
    print(f"  scribe) or starts with any of {PRODUCTION_PREFIXES}. Counting a")
    print("  hand-picked path list instead of this rule is what produced the earlier")
    print("  'six events' figure -- accurate for the list, wrong for the class.")

    by = collections.Counter(str(r.get("agent_type", "")) for r in rows)
    print("\nDISCLOSED RESIDUAL -- NOT matched by is_qa_role(), queued as step 86.33:")
    for k in ("workflow-subagent", "general-purpose"):
        print(f"    {k:20s} {by[k]:4d} events -- indistinguishable from a legitimate researcher write")
    print("\nFor reference, the identities the qa-role predicate now covers are matched by")
    print("`agent_type` == 'qa' or lowercased-prefix 'qa-' / 'qa_'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
