#!/usr/bin/env python3
"""phase-86.33 criterion 1 -- derive the agent_type population from the guard log.

RE-RUNNABLE AND COMMITTED, because the criterion says so: the population must be
derived by a script, not transcribed from the step text.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
-----------------------------------------
It does NOT claim to separate real subagent writes from synthetic ones, because
`handoff/logs/qa_write_guard.log` cannot support that split -- see the comment block
below `P0_UTC`. A test harness driving the hook fabricates every field the log
records, including `agent_type`, so a synthetic row and a real one are
indistinguishable in it.

That is not a limitation to work around; it is the step's central finding arriving
one level up. `agent_type` cannot carry authorization because the caller chooses it,
and for exactly the same reason this log cannot certify its own population.

THREE MEASUREMENTS WENT WRONG HERE BEFORE THIS SCRIPT SETTLED
-------------------------------------------------------------
  * Main, "24 of 68 qa rows lack agent_id"     -- counted harness payloads as real
  * Main, "23 of 33 role-typed rows (70%)"     -- same filter, same artifact
  * this script's own first revision, "85%"    -- a hand-written path list whose
    docstring claimed it was "derived by reading the prover, not guessed". THAT
    CLAIM WAS FALSE and the list was incomplete.

Every one of them quoted a rate whose population rule was unsound. The script now
reports coverage UNFILTERED and names what it cannot establish.

Run:  python scripts/qa/derive_agent_type_population_86_33.py
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
LOG = REPO / "handoff" / "logs" / "qa_write_guard.log"

# The 86.33 P0 commit (8a9a4293) that added agent_id logging, in UTC. Rows before
# it CANNOT carry the field, so including them measures the deploy, not the runtime.
P0_UTC = "2026-08-11T07:55:14"

# THE SYNTHETIC/REAL SPLIT CANNOT BE DERIVED FROM THIS LOG, AND THAT IS A FINDING.
#
# An earlier revision of this script carried a hand-written PROVER_TARGETS list and
# a docstring claiming it was "derived by reading the prover, not guessed". THAT
# CLAIM WAS FALSE -- I wrote the list from memory of an earlier console output. It
# was also incomplete, which is why this script reported 85% where the research gate
# measured 100%.
#
# Deriving the list properly makes the problem worse, not better. The prover's CASES
# table (extracted from source) targets 8 paths:
#
#   .claude/agent-memory/qa/MEMORY.md          backend/main.py
#   .claude/agent-memory/qa/verdicts/...       backend/services/kill_switch.py
#   handoff/current/evaluator_critique.md      handoff/current/experiment_results_86.34.md
#   handoff/current/research_brief_86.38.md    .../qa/../../../backend/services/kill_switch.py
#
# -- and NONE of them is `/tmp/evil.md` or `../../../etc/x`, which also appear in the
# log. Those come from a DIFFERENT probe (`/T/qa_guard_probe_*/g.sh`, 2026-08-10).
# Several prover targets are also paths REAL agents legitimately write.
#
# So: a synthetic row and a real row are indistinguishable in this log, because a
# harness fabricates every field it contains. **That is the same defect as
# `agent_type` being caller-chosen, one level up** -- and it means any coverage
# figure quoted from this log is conditional on a population rule that cannot be
# soundly derived from the log itself.
#
# This script therefore reports agent_id coverage WITHOUT asserting a clean split,
# and names the limitation rather than hiding it behind a curated list.

def load_rows() -> list[dict]:
    rows = []
    for line in LOG.read_text(errors="replace").splitlines():
        i = line.find("{")
        if i < 0:
            continue
        try:
            rows.append(json.loads(line[i:]))
        except Exception:  # noqa: BLE001 -- a malformed line is not a row
            continue
    return rows


def coverage_by_agent_id(rows: list[dict]) -> tuple[int, int]:
    """agent_id coverage, reported WITHOUT a synthetic/real split.

    Deliberately not filtered: see the comment block above. Any filter would rest
    on a population rule this log cannot support, and inventing one is how the two
    earlier mis-measurements happened.
    """
    with_id = sum(1 for r in rows if r.get("agent_id"))
    return with_id, len(rows)


def main() -> int:
    if not LOG.exists():
        print(f"  LOG NOT FOUND: {LOG}")
        return 2
    rows = load_rows()
    print("=" * 78)
    print("phase-86.33 criterion 1 -- agent_type population, DERIVED")
    print("=" * 78)
    print(f"\nsource: {LOG.relative_to(REPO)}   rows parsed: {len(rows)}")

    # ── the full distinct set, all time ──────────────────────────────────
    print(f"\n{'ALL-TIME distinct agent_type values':-<78}")
    print(f"  {'agent_type':<30} {'rows':>6}")
    for k, v in collections.Counter(r.get("agent_type") or "(empty)" for r in rows).most_common():
        print(f"  {k!r:<30} {v:>6}")

    # ── the P0 split, with prover rows separated ─────────────────────────
    post = [r for r in rows if (r.get("ts") or "") >= P0_UTC]
    sub = [r for r in post if (r.get("agent_type") or "")]
    main_rows = [r for r in post if not (r.get("agent_type") or "")]
    s_id, s_n = coverage_by_agent_id(sub)
    m_id, m_n = coverage_by_agent_id(main_rows)
    print(f"\n{'AFTER the P0 commit (agent_id exists)':-<78}")
    print(f"  rows after {P0_UTC}: {len(post)}")
    print(f"    with agent_type set (subagent-shaped) : {s_id}/{s_n}"
          + (f"  = {s_id/s_n:.0%} carry agent_id" if s_n else ""))
    print(f"    agent_type empty    (Main-shaped)     : {m_id}/{m_n}"
          + (f"  = {m_id/m_n:.0%} carry agent_id" if m_n else ""))
    print("\n  UNFILTERED ON PURPOSE. Some subagent-shaped rows are synthetic: a")
    print("  harness driving the hook sets agent_type itself. This log cannot")
    print("  distinguish them, so a 'real spawns only' rate cannot be soundly")
    print("  derived from it -- the research gate reports 63/63 under its own")
    print("  population rule; that rule is not reproducible from this file alone.")
    print("  Two earlier Main measurements (24/68 and 23/33) went wrong exactly here.")

    # ── the researcher population, WITH ITS RULE STATED ──────────────────
    print(f"\n{'RESEARCHER population -- every rate carries its rule':-<78}")
    rules = {
        "exact 'researcher'": lambda a: a == "researcher",
        "startswith('research')": lambda a: a.startswith("research"),
        "startswith('research') or 'res-'": lambda a: a.startswith(("research", "res-")),
    }
    for label, pred in rules.items():
        m = [r for r in rows if pred(r.get("agent_type") or "")]
        print(f"  {label:<38} events={len(m):<5} spellings={len(set(r.get('agent_type') for r in m))}")
    print("\n  Two derivations that differ are not a contradiction; a rate quoted")
    print("  without its rule is. Both are reported so neither can be quoted bare.")

    # ── what the guard's role-prefix match would and would not catch ─────
    print(f"\n{'What a qa-ROLE prefix match covers':-<78}")
    types = {r.get("agent_type") or "" for r in rows}
    # THE GUARD'S OWN PREDICATE, not a reimplementation of it.
    # qa-write-guard.sh:120-121 lowercases first:
    #     n = (name or "").strip().lower()
    #     return n == "qa" or n.startswith("qa-") or n.startswith("qa_")
    # An earlier revision of this script wrote startswith(("qa-","qa_","QA-","QA_"))
    # instead, which DIVERGES on mixed case: the guard MATCHES "Qa-Mixed" while that
    # version reported it as evading. Reimplementing a predicate is how a checker
    # ends up disagreeing with the thing it checks.
    def is_qa_role(name: str) -> bool:
        n = (name or "").strip().lower()
        return n == "qa" or n.startswith("qa-") or n.startswith("qa_")

    qa_like = {t for t in types if is_qa_role(t)}
    # EMPTY is a real member of this population (Main-shaped writes) and is the
    # LARGEST bucket. Dropping it silently makes the partition not add up -- an
    # earlier revision did exactly that and printed 34 + 37 against a total of 72.
    evade = sorted((t for t in types if t not in qa_like), key=lambda s: (s == "", s))
    print(f"  total distinct agent_type values : {len(types)}")
    print(f"    matched by the guard predicate : {len(qa_like)}")
    print(f"    NOT matched                    : {len(evade)}")
    print(f"    -> {len(qa_like)} + {len(evade)} = {len(qa_like) + len(evade)}"
          f"  (must equal {len(types)}; EMPTY is counted, not dropped)")
    for t in evade:
        n = sum(1 for r in rows if (r.get("agent_type") or "") == t)
        label = "(EMPTY -- Main-shaped writes)" if t == "" else repr(t)
        print(f"    {label:<32} {n:>6}")
    print("\n  Every value above is CALLER-CHOSEN -- it is whatever was passed as")
    print("  `agentType` at spawn. 'general-purpose' was this repo's own former pin")
    print("  (.claude/workflows/qa-verdict.js:207). Widening the prefix cannot fix a")
    print("  field the caller controls; it only renames the bypass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
