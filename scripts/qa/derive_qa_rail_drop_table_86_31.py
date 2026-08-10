#!/usr/bin/env python3
"""phase-86.31 criterion 6 -- the Q/A rail drop table, DERIVED not transcribed.

Walks every Workflow run record Claude Code has written for this project
(`~/.claude/projects/<slug>/*/workflows/wf_*.json`) and reports one row per
`qa-verdict` run: run id, step, outcome, tokens, tool calls, duration.

WHY IT IS DERIVED. The masterplan's audit_basis quotes three dropped runs and a
completed range. Those numbers are correct -- this script reproduces them -- but
a table retyped from prose cannot be re-checked by a future reader, and the
population ("8 spawns") was a claim about a SET whose membership rule was never
written down. Here the rule is written down and executable: every run record on
disk whose `workflowName == "qa-verdict"`, optionally filtered by step id.

WHAT THE TABLE IS FOR. It exists so nobody re-tries the falsified hypothesis.
Main's first theory was evidence VOLUME; it was killed by experiment (the
mandatory read path was compacted 2,572 -> 663 lines and the very next spawn
dropped anyway), and the populations OVERLAP -- sort by tokens and dropped and
completed runs interleave. Read the overlap line at the bottom before proposing
any token-budget fix.

CAVEAT, stated because it bounds the evidence: these records live under
`~/.claude/`, outside the repository and outside git. They are session-scoped
and will eventually be pruned. That is exactly why the derived table is pasted
verbatim into `handoff/.../live_check_86.31.md` -- the artifact outlives its
source.

    $ python scripts/qa/derive_qa_rail_drop_table_86_31.py
    $ python scripts/qa/derive_qa_rail_drop_table_86_31.py --step 86.28
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

PROJECT_SLUG = "-Users-ford--openclaw-workspace-pyfinagent"
ROOT = pathlib.Path.home() / ".claude" / "projects" / PROJECT_SLUG


def step_of(rec: dict) -> str:
    raw = rec.get("args")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return "?"
    if isinstance(raw, dict):
        return str(raw.get("step_id") or raw.get("stepId") or "?")
    return "?"


def collect(workflow_name: str, step: str | None):
    rows = []
    if not ROOT.is_dir():
        return rows
    for rec_path in sorted(ROOT.glob("*/workflows/wf_*.json")):
        try:
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("workflowName") != workflow_name:
            continue
        sid = step_of(rec)
        if step and sid != step:
            continue
        status = rec.get("status") or "?"
        err = (rec.get("error") or "").strip()
        if status == "completed":
            outcome = "COMPLETED"
        elif "without calling StructuredOutput" in err:
            outcome = "DROPPED"
        else:
            outcome = f"FAILED({status})"
        rows.append({
            "run": rec.get("runId", rec_path.stem),
            "step": sid,
            "outcome": outcome,
            "tokens": rec.get("totalTokens") or 0,
            "tools": rec.get("totalToolCalls") or 0,
            "sec": round((rec.get("durationMs") or 0) / 1000),
            "ts": (rec.get("timestamp") or "")[:19],
            "err": err[:70],
        })
    rows.sort(key=lambda r: r["ts"])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default=None, help="filter to one step id (e.g. 86.28)")
    ap.add_argument("--workflow", default="qa-verdict")
    ns = ap.parse_args()

    rows = collect(ns.workflow, ns.step)
    if not rows:
        print(f"NO RUN RECORDS for workflow={ns.workflow!r} step={ns.step!r} under {ROOT}")
        print("(these records are session-scoped and outside git -- see the module docstring)")
        return 1

    scope = f"step {ns.step}" if ns.step else "ALL steps"
    print(f"{ns.workflow} runs -- {scope}   (source: {ROOT}/*/workflows/wf_*.json)")
    print(f"{'run id':<18} {'step':<8} {'outcome':<11} {'tokens':>8} {'tools':>6} {'sec':>5}  timestamp")
    print("-" * 92)
    for r in rows:
        print(f"{r['run']:<18} {r['step']:<8} {r['outcome']:<11} {r['tokens']:>8} "
              f"{r['tools']:>6} {r['sec']:>5}  {r['ts']}")

    dropped = [r for r in rows if r["outcome"] == "DROPPED"]
    done = [r for r in rows if r["outcome"] == "COMPLETED"]
    print("-" * 92)
    print(f"total={len(rows)}  DROPPED={len(dropped)}  COMPLETED={len(done)}  "
          f"other={len(rows) - len(dropped) - len(done)}")
    if dropped and done:
        rate = 100.0 * len(dropped) / (len(dropped) + len(done))
        print(f"drop rate over DROPPED+COMPLETED: {len(dropped)}/{len(dropped) + len(done)} = {rate:.1f}%")
        dt = sorted(r["tokens"] for r in dropped)
        ct = sorted(r["tokens"] for r in done)
        print(f"dropped tokens  : min={dt[0]:,} max={dt[-1]:,}")
        print(f"completed tokens: min={ct[0]:,} max={ct[-1]:,}")
        overlap = [r for r in done if r["tokens"] > dt[0]]
        if overlap:
            worst = max(overlap, key=lambda r: r["tokens"])
            print(f"\nTHE POPULATIONS OVERLAP -- this is the falsification, read it before")
            print(f"proposing any token-budget fix:")
            print(f"  a DROPPED run at {dt[0]:,} tokens sits BELOW")
            print(f"  a COMPLETED run at {worst['tokens']:,} tokens ({worst['run']}).")
            print(f"  {len(overlap)} completed run(s) ran HOTTER than the coolest drop.")
            print("THE FAILURE IS INTERMITTENT, NOT A THRESHOLD.")
        else:
            print("\nNo overlap in THIS slice -- do not read that as a threshold; check the "
                  "unfiltered population before concluding anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
