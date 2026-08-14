#!/usr/bin/env python3
"""Measure the Layer-3 workflow rail's StructuredOutput drop rate, over time.

WHY THIS EXISTS
---------------
On 2026-08-14 the rail was losing ~1 run in 5: the runtime throws
`agent({schema}): subagent completed without calling StructuredOutput` -- the
turn ends, no schema call is emitted, tokens are spent, nothing returns. A
retry was added (commit 6b4df8f9) because the failure is STOCHASTIC (the
identical script dropped 39 times and completed 34).

A single run cannot measure a rate. This script reads the run records that
Claude Code already writes -- nothing new is instrumented, no hook is added --
so the measurement accumulates by itself as the masterplan is drained. Run it
whenever you want to know whether the harness is actually getting better.

    python3 scripts/qa/rail_drop_rate.py
    python3 scripts/qa/rail_drop_rate.py --since 2026-08-14

WHAT THE NUMBERS MEAN
---------------------
  EXHAUSTED   the run ended with no envelope after every retry. This is the
              number that matters: it is a lost run.
  RETRIED     a drop happened and a retry recovered it. Each one is a run the
              old code would have LOST. This is the fix's yield, and it is
              only visible because the retry logs each attempt.

Pre-fix runs have no retry logs at all, so RETRIED is 0 for them by
construction -- do not read that as "the old code never recovered". It could
not.

CAVEAT ON THE DENOMINATOR: these are run records on THIS machine, under the
current Claude Code projects dir. Sessions pruned or run elsewhere are absent.
Retention here began 2026-07-13, which is why no pre-Workflow (Agent-tool)
baseline exists to compare against -- see the harness notes in CLAUDE.md.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

DROP = "without calling StructuredOutput"
RETRY_LOG = "StructuredOutput DROP on attempt"
PROJECT_DIR = pathlib.Path.home() / ".claude/projects/-Users-ford--openclaw-workspace-pyfinagent"


def load(project_dir: pathlib.Path):
    """Yield one dict per workflow run record."""
    for p in project_dir.glob("*/workflows/*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(d, dict) or "runId" not in d:
            continue
        blob = json.dumps(d)
        # `logs` carries the workflow's own log() lines, so a recovered drop is
        # visible even though the run SUCCEEDED and its status is clean.
        retries = blob.count(RETRY_LOG)
        yield {
            "date": (d.get("timestamp") or "")[:10],
            "model": d.get("defaultModel"),
            "wf": d.get("workflowName"),
            "exhausted": DROP in str(d.get("error") or "") or (DROP in blob and d.get("status") == "failed"),
            "retries": retries,
            "tokens": d.get("totalTokens") or 0,
        }


def pct(n: int, d: int) -> str:
    return f"{n / d * 100:5.1f}%" if d else "    -"


def table(title: str, groups: dict, key_w: int = 24) -> None:
    print(f"\n=== {title} ===")
    print(f"  {'key':{key_w}} {'runs':>5} {'exhausted':>10} {'rate':>7} {'retried':>8}")
    for k, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        n = len(rows)
        ex = sum(1 for r in rows if r["exhausted"])
        rt = sum(r["retries"] for r in rows)
        print(f"  {str(k)[:key_w]:{key_w}} {n:5} {ex:10} {pct(ex, n):>7} {rt:8}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default=None, metavar="YYYY-MM-DD", help="only runs on/after this date")
    ap.add_argument("--project-dir", default=str(PROJECT_DIR))
    ap.add_argument("--fix-date", default="2026-08-14",
                    help="date the retry landed (commit 6b4df8f9); used for the before/after split")
    args = ap.parse_args()

    pdir = pathlib.Path(args.project_dir).expanduser()
    if not pdir.is_dir():
        print(f"ERROR: project dir not found: {pdir}", file=sys.stderr)
        return 2

    rows = [r for r in load(pdir) if r["date"]]
    if args.since:
        rows = [r for r in rows if r["date"] >= args.since]
    if not rows:
        print("No workflow run records found for that window.")
        return 1

    n = len(rows)
    ex = sum(1 for r in rows if r["exhausted"])
    rt = sum(r["retries"] for r in rows)
    span = f"{min(r['date'] for r in rows)} .. {max(r['date'] for r in rows)}"
    print(f"Layer-3 rail drop rate   {span}   ({n} runs)")
    print(f"  EXHAUSTED (lost runs) : {ex:4}  {pct(ex, n)}")
    print(f"  RETRIED   (recovered) : {rt:4}   <- runs the pre-fix code would have LOST")

    by = collections.defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)
    table("by model", by)

    by = collections.defaultdict(list)
    for r in rows:
        by[r["wf"]].append(r)
    table("by workflow", by)

    by = collections.defaultdict(list)
    for r in rows:
        by[r["date"]].append(r)
    print("\n=== by date ===")
    print(f"  {'date':12} {'runs':>5} {'exhausted':>10} {'rate':>7} {'retried':>8}")
    for k in sorted(by):
        g = by[k]
        e = sum(1 for r in g if r["exhausted"])
        print(f"  {k:12} {len(g):5} {e:10} {pct(e, len(g)):>7} {sum(r['retries'] for r in g):8}")

    pre = [r for r in rows if r["date"] < args.fix_date]
    post = [r for r in rows if r["date"] >= args.fix_date]
    print(f"\n=== BEFORE vs AFTER the retry ({args.fix_date}, commit 6b4df8f9) ===")
    for lbl, g in (("before", pre), ("on/after", post)):
        if g:
            e = sum(1 for r in g if r["exhausted"])
            print(f"  {lbl:9} runs={len(g):4}  exhausted={e:3} ({pct(e, len(g)).strip()})"
                  f"  retried={sum(r['retries'] for r in g)}")
        else:
            print(f"  {lbl:9} (no runs in window)")
    if post and len(post) < 20:
        print(f"\n  NOTE: only {len(post)} run(s) since the fix -- too few to call a rate.")
        print("  A 21.8% -> 4.8% claim needs ~20+ runs before it means anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
