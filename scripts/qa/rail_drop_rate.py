#!/usr/bin/env python3
"""Measure the Layer-3 workflow rail's StructuredOutput drop rate, over time.

WHY THIS EXISTS
---------------
The runtime throws `agent({schema}): subagent completed without calling
StructuredOutput` -- the turn ends, no schema call is emitted, tokens are spent,
nothing returns. A retry was added (commit 6b4df8f9, 2026-08-14T10:15:17Z)
because the failure is STOCHASTIC: grouping runs by sha1 of the embedded
`script`, EIGHT distinct byte-identical script versions produced BOTH outcomes,
the largest dropping 17 times and completing 179.

Measured over the 565 records on disk when phase-86.81 was written, classified
from the `error` field alone: 44/565 = 7.8% overall; by model
claude-opus-5[1m] 40/351 = 11.4%, claude-fable-5 4/135 = 3.0%,
claude-opus-4-8[1m] 0/73 = 0.0%; by workflow qa-verdict 35/368 = 9.5% and
research-gate 6/74 = 8.1% -- indistinguishable, so there is no per-workflow
amplification to hunt.

A single run cannot measure a rate. This script reads the run records that
Claude Code already writes -- nothing new is instrumented, no hook is added --
so the measurement accumulates by itself as the masterplan is drained. Run it
whenever you want to know whether the harness is actually getting better.

    python3 scripts/qa/rail_drop_rate.py
    python3 scripts/qa/rail_drop_rate.py --since 2026-08-14
    python3 scripts/qa/rail_drop_rate.py --json      # machine-readable summary

WHAT THE NUMBERS MEAN
---------------------
  EXHAUSTED   the run ended with no envelope after every retry. This is the
              number that matters: it is a lost run. Classified ONLY from the
              record's `error` field -- see THE SELF-MATCH TRAP below.
  RETRIED     a recovery: a drop happened and a later attempt succeeded. Read
              ONLY from the record's `logs` array.

RETRIED IS A RECOVERY COUNT, NOT AN ATTEMPT COUNT, AND THE DIFFERENCE IS
STRUCTURAL (phase-86.81, finding I-2). Measured: `logs` is EMPTY on 44 of 44
dropped runs, while 83 non-dropped runs carry logs -- so a recovered run's retry
lines persist and an EXHAUSTED run's do not. The reader therefore cannot
distinguish "the retry never ran" from "it ran twice and exhausted" on a lost
run, and it says so in its own output rather than implying a zero it cannot see.

THE SELF-MATCH TRAP -- READ BEFORE CHANGING EITHER PREDICATE. A run record
EMBEDS the dispatched workflow SOURCE in its `script` field (measured: 52,089
chars on one research-gate run), and both workflow files quote the drop string
in comments. So ANY predicate that scans the whole record matches its own
subject matter. That is not hypothetical: it produced 38 phantom drops out of 81
and was corrected in commit f88f8190 -- but only in the `exhausted` predicate's
first disjunct. The RETRIED counter kept scanning the blob, and
`.claude/workflows/qa-verdict.js` contains the literal `StructuredOutput DROP on
attempt` exactly once -- in the log call itself -- so every future qa-verdict run
would have reported a phantom retry. Both predicates now read a named field:
`error` for drops, `logs` for retries. Neither may go back to the blob.

WHY THERE IS NO PREDICTED EFFECTIVENESS FIGURE HERE. Retry math assumes
independence, and that assumption is refuted in both directions: ReliabilityBench
(arXiv 2601.06112) measures Gemini 2.0 Flash pass^2 at 91.04% where independence
predicts 93.86%, while GPT-4o lands essentially AT independence. So any p^2/p^3
number is an UPPER BOUND on benefit, never a forecast. The only honest
effectiveness number is a measured conditional rate P(drop on attempt 2 | drop on
attempt 1), which requires real second attempts on real drops. This reader
reports what it observes and refuses to extrapolate.

CAVEAT ON THE DENOMINATOR: these are run records on THIS machine, under the
current Claude Code projects dir. Sessions pruned or run elsewhere are absent.
Retention here began 2026-07-13, which is why no pre-Workflow (Agent-tool)
baseline exists to compare against -- see the harness notes in CLAUDE.md.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import pathlib
import sys

DROP = "without calling StructuredOutput"

# One entry per retry SITE. Kept explicit rather than a generic /attempt \d+\/\d+/
# because a future non-retry log line carrying "attempt 1/3" must not be counted
# as a recovery. If a fourth retry site is added, add its literal here -- a
# missing entry is a SILENT undercount, which is why the checker asserts that
# every `(attempt ` log emitted by either workflow matches one of these.
RETRY_LOGS = (
    "StructuredOutput DROP on attempt",        # qa-verdict.js  agentRetryingDrops
    "STAGE-1 RAIL DROPPED (attempt",           # research-gate.js stage 1
    "STAGE-2 brief-verify failed (attempt",    # research-gate.js stage 2
)

# Commit 6b4df8f9, the instant the retry landed. UTC -- `git log` prints +0200
# locally and comparing that against a UTC run timestamp is a two-hour error that
# already inverted this analysis once (phase-86.81).
FIX_INSTANT = "2026-08-14T10:15:17Z"

PROJECT_DIR = pathlib.Path.home() / ".claude/projects/-Users-ford--openclaw-workspace-pyfinagent"


def _iso(epoch_ms: int) -> str:
    return datetime.datetime.fromtimestamp(
        epoch_ms / 1000, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def launched_at(d: dict) -> str:
    """The instant the run STARTED.

    A run is governed by the code present when it LAUNCHED, so `startTime` is the
    only correct split key. `timestamp` is the COMPLETION instant -- reading it as
    the launch instant is how phase-86.81 first concluded, wrongly, that two runs
    had ignored a fix that did not yet exist when they started. The two fields can
    even disagree on ORDER: the drop that ended later started earlier.
    """
    st = d.get("startTime")
    if isinstance(st, (int, float)) and st > 0:
        return _iso(int(st))
    return ""            # deliberately empty, never a silent fallback to `timestamp`


def load(project_dir: pathlib.Path):
    """Yield one dict per workflow run record."""
    for p in project_dir.glob("*/workflows/*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(d, dict) or "runId" not in d:
            continue
        logs = d.get("logs") or []
        # `logs` ONLY -- never the whole record. See THE SELF-MATCH TRAP above.
        retries = sum(1 for line in logs
                      if any(pat in str(line) for pat in RETRY_LOGS))
        yield {
            "date": (d.get("timestamp") or "")[:10],
            "started": launched_at(d),
            "model": d.get("defaultModel"),
            "wf": d.get("workflowName"),
            # `error` ONLY -- the blob disjunct that used to sit here was a latent
            # self-match: the DROP string occurs in the `script` field of 31
            # records, so the first non-drop failure of a workflow whose source
            # quotes it would have been silently reclassified as a drop.
            "exhausted": DROP in str(d.get("error") or ""),
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
    ap.add_argument("--fix-instant", default=FIX_INSTANT,
                    help="UTC instant the retry landed (commit 6b4df8f9); the before/after split key")
    ap.add_argument("--json", action="store_true", help="emit a machine-readable summary and exit")
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
    # The split is on the LAUNCH instant. A record with no usable `startTime` is
    # counted separately rather than being guessed into one bucket or the other.
    pre = [r for r in rows if r["started"] and r["started"] < args.fix_instant]
    post = [r for r in rows if r["started"] and r["started"] >= args.fix_instant]
    undated = [r for r in rows if not r["started"]]

    if args.json:
        print(json.dumps({
            "runs": n,
            "exhausted": ex,
            "retried": rt,
            "pre_fix_runs": len(pre),
            "post_fix_runs": len(post),
            "undated_runs": len(undated),
            "post_fix_exhausted": sum(1 for r in post if r["exhausted"]),
            "post_fix_retried": sum(r["retries"] for r in post),
            "fix_instant": args.fix_instant,
        }, sort_keys=True))
        return 0

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

    print(f"\n=== BEFORE vs AFTER the retry (launch instant {args.fix_instant}, commit 6b4df8f9) ===")
    for lbl, g in (("before", pre), ("on/after", post)):
        if g:
            e = sum(1 for r in g if r["exhausted"])
            print(f"  {lbl:9} runs={len(g):4}  exhausted={e:3} ({pct(e, len(g)).strip()})"
                  f"  retried={sum(r['retries'] for r in g)}")
        else:
            print(f"  {lbl:9} (no runs in window)")
    if undated:
        print(f"  {'undated':9} runs={len(undated):4}  (no usable startTime -- excluded from BOTH buckets,"
              f" never guessed into one)")

    if post and len(post) < 20:
        print(f"\n  NOTE: only {len(post)} run(s) have LAUNCHED since the fix -- too few to call a rate.")
        print("  Splitting on the run DATE instead would put every run from that whole day")
        print("  in this bucket, including ones that launched before the fix existed.")
    print("\n  READ RETRIED AS A RECOVERY COUNT. `logs` is empty on every dropped run,")
    print("  so the attempts burned by a run that exhausted are NOT observable here;")
    print("  retried=0 on a lost run means 'not visible', not 'did not happen'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
