#!/usr/bin/env python3
"""phase-86.31 criterion 3 -- what a DROPPED Q/A leaves behind, demonstrated.

THE FAULT BEING SIMULATED, stated precisely so the simulation can be judged
faithful or not. On the Workflow rail a drop presents as
`agent({schema}): subagent completed without calling StructuredOutput`. The
subagent's turns HAPPENED -- tool calls ran, findings were established -- and
then no structured return was emitted, so the orchestrator gets nothing. From
the artifact's point of view that is indistinguishable from the process being
killed between two writes with no chance to run a finalizer.

So the simulation is: a child process runs the write-first pattern (create
INCOMPLETE, append findings incrementally, flip to COMPLETE as its final act)
and is **SIGKILLed** partway through. SIGKILL, not SIGTERM, on purpose: nothing
gets to flush, close, or tidy up, which is the whole point.

WHAT THIS SIMULATION DOES *NOT* CLAIM. It does not reproduce the LLM-side cause
of the drop -- that cause is unknown and measured NOT to be a token threshold
(see the run table: a dropped run at 174,664 tokens sits BELOW a completed one
at 176,900). It reproduces the CONSEQUENCE for the artifact, which is the only
thing criterion 3 asks about.

    $ python scripts/qa/simulate_qa_drop_86_31.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("qa_wip_sim", HERE / "qa_wip.py")
qa_wip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qa_wip)

STEP = "99.99"

#: What a real Q/A establishes, in the order it establishes it. Modelled on
#: wf_e03ec2d0-c07, the run that died holding a genuine surviving mutant.
FINDINGS = [
    "## A. harness compliance",
    "- research_brief present, gate_passed=true, 12 sources  -> OK",
    "- contract mtime AFTER brief mtime                      -> OK",
    "## B. deterministic",
    "- immutable verification command: exit=0",
    "- ruff F821,F401,F811 over 4 changed .py files: exit=0",
    "## C. mutation matrix",
    "- M1 KILLED  - M2 KILLED  - M3 KILLED",
    "- M4 SURVIVED  <-- REAL FINDING: the driver drifts to false",   # the payload
    "## D. criteria",
    "- criterion 1 MET (guard decisions captured verbatim)",
    "- criterion 2 MET (no deny removed; mechanism named)",
]

CHILD = r'''
import pathlib, sys, time
path = pathlib.Path(sys.argv[1]); incomplete = sys.argv[2]; complete = sys.argv[3]
written_at = sys.argv[4]; lines = sys.argv[5:]
path.parent.mkdir(parents=True, exist_ok=True)
# born inert AND identified: the marker and the WRITTEN stamp are both in the
# FIRST write. The stamp is what lets a later reader tell this artifact from a
# previous cycle's at the same fixed path (phase-86.31 cycle 2).
path.write_text(incomplete + "\nSTEP: 99.99\nWRITTEN: " + sys.argv[4] + "\n\n", encoding="utf-8")
for line in lines:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
    time.sleep(0.05)
# the final act -- only reached by a run that is NOT dropped
body = path.read_text(encoding="utf-8")
path.write_text(body.replace(incomplete, complete, 1), encoding="utf-8")
print("CHILD COMPLETED", flush=True)
'''


def run(sandbox: pathlib.Path, kill_after: float | None, written_at: str = "2026-08-10T12:00:00Z"):
    target = qa_wip.resolve_wip_path(STEP, repo=sandbox)
    if target.exists():
        target.unlink()
    proc = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(target),
         qa_wip.INCOMPLETE_MARKER, qa_wip.COMPLETE_MARKER, written_at, *FINDINGS],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    killed = False
    if kill_after is not None:
        time.sleep(kill_after)
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGKILL)   # no finalizer runs. that is the point.
            killed = True
    out, _err = proc.communicate(timeout=60)
    return proc.returncode, killed, out.strip(), target


def main() -> int:
    sandbox = pathlib.Path(tempfile.mkdtemp(prefix="qa_drop_86_31_"))
    failures = []
    try:
        print("=" * 78)
        print("RUN 1 -- CONTROL: the run is allowed to finish")
        print("=" * 78)
        rc, killed, out, target = run(sandbox, kill_after=None)
        rep = qa_wip.report(STEP, repo=sandbox)
        print(f"  child exit={rc} killed={killed} stdout={out!r}")
        print(f"  recovered: status={rep['status']} bytes={rep['bytes']} "
              f"recoverable={rep['recoverable']} is_verdict={rep['is_verdict']}")
        if rep["status"] != "COMPLETE":
            failures.append("CONTROL did not reach COMPLETE -- the harness is broken")
        complete_lines = target.read_text(encoding="utf-8").count("\n")

        print("\n" + "=" * 78)
        print("RUN 2..N -- DROPPED: SIGKILL mid-stream, no finalizer runs")
        print("=" * 78)
        rows = []
        for kill_after in (0.10, 0.25, 0.40, 0.55):
            rc, killed, out, target = run(sandbox, kill_after=kill_after)
            rep = qa_wip.report(STEP, repo=sandbox)
            body = target.read_text(encoding="utf-8") if target.exists() else ""
            kept = [l for l in FINDINGS if l in body]
            lost = [l for l in FINDINGS if l not in body]
            rows.append((kill_after, rep["status"], len(kept), len(lost), rep["bytes"]))
            print(f"\n  kill at t+{kill_after:.2f}s: child rc={rc} sigkilled={killed} "
                  f"stdout={out!r}")
            print(f"    marker      : {rep['status']}   (NEVER 'COMPLETE' -- the final act never ran)")
            print(f"    RECOVERED   : {len(kept)}/{len(FINDINGS)} findings, {rep['bytes']} bytes")
            print(f"    LOST        : {len(lost)}/{len(FINDINGS)} findings")
            if "- M4 SURVIVED" in " ".join(kept):
                print("    ** the surviving-mutant finding SURVIVED the drop **")
            if rep["status"] == "COMPLETE":
                failures.append(f"a SIGKILLed run reported COMPLETE at t+{kill_after}")
            if rep["is_verdict"] or "verdict" in rep:
                failures.append("the recovery report exposed something scrapable as a verdict")

        print("\n" + "=" * 78)
        print("WHAT IS RECOVERABLE VERSUS LOST")
        print("=" * 78)
        print(f"  {'kill at':>9}  {'marker':<11} {'recovered':>9} {'lost':>5} {'bytes':>7}")
        for ka, st, k, l, b in rows:
            print(f"  t+{ka:.2f}s   {st:<11} {k:>6}/{len(FINDINGS)} {l:>5} {b:>7}")
        print(f"  (a COMPLETE run writes {complete_lines} lines and flips the marker)")
        print("\n  RECOVERABLE: every finding written before the kill, verbatim.")
        print("  LOST       : findings not yet written, and the structured return object")
        print("               -- which is the ONLY thing that can ever be a verdict.")
        print("\n  BEFORE phase-86.31 the recoverable column was 0, and that is MEASURED,")
        print("  not assumed. Nothing directed the Q/A to write a WIP at all; and the one")
        print("  time a spawn prompt did (step 82.10 cycle 1), the evaluator reported in")
        print("  its own return -- handoff/current/qa_returns/82.10_cycle1.output.json,")
        print("  key `notes` -- that BOTH the directed path and a scratchpad fallback were")
        print("  blocked by qa-write-guard, concluding verbatim: \"this structured return")
        print("  is the ONLY copy of the verdict -- if the rail drops it, nothing is")
        print("  recoverable, which is exactly the risk the directive was written to")
        print("  remove.\" phase-86.31 aims the directive at a path the guard ALREADY")
        print("  permits, which is why no allowlist was added and no deny removed.")

        print("\n" + "=" * 78)
        print("AND IT IS STILL NOT A VERDICT")
        print("=" * 78)
        print(json.dumps(qa_wip.report(STEP, repo=sandbox), indent=2, sort_keys=True))
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nALL GREEN -- a dropped run leaves recoverable INCOMPLETE evidence; "
          "a completed run leaves COMPLETE; neither is a verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
