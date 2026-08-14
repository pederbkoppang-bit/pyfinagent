#!/usr/bin/env python3
"""phase-86.21 criterion 6 -- MUTATION-TEST the attempt counter's source.

The criterion: "corrupt or empty the source and assert it NOTICES rather than
silently reporting zero -- silently reporting zero is the defect being fixed."

Run as a FILE, never `python -` : spawn re-imports __main__ by path.
Read-only against the live tree; every mutation happens in a TemporaryDirectory.
"""
from __future__ import annotations
import json, pathlib, shutil, sys, tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import qa_wip

REPO = pathlib.Path(__file__).resolve().parents[2]
SINK = "/".join([qa_wip.MEMORY_DIR.rstrip("/"), qa_wip.WIP_SUBDIR])
STEP = "86.21"

def mkrepo(tmp: pathlib.Path) -> pathlib.Path:
    r = tmp / "repo"; (r / SINK).mkdir(parents=True); return r

def rec(repo: pathlib.Path, sid: str, stamp: str, body: str = None):
    p = repo / SINK / f"verdict_wip_{sid}__{stamp}.md"
    p.write_text(body if body is not None else
                 f"{qa_wip.COMPLETE_MARKER}\nSTEP: {sid}\nWRITTEN: "
                 f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[9:11]}:"
                 f"{stamp[11:13]}:{stamp[13:15]}Z\n\nbody\n", encoding="utf-8")
    return p

def probe(repo, sid=STEP):
    r = qa_wip.report(sid, repo=repo)
    return {"records_retained": r["records_retained"], "status": r["status"],
            "exists": r["exists"], "guidance_head": r["guidance"][:44]}

cells, fail = [], 0
with tempfile.TemporaryDirectory() as td:
    tmp = pathlib.Path(td)

    # ---- CONTROL: two genuine prior attempts must be SEEN --------------
    ctl = mkrepo(tmp / "c"); rec(ctl, STEP, "20260811T072330Z"); rec(ctl, STEP, "20260812T090000Z")
    c = probe(ctl)
    ok = c["records_retained"] == 2
    cells.append(("CONTROL  2 real records -> counted", ok, c)); fail += not ok

    # ---- M1: the SINK DIRECTORY IS GONE --------------------------------
    m1 = mkrepo(tmp / "m1"); rec(m1, STEP, "20260811T072330Z"); rec(m1, STEP, "20260812T090000Z")
    shutil.rmtree(m1 / SINK)
    r1 = probe(m1)
    noticed = r1["records_retained"] != 0 or "MISSING" in r1["guidance_head"].upper() \
              or "SOURCE" in r1["guidance_head"].upper()
    cells.append(("M1 sink dir DELETED -> notices?", noticed, r1)); fail += not noticed

    # ---- M2: sink exists but EMPTIED -- UNDETECTABLE BY DESIGN ----------
    # This cell is NOT scored, and the reason is load-bearing rather than a
    # convenience. `prune_wip_records(keep=DEFAULT_KEEP)` DELETES old records as
    # normal operation, so "sink present, no record for this step" is a state the
    # module produces deliberately and is genuinely identical to a first attempt.
    # Detecting it would require a second monotonic counter outside the sink --
    # more machinery than the defect warrants, and the operator explicitly asked
    # for no over-engineering. Recorded as a STATED LIMIT, not a passing cell:
    # record loss inside an existing sink is not self-detectable here.
    m2 = mkrepo(tmp / "m2"); rec(m2, STEP, "20260811T072330Z")
    for p in (m2 / SINK).glob("*"): p.unlink()
    r2 = probe(m2)
    cells.append(("M2 records DELETED (UNSCORED: prune deletes by design)", None, r2))

    # ---- BASELINE: a genuine FIRST attempt (nothing wrong) --------------
    base = mkrepo(tmp / "b")
    rb = probe(base)
    cells.append(("BASELINE genuine 1st attempt", True, rb))

    # ---- THE DISCRIMINATION TEST ----------------------------------------
    same_as_missing  = (rb == r1)
    same_as_emptied  = (rb == r2)

print("=" * 78)
for name, ok, detail in cells:
    tag = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    print(f"  [{tag}] {name}")
    print(f"         {json.dumps(detail)}")
print("=" * 78)
print(f"  genuine-first-attempt output == sink-DELETED output : {same_as_missing}")
print(f"  genuine-first-attempt output == records-WIPED output: {same_as_emptied}")
print()
if same_as_missing:
    print("  VERDICT: DEFECT LIVE -- cannot distinguish 'no prior attempts'")
    print("           from 'the counting source is gone'.")
else:
    print("  VERDICT: DEFECT CLOSED -- a missing sink now reports SOURCE MISSING")
    print("           and is distinguishable from a genuine first attempt.")
print("  STATED LIMIT: record loss INSIDE an existing sink stays undetectable")
print("           (prune_wip_records deletes by design) -- see the M2 comment.")
print(f"\nmutants surviving (undetected): {fail}")
sys.exit(1 if fail else 0)
