#!/usr/bin/env python3
"""phase-86.36 criteria 2, 3 and 4 -- retention, resolution, and non-verdict-ness.

Runs against a SCRATCH sink, never the live one: a concurrent peer session has
Q/A runs writing to `.claude/agent-memory/qa/verdicts/` right now, and driving
retention against it would destroy their evidence -- the defect under study.

The one exception is the audit_memory.py leg (criterion 2), which must be run
against the REAL corpus to mean anything; it is read-only there and asserts the
output is UNCHANGED.

EVERY SECTION CARRIES A CARDINALITY FLOOR. A loop over an empty set emits no
assertions and reports success; that shape has already produced a false green in
this project, so each section asserts how many subjects it actually examined.

Exit 0 = all green.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# TEST-ONLY SEAM. `PYFIN_QA_WIP_OVERRIDE` points at a mutant copy of qa_wip.py
# so mutation_matrix_86_36.py can drive THIS checker against a mutated subject
# without ever writing to the tracked file. Unset (the normal case) this imports
# the real module, so a plain run is unaffected. Same shape as the
# PYFINAGENT_86_34_SWEEP_ROOT seam and for the same reason: the RED half of a
# criterion is unprovable without one.
import importlib.util  # noqa: E402
import os  # noqa: E402

_override = os.environ.get("PYFIN_QA_WIP_OVERRIDE")
if _override:
    _spec = importlib.util.spec_from_file_location("qa_wip", _override)
    qa_wip = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(qa_wip)
else:
    import qa_wip  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
PASS = []
FAIL = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  -- {detail}" if detail else ""))
    return cond


def _write(sink, step, stamp, written, body, complete):
    p = qa_wip.resolve_wip_path(step, repo=sink, run_stamp=stamp)
    p.parent.mkdir(parents=True, exist_ok=True)
    marker = qa_wip.COMPLETE_MARKER if complete else qa_wip.INCOMPLETE_MARKER
    p.write_text(f"{marker}\nSTEP: {step}\nWRITTEN: {written}\n"
                 + (f"COMPLETED: {written}\n" if complete else "")
                 + "\n" + body + "\n", encoding="utf-8")
    return p


def main() -> int:
    step = "86.36"
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="pyfin_86_36_verify_"))
    try:
        # ---------- CRITERION 2: coexistence -----------------------------
        print("\n[2] TWO CYCLES' RECORDS COEXIST")
        p1 = _write(tmp, step, "20260811T060000Z", "2026-08-11T06:00:00Z",
                    "\n".join(f"- cycle-1 finding {i}" for i in range(40)), True)
        p2 = _write(tmp, step, "20260811T064000Z", "2026-08-11T06:40:00Z",
                    "(first tool call, nothing yet)", False)
        b1, b2 = p1.stat().st_size, p2.stat().st_size
        check("both records exist on disk", p1.exists() and p2.exists(),
              f"{p1.name}={b1}B  {p2.name}={b2}B")
        check("the paths are DISTINCT", p1 != p2)
        check("cycle 1 survived cycle 2's first write", "cycle-1 finding 39" in p1.read_text(),
              f"{b1} bytes retained (pre-fix this went to ~124)")
        recs = qa_wip.list_wip_records(step, repo=tmp)
        check("list_wip_records sees BOTH, newest first", len(recs) == 2 and recs[0] == p2,
              f"{[r.name for r in recs]}")
        h1 = qa_wip.headers(p1.read_text()); h2 = qa_wip.headers(p2.read_text())
        check("each carries its OWN WRITTEN stamp",
              h1.get("WRITTEN") == "2026-08-11T06:00:00Z"
              and h2.get("WRITTEN") == "2026-08-11T06:40:00Z")
        check("cycle 1 carries COMPLETED, cycle 2 does not",
              bool(h1.get("COMPLETED")) and not h2.get("COMPLETED"))

        # ---------- CRITERION 3: resolution + staleness -------------------
        print("\n[3] RESOLUTION FOR A GIVEN CYCLE, AND STALENESS STILL FIRES")
        r2 = qa_wip.report(step, repo=tmp, spawned_at="2026-08-11T06:40:00Z")
        check("spawn 2 resolves to cycle 2's record", r2["path"] == str(p2),
              f"status={r2['status']}")
        check("and lists cycle 1 as a PRIOR, not merged", r2["prior_records"] == [str(p1)])
        r1 = qa_wip.report(step, repo=tmp, spawned_at="2026-08-11T06:00:00Z")
        check("spawn 1 resolves to cycle 1's record", r1["path"] == str(p1),
              f"status={r1['status']}")
        r_future = qa_wip.report(step, repo=tmp, spawned_at="2026-08-11T09:00:00Z")
        check("a spawn NEWER than every record reports STALE",
              r_future["status"] == "STALE", f"got {r_future['status']}")
        r_bad = qa_wip.report(step, repo=tmp, spawned_at="not-a-timestamp")
        check("an unparseable spawned_at reports IDENTITY_UNKNOWN",
              r_bad["status"] == "IDENTITY_UNKNOWN", f"got {r_bad['status']}")

        # ---------- CRITERION 4: never a verdict, over EVERY record -------
        print("\n[4] NO RECORD IS EVER READABLE AS A VERDICT")
        examined = 0
        for spawn in ("2026-08-11T06:00:00Z", "2026-08-11T06:40:00Z", None):
            rep = qa_wip.report(step, repo=tmp, spawned_at=spawn)
            examined += 1
            if not check(f"report(spawned_at={spawn!r}) has NO 'verdict' key",
                         "verdict" not in rep):
                break
            if not check(f"report(spawned_at={spawn!r}) is_verdict is False",
                         rep["is_verdict"] is False):
                break
        check("criterion-4 cardinality floor: >=3 reports examined", examined >= 3,
              f"examined={examined}")

        # ---------- CRITERION 2b: bounded retention -----------------------
        print("\n[2b] RETENTION IS BOUNDED")
        for i in range(6):
            _write(tmp, step, f"2026081{i}T120000Z".replace("2026081", "2026081"),
                   f"2026-08-1{i}T12:00:00Z", f"filler {i}", True)
        before_n = len(qa_wip.list_wip_records(step, repo=tmp))
        removed = qa_wip.prune_wip_records(step, repo=tmp, keep=3)
        after = qa_wip.list_wip_records(step, repo=tmp)
        check("prune keeps exactly `keep` records", len(after) == 3,
              f"{before_n} -> {len(after)}, removed {len(removed)}")
        check("prune removed the OLDEST, kept the newest", after[0].stat().st_size > 0)
        try:
            qa_wip.prune_wip_records(step, repo=tmp, keep=0)
            check("keep=0 is REFUSED", False, "no exception raised")
        except ValueError:
            check("keep=0 is REFUSED (retaining zero is the defect, not a setting)", True)

        # ---------- CRITERION 2c: the memory auditor is unaffected --------
        print("\n[2c] audit_memory.py OUTPUT UNCHANGED (real corpus, read-only)")
        aud = REPO / "scripts/housekeeping/audit_memory.py"
        if not aud.exists():
            check("audit_memory.py present", False, f"missing at {aud}")
        else:
            def run_aud():
                r = subprocess.run([sys.executable, str(aud)], cwd=REPO,
                                   capture_output=True, text=True, timeout=120)
                return r.returncode, (r.stdout + r.stderr)
            rc_a, out_a = run_aud()
            probe = (REPO / qa_wip.MEMORY_DIR / qa_wip.WIP_SUBDIR
                     / f"verdict_wip_{step}__20260811T235959Z.md")
            probe.parent.mkdir(parents=True, exist_ok=True)
            probe.write_text(f"{qa_wip.INCOMPLETE_MARKER}\nSTEP: {step}\n"
                             f"WRITTEN: 2026-08-11T23:59:59Z\n\nprobe\n", encoding="utf-8")
            try:
                rc_b, out_b = run_aud()
            finally:
                probe.unlink(missing_ok=True)
            check("a stamped record does NOT change the auditor's exit code",
                  rc_a == rc_b, f"{rc_a} vs {rc_b}")
            check("a stamped record does NOT change the auditor's output",
                  out_a == out_b,
                  "identical" if out_a == out_b else f"differs by {abs(len(out_a)-len(out_b))} chars")

        print(f"\n{'ALL GREEN' if not FAIL else 'RED'} -- {len(PASS)} passed, {len(FAIL)} failed")
        if FAIL:
            for f in FAIL:
                print(f"  FAILED: {f}")
        return 1 if FAIL else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
