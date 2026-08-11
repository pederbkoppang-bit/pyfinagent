#!/usr/bin/env python3
"""phase-86.36 criterion 6 -- mutation matrix for the retention change.

Criterion 6 names TWO cells and both are here:
  * revert the retention change  -> a NAMED assertion must go red
  * keep only ONE record         -> the coexistence assertion must fire

Each mutant is a COPY of `qa_wip.py` under a temp name; the tracked file is
never opened for writing, and its sha256 is compared before and after so
"I did not touch it" is proven rather than asserted.

A GREEN CONTROL RUNS FIRST. A cell that kills on an already-red suite proves
nothing, and this project has scored exactly that kind of false kill before.

Anchor uniqueness is checked: a `str.replace` that matches nothing returns a
perfectly normal string, so a non-mutation would otherwise be scored as a kill.

Exit 0 = every cell KILLED.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
SUBJECT = REPO / "scripts/qa/qa_wip.py"
CHECKER = REPO / "scripts/qa/verify_wip_retention_86_36.py"

MUTATIONS = [
    dict(
        id="M1-REVERT-RUN-STAMP",
        anchor='    if run_stamp is None:\n        return sink / f"verdict_wip_{sid}.md"',
        repl='    if True:  # MUTANT: ignore run_stamp, revert to the fixed path\n'
             '        return sink / f"verdict_wip_{sid}.md"',
        desc="revert the path to fixed-per-step -- both cycles collide again",
        expect_named=("the paths are DISTINCT", "cycle 1 survived cycle 2's first write"),
    ),
    dict(
        id="M2-RETAIN-ONLY-ONE",
        anchor="def prune_wip_records(step_id: str, repo: pathlib.Path | None = None,\n"
               "                      keep: int = DEFAULT_KEEP) -> list[pathlib.Path]:",
        repl="def prune_wip_records(step_id: str, repo: pathlib.Path | None = None,\n"
             "                      keep: int = 1) -> list[pathlib.Path]:\n"
             "    keep = 1  # MUTANT: retain only the newest",
        desc="cap retention at ONE -- the bounded-retention assertion must fire",
        expect_named=("prune keeps exactly `keep` records",),
    ),
    dict(
        id="M3-LIST-ONLY-NEWEST",
        anchor="    return sorted(set(found), key=_key, reverse=True)",
        repl="    return sorted(set(found), key=_key, reverse=True)[:1]  # MUTANT",
        desc="make list_wip_records report only the newest -- coexistence must fail",
        expect_named=("list_wip_records sees BOTH, newest first",),
    ),
    dict(
        id="M4-NEWEST-FIRST-SELECTION",
        anchor="            for cand in reversed(records):",
        repl="            for cand in records:  # MUTANT: newest-first (the real bug)",
        desc="reintroduce the newest-first selection bug this step already hit once",
        expect_named=("spawn 1 resolves to cycle 1's record",),
    ),
    dict(
        id="M5-LEAK-A-VERDICT-KEY",
        anchor='        "is_verdict": False,',
        repl='        "is_verdict": False,\n        "verdict": "PASS",  # MUTANT',
        desc="leak a scrapeable verdict key -- criterion 4 must fire",
        expect_named=("has NO 'verdict' key",),
    ),
]


def digest(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def run_checker(subject_override: pathlib.Path | None = None) -> tuple[int, str]:
    """Run the criterion checker. If given a mutant, import it AS qa_wip."""
    env = {**os.environ}
    if subject_override is not None:
        env["PYFIN_QA_WIP_OVERRIDE"] = str(subject_override)
    p = subprocess.run([sys.executable, str(CHECKER)], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=600)
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    before = digest(SUBJECT)
    print(f"subject : {SUBJECT.relative_to(REPO)}  sha256[:16]={before}")

    rc, out = run_checker()
    print(f"\n[CONTROL] unmutated checker -> exit {rc}")
    if rc != 0:
        print("FAIL -- the control is RED, so no kill below would mean anything.")
        print(out[-1500:])
        return 1
    print("  ok -- green control established")

    src = SUBJECT.read_text()
    rows, survived = [], []
    for cell in MUTATIONS:
        mid, anchor, repl = cell["id"], cell["anchor"], cell["repl"]
        n = src.count(anchor)
        if n != 1:
            rows.append((mid, "ANCHOR", f"matched {n} time(s) -- stale anchor"))
            survived.append(mid)
            continue
        mut_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"pyfin_86_36_{mid.lower()}_"))
        mut = mut_dir / "qa_wip.py"
        try:
            mut.write_text(src.replace(anchor, repl, 1))
            mrc, mout = run_checker(mut)
            if mrc == 0:
                rows.append((mid, "SURVIVED", cell["desc"]))
                survived.append(mid)
            else:
                # The name must appear ON A FAILING LINE -- matched line-wise,
                # not by prefix. Two earlier versions of this were wrong in
                # OPPOSITE directions and both were caught by running it:
                #   * `n_ in mout` accepted the name anywhere, including the
                #     `ok` lines the checker prints for every assertion, so
                #     MIS-ATTRIB could never fire and any red scored as a named
                #     kill. That produced a FALSE KILL on M5.
                #   * `f"FAIL {n_}"` required the name to IMMEDIATELY follow the
                #     verdict word, but these names are suffixes of their line
                #     (`FAIL report(spawned_at=...) has NO 'verdict' key`), so a
                #     genuine kill was mis-scored MIS-ATTRIB.
                # Line-wise containment is the property actually wanted.
                fail_lines = [ln for ln in mout.splitlines()
                              if ln.strip().startswith(("FAIL ", "FAILED:"))]
                named = [n_ for n_ in cell["expect_named"]
                         if any(n_ in ln for ln in fail_lines)]
                if named:
                    rows.append((mid, "KILLED", f"{cell['desc']}  [named: {named[0][:46]}]"))
                else:
                    rows.append((mid, "MIS-ATTRIB",
                                 "went red but not on a NAMED assertion from this cell"))
                    survived.append(mid)
        finally:
            shutil.rmtree(mut_dir, ignore_errors=True)

    after = digest(SUBJECT)
    w = max(len(r[0]) for r in rows)
    print()
    for mid, verdict, note in rows:
        print(f"  {mid:<{w}}  {verdict:<11} {note[:120]}")
    print(f"\ntracked subject UNCHANGED: {before == after}  ({before} -> {after})")
    if survived:
        print(f"\nFAIL -- {len(survived)} cell(s) did not kill: {survived}")
        return 1
    print(f"\nOK -- all {len(rows)} cells KILLED on a named assertion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
