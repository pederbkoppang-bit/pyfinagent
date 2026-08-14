#!/usr/bin/env python3
"""phase-86.79 criterion 7 -- mutation matrix for the attempt-counter fix.

    source .venv/bin/activate && python scripts/qa/mutation_matrix_86_79.py

Each mutant is a COPY of `qa_wip.py` under a temp name, fed to
`verify_counter_86_79.py` through the `PYFIN_QA_WIP_OVERRIDE` seam. The tracked
file is never opened for writing, and its sha256 is compared before and after so
"I did not touch it" is PROVEN rather than asserted.

A GREEN CONTROL RUNS FIRST. A cell that "kills" an already-red checker proves
nothing, and this project has scored exactly that kind of false kill before.

ANCHOR UNIQUENESS IS CHECKED. `str.replace` on a non-matching anchor returns a
perfectly normal string, so a non-mutation would otherwise be scored as a kill --
an operation that cannot fail loudly.

EACH CELL MUST DISCRIMINATE. A cell is only a kill if the checker goes red AND
at least one of the NAMED assertions it is supposed to break is among the
failures. Without that, a mutant that reddens the checker for an unrelated
reason (an import error, say) counts as a kill it did not earn.

Exit 0 = every cell KILLED, by the assertion it was aimed at.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
SUBJECT = REPO / "scripts/qa/qa_wip.py"
CHECKER = REPO / "scripts/qa/verify_counter_86_79.py"

MUTATIONS = [
    dict(
        id="M1-DROP-IDENTITY-GUARD",
        desc="compute attempt_number even when no record belongs to this spawn "
             "-- reintroduces the write-first temporal coupling",
        anchor="    if not matched_current:",
        repl="    if False:  # MUTANT: trust the count without identity",
        expect_named=("attempt_number REFUSES before the write-first record exists",
                      "...and it refuses with the right reason"),
    ),
    dict(
        id="M2-PRUNE-STOPS-ACCOUNTING",
        desc="prune deletes without recording the loss -- the PERF_RECORD_LOST "
             "remedy removed, so the count saturates again",
        anchor="    _record_loss(step_id, len(doomed), repo)",
        repl="    pass  # MUTANT: delete without recording the loss",
        expect_named=("attempt_number SURVIVES the prune and still reports 6",
                      "NEW number DOES escalate after a prune (the fix)"),
    ),
    dict(
        id="M3-LOSS-LEDGER-CAN-DECREASE",
        desc="the loss account stops being monotonic -- Prometheus's 'do not use "
             "a counter to expose a value that can decrease', reintroduced",
        anchor="    total = max(prior, prior + max(0, int(n_lost)))",
        repl="    total = int(n_lost)  # MUTANT: overwrite, do not accumulate",
        expect_named=("the loss account is MONOTONIC across a no-op prune",),
    ),
    dict(
        id="M4-FAIL-OPEN-WITH-ZERO",
        desc="an uncomputable count reports 0 instead of None -- fails OPEN, "
             "which is the direction that SUPPRESSES escalation",
        anchor='    out = {"attempt_number": None, "prior_attempts": None,',
        repl='    out = {"attempt_number": 0, "prior_attempts": 0,  # MUTANT',
        expect_named=("missing sink -> attempt_number is None, not 0",
                      "no spawned_at -> attempt_number is None, not 0"),
    ),
    dict(
        id="M5-RESTORE-OFF-BY-ONE-COMMENT",
        desc="restore DEFAULT_KEEP's original off-by-one wording -- the doc/code "
             "divergence criterion 4 forbids leaving in place",
        anchor="#: TOTAL records retained, **INCLUSIVE of the current one**",
        repl="#: Current record + this many prior attempts.  # MUTANT",
        expect_named=("the comment states the UNIT (TOTAL / INCLUSIVE), not just a name",),
    ),
    # ── M8/M9 were found by the CYCLE-1 Q/A, not by this matrix. Both SURVIVED
    # the cycle-1 checker with executed behavioural differentials. They are added
    # permanently so the guards written for them are proven to be able to fail.
    dict(
        id="M8-DROP-LOSS-ADDBACK-ON-NO-RECORD-PATH",
        desc="[cycle-1 Q/A survivor] drop the pruned-away add-back on the "
             "no_record_for_this_spawn branch -- reintroduces the exact "
             "under-count-suppresses-escalation defect this step exists to remove, "
             "on a branch this step created",
        anchor='        out["prior_attempts"] = lost_n + len(records)',
        repl='        out["prior_attempts"] = len(records)  # MUTANT: ignore pruned-away',
        expect_named=("prior_attempts counts PRUNED-AWAY records too, not just retained ones",
                      "...and that difference is what makes F1b's ceiling reachable on this path"),
    ),
    dict(
        id="M9-LOWER-BOUND-FLAG-NEVER-SET",
        desc="[cycle-1 Q/A survivor] make attempt_number_is_lower_bound always "
             "False -- the mutant silently claims exactness in precisely the "
             "regime the field exists to flag",
        anchor='        out["attempt_number_is_lower_bound"] = True',
        repl='        out["attempt_number_is_lower_bound"] = False  # MUTANT',
        # Deliberately pointed at the ABOVE-the-window assertion, which M10's
        # boundary mutation canNOT break -- so M9 and M10 stay distinguishable
        # rather than both being scored by whichever assertion fires first.
        expect_named=("above the window with no loss account -> FLAGGED as a floor",),
    ),
    # ── M10/M11 were found by the CYCLE-2 Q/A. M10 is a NEW instance of M9's class
    # INSIDE the guard written to close M9 -- the remediation was narrower than the
    # class, which is the whole reason it is pinned here.
    dict(
        id="M10-LOWER-BOUND-MISSES-THE-BOUNDARY",
        desc="[cycle-2 Q/A survivor] >= DEFAULT_KEEP -> > DEFAULT_KEEP. At exactly "
             "keep retained with no ledger -- the state a legacy prune leaves "
             "behind -- the mutant claims exactness where the flag exists to warn",
        anchor="    if lost is None and len(records) >= DEFAULT_KEEP:",
        repl="    if lost is None and len(records) > DEFAULT_KEEP:  # MUTANT: off-by-one",
        expect_named=("EXACTLY AT the window with no ledger -> FLAGGED (the boundary itself)",
                      "the field discriminates (not all regimes agree)"),
    ),
    dict(
        id="M11-RECORD-LOSS-AFTER-THE-UNLINK",
        desc="[cycle-2 Q/A survivor] record the loss AFTER the unlink loop and from "
             "what was actually removed -- a crash mid-prune then UNDER-counts, the "
             "direction that SUPPRESSES escalation",
        # The mutation must MOVE the call, not delete it -- deleting it is already
        # M2, and a cell that duplicates another proves nothing new.
        anchor=("    _record_loss(step_id, len(doomed), repo)\n"
                "    removed = []\n"
                "    for p in doomed:\n"
                "        try:\n"
                "            p.unlink()\n"
                "            removed.append(p)\n"
                "        except OSError:\n"
                "            pass\n"
                "    return removed"),
        repl=("    removed = []  # MUTANT: account AFTER the unlink, from what survived\n"
              "    for p in doomed:\n"
              "        try:\n"
              "            p.unlink()\n"
              "            removed.append(p)\n"
              "        except OSError:\n"
              "            pass\n"
              "    _record_loss(step_id, len(removed), repo)\n"
              "    return removed"),
        expect_named=("the loss was ALREADY recorded when the crash hit -> OVER-count, not None",
                      "...so the derived count errs HIGH after a crash, never low"),
    ),
    dict(
        id="M6-LEAK-A-VERDICT-KEY",
        desc="leak a scrapeable verdict key -- the no-self-verdict invariant "
             "criterion 6 turns on",
        anchor='        "is_verdict": False,',
        repl='        "is_verdict": False,\n        "verdict": "PASS",  # MUTANT',
        expect_named=("NO report() variant ever carries a `verdict` key",),
    ),
    dict(
        id="M7-DROP-THE-UNIT",
        desc="remove the unit from the payload -- E1's remedy ('a name is not a "
             "unit') removed, and with it the mitigation standing in for the "
             "un-applied qa.md correction",
        # NOTE: the replacement must stay SYNTACTICALLY VALID. The first draft of
        # this cell produced a broken string literal, so the mutant crashed and
        # the checker went red with NO [FAIL] lines -- red for the wrong reason.
        # The discrimination guard caught it and refused to score it as a kill.
        anchor='            "the current spawn\'s own write-first record. This is a GAUGE (pruning "',
        repl='            "the current spawn\'s own write-first record. This is a (pruning "  # MUTANT: unit word stripped',
        expect_named=("records_retained_unit states it is a GAUGE, not a counter",),
    ),
]

FAIL_LINE = re.compile(r"^\s*\[FAIL\] (.+?)(?: --.*)?$", re.M)


def digest(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def run_checker(subject_override: pathlib.Path | None = None):
    env = {**os.environ}
    if subject_override is not None:
        env["PYFIN_QA_WIP_OVERRIDE"] = str(subject_override)
    else:
        env.pop("PYFIN_QA_WIP_OVERRIDE", None)
    p = subprocess.run([sys.executable, str(CHECKER)], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=900)
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    before = digest(SUBJECT)
    print(f"subject : {SUBJECT.relative_to(REPO)}  sha256[:16]={before}")
    print(f"checker : {CHECKER.relative_to(REPO)}")

    rc, out = run_checker()
    print(f"\n[CONTROL] unmutated checker -> exit {rc}")
    if rc != 0:
        print("ABORT -- the control is RED, so no kill below would mean anything.")
        print(out[-2000:])
        return 1
    ran = re.search(r"checks run : (\d+)", out)
    print(f"  ok -- GREEN control established ({ran.group(1) if ran else '?'} checks)")

    src = SUBJECT.read_text(encoding="utf-8")
    rows, survived = [], []
    for cell in MUTATIONS:
        mid, anchor, repl = cell["id"], cell["anchor"], cell["repl"]
        n = src.count(anchor)
        if n != 1:
            rows.append((mid, "ANCHOR-STALE", f"matched {n} time(s), expected 1"))
            survived.append(mid)
            continue
        d = pathlib.Path(tempfile.mkdtemp(prefix=f"pyfin_86_79_{mid.lower()}_"))
        mut = d / "qa_wip.py"
        try:
            mut.write_text(src.replace(anchor, repl, 1), encoding="utf-8")
            mrc, mout = run_checker(mut)
            if mrc == 0:
                rows.append((mid, "SURVIVED", cell["desc"]))
                survived.append(mid)
                continue
            failures = set(FAIL_LINE.findall(mout))
            aimed = [a for a in cell["expect_named"] if a in failures]
            if not aimed:
                # RED, but not for the reason this cell exists. That is not a kill.
                rows.append((mid, "RED-WRONG-REASON",
                             f"expected one of {cell['expect_named']}, "
                             f"got {sorted(failures)[:4] or 'no [FAIL] lines'}"))
                survived.append(mid)
            else:
                rows.append((mid, "KILLED", f"by: {aimed[0]}"))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    after = digest(SUBJECT)
    print(f"\n{'=' * 74}\nMUTATION MATRIX\n{'=' * 74}")
    for mid, status, detail in rows:
        print(f"  {status:17s} {mid:30s} {detail}")
    print(f"\n  subject sha256[:16] before={before} after={after} "
          f"-> tracked file {'UNCHANGED' if before == after else '*** MODIFIED ***'}")
    print(f"  cells: {len(MUTATIONS)}   killed: {len(MUTATIONS) - len(survived)}   "
          f"survived/unearned: {len(survived)}")
    ok = not survived and before == after
    print(f"\n  {'ALL CELLS KILLED' if ok else 'MATRIX INCOMPLETE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
