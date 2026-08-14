#!/usr/bin/env python3
"""phase-86.31 criterion 5 -- mutation matrix for the Q/A write-first rail.

RE-RUNNABLE AND HERMETIC. Every mutation is applied to a COPY inside a
`mkdtemp` MINI-REPO laid out with the real relative paths, and the AUTHOR'S OWN
checker (`verify_qa_write_first_86_31.py`) is executed from inside it -- so the
matrix drives the real assertions rather than a re-implementation, and the
tracked tree is never written. That also matters because a concurrent Claude
session works in this same tree.

A cell is KILLED when the checker's exit code flips 0 -> 1 AND the named
assertion appears in its failure list. A cell that merely turns the run red for
some OTHER reason is NOT a kill -- that is a broken harness wearing a kill's
clothes, and it is scored SURVIVED-MISATTRIBUTED so a reader can see it.

ANCHOR UNIQUENESS IS ASSERTED FOR EVERY MUTATION. A `str.replace` that matches
nothing looks exactly like a successful edit; without this assertion a
non-mutation scores as a kill.

THE CONTROL RUNS FIRST. A green control is what separates "the guard died" from
"my mini-repo is broken".

    $ python scripts/qa/mutation_matrix_86_31.py
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
CHECKER_REL = "scripts/qa/verify_qa_write_first_86_31.py"

#: Every file the checker touches, at its real repo-relative path.
MINI_REPO_FILES = [
    CHECKER_REL,
    "scripts/qa/qa_wip.py",
    "scripts/housekeeping/audit_memory.py",
    ".claude/hooks/qa-write-guard.sh",
    ".claude/agents/qa.md",
    ".claude/workflows/qa-verdict.js",
    "docs/runbooks/per-step-protocol.md",
]

#: Sentinel target for a cell that removes the LIVE WIP artifacts instead of
#: editing a file. Section [9] asserts the rail actually produced one, and the
#: only way to test that assertion is to take the subject away.
DELETE_WIP = "@DELETE_WIP_ARTIFACTS"

WIP_DIR_REL = ".claude/agent-memory/qa/verdicts"


#: (id, target_rel, anchor, replacement, expected_failure_substring, description)
#:
#: `anchor` is EITHER a literal string (replaced once) OR a 2-tuple
#: (start, end) naming a REGION: everything from `start` up to but excluding
#: `end` is replaced. The region form exists because the first draft of P1/P2
#: replaced only a SECTION HEADING and left the whole body in place -- the
#: cells "SURVIVED", which read as "the anchor is not load-bearing" when it
#: actually meant "the probe did not do what its own description said". A
#: mutation whose description says "strip the section" must strip the section.
MUTATIONS = [
    # ---- the guard itself -------------------------------------------------
    ("G1", ".claude/hooks/qa-write-guard.sh",
     'MEMORY_DIR = ".claude/agent-memory/qa/"',
     'MEMORY_DIR = ".claude/"',
     "DENY   [qa] Write the masterplan",
     "widen the allowlist to all of .claude/ -- the masterplan becomes writable"),

    ("G2", ".claude/hooks/qa-write-guard.sh",
     '    return n == "qa" or n.startswith("qa-") or n.startswith("qa_")',
     '    return False',
     "DENY   [qa] Write production code",
     "the guard never fires -- the evaluator may edit the work it is grading"),

    ("G3", ".claude/hooks/qa-write-guard.sh",
     'if MEMORY_DIR.rstrip("/") + "/" not in norm + "/":',
     'if MEMORY_DIR.rstrip("/") + "/" in norm + "/":',
     "ALLOW  [qa] verdict WIP for the step under evaluation",
     "invert the membership test -- write-first itself becomes impossible"),

    ("G4", ".claude/hooks/qa-write-guard.sh",
     'print("deny qa-write-outside-memory")',
     'print("allow deny-suppressed")',
     "DENY   [qa] Write production code",
     "suppress the deny verdict -- the hook reports but never blocks"),

    ("G5", ".claude/hooks/qa-write-guard.sh",
     'tool_name in ("Write", "Edit")',
     'tool_name in ("Write",)',
     "DENY   [qa] Edit  production code",
     "drop Edit from the intercepted tools -- Write blocked, Edit wide open"),

    ("G6", ".claude/hooks/qa-write-guard.sh",
     '    norm = os.path.normpath(file_path.replace("\\\\", "/"))',
     '    norm = file_path.replace("\\\\", "/")',
     "DENY   [qa] Write traversal OUT of the memory dir",
     "drop the normpath collapse -- ../ traversal escapes the substring test"),

    # G7 is THE cycle-1 finding, pinned as a permanent regression cell.
    ("G7", ".claude/hooks/qa-write-guard.sh",
     '    return n == "qa" or n.startswith("qa-") or n.startswith("qa_")',
     '    return n == "qa"',
     "DENY   [qa-80-2-c2] Write production FRONTEND source",
     "revert to the EXACT-name predicate -- 27 real qa-* identities walk past the guard again"),

    # G8 is the FAIL-OPEN trap: the guard is embedded in a bash single-quoted
    # block, so one apostrophe kills python and the hook allows everything.
    # This is not hypothetical -- it was introduced while fixing G7.
    ("G8", ".claude/hooks/qa-write-guard.sh",
     '    n = (name or "").strip().lower()',
     "    n = (name or \"\").strip().lower()  # Main's own calls carry no name",
     "no apostrophe inside the single-quoted python block",
     "inject one apostrophe -- python dies and the FAIL-OPEN hook allows everything"),

    # ---- the marker reader (the CALLER criterion 3 requires) ---------------
    ("M1", "scripts/qa/qa_wip.py",
     '        return m.group(1) if m else "UNMARKED"',
     '        return "COMPLETE"',
     "a TRUNCATED artifact stays INCOMPLETE",
     "always report COMPLETE -- a partial verdict becomes indistinguishable"),

    ("M2", "scripts/qa/qa_wip.py",
     '        return m.group(1) if m else "UNMARKED"',
     '        return m.group(1) if m else "INCOMPLETE"',
     "a file with no marker is UNMARKED",
     "fold UNMARKED into INCOMPLETE -- a hand-written file poses as write-first output"),

    # phase-86.21: anchor repointed. The original pinned '"guidance": "",\n    }',
    # which stopped matching once later phases added keys AFTER guidance -- the
    # cell then reported ANCHOR-BAD, i.e. it silently stopped testing anything.
    ("M3", "scripts/qa/qa_wip.py",
     '        "records_retained": len(records),\n',
     '        "records_retained": len(records),\n        "verdict": "PASS",\n',
     "carries no key to scrape as one",
     "give the report a verdict key -- the recovery path becomes scrapable as PASS"),

    ("M4", "scripts/qa/qa_wip.py",
     r'_STEP_ID_RE = re.compile(r"\A[0-9]+(?:\.[0-9]+)*\Z")',
     r'_STEP_ID_RE = re.compile(r"\A.+\Z", re.S)',
     "hostile step id refused",
     "accept any step id -- path traversal out of the memory dir via the resolver"),

    # phase-86.21: anchor repointed. The original pinned a one-line return that
    # phase-86.36 refactored into a `sink` local, so this cell had been dead too.
    ("M5", "scripts/qa/qa_wip.py",
     '    sink = root / MEMORY_DIR / WIP_SUBDIR\n    if run_stamp is None:',
     '    sink = root / MEMORY_DIR\n    if run_stamp is None:',
     "the resolved WIP path sits under",
     "revert the sink to the memory dir TOP LEVEL -- the audit_memory collision returns"),

    ("M6", "scripts/qa/qa_wip.py",
     '\n            "is NO VERDICT, NEVER PASS."',
     '\n            "you may transcribe it."',
     "restates NO VERDICT, NEVER PASS",
     "soften the COMPLETE guidance -- recovery quietly becomes verdict-shopping"),

    # ---- identity / staleness (cycle-1 finding 3) -------------------------
    ("I1", "scripts/qa/qa_wip.py",
     '        elif written_dt < spawn_dt:',
     '        elif False:',
     "written BEFORE the spawn is STALE",
     "stop comparing timestamps -- a PRIOR cycle's COMPLETE artifact reads as current"),

    ("I2", "scripts/qa/qa_wip.py",
     '    if out["status"] in ("STALE", "IDENTITY_UNKNOWN"):\n        out["recoverable"] = False',
     '    if out["status"] in ("STALE", "IDENTITY_UNKNOWN"):\n        out["recoverable"] = True',
     "a STALE artifact is NOT offered as recoverable evidence",
     "offer stale evidence as recoverable -- Main acts on the previous cycle's findings"),

    ("I3", "scripts/qa/qa_wip.py",
     '_HEADER_RE = re.compile(r"(?m)^\\s*(WRITTEN|COMPLETED|STEP):\\s*(\\S.*?)\\s*$")',
     '_HEADER_RE = re.compile(r"(?m)^\\s*(COMPLETED|STEP):\\s*(\\S.*?)\\s*$")',
     "written BEFORE the spawn is STALE",
     "stop parsing the WRITTEN stamp -- every artifact becomes identity-unknown or trusted"),

    # ---- the prose anchors (vacuity shapes #2/#3) -------------------------
    # P* are REGION DELETES. Q* are REWORD-INVERSIONS -- the mutation the
    # cycle-1 Q/A used to prove the old bare-token scan was illusory. Q* keep
    # EVERY scanned literal and invert only the meaning.
    ("P1", ".claude/agents/qa.md",
     ("## Write-first for your VERDICT FILE ONLY (phase-86.31, BINDING)",
      "## Verification order (deterministic FIRST)"),
     "",
     ".claude/agents/qa.md :: write-first section",
     "region-DELETE the whole write-first section from qa.md"),

    ("P2", ".claude/workflows/qa-verdict.js",
     ("  'STEP 0b (binding, phase-86.31, path revised by phase-86.36): WRITE-FIRST FOR YOUR VERDICT FILE ONLY.",
      "  'You are INDEPENDENT of the author"),
     "",
     ".claude/workflows/qa-verdict.js :: STEP 0b",
     "region-DELETE the whole STEP 0b block from the Workflow prompt"),

    ("P3", "docs/runbooks/per-step-protocol.md",
     ("**RECOVERY AFTER A DROPPED Q/A", "\n**Q/A rubric hardening"),
     "",
     "docs/runbooks/per-step-protocol.md :: recovery contract",
     "region-DELETE Main's whole recovery contract"),

    ("Q1", ".claude/agents/qa.md",
     ("## Write-first for your VERDICT FILE ONLY (phase-86.31, BINDING)",
      "## Verification order (deterministic FIRST)"),
     ("## Write-first for your VERDICT FILE ONLY (phase-86.31, BINDING)\n\n"
      "RETIRED 2026-08-11. Do NOT create "
      "`.claude/agent-memory/qa/verdicts/verdict_wip_<step_id>.md`. The "
      "`STATUS: INCOMPLETE` and `STATUS: COMPLETE` markers, the `verdict_wip_` "
      "naming, the instruction to Append findings incrementally and the final act "
      "are all obsolete. Write nothing at all.\n\n"),
     ".claude/agents/qa.md :: write-first section",
     "REWORD-INVERSION of qa.md: every scanned literal kept, meaning inverted (the cycle-1 Q/A's own mutant)"),

    ("Q1b", ".claude/agents/qa.md",
     ("## Write-first for your VERDICT FILE ONLY (phase-86.31, BINDING)",
      "## Verification order (deterministic FIRST)"),
     ("## Write-first for your VERDICT FILE ONLY (phase-86.31, BINDING)\n\n"
      "This section is RETIRED. **Create** nothing.\n\n"
      + ("Historical note preserved for auditors: the rail formerly asked for "
         "`.claude/agent-memory/qa/verdicts/verdict_wip_<step_id>.md` with a first line "
         "`STATUS: INCOMPLETE -- not a verdict`, an instruction to Append findings as they "
         "were established, and a final act flipping it to `STATUS: COMPLETE`. "
         "None of that is required now. ") * 6 + "\n\n"),
     ".claude/agents/qa.md :: write-first section: contains NO anti-directive language",
     "LONG reword-inversion: passes the length floor AND keeps the **Create** literal, so ONLY the anti-directive leg can kill it"),

    ("Q2", ".claude/workflows/qa-verdict.js",
     ("  'STEP 0b (binding, phase-86.31, path revised by phase-86.36): WRITE-FIRST FOR YOUR VERDICT FILE ONLY.",
      "  'You are INDEPENDENT of the author"),
     # The start marker is preserved VERBATIM. The first draft of this cell
     # replaced it too, so the section became UNLOCATABLE and the cell died on
     # "the section is locatable" -- a kill, but by the wrong assertion, which
     # is vacuity shape #11 (mis-attributed kill mechanism) wearing a green
     # badge. Keeping the marker forces the anti-directive / imperative legs to
     # be the thing that fires.
     ("  'STEP 0b (binding, phase-86.31, path revised by phase-86.36): WRITE-FIRST FOR YOUR VERDICT FILE ONLY. "
      "RETIRED -- do not create .claude/agent-memory/qa/verdicts/verdict_wip_<step_id>.md. "
      "The STATUS: INCOMPLETE and STATUS: COMPLETE markers are obsolete and there is no "
      "FINAL act.',\n"
      + "  'Historical note for auditors: the rail formerly asked for a write-first record "
        "at .claude/agent-memory/qa/verdicts/verdict_wip_<step_id>.md whose first line was "
        "STATUS: INCOMPLETE, appended incrementally, with a FINAL act flipping it to "
        "STATUS: COMPLETE. None of that is required now.',\n" * 4
      + "  '',\n"),
     ".claude/workflows/qa-verdict.js :: STEP 0b: contains NO anti-directive language",
     "REWORD-INVERSION of the Workflow prompt with the section marker PRESERVED: literals kept, length floor cleared, directive negated"),

    # ---- the BEHAVIOURAL guard (cycle-2 finding A) ------------------------
    # experiment_results calls section [9] the ONLY non-circular evidence that
    # the directive reaches the agent. A claim that specific must be falseable.
    # Before the fix this cell SURVIVED: deleting every artifact left the
    # checker at exit=0 with zero red, because [9] had no floor.
    ("B1", DELETE_WIP, None, None,
     "the LIVE rail has produced at least one WIP artifact",
     "remove every live WIP artifact -- the exact state section [9] exists to detect"),
]


def build_mini_repo(dest: pathlib.Path) -> None:
    # The live WIP artifacts come too: without them section [9] has no subject
    # and the CONTROL would be red for a reason that has nothing to do with any
    # mutation.
    (dest / WIP_DIR_REL).mkdir(parents=True, exist_ok=True)
    for p in (REPO / WIP_DIR_REL).glob("*.md"):
        shutil.copy2(p, dest / WIP_DIR_REL / p.name)
    for rel in MINI_REPO_FILES:
        src = REPO / rel
        if not src.exists():
            raise SystemExit(f"mini-repo source missing: {rel}")
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)


def run_checker(root: pathlib.Path):
    proc = subprocess.run(
        [sys.executable, str(root / CHECKER_REL)],
        capture_output=True, text=True, timeout=600, cwd=str(root),
    )
    return proc.returncode, proc.stdout + proc.stderr


def failure_lines(output: str) -> list[str]:
    """The trailing `  FAIL <label>` summary block."""
    return [m.group(1).strip() for m in re.finditer(r"^  FAIL (.+)$", output, re.M)]


def main() -> int:
    root = pathlib.Path(tempfile.mkdtemp(prefix="mut_86_31_"))
    rows = []
    try:
        build_mini_repo(root)

        print("=" * 78)
        print("CONTROL -- unmutated mini-repo (must be GREEN or the matrix means nothing)")
        print("=" * 78)
        rc, out = run_checker(root)
        tail = [l for l in out.splitlines() if "ALL GREEN" in l or l.startswith("FAILED")]
        print(f"  exit={rc}  {tail[-1].strip() if tail else '(no summary)'}")
        if rc != 0:
            print("\nCONTROL IS RED -- the harness is broken, not the guard. Aborting.")
            print(out[-3000:])
            return 1
        print("  CONTROL GREEN\n")

        for mid, rel, anchor, repl, expect, desc in MUTATIONS:
            if rel == DELETE_WIP:
                wipdir = root / WIP_DIR_REL
                saved = {p.name: p.read_bytes() for p in wipdir.glob("*.md")}
                if not saved:
                    rows.append((mid, "ANCHOR-BAD", desc,
                                 "no live WIP artifacts to remove -- NOT a kill"))
                    print(f"[{mid}] ANCHOR-BAD  {desc}\n      nothing to delete")
                    continue
                for p in list(wipdir.glob("*.md")):
                    p.unlink()
                rc, out = run_checker(root)
                fails = failure_lines(out)
                hit = [f for f in fails if expect in f]
                verdict = ("SURVIVED" if rc == 0
                           else "KILLED" if hit else "SURVIVED-MISATTRIBUTED")
                note = (f"removed {len(saved)} artifact(s); "
                        + ("checker still GREEN -- section [9] asserts nothing when its subject is absent"
                           if rc == 0 else f"{len(fails)} red; named one: {(hit or fails)[0][:80]}"))
                rows.append((mid, verdict, desc, note))
                print(f"[{mid}] {verdict:22s} {desc}\n      {note}")
                for name, data in saved.items():
                    (wipdir / name).write_bytes(data)
                continue

            target = root / rel
            pristine = (REPO / rel).read_text(encoding="utf-8")

            if isinstance(anchor, tuple):
                start, end = anchor
                ns, ne = pristine.count(start), pristine.count(end)
                i, j = pristine.find(start), pristine.find(end)
                bad = (ns != 1 or ne != 1 or not (0 <= i < j))
                if bad:
                    rows.append((mid, "ANCHOR-BAD", desc,
                                 f"region markers in {rel}: start x{ns} @{i}, end x{ne} @{j} "
                                 f"(need exactly 1 each, start BEFORE end) -- NOT a kill"))
                    print(f"[{mid}] ANCHOR-BAD  {desc}\n      start x{ns} @{i}, end x{ne} @{j}")
                    target.write_text(pristine, encoding="utf-8")
                    continue
                mutated = pristine[:i] + repl + pristine[j:]
                removed = j - i
                # A region delete that removes nothing is a non-mutation.
                if removed <= 0:
                    rows.append((mid, "ANCHOR-BAD", desc, "region is empty -- NOT a kill"))
                    print(f"[{mid}] ANCHOR-BAD  {desc}\n      region is empty")
                    target.write_text(pristine, encoding="utf-8")
                    continue
                print(f"      (region delete: {removed} chars)")
            else:
                n = pristine.count(anchor)
                if n != 1:
                    rows.append((mid, "ANCHOR-BAD", desc,
                                 f"anchor occurs {n}x in {rel} (need exactly 1) -- NOT a kill"))
                    print(f"[{mid}] ANCHOR-BAD  {desc}\n      anchor occurs {n}x in {rel}")
                    target.write_text(pristine, encoding="utf-8")
                    continue
                mutated = pristine.replace(anchor, repl, 1)

            target.write_text(mutated, encoding="utf-8")
            rc, out = run_checker(root)
            fails = failure_lines(out)
            hit = [f for f in fails if expect in f]

            if rc == 0:
                verdict = "SURVIVED"
                note = "checker still GREEN -- the assertion this mutation targets is not load-bearing"
            elif hit:
                verdict = "KILLED"
                note = f"{len(fails)} assertion(s) red; named one: {hit[0][:90]}"
            else:
                verdict = "SURVIVED-MISATTRIBUTED"
                note = (f"red, but NOT on {expect!r}; red on: "
                        + "; ".join(f[:60] for f in fails[:2]))

            rows.append((mid, verdict, desc, note))
            print(f"[{mid}] {verdict:22s} {desc}\n      {note}")
            target.write_text(pristine, encoding="utf-8")  # restore the copy

        print("\n" + "=" * 78)
        killed = sum(1 for r in rows if r[1] == "KILLED")
        print(f"MATRIX: {killed}/{len(MUTATIONS)} KILLED")
        for mid, verdict, desc, note in rows:
            if verdict != "KILLED":
                print(f"  !! {mid} {verdict}: {desc}\n     {note}")
        print("This licenses exactly one claim: THESE mutations were killed.")
        print("It licenses no global 'no vacuous guards' claim (Goodenough-Gerhart).")
        return 0 if killed == len(MUTATIONS) else 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
