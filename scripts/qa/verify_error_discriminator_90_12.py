#!/usr/bin/env python3
"""phase-90.12 -- re-runnable proof that the ERROR discriminator can FIRE.

    python3 scripts/qa/verify_error_discriminator_90_12.py --self-test

Exit 0 iff every check passes AND the cardinality floor is met.

THE DEFECT THIS CLOSES
----------------------
`mutation_matrix_90_1._drive_unresolvable` (formerly `_drive_traceback`) decided
"this mutant could not run" by requiring the literal string
"Traceback (most recent call last)" on a drive's stderr.
`attempt_gate.handle_hook` ends in a blanket `except Exception` that prints ONE
LINE -- "[attempt-gate] INTERNAL ERROR -- NameError: ... -- failing OPEN" -- and
returns 0. That handler is correct and must stay. But it means NO failure raised
inside the hook's try block ever produces a traceback, so the scan returned None
for the whole class and those mutants scored KILLED where step 90.1's criterion 5
clause 3 requires ERROR. The step 90.1 cycle-5 Q/A FAILED the step on it.

The harm is not nominal: a call-site rename defeats NO guard yet fails many
checks, so a build that never runs green-washes several criteria at once.

RED-FIRST IS STRUCTURAL HERE, NOT NARRATED
------------------------------------------
The "before" discriminator is not re-typed from memory -- it is EXTRACTED FROM
GIT at the commit that carried it, so the red half is the real prior code. Each
mutant is observed ONCE and then scored by BOTH discriminators, so the pair is a
true differential on identical evidence rather than two runs that might differ
for some other reason.

WHAT KEEPS IT FROM BECOMING OVER-EAGER
--------------------------------------
Scoring "any exception" as ERROR silently DELETES legitimate cells from a matrix
-- that already happened once, during 90.1 cycle 4, to a cell whose entire
purpose was to reintroduce a bug raising AssertionError. So a DOMAIN exception
must still score KILLED, including when the same fail-open handler formats it
into the same one-line shape. That case is drilled here, in both directions.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MATRIX = REPO / "scripts" / "qa" / "mutation_matrix_90_1.py"
GATE = REPO / "scripts" / "harness" / "attempt_gate.py"
VERDICT_LEDGER = REPO / "handoff" / "verdict_ledger.jsonl"

#: The commit whose `_drive_traceback` is the RED baseline. Chosen because it is
#: the tree the 90.1 cycle-5 Q/A actually evaluated.
PRE_FIX_REV = "d564ad58"

_DIRTY_AT_START: set = set()

EXPECTED_CHECKS = 24

results: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    results.append((label, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))


def section(t: str) -> None:
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "ABSENT"


def load_matrix():
    spec = importlib.util.spec_from_file_location("mm901", MATRIX)
    m = importlib.util.module_from_spec(spec)
    sys.modules["mm901"] = m
    spec.loader.exec_module(m)
    return m


def extract_pre_fix_discriminator():
    """The REAL prior implementation, lifted from git -- never re-typed."""
    out = subprocess.run(["git", "show", f"{PRE_FIX_REV}:scripts/qa/mutation_matrix_90_1.py"],
                         cwd=REPO, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"git show {PRE_FIX_REV} failed: {out.stderr[:200]}")
    src = out.stdout
    start = src.index("def _drive_traceback(")
    end = src.index("def observations(", start)
    body = src[start:end]
    ns: dict = {"UNRESOLVABLE_ERRORS": ("ModuleNotFoundError", "ImportError",
                                        "NameError", "AttributeError")}
    exec(compile(body, "<pre-fix>", "exec"), ns)  # noqa: S102
    return ns["_drive_traceback"], body


#: (id, description, apply). Each `apply` mutates attempt_gate.py source.
def _rename_call_site(name: str, occurrence: int = 0):
    """Rename ONE CALL SITE, never the definition -- a realistic authoring slip."""
    def apply(src: str) -> str:
        lines = src.splitlines(True)
        seen = 0
        for i, ln in enumerate(lines):
            if f"{name}(" in ln and not ln.lstrip().startswith("def "):
                if seen == occurrence:
                    lines[i] = ln.replace(f"{name}(", f"{name}_v2(")
                    return "".join(lines)
                seen += 1
        raise SystemExit(f"ANCHOR MISSING: no call site #{occurrence} of {name}")
    return apply


def _rename_def(name: str):
    def apply(src: str) -> str:
        a = f"def {name}("
        if src.count(a) != 1:
            raise SystemExit(f"ANCHOR NOT UNIQUE: {a}")
        return src.replace(a, f"def {name}_v2(")
    return apply


def _plant_unbound_local(src: str) -> str:
    """A binding used BEFORE assignment inside the hook -- raises UnboundLocalError.

    The research gate's finding, and it was a live blind spot: UnboundLocalError
    SUBCLASSES NameError, but the printed name is "UnboundLocalError:", which does
    not contain the substring "NameError". This scan matches type names as STRINGS,
    so subclass relationships do not carry, and the mutant was scored KILLED.
    """
    a = "        claim = extract_step_id_claim(tool_input)"
    if src.count(a) != 1:
        raise SystemExit(f"ANCHOR NOT UNIQUE: {a}")
    return src.replace(a, "        _later = _later + 1\n        _later = 0\n" + a)


def _plant_domain_error(src: str) -> str:
    """A DOMAIN exception raised INSIDE the hook's try block, so it reaches
    stderr through the SAME fail-open handler as a NameError. This is the cell
    that must stay a KILL: if the discriminator keys on the handler's shape
    rather than the exception TYPE, it deletes this one."""
    a = "        claim = extract_step_id_claim(tool_input)"
    if src.count(a) != 1:
        raise SystemExit(f"ANCHOR NOT UNIQUE: {a}")
    return src.replace(a, a + "\n        assert False, 'planted domain failure'")


CELLS = [
    ("QA1", "call-site rename: read_ledger -> read_ledger_v2", "ERROR",
     _rename_call_site("read_ledger")),
    ("QA1b", "call-site rename: extract_step_id_claim -> ..._v2", "ERROR",
     _rename_call_site("extract_step_id_claim")),
    ("QA1c", "call-site rename: extract_step_id -> ..._v2", "ERROR",
     _rename_call_site("extract_step_id")),
    ("QX2", "DEFINITION rename: def handle_hook -> handle_hook_v2 (the control "
            "the pre-fix scan already caught)", "ERROR", _rename_def("handle_hook")),
    ("UBL", "a binding used before assignment -- UnboundLocalError, whose printed "
            "name does NOT contain 'NameError' despite subclassing it", "ERROR",
     _plant_unbound_local),
    ("DOM", "a DOMAIN exception (AssertionError) raised inside the hook's try "
            "block, reaching stderr through the SAME fail-open handler", "KILL",
     _plant_domain_error),
    ("N0", "NULL MUTANT (comment only)", "KILL_NONE",
     lambda s: s.replace("def handle_hook(", "def handle_hook(  # null mutant\n", 1)
     if False else s + "\n# null mutant\n"),
]


def observe(mm, mutated_src: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        for f in (GATE, GATE.parent / "attempt_budget.py",
                  GATE.parent / "attempt_outcomes.py"):
            shutil.copy2(f, work / f.name)
        (work / GATE.name).write_text(mutated_src, encoding="utf-8")
        return mm.observations(work / GATE.name)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" not in argv:
        print(__doc__.split("\n\n")[0])
        return 0

    vl_before = sha256(VERDICT_LEDGER)
    global _DIRTY_AT_START
    _DIRTY_AT_START = {ln.split(" -> ")[-1].strip().strip('"')
                       for ln in (l[3:] for l in subprocess.run(
                           ["git", "status", "--porcelain", "--", "scripts/harness",
                            "scripts/qa"], cwd=REPO, capture_output=True,
                           text=True).stdout.splitlines() if l.strip())}
    gate_md5_before = hashlib.md5(GATE.read_bytes()).hexdigest()
    mm = load_matrix()
    pre_fix, pre_fix_src = extract_pre_fix_discriminator()
    new = mm._drive_unresolvable
    src = GATE.read_text(encoding="utf-8")

    section("A. THE RED BASELINE IS THE REAL PRIOR CODE, NOT A RETYPING")
    check("the pre-fix discriminator was extracted from git, not re-typed",
          "Traceback (most recent call last)" in pre_fix_src
          and "def _drive_traceback(" in pre_fix_src,
          f"{PRE_FIX_REV}, {len(pre_fix_src)} chars")
    check("...and it is the TRACEBACK-ONLY implementation, which is what made it "
          "blind", "INTERNAL ERROR" not in pre_fix_src)
    check("the shipped discriminator is a DIFFERENT function, renamed to say what "
          "it does", callable(new) and new.__name__ == "_drive_unresolvable",
          getattr(new, "__name__", "?"))

    section("B. THE DIFFERENTIAL -- one observation per mutant, scored BOTH ways")
    print(f"  {'cell':<6} {'BEFORE (traceback-only)':<26} {'AFTER (typed)':<26} expected")
    rows = []
    for cid, desc, want, apply in CELLS:
        mutated = src if cid == "N0" else apply(src)
        obs = observe(mm, mutated)
        old_r = pre_fix(obs)
        new_r = new(obs)
        old_s = "ERROR" if old_r else "not-ERROR"
        new_s = "ERROR" if new_r else "not-ERROR"
        rows.append((cid, desc, want, old_s, new_s, new_r or old_r or ""))
        print(f"  {cid:<6} {old_s:<26} {new_s:<26} {want}")
    by = {r[0]: r for r in rows}

    section("C. CRITERION 2 -- CALL-SITE renames score ERROR, and the DEFINITION "
            "control still does")
    check("UBL scores ERROR -- UnboundLocalError subclasses NameError but its printed "
          "name does not contain that substring, and this scan matches type names as "
          "STRINGS, so the subclass relationship does not carry (research gate 90.12)",
          by["UBL"][4] == "ERROR", by["UBL"][5][:90])
    check("...and it scored NOT-ERROR before the type list was widened, so the gap was "
          "real rather than theoretical", by["UBL"][3] == "not-ERROR")
    check("TypeError is DELIBERATELY absent from the type list: cosmic-ray issue #310 is "
          "this defect in the wild in reverse, where a TypeError -- a legitimate domain "
          "error -- was classed non-viable and the mutant mis-scored. A false positive "
          "here silently DELETES a cell",
          "TypeError" not in __import__("mm901").UNRESOLVABLE_ERRORS)

    for cid in ("QA1", "QA1b", "QA1c"):
        r = by[cid]
        check(f"{cid} scores ERROR after the fix -- {r[1]}", r[4] == "ERROR", r[5][:90])
        check(f"...and scored NOT-ERROR before it, so the cell is RED-FIRST rather "
              f"than already covered ({cid})", r[3] == "not-ERROR")
    # CORRECTED after the 90.9 cycle-1 Q/A. This check used to assert that QX2 scored
    # ERROR in BOTH columns, on the reasoning that the pre-fix scan already caught the
    # definition-rename sub-class. That was measured on system python3 3.9.6. Under the
    # project venv (3.14.4) with the runtime's FORCE_COLOR, the pre-fix scan is blind to
    # QX2 as well -- its `tail.startswith("NameError")` cannot match a tail that begins
    # with an ANSI escape. So the BEFORE column is INTERPRETER-DEPENDENT and asserting a
    # value for it was asserting a property of my shell. What must hold, and does, is
    # the AFTER column.
    check("QX2 (definition rename) scores ERROR AFTER the fix",
          by["QX2"][4] == "ERROR", by["QX2"][5][:80])
    check("...and its BEFORE value is reported, not asserted -- it depends on whether "
          "the interpreter colorizes tracebacks, which is exactly the defect the 90.9 "
          "Q/A found in the sibling matrices",
          by["QX2"][3] in ("ERROR", "not-ERROR"), "BEFORE=" + by["QX2"][3])

    section("D. CRITERION 3 -- IT STILL DISCRIMINATES (the over-eager failure mode)")
    check("a DOMAIN exception through the SAME fail-open handler is NOT scored ERROR "
          "-- it stays a KILL", by["DOM"][4] == "not-ERROR",
          "AssertionError via '[attempt-gate] INTERNAL ERROR -- ...'")
    check("...and the drive really did exercise that handler, so the check is not "
          "vacuous",
          any("INTERNAL ERROR" in (v or {}).get("stderr", "")
              for v in observe(mm, _plant_domain_error(src)).values()
              if isinstance(v, dict)))
    check("the NULL mutant is NOT scored ERROR", by["N0"][4] == "not-ERROR")

    section("E. CRITERION 5 -- NO SILENT CELL LOSS IN THE SHIPPED MATRIX")
    r = subprocess.run([sys.executable, str(MATRIX), "--verify"],
                       cwd=REPO, capture_output=True, text=True, timeout=1800)
    # id -> score, NOT score -> id. The first version wrote `dict(findall(...))`
    # over (score, id) pairs, which collapses 16 cells onto 3 keys and reported
    # "2 cells scored". A roster keyed by the wrong element is not a roster.
    scores = {cid: sc for sc, cid in
              re.findall(r"^\s+(SURVIVED|KILLED|ERROR)\s+(\S+)", r.stdout, re.M)}
    tally = re.search(r"KILLED (\d+) \| SURVIVED (\d+) \(excl\. N0\) \| ERROR (\d+) \| "
                      r"null mutant survived: (\w+)", r.stdout)
    print(f"  shipped matrix: exit={r.returncode}  {tally.group(0) if tally else 'NO TALLY'}")
    check("the shipped matrix still exits 0", r.returncode == 0)
    check("its tally is UNCHANGED by this fix -- 15 KILLED / 0 SURVIVED / 0 ERROR, "
          "null survived. Measured by me BEFORE the edit and again after; no shipped "
          "cell changed score, so no cell was silently deleted",
          bool(tally) and tally.groups() == ("15", "0", "0", "True"),
          tally.group(0) if tally else "NO TALLY")
    check("...and the cell roster is non-empty, so the tally is not vacuous",
          len(scores) >= 15, f"{len(scores)} cells scored")

    section("F. CRITERION 6 + CONTAINMENT")
    vl_after = sha256(VERDICT_LEDGER)
    check("handoff/verdict_ledger.jsonl sha256 byte-identical before and after",
          vl_before == vl_after, f"{vl_before[:16]} -> {vl_after[:16]}")
    check("the real attempt_gate.py is byte-identical -- every mutant ran from a "
          "temp copy", hashlib.md5(GATE.read_bytes()).hexdigest() == gate_md5_before)
    # NOT a source grep for write verbs: that probe would match its own literal
    # list, which is the self-referential vacuity that broke the first version of
    # the equivalent check in criteria_shape_90_9.py. Measured empirically instead.
    # COMPARE THE RUN AGAINST ITSELF, not against a hardcoded list. The first version
    # pinned an expected-dirty set, which went red the moment this session legitimately
    # edited a different file -- so it was measuring "has anything changed today",
    # not "did THIS RUN write anything". The baseline is captured at the top of main().
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            "scripts/harness", "scripts/qa"],
                           cwd=REPO, capture_output=True, text=True).stdout
    expected_dirty = _DIRTY_AT_START
    # NOT `stdout.strip()` then `ln[3:]`: stripping the whole blob removes the
    # leading space of the FIRST porcelain line only, so a fixed-width slice eats
    # one character of exactly one path and reports 'cripts/qa/...'. Parse each
    # line on its own terms instead of assuming a column.
    touched = {ln.split(" -> ")[-1].strip().strip('"')
               for ln in (l[3:] for l in dirty.splitlines() if l.strip())}
    check("this run wrote nothing under scripts/ that was not already this step's "
          "own edit -- measured with git, not by grepping my own source for write "
          "verbs (that probe matches its own list)",
          touched <= expected_dirty, ", ".join(sorted(touched - expected_dirty)) or "none")
    check("...and the three subject modules are individually unmodified in the tree",
          not subprocess.run(["git", "diff", "--quiet", "--", "scripts/harness"],
                             cwd=REPO).returncode)

    section("SUMMARY")
    failed = [x for x in results if not x[1]]
    print(f"  checks run: {len(results)} (floor {EXPECTED_CHECKS})")
    print(f"  failed:     {len(failed)}")
    for label, _, detail in failed:
        print(f"    FAIL {label} {detail}")
    if len(results) < EXPECTED_CHECKS:
        print(f"  CARDINALITY FLOOR NOT MET: {len(results)} < {EXPECTED_CHECKS}. "
              "A checker whose loop covers nothing exits 0 and looks like success.")
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
