#!/usr/bin/env python3
"""phase-90.9 -- mutation matrix for the criterion-shape classifier.

    python3 scripts/qa/mutation_matrix_90_9.py --verify   # exit 1 on a survivor

CONTROL OBSERVED GREEN FIRST. Every cell runs the REAL
`scripts/qa/criteria_shape_90_9.py --verify` as a SUBPROCESS, against a mutated
copy in a SANDBOX. Nothing in the repository is written by any cell, and that is
enforced rather than promised:

  * the sandbox gets its OWN COPIES of `.claude/masterplan.json` and
    `handoff/verdict_ledger.jsonl`, never symlinks to them, so a mutant that turns
    a read into a write hits the copy;
  * `.git` IS symlinked, because `git show <rev>:<path>` is the corpus pin and is
    read-only; no cell touches git;
  * a CONTAINMENT GUARD refuses to run at all if any sandbox path resolves inside
    the repository -- 90.1 cycle 4 found that ONE deleted redirect line stood
    between a self-test and the project's verdict history, and the test reported
    PASSED while truncating it;
  * the real masterplan and verdict-ledger md5s are captured before and after the
    whole run and compared.

THE ERROR DISCRIMINATOR, AND WHY IT IS TYPED
--------------------------------------------
"A mutant that fails to run scores ERROR and never counts as a kill." Two lessons
are built into the implementation, both paid for in real cycles:

  90.1 cycle 4 -- "fails to run" is not "fails to PARSE". A mutant can parse and
  still not run. So a cell is driven, not just compiled, and the drive includes an
  IMPORT.

  90.1 cycle 4 again -- an over-eager probe is as bad as a blind one. Scoring "any
  traceback" as ERROR flagged a legitimate cell whose entire purpose was to raise
  AssertionError, silently DELETING it from the matrix. So the discriminator reads
  the exception TYPE: a name-resolution failure means the code is not there; a
  domain exception means it ran and misbehaved.

  90.1 cycle 5 -- the type must be read from the MESSAGE, not from the shape of a
  traceback. A NameError raised inside a fail-open `except Exception` handler
  prints one line and no traceback at all, and a discriminator keyed on the
  literal "Traceback (most recent call last)" scored three such mutants KILLED
  when the criterion required ERROR. That FAILED step 90.1. The check below is
  keyed on the type name wherever it appears, and a unit cell proves it fires on a
  swallowed one-liner with no traceback.

A NULL control and a REAL-KILL control are drilled on EVERY run, so neither an
always-kill nor an always-survive harness can go unnoticed.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUBJECT = REPO / "scripts" / "qa" / "criteria_shape_90_9.py"
MASTERPLAN = REPO / ".claude" / "masterplan.json"
VERDICT_LEDGER = REPO / "handoff" / "verdict_ledger.jsonl"

UNRESOLVABLE = ("NameError", "ModuleNotFoundError", "ImportError",
                "AttributeError", "UnboundLocalError")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else "ABSENT"


#: phase-90.9 cycle-1 Q/A: Python 3.13+ COLORIZES tracebacks, and the agent runtime
#: injects FORCE_COLOR=3, so a subprocess stderr carries
#: '\x1b[1;35mNameError\x1b[0m: ' and a literal `"NameError:" in stderr` NEVER
#: matches. Measured: the same command exits 0 on system python3 3.9.6 and 1 under
#: the project venv (3.14.4), and the direction is FAIL-DANGEROUS -- it turns ERROR
#: into KILLED and INFLATES the kill count. Strip the formatting before reading the
#: type, and drive with colour disabled as well; belt and braces, because either one
#: alone is a bet on an environment.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def decolorize(s: str) -> str:
    return _ANSI.sub("", s or "")


def score_error(observed: str) -> str | None:
    """The exception TYPE, if the drive failed to RESOLVE A NAME. Keyed on the type
    name wherever it appears -- a traceback is not required, because a fail-open
    handler emits one line and no traceback (the 90.1 cycle-5 finding)."""
    for t in UNRESOLVABLE:
        if f"{t}:" in decolorize(observed):
            return t
    return None


def build_sandbox(src_text: str) -> Path:
    """A sandbox whose plan and ledger are COPIES. Refuses to hand back a path
    that resolves inside the repository."""
    root = Path(tempfile.mkdtemp(prefix="mm909_"))
    (root / "scripts" / "qa").mkdir(parents=True)
    (root / ".claude").mkdir()
    (root / "handoff").mkdir()
    (root / "scripts" / "qa" / "criteria_shape_90_9.py").write_text(src_text, encoding="utf-8")
    shutil.copy2(MASTERPLAN, root / ".claude" / "masterplan.json")
    shutil.copy2(VERDICT_LEDGER, root / "handoff" / "verdict_ledger.jsonl")
    (root / ".git").symlink_to(REPO / ".git")
    # CONTAINMENT GUARD -- fail loudly, never quietly proceed.
    for p in (root, root / ".claude" / "masterplan.json",
              root / "handoff" / "verdict_ledger.jsonl"):
        rp = p.resolve()
        if rp == REPO or REPO in rp.parents:
            raise SystemExit(
                f"CONTAINMENT GUARD: sandbox path {rp} resolves inside the repository "
                f"({REPO}). Refusing to run -- a mutation matrix that can reach the "
                f"real plan of record is the failure 90.1 cycle 4 found.")
    return root


def drive(root: Path) -> tuple[int, str]:
    out = subprocess.run([sys.executable, "scripts/qa/criteria_shape_90_9.py", "--verify"],
                         cwd=root, capture_output=True, text=True, timeout=600,
                         env={**os.environ, "NO_COLOR": "1", "PYTHON_COLORS": "0",
                              "FORCE_COLOR": ""})
    return out.returncode, (out.stdout + "\n" + out.stderr)


# --- cells -----------------------------------------------------------------

def _sub(old: str, new: str):
    def apply(s: str) -> str:
        if old not in s:
            raise SystemExit(f"ANCHOR MISSING, cell would be a no-op: {old[:70]!r}")
        if s.count(old) != 1:
            raise SystemExit(f"ANCHOR NOT UNIQUE ({s.count(old)}x): {old[:70]!r}")
        return s.replace(old, new)
    return apply


CELLS = [
    ("N0", "SURVIVED",
     "NULL MUTANT (comment only). If this scores KILLED the harness is broken and "
     "every other kill this run is meaningless.",
     _sub("RULE_NAME = \"MID\"", "RULE_NAME = \"MID\"  # null mutant")),
    ("M1", "KILLED",
     "criterion 2, NAMED: classify() labels EVERY criterion evidence-class. The "
     "all-product-behaviour control must kill this.",
     _sub("    return \"PRODUCT_BEHAVIOUR\"", "    return \"EVIDENCE_APPARATUS\"")),
    ("M2", "KILLED",
     "classify() labels every criterion product-behaviour -- the opposite constant. "
     "The all-product control alone CANNOT catch this, which is why a discriminating "
     "check exists.",
     _sub("            return \"EVIDENCE_APPARATUS\"\n    return \"PRODUCT_BEHAVIOUR\"",
          "            return \"PRODUCT_BEHAVIOUR\"\n    return \"PRODUCT_BEHAVIOUR\"")),
    ("M3", "KILLED",
     "the unbounded detector never fires, so the one condition this tool exits "
     "non-zero on becomes unreachable.",
     _sub("def unbounded_reason(criterion: str) -> str | None:",
          "def unbounded_reason(criterion: str) -> str | None:\n    return None")),
    ("M4", "KILLED",
     "the unbounded detector always fires, so every step is flagged and the gate "
     "stops discriminating.",
     _sub("def is_unbounded(criterion: str) -> bool:\n    return unbounded_reason(criterion) is not None",
          "def is_unbounded(criterion: str) -> bool:\n    return True")),
    ("M5", "KILLED",
     "the numeric-bound escape is removed, so an enumerated population "
     "('all 155 steps') is misread as unbounded.",
     _sub("        if _NUMERAL.search(clause):\n            continue  # an enumerated population is finite by construction\n", "")),
    ("M6", "KILLED",
     "the corpus-population check is removed, so quantifying over a fixed corpus "
     "('every attempt row') is misread as growing with the work.",
     _sub("        if cm and cm.start(2) < m.start(2):\n            continue  # the quantifier governs a corpus population, not the artifact\n", "")),
    ("M7", "KILLED",
     "criterion 1: the step-inclusion rule stops excluding step 90.9 itself, so the "
     "one figure that DOES reproduce stops reproducing.",
     _sub('def collect_steps(plan: dict, exclude: str = "90.9") -> list[dict]:',
          'def collect_steps(plan: dict, exclude: str = "__none__") -> list[dict]:')),
    ("M8", "KILLED",
     "criterion 4: the AST write-scan returns nothing, so 'no write path' becomes "
     "vacuously true.",
     _sub("def write_capable_calls(src: str) -> list[str]:",
          "def write_capable_calls(src: str) -> list[str]:\n    return []")),
    ("M9", "KILLED",
     "criterion 7: the consequence scan returns nothing, so 'never reads a verdict "
     "history' becomes vacuously true.",
     _sub("def classifier_consequence_refs(src: str, fns=None) -> list[str]:",
          "def classifier_consequence_refs(src: str, fns=None) -> list[str]:\n    return []")),
    ("M10", "KILLED",
     "criteria 4 and 5: sha256 returns a constant, so every byte-identity check "
     "passes while proving nothing.",
     _sub("def sha256(p: Path) -> str:",
          "def sha256(p: Path) -> str:\n    return \"constant\"")),
    ("QX", "ERROR",
     "ERROR CONTROL 1: a CALL SITE is renamed. The code parses, imports, and then "
     "cannot RESOLVE A NAME at run time. It must score ERROR, never a kill.",
     _sub("    if ns.verify:\n        return verify()",
          "    if ns.verify:\n        return verify_v2()")),
    ("QXI", "ERROR",
     "ERROR CONTROL 2: a MODULE-SCOPE import that does not exist. It parses "
     "cleanly and dies on IMPORT -- 'fails to run' is not 'fails to parse'.",
     _sub("import hashlib", "import hashlib\nimport nonexistent_module_90_9")),
]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    verify = "--verify" in argv

    mp_before, vl_before = md5(MASTERPLAN), md5(VERDICT_LEDGER)
    src = SUBJECT.read_text(encoding="utf-8")

    print("=" * 78)
    print("CONTROL (the real, unmutated classifier)")
    print("=" * 78)
    root = build_sandbox(src)
    rc, obs = drive(root)
    print(f"  exit={rc}   failed-check lines: "
          f"{sum(1 for ln in obs.splitlines() if ln.strip().startswith('[FAIL]'))}")
    if rc != 0:
        print("  CONTROL IS NOT GREEN. Every kill below would be meaningless; stopping.")
        for ln in obs.splitlines():
            if "[FAIL]" in ln:
                print("   ", ln.strip())
        return 1
    print("  CONTROL GREEN")

    print("\n" + "=" * 78)
    print("THE ERROR DISCRIMINATOR IS TYPED, AND IT FIRES WITHOUT A TRACEBACK")
    print("=" * 78)
    swallowed = ("[some-gate] INTERNAL ERROR -- NameError: name 'handle_x' is not "
                 "defined -- failing OPEN")
    domain = "Traceback (most recent call last):\n  ...\nAssertionError: guard tripped"
    probe_ok = (score_error(swallowed) == "NameError") and (score_error(domain) is None)
    print(f"  a SWALLOWED one-liner with no traceback  -> {score_error(swallowed)}")
    print(f"  a real traceback ending in AssertionError -> {score_error(domain)}")
    print(f"  {'ok' if probe_ok else 'BAD'}: the discriminator reads the TYPE, not the shape")

    print("\n" + "=" * 78)
    print("CELLS")
    print("=" * 78)
    rows = []
    for cid, want, why, apply in CELLS:
        mutated = apply(src)
        r = build_sandbox(mutated)
        code, observed = drive(r)
        err = score_error(observed)
        got = "ERROR" if err else ("KILLED" if code != 0 else "SURVIVED")
        rows.append((cid, got, want, why, err or ""))

    bad = 0
    for cid, got, want, why, err in rows:
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BAD '} {cid:<4} {got:<9} expected {want:<9} {err}")
        print(f"         {why}")

    mp_after, vl_after = md5(MASTERPLAN), md5(VERDICT_LEDGER)
    print("\n" + "=" * 78)
    print("CONTAINMENT")
    print("=" * 78)
    print(f"  .claude/masterplan.json md5      {mp_before} -> {mp_after}")
    print(f"  handoff/verdict_ledger.jsonl md5 {vl_before} -> {vl_after}")
    contained = (mp_before == mp_after) and (vl_before == vl_after)
    print(f"  real tree untouched: {contained}")

    nulls = [r for r in rows if r[0] == "N0"]
    kills = [r for r in rows if r[0] == "M1"]
    errs = [r for r in rows if r[0] in ("QX", "QXI")]
    n_killed = sum(1 for r in rows if r[1] == "KILLED")
    n_surv = sum(1 for r in rows if r[1] == "SURVIVED" and r[0] != "N0")
    n_err = sum(1 for r in rows if r[1] == "ERROR")

    print("\n" + "=" * 78)
    print(f"KILLED {n_killed} | SURVIVED {n_surv} (excl. N0) | ERROR {n_err} | "
          f"null mutant survived: {bool(nulls) and nulls[0][1] == 'SURVIVED'} | "
          f"real-kill control killed: {bool(kills) and kills[0][1] == 'KILLED'} | "
          f"error controls: {[r[1] for r in errs]}")
    print("=" * 78)

    if not verify:
        return 0
    problems = bad + (0 if contained else 1) + (0 if probe_ok else 1)
    if problems:
        print(f"  {problems} problem(s): {bad} unexpected cell score(s)"
              + ("" if contained else ", CONTAINMENT BREACHED")
              + ("" if probe_ok else ", ERROR PROBE DOES NOT DISCRIMINATE"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
