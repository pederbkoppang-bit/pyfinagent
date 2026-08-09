#!/usr/bin/env python3
"""phase-86.22 -- re-runnable mutation matrix for the shared vocabulary AND its detector.

WHY BOTH ARE MUTATED
--------------------
This step ships two things that can rot independently:

  * the shared vocabulary (`backend/services/recommendation_vocab.py` and the
    six consumers that now call it), and
  * the DETECTOR that proves no seventh consumer has re-forked it
    (`scripts/qa/derive_recommendation_consumers_86_22.py`).

Mutating only the first would leave the detector unmeasured -- and a detector
whose recall is unmeasured is precisely the failure this step exists to close.
The detector already proved that once tonight: its first version was blind to
the substring shape and silently missed `conflict_detector.py` entirely, while
reporting a confident population of 10 across 4 files.

TWO KILL CRITERIA, BECAUSE THERE ARE TWO KINDS OF SUBJECT
---------------------------------------------------------
  VOCAB cells    -- the mutated module is injected into `sys.modules` before
                    pytest imports anything; KILLED iff the test module fails.
  DETECTOR cells -- the mutated detector is written to a TEMPORARY file and its
                    own `validate()` is run; KILLED iff validation reports a
                    recall or precision failure (non-zero).

A detector cell deliberately does NOT run pytest: the test loads the detector
from its real path via `spec_from_file_location`, so a `sys.modules` injection
would not reach it. Adding an env-var override to the test so the harness could
redirect it would be a test-only trapdoor -- the same class of hole as a
`pytest.skip` escape -- so the harness adapts instead of the subject.

SAFETY -- THIS HARNESS NEVER WRITES TO THE REPOSITORY
-----------------------------------------------------
Every mutant is applied in memory or to a tempfile. Both targets are digested
before and after the whole run and the harness fails if either moved.

DISCIPLINE ENCODED HERE
-----------------------
* Every anchor must occur EXACTLY once. A no-match `str.replace` looks exactly
  like success, so 0 or 2+ matches is a hard error, never a skipped cell.
* The mutated source must differ from the original -- asserted, not assumed.
* The un-mutated baseline must be GREEN first; a kill proves nothing against an
  already-red tree.
* A restored run executes last, so the transcript is green on both sides.

USAGE
-----
    source .venv/bin/activate
    python scripts/qa/mutation_matrix_86_22.py

Do NOT run concurrently with another pytest invocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VOCAB = REPO_ROOT / "backend" / "services" / "recommendation_vocab.py"
VOCAB_MOD = "backend.services.recommendation_vocab"
DETECTOR = REPO_ROOT / "scripts" / "qa" / "derive_recommendation_consumers_86_22.py"
TEST_MODULE = ("backend/tests/"
               "test_phase_86_22_outcome_tracker_bias_detector_vocabulary.py")

# ── the matrix ───────────────────────────────────────────────────────────────
# kind: "vocab" (inject + pytest) | "detector" (tempfile + validate())

MUTANTS: list[dict] = [
    {
        "id": "V1", "kind": "vocab",
        "desc": "stop folding the separator (revert to the pre-86.20 behaviour)",
        "proves": "the whole defect -- 'Strong Buy' must not fall back to UNKNOWN",
        "old": '    folded = _SEPARATORS.sub("_", value.strip().upper())',
        "new": '    folded = value.strip().upper()  # MUTANT V1',
    },
    {
        "id": "V2", "kind": "vocab",
        "desc": "fold whitespace only -- hyphen and underscore stop being separators",
        "proves": "'STRONG-BUY' / 'Strong_Buy' are in the parametrised buy set",
        "old": '_SEPARATORS = re.compile(r"[\\s\\-_]+")',
        "new": '_SEPARATORS = re.compile(r"[\\s]+")  # MUTANT V2',
    },
    {
        "id": "V3", "kind": "vocab",
        "desc": "WIDEN is_buy_intent to any recognised value (HOLD becomes a buy)",
        "proves": "the fix must not become an over-permissive gate",
        "old": "    return canonical_recommendation(value) in BUY_INTENT",
        "new": ("    return canonical_recommendation(value) is not None"
                "  # MUTANT V3"),
    },
    {
        "id": "V4", "kind": "vocab",
        "desc": "put HOLD in BUY_INTENT",
        "proves": "a considered HOLD is not a directional call",
        "old": "BUY_INTENT: frozenset[str] = frozenset({STRONG_BUY, BUY})",
        "new": ("BUY_INTENT: frozenset[str] = frozenset({STRONG_BUY, BUY, HOLD})"
                "  # MUTANT V4"),
    },
    {
        "id": "V5", "kind": "vocab",
        "desc": "alias is_sell_intent to the buy set (direction inverted)",
        "proves": "sell must not be graded as buy -- the substring defect's core",
        "old": "    return canonical_recommendation(value) in SELL_INTENT",
        "new": "    return canonical_recommendation(value) in BUY_INTENT  # MUTANT V5",
    },
    {
        "id": "V6", "kind": "vocab",
        "desc": "accept non-strings by coercing with str()",
        "proves": "a dict or enum reaching the gate is a caller bug, not a token",
        "old": "    if not isinstance(value, str):",
        "new": "    if False:  # MUTANT V6",
    },
    {
        "id": "V7", "kind": "vocab",
        "desc": "make is_directional() true for HOLD as well",
        "proves": "'unparseable' and 'considered hold' must stay distinguishable",
        "old": "    return is_buy_intent(value) or is_sell_intent(value)",
        "new": ("    return canonical_recommendation(value) is not None"
                "  # MUTANT V7"),
    },
    {
        "id": "D1", "kind": "detector",
        "desc": "delete rule R3 (the substring shape becomes invisible again)",
        "proves": "recall -- this is the exact blindness that missed conflict_detector",
        "old": '                if _canon(left.value) is not None:',
        "new": '                if False:  # MUTANT D1',
    },
    {
        "id": "D2", "kind": "detector",
        "desc": "delete rule R1 (strong-conviction tokens no longer in scope)",
        "proves": "recall -- R1 is what catches a site with an unhelpful variable name",
        "old": "    if any(c in _STRONG for c in canon):",
        "new": "    if False:  # MUTANT D2",
    },
    {
        "id": "D3", "kind": "detector",
        "desc": "flag EVERY literal membership test (perfect recall, no precision)",
        "proves": "precision -- a detector that flags everything is not a detector",
        "old": '    if "recommend" in tested_src.lower():',
        "new": "    if True:  # MUTANT D3",
    },
    {
        "id": "D4", "kind": "detector",
        "desc": "drop the R2 requirement that the literals be recommendation-shaped",
        "proves": "precision -- `mode in ('fast','slow')` must not enter the population",
        "old": "        if any(c is not None for c in canon):",
        "new": "        if True:  # MUTANT D4",
    },
]

BOOTSTRAP_VOCAB = r"""
import importlib.util, json, os, sys

spec = json.load(open(os.environ["MUT8622_SPEC"], encoding="utf-8"))
target, modname = spec["target"], spec["modname"]
src = open(target, encoding="utf-8").read()

old, new = spec.get("old"), spec.get("new")
if old is not None:
    n = src.count(old)
    if n != 1:
        print(f"ANCHOR-ERROR: occurrences={n} (must be exactly 1)", file=sys.stderr)
        raise SystemExit(97)
    mutated = src.replace(old, new)
    if mutated == src:
        print("ANCHOR-ERROR: mutation was a no-op", file=sys.stderr)
        raise SystemExit(97)
    src = mutated

modspec = importlib.util.spec_from_loader(modname, loader=None, origin=target)
mod = importlib.util.module_from_spec(modspec)
mod.__file__ = target
sys.modules[modname] = mod
exec(compile(src, target, "exec"), mod.__dict__)

import pytest
raise SystemExit(pytest.main(["-q", "-p", "no:randomly", spec["test_module"]]))
"""

BOOTSTRAP_DETECTOR = r"""
import importlib.util, json, os, sys

spec = json.load(open(os.environ["MUT8622_SPEC"], encoding="utf-8"))
src = open(spec["target"], encoding="utf-8").read()
old, new = spec["old"], spec["new"]
n = src.count(old)
if n != 1:
    print(f"ANCHOR-ERROR: occurrences={n} (must be exactly 1)", file=sys.stderr)
    raise SystemExit(97)
mutated = src.replace(old, new)
if mutated == src:
    print("ANCHOR-ERROR: mutation was a no-op", file=sys.stderr)
    raise SystemExit(97)

import tempfile, pathlib
tmp = pathlib.Path(tempfile.mkdtemp()) / "mutated_detector.py"
tmp.write_text(mutated, encoding="utf-8")

ms = importlib.util.spec_from_file_location("mutated_detector", tmp)
mod = importlib.util.module_from_spec(ms)
ms.loader.exec_module(mod)

missed = [n for n, s in mod.KNOWN_POSITIVES if not mod.scan_source(s, "<f>")]
false_pos = [n for n, s in mod.KNOWN_NEGATIVES if mod.scan_source(s, "<f>")]
print(f"recall misses : {missed}")
print(f"false positives: {false_pos}")
raise SystemExit(1 if (missed or false_pos) else 0)
"""


def digest(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def run_cell(cell: dict | None, label: str) -> tuple[bool, str]:
    """Return (tests_passed, tail_of_output)."""
    kind = (cell or {}).get("kind", "vocab")
    target = DETECTOR if kind == "detector" else VOCAB
    spec = {
        "target": str(target),
        "modname": VOCAB_MOD,
        "test_module": TEST_MODULE,
        "old": (cell or {}).get("old"),
        "new": (cell or {}).get("new"),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(spec, fh)
        spec_path = fh.name
    env = dict(os.environ, MUT8622_SPEC=spec_path, PYTHONPATH=str(REPO_ROOT))
    boot = BOOTSTRAP_DETECTOR if (cell and kind == "detector") else BOOTSTRAP_VOCAB
    proc = subprocess.run([sys.executable, "-c", boot], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=900)
    os.unlink(spec_path)
    out = (proc.stdout + proc.stderr).strip().splitlines()
    if proc.returncode == 97:
        raise SystemExit(f"{label}: ANCHOR ERROR -- {out[-1] if out else '?'}")
    return proc.returncode == 0, "\n".join(out[-3:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-clean", action="store_true", default=True)
    ap.parse_args()

    before = {p: digest(p) for p in (VOCAB, DETECTOR)}
    print("phase-86.22 mutation matrix")
    for p, d in before.items():
        print(f"  target {p.relative_to(REPO_ROOT)}  md5={d}")
    print()

    ok, tail = run_cell(None, "baseline")
    print(f"BASELINE (un-mutated): {'GREEN' if ok else 'RED'}\n  {tail}\n")
    if not ok:
        print("Baseline is RED -- every kill below would be meaningless. Aborting.")
        return 1

    killed = survived = 0
    print(f"{'id':<5}{'kind':<10}{'result':<10}mutation")
    print("-" * 100)
    for cell in MUTANTS:
        passed, tail = run_cell(cell, cell["id"])
        if passed:
            survived += 1
            verdict = "SURVIVED"
        else:
            killed += 1
            verdict = "killed"
        print(f"{cell['id']:<5}{cell['kind']:<10}{verdict:<10}{cell['desc']}")
        print(f"{'':15}proves: {cell['proves']}")
        print(f"{'':15}{tail.splitlines()[-1] if tail else ''}")

    ok, tail = run_cell(None, "restored")
    print(f"\nRESTORED (un-mutated): {'GREEN' if ok else 'RED'}\n  {tail}")

    after = {p: digest(p) for p in (VOCAB, DETECTOR)}
    for p in (VOCAB, DETECTOR):
        same = before[p] == after[p]
        print(f"  {p.relative_to(REPO_ROOT)} unchanged: {same} ({after[p]})")
        if not same:
            print("  !! TARGET MODIFIED ON DISK -- this harness must never do that.")
            return 1

    print(f"\n{killed} killed / {survived} survived of {len(MUTANTS)}")
    if survived:
        print("A SURVIVING mutant means a guard in this matrix cannot fail.")
        return 1
    print("Every guard IN THIS MATRIX can fail. That is the scope of this claim:")
    print("it says nothing about guards the matrix does not mutate.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
