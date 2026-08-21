#!/usr/bin/env python3
"""phase-90.3 -- mutation matrix for the progress digest.

    python3 scripts/qa/mutation_matrix_90_3.py --verify   # exit 1 on a survivor

CONTROL OBSERVED GREEN FIRST. Every cell mutates a COPY of
`scripts/harness/attempt_gate.py` in a sandbox and drives the real
`--self-test` as a subprocess. The repository is never written.

WHAT THIS MECHANISM IS, so no cell is read as claiming more than it does: a
byte-digest is the WEAKEST of the three published stagnation signals -- CUDABeaver
measures SHA-256 duplicate_code at 0-50.8% and code_cycle at 0.7-3.8%, against
SEMANTIC no_progress at 44.6-84.6%. It detects an exact repeat and an A->B->A
oscillation, and NOTHING ELSE. Criterion 5 forbids any consumer from reading a
changed digest as progress; optimal-stopping work triggers on an ABSOLUTE score,
explicitly not on inter-iteration change.

THE CELL THAT MATTERS MOST is D1: the research gate measured, before any of this
existed, that criterion 1's file set resolves to files the GATE ITSELF writes on
every launch. Without the exclusion the digest advances by construction and the
whole check is vacuous -- 89.1's defect through a different door. D1 removes the
exclusion and must be KILLED.

ERROR discipline is inherited from phase-90.12: a mutant that cannot RESOLVE A
NAME scores ERROR and is never a kill, and the type is read from the message as
well as from a traceback, because a fail-open handler emits one line and none.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GATE = REPO / "scripts" / "harness" / "attempt_gate.py"
MASTERPLAN = REPO / ".claude" / "masterplan.json"
VERDICT_LEDGER = REPO / "handoff" / "verdict_ledger.jsonl"

UNRESOLVABLE = ("NameError", "ModuleNotFoundError", "ImportError",
                "UnboundLocalError", "AttributeError")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else "ABSENT"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "ABSENT"


def score_error(observed: str) -> str | None:
    """The exception TYPE if the drive failed to RESOLVE A NAME (phase-90.12).

    Keyed on the type name wherever it appears -- a traceback is not required,
    because a fail-open handler prints one line and no traceback.
    """
    for t in UNRESOLVABLE:
        if f"{t}:" in observed:
            return t
    return None


def sandbox(src: str) -> Path:
    """A sandbox with COPIES, never symlinks, so a mutant that turns a read into
    a write hits the copy. Refuses to hand back a path inside the repository."""
    root = Path(tempfile.mkdtemp(prefix="mm903_"))
    (root / "scripts" / "harness").mkdir(parents=True)
    (root / ".claude").mkdir()
    (root / "handoff").mkdir()
    # EVERY sibling module, not a hand-picked two. The first version copied only
    # attempt_budget.py and attempt_outcomes.py; `verdict_outcomes` also does
    # `from verdict_ledger_write import emit_sequence`, that import failed in the
    # sandbox, the function failed CLOSED as designed, and the control went red on
    # a check that has nothing to do with any mutation. A sandbox that is missing
    # an import does not test the subject -- it tests the sandbox.
    for f in sorted(GATE.parent.glob("*.py")):
        if f.name != GATE.name:
            shutil.copy2(f, root / "scripts" / "harness" / f.name)
    # The gate puts BOTH scripts/harness AND scripts/qa on sys.path (:83-84), and
    # `verdict_outcomes` imports verdict_ledger_write from the qa side. Copying only
    # the harness siblings still left that import failing, which the function then
    # handled fail-CLOSED exactly as designed -- so the control went red on a check
    # unrelated to any mutation. Mirror the real import surface, not a guess at it.
    (root / "scripts" / "qa").mkdir(parents=True, exist_ok=True)
    for f in sorted((REPO / "scripts" / "qa").glob("*.py")):
        shutil.copy2(f, root / "scripts" / "qa" / f.name)
    (root / "scripts" / "harness" / GATE.name).write_text(src, encoding="utf-8")
    shutil.copy2(MASTERPLAN, root / ".claude" / "masterplan.json")
    # An EMPTY verdict ledger, not a copy of the real one. The self-test manages
    # its own verdict fixtures, and seeding 146 real rows both collides with them
    # and puts real verdict history in a temp dir for no reason. Criterion 7's
    # byte-identity is measured on the REAL path, outside every sandbox.
    (root / "handoff" / "verdict_ledger.jsonl").write_text("", encoding="utf-8")
    (root / "handoff" / "current").mkdir(exist_ok=True)
    rp = root.resolve()
    if rp == REPO or REPO in rp.parents:
        raise SystemExit(f"CONTAINMENT GUARD: {rp} resolves inside {REPO}. Refusing.")
    return root


def drive(root: Path) -> tuple[int, str]:
    out = subprocess.run([sys.executable, "scripts/harness/attempt_gate.py", "--self-test"],
                         cwd=root, capture_output=True, text=True, timeout=600)
    return out.returncode, out.stdout + "\n" + out.stderr


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
     "NULL MUTANT (comment only). If this scores KILLED the harness is broken and every "
     "other kill in this run is meaningless.",
     _sub('DIGEST_STATUS_OK = "ok"', 'DIGEST_STATUS_OK = "ok"  # null mutant')),

    ("D1", "KILLED",
     "THE CELL THIS STEP EXISTS FOR: handoff/audit/ is removed from the exclusions, so "
     "the gate's OWN audit stream re-enters the digest and it advances by construction -- "
     "89.1's defect through a different door, measured by the 90.3 research gate before "
     "any code existed.",
     _sub('DIGEST_EXCLUDED_ROOTS = ("handoff/audit/", "handoff/logs/", ".claude/agent-memory/",',
          'DIGEST_EXCLUDED_ROOTS = ("handoff/logs/", ".claude/agent-memory/",')),

    ("D2", "KILLED",
     "criterion 1: the digest mixes in mtime, so os.utime() on an unchanged file moves it "
     "and a touched-but-identical relaunch is admitted.",
     _sub("        h.update(hashlib.sha256(data).digest())",
          "        h.update(hashlib.sha256(data).digest())\n"
          "        h.update(str(f.stat().st_mtime).encode())")),

    ("D3", "KILLED",
     "criterion 1: the digest becomes a constant, so nothing ever differs and every "
     "relaunch after the first is denied.",
     _sub("    return h.hexdigest(), DIGEST_STATUS_OK, []",
          "    return 'constant', DIGEST_STATUS_OK, []")),

    ("D4", "KILLED",
     "criterion 4: a MISSING declared input becomes a silent skip instead of a DENY, so a "
     "digest computed over a SUBSET masquerades as the digest of the whole set.",
     _sub("    if missing:\n        return None, DIGEST_STATUS_INPUTS_INCOMPLETE, missing",
          "    if missing:\n        return h.hexdigest(), DIGEST_STATUS_OK, []")),

    ("D5", "KILLED",
     "criterion 2: the NO_VERDICT exemption is removed, so a byte-identical relaunch after "
     "a dropped rail is DENIED -- the doctrine-mandated retry 89.1 would have blocked on "
     "14 of 16 real drops.",
     _sub('        return ("the most recent verdict row is NO_VERDICT -- a dropped rail "\n'
          '                "produces nothing to fix, so a byte-identical relaunch is correct")',
          "        return None")),

    ("D6", "KILLED",
     "criterion 3: comparison narrows to the PREVIOUS digest only, so an A->B->A revert "
     "oscillates forever instead of denying on the third launch.",
     _sub("    return out", "    return out[-1:]")),

    ("D7", "KILLED",
     "criterion 3: another step's digests leak into this step's comparison, so an unrelated "
     "step can deny this one.",
     _sub('        if str(r.get("step_id")) != str(step_id):\n            continue\n', "")),

    ("QX", "ERROR",
     "ERROR CONTROL: a call site is renamed, so the code parses, imports, and then cannot "
     "RESOLVE A NAME at run time. It must score ERROR, never a kill (phase-90.12).",
     _sub("    exempt = (None if DIGEST_ENABLED", "    exempt = (None if DIGEST_ENABLED_v2")),
]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    verify = "--verify" in argv

    gate_md5_before, vl_sha_before = md5(GATE), sha256(VERDICT_LEDGER)
    src = GATE.read_text(encoding="utf-8")

    print("=" * 74 + "\nCONTROL (the real, unmutated gate)\n" + "=" * 74)
    rc, obs = drive(sandbox(src))
    print(f"  exit={rc}  FAIL lines={sum(1 for l in obs.splitlines() if l.strip().startswith('FAIL'))}")
    if rc != 0:
        print("  CONTROL IS NOT GREEN -- every kill below would be meaningless. Stopping.")
        for ln in obs.splitlines():
            if ln.strip().startswith("FAIL"):
                print("   ", ln.strip())
        return 1
    print("  CONTROL GREEN")

    print("\n" + "=" * 74 + "\nCELLS\n" + "=" * 74)
    rows = []
    for cid, want, why, apply in CELLS:
        mutated = apply(src)
        code, observed = drive(sandbox(mutated))
        err = score_error(observed)
        got = "ERROR" if err else ("KILLED" if code != 0 else "SURVIVED")
        rows.append((cid, got, want, why, err or ""))

    bad = 0
    for cid, got, want, why, err in rows:
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BAD '} {cid:<4} {got:<9} expected {want:<9} {err}")
        print(f"         {why}")

    gate_md5_after, vl_sha_after = md5(GATE), sha256(VERDICT_LEDGER)
    print("\n" + "=" * 74 + "\nCONTAINMENT (criteria 6, 7)\n" + "=" * 74)
    print(f"  scripts/harness/attempt_gate.py md5   {gate_md5_before} -> {gate_md5_after}")
    print(f"  handoff/verdict_ledger.jsonl sha256   {vl_sha_before[:32]} -> {vl_sha_after[:32]}")
    contained = gate_md5_before == gate_md5_after and vl_sha_before == vl_sha_after
    print(f"  real tree untouched: {contained}   (a denial is NOT a verdict, so the "
          "verdict ledger must not move)")

    n_k = sum(1 for r in rows if r[1] == "KILLED")
    n_s = sum(1 for r in rows if r[1] == "SURVIVED" and r[0] != "N0")
    n_e = sum(1 for r in rows if r[1] == "ERROR")
    null_ok = any(r[0] == "N0" and r[1] == "SURVIVED" for r in rows)
    kill_ok = any(r[0] == "D1" and r[1] == "KILLED" for r in rows)
    print("\n" + "=" * 74)
    print(f"KILLED {n_k} | SURVIVED {n_s} (excl. N0) | ERROR {n_e} | "
          f"null mutant survived: {null_ok} | real-kill control killed: {kill_ok}")
    print("=" * 74)

    if not verify:
        return 0
    problems = bad + (0 if contained else 1) + (0 if null_ok else 1) + (0 if kill_ok else 1)
    if problems:
        print(f"  {problems} problem(s): {bad} unexpected score(s)"
              + ("" if contained else ", CONTAINMENT BREACHED")
              + ("" if null_ok else ", NULL MUTANT DID NOT SURVIVE")
              + ("" if kill_ok else ", REAL-KILL CONTROL DID NOT DIE"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
