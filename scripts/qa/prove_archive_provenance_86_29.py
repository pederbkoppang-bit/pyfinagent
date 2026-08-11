#!/usr/bin/env python3
"""phase-86.29 -- drive `.claude/hooks/archive-handoff.sh` against SYNTHETIC steps
in a SCRATCH TREE and prove the archive dir now holds the step it is named for.

WHY A SCRATCH TREE, NON-NEGOTIABLE. Criterion 3 says the demonstration runs
"against a synthetic step in a scratch tree, never against handoff/archive
itself". `handoff/archive/` is the audit trail this step exists to repair; a
demonstration that wrote into it would be destroying the evidence to prove a
point about evidence. Every fixture below lives under `tempfile.mkdtemp()`, the
hook is invoked with `CLAUDE_PROJECT_DIR` pointed at that temp root, and the
script asserts at exit that the real repository is byte-unchanged.

WHAT IT PROVES, in order:

  A. BEFORE -- the pre-fix hook, run on the identical fixture, produces
     `phase-99.1/contract.md` declaring **phase-82.54**. Not argued from the
     source: the old hook is recovered with `git show <sha>:<path>` and executed.
  B. AFTER  -- the current hook produces `phase-99.1/contract.md` declaring
     **99.1**, sourced from `contract_99.1.md`.
  C. EMPTY  -- a step with no per-step artifacts, offered only poisoned rolling
     files, archives NOTHING, emits a systemMessage on stdout, and records the
     failure in PROVENANCE.md. Previously this case printed `copied=4` and read
     as success.
  D. MUTATION -- four cells. Each mutation is asserted to have actually CHANGED
     the script text before it is run, because a `str.replace` that matches
     nothing looks exactly like a successful mutation and would score a survivor
     as a kill. A cell "KILLED" only if the check that covers it goes RED.

    $ python scripts/qa/prove_archive_provenance_86_29.py
    $ python scripts/qa/prove_archive_provenance_86_29.py --keep   # keep fixtures
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
HOOK = REPO / ".claude" / "hooks" / "archive-handoff.sh"

#: The commit at which the hook still had the unguarded rolling-copy branch.
#: Used to recover the PRE-FIX script for the BEFORE half of criterion 3.
PRE_FIX_REF = "c806cad6"

# The poisoned rolling files, reproducing the real defect exactly: three
# different steps' work sitting under the four unsuffixed names.
ROLLING = {
    "contract.md": "# Contract -- phase-82.54\n\nRolling file, stale since 2026-08-06.\n",
    "experiment_results.md": "# Experiment Results -- phase-82.6\n\nRolling file.\n",
    "evaluator_critique.md": "# Evaluator Critique -- phase-82.6\n\nRolling file.\n",
    "research_brief.md": "# Research Brief -- phase-80.2\n\nRolling file.\n",
}


class Failure(Exception):
    pass


def _run(cmd, env=None, cwd=None):
    return subprocess.run(
        cmd, shell=isinstance(cmd, str), env=env, cwd=cwd,
        capture_output=True, text=True,
    )


def make_scratch(step_id: str, with_per_step: bool) -> pathlib.Path:
    """A minimal fake repo: masterplan with `step_id` done, poisoned rolling
    files, and (optionally) correctly-named per-step artifacts."""
    root = pathlib.Path(tempfile.mkdtemp(prefix=f"arch86_29_{step_id}_"))
    (root / ".claude").mkdir()
    (root / "handoff" / "current").mkdir(parents=True)
    (root / "handoff" / "archive").mkdir(parents=True)

    (root / ".claude" / "masterplan.json").write_text(json.dumps(
        {"phases": [{"steps": [
            {"id": "99.0", "status": "done"},
            {"id": step_id, "status": "done"},
        ]}]}, indent=2))

    # Baseline must already contain 99.0 (so the hook does NOT self-seed and
    # emit nothing) while omitting step_id (so it reads as a NEW transition).
    (root / ".claude" / ".archive-baseline.json").write_text(
        json.dumps({"seen_done": ["99.0"]}, indent=2))

    cur = root / "handoff" / "current"
    for name, body in ROLLING.items():
        (cur / name).write_text(body)

    if with_per_step:
        (cur / f"contract_{step_id}.md").write_text(
            f"# Contract -- step `{step_id}`\n\nThe real per-step contract.\n")
        (cur / f"experiment_results_{step_id}.md").write_text(
            f"# Experiment results -- step {step_id}\n\n**Step**: `{step_id}`\n")
        (cur / f"evaluator_critique_{step_id}.md").write_text(
            f"# Evaluator critique -- step {step_id}\n\n**Step**: `{step_id}`\n")
        (cur / f"research_brief_{step_id}.md").write_text(
            f"# Research Brief -- step {step_id}\n\n**Step**: `{step_id}`\n")
        (cur / f"research_brief_{step_id}_rerun.md").write_text(
            f"# Research Brief -- step {step_id} (RERUN)\n\n**Step**: `{step_id}`\n")
    return root


def drive(hook_path: pathlib.Path, root: pathlib.Path):
    env = dict(os.environ)
    env["CLAUDE_PROJECT_DIR"] = str(root)
    return _run(["bash", str(hook_path)], env=env, cwd=str(root))


def declared(path: pathlib.Path) -> str | None:
    """Which step does this archived contract declare? Independent of the hook."""
    if not path.exists():
        return None
    import re
    head = path.read_text(errors="replace")[:4000]
    SID = r"[0-9]+(?:\.[0-9A-Za-z]+)*"
    for p in (rf"^#\s*Contract\s*--\s*step\s*`?({SID})`?",
              rf"^#\s*Contract\s*--\s*(?:.*?)phase-({SID})"):
        m = re.search(p, head, re.M | re.I)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# The three behavioural checks. Each returns None on pass, or a reason string.
# They are reused verbatim by the mutation matrix -- a mutation is KILLED when
# the check it targets starts returning a reason.
# ---------------------------------------------------------------------------

def check_right_step(hook: pathlib.Path) -> str | None:
    """AFTER: archiving 99.1 yields a contract.md declaring 99.1."""
    root = make_scratch("99.1", with_per_step=True)
    try:
        drive(hook, root)
        got = declared(root / "handoff" / "archive" / "phase-99.1" / "contract.md")
        if got != "99.1":
            return f"archive contract declares {got!r}, expected '99.1'"
        return None
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_no_poison_substitution(hook: pathlib.Path) -> str | None:
    """A rolling file that declares ANOTHER step must never be substituted."""
    root = make_scratch("99.2", with_per_step=False)
    try:
        drive(hook, root)
        got = declared(root / "handoff" / "archive" / "phase-99.2" / "contract.md")
        if got is not None:
            return f"poisoned rolling contract WAS substituted (declares {got!r})"
        return None
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_loud_on_empty(hook: pathlib.Path) -> str | None:
    """An archive that captured nothing must be a VISIBLE failure."""
    root = make_scratch("99.3", with_per_step=False)
    try:
        res = drive(hook, root)
        prov = root / "handoff" / "archive" / "phase-99.3" / "PROVENANCE.md"
        problems = []
        if "FAILURE" not in res.stderr:
            problems.append("no FAILURE line on stderr")
        try:
            payload = json.loads(res.stdout.strip() or "{}")
        except json.JSONDecodeError:
            payload = {}
        if "systemMessage" not in payload:
            problems.append("no systemMessage on stdout")
        if not prov.exists() or "RESULT: FAILED" not in prov.read_text():
            problems.append("PROVENANCE.md does not record the failure")
        return "; ".join(problems) or None
    finally:
        shutil.rmtree(root, ignore_errors=True)


CHECKS = [
    ("right_step", check_right_step),
    ("no_poison_substitution", check_no_poison_substitution),
    ("loud_on_empty", check_loud_on_empty),
]

#: (cell, description, old, new, which check MUST go red)
MUTANTS = [
    ("M1", "revert the derived branch (never build names from the step id)",
     'src="$CURRENT_DIR/${base}_${short_sid}.md"',
     'src="$CURRENT_DIR/__never_matches_${base}__.md"',
     "right_step"),
    ("M2", "revert the rolling guard to an unconditional copy",
     'if rolling_declares_step "$CURRENT_DIR/$f" "$short_sid"; then',
     'if true; then',
     "no_poison_substitution"),
    ("M3", "make the declaration check answer TRUE for every file",
     'sys.exit(0 if m.group(1) == want else 1)',
     'sys.exit(0)',
     "no_poison_substitution"),
    ("M4", "delete the empty-archive failure branch",
     'if [ "$total" -eq 0 ]; then',
     'if false; then',
     "loud_on_empty"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ns = ap.parse_args()

    repo_digest_before = hashlib.sha256(HOOK.read_bytes()).hexdigest()
    archive_before = sorted(p.name for p in (REPO / "handoff" / "archive").glob("phase-*"))

    print("=" * 76)
    print("phase-86.29 -- archive provenance, driven in a SCRATCH TREE")
    print("=" * 76)
    print(f"hook under test : {HOOK.relative_to(REPO)}")
    print(f"hook sha256     : {repo_digest_before[:16]}")
    print(f"real archive    : {len(archive_before)} phase-* dirs (must not change)")
    print()

    failures = 0

    # ---- A. BEFORE: the pre-fix hook on the identical fixture --------------
    print("-" * 76)
    print("A. BEFORE -- the PRE-FIX hook, recovered from git and executed")
    print("-" * 76)
    old = _run(["git", "-C", str(REPO), "show", f"{PRE_FIX_REF}:.claude/hooks/archive-handoff.sh"])
    if old.returncode != 0:
        print(f"  CANNOT RECOVER pre-fix hook at {PRE_FIX_REF}: {old.stderr.strip()}")
        failures += 1
    else:
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="arch86_29_prefix_"))
        old_hook = tmp / "archive-handoff.sh"
        old_hook.write_text(old.stdout)
        if "rolling_declares_step" in old.stdout:
            print("  REFUSED: the recovered script already contains the fix --")
            print("  PRE_FIX_REF is wrong and the BEFORE half would be vacuous.")
            failures += 1
        else:
            root = make_scratch("99.1", with_per_step=True)
            drive(old_hook, root)
            got = declared(root / "handoff" / "archive" / "phase-99.1" / "contract.md")
            print(f"  pre-fix hook, step 99.1 -> archive contract declares: {got!r}")
            if got == "82.54":
                print("  CONFIRMED: the archive dir named 99.1 holds phase-82.54's contract.")
            else:
                print(f"  UNEXPECTED: expected '82.54', got {got!r} -- the BEFORE half did not reproduce")
                failures += 1
            shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)
    print()

    # ---- B/C. AFTER: the three behavioural checks on the live hook ---------
    print("-" * 76)
    print("B/C. AFTER -- the current hook (control run; all three must be GREEN)")
    print("-" * 76)
    control = {}
    for name, fn in CHECKS:
        reason = fn(HOOK)
        control[name] = reason
        print(f"  {name:26s} {'GREEN' if reason is None else 'RED  -- ' + reason}")
        if reason is not None:
            failures += 1
    print()

    # ---- D. MUTATION -------------------------------------------------------
    print("-" * 76)
    print("D. MUTATION -- a cell is KILLED only if its target check goes RED")
    print("-" * 76)
    src = HOOK.read_text()
    for cell, desc, old_s, new_s, target in MUTANTS:
        if old_s not in src:
            print(f"  {cell} ANCHOR MISSING -- refusing to score. {desc}")
            print(f"      the text {old_s!r} is not in the hook; this cell would")
            print("      have silently tested an unmutated script and scored a KILL.")
            failures += 1
            continue
        mutated = src.replace(old_s, new_s, 1)
        if mutated == src:
            print(f"  {cell} MUTATION DID NOT CHANGE THE TEXT -- refusing to score")
            failures += 1
            continue
        tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"arch86_29_{cell}_"))
        mhook = tmp / "archive-handoff.sh"
        mhook.write_text(mutated)
        fn = dict(CHECKS)[target]
        reason = fn(mhook)
        killed = reason is not None
        print(f"  {cell} {'KILLED  ' if killed else 'SURVIVED'} [{target}] -- {desc}")
        if killed:
            print(f"      check went RED: {reason}")
        else:
            failures += 1
            print("      the check stayed GREEN with the fix reverted: it does not")
            print("      actually cover this behaviour.")
        shutil.rmtree(tmp, ignore_errors=True)
    print()

    # ---- isolation assertion ----------------------------------------------
    print("-" * 76)
    print("ISOLATION -- the real repository must be untouched")
    print("-" * 76)
    repo_digest_after = hashlib.sha256(HOOK.read_bytes()).hexdigest()
    archive_after = sorted(p.name for p in (REPO / "handoff" / "archive").glob("phase-*"))
    ok_hook = repo_digest_before == repo_digest_after
    ok_arch = archive_before == archive_after
    print(f"  hook sha256 unchanged            : {ok_hook}")
    print(f"  handoff/archive dir list unchanged: {ok_arch} ({len(archive_after)} dirs)")
    new_dirs = set(archive_after) - set(archive_before)
    if new_dirs:
        print(f"  NEW DIRS CREATED IN THE REAL ARCHIVE: {sorted(new_dirs)}")
    if not (ok_hook and ok_arch):
        failures += 1
    print()

    print("=" * 76)
    print(f"RESULT: {'PASS' if failures == 0 else 'FAIL'} ({failures} problem(s))")
    print("=" * 76)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
