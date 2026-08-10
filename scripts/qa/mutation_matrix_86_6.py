#!/usr/bin/env python3
"""phase-86.6 criterion 6 -- mutation-test the live-state PREVENTER.

CRITERION 6, VERBATIM: "removing the preventer must fail the criterion-2 test,
and the mutation must be run against a tmp COPY of the journal, never the live
file."

THE SECOND HALF IS THE HARD PART, AND IT SHAPES THE WHOLE HARNESS
-----------------------------------------------------------------
The obvious harness -- disable the guard, run the criterion-2 test, watch it
fail -- WRITES TO THE LIVE JOURNAL, because with the guard gone nothing stops
the write. "Restore it afterwards from a backup" does not satisfy the criterion
either: the bytes still land on the operator's live safety journal, and a crash
between write and restore leaves them there. A mutation harness that must not
be interrupted is not a safe harness.

So the mutation runs against a TMP COPY end to end. Each cell:

  1. copies the live journal to a fresh tmp dir (a real copy, so the subject is
     byte-identical to production, not a synthetic stub);
  2. re-executes `conftest.py`'s guard source in a child process with
     `_BLOCKED_PATHS` REDIRECTED to that copy;
  3. attempts the same append the criterion-2 test attempts;
  4. reports whether it was REFUSED and whether the copy's sha256 moved.

The live journal is digested before and after the entire run and the harness
fails if it moved by a single byte. It is never opened for writing at any point.

WHAT EACH CELL PROVES
---------------------
A guard is only load-bearing if its removal is observable. CONTROL shows the
refusal happens; each mutant removes one load-bearing part and must show the
write LANDING on the copy.

    source .venv/bin/activate
    python scripts/qa/mutation_matrix_86_6.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFTEST = REPO_ROOT / "conftest.py"
LIVE_JOURNAL = REPO_ROOT / "handoff" / "kill_switch_audit.jsonl"

MUTANTS: list[dict] = [
    {
        "id": "CONTROL",
        "desc": "guard intact",
        "proves": "the refusal fires at all -- every kill below is measured against this",
        "old": None, "new": None,
        "expect_refused": True,
    },
    {
        "id": "M1",
        "desc": "remove the preventer entirely (never install the audit hook)",
        "proves": "criterion 6 -- without the hook the write LANDS, so the hook is what prevents it",
        "old": "_sys.addaudithook(_live_state_audit_hook)",
        "new": "pass  # MUTANT M1: preventer not installed",
        "expect_refused": False,
    },
    {
        "id": "M2",
        "desc": "downgrade the refusal to a log line (detect, do not prevent)",
        "proves": "detection is not prevention -- 36.12 and 86.3 both fired AFTER the bytes landed",
        "old": "    raise LiveStateWriteRefused(msg)",
        "new": "    print(msg, file=_sys.stderr)  # MUTANT M2: detect only",
        "expect_refused": False,
    },
    {
        "id": "M3",
        "desc": "make LiveStateWriteRefused an Exception instead of a BaseException",
        "proves": "the house RuntimeError pattern would be SWALLOWED by _append_audit's except Exception",
        "old": "class LiveStateWriteRefused(BaseException):",
        "new": "class LiveStateWriteRefused(Exception):  # MUTANT M3",
        "expect_refused": True,          # still raises HERE; see the note below
        "also_check_swallow": True,
    },
    {
        "id": "M4",
        "desc": "drop 'a' from the write-intent MODE chars (mode leg only)",
        # PREDICTED False, MEASURED True -- and the guard was right, not the
        # matrix. `_is_write_intent` has TWO independent legs: the mode string
        # AND the os flags. CPython's `open` audit event carries both, and
        # open(p, "a") sets O_APPEND in the flags regardless of what the mode
        # chars say, so the flags leg still catches it. That is defence in
        # depth, so the expectation is corrected rather than the guard
        # weakened -- and M5 exists precisely because a redundant leg is only
        # PROVEN redundant when removing BOTH opens the hole.
        "proves": "the FLAGS leg is an independent second line -- the mode leg alone is not load-bearing for an append",
        "old": '_WRITE_MODE_CHARS = frozenset("wax+")',
        "new": '_WRITE_MODE_CHARS = frozenset("wx+")  # MUTANT M4',
        "expect_refused": True,
    },
    {
        "id": "M5",
        "desc": "drop BOTH append legs (mode 'a' AND O_APPEND from the flags)",
        "proves": "the two legs together ARE the complete set for an append -- remove both and the write lands",
        "old": '_WRITE_FLAGS = (\n    _os.O_WRONLY | _os.O_RDWR | _os.O_APPEND | _os.O_CREAT | _os.O_TRUNC\n)',
        "new": '_WRITE_FLAGS = 0  # MUTANT M5: flags leg disabled',
        "extra_old": '_WRITE_MODE_CHARS = frozenset("wax+")',
        "extra_new": '_WRITE_MODE_CHARS = frozenset("wx+")  # MUTANT M5',
        "expect_refused": False,
    },
]

# The child re-executes the guard source with _BLOCKED_PATHS pointed at a COPY.
CHILD = r"""
import hashlib, json, os, sys
from pathlib import Path

spec = json.load(open(os.environ["MUT866_SPEC"], encoding="utf-8"))
# .resolve() is NOT cosmetic: on macOS /var is a symlink to /private/var, and
# the guard resolves the path it is handed before comparing. An unresolved copy
# path compares unequal to every blocked entry, so the CONTROL cell reported
# "not refused" and looked like a broken guard when it was a broken harness.
copy = Path(spec["copy"]).resolve()
src  = spec["source"]

ns = {"__name__": "_mutated_conftest", "__file__": spec["conftest"]}
exec(compile(src, spec["conftest"], "exec"), ns)

# Redirect the guard at the COPY. The hook reads the module global at call
# time, so rebinding it here is enough and no live path is ever a target.
ns["_BLOCKED_PATHS"] = (copy,)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None

before = sha(copy)
refused, exc_type, swallowed_by_except_Exception = False, None, None
try:
    with open(copy, "a", encoding="utf-8") as fh:
        fh.write('{"phase-86.6": "mutation probe"}\n')
except BaseException as e:
    refused, exc_type = True, type(e).__name__
    # Would the PRODUCTION swallow shape have absorbed it?
    swallowed_by_except_Exception = isinstance(e, Exception)

after = sha(copy)
print(json.dumps({
    "refused": refused,
    "exc_type": exc_type,
    "swallowed_by_except_Exception": swallowed_by_except_Exception,
    "copy_changed": before != after,
}))
"""


def digest(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def run_cell(cell: dict) -> dict:
    src = CONFTEST.read_text(encoding="utf-8")
    if cell["old"] is not None:
        n = src.count(cell["old"])
        if n != 1:
            raise SystemExit(
                f"{cell['id']}: anchor occurs {n} times, must be exactly 1 -- "
                f"a no-match str.replace is indistinguishable from success")
        mutated = src.replace(cell["old"], cell["new"])
        if mutated == src:
            raise SystemExit(f"{cell['id']}: mutation was a no-op")
        src = mutated
    if cell.get("extra_old") is not None:
        n = src.count(cell["extra_old"])
        if n != 1:
            raise SystemExit(
                f"{cell['id']}: extra anchor occurs {n} times, must be exactly 1")
        mutated = src.replace(cell["extra_old"], cell["extra_new"])
        if mutated == src:
            raise SystemExit(f"{cell['id']}: extra mutation was a no-op")
        src = mutated

    tmp = Path(tempfile.mkdtemp())
    copy = tmp / "kill_switch_audit.jsonl"
    if LIVE_JOURNAL.exists():
        shutil.copy2(LIVE_JOURNAL, copy)          # a REAL copy, not a stub
    else:
        copy.write_text("", encoding="utf-8")

    spec_path = tmp / "spec.json"
    spec_path.write_text(json.dumps({
        "copy": str(copy), "source": src, "conftest": str(CONFTEST),
    }), encoding="utf-8")

    env = dict(os.environ, MUT866_SPEC=str(spec_path), PYTHONPATH=str(REPO_ROOT))
    proc = subprocess.run([sys.executable, "-c", CHILD], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise SystemExit(f"{cell['id']}: child failed\n{proc.stderr[-1500:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    before_live = digest(LIVE_JOURNAL)
    print("phase-86.6 criterion 6 -- preventer mutation matrix")
    print(f"  live journal (NEVER written by this harness): {before_live}")
    print(f"  every cell operates on a fresh tmp COPY\n")

    hdr = f"{'id':<9}{'refused':<10}{'copy changed':<15}{'verdict':<10}mutation"
    print(hdr); print("-" * len(hdr))
    failures = 0
    for cell in MUTANTS:
        r = run_cell(cell)
        ok = (r["refused"] == cell["expect_refused"])
        # A refusal that leaves the copy CHANGED is detection, not prevention.
        if cell["expect_refused"] and r["copy_changed"]:
            ok = False
        if not cell["expect_refused"] and not r["copy_changed"]:
            ok = False                            # the mutant should have written
        verdict = "ok" if ok else "UNEXPECTED"
        if not ok:
            failures += 1
        print(f"{cell['id']:<9}{str(r['refused']):<10}{str(r['copy_changed']):<15}"
              f"{verdict:<10}{cell['desc']}")
        print(f"{'':19}proves: {cell['proves']}")
        if cell.get("also_check_swallow"):
            swallowed = r["swallowed_by_except_Exception"]
            print(f"{'':19}-> raised {r['exc_type']}; would production's "
                  f"`except Exception` swallow it? {swallowed}")
            if not swallowed:
                print(f"{'':19}   UNEXPECTED: M3 makes it an Exception, so it "
                      f"MUST be swallowable")
                failures += 1
            else:
                print(f"{'':19}   That is the trap: the write is blocked, the "
                      f"refusal absorbed, and the test stays GREEN.")

    after_live = digest(LIVE_JOURNAL)
    print(f"\n  live journal after: {after_live}")
    if after_live != before_live:
        print("  !! THE LIVE JOURNAL MOVED -- this harness must never do that.")
        return 1
    print("  live journal UNCHANGED (byte-identical across the whole matrix)")

    if failures:
        print(f"\n{failures} cell(s) behaved unexpectedly.")
        return 1
    print("\nEvery cell behaved as predicted: the preventer is load-bearing, and")
    print("removing any load-bearing part lets the write land on the copy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
