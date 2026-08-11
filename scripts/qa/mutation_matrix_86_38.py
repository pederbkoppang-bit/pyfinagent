#!/usr/bin/env python3
"""phase-86.38 mutation matrix -- every new guard must be observed FAILING.

METHOD, and its risk stated plainly. Each cell rewrites a production source file
on disk, runs ONLY the test that is supposed to cover that change, and restores
the file in a `finally`. This is the same method `scripts/qa/mutation_matrix_86_24.py`
uses, and it is chosen over re-implementing the assertions inside this script
because a matrix that re-implements its own guard tests nothing (qa.md 4c
vacuity shape #7 -- the guard must be the SHIPPED one).

SAFETY, because these are live trading modules:
  * every file's sha256 is captured BEFORE any mutation and re-verified after
    every cell AND at exit; a mismatch is a hard failure, not a warning;
  * the restore is in a `finally`, so a KeyboardInterrupt still restores;
  * the run is refused outright if the working tree for the target files is
    already dirty, because then "restored" could not be distinguished from
    "clobbered someone else's edit".

A cell is KILLED only when its named test goes RED. A cell whose anchor is
missing is BROKEN and scores nothing -- a `str.replace` that matches nothing
looks exactly like a successful mutation and would otherwise be credited a kill.

    $ python scripts/qa/mutation_matrix_86_38.py
"""
from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
LOOP = REPO / "backend" / "services" / "autonomous_loop.py"
HEALTH = REPO / "backend" / "services" / "cycle_health.py"
TESTS = "backend/tests/test_phase_86_38_degradation_visibility.py"

#: (cell, file, old, new, test node, description)
MUTANTS = [
    ("M1", LOOP,
     'if _n_fb_total:\n                    summary["fallback_rate"]',
     'if False:\n                    summary["fallback_rate"]',
     "test_degradation_fields_are_set_outside_the_fire_branch",
     "stop recording the rate unless the alarm fires (the original defect)"),
    ("M2", LOOP,
     'summary["fallback_alarm_fired"] = bool(_fb_fire)',
     '_unused_fired = bool(_fb_fire)',
     "test_degradation_fields_are_set_outside_the_fire_branch",
     "record the rate but drop whether it paged"),
    ("M3", LOOP,
     '_lite["_fallback_reason"] = _fb_reason[:500]',
     '_lite["_fallback_reason"] = _fb_reason[:500]\n            _lite["_intended_path"] = "full"',
     "test_intended_path_is_gone_and_not_merely_unused",
     "restore the write-only _intended_path field"),
    ("M4", HEALTH,
     '"degradation": degradation or {},',
     '"_degradation_dropped": degradation or {},',
     "test_degradation_is_persisted_on_a_quiet_cycle",
     "stop persisting degradation under the key readers look for"),
    ("M5", HEALTH,
     '"degradation": degradation or {},',
     '"degradation": funnel or {},',
     "test_degradation_defaults_empty_and_breaks_no_existing_caller",
     "fold degradation into the 66.2 funnel instead of keeping it separate"),
    ("M6", LOOP,
     "fire = n_total > 0 and (n_fallback / n_total) > threshold",
     "fire = n_total > 0 and (n_fallback / n_total) >= threshold",
     "test_the_2026_08_10_boundary_does_not_page",
     "loosen the alarm to >= (paging behaviour MUST be pinned)"),
]


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_test(node: str) -> int:
    return subprocess.run(
        [sys.executable, "-m", "pytest", f"{TESTS}::{node}", "-q", "--no-header"],
        cwd=REPO, capture_output=True, text=True,
    ).returncode


def main() -> int:
    targets = [LOOP, HEALTH]

    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--"] + [str(p) for p in targets],
        capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        print("REFUSING TO RUN -- the target files have uncommitted changes:")
        print(dirty)
        print("\nA restore could not then be distinguished from clobbering an edit.")
        print("Commit or stash first.")
        return 2

    before = {p: sha(p) for p in targets}
    originals = {p: p.read_text() for p in targets}
    print("=" * 78)
    print("phase-86.38 mutation matrix")
    print("=" * 78)
    for p in targets:
        print(f"  {p.relative_to(REPO)}  sha256={before[p][:16]}")
    print()

    print("CONTROL -- the whole module must be GREEN before any mutation")
    rc = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", "--no-header"],
        cwd=REPO, capture_output=True, text=True,
    ).returncode
    print(f"  control rc={rc}  {'GREEN' if rc == 0 else 'RED -- matrix is meaningless, aborting'}")
    if rc != 0:
        return 1
    print()

    killed, survived, broken = 0, [], []
    try:
        for cell, path, old, new, node, desc in MUTANTS:
            src = originals[path]
            if old not in src:
                broken.append(cell)
                print(f"  {cell} BROKEN  | {desc}")
                print(f"          anchor not found in {path.name}; scoring NOTHING")
                continue
            mutated = src.replace(old, new, 1)
            if mutated == src:
                broken.append(cell)
                print(f"  {cell} BROKEN  | replace was a no-op; scoring NOTHING")
                continue
            try:
                path.write_text(mutated)
                trc = run_test(node)
            finally:
                path.write_text(src)
            assert sha(path) == before[path], f"RESTORE FAILED for {path}"
            if trc != 0:
                killed += 1
                print(f"  {cell} KILLED  | {desc}")
                print(f"          {node} went RED (rc={trc})")
            else:
                survived.append(cell)
                print(f"  {cell} SURVIVED| {desc}")
                print(f"          {node} stayed GREEN -- it does not cover this")
    finally:
        for p in targets:
            p.write_text(originals[p])

    print()
    after = {p: sha(p) for p in targets}
    intact = all(before[p] == after[p] for p in targets)
    print(f"[integrity] tracked sources unchanged: {intact}")
    for p in targets:
        print(f"    {p.relative_to(REPO)}  {after[p][:16]}")
    print()
    if broken:
        print(f"MATRIX BROKEN -- anchors failed for: {', '.join(broken)}")
        return 3
    if survived:
        print(f"{killed} of {len(MUTANTS)} killed. SURVIVORS: {', '.join(survived)}")
        return 1
    if not intact:
        print("SOURCES CHANGED DURING THE RUN -- investigate.")
        return 4
    print(f"ALL {len(MUTANTS)} MUTANTS KILLED -- every guard in this matrix can fail.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
