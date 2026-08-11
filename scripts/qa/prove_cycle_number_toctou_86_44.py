#!/usr/bin/env python3
"""phase-86.44: demonstrate the cycle-NUMBER race, and show it is a
different defect from the cycle-DATA race that 86.44 fixed.

Two distinct races live in this project's harness_log writers:

  D1 (FIXED in 86.44) -- run_harness.py did read_text + write_text, so a
     concurrent writer's whole block was destroyed. That is DATA loss.

  D4 (FOUND here, NOT fixed) -- scripts/smoketest/steps/finalize.py does
     `_next_cycle_number(read_text())` and then `open("a")`. The append is
     correct, so no data is lost; but the NUMBER is computed before the
     write with no lock, so two writers that read the same max both claim
     max+1. That is a TOCTOU on the identifier.

This runs as a FILE and not a heredoc on purpose: multiprocessing's spawn
start method (the macOS default) re-imports __main__ by path, and a `python3 -`
heredoc has no path -- the first attempt at this probe died in
spawn_main/_fixup_main_from_path for exactly that reason.
"""
from __future__ import annotations

import multiprocessing as mp
import pathlib
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
N_WRITERS = 16


_BARRIER = None


def _init(barrier):
    global _BARRIER
    _BARRIER = barrier


def _worker(target: str) -> int:
    sys.path.insert(0, str(ROOT))
    import importlib
    m = importlib.import_module("scripts.smoketest.steps.finalize")
    # A TOCTOU is a WINDOW. Worker startup (import, interpreter spawn) dominates
    # the window, so an unsynchronised pool serialises and the race never fires --
    # the first run of this probe reported 16/16 distinct and would have been read
    # as "safe". The barrier does not create the defect; it stops process-startup
    # jitter from hiding it. Every worker is inside the real, unmodified
    # append_cycle_row when it runs.
    if _BARRIER is not None:
        _BARRIER.wait()
    return m.append_cycle_row(pathlib.Path(target), "99.9", "PASS", "86.44 race probe")["cycle"]


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        target = pathlib.Path(td) / "log.md"
        target.write_text("## Cycle 100 -- 2026-01-01 -- phase=1.0 result=PASS\n", encoding="utf-8")
        barrier = mp.Barrier(N_WRITERS)
        with mp.Pool(N_WRITERS, initializer=_init, initargs=(barrier,)) as pool:
            assigned = pool.map(_worker, [str(target)] * N_WRITERS, chunksize=1)
        text = target.read_text(encoding="utf-8")
        rows = re.findall(r"^## Cycle (\d+) --", text, re.M)

    distinct = len(set(assigned))
    collisions = N_WRITERS - distinct
    print(f"  seed max                 : 100")
    print(f"  concurrent appenders     : {N_WRITERS}")
    print(f"  numbers assigned         : {sorted(assigned)}")
    print(f"  DISTINCT numbers         : {distinct} of {N_WRITERS}")
    print(f"  collisions               : {collisions}")
    print(f"  rows actually written    : {len(rows)} (seed 1 + {len(rows)-1} new)")
    print()
    if len(rows) - 1 == N_WRITERS:
        print("  DATA: every append survived -- finalize.py already uses open('a'),")
        print("        so it never had D1's read-modify-write data loss.")
    else:
        print(f"  DATA LOSS: {N_WRITERS - (len(rows)-1)} appends vanished (unexpected here).")
    if collisions:
        print(f"  NUMBER: {collisions} writers claimed a number another writer also claimed.")
        print("        _next_cycle_number() reads max BEFORE the append, with no lock:")
        print("        scripts/smoketest/steps/finalize.py:70-72 then :83-85.")
        print("        THIS is the mechanism behind the duplicate integers in history.")
    else:
        print("  NUMBER: no collision observed in this run. The race is real in the")
        print("        code path but is timing-dependent; absence here is NOT a proof")
        print("        of safety -- re-run, or widen N_WRITERS.")
    # Deliberately NOT an assertion-based test: a timing-dependent race that
    # happens not to fire must not be reported as a passing guard.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
