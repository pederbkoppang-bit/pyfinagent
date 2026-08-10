#!/usr/bin/env python3
"""phase-86.24 criterion 6 -- mutation matrix for the clock-dependence repairs.

    $ python scripts/qa/mutation_matrix_86_24.py

HOW MUTATION IS DONE HERE, AND WHY NOT IN PLACE
-----------------------------------------------
Each mutant is written as a COPY under `backend/tests/` with a temporary name and
run on its own; the tracked file is never opened for writing. A copy INSIDE the
test tree (rather than in `/tmp`) is deliberate: pytest's rootdir, the repo-root
`conftest.py` egress guards and the `backend.*` import path all have to apply
exactly as they do for the real module, or the mutant would be running under
different rules than the thing it is meant to mutate.

The copies are removed in a `finally`, and the tracked sources are digested
before and after so "I did not write to them" is proven rather than asserted --
a second Claude session is live in this tree today.

Anchor uniqueness is checked for every mutation: a `str.replace` that matches
nothing returns a perfectly normal string, so a non-mutation would otherwise be
scored as a kill.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
TESTS = REPO / "backend" / "tests"

MACRO = TESTS / "test_phase_82_0_macro_ingestion.py"
POISON = TESTS / "test_phase_86_2_replay_poison_row.py"
NEWMOD = TESTS / "test_phase_86_24_clock_dependence.py"

#: (id, source file, anchor, replacement, TZ for the run, description)
#: TZ=None means "run under the system clock".
MUTATIONS = [
    ("M1", MACRO,
     "    return datetime.now(timezone.utc).date().isoformat()",
     "    return date.today().isoformat()",
     "Pacific/Midway",
     "revert the macro tests to the LOCAL clock domain"),

    ("M2", POISON,
     "_UTC_TODAY = _dt.datetime.now(_dt.timezone.utc).date()",
     "_UTC_TODAY = _dt.date(2026, 8, 9)",
     None,
     "re-pin the poison-row fixture to the day it was written"),

    ("M3", NEWMOD,
     "    _journal(isolated, _day(-1))\n    st = ks.KillSwitchState()\n"
     "    monkeypatch.setattr(ks, \"_state\", st)\n"
     "    r = ks.evaluate_breach(80.0, 4.0, 10.0)\n\n"
     "    assert r[\"daily_baseline_stale\"] is True, r",
     "    _journal(isolated, _day(0))\n    st = ks.KillSwitchState()\n"
     "    monkeypatch.setattr(ks, \"_state\", st)\n"
     "    r = ks.evaluate_breach(80.0, 4.0, 10.0)\n\n"
     "    assert r[\"daily_baseline_stale\"] is True, r",
     None,
     "give the STALE-anchor test a FRESH anchor -- does it actually discriminate?"),

    ("M4", NEWMOD,
     '    env = {**os.environ, "TZ": "Pacific/Midway"}',
     '    env = {**os.environ}',
     None,
     "remove the clock shift from the differential test -- its positive control "
     "must fire rather than the test passing for free"),

    ("M5", NEWMOD,
     '@pytest.mark.parametrize("offset", [-1, -2, -7, -365])',
     '@pytest.mark.parametrize("offset", [0])',
     None,
     "point the how-stale sweep at a FRESH anchor"),
]


def digest(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def run(path: pathlib.Path, tz: str | None) -> tuple[int, str]:
    env = {**os.environ}
    if tz:
        env["TZ"] = tz
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(path), "-q", "-p", "no:randomly"],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600)
    return proc.returncode, (proc.stdout or "")[-300:]


def main() -> int:
    tracked = [MACRO, POISON, NEWMOD]
    before = {p: digest(p) for p in tracked}
    rows, survived = [], []

    for mid, src, anchor, repl, tz, desc in MUTATIONS:
        text = src.read_text()
        n = text.count(anchor)
        if n != 1:
            rows.append((mid, "ANCHOR", f"matched {n} time(s)", desc))
            survived.append(mid)
            continue

        control_rc, _ = run(src, tz)
        mut = src.with_name(f"test_zz_mutant_{mid.lower()}_{os.getpid()}.py")
        try:
            mut.write_text(text.replace(anchor, repl, 1))
            mut_rc, tail = run(mut, tz)
        finally:
            mut.unlink(missing_ok=True)

        killed = control_rc == 0 and mut_rc != 0
        detail = f"control rc={control_rc} mutant rc={mut_rc}"
        if control_rc != 0:
            detail += "  <-- CONTROL NOT GREEN, cell is meaningless"
        rows.append((mid, "KILLED" if killed else "SURVIVED", detail, desc))
        if not killed:
            survived.append(mid)

    after = {p: digest(p) for p in tracked}
    unchanged = before == after
    strays = sorted(TESTS.glob("test_zz_mutant_*.py"))

    w = max(len(r[3]) for r in rows)
    print(f"{'id':4} {'verdict':9} {'probe':46} mutation")
    print("-" * (62 + w))
    for mid, verdict, detail, desc in rows:
        print(f"{mid:4} {verdict:9} {detail[:46]:46} {desc}")
    print("-" * (62 + w))
    print(f"tracked sources UNCHANGED: {unchanged}  {[(p.name, before[p]) for p in tracked]}")
    print(f"stray mutant files left behind: {[p.name for p in strays] or 'none'}")
    if strays or not unchanged:
        print("\nHARNESS FAULT -- refusing to report a clean result")
        return 2
    if survived:
        print(f"\nSURVIVING MUTANTS: {survived} -- those guards are NOT proven "
              f"load-bearing and do not count as covered.")
        return 1
    print(f"\nAll {len(rows)} mutants killed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
