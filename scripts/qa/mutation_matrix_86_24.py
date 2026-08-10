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

import datetime as _dt
import hashlib
import os
import pathlib
import subprocess
import sys
from zoneinfo import ZoneInfo

REPO = pathlib.Path(__file__).resolve().parents[2]


def _date_shifting_tz() -> str:
    """A zone whose LOCAL date differs from the UTC date RIGHT NOW.

    phase-86.34. M1 used to hardcode `Pacific/Midway`. That zone (UTC-11) only
    shifts the calendar date during 00:00-10:59 UTC, so for the other 13 hours
    the M1 mutant -- "revert the macro tests to the LOCAL clock domain" -- is
    behaviourally identical to the original and SURVIVES. Measured 2026-08-10 at
    19:23 UTC: M1 SURVIVED. It was killed earlier the same day at ~10:5x UTC,
    inside Midway's window. The cell was reporting the wall clock, not the guard.

    Kiritimati (UTC+14) covers 10:00-23:59, so the pair spans all 24 hours.
    Returns the first candidate that actually shifts the date now.
    """
    now = _dt.datetime.now(_dt.timezone.utc)
    for name in ("Pacific/Midway", "Pacific/Kiritimati"):
        if now.astimezone(ZoneInfo(name)).date() != now.date():
            return name
    # Unreachable given the coverage above; never silently return a
    # non-shifting zone -- that is the defect this function exists to remove.
    raise RuntimeError("no candidate zone shifts the calendar date right now")
TESTS = REPO / "backend" / "tests"

MACRO = TESTS / "test_phase_82_0_macro_ingestion.py"
POISON = TESTS / "test_phase_86_2_replay_poison_row.py"
NEWMOD = TESTS / "test_phase_86_24_clock_dependence.py"

#: Each cell is a dict so the fields cannot drift positionally:
#:   id, src      -- the file to mutate (a COPY is written; src is never touched)
#:   anchor/repl  -- the mutation; anchor uniqueness is asserted
#:   tz           -- TZ for the run, or None for the system clock
#:   run          -- which file pytest actually runs. Defaults to the mutant.
#:                   M6 must run a DIFFERENT module than the one it mutates,
#:                   because the assertion that kills it lives there.
#:   env          -- extra env; the literal "<MUTANT>" is replaced with the
#:                   mutant copy's path.
MUTATIONS = [
    dict(id="M1", src=MACRO, tz=_date_shifting_tz(),
         anchor="    return datetime.now(timezone.utc).date().isoformat()",
         repl="    return date.today().isoformat()",
         desc="revert the macro tests to the LOCAL clock domain"),

    dict(id="M2", src=POISON, tz=None,
         anchor="def _day(offset_days: int = 0) -> str:\n"
                "    return (_dt.datetime.now(_dt.timezone.utc).date()\n"
                "            + _dt.timedelta(days=offset_days)).isoformat()",
         repl="def _day(offset_days: int = 0) -> str:\n"
              "    return (_dt.date(2026, 8, 9)\n"
              "            + _dt.timedelta(days=offset_days)).isoformat()",
         desc="re-pin the poison-row fixture to the day it was written"),

    # M6 is the cycle-1 Q/A's second finding turned into a cell. The FIRST
    # version of the repair snapshotted the date at import while production
    # recomputes at call time -- the masterplan's own case (a), introduced
    # inside the file this step was repairing. An ordinary run CANNOT kill that
    # mutant (it only misfires if the clock crosses midnight mid-run), so the
    # property is asserted directly, and this cell mutates POISON but RUNS the
    # module holding that assertion, pointed at the mutant via the test seam.
    dict(id="M6", src=POISON, tz=None, run=NEWMOD,
         env={"PYFINAGENT_86_24_PROW_PATH": "<MUTANT>"},
         anchor="def _day(offset_days: int = 0) -> str:\n"
                "    return (_dt.datetime.now(_dt.timezone.utc).date()\n"
                "            + _dt.timedelta(days=offset_days)).isoformat()",
         repl="_SNAPSHOT = _dt.datetime.now(_dt.timezone.utc).date()\n\n\n"
              "def _day(offset_days: int = 0) -> str:\n"
              "    return (_SNAPSHOT + _dt.timedelta(days=offset_days)).isoformat()",
         desc="SNAPSHOT the fixture date at import instead of recomputing per call"),

    dict(id="M7", src=NEWMOD, tz=None,
         anchor="    for nav in (95.0, 92.0):                 # 5% and 8% -- inside (4%, 10%)",
         repl="    for nav in (80.0, 80.0):                 # 20% -- OUTSIDE the band",
         desc="point the band test OUTSIDE the band -- does it discriminate?"),

    dict(id="M3", src=NEWMOD, tz=None,
         anchor='    _journal(isolated, _day(-1))\n    st = ks.KillSwitchState()\n'
                '    monkeypatch.setattr(ks, "_state", st)\n'
                '    r = ks.evaluate_breach(80.0, 4.0, 10.0)\n\n'
                '    assert r["daily_baseline_stale"] is True, r',
         repl='    _journal(isolated, _day(0))\n    st = ks.KillSwitchState()\n'
              '    monkeypatch.setattr(ks, "_state", st)\n'
              '    r = ks.evaluate_breach(80.0, 4.0, 10.0)\n\n'
              '    assert r["daily_baseline_stale"] is True, r',
         desc="give the STALE-anchor test a FRESH anchor -- does it discriminate?"),

    dict(id="M4", src=NEWMOD, tz=None,
         # phase-86.34 RE-ANCHORED: this used to bind the literal
         # `"TZ": "Pacific/Midway"`. 86.34 replaced that hardcoded zone with the
         # runtime selector `_date_shifting_tz()`, so the old anchor matched 0
         # times and this cell reported ANCHOR/survived -- a stale anchor, not a
         # real survivor. The cell's INTENT is unchanged: strip the clock shift
         # and require the positive control to fire.
         anchor='    env = {**os.environ, "TZ": _date_shifting_tz()}',
         repl='    env = {**os.environ}',
         desc="remove the clock shift from the differential test -- its positive "
              "control must fire rather than the test passing for free"),

    dict(id="M5", src=NEWMOD, tz=None,
         anchor='@pytest.mark.parametrize("offset", [-1, -2, -7, -365])',
         repl='@pytest.mark.parametrize("offset", [0])',
         desc="point the how-stale sweep at a FRESH anchor"),
]


def digest(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def run(path: pathlib.Path, tz: str | None, extra: dict | None = None,
        mutant: pathlib.Path | None = None) -> tuple[int, str]:
    env = {**os.environ}
    if tz:
        env["TZ"] = tz
    for k, v in (extra or {}).items():
        env[k] = str(mutant) if v == "<MUTANT>" and mutant else v
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(path), "-q", "-p", "no:randomly"],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600)
    return proc.returncode, (proc.stdout or "")[-300:]


def main() -> int:
    tracked = [MACRO, POISON, NEWMOD]
    before = {p: digest(p) for p in tracked}
    rows, survived = [], []

    for cell in MUTATIONS:
        mid, src, anchor, repl = cell["id"], cell["src"], cell["anchor"], cell["repl"]
        tz, desc = cell.get("tz"), cell["desc"]
        run_target, extra = cell.get("run"), cell.get("env")
        text = src.read_text()
        n = text.count(anchor)
        if n != 1:
            rows.append((mid, "ANCHOR", f"matched {n} time(s)", desc))
            survived.append(mid)
            continue

        # The control runs the SAME file the mutant run will, unmutated, so a
        # cell can never score a kill on a target that was already red.
        control_rc, _ = run(run_target or src, tz)
        mut = src.with_name(f"test_zz_mutant_{mid.lower()}_{os.getpid()}.py")
        try:
            mut.write_text(text.replace(anchor, repl, 1))
            mut_rc, tail = run(run_target or mut, tz, extra, mut)
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
