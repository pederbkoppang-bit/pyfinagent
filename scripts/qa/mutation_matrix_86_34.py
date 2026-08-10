#!/usr/bin/env python3
"""Mutation matrix for phase-86.34 -- the three guards 86.34 added.

Successor to `scripts/qa/mutation_matrix_86_24.py` (criterion 3 permits
"or a successor"). It exists because 86.34 added three assertions to
`backend/tests/test_phase_86_24_clock_dependence.py` and an assertion nobody
has watched fail is not yet a guard.

WHY THE N1 CELL IS NOT "HARDCODE Pacific/Midway"
------------------------------------------------
The obvious mutant for `_date_shifting_tz()` is to revert it to the literal
`"Pacific/Midway"` it replaced. That cell is USELESS FOR HALF THE DAY: Midway
(UTC-11) genuinely shifts the calendar date during 00:00-10:59 UTC, so between
those hours the mutant is CORRECT and survives -- the matrix would report a
survivor that is really a fact about the wall clock. phase-86.34's own
experiment_results recorded that trap after hitting it.

So the cell instead hardcodes whichever candidate zone does NOT shift the date
at THIS moment, chosen at run time. That mutates the exact property the
function exists to establish -- "the zone is picked so that it always shifts"
-- and it kills at every hour of the day. The zone actually used is printed so
a reader can reproduce the run.

Run:  source .venv/bin/activate && python scripts/qa/mutation_matrix_86_34.py
Exit: 0 = every cell KILLED; 1 = at least one SURVIVED or a bad anchor.
"""
from __future__ import annotations

import datetime as _dt
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

REPO = pathlib.Path(__file__).resolve().parents[2]
NEWMOD = REPO / "backend/tests/test_phase_86_24_clock_dependence.py"

CANDIDATES = ("Pacific/Midway", "Pacific/Kiritimati")


def _non_shifting_zone() -> str:
    """A zone whose local date EQUALS the UTC date right now.

    This is the mirror of the helper under test. If every candidate happens to
    shift (impossible given the two offsets cover 24/24 hours, but never
    assumed), fall back to UTC, which shifts by definition never.
    """
    now = _dt.datetime.now(_dt.timezone.utc)
    for name in CANDIDATES:
        if now.astimezone(ZoneInfo(name)).date() == now.date():
            return name
    return "UTC"


def run(target: pathlib.Path, tz: str | None = None,
        extra_env: dict | None = None, kexpr: str | None = None) -> tuple[int, str]:
    env = {**os.environ}
    if tz:
        env["TZ"] = tz
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, "-m", "pytest", str(target), "-q"]
    if kexpr:
        cmd += ["-k", kexpr]
    p = subprocess.run(
        cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=600,
    )
    # FULL output: the poison cell discriminates on the NAMED assertion text,
    # and a 600-char tail can hold nothing but pip warnings. Callers truncate.
    return p.returncode, (p.stdout + p.stderr)


def _poisoned_repo() -> pathlib.Path:
    """A fake repo root whose only conftest declares a GLOBAL time freeze.

    The real repo contains no such file by construction, so without this the
    RED half of criterion 3 is unprovable and the guard would only ever be
    observed passing. Written under the system temp dir, never inside REPO.
    """
    root = pathlib.Path(tempfile.mkdtemp(prefix="pyfin_86_34_poison_")) / "sub"
    root.mkdir(parents=True)
    (root / "conftest.py").write_text(
        "import pytest\n"
        "from freezegun import freeze_time\n\n"
        "@pytest.fixture(autouse=True, scope='session')\n"
        "def _global_freeze():\n"
        "    with freeze_time('2026-01-01'):\n"
        "        yield\n"
    )
    return root.parent


FROZEN_ZONE = _non_shifting_zone()

MUTATIONS = [
    dict(
        id="N1-HARDCODE-NONSHIFTING-TZ",
        src=NEWMOD,
        anchor='    env = {**os.environ, "TZ": _date_shifting_tz()}',
        repl=f'    env = {{**os.environ, "TZ": "{FROZEN_ZONE}"}}',
        desc=f"replace the runtime zone selector with a zone that does NOT shift "
             f"the date now ({FROZEN_ZONE}) -- the positive control must fire",
    ),
    dict(
        id="N2-REVERT-EXCLUSION",
        src=NEWMOD,
        anchor='        return not any(part.startswith(".venv") or part == "node_modules"',
        repl='        return not any(part == ".venv" or part == "node_modules"',
        desc="revert the sweep to the EXACT-element rule -- it re-admits the "
             "vendored .venv.py313.bak corpus, so the first-party assertion must fire",
    ),
    dict(
        id="N2-EMPTY-POPULATION",
        src=NEWMOD,
        anchor="    def _first_party(path) -> bool:",
        repl="    def _first_party(path) -> bool:\n        return False  # MUTANT",
        desc="make the sweep match NOTHING -- the non-vacuity assertion must fire "
             "(the guard is broken at its SUBJECT, not deleted)",
    ),
    # NOT a source mutation. Criterion 3 has TWO halves and this is the first:
    # "inject a conftest containing a global time-freezing fixture into a fake
    # repo root and require the guard to go RED". The subject is the ENVIRONMENT,
    # so the file is run unmutated with the sweep pointed at a poisoned root.
    dict(
        id="N2-POISONED-CONFTEST",
        src=NEWMOD,
        anchor=None,
        repl=None,
        kexpr="test_no_global_time_freezing",
        poison=True,
        desc="inject a conftest declaring a session-scoped freeze_time into a fake "
             "repo root -- the named offenders assertion must fire",
    ),
]


def main() -> int:
    print(f"repo   : {REPO}")
    print(f"utc    : {_dt.datetime.now(_dt.timezone.utc):%Y-%m-%d %H:%M} "
          f"-> non-shifting zone chosen for N1: {FROZEN_ZONE}")
    print()

    rows, survived = [], []
    for cell in MUTATIONS:
        mid, src, anchor, repl, desc = (
            cell["id"], cell["src"], cell["anchor"], cell["repl"], cell["desc"])
        text = src.read_text()

        if cell.get("poison"):
            # Control FIRST: the same -k selection must be GREEN against the
            # real repo, else a RED under the poisoned root proves nothing.
            ctl_rc, ctl_tail = run(src, kexpr=cell["kexpr"])
            if ctl_rc != 0:
                rows.append((mid, "CONTROL-RED", f"already failing clean: {ctl_tail[-120:]}"))
                survived.append(mid)
                continue
            fake = _poisoned_repo()
            try:
                rc, tail = run(src, extra_env={"PYFINAGENT_86_34_SWEEP_ROOT": str(fake)},
                               kexpr=cell["kexpr"])
            finally:
                shutil.rmtree(fake.parent, ignore_errors=True)
            # A kill must come from the NAMED assertion, not from any old error.
            if rc != 0 and "a time-freezing helper appeared in a conftest" in tail:
                rows.append((mid, "KILLED", desc))
            elif rc != 0:
                rows.append((mid, "MIS-ATTRIBUTED",
                             f"went RED but not on the named assertion: {tail[-160:]}"))
                survived.append(mid)
            else:
                rows.append((mid, "SURVIVED", f"{desc} || stayed GREEN: {tail[-160:]}"))
                survived.append(mid)
            continue

        n = text.count(anchor)
        if n != 1:
            rows.append((mid, "ANCHOR", f"matched {n} time(s) -- STALE ANCHOR"))
            survived.append(mid)
            continue

        # Control: the unmutated file must be GREEN, else a "kill" proves nothing.
        control_rc, control_tail = run(src)
        if control_rc != 0:
            rows.append((mid, "CONTROL-RED", f"unmutated suite already fails: {control_tail[-120:]}"))
            survived.append(mid)
            continue

        mut = src.with_name(f"test_zz_mut_{mid.lower().replace('-', '_')}_{os.getpid()}.py")
        try:
            mut.write_text(text.replace(anchor, repl, 1))
            mut_rc, tail = run(mut)
        finally:
            mut.unlink(missing_ok=True)

        if mut_rc != 0:
            rows.append((mid, "KILLED", desc))
        else:
            rows.append((mid, "SURVIVED", f"{desc} || mutant stayed GREEN: {tail[-160:]}"))
            survived.append(mid)

    width = max(len(r[0]) for r in rows)
    for mid, verdict, note in rows:
        print(f"  {mid:<{width}}  {verdict:<12} {note[:140]}")
    print()
    if survived:
        print(f"FAIL -- {len(survived)} cell(s) did not kill: {survived}")
        return 1
    print(f"OK -- all {len(rows)} cell(s) KILLED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
