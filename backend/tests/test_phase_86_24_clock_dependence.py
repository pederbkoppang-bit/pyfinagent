"""phase-86.24 -- the suite must not change colour with the wall clock.

WHAT THIS MODULE IS FOR
-----------------------
Three tests changed state at midnight CEST on 2026-08-09/10 with no code change,
and two healed themselves 45 minutes later. The step required the kill-switch
case to be **adjudicated rather than patched**, because "the daily safety anchor
goes stale at midnight" would be a real defect in a long-running backend.

**The adjudication came out: production is CORRECT, the tests were wrong -- and
the three tests had TWO DIFFERENT mechanisms, which is why one explanation could
never have covered them.**

1. `test_phase_86_2_replay_poison_row.py` pinned `2026-08-09` in a fixture and
   asserted the daily leg could fire. The kill switch judges the anchor against
   `datetime.now(timezone.utc).date()`, so from the next day the leg was
   correctly DISARMED and the assertion correctly failed. **A pinned fixture
   date judged against now.** Fixed by making the fixture relative.
2. `test_phase_82_0_macro_ingestion.py` (x2) asserted against **local**
   `date.today()` while production resolves in **UTC**. On CEST those disagree
   for exactly 00:00-02:00 nightly, then agree again. **A timezone-domain
   mismatch**, not a time bomb. Fixed by asking production's question in
   production's units.

The two mechanisms have flip instants two hours apart. No static analysis finds
the second class at all -- both sides call the clock, and nothing in the syntax
says which one is wrong.

THE COVERAGE THIS MODULE ADDS
-----------------------------
Making the poison-row fixture relative removes the accidental daily visit to the
STALE path. That path is a real safety rule (phase-36.9, installed against a
measured live incident) and it deserves a test that runs every day rather than
one that arrives by accident once a day. So it gets one here, deliberately.
"""
from __future__ import annotations

import datetime as _dt
import json
import pathlib

import pytest

import backend.services.kill_switch as ks

REPO = pathlib.Path(__file__).resolve().parents[2]


def _day(offset: int) -> str:
    return (_dt.datetime.now(_dt.timezone.utc).date()
            + _dt.timedelta(days=offset)).isoformat()


def _journal(tmp_path, sod_date: str) -> pathlib.Path:
    """A minimal well-formed journal with a SOD anchor on `sod_date`."""
    rows = [
        {"ts": f"{sod_date}T00:01:00+00:00", "event": "sod_snapshot",
         "nav": 100.0, "date": sod_date},
        {"ts": f"{sod_date}T00:02:00+00:00", "event": "peak_update", "nav": 100.0},
    ]
    p = tmp_path / "kill_switch_audit.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """Redirect the journal to tmp. The live operator journal is never touched."""
    monkeypatch.setattr(ks, "_AUDIT_PATH", tmp_path / "kill_switch_audit.jsonl")
    monkeypatch.setattr(ks, "_state", None, raising=False)
    return tmp_path


# ── the adjudication, pinned as a test that runs EVERY day ──────────────────

def test_a_TODAY_anchor_arms_the_daily_leg(isolated, monkeypatch):
    """Control for the test below. Without it, a kill switch that never armed
    would pass the staleness test perfectly."""
    _journal(isolated, _day(0))
    st = ks.KillSwitchState()
    monkeypatch.setattr(ks, "_state", st)
    assert st.snapshot()["sod_date"] == _day(0)
    r = ks.evaluate_breach(80.0, 4.0, 10.0)
    assert r["daily_baseline_stale"] is False, r
    assert r["armed"] is True, r
    assert r["daily_loss_breached"] is True, r


def test_a_YESTERDAY_anchor_DISARMS_the_daily_leg_but_the_trailing_leg_still_fires(
        isolated, monkeypatch):
    """THE ADJUDICATION. This is the behaviour phase-86.24 examined and found
    CORRECT, so it is pinned rather than removed.

    phase-36.9 made the daily leg unevaluable on a stale anchor after a MEASURED
    live incident on 2026-07-26: the badge served `sod_date=2026-07-24` with
    `armed: true`, and a TWO-DAY move was being reported as a same-day loss --
    losing same-day coverage and biasing toward a spurious flatten at once.

    The rule is PER-LEG, and that is the part worth asserting: the trailing
    drawdown limit is a high-water mark, not date-scoped, so it keeps firing
    while the daily leg sleeps. A reader who sees `armed: false` overnight
    should be able to point at this test for why the book is still covered.
    """
    _journal(isolated, _day(-1))
    st = ks.KillSwitchState()
    monkeypatch.setattr(ks, "_state", st)
    r = ks.evaluate_breach(80.0, 4.0, 10.0)

    assert r["daily_baseline_stale"] is True, r
    assert r["armed"] is False, r
    assert r["daily_loss_breached"] is False, (
        "an UNEVALUABLE leg must not fire -- that is the whole point of 36.9", r)
    # ...and the book is NOT uncovered while it sleeps:
    assert r["trailing_dd_breached"] is True, r
    assert r["any_breached"] is True, (
        "the trailing leg is date-independent and must still fire on a stale "
        "daily anchor; if this ever goes False the overnight window is naked", r)


@pytest.mark.parametrize("offset", [-1, -2, -7, -365])
def test_staleness_does_not_depend_on_HOW_stale(isolated, monkeypatch, offset):
    """A rule that only fired for yesterday would leave a hole for every older
    anchor. Derived offsets rather than one named case."""
    _journal(isolated, _day(offset))
    st = ks.KillSwitchState()
    monkeypatch.setattr(ks, "_state", st)
    r = ks.evaluate_breach(80.0, 4.0, 10.0)
    assert r["daily_baseline_stale"] is True, (offset, r)
    assert r["any_breached"] is True, (offset, r)


# ── criterion 5: no global time-freezing fixture ────────────────────────────

def test_no_global_time_freezing_fixture_is_introduced():
    """The step forbids this explicitly, and the reason is sharp: a global
    freeze would make the staleness rule above permanently unevaluable, hiding
    exactly the class of defect the rule exists to surface.

    Asserted over every conftest in the repo rather than the two this step
    touched -- a global fixture is global wherever it is declared.
    """
    suspects = ("freeze_time", "freezegun", "time_machine", "time-machine",
                "libfaketime", "FrozenDateTimeFactory", "travel(")
    offenders = []
    for cf in list(REPO.glob("conftest.py")) + list(REPO.glob("**/conftest.py")):
        if ".venv" in cf.parts or "node_modules" in cf.parts:
            continue
        text = cf.read_text(errors="replace")
        for s in suspects:
            if s in text:
                offenders.append(f"{cf.relative_to(REPO)}: {s}")
    assert not offenders, (
        "a time-freezing helper appeared in a conftest; phase-86.24 forbids a "
        f"GLOBAL freeze because it disarms the staleness rule: {offenders}"
    )


def test_the_two_repaired_modules_PASS_AT_A_SHIFTED_CLOCK():
    """The real property, asserted directly instead of by proxy.

    THE FIRST VERSION OF THIS TEST WAS UNSOUND AND IS RECORDED RATHER THAN
    QUIETLY REPLACED. It asserted "neither module carries a pinned calendar
    date", and it fired on sixteen lines that are all perfectly correct: the
    backtest cap `2025-12-31` (whose pinning is the DEFECT under test), the
    operator pin `2026-01-15` (whose being honoured is a criterion), receipt
    fixtures, and prose in docstrings. "No literal dates" is not the property --
    plenty of pinned dates are exactly what a test should pin.

    The property is: **the verdict must not depend on what day it is.** That is
    not statically decidable -- as the phase-86.24 research established, the
    timezone-domain class has literals on neither side, and only executing both
    sides reveals it. So it is checked the only way it can be: run the modules
    again with the clock shifted into a different calendar day and require the
    same result.

    `TZ=Pacific/Midway` puts the LOCAL date one day behind UTC, which is exactly
    the 00:00-02:00 CEST window in which the two macro tests used to fail. It
    does not move UTC, which is a stated limit rather than a hidden one: the
    "pinned fixture ages past UTC today" axis needs a real clock offset, and the
    step's artifacts carry that as an open operator ask.
    """
    import os
    import subprocess
    import sys

    targets = ["backend/tests/test_phase_86_2_replay_poison_row.py",
               "backend/tests/test_phase_82_0_macro_ingestion.py"]
    env = {**os.environ, "TZ": "Pacific/Midway"}
    proc = subprocess.run([sys.executable, "-m", "pytest", *targets, "-q",
                           "-p", "no:randomly"],
                          cwd=str(REPO), env=env, capture_output=True,
                          text=True, timeout=300)

    # Positive control: the shift must actually have taken effect, or this test
    # would pass by simply not shifting anything.
    shifted = subprocess.run(
        [sys.executable, "-c",
         "import datetime,time;print(datetime.date.today().isoformat(),"
         "datetime.datetime.now(datetime.timezone.utc).date().isoformat())"],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=60)
    local_d, utc_d = shifted.stdout.split()
    assert local_d != utc_d, (
        f"the TZ shift did not move the local date ({local_d} == {utc_d}); this "
        "test would have passed without testing anything"
    )

    assert proc.returncode == 0, (
        "the repaired modules FAIL when the local calendar day differs from the "
        f"UTC day -- the clock-dependence is still there:\n{proc.stdout[-2500:]}"
    )
