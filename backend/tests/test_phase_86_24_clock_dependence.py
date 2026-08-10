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
from zoneinfo import ZoneInfo

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


def test_a_YESTERDAY_anchor_DISARMS_the_daily_leg(isolated, monkeypatch):
    """THE ADJUDICATION -- and the rationale here was WRONG in cycle 1.

    phase-36.9 makes the daily leg unevaluable on a stale anchor, after a
    MEASURED live incident on 2026-07-26 in which the badge served
    `sod_date=2026-07-24` with `armed: true` and a TWO-DAY move was reported as
    a same-day loss -- losing same-day coverage and biasing toward a spurious
    flatten at once.

    **What cycle 1 claimed, and what is actually true.** This test was called
    `..._but_the_trailing_leg_still_fires` and asserted that the overnight
    window "is not naked" because the date-independent trailing leg keeps
    firing. The cycle-1 Q/A measured that to be FALSE IN A BAND, and I
    reproduced it:

        anchor    nav   armed  stale  daily  trailing  ANY
        STALE     99.0  False  True   False  False     False
        STALE     95.0  False  True   False  False     False     <-- 5% loss
        STALE     92.0  False  True   False  False     False     <-- 8% loss
        STALE     89.0  False  True   False  True      True
        TODAY     95.0  True   False  True   False     True

    Between the daily limit (4%) and the trailing limit (10%) a stale anchor
    leaves **nothing** firing. The cycle-1 guard only exercised `nav=80.0` -- a
    20% drop, above the trailing limit -- so it could not detect the gap it
    claimed to close. That is asserting a general property from a single point.

    **The conclusion is unchanged and still correct, but for a different
    reason**: the enforcement path never evaluates against a stale anchor.
    Measured in `paper_trader.check_and_enforce_kill_switch`: the re-anchor at
    `:1413` (`if sod_anchor_needs_reroll(snap, today)`) runs BEFORE
    `evaluate_breach` at `:1460`, and the flatten branch at `:1468` keys on
    `breach["any_breached"]`, never on `armed`. The order gate at `:1372` reads
    `baselines_present`, also never `armed`.

    So the band above is reachable only by a READ-ONLY caller (the badge
    endpoint), never by the code that decides whether to flatten. That is why
    this is not a live defect -- not because the trailing leg covers it.
    """
    _journal(isolated, _day(-1))
    st = ks.KillSwitchState()
    monkeypatch.setattr(ks, "_state", st)
    r = ks.evaluate_breach(80.0, 4.0, 10.0)

    assert r["daily_baseline_stale"] is True, r
    assert r["armed"] is False, r
    assert r["daily_loss_breached"] is False, (
        "an UNEVALUABLE leg must not fire -- that is the whole point of 36.9", r)
    # Above the TRAILING limit the trailing leg does still fire. This is true
    # here and is NOT true across the whole range -- see the band test below.
    assert r["trailing_dd_breached"] is True, r
    assert r["any_breached"] is True, r


def test_a_stale_anchor_leaves_the_band_between_the_two_limits_UNCOVERED(
        isolated, monkeypatch):
    """The measurement that refutes cycle 1's rationale, pinned as a test.

    This asserts an UNCOMFORTABLE fact rather than a reassuring one, and that is
    deliberate: the previous version asserted the comfortable claim and was
    wrong. A loss between the daily and trailing limits, evaluated against a
    STALE anchor, breaches neither leg.

    **This is safe ONLY because the enforcement path re-anchors first**
    (`paper_trader.py:1413` before `:1460`), so the flatten decision never sees
    a stale anchor. If that ordering is ever changed, this test is the one that
    should be re-read: it documents exactly what the ordering is protecting.
    """
    _journal(isolated, _day(-1))
    st = ks.KillSwitchState()
    monkeypatch.setattr(ks, "_state", st)

    for nav in (95.0, 92.0):                 # 5% and 8% -- inside (4%, 10%)
        r = ks.evaluate_breach(nav, 4.0, 10.0)
        assert r["armed"] is False, (nav, r)
        assert r["any_breached"] is False, (
            f"nav={nav} against a stale anchor now reports a breach; the band "
            "this test documents has changed and the adjudication in "
            "experiment_results_86.24.md must be re-derived", r)

    # CONTROL: the same losses against a FRESH anchor DO breach, so the result
    # above is the staleness rule and not an inert threshold.
    _journal(isolated, _day(0))
    fresh = ks.KillSwitchState()
    monkeypatch.setattr(ks, "_state", fresh)
    for nav in (95.0, 92.0):
        r = ks.evaluate_breach(nav, 4.0, 10.0)
        assert r["armed"] is True and r["any_breached"] is True, (nav, r)


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
    # phase-86.34: SCOPE THE POPULATION, and report it.
    # The rule was `".venv" in cf.parts` -- an EXACT path-element match, which
    # does not exclude sibling virtualenvs like `.venv.py313.bak` (gitignored at
    # .gitignore:16 as `.venv*/`). MEASURED before the fix: 70 conftest.py in the
    # tree, 34 kept, of which **32 were vendored third-party files** and only 2
    # were the project's own. The guard was green by the luck of what the present
    # vendored corpus happens to contain -- a future `pip install` of a package
    # whose conftest mentions freezegun would have turned it red for a reason
    # unrelated to any change here.
    def _first_party(path) -> bool:
        return not any(part.startswith(".venv") or part == "node_modules"
                       for part in path.parts)

    # phase-86.34: a TEST-ONLY seam so the guard can be pointed at a fake repo
    # root. Criterion 3 requires proving this assertion goes RED when a conftest
    # declaring a global time-freezing fixture exists -- and the repo, by
    # construction, contains no such file. Without a seam the RED half of the
    # cell is unprovable and the guard is only ever observed passing.
    # Defaults to the real REPO; mirrors the existing seam at
    # PYFINAGENT_86_24_PROW_PATH below. Read in this test only.
    import os

    sweep_root = pathlib.Path(os.environ.get("PYFINAGENT_86_34_SWEEP_ROOT", str(REPO)))

    offenders = []
    swept = [cf for cf in set(list(sweep_root.glob("conftest.py"))
                              + list(sweep_root.glob("**/conftest.py")))
             if _first_party(cf)]
    # A sweep that resolves to nothing reports "no offenders" identically to a
    # sweep that resolves to everything and finds none. Assert the population.
    assert swept, "the conftest sweep matched ZERO files -- the scan is vacuous, not clean"
    # ...and assert the population is FIRST-PARTY, not merely non-empty.
    # phase-86.34 mutation N2-REVERT-EXCLUSION: reverting to the exact-element
    # `".venv" in parts` rule left the suite GREEN, because the 32 vendored
    # conftests it re-admits happen to contain no suspect token today. Asserting
    # only "non-empty" cannot see that. This asserts the PROPERTY the exclusion
    # rule exists to establish, so weakening the rule fires a named assertion
    # instead of waiting for a future `pip install` to make it red by accident.
    vendored = [p for p in swept if any(x.startswith(".venv") or x == "node_modules"
                                        for x in p.parts)]
    assert not vendored, (
        f"the conftest sweep admitted {len(vendored)} VENDORED file(s) -- the scan "
        f"population must be first-party only, else it is green by the luck of what "
        f"the vendored corpus happens to contain: {[str(x) for x in vendored[:4]]}"
    )
    print(f"[86.34] conftest sweep population: {len(swept)} first-party file(s): "
          f"{sorted(str(p.relative_to(sweep_root)) for p in swept)}")
    for cf in swept:
        text = cf.read_text(errors="replace")
        for s in suspects:
            if s in text:
                offenders.append(f"{cf.relative_to(sweep_root)}: {s}")
    assert not offenders, (
        "a time-freezing helper appeared in a conftest; phase-86.24 forbids a "
        f"GLOBAL freeze because it disarms the staleness rule: {offenders}"
    )



def _date_shifting_tz() -> str:
    """A timezone whose LOCAL date differs from the UTC date RIGHT NOW.

    phase-86.34. A fixed offset shifts the date on only `|offset|` of the 24 UTC
    hours, so a hardcoded zone makes this suite pass or fail by the clock rather
    than by the code. Midway (UTC-11, hours 00:00-10:59) and Kiritimati (UTC+14,
    hours 10:00-23:59) cover the whole day between them.

    Returns the FIRST candidate that actually shifts the date now. If none does
    -- which should be impossible given the coverage above -- it returns the
    first candidate anyway and lets the positive control in the caller fail
    loudly. Silently returning something that does not shift the date is exactly
    the failure this function exists to remove, so it is never done here.
    """
    now = _dt.datetime.now(_dt.timezone.utc)
    for name in ("Pacific/Midway", "Pacific/Kiritimati"):
        if now.astimezone(ZoneInfo(name)).date() != now.date():
            return name
    return "Pacific/Midway"

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

    THE ZONE IS CHOSEN AT RUNTIME, and phase-86.34 is why.

    This used to hardcode `TZ=Pacific/Midway` and claim it "puts the LOCAL date
    one day behind UTC, which is exactly the 00:00-02:00 CEST window in which the
    two macro tests used to fail". Both halves were wrong:

      * DIRECTION -- at 00:30/01:30 CEST the local date is one day AHEAD of UTC.
        Midway (UTC-11) is BEHIND. The fixture was the MIRROR of the window it
        named. Harmless on its own: the operative property is `local date !=
        UTC date`, and both directions satisfy it.
      * COVERAGE -- and this is the real defect. A FIXED offset does not
        guarantee a date shift at all; for an offset of `o` hours it holds on
        exactly `|o|` of the 24 UTC hours. MEASURED: Midway shifts the date only
        for UTC hours 00:00-10:59 (11/24). For the other 13 hours the fixture
        established NOTHING, and the positive control below correctly turned the
        whole suite RED -- for a reason unrelated to any code change.

    So the suite was green only if you ran it in the right third of the day.
    phase-86.24 closed on a PASS at roughly 10:5x UTC, about five minutes inside
    Midway's window; the same command that evening was red.

    The fix is not a luckier timezone. `_date_shifting_tz()` picks a zone that
    PROVABLY shifts the date at this moment -- Midway (UTC-11) and Kiritimati
    (UTC+14) cover 24/24 hours between them -- and the positive control is KEPT
    so the choice is verified rather than trusted. Reproduce the coverage with
    `python scripts/qa/measure_tz_fixture_coverage_86_34.py`.

    It still does not move UTC, which is a stated limit rather than a hidden
    one: the "pinned fixture ages past UTC today" axis needs a real clock offset,
    and the step's artifacts carry that as an open operator ask.
    """
    import os
    import subprocess
    import sys

    targets = ["backend/tests/test_phase_86_2_replay_poison_row.py",
               "backend/tests/test_phase_82_0_macro_ingestion.py"]
    env = {**os.environ, "TZ": _date_shifting_tz()}
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


def test_the_poison_row_fixture_date_is_RECOMPUTED_not_snapshotted():
    """The cycle-1 Q/A's second finding, pinned so it cannot come back.

    `test_phase_86_2_replay_poison_row.py` originally read
    `_UTC_TODAY = datetime.now(timezone.utc).date()` ONCE at module import,
    while the value it is judged against (`kill_switch.py:986`) recomputes at
    call time. If UTC midnight fell between collection and execution the fixture
    would write yesterday's anchor and the test would go red -- the masterplan's
    own definition of case (a), "a fixture that hard-codes or ONCE-COMPUTES a
    date while the assertion recomputes it", introduced by the very step that
    exists to remove it.

    Why this test rather than another mutation cell: the failure only manifests
    if the clock crosses midnight mid-run, so a mutant that re-snapshots CANNOT
    be killed by an ordinary run -- it is an equivalent mutant under normal
    conditions. Asserting the PROPERTY directly, with an injected clock that
    advances a day between calls, makes it killable.
    """
    import importlib.util
    import types

    # TEST SEAM, disclosed: the mutation matrix needs to point this at a mutant
    # COPY, because it reads the module by PATH and a copy elsewhere would
    # otherwise never be exercised. Unset in every normal run.
    import os
    target = os.environ.get(
        "PYFINAGENT_86_24_PROW_PATH",
        str(REPO / "backend" / "tests" / "test_phase_86_2_replay_poison_row.py"))
    spec = importlib.util.spec_from_file_location(
        "_prow_for_recompute_check", target)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    real_day = _dt.datetime.now(_dt.timezone.utc)
    seq = iter([real_day, real_day + _dt.timedelta(days=1)])
    mod._dt = types.SimpleNamespace(
        datetime=types.SimpleNamespace(now=lambda tz=None: next(seq)),
        timezone=_dt.timezone,
        timedelta=_dt.timedelta,
    )

    first, second = mod._day(0), mod._day(0)
    assert first != second, (
        "the fixture date is SNAPSHOTTED, not recomputed: two calls straddling "
        f"a clock advance both returned {first!r}. That is the once-computes "
        "shape phase-86.24 exists to remove."
    )
    assert second == (real_day + _dt.timedelta(days=1)).date().isoformat()
