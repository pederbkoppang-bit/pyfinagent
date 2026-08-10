"""phase-86.2: one malformed audit row must not strand the whole kill switch.

THE DEFECT. `_coerce_nav` caught only `(TypeError, ValueError)`. `OverflowError`
is `ArithmeticError`'s child -- a SIBLING branch -- so `float(a JSON integer of
310..4300 decimal digits)` escaped it, and the broad handler around the replay
loop swallowed the abort. Every row after the bad one was silently lost.
Measured before the fix, with the bad row FIRST in `ts` order:

    sod_nav=None sod_date=None peak_nav=None
    armed: False   daily 0.0%   trailing 0.0%   any_breached: FALSE

on a 20% drawdown. That is the only known path to a TOTAL disarm -- every other
degraded state leaves the date-independent trailing leg firing.

ORDERING, DISCLOSED. Criterion 1 asks for the red test FIRST. The RED state was
captured before any change via the diagnostic (contract SS3), but the research
gate then established that the diagnostic PRINTS case E without ASSERTING it --
it exits 0 with the defect fully present -- so it was a witness, not a test.
This file was therefore written AFTER the production fix. It is not taken on
trust: `test_c4_mutation_*` reverts the widened `except` against an isolated
copy of the module and shows these same assertions fail, which is the evidence
criterion 1 wanted. Stated here rather than left for a reader to notice.

ISOLATION follows the phase-86.1 idioms landed the same day: an autouse
byte-identity fixture over the live journal, a DETACHED `KillSwitchState`, and
redirect-before-construction (`__init__` replays).
"""

from __future__ import annotations

import importlib.util
import datetime as _dt
import json
import pathlib

import pytest

import backend.services.kill_switch as ks

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_LIVE_AUDIT = REPO_ROOT / "handoff" / "kill_switch_audit.jsonl"

# The measured trigger window: a JSON integer of 310..4300 decimal digits.
# Below 310 `float()` succeeds; above 4300 `json.loads` itself raises ValueError
# (int_max_str_digits) and the pre-existing per-line handler already caught it.
_POISON_INT = "1" + "0" * 400

# ── phase-86.24: the fixture's day must track the clock ─────────────────────
# This module's fixtures used to pin "2026-08-09" -- the day they were written.
# The kill switch judges a daily anchor against `datetime.now(timezone.utc).date()`
# (kill_switch.py:986 via `_sod_date_is_stale`), so from 2026-08-10 onward the
# anchor read STALE, the daily leg disarmed, and the assertion below that the
# switch "can now actually fire" failed -- on any day but one.
#
# ADJUDICATED in phase-86.24: the staleness rule is CORRECT. It is per-LEG (the
# date-independent trailing leg still fires), the order gate reads
# `baselines_present` rather than `armed`, and phase-36.9 installed it against a
# MEASURED live incident where a two-day move was reported as a same-day loss.
# So the assertion is NOT relaxed -- the fixture is made relative, and the stale
# case gets its own test below instead of arriving by accident once a day.
# RECOMPUTED PER CALL, NOT SNAPSHOTTED AT IMPORT -- and the first version of
# this helper got that wrong, in the file this very step was repairing.
#
# It read `_UTC_TODAY = datetime.now(timezone.utc).date()` at module level. The
# value it is judged against (`kill_switch.py:986`, `_sod_date_is_stale`)
# recomputes at CALL time, so if UTC midnight fell between this module's import
# at collection and the execution of the test below, the fixture would write
# YESTERDAY's anchor, the daily leg would correctly disarm, and the test would
# go red -- which is precisely the failure phase-86.24 exists to remove, and
# precisely the masterplan's own definition of case (a): "a fixture that
# hard-codes or ONCE-COMPUTES a date while the assertion recomputes it".
#
# The evidence was already in this step's own mutation matrix and I misread it:
# cell M2 pins `_UTC_TODAY` to a past date and is scored KILLED, i.e. the module
# is PROVEN to go red whenever the import-time snapshot is a day behind
# evaluation time. I read that as the guard working rather than as my own helper
# being a member of the class. Found by the cycle-1 Q/A.
#
# The window was minutes rather than 24h (collection -> execution, ~376s on a
# full run), but a narrower instance of a defect is still an instance of it.


def _day(offset_days: int = 0) -> str:
    return (_dt.datetime.now(_dt.timezone.utc).date()
            + _dt.timedelta(days=offset_days)).isoformat()


def _ts(offset_days: int = 0, hhmmss: str = "00:00:00") -> str:
    return f"{_day(offset_days)}T{hhmmss}+00:00"


@pytest.fixture(autouse=True)
def _live_kill_switch_journal_is_byte_identical():
    """No test here may touch the operator's journal (phase-86.1 idiom)."""
    before = _LIVE_AUDIT.read_bytes() if _LIVE_AUDIT.exists() else None
    yield
    after = _LIVE_AUDIT.read_bytes() if _LIVE_AUDIT.exists() else None
    assert after == before, (
        "this test wrote to the LIVE kill-switch journal: "
        f"{0 if before is None else len(before.splitlines())} -> "
        f"{0 if after is None else len(after.splitlines())} lines"
    )


def _write(tmp_path, *rows: str) -> pathlib.Path:
    p = tmp_path / "kill_switch_audit.jsonl"
    p.write_text("".join(r + "\n" for r in rows), encoding="utf-8")
    return p


def _row(ts: str, **fields) -> str:
    return json.dumps({"ts": ts, **fields})


def _isolate(monkeypatch, tmp_path):
    """Redirect BEFORE constructing state -- __init__ replays."""
    monkeypatch.setattr(ks, "_AUDIT_PATH", tmp_path / "kill_switch_audit.jsonl")
    assert ks._audit_archive_dir() == tmp_path / "audit"
    assert not any(p == _LIVE_AUDIT for p in ks._audit_source_paths()), (
        f"a LIVE audit file is in the replay set: {ks._audit_source_paths()}"
    )


# ---------------------------------------------------------------------------
# Criterion 1 + 2 -- rows on BOTH sides of the poison row still apply.
# ---------------------------------------------------------------------------

def test_c1_c2_a_poison_row_first_no_longer_strands_the_replay(monkeypatch, tmp_path):
    """The reproduction. Poison row FIRST in ts order -- the ordering matters and
    a previous author got it wrong: placed LAST, the good rows have already
    applied and nothing is stranded, so the run contradicts the claim above it.

    Criterion 2 additionally requires a well-formed row BEFORE the bad one as
    well as after, so this fixture brackets it.
    """
    _isolate(monkeypatch, tmp_path)
    _write(
        tmp_path,
        # BEFORE the poison row (criterion 2's "before" half)
        _row(_ts(0, "00:00:00"), event="peak_update", nav=90.0),
        # the poison row
        '{"ts": "%s", "event": "peak_update", "nav": %s}' % (_ts(0, "00:00:30"), _POISON_INT),
        # AFTER it (criterion 2's "after" half)
        _row(_ts(0, "00:01:00"), event="sod_snapshot", nav=100.0, date=_day(0)),
        _row(_ts(0, "00:02:00"), event="peak_update", nav=100.0),
    )

    st = ks.KillSwitchState()          # DETACHED, replays on construction
    snap = st.snapshot()

    # Every WELL-FORMED row applied -- the ones before AND after the bad one.
    assert snap["sod_nav"] == 100.0, snap
    assert snap["sod_date"] == _day(0), snap
    assert snap["peak_nav"] == 100.0, snap

    # ...and the switch can now actually fire on a real 20% drawdown.
    monkeypatch.setattr(ks, "_state", st)
    r = ks.evaluate_breach(80.0, 4.0, 10.0)
    assert r["daily_loss_breached"] is True, r
    assert r["trailing_dd_breached"] is True, r
    assert r["any_breached"] is True, r


def test_c2_the_poison_value_itself_is_still_rejected(monkeypatch, tmp_path):
    """Skipping the ROW must not mean accepting the VALUE. The bad peak_update
    must not become the peak."""
    _isolate(monkeypatch, tmp_path)
    _write(
        tmp_path,
        '{"ts": "%s", "event": "peak_update", "nav": %s}' % (_ts(0, "00:00:30"), _POISON_INT),
        _row(_ts(0, "00:01:00"), event="peak_update", nav=100.0),
    )
    st = ks.KillSwitchState()
    assert st.snapshot()["peak_nav"] == 100.0, "the oversized value must never become the peak"


def test_the_trigger_window_is_what_we_think_it_is():
    """Pin the measured bound, so a future reader need not re-derive it.

    `_coerce_nav` must return None (not raise) across the whole window, and the
    values that were ALREADY safe must stay safe.
    """
    assert ks._coerce_nav(int("1" + "0" * 400)) is None      # in-window -> OverflowError path
    assert ks._coerce_nav(float("inf")) is None              # already guarded by isfinite
    assert ks._coerce_nav(float("nan")) is None
    assert ks._coerce_nav("1e400") is None                   # string -> inf -> isfinite
    assert ks._coerce_nav(None) is None
    assert ks._coerce_nav("abc") is None
    assert ks._coerce_nav([1]) is None
    # do-no-harm: values that worked before still work
    assert ks._coerce_nav(100.0) == 100.0
    assert ks._coerce_nav("100.0") == 100.0
    assert ks._coerce_nav(" 12 ") == 12.0


# ---------------------------------------------------------------------------
# Criterion 3 -- the fail-safe direction.
# ---------------------------------------------------------------------------

def test_c3_a_skipped_row_marks_history_INCOMPLETE(monkeypatch, tmp_path):
    """The load-bearing half: a skipped row must mark history INCOMPLETE, or a
    later anchor-from-None can claim authority over history it never saw --
    reopening the hole phase-36.8 closed.

    THIS TEST WAS VACUOUS IN ITS FIRST FORM AND THE Q/A PROVED IT (EVALUATE
    pass 1, wf_f5f6f4b7-5cd). It asserted `_history_complete is False` while its
    fixture omitted `(tmp_path/"audit").mkdir()` -- so the MISSING ARCHIVE DIR
    already forced the flag False before any row was applied. Deleting
    `self._history_complete = False` from the production per-row handler left
    this test GREEN. The Q/A's 2x2 (poison x archive-dir) showed the flag
    tracked the archive dir and NEVER the poison row.

    Two corrections, both required for this to assert anything:

    1. The archive dir is CREATED, so the flag starts True and only the skip can
       move it. Without this the assertion is fixture-guaranteed.
    2. The fault is INJECTED at the seam. With the widened `_coerce_nav`, the
       case-E poison row no longer RAISES -- it coerces to None and `peak_update`
       no-ops -- so the per-row `except` is never entered by that input. The Q/A
       tried five candidate rows and reached it with none; neither of us found a
       natural trigger. Rather than assert a branch nothing executes, the fault
       is injected by making `_apply_audit_row` raise for one specific row. That
       tests the HANDLER and its bookkeeping, which is what criterion 3 is about,
       and it says plainly that the trigger is synthetic.
    """
    _isolate(monkeypatch, tmp_path)
    (tmp_path / "audit").mkdir(exist_ok=True)      # (1) -- else fixture-guaranteed
    _write(
        tmp_path,
        _row(_ts(0, "00:00:00"), event="peak_update", nav=90.0),
        _row(_ts(0, "00:00:30"), event="peak_update", nav=95.0),   # made to raise
        _row(_ts(0, "00:01:00"), event="sod_snapshot", nav=100.0, date=_day(0)),
    )

    # PRECONDITION: without an injected fault this replay is CLEAN and the flag
    # is True. If this ever fails, the assertion below proves nothing again.
    assert ks.KillSwitchState()._history_complete is True, (
        "PRECONDITION: a clean replay of this fixture must report COMPLETE, "
        "otherwise the post-injection assertion is fixture-guaranteed"
    )

    real_apply = ks.KillSwitchState._apply_audit_row

    def _raise_on_the_middle_row(self, row):                      # (2)
        if row.get("nav") == 95.0:
            raise RuntimeError("injected fault at the per-row seam")
        return real_apply(self, row)

    monkeypatch.setattr(ks.KillSwitchState, "_apply_audit_row", _raise_on_the_middle_row)
    st = ks.KillSwitchState()

    assert st._history_complete is False, (
        "a skipped row must mark history INCOMPLETE"
    )
    # ...and the replay CONTINUED: the rows either side of the fault applied.
    snap = st.snapshot()
    assert snap["peak_nav"] == 90.0, snap        # before the fault
    assert snap["sod_nav"] == 100.0, snap        # after it


def test_c3_MUTATION_the_bookkeeping_is_load_bearing(monkeypatch, tmp_path):
    """The mutation the previous form could not survive: delete
    `self._history_complete = False` from the per-row handler and this must go
    RED. Run against an isolated module COPY, never the real module.

    This is the guard the Q/A named as missing -- without it, criterion 3's
    assertion cannot distinguish a working bookkeeping from a deleted one.
    """
    mut = _load_mutated_kill_switch_multi(
        tmp_path,
        [("                except Exception as e:\n                    self._history_complete = False",
          "                except Exception as e:\n                    pass  # MUTANT: bookkeeping deleted")],
    )
    audit = tmp_path / "c3_mutant.jsonl"
    (tmp_path / "audit").mkdir(exist_ok=True)
    audit.write_text(
        _row(_ts(0, "00:00:30"), event="peak_update", nav=95.0) + "\n"
        + _row(_ts(0, "00:01:00"), event="sod_snapshot", nav=100.0, date=_day(0)) + "\n",
        encoding="utf-8",
    )
    mut._AUDIT_PATH = audit

    real_apply = mut.KillSwitchState._apply_audit_row

    def _raise_on_the_middle_row(self, row):
        if row.get("nav") == 95.0:
            raise RuntimeError("injected fault at the per-row seam")
        return real_apply(self, row)

    monkeypatch.setattr(mut.KillSwitchState, "_apply_audit_row", _raise_on_the_middle_row)
    st = mut.KillSwitchState()
    assert st._history_complete is True, (
        "MUTANT SURVIVED: deleting the per-row bookkeeping did not change the "
        "flag, so the criterion-3 assertion is not load-bearing"
    )


def test_c3_a_clean_replay_still_reports_history_COMPLETE(monkeypatch, tmp_path):
    """Do-no-harm: the flag must not become permanently False.

    NOTE, measured against HEAD before relying on it: a tmp replay reports
    `_history_complete=False` even with NO bad rows, because the derived archive
    directory does not exist and an unreadable SOURCE already sets the flag --
    pre-existing phase-36.8 behaviour, not a phase-86.2 regression. My first
    version of this test asserted True and failed for that reason. The archive
    dir is therefore created, so this asserts what it claims to.
    """
    _isolate(monkeypatch, tmp_path)
    (tmp_path / "audit").mkdir(exist_ok=True)
    _write(tmp_path, _row(_ts(0, "00:01:00"), event="peak_update", nav=100.0))
    st = ks.KillSwitchState()
    assert st._history_complete is True, (
        "a replay with every source readable and no bad row must report COMPLETE"
    )
    assert st.snapshot()["peak_nav"] == 100.0


# ---------------------------------------------------------------------------
# Criterion 4 -- the mutation. Also the evidence for the ordering disclosure.
# ---------------------------------------------------------------------------

def _load_mutated_kill_switch_multi(tmp_path, subs):
    """Import a COPY of kill_switch.py with N substitutions applied.

    A copy, never the real module: mutating the installed module would leak into
    every other test in the session.
    """
    src = pathlib.Path(ks.__file__).read_text(encoding="utf-8")
    for old, new in subs:
        assert src.count(old) == 1, f"mutation anchor not unique ({src.count(old)}): {old!r}"
        src = src.replace(old, new, 1)
    mutated = tmp_path / "kill_switch_mutant.py"
    mutated.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"ks_mutant_86_2_{tmp_path.name}", mutated)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _unused_load_mutated_kill_switch(tmp_path, old: str, new: str):
    """Import a COPY of kill_switch.py with one substitution applied.

    A copy, never the real module: mutating the installed module would leak into
    every other test in the session.
    """
    src = pathlib.Path(ks.__file__).read_text(encoding="utf-8")
    assert src.count(old) == 1, f"mutation anchor not unique ({src.count(old)}): {old!r}"
    mutated = tmp_path / "kill_switch_mutant.py"
    mutated.write_text(src.replace(old, new, 1), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("ks_mutant_86_2", mutated)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_c4_mutation_reverting_BOTH_guards_strands_the_replay_again(monkeypatch, tmp_path):
    """CRITERION 4 -- and a tension with its literal wording, disclosed.

    The criterion says "a mutation reverting the widened except makes the
    reproduction test fail again". MEASURED: it does NOT, and that is because
    the fix is better than the criterion assumed. Reverting ONLY the `except`
    lets `OverflowError` escape `_coerce_nav` -- and the NEW per-row `try` then
    catches it, skips that row, marks history incomplete and CONTINUES. The
    later rows still apply. So the single-guard mutant SURVIVES by design.

    That is defence in depth, not a hole, and it is asserted as such in
    `test_c4_reverting_only_the_except_is_now_HARMLESS` below. To satisfy what
    the criterion is actually for -- proving the widened tuple is load-bearing
    rather than decorative -- BOTH guards are reverted here, which restores the
    pre-fix stranding exactly.
    """
    mut = _load_mutated_kill_switch_multi(
        tmp_path,
        [("except (TypeError, ValueError, ArithmeticError):",
          "except (TypeError, ValueError):"),
         ("                try:\n                    self._apply_audit_row(row)\n                except Exception as e:",
          "                if True:\n                    self._apply_audit_row(row)\n                if False:")],
    )
    audit = tmp_path / "mutant_audit.jsonl"
    audit.write_text(
        '{"ts": "%s", "event": "peak_update", "nav": %s}\n' % (_ts(0, "00:00:30"), _POISON_INT)
        + _row(_ts(0, "00:01:00"), event="sod_snapshot", nav=100.0, date=_day(0)) + "\n"
        + _row(_ts(0, "00:02:00"), event="peak_update", nav=100.0) + "\n",
        encoding="utf-8",
    )
    mut._AUDIT_PATH = audit
    snap = mut.KillSwitchState().snapshot()
    assert snap["sod_nav"] is None and snap["peak_nav"] is None, (
        f"MUTANT SURVIVED: reverting BOTH guards still applied the later rows -- "
        f"the fix is not load-bearing. {snap}"
    )
    # ...and the shipped module, on the identical rows, does not strand.
    monkeypatch.setattr(ks, "_AUDIT_PATH", audit)
    assert ks.KillSwitchState().snapshot()["sod_nav"] == 100.0


def test_c4_reverting_only_the_except_is_now_HARMLESS(monkeypatch, tmp_path):
    """The defence-in-depth property, asserted rather than assumed.

    With only the `except` reverted, the per-row `try` still contains the fault:
    the replay continues, the well-formed rows apply, and history is marked
    INCOMPLETE. This is what makes the criterion-4 mutant survive, so it is
    pinned explicitly -- if a future edit removes the per-row try, THIS test
    goes red and names why.
    """
    mut = _load_mutated_kill_switch_multi(
        tmp_path,
        [("except (TypeError, ValueError, ArithmeticError):",
          "except (TypeError, ValueError):")],
    )
    audit = tmp_path / "single_mutant_audit.jsonl"
    audit.write_text(
        '{"ts": "%s", "event": "peak_update", "nav": %s}\n' % (_ts(0, "00:00:30"), _POISON_INT)
        + _row(_ts(0, "00:01:00"), event="sod_snapshot", nav=100.0, date=_day(0)) + "\n",
        encoding="utf-8",
    )
    mut._AUDIT_PATH = audit
    st = mut.KillSwitchState()
    # The DISCRIMINATING assertion -- this is what pins the property.
    assert st.snapshot()["sod_nav"] == 100.0, "the per-row try must still contain the fault"
    # NOTE (Q/A pass 1): the `_history_complete is False` assertion that used to
    # sit here was fixture-guaranteed by the same missing-archive-dir path, so it
    # killed nothing. Removed rather than left looking like coverage; the
    # bookkeeping is pinned by test_c3_MUTATION_the_bookkeeping_is_load_bearing.
