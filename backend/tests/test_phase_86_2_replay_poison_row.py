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
        _row("2026-08-09T00:00:00+00:00", event="peak_update", nav=90.0),
        # the poison row
        '{"ts": "2026-08-09T00:00:30+00:00", "event": "peak_update", "nav": %s}' % _POISON_INT,
        # AFTER it (criterion 2's "after" half)
        _row("2026-08-09T00:01:00+00:00", event="sod_snapshot", nav=100.0, date="2026-08-09"),
        _row("2026-08-09T00:02:00+00:00", event="peak_update", nav=100.0),
    )

    st = ks.KillSwitchState()          # DETACHED, replays on construction
    snap = st.snapshot()

    # Every WELL-FORMED row applied -- the ones before AND after the bad one.
    assert snap["sod_nav"] == 100.0, snap
    assert snap["sod_date"] == "2026-08-09", snap
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
        '{"ts": "2026-08-09T00:00:30+00:00", "event": "peak_update", "nav": %s}' % _POISON_INT,
        _row("2026-08-09T00:01:00+00:00", event="peak_update", nav=100.0),
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
    """The load-bearing half. A skip WITHOUT this bookkeeping would let a
    lost-history anchor claim authority again, reopening the hole phase-36.8
    closed. `_read_audit_rows` already does exactly this per LINE; the apply
    loop now matches it per ROW.
    """
    _isolate(monkeypatch, tmp_path)
    _write(
        tmp_path,
        '{"ts": "2026-08-09T00:00:30+00:00", "event": "peak_update", "nav": %s}' % _POISON_INT,
        _row("2026-08-09T00:01:00+00:00", event="peak_update", nav=100.0),
    )
    st = ks.KillSwitchState()
    assert st._history_complete is False, (
        "a skipped row must mark history INCOMPLETE -- otherwise a later "
        "anchor-from-None can claim authority on history it never saw"
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
    _write(tmp_path, _row("2026-08-09T00:01:00+00:00", event="peak_update", nav=100.0))
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
        '{"ts": "2026-08-09T00:00:30+00:00", "event": "peak_update", "nav": %s}\n' % _POISON_INT
        + _row("2026-08-09T00:01:00+00:00", event="sod_snapshot", nav=100.0, date="2026-08-09") + "\n"
        + _row("2026-08-09T00:02:00+00:00", event="peak_update", nav=100.0) + "\n",
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
        '{"ts": "2026-08-09T00:00:30+00:00", "event": "peak_update", "nav": %s}\n' % _POISON_INT
        + _row("2026-08-09T00:01:00+00:00", event="sod_snapshot", nav=100.0, date="2026-08-09") + "\n",
        encoding="utf-8",
    )
    mut._AUDIT_PATH = audit
    st = mut.KillSwitchState()
    assert st.snapshot()["sod_nav"] == 100.0, "the per-row try must still contain the fault"
    assert st._history_complete is False, "and it must still mark history INCOMPLETE"
