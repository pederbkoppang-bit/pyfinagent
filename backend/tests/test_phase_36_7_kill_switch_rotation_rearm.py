"""phase-36.7 (P0): the kill switch could not fire on a live book.

FILENAME IS LOAD-BEARING. The step's immutable verification command is
`pytest backend/tests/ -q -k kill_switch`, so a module name without the literal
`kill_switch` substring would not be selected and every criterion below would
pass vacuously.

THE DEFECT, MEASURED READ-ONLY 2026-07-26 against the running backend (pid 70791):

    $ curl -s http://localhost:8000/api/paper-trading/kill-switch
    {"paused": false, "sod_nav": null, "peak_nav": null, "current_nav": 23838.16,
     "breach": {"daily_loss_breached": false, "trailing_dd_breached": false,
                "any_breached": false}}

`evaluate_breach` gated both legs behind `if sod and sod > 0:` / `if peak and
peak > 0:`. `None` is falsy, so with both baselines null BOTH branches were
skipped, both percentages kept their 0.0 initialisers, and `any_breached` was
False FOR ANY current_nav -- a 50% drawdown did not trip it.

ROOT CAUSE, and why it was self-perpetuating rather than a one-off:
`scripts/housekeeping/verify_handoff_layout.py` classified the kill switch's ONLY
persistence file as layout dirt ("handoff/kill_switch_audit.jsonl is audit
output; move to handoff/audit/") and `backfill_handoff_archive.py` silently
executed the move -- no per-file print, so not even --dry-run named it. Git
records two prior executions: fa9aaf8e -> `-v3`, 77bc7db5 -> `-v4`. MEASURED
2026-07-26 (the production state, per HEAD and the masterplan step text): the
live file held 8 rows, {'pause': 4, 'resume': 4}, ZERO baseline rows; the
baselines were stranded in `handoff/audit/`.

CORRECTION (adversarial Q/A, same day): an earlier revision of this docstring
asserted "20 rows, {'pause': 10, 'resume': 10}" as the measurement. That was a
TEST-POLLUTION TRANSIENT -- 20 rows is what `handoff/kill_switch_audit.jsonl`
briefly held mid-session after running the FULL test suite (a different,
pre-existing test genuinely pauses/resumes the live book for real), not the
production state this step is about. It was restored via `git show HEAD:...`
before this file was committed. A permanent test file must record the actual
measurement, not a self-inflicted artifact from the session that wrote it --
this is exactly the failure `feedback_measure_dont_assert_claims` exists to
catch, caught here by an independent evaluator rather than by re-deriving it
before writing it down the first time.

VERBATIM PRE-FIX OUTPUT (recorded before any code change -- criterion 1):

    === B. evaluate_breach with sod_nav=None peak_nav=None, current_nav FAR below any baseline ===
    current_nav=11919.08 (a 50% drawdown from the live 23838.16 book)
    evaluate_breach -> {"any_breached": false, "daily_loss_breached": false,
      "daily_loss_limit_pct": 4.0, "daily_loss_pct": 0.0,
      "trailing_dd_breached": false, "trailing_dd_limit_pct": 10.0,
      "trailing_dd_pct": 0.0}
    any_breached      = False
    daily_loss_pct    = 0.0
    trailing_dd_pct   = 0.0
    disarmed marker present? False

    === C. Even a 99.99% drawdown does not breach ===
    current_nav=1.00 -> any_breached = False

WHAT THIS FILE LOCKS DOWN
  1. The defect itself: no-baseline + catastrophic NAV. Still `any_breached=False`
     (setting it True would flatten a healthy book on a housekeeping file move --
     see `evaluate_breach`'s docstring), but now `armed=False` + per-leg markers.
  2. Rotation-survival, reproducing the exact 2026-07-26 file shape.
  3. Glob-proofing against a synthesized `-v5` (a hardcoded suffix would ship
     green and be disarmed by the next sweep).
  4. `ts`-ordering, not filename ordering.
  5. The peak RATCHET across files (assignment measurably lowered the peak).
  6. Healthy-path arithmetic against fixed fixtures, just-under AND just-over
     both thresholds.
  7. No `peak_reset` row is ever written by the re-arm (KS-PEAK-RESET is a
     separate owed operator token, step 79.6).
  8. Every consumer key of the breach dict.
  9. The housekeeping allowlist, exercised by RUNNING both scripts against a tmp
     tree -- not by scanning their source.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# The measured live book on 2026-07-26.
LIVE_NAV = 23838.16
LIVE_SOD = 23838.19          # handoff/audit/kill_switch_audit-v4.jsonl
LIVE_PEAK_TRUE_MAX = 24666.57  # handoff/audit/kill_switch_audit.jsonl, 2026-06-03
LIVE_PEAK_LAST_BY_TS = 24124.77  # -v3.jsonl, 2026-06-22 (what assignment yields)
CATASTROPHIC_NAV = 11919.08  # a 50% drawdown from LIVE_NAV


# ── helpers ────────────────────────────────────────────────────────────────


# phase-36.9: COMPUTED, never hardcoded. `evaluate_breach` now treats a `sod_date`
# older than today (UTC) as an unevaluable daily leg, so a literal date string here
# would pass on the day it was written and fail every day after. Tests that want the
# daily leg to FIRE must anchor to today; tests that want staleness say so explicitly
# via STALE_SOD_DATE.
TODAY_UTC = datetime.now(timezone.utc).date().isoformat()
STALE_SOD_DATE = "2026-05-22"  # deliberately old: used where staleness is the subject


def _row(event: str, ts: str, **fields: Any) -> str:
    return json.dumps({"ts": ts, "event": event, **fields})


def _write(path: Path, *lines: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def ks_tmp_audit(tmp_path, monkeypatch):
    """Point kill_switch's audit I/O at a tmp tree.

    Yields (ks_module, live_path, archive_dir). Construct a fresh
    `ks.KillSwitchState()` to exercise the boot replay. The module-global
    `_state` is left untouched, so nothing here can leak into the real book or
    into another test's view of prod state.

    Patching `_AUDIT_PATH` ALONE is deliberate: the archive dir is derived from
    it, so an existing caller that only knew about `_AUDIT_PATH`
    (test_dod4_tier1_coverage_investment.py) still gets full isolation instead
    of a tmp live file merged with the real production archives.
    """
    import backend.services.kill_switch as ks

    live = tmp_path / "handoff" / "kill_switch_audit.jsonl"
    archive = tmp_path / "handoff" / "audit"
    live.parent.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ks, "_AUDIT_PATH", live)
    assert ks._audit_archive_dir() == archive
    yield ks, live, archive


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Swap the module-global `_state` for a throwaway instance.

    `evaluate_breach` / `check_auto_resume` read `_state` directly, so mutating
    the real one would corrupt the live book's in-memory baseline for the rest
    of the session. This installs a detached instance instead.

    `_AUDIT_PATH` IS REDIRECTED TOO, AND THAT IS NOT OPTIONAL. `_state.resume()`
    appends to `_AUDIT_PATH`, so the auto-resume and /resume tests below would
    otherwise write REAL rows -- including a fabricated
    `trigger=auto_resume_hysteresis` resume at a fake NAV -- into the live
    production audit trail `handoff/kill_switch_audit.jsonl`. That both corrupts
    the trail a future auditor reads AND breaks
    `test_phase_23_2_4_pause_resume_no_deadlock_live.py::
    test_phase_23_2_4_audit_log_clean_transitions`, whose allowed-trigger set is
    a regression lock that must not be widened to accommodate test junk. This
    happened once during development (12 rows written, then removed); the
    redirect + `test_phase_36_7_..._no_test_writes_reach_the_live_audit_file`
    below exist so it cannot recur silently.
    """
    import backend.services.kill_switch as ks

    monkeypatch.setattr(ks, "_AUDIT_PATH", tmp_path / "kill_switch_audit.jsonl")

    fresh = object.__new__(ks.KillSwitchState)
    import threading

    fresh._lock = threading.Lock()
    fresh._paused = False
    fresh._pause_reason = None
    fresh._sod_nav = None
    fresh._sod_date = None
    fresh._peak_nav = None
    fresh._paused_at = None
    fresh._auto_resume_alerted_at = None
    monkeypatch.setattr(ks, "_state", fresh)
    # The disarmed ERROR log is one-shot per process; reset so tests can assert
    # on it independently of execution order.
    monkeypatch.setattr(ks, "_disarmed_logged", False)
    yield ks, fresh


@pytest.fixture(autouse=True)
def _live_audit_file_is_write_protected(monkeypatch):
    """Nothing in THIS module may append to the real audit trail.

    Belt to the `isolated_state` / `ks_tmp_audit` braces: if a future test forgets
    to redirect `_AUDIT_PATH`, this fails it loudly instead of quietly writing
    fabricated pause/resume/peak rows into production state. `_append_audit`
    swallows write errors with a `logger.warning`, so a raise here cannot break
    the code under test -- it can only be caught by the assertion below.
    """
    live = REPO_ROOT / "handoff" / "kill_switch_audit.jsonl"
    before = live.read_bytes() if live.exists() else None
    yield
    after = live.read_bytes() if live.exists() else None
    assert after == before, (
        "phase-36.7: a test in this module wrote to the LIVE audit trail "
        f"{live}. Redirect ks._AUDIT_PATH to tmp_path."
    )


# ── criterion 1: the defect, and what replaced it ──────────────────────────


def test_phase_36_7_kill_switch_no_baseline_catastrophic_nav_is_loudly_disarmed(
    isolated_state,
):
    """THE DEFECT. sod_nav=None + peak_nav=None + a NAV 50% below any plausible
    baseline used to return a bare any_breached=False that was indistinguishable
    from health.

    `any_breached` stays False on purpose: paper_trader.py:1097 flattens the
    whole book on True, and a housekeeping file move must not do that. What
    changed is that absence is now REPORTED.
    """
    ks, st = isolated_state
    assert st.snapshot()["sod_nav"] is None
    assert st.snapshot()["peak_nav"] is None

    r = ks.evaluate_breach(
        current_nav=CATASTROPHIC_NAV,
        daily_loss_limit_pct=4.0,
        trailing_dd_limit_pct=10.0,
    )

    # Unchanged (deliberately): no phantom flatten on a missing baseline.
    assert r["any_breached"] is False
    # THE FIX: absence is now visible, per leg and rolled up.
    assert r["armed"] is False, (
        "phase-36.7 REGRESSION: a missing baseline reports as ARMED. The "
        "pre-fix dict carried no marker at all, so KillSwitchPanel.tsx computed "
        "alarm=false and rendered an emerald ACTIVE badge next to "
        "'daily 0.00% / 4%  trail 0.00% / 10%' -- absence rendered as the most "
        "reassuring possible reading."
    )
    assert r["daily_baseline_missing"] is True
    assert r["trailing_baseline_missing"] is True


def test_phase_36_7_kill_switch_disarmed_marker_is_per_leg_not_wholesale(
    isolated_state,
):
    """A present-sod / absent-peak state must STILL evaluate the daily leg.

    Modelling the disarmed state on the `nav_invalid` early return would skip
    BOTH legs whenever EITHER baseline was missing -- strictly less likely to
    pause than the buggy code it replaced. Measured on the real archives: the
    `-v4` file alone (peak_nav=None) fires correctly via the daily leg.
    """
    ks, st = isolated_state
    st._sod_nav = LIVE_SOD
    st._sod_date = TODAY_UTC
    st._peak_nav = None  # exactly what -v4-alone restores

    r = ks.evaluate_breach(CATASTROPHIC_NAV, 4.0, 10.0)

    assert r["daily_loss_breached"] is True, (
        "the daily leg still has a baseline and MUST still fire"
    )
    assert r["any_breached"] is True
    assert r["daily_loss_pct"] == pytest.approx(50.0001, abs=1e-4)
    # ... while the peak leg honestly reports itself unevaluable.
    assert r["daily_baseline_missing"] is False
    assert r["trailing_baseline_missing"] is True
    assert r["armed"] is False


def test_phase_36_7_kill_switch_disarmed_discriminates_on_presence_not_value(
    isolated_state,
):
    """A book sitting exactly AT its high-water mark reads 0.00% drawdown. That
    is a legitimate healthy reading and must stay ARMED (phase-80.36 rule:
    discriminate on presence, never on value)."""
    ks, st = isolated_state
    st._sod_nav = 10000.0
    st._sod_date = TODAY_UTC
    st._peak_nav = 10000.0

    r = ks.evaluate_breach(10000.0, 4.0, 10.0)

    assert r["daily_loss_pct"] == 0.0
    assert r["trailing_dd_pct"] == 0.0
    assert r["armed"] is True, "0.0%% drawdown is health, not absence"
    assert r["daily_baseline_missing"] is False
    assert r["trailing_baseline_missing"] is False
    assert r["any_breached"] is False


def test_phase_36_7_kill_switch_disarmed_logs_error_once_per_process(
    isolated_state, caplog
):
    """The operator-visible log leg. One-shot: `evaluate_breach` is called by a
    polled GET, so it must not spam."""
    ks, _st = isolated_state
    with caplog.at_level("ERROR", logger="backend.services.kill_switch"):
        for _ in range(5):
            ks.evaluate_breach(CATASTROPHIC_NAV, 4.0, 10.0)
    disarmed = [r for r in caplog.records if "DISARMED" in r.getMessage()]
    assert len(disarmed) == 1, f"expected exactly 1 ERROR, got {len(disarmed)}"


# ── criterion 3: rotation survival ─────────────────────────────────────────


def test_phase_36_7_kill_switch_baseline_restored_from_rotated_v4_file(ks_tmp_audit):
    """THE 2026-07-26 CONDITION, reproduced byte-for-byte in shape.

    Live `handoff/kill_switch_audit.jsonl` holds ONLY pause/resume rows; the
    baselines live in `handoff/audit/kill_switch_audit-v4.jsonl`.
    """
    ks, live, archive = ks_tmp_audit
    # Live file: pause/resume only -- the exact post-rotation content shape.
    _write(
        live,
        _row("pause", "2026-07-25T09:00:00+00:00", trigger="manual", details={}),
        _row("resume", "2026-07-25T09:05:00+00:00", trigger="manual", details={}),
    )
    # Rotated: the real -v4 row, verbatim from handoff/audit/.
    _write(
        archive / "kill_switch_audit-v4.jsonl",
        json.dumps({
            "ts": "2026-07-24T18:36:35.405342+00:00",
            "event": "sod_snapshot",
            "nav": LIVE_SOD,
            "date": "2026-07-24",
        }),
    )
    _write(
        archive / "kill_switch_audit-v3.jsonl",
        _row("peak_update", "2026-06-22T18:10:36.839708+00:00", nav=LIVE_PEAK_LAST_BY_TS),
    )

    st = ks.KillSwitchState()
    snap = st.snapshot()

    assert snap["sod_nav"] == pytest.approx(LIVE_SOD), (
        "phase-36.7 REGRESSION: the rotated sod_snapshot was not restored -- "
        "the loader is reading only the live file again"
    )
    assert snap["sod_date"] == "2026-07-24"
    assert snap["peak_nav"] == pytest.approx(LIVE_PEAK_LAST_BY_TS)
    # Pause/resume replay from the live file is unaffected.
    assert snap["paused"] is False

    # ... and with the baseline back, the switch FIRES.
    #
    # phase-36.9 AMENDED, and the amendment is the point of that step. This row is
    # dated 2026-07-24, so what it restores is a REAL but STALE daily anchor. Until
    # 36.9 this asserted `armed is True` -- which is exactly the live defect 36.9
    # was filed for: measured on :8000 on 2026-07-26, the operator's badge endpoint
    # served sod_date=2026-07-24 with armed:true, and the daily leg's 4% point sat
    # at 22884.66, i.e. a TWO-DAY move reported as a same-day loss. The daily leg is
    # now unevaluable on a stale anchor and says so.
    #
    # 36.7's OWN guarantee is preserved and still asserted here: restoration works,
    # and the switch still PROTECTS -- the trailing leg is a high-water mark, not
    # date-scoped, so it fires on the catastrophic NAV exactly as before. What
    # changed is only that a stale daily anchor no longer claims it can fire.
    import backend.services.kill_switch as ks_mod
    orig = ks_mod._state
    try:
        ks_mod._state = st
        r = ks_mod.evaluate_breach(CATASTROPHIC_NAV, 4.0, 10.0)
    finally:
        ks_mod._state = orig
    assert r["daily_baseline_stale"] is True, (
        "phase-36.9: a 2026-07-24 anchor evaluated today is stale and must say so"
    )
    assert r["armed"] is False, (
        "phase-36.9 REGRESSION: a stale daily anchor must not report armed:true -- "
        "that is the live 2026-07-26 defect this step closes"
    )
    assert r["daily_loss_breached"] is False, (
        "phase-36.9: a percentage must never be computed from a stale anchor"
    )
    assert r["daily_loss_pct"] == 0.0
    # PROTECTION IS NOT LOST -- the date-independent leg still fires.
    assert r["trailing_dd_breached"] is True
    assert r["any_breached"] is True

    # And with the SAME restore carrying a FRESH date, 36.7's original guarantee
    # holds unchanged: both legs evaluate and both fire.
    st._sod_date = TODAY_UTC
    try:
        ks_mod._state = st
        r2 = ks_mod.evaluate_breach(CATASTROPHIC_NAV, 4.0, 10.0)
    finally:
        ks_mod._state = orig
    assert r2["armed"] is True
    assert r2["any_breached"] is True
    assert r2["daily_loss_breached"] is True
    assert r2["trailing_dd_breached"] is True


def test_phase_36_7_kill_switch_restore_survives_a_future_v5_rotation(ks_tmp_audit):
    """GLOB, don't enumerate. `_safe_target` mints a fresh `-vN` on every sweep,
    so a fix that hardcodes `-v4` passes its own test, ships, and is disarmed
    again by the next rotation -- with a green suite."""
    ks, live, archive = ks_tmp_audit
    _write(live, _row("pause", "2026-08-01T09:00:00+00:00", trigger="manual", details={}))
    _write(
        archive / "kill_switch_audit-v5.jsonl",
        _row("sod_snapshot", "2026-08-01T08:00:00+00:00", nav=20000.0, date="2026-08-01"),
        _row("peak_update", "2026-08-01T08:00:01+00:00", nav=21000.0),
    )
    _write(
        archive / "kill_switch_audit-v17.jsonl",
        _row("peak_update", "2026-08-01T08:00:02+00:00", nav=21500.0),
    )

    snap = ks.KillSwitchState().snapshot()
    assert snap["sod_nav"] == pytest.approx(20000.0), (
        "phase-36.7 REGRESSION: a future -vN rotation is not read. The suffix "
        "must be globbed, never enumerated."
    )
    assert snap["peak_nav"] == pytest.approx(21500.0)


def test_phase_36_7_kill_switch_merge_orders_by_ts_not_by_filename(ks_tmp_audit):
    """FILE ORDER IS NOT TIME ORDER. `-vN` reflects discovery order in
    `_safe_target`. Measured on the real tree the two happen to agree today --
    that is luck. Here the LATER sod_snapshot sits in the EARLIER-sorting
    filename, so a filename-ordered merge would restore the stale one."""
    ks, live, archive = ks_tmp_audit
    _write(live, _row("resume", "2026-01-01T00:00:00+00:00", trigger="manual", details={}))
    # -v2 sorts BEFORE -v4 by name but holds the NEWER row.
    _write(
        archive / "kill_switch_audit-v2.jsonl",
        _row("sod_snapshot", "2026-07-24T18:00:00+00:00", nav=23838.19, date="2026-07-24"),
    )
    _write(
        archive / "kill_switch_audit-v4.jsonl",
        _row("sod_snapshot", "2026-06-09T18:00:00+00:00", nav=19000.0, date="2026-06-09"),
    )

    snap = ks.KillSwitchState().snapshot()
    assert snap["sod_nav"] == pytest.approx(23838.19), (
        f"expected the ts-latest sod (23838.19 @ 2026-07-24); got "
        f"{snap['sod_nav']} -- the merge is ordering by filename"
    )
    assert snap["sod_date"] == "2026-07-24"


# ── the peak ratchet (hard-stop 2) ─────────────────────────────────────────


def test_phase_36_7_kill_switch_peak_replay_ratchets_never_assigns(ks_tmp_audit):
    """`update_peak` (the writer) enforces "never moves down"; the replay used a
    bare assignment, so the ts-last row won even when an EARLIER row held a
    higher mark.

    MEASURED ON THE REAL ARCHIVES: the true high-water mark is 24666.57
    (2026-06-03, in kill_switch_audit.jsonl) but assignment yields 24124.77
    (2026-06-22, in -v3) -- 541.80 of peak = 2.17 percentage points of
    trailing-DD headroom destroyed BY THE RESTORE ITSELF. This test replays that
    exact pair.
    """
    ks, live, archive = ks_tmp_audit
    _write(live, _row("resume", "2026-07-25T00:00:00+00:00", trigger="manual", details={}))
    _write(
        archive / "kill_switch_audit.jsonl",
        _row("peak_update", "2026-06-03T19:04:39.221046+00:00", nav=LIVE_PEAK_TRUE_MAX),
    )
    _write(
        archive / "kill_switch_audit-v3.jsonl",
        _row("peak_update", "2026-06-22T18:10:36.839708+00:00", nav=LIVE_PEAK_LAST_BY_TS),
    )

    snap = ks.KillSwitchState().snapshot()
    assert snap["peak_nav"] == pytest.approx(LIVE_PEAK_TRUE_MAX), (
        f"phase-36.7 REGRESSION: peak replay assigned instead of ratcheting. "
        f"Got {snap['peak_nav']}, expected the true high-water mark "
        f"{LIVE_PEAK_TRUE_MAX}. Assignment silently LOWERS the peak, which "
        f"loosens the trailing-DD limit while the fix advertises a re-arm."
    )
    # Quantify the headroom the naive version gave away, so the number in the
    # comment above is asserted rather than asserted-about.
    loosened = (LIVE_PEAK_TRUE_MAX - LIVE_NAV) / LIVE_PEAK_TRUE_MAX * 100.0
    naive = (LIVE_PEAK_LAST_BY_TS - LIVE_NAV) / LIVE_PEAK_LAST_BY_TS * 100.0
    assert round(loosened - naive, 2) == pytest.approx(2.17, abs=0.01)


def test_phase_36_7_kill_switch_peak_reset_row_still_moves_the_peak_down(ks_tmp_audit):
    """The ratchet must not swallow phase-69.1's audited downward move. A
    `peak_reset` is AUTHORITATIVE in `ts` order, and later `peak_update` rows
    ratchet up from the reset value, not from the pre-reset high."""
    ks, live, archive = ks_tmp_audit
    _write(live, _row("resume", "2026-07-01T00:00:00+00:00", trigger="manual", details={}))
    _write(
        archive / "kill_switch_audit-v9.jsonl",
        _row("peak_update", "2026-06-01T00:00:00+00:00", nav=25000.0),
        _row("peak_reset", "2026-06-02T00:00:00+00:00", old_peak=25000.0,
             new_peak=20000.0, trigger="resume:manual"),
        _row("peak_update", "2026-06-03T00:00:00+00:00", nav=20500.0),
    )

    snap = ks.KillSwitchState().snapshot()
    assert snap["peak_nav"] == pytest.approx(20500.0), (
        f"expected the post-reset ratchet value 20500.0; got {snap['peak_nav']}. "
        f"25000.0 means max() swallowed the peak_reset; 20000.0 means the "
        f"post-reset peak_update was dropped."
    )


def test_phase_36_7_kill_switch_coerce_nav_rejects_zero_and_negative(ks_tmp_audit):
    """`float(x or 0.0) or None` mapped a legitimate 0.0 to None by accident and
    let a NEGATIVE nav through as a usable baseline. A negative peak silently
    skipped its leg with no marker; None now raises the loud disarmed state."""
    ks, live, archive = ks_tmp_audit
    assert ks._coerce_nav(None) is None
    assert ks._coerce_nav(0.0) is None
    assert ks._coerce_nav(-1.0) is None, (
        "a negative NAV must not become a usable baseline"
    )
    assert ks._coerce_nav("not-a-number") is None
    assert ks._coerce_nav(1.5) == pytest.approx(1.5)
    assert ks._coerce_nav("23838.19") == pytest.approx(23838.19)


def test_phase_36_7_kill_switch_coerce_nav_rejects_non_finite(ks_tmp_audit):
    """R1 (2026-07-26 adversarial verification): `inf > 0` is True in Python,
    so a non-finite NAV previously passed as a "valid" baseline. json.dumps
    round-trips a bare `Infinity` token, so a corrupted upstream NAV reaching
    the audit log would be replayed as a permanent baseline every future boot.
    Worse: the max()-ratchet in update_peak (this same step) makes an inf peak
    IRREVERSIBLE -- the prior assignment-replay could still heal on the next
    sane row, but max() cannot heal downward. With peak_nav=inf,
    trailing_dd_pct becomes nan and every `nan >= limit` comparison is False,
    so the trailing leg goes silently, permanently dead while `armed` still
    reports True."""
    import math
    ks, live, archive = ks_tmp_audit
    assert ks._coerce_nav(float("inf")) is None
    assert ks._coerce_nav(float("-inf")) is None
    assert ks._coerce_nav(float("nan")) is None
    # the exact round-trip that makes this reachable: a bad row gets written,
    # the audit is replayed, the resulting peak must not be usable.
    ks.KillSwitchState._append_audit("peak_update", nav=float("inf"))
    fresh = ks.KillSwitchState()
    snap = fresh.snapshot()
    assert snap["peak_nav"] != float("inf"), (
        "an Infinity audit row was replayed into a usable peak baseline"
    )
    assert snap["peak_nav"] is None or math.isfinite(snap["peak_nav"])

    _write(live, _row("resume", "2026-07-01T00:00:00+00:00", trigger="manual", details={}))
    _write(
        archive / "kill_switch_audit-v2.jsonl",
        _row("sod_snapshot", "2026-06-01T00:00:00+00:00", nav=-500.0, date="2026-06-01"),
        _row("peak_update", "2026-06-01T00:00:01+00:00", nav=-900.0),
    )
    snap = ks.KillSwitchState().snapshot()
    assert snap["sod_nav"] is None
    assert snap["peak_nav"] is None


def test_phase_36_7_kill_switch_unreadable_archive_does_not_lose_the_live_file(
    ks_tmp_audit,
):
    """One corrupt/half-written archive must not cost us the other sources."""
    ks, live, archive = ks_tmp_audit
    _write(
        live,
        _row("sod_snapshot", "2026-07-26T00:00:00+00:00", nav=23838.19, date="2026-07-26"),
    )
    (archive / "kill_switch_audit-v3.jsonl").write_bytes(b"\x00\x01not json at all\n")
    _write(
        archive / "kill_switch_audit-v4.jsonl",
        _row("peak_update", "2026-07-01T00:00:00+00:00", nav=24666.57),
    )

    snap = ks.KillSwitchState().snapshot()
    assert snap["sod_nav"] == pytest.approx(23838.19)
    assert snap["peak_nav"] == pytest.approx(24666.57)


def test_phase_36_7_kill_switch_missing_files_boot_clean(ks_tmp_audit):
    """No live file, no archive dir -> a clean (disarmed) boot, no exception."""
    ks, live, archive = ks_tmp_audit
    live.unlink(missing_ok=True)
    for p in archive.iterdir():
        p.unlink()
    archive.rmdir()
    snap = ks.KillSwitchState().snapshot()
    assert snap["sod_nav"] is None and snap["peak_nav"] is None
    assert ks._audit_source_paths() == []


# ── criterion 4: healthy-path arithmetic is byte-identical in behaviour ────

# (sod, peak, current_nav, daily_pct, trailing_pct, daily_breach, trailing_breach)
# Thresholds are the authoritative settings pair: 4.0% daily / 10.0% trailing
# (backend/config/settings.py:536-537). NOT changed by this step.
HEALTHY_PATH_FIXTURES = [
    # daily leg, just UNDER 4.0
    ("daily_just_under", 10000.0, None, 9601.0, 3.99, 0.0, False, False),
    # daily leg, exactly AT 4.0 -> breach (>= comparison)
    ("daily_at_limit", 10000.0, None, 9600.0, 4.0, 0.0, True, False),
    # daily leg, just OVER 4.0
    ("daily_just_over", 10000.0, None, 9599.0, 4.01, 0.0, True, False),
    # trailing leg, just UNDER 10.0
    ("trailing_just_under", None, 10000.0, 9001.0, 0.0, 9.99, False, False),
    # trailing leg, exactly AT 10.0 -> breach
    ("trailing_at_limit", None, 10000.0, 9000.0, 0.0, 10.0, False, True),
    # trailing leg, just OVER 10.0
    ("trailing_just_over", None, 10000.0, 8999.0, 0.0, 10.01, False, True),
    # profit: the 2026-05-05 nine-false-fire shape -- NEGATIVE daily_loss_pct
    ("profit_negative_pct", 10000.0, 10000.0, 10250.0, -2.5, -2.5, False, False),
    # both legs live, both healthy (the live book's own shape)
    ("live_book_shape", LIVE_SOD, LIVE_PEAK_TRUE_MAX, LIVE_NAV, 0.0, 3.3584,
     False, False),
]


@pytest.mark.parametrize(
    "name,sod,peak,nav,exp_daily,exp_trailing,exp_daily_breach,exp_trailing_breach",
    HEALTHY_PATH_FIXTURES,
    ids=[f[0] for f in HEALTHY_PATH_FIXTURES],
)
def test_phase_36_7_kill_switch_healthy_path_arithmetic_unchanged(
    isolated_state, name, sod, peak, nav, exp_daily, exp_trailing,
    exp_daily_breach, exp_trailing_breach,
):
    """Criterion 4: with a valid baseline the percentages and their limit
    comparisons are unchanged. Fixed numeric fixtures, both just-under and
    just-over each threshold, so a refactor of the gate condition cannot move
    the boundary without failing here."""
    ks, st = isolated_state
    st._sod_nav = sod
    st._sod_date = TODAY_UTC if sod is not None else None
    st._peak_nav = peak

    r = ks.evaluate_breach(nav, 4.0, 10.0)

    assert r["daily_loss_pct"] == pytest.approx(exp_daily, abs=1e-4), name
    assert r["trailing_dd_pct"] == pytest.approx(exp_trailing, abs=1e-4), name
    assert r["daily_loss_breached"] is exp_daily_breach, name
    assert r["trailing_dd_breached"] is exp_trailing_breach, name
    assert r["any_breached"] is (exp_daily_breach or exp_trailing_breach), name
    # Thresholds echoed back untouched.
    assert r["daily_loss_limit_pct"] == 4.0
    assert r["trailing_dd_limit_pct"] == 10.0


def test_phase_36_7_kill_switch_thresholds_untouched_by_this_step():
    """Criterion 7: no change to limit VALUES. The authoritative pair lives in
    settings; the MCP risk server keeps its own copy that agrees today."""
    from backend.config.settings import get_settings

    s = get_settings()
    assert s.paper_daily_loss_limit_pct == 4.0
    assert s.paper_trailing_dd_limit_pct == 10.0
    from backend.agents.mcp_servers import risk_server

    assert risk_server.DEFAULT_DAILY_LOSS_LIMIT_PCT == 4.0
    assert risk_server.DEFAULT_MAX_DD_CAP_PCT == 10.0


def test_phase_36_7_kill_switch_zero_sod_no_div_zero_and_flagged(isolated_state):
    """sod_nav=0.0 still must not raise, and now says so out loud instead of
    silently skipping the leg."""
    ks, st = isolated_state
    st._sod_nav = 0.0
    st._sod_date = TODAY_UTC
    st._peak_nav = 10000.0
    r = ks.evaluate_breach(10000.0, 4.0, 10.0)
    assert r["daily_loss_breached"] is False
    assert r["daily_baseline_missing"] is True
    assert r["trailing_baseline_missing"] is False
    assert r["armed"] is False


# ── criterion 6: NO peak_reset row is written ─────────────────────────────


def test_phase_36_7_kill_switch_rearm_writes_no_peak_reset_row(ks_tmp_audit):
    """KS-PEAK-RESET is a separate owed operator token (step 79.6). The re-arm
    must not write a `peak_reset` row -- AND must not achieve the same effect by
    omission: letting the next cycle's `update_peak` establish the peak from
    scratch would silently drop it from 24666.57 to the current 23838.16,
    giving away 3.36 percentage points of trailing-DD headroom un-audited.
    """
    ks, live, archive = ks_tmp_audit
    _write(live, _row("resume", "2026-07-25T00:00:00+00:00", trigger="manual", details={}))
    _write(
        archive / "kill_switch_audit-v3.jsonl",
        _row("peak_update", "2026-06-03T19:04:39.221046+00:00", nav=LIVE_PEAK_TRUE_MAX),
        _row("sod_snapshot", "2026-07-24T18:36:35.405342+00:00", nav=LIVE_SOD,
             date="2026-07-24"),
    )

    before = live.read_text(encoding="utf-8")

    st = ks.KillSwitchState()
    # Exercise the full restore + evaluate + a monotonic peak nudge at the
    # CURRENT (lower) NAV -- the shape the next live cycle produces.
    import backend.services.kill_switch as ks_mod
    orig = ks_mod._state
    try:
        ks_mod._state = st
        st.update_peak(LIVE_NAV)  # lower than the peak -> must be a no-op
        ks_mod.evaluate_breach(LIVE_NAV, 4.0, 10.0)
        snap = st.snapshot()
    finally:
        ks_mod._state = orig

    # The restore did NOT re-anchor the peak to the current NAV.
    assert snap["peak_nav"] == pytest.approx(LIVE_PEAK_TRUE_MAX), (
        "the peak must survive the re-arm at its true high-water mark; "
        "re-anchoring to current NAV is an un-audited KS-PEAK-RESET"
    )

    # And no peak_reset row was appended anywhere.
    for path in [live] + sorted(archive.glob("kill_switch_audit*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            assert row.get("event") != "peak_reset", (
                f"phase-36.7 wrote a peak_reset row to {path.name}: {row}. "
                f"KS-PEAK-RESET is a separate owed operator token (step 79.6)."
            )
    # The live file is byte-unchanged: a boot replay is READ-ONLY.
    assert live.read_text(encoding="utf-8") == before


def test_phase_36_7_kill_switch_reset_peak_still_dark_by_default(ks_tmp_audit):
    """phase-69.1's `reset_peak` stays DARK behind
    settings.kill_switch_peak_reset_enabled. This step does not flip it.

    ISOLATION IS NOT OPTIONAL HERE. A prior revision of this test built `fresh`
    via `KillSwitchState.__new__` with no `ks_tmp_audit` / `_AUDIT_PATH`
    redirect at all. reset_peak() returning None today (flag off) made that
    look safe -- but the moment `kill_switch_peak_reset_enabled` flips True
    (the owed KS-PEAK-RESET token, step 79.6) and this exact test file is
    re-run, `reset_peak` stops being a no-op and calls `_append_audit`, which
    would write a REAL `peak_reset new_peak=1.0` row into the LIVE production
    trail -- and because the replay treats peak_reset as authoritative
    (assignment, not max), a subsequent boot would restore `peak_nav=1.0` and
    the trailing leg would go permanently dead while reporting `armed=True`.
    Strictly worse than the bug 36.7 fixes. The `_live_audit_file_is_write_protected`
    autouse guard above only catches this AFTER the write, at teardown -- it
    does not undo the row. `ks_tmp_audit` redirects `_AUDIT_PATH` before this
    test can touch it at all, matching every other test in this module.
    """
    ks, live, _archive = ks_tmp_audit
    from backend.config.settings import get_settings

    assert getattr(get_settings(), "kill_switch_peak_reset_enabled", False) is False
    fresh = ks.KillSwitchState.__new__(ks.KillSwitchState)
    import threading

    fresh._lock = threading.Lock()
    fresh._peak_nav = 25000.0
    assert fresh.reset_peak(1.0, trigger="test") is None, (
        "reset_peak must remain a no-op until KS-PEAK-RESET: APPROVED"
    )
    assert fresh._peak_nav == 25000.0
    # Even though reset_peak was a no-op, prove the isolation actually would
    # have caught a write: the tmp live file must be untouched too.
    assert not live.exists() or live.stat().st_size == 0


def test_phase_36_7_reset_peak_write_is_isolated_when_flag_is_on(isolated_state, monkeypatch):
    """Proves R6's fix under the ACTUAL dangerous condition, not just the
    off-flag no-op above.

    Flips `kill_switch_peak_reset_enabled` on (simulating a future
    KS-PEAK-RESET: APPROVED) and calls reset_peak for real. The write MUST
    land in the tmp path `isolated_state` redirects `_AUDIT_PATH` to -- never
    in the real repo's handoff/kill_switch_audit.jsonl. This is the test that
    would have caught the original defect: pre-fix, the equivalent call
    sequence (a bare `__new__` object with no redirect) hit the live file.
    """
    ks, fresh = isolated_state
    from backend.config import settings as settings_module

    monkeypatch.setattr(
        settings_module.get_settings(), "kill_switch_peak_reset_enabled", True
    )
    fresh._peak_nav = 25000.0

    real_live = Path(__file__).resolve().parents[2] / "handoff" / "kill_switch_audit.jsonl"
    real_before = real_live.read_bytes() if real_live.exists() else None

    result = fresh.reset_peak(1.0, trigger="test-flag-on")

    assert result is not None, "with the flag on, reset_peak must actually run"
    assert ks._AUDIT_PATH.exists(), "the peak_reset row must land in the ISOLATED tmp file"
    assert "peak_reset" in ks._AUDIT_PATH.read_text(encoding="utf-8")
    real_after = real_live.read_bytes() if real_live.exists() else None
    assert real_after == real_before, (
        "reset_peak wrote to the REAL live audit trail even with isolation active"
    )


# ── criterion 2: the refusals -- resume paths must not trust a disarmed read ──


def test_phase_36_7_kill_switch_resume_endpoint_409s_when_disarmed(
    isolated_state, monkeypatch
):
    """POST /api/paper-trading/resume gated SOLELY on `breach["any_breached"]`,
    so while disarmed EVERY resume succeeded regardless of the real drawdown --
    defeating the module docstring's "explicit human resume required once both
    limits read healthy". You cannot verify healthy without a baseline."""
    from fastapi import HTTPException

    import backend.api.paper_trading as pt

    ks, st = isolated_state
    st._sod_nav = None
    st._peak_nav = None
    st._paused = True

    class _BQ:
        def get_paper_portfolio(self, _pid):
            return {"total_nav": LIVE_NAV}

    monkeypatch.setattr(pt, "get_bq_client", lambda: _BQ())
    monkeypatch.setattr(pt, "_get_ks_state", lambda: st)

    req = pt.KillSwitchActionRequest(confirmation="RESUME")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(pt.resume_trading(req))
    assert exc.value.status_code == 409
    assert "DISARMED" in str(exc.value.detail)
    assert st.is_paused() is True, "a refused resume must leave the pause intact"

    # Control: with the baseline restored, the SAME call succeeds -- the gate is
    # about the disarmed state, not a blanket refusal.
    st._sod_nav = LIVE_SOD
    st._sod_date = TODAY_UTC
    st._peak_nav = LIVE_PEAK_TRUE_MAX
    out = asyncio.run(pt.resume_trading(req))
    assert out["status"] == "resumed"
    assert st.is_paused() is False


def test_phase_36_7_kill_switch_auto_resume_refuses_while_disarmed(
    isolated_state, monkeypatch
):
    """`check_auto_resume` treated `not any_breached` as proof of health and
    resumed at T+2h. With null baselines that proof is vacuous, so a disarmed
    switch would auto-resume out of a real drawdown -- breaking
    docs/runbooks/away-ops-rules.md rail 5."""
    from datetime import datetime, timedelta, timezone

    ks, st = isolated_state
    st._paused = True
    st._paused_at = (
        datetime.now(timezone.utc) - timedelta(seconds=ks.AUTO_RESUME_TRIGGER_AT_SEC + 60)
    ).isoformat()
    st._sod_nav = None
    st._peak_nav = None

    out = ks.check_auto_resume(CATASTROPHIC_NAV, 4.0, 10.0, enabled=True)
    assert out["action"] == "no_op", (
        f"a disarmed switch must never auto-resume; got {out}"
    )
    assert out["reason"] == "kill_switch_disarmed_baseline_missing"
    assert st.is_paused() is True

    # Control: armed + genuinely healthy -> the T+2h resume still fires, so this
    # guard did not disable phase-38.1's hysteresis.
    st._sod_nav = 10000.0
    st._sod_date = TODAY_UTC
    st._peak_nav = 10000.0
    out2 = ks.check_auto_resume(10000.0, 4.0, 10.0, enabled=True)
    assert out2["action"] == "resume", out2
    assert st.is_paused() is False


# ── criterion 8-adjacent: the breach-dict consumer contract ────────────────


def test_phase_36_7_kill_switch_breach_dict_shape_contract(isolated_state):
    """Every enumerated consumer of the breach dict, and what it requires.

    HARD-INDEXED (`breach["any_breached"]`) -- that key can never be removed:
      * backend/api/paper_trading.py::resume_trading
      * backend/services/paper_trader.py::check_and_enforce_kill_switch
      * backend/services/kill_switch.py::check_auto_resume
    .get()-ACCESSED / pass-through:
      * backend/api/paper_trading.py::get_kill_switch_state  (returns verbatim)
      * backend/agents/mcp_servers/risk_server.py::kill_switch (MCP tool)
      * backend/slack_bot/scheduler.py + scripts/ops/send_confirmation_digest.py
        (`br.get("any_breached")`)
      * backend/services/autonomous_loop.py (reads ks_check["triggered"], not
        the breach dict)
    FRONTEND (typed interfaces, so added fields are type-safe):
      * KillSwitchPanel.tsx, OpsStatusBar.tsx
    """
    ks, st = isolated_state
    st._sod_nav = 10000.0
    st._sod_date = TODAY_UTC
    st._peak_nav = 10000.0

    normal = ks.evaluate_breach(9900.0, 4.0, 10.0)
    invalid = ks.evaluate_breach(0.0, 4.0, 10.0)

    required = {
        "daily_loss_breached", "daily_loss_pct", "daily_loss_limit_pct",
        "trailing_dd_breached", "trailing_dd_pct", "trailing_dd_limit_pct",
        "any_breached",
        # phase-36.7 additions -- in BOTH shapes, so no consumer must branch.
        "daily_baseline_missing", "trailing_baseline_missing", "armed",
    }
    assert required <= set(normal), sorted(required - set(normal))
    assert required <= set(invalid), sorted(required - set(invalid))
    # `nav_invalid` keeps its established optional-key discipline: present only
    # in the early return, so every consumer must use .get() for it.
    assert "nav_invalid" not in normal
    assert invalid["nav_invalid"] is True
    # The invalid-NAV branch stays fail-safe (no phantom 100% breach).
    assert invalid["any_breached"] is False
    assert invalid["daily_loss_pct"] == 0.0


def test_phase_36_7_kill_switch_mcp_risk_server_surfaces_armed(monkeypatch):
    """The MCP `kill_switch` tool hands `breach` to MAS agents verbatim, so the
    disarmed marker reaches them with no server-side change."""
    import backend.services.kill_switch as ks

    r = ks.evaluate_breach(LIVE_NAV, 4.0, 10.0)
    assert "armed" in r
    src = (REPO_ROOT / "backend/agents/mcp_servers/risk_server.py").read_text(
        encoding="utf-8"
    )
    assert '"breach": breach' in src, (
        "risk_server no longer passes the breach dict through verbatim; the "
        "disarmed marker may no longer reach MAS agents"
    )


# ── the root cause: the housekeeping sweep must not move the state file ────


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / "housekeeping" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_ks367_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _tmp_handoff_tree(tmp_path: Path) -> Path:
    handoff = tmp_path / "handoff"
    (handoff / "current").mkdir(parents=True)
    (handoff / "audit").mkdir()
    (handoff / "logs").mkdir()
    (handoff / "archive" / "misc").mkdir(parents=True)
    (handoff / "kill_switch_audit.jsonl").write_text(
        json.dumps({"ts": "2026-07-24T18:36:35+00:00", "event": "sod_snapshot",
                    "nav": LIVE_SOD, "date": "2026-07-24"}) + "\n",
        encoding="utf-8",
    )
    (handoff / "some_other_audit.jsonl").write_text("{}\n", encoding="utf-8")
    (handoff / "stray.log").write_text("x\n", encoding="utf-8")
    return handoff


def test_phase_36_7_kill_switch_state_file_not_swept_by_backfill(tmp_path, capsys):
    """ROOT CAUSE. `backfill_handoff_archive.py` shutil.move'd the live state
    file into handoff/audit/ with a `-vN` suffix. Runs the real script against a
    tmp tree and asserts the file stays put -- while an actually-log-shaped
    sibling still moves, so the exclusion is narrow."""
    mod = _load_script("backfill_handoff_archive")
    handoff = _tmp_handoff_tree(tmp_path)
    mp = tmp_path / "masterplan.json"
    mp.write_text(json.dumps({"phases": []}), encoding="utf-8")

    mod.REPO = tmp_path
    mod.HANDOFF = handoff
    mod.CURRENT = handoff / "current"
    mod.ARCHIVE = handoff / "archive"
    mod.AUDIT = handoff / "audit"
    mod.LOGS = handoff / "logs"
    mod.MISC = handoff / "archive" / "misc"
    mod.MASTERPLAN = mp

    assert mod.main(dry_run=False) == 0
    out = capsys.readouterr().out

    assert (handoff / "kill_switch_audit.jsonl").exists(), (
        "phase-36.7 REGRESSION: the housekeeping sweep moved the kill switch's "
        "live state file again. That is the exact action that disarmed the "
        "switch on 2026-07-26 (git fa9aaf8e -> -v3, 77bc7db5 -> -v4)."
    )
    assert not list((handoff / "audit").glob("kill_switch_audit*.jsonl"))
    # Narrowness: a genuinely log-shaped sibling STILL moves.
    assert not (handoff / "some_other_audit.jsonl").exists()
    assert (handoff / "audit" / "some_other_audit.jsonl").exists()
    assert (handoff / "logs" / "stray.log").exists()
    # Visibility: the root loop used to move files with NO print, so not even
    # --dry-run named the state file. Every root decision is now printed.
    assert "KEEP (live state file, phase-36.7): kill_switch_audit.jsonl" in out
    assert "some_other_audit.jsonl -> handoff/audit/some_other_audit.jsonl" in out
    assert "root-kept=1" in out


def test_phase_36_7_kill_switch_state_file_not_flagged_by_layout_verifier(tmp_path):
    """The verifier ACTIVELY DEMANDED the move ("handoff/kill_switch_audit.jsonl
    is audit output; move to handoff/audit/"), which is what made the defect
    self-perpetuating: fix the loader only, and the next housekeeping run
    re-breaks it."""
    mod = _load_script("verify_handoff_layout")
    handoff = _tmp_handoff_tree(tmp_path)
    mp = tmp_path / "masterplan.json"
    mp.write_text(json.dumps({"phases": []}), encoding="utf-8")

    mod.REPO = tmp_path
    mod.HANDOFF = handoff
    mod.CURRENT = handoff / "current"
    mod.MASTERPLAN = mp

    rc = mod.main()
    # The tree still has a genuinely misplaced audit file + log, so a FAIL is
    # expected -- what matters is WHICH violations are reported.
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.main()
    printed = buf.getvalue()
    assert "kill_switch_audit.jsonl" not in printed, (
        f"phase-36.7 REGRESSION: the verifier still tells operators to move the "
        f"live kill-switch state file:\n{printed}"
    )
    # Narrowness again: the real violations are still reported.
    assert "some_other_audit.jsonl is audit output" in printed
    assert "stray.log is a log" in printed
    assert rc == 1


def test_phase_36_7_kill_switch_allowlists_agree_between_the_two_scripts():
    """The verifier and the backfill must agree. If only one allowlists the
    file, the verifier keeps demanding a move the backfill refuses (or worse,
    the reverse), and the loop reopens."""
    v = _load_script("verify_handoff_layout")
    b = _load_script("backfill_handoff_archive")
    assert v.HANDOFF_ROOT_KEEP == b.HANDOFF_ROOT_KEEP
    assert "kill_switch_audit.jsonl" in v.HANDOFF_ROOT_KEEP
    # The file the allowlist protects must be the file the loader actually reads.
    import backend.services.kill_switch as ks

    assert ks._AUDIT_PATH.name in v.HANDOFF_ROOT_KEEP, (
        "the allowlisted name drifted from kill_switch._AUDIT_PATH"
    )
    # The loader must read the same directory the sweep moves files INTO.
    assert ks._audit_archive_dir() == ks._AUDIT_PATH.parent / "audit", (
        "the loader's archive dir drifted from where the sweep moves files"
    )
    assert ks._audit_archive_dir() == b.AUDIT, (
        f"loader reads {ks._audit_archive_dir()} but the sweep writes to {b.AUDIT}"
    )
