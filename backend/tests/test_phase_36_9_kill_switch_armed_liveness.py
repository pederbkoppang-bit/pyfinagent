"""phase-36.9 -- `armed:true` must mean "this leg can actually fire NOW".

Three ways the kill switch reported `armed:true` while a leg was structurally
unable to fire. All three were found by adversarial verification of step 36.7 and
all three are LIVE-REPRODUCIBLE, not hypothetical:

  F1 STALE ANCHOR. `evaluate_breach` never compared the restored `sod_date` to
     today, though the snapshot carries it. Measured on the operator's running
     :8000 on 2026-07-26: GET /api/paper-trading/kill-switch served
     `sod_date: "2026-07-24"` with `armed: true`, and the daily leg's 4.0% point
     sat at 23838.19 * 0.96 = 22884.66 -- a TWO-DAY price move would have been
     reported as a same-day 4% loss. Research (IEC 61508 diagnostic coverage, SEC
     market-wide circuit breakers, which are "calculated daily based on the prior
     day's closing price") says this is the WORST quadrant, not merely a weaker
     one: it loses same-day coverage AND biases toward a spurious flatten -- a
     nuisance trip and a diagnostic failure at the same time.

  F2 nav_invalid RETURNED armed:true. `armed` was computed BEFORE the nav_invalid
     early return, so an unmeasurable NAV (a BQ timeout's `or 0.0` fallback)
     produced `any_breached:False` AND `armed:true` together. The cockpit then
     renders an emerald ACTIVE badge beside two 0.00% readouts -- the exact
     failure mode KillSwitchPanel's own comment claims to eliminate -- and POST
     /resume could succeed against a NAV nobody had measured. Unknown != healthy.

  F3 sod_nav=0.0 LATCHED AND WEDGED ITS OWN REPAIR. `update_sod_nav` wrote
     `float(nav)` with no positivity guard, so 0.0 became a REAL
     sod_snapshot row with today's date. `evaluate_breach` then correctly reported
     the leg missing and /resume correctly 409'd -- promising "the next cycle
     re-anchors both baselines" -- but the re-anchor predicate tested
     `sod_nav is None`, and 0.0 is not None while the date WAS today. The book sat
     paused up to 24h behind an operator-facing message that was FALSE. In IEC
     61511 terms (Cl. 16.2.4) that is a bypass with no exit, which is why the fix
     is at the root (refuse the anchor) and not only at the consumer.

WHAT THIS FILE DOES NOT DO: it does not assert a threshold. Limits, stops, sector
caps, DSR and PBO are byte-untouched by this step; only the question "can this leg
fire right now?" changed.

Filename carries `kill_switch` deliberately -- the immutable verification command
selects with `-k kill_switch`, and a module named without it is silently deselected
(phase-36.8 shipped a cycle with zero of its tests selected for exactly that reason).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

TODAY_UTC = datetime.now(timezone.utc).date().isoformat()
YESTERDAY_UTC = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
TWO_DAYS_AGO_UTC = (datetime.now(timezone.utc).date() - timedelta(days=2)).isoformat()

# The real live shape, measured 2026-07-26 from :8000 (GET only).
LIVE_SOD = 23838.19
LIVE_PEAK = 24666.57
LIVE_NAV = 23838.16
# 23838.19 * 0.96 -- the exact NAV at which the daily leg reads 4.0%.
NAV_AT_4PCT = 22884.66


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Detached state + redirected audit path.

    `_AUDIT_PATH` is redirected BEFORE anything can append: `update_sod_nav`
    writes a `sod_snapshot` row, and `handoff/kill_switch_audit.jsonl` is
    git-tracked LIVE safety state. An evaluator wrote 54 rows into it during this
    phase by skipping this step.
    """
    import backend.services.kill_switch as ks

    monkeypatch.setattr(ks, "_AUDIT_PATH", tmp_path / "kill_switch_audit.jsonl")
    monkeypatch.setattr(ks, "_audit_archive_dir", lambda: tmp_path / "audit")
    # A REAL constructor against the redirected paths, not `object.__new__` with
    # hand-set fields. The hand-built form silently omits whatever attribute the
    # snapshot contract grows next (it already missed `_paused_at` and
    # `_auto_resume_alerted_at`), which fails as an AttributeError inside the code
    # under test and reads like a production bug. The tmp audit tree is empty, so
    # the replay yields a clean all-None state.
    st = ks.KillSwitchState()
    monkeypatch.setattr(ks, "_state", st)
    monkeypatch.setattr(ks, "_disarmed_logged", False)
    return ks, st


# --------------------------------------------------------------------------
# F1 -- a stale daily anchor cannot fire, and must not claim it can
# --------------------------------------------------------------------------

def test_phase_36_9_the_live_2026_07_26_shape_no_longer_reports_armed(isolated_state):
    """THE DEFECT, in the exact shape the running backend served.

    Pre-fix this returned armed:True with daily_loss_pct=4.0 at NAV_AT_4PCT --
    a two-day move read as a same-day 4% loss, one cycle away from flatten_all.
    """
    ks, st = isolated_state
    st._sod_nav = LIVE_SOD
    st._sod_date = TWO_DAYS_AGO_UTC
    st._peak_nav = LIVE_PEAK

    r = ks.evaluate_breach(NAV_AT_4PCT, 4.0, 10.0)

    assert r["daily_baseline_stale"] is True
    assert r["armed"] is False, (
        "a 2-day-old anchor must not report armed:true -- this is the live defect"
    )
    assert r["daily_loss_breached"] is False
    assert r["daily_loss_pct"] == 0.0, (
        "a percentage must NEVER be computed from a stale anchor; pre-fix this was 4.0"
    )


def test_phase_36_9_stale_anchor_does_not_disable_the_trailing_leg(isolated_state):
    """PER LEG, never wholesale. The trailing mark is a high-water mark, not
    date-scoped, so staleness of the DAILY anchor must not cost us the
    date-independent protection. Disarming both would be strictly less likely to
    pause -- the trap phase-36.7 documented and deliberately avoided."""
    ks, st = isolated_state
    st._sod_nav = LIVE_SOD
    st._sod_date = TWO_DAYS_AGO_UTC
    st._peak_nav = LIVE_PEAK

    r = ks.evaluate_breach(LIVE_PEAK * 0.50, 4.0, 10.0)  # catastrophic

    assert r["trailing_dd_breached"] is True
    assert r["any_breached"] is True, "protection must survive a stale daily anchor"


def test_phase_36_9_yesterday_is_already_stale(isolated_state):
    """The boundary is TODAY, not "recent". A daily-loss limit is measured from
    today's open; yesterday's open is a different measurement."""
    ks, st = isolated_state
    st._sod_nav = LIVE_SOD
    st._sod_date = YESTERDAY_UTC
    st._peak_nav = LIVE_PEAK

    assert ks.evaluate_breach(LIVE_NAV, 4.0, 10.0)["daily_baseline_stale"] is True


@pytest.mark.parametrize("bad_date", [None, "", "not-a-date", "2026-13-45", 20260726, ["2026-07-26"]])
def test_phase_36_9_an_unprovable_date_counts_as_stale(isolated_state, bad_date):
    """Freshness is a CLAIM THAT MUST BE PROVABLE. A present baseline whose date
    is missing, malformed or the wrong type cannot prove it is today's open, so it
    is treated as stale rather than trusted. Erring the other way would let the
    defect back in through one malformed audit row."""
    ks, st = isolated_state
    st._sod_nav = LIVE_SOD
    st._sod_date = bad_date
    st._peak_nav = LIVE_PEAK

    r = ks.evaluate_breach(LIVE_NAV, 4.0, 10.0)
    assert r["daily_baseline_stale"] is True
    assert r["armed"] is False


def test_phase_36_9_stale_is_not_a_second_name_for_absent(isolated_state):
    """When there is no daily baseline at all, `daily_baseline_missing` already
    says so. Reporting `daily_baseline_stale` too would be two names for one
    condition and would muddy which repair an operator needs."""
    ks, st = isolated_state
    st._sod_nav = None
    st._sod_date = None
    st._peak_nav = LIVE_PEAK

    r = ks.evaluate_breach(LIVE_NAV, 4.0, 10.0)
    assert r["daily_baseline_missing"] is True
    assert r["daily_baseline_stale"] is False


# --------------------------------------------------------------------------
# F2 -- an unmeasurable NAV cannot fire either
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad_nav", [0.0, None, -1.0])
def test_phase_36_9_nav_invalid_must_not_report_armed(isolated_state, bad_nav):
    """Pre-fix: any_breached:False AND armed:true simultaneously, because `armed`
    was computed from baseline presence before the nav_invalid early return."""
    ks, st = isolated_state
    st._sod_nav = LIVE_SOD
    st._sod_date = TODAY_UTC          # baselines are PERFECT here
    st._peak_nav = LIVE_PEAK

    r = ks.evaluate_breach(bad_nav, 4.0, 10.0)

    assert r["nav_invalid"] is True
    assert r["any_breached"] is False
    assert r["armed"] is False, (
        "a leg that cannot measure cannot fire -- unknown is not healthy"
    )
    assert r["daily_baseline_missing"] is False, (
        "the BASELINES are fine; the disarm must be attributed to the NAV, not to them"
    )
    assert r["nav_invalid_disarmed"] is True


def test_phase_36_9_nav_invalid_armed_is_consistent_with_any_breached(isolated_state):
    """The immutable criterion, stated directly: armed must never be True while
    any_breached is False *because nothing could be measured*."""
    ks, st = isolated_state
    st._sod_nav = LIVE_SOD
    st._sod_date = TODAY_UTC
    st._peak_nav = LIVE_PEAK

    r = ks.evaluate_breach(0.0, 4.0, 10.0)
    assert not (r["armed"] and not r["any_breached"] and r.get("nav_invalid"))


# --------------------------------------------------------------------------
# F3 -- 0.0 must not latch, and must not wedge its own repair
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_phase_36_9_update_sod_nav_refuses_a_non_positive_anchor(isolated_state, bad):
    ks, st = isolated_state
    st.update_sod_nav(bad, date=TODAY_UTC)

    snap = st.snapshot()
    assert snap["sod_nav"] is None, f"{bad!r} must not latch as a baseline"
    assert snap["sod_date"] is None, "a refused anchor must not stamp a date either"


def test_phase_36_9_a_refused_anchor_writes_no_audit_row(isolated_state, tmp_path):
    """A refusal must not leave a `sod_snapshot` row, or the next boot replays the
    very 0.0 we refused."""
    ks, st = isolated_state
    st.update_sod_nav(0.0, date=TODAY_UTC)

    path = tmp_path / "kill_switch_audit.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    assert [r for r in rows if r.get("event") == "sod_snapshot"] == []


def test_phase_36_9_refusing_leaves_the_state_the_re_anchor_check_repairs(isolated_state):
    """THE WEDGE, closed at the root. After a refusal `sod_nav` is None, which is
    exactly what paper_trader's daily-roll predicate re-anchors on -- so the /resume
    409's promise ("the next cycle re-anchors both baselines") becomes TRUE."""
    ks, st = isolated_state
    st.update_sod_nav(0.0, date=TODAY_UTC)
    snap = st.snapshot()

    # THE REAL predicate, imported -- not a copy of it. A hand-copied duplicate
    # here passed even with the production predicate reverted (mutation
    # F3_REVERT_consumer_predicate_is_None_only survived), which is the same
    # silent-drift defect that already exists in tests/services/test_sod_daily_roll.py.
    from backend.services.paper_trader import sod_anchor_needs_reroll
    assert sod_anchor_needs_reroll(snap, TODAY_UTC) is True

    # ... and the repair actually works.
    st.update_sod_nav(LIVE_SOD, date=TODAY_UTC)
    assert st.snapshot()["sod_nav"] == pytest.approx(LIVE_SOD)


def test_phase_36_9_a_latched_zero_still_re_anchors_defense_in_depth(isolated_state):
    """For a process that latched 0.0 BEFORE this fix and has not restarted: the
    consumer-side predicate must treat 0.0 as absent too. `is None` alone left the
    book wedged because 0.0 is not None and the date was today."""
    ks, st = isolated_state
    st._sod_nav = 0.0                 # simulate the pre-fix latched state
    st._sod_date = TODAY_UTC
    snap = st.snapshot()

    from backend.services.paper_trader import sod_anchor_needs_reroll
    assert sod_anchor_needs_reroll(snap, TODAY_UTC) is True, (
        "the pre-36.9 predicate `snap.get('sod_nav') is None` returned False here "
        "and the wedge persisted for up to 24h"
    )

    # ... and a genuinely healthy anchor must NOT be re-rolled (the predicate has
    # to discriminate, not just always return True).
    st._sod_nav = LIVE_SOD
    assert sod_anchor_needs_reroll(st.snapshot(), TODAY_UTC) is False


# --------------------------------------------------------------------------
# Criterion 4 -- the healthy path is unchanged, asserted against a FIXED fixture
# --------------------------------------------------------------------------

def test_phase_36_9_healthy_path_is_byte_for_byte_unchanged(isolated_state):
    """Fixed numbers, full-dict comparison. If any fix leaks into the healthy
    path -- an extra disarm, a changed percentage, a renamed key -- this fails.
    The two new keys are asserted as present-and-False rather than ignored."""
    ks, st = isolated_state
    st._sod_nav = 10000.0
    st._sod_date = TODAY_UTC
    st._peak_nav = 12000.0

    r = ks.evaluate_breach(11000.0, 4.0, 10.0)

    assert r == {
        "daily_loss_breached": False,
        "daily_loss_pct": -10.0,          # a PROFIT vs today's open
        "daily_loss_limit_pct": 4.0,
        "trailing_dd_breached": False,
        "trailing_dd_pct": 8.3333,
        "trailing_dd_limit_pct": 10.0,
        "any_breached": False,
        "daily_baseline_missing": False,
        "daily_baseline_stale": False,
        "trailing_baseline_missing": False,
        # Added in cycle 2. This whole-dict comparison is what CAUGHT the new key
        # appearing, which is the point of comparing the dict rather than probing
        # individual fields: every schema change has to be acknowledged here.
        "baselines_present": True,
        "armed": True,
    }


def test_phase_36_9_a_fresh_anchor_still_fires_the_daily_leg(isolated_state):
    """The other half of criterion 4: the fix must not cost us a real daily
    breach on a legitimately fresh anchor."""
    ks, st = isolated_state
    st._sod_nav = LIVE_SOD
    st._sod_date = TODAY_UTC
    st._peak_nav = LIVE_PEAK

    r = ks.evaluate_breach(NAV_AT_4PCT, 4.0, 10.0)
    assert r["armed"] is True
    assert r["daily_loss_breached"] is True
    assert r["daily_loss_pct"] == pytest.approx(4.0, abs=0.01)


def test_phase_36_9_the_disarm_log_names_staleness_not_absence(isolated_state, caplog):
    """An operator-facing string must not describe a defect that is not there. The
    pre-36.9 line read `baseline missing (sod_nav=23838.19 ...)` -- reporting a
    number it had just called missing, sending the reader to look for an absent
    baseline that was sitting right there with the wrong date on it."""
    ks, st = isolated_state
    st._sod_nav = LIVE_SOD
    st._sod_date = TWO_DAYS_AGO_UTC
    st._peak_nav = LIVE_PEAK

    with caplog.at_level("ERROR"):
        ks.evaluate_breach(LIVE_NAV, 4.0, 10.0)

    msg = "\n".join(r.getMessage() for r in caplog.records)
    assert "STALE" in msg
    assert "baseline missing (" not in msg


# --------------------------------------------------------------------------
# THE ORDER-PLACING PATH -- added in cycle 2 after a Q/A found a regression
# that all 162 tests were blind to.
#
# `armed` is read by a FOURTH consumer the first cycle of this step never
# considered: paper_trader.check_and_enforce_kill_switch, which measures it
# BEFORE the daily roll (36.12's deliberate ordering). Folding staleness into
# `armed` therefore made the ORDINARY first cycle of every UTC day look like
# lost history -- measured end-to-end: blocked=True,
# block_reason=kill_switch_disarmed_lost_history, a P1 page, and a fabricated
# `lost_history_anchor` row written into the live audit trail. Every morning.
#
# The fix splits the question: `baselines_present` (did we LOSE the baselines --
# a durable fault) versus `armed` (can the leg fire RIGHT NOW). The order gate
# reads the former; the read surfaces keep the latter. These tests exist because
# a unit test of `evaluate_breach` cannot see any of that.
# --------------------------------------------------------------------------

@pytest.fixture
def cycle_probe(isolated_state, monkeypatch):
    """Drive the REAL check_and_enforce_kill_switch with the pager captured.

    Captures rather than mocks-out the alert so a test can assert the P1 is NOT
    raised; 17 false P1s reached the operator's Slack earlier in this phase
    because a test exercised this path with the real dispatcher attached.
    """
    from backend.config.settings import get_settings
    from backend.services.paper_trader import PaperTrader
    import backend.services.observability.alerting as al

    ks, st = isolated_state
    alerts: list = []
    monkeypatch.setattr(al, "raise_cron_alert_sync", lambda *a, **k: alerts.append(k or a))

    def run(sod, sod_date, peak, nav=23800.0, starting_capital=20000.0):
        st._sod_nav, st._sod_date, st._peak_nav = sod, sod_date, peak
        st._paused = False
        alerts.clear()
        tr = object.__new__(PaperTrader)
        tr.settings = get_settings()
        tr.get_or_create_portfolio = lambda: {
            "total_nav": nav, "starting_capital": starting_capital}
        tr.flatten_all_positions = lambda *a, **k: []
        return tr.check_and_enforce_kill_switch(), alerts

    return run


def test_phase_36_9_an_overnight_anchor_does_not_halt_the_morning_cycle(cycle_probe):
    """THE REGRESSION, and the reason this section exists.

    A healthy funded book whose anchor is simply from yesterday must trade
    normally. Pre-fix (staleness folded into the flag this gate reads) it was
    blocked, paged and audit-stamped as lost history on the first cycle of every
    UTC day.
    """
    r, alerts = cycle_probe(LIVE_SOD, YESTERDAY_UTC, LIVE_PEAK)

    assert r.get("blocked") is not True, (
        "an overnight anchor is repaired by the daily roll on the very next line -- "
        "it is not lost history and must not halt the cycle"
    )
    assert r.get("block_reason") is None
    assert alerts == [], "no P1 may be raised for an ordinary overnight cycle"


def test_phase_36_9_genuine_lost_history_still_blocks(cycle_probe):
    """The 36.12 guarantee this must not cost us: absent baselines on a book that
    HAS traded still block orders and still page."""
    r, alerts = cycle_probe(None, None, None, nav=23800.0, starting_capital=20000.0)

    assert r["blocked"] is True
    assert r["block_reason"] == "kill_switch_disarmed_lost_history"
    assert len(alerts) == 1


def test_phase_36_9_a_lost_peak_still_blocks_even_with_a_stale_sod(cycle_probe):
    """Mixed state: one baseline genuinely absent, the other merely stale. Absence
    wins -- this is lost history and must block."""
    r, _ = cycle_probe(LIVE_SOD, YESTERDAY_UTC, None)

    assert r["blocked"] is True
    assert r["block_reason"] == "kill_switch_disarmed_lost_history"


def test_phase_36_9_the_two_questions_are_not_the_same_flag(isolated_state):
    """`baselines_present` and `armed` must actually DIVERGE on a stale anchor --
    otherwise the split is cosmetic and the order gate is still reading freshness."""
    ks, st = isolated_state
    st._sod_nav, st._sod_date, st._peak_nav = LIVE_SOD, TWO_DAYS_AGO_UTC, LIVE_PEAK

    r = ks.evaluate_breach(LIVE_NAV, 4.0, 10.0)
    assert r["baselines_present"] is True, "the baselines are present -- nothing is lost"
    assert r["armed"] is False, "but the daily leg cannot fire today"


def test_phase_36_9_the_resume_409_names_staleness_not_absence(isolated_state, monkeypatch):
    """Cycle 3, from a Q/A finding on the OPERATOR-FACING surface.

    The 36.7 resume gate refuses on the new staleness cause but kept the ABSENCE
    text: it told the operator "the loss baselines could not be restored" while
    printing `daily_baseline_missing=False, trailing_baseline_missing=False` -- a
    message its own diagnostics refute -- and pointed at a lost-history
    remediation that no longer happens for this cause, since the cycle-2 split
    means a stale anchor does NOT trip the order block.

    Same defect class already fixed in the disarm LOG this step; it survived here.
    Not a wedge: a PAUSED book still reaches the daily roll, so it self-clears in
    one cycle -- which is what the new message tells the operator.
    """
    import asyncio
    from fastapi import HTTPException
    import backend.api.paper_trading as api

    ks, st = isolated_state
    st._sod_nav, st._sod_date, st._peak_nav = LIVE_SOD, YESTERDAY_UTC, LIVE_PEAK
    st._paused = True

    class _BQ:
        def get_paper_portfolio(self, _):
            return {"total_nav": LIVE_NAV, "starting_capital": 20000.0}

    monkeypatch.setattr(api, "get_bq_client", lambda: _BQ())
    monkeypatch.setattr(api, "_get_ks_state", lambda: st)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(api.resume_trading(
            api.KillSwitchActionRequest(confirmation="RESUME")))

    assert exc.value.status_code == 409
    detail = exc.value.detail
    assert "STALE" in detail, "the refusal must name the ACTUAL cause"
    assert "could not be restored" not in detail, (
        "the absence text asserts a cause this state's own diagnostics refute"
    )
    assert "NO operator action is required" in detail, (
        "the remediation must describe the daily roll that actually clears it, "
        "not the lost-history block that does not fire for a stale anchor"
    )
    assert YESTERDAY_UTC in detail, "name the offending date so it is actionable"
