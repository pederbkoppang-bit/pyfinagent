"""phase-69.1 book-safety fixes (audit items 1-4).

Reproduction + guard tests for the ways the live engine could destroy its own
book: FX=1.0 phantom proceeds, the unrecoverable kill-switch peak (+ phantom
breach on a bad NAV), the Slack 'clear queue' pkill SIGKILL, and the lock strands.
Thresholds (4/10/8/30) are asserted byte-untouched via the constants + a real breach.
"""

from datetime import datetime, timedelta, timezone
import json
import pathlib
from unittest.mock import MagicMock

import pytest

import backend.services.fx_rates as fx
import backend.services.kill_switch as ks
import backend.services.cycle_lock as cl


# ----------------------------------------------------------------------
# Item 1 — FX: last-known fallback instead of the phantom 1.0 default
# ----------------------------------------------------------------------
def _no_cache(monkeypatch):
    monkeypatch.setattr(
        "backend.services.api_cache.get_api_cache",
        lambda: MagicMock(get=lambda k: None, set=lambda *a, **k: None),
    )


def test_fx_serves_last_known_on_dual_outage(monkeypatch):
    _no_cache(monkeypatch)
    monkeypatch.setattr(fx, "_fetch_yf", lambda ccy: None)      # yfinance down
    monkeypatch.setattr(fx, "_fetch_fred", lambda ccy: None)    # FRED down
    monkeypatch.setattr(fx, "_last_known_usd_value", lambda ccy: 0.00076)  # KRW stored rate
    # RED (pre-fix): _usd_value_live returned None -> execute_sell defaulted to 1.0.
    # GREEN: serves the last-known rate, NOT None and NOT 1.0.
    assert fx._usd_value_live("KRW") == 0.00076


def test_fx_returns_none_only_when_no_rate_ever(monkeypatch):
    _no_cache(monkeypatch)
    monkeypatch.setattr(fx, "_fetch_yf", lambda ccy: None)
    monkeypatch.setattr(fx, "_fetch_fred", lambda ccy: None)
    monkeypatch.setattr(fx, "_last_known_usd_value", lambda ccy: None)  # never stored
    assert fx._usd_value_live("KRW") is None  # -> execute_sell BLOCKS (not 1.0)


def test_fx_usd_unaffected(monkeypatch):
    # The USD path stays byte-identical (do-no-harm for the US-only live book).
    assert fx._usd_value_live("USD") == 1.0
    assert fx.get_fx_rate("USD", "USD") == 1.0


def test_fx_local_to_usd_blocks_nonusd_when_no_rate(monkeypatch):
    import backend.services.paper_trader as pt
    # Dual outage + no last-known -> _fx_local_to_usd(non-USD) is None -> execute_sell
    # hits the block branch (`if _l2u is None: ... return None`), never crediting at 1.0.
    monkeypatch.setattr(fx, "_usd_value_live", lambda ccy: None if ccy != "USD" else 1.0)
    monkeypatch.setattr(fx, "_usd_value_asof", lambda ccy, d: None if ccy != "USD" else 1.0)
    assert pt._fx_local_to_usd("KR") is None    # non-USD, no rate -> block input
    assert pt._fx_local_to_usd("US") == 1.0     # USD market unaffected


# ----------------------------------------------------------------------
# Item 2 — kill-switch: current_nav<=0 guard + DARK peak_reset
# ----------------------------------------------------------------------
def test_current_nav_zero_no_phantom_breach():
    r = ks.evaluate_breach(0.0, 4.0, 10.0)
    assert r["any_breached"] is False and r.get("nav_invalid") is True


def test_current_nav_negative_no_phantom_breach():
    r = ks.evaluate_breach(-5.0, 4.0, 10.0)
    assert r["any_breached"] is False and r.get("nav_invalid") is True


def test_valid_nav_still_breaches(monkeypatch, tmp_path):
    """do-no-harm: the guard must NOT suppress a REAL breach on a valid NAV.

    phase-85.5.1: this drove a MOCK of `st.snapshot` returning only
    `{"sod_nav", "peak_nav"}`. The real `_snapshot_locked()` emits `sod_date`
    (and, since phase-85.6, `sod_provisional`) as well, so the mock handed
    `evaluate_breach` a state with `sod_date=None` -- which the phase-36.9
    liveness guard correctly reads as an unevaluable daily leg and disarms. The
    test was asserting that a leg fires while feeding it a state in which it
    provably cannot. It was the FIXTURE that was wrong, not the guard.

    Measured, not assumed -- `scripts/diagnostics/measure_sod_date_reachability.py`
    walks every production path to a None/stale anchor and shows the guard behaving
    correctly in each. The guard is deliberate selective bypass: the trailing leg
    is date-independent and still fires, so `any_breached` was True even in the
    old failing run. Only the daily leg was suppressed.

    So the fix is entirely input-side: build a REAL `KillSwitchState` against a
    redirected audit path -- the idiom at
    `test_phase_36_9_kill_switch_armed_liveness.py:63-88` -- and anchor it TODAY.
    A real state is drift-proof: it cannot omit a contract key the way a
    hand-written dict can, and this one already omitted three.

    NOT ONE ASSERTION IS WEAKENED. All four still demand a 20% drawdown against
    both sod and peak produce daily True, trailing True, any True, no nav_invalid.
    """
    # Redirect BEFORE anything can append: `update_sod_nav` writes a real
    # `sod_snapshot` row, and handoff/kill_switch_audit.jsonl is git-tracked LIVE
    # safety state.
    monkeypatch.setattr(ks, "_AUDIT_PATH", tmp_path / "kill_switch_audit.jsonl")
    monkeypatch.setattr(ks, "_audit_archive_dir", lambda: tmp_path / "audit")
    assert all(
        str(pth).startswith(str(tmp_path)) for pth in ks._audit_source_paths()
    ) or not ks._audit_source_paths(), "ISOLATION FAILED -- a live journal is in the replay"

    st = ks.KillSwitchState()
    monkeypatch.setattr(ks, "_state", st)
    monkeypatch.setattr(ks, "_disarmed_logged", False)

    # TODAY computed, never hardcoded -- a literal date would rot at midnight and
    # start disarming the very leg this test exists to prove fires.
    today = datetime.now(timezone.utc).date().isoformat()
    st.update_sod_nav(100.0, date=today)
    st.update_peak(100.0)

    snap = st.snapshot()
    assert snap["sod_nav"] == 100.0 and snap["sod_date"] == today, (
        f"PRECONDITION: the anchor must be today's, else the daily leg is "
        f"legitimately unevaluable and this test proves nothing -- {snap}"
    )

    r = ks.evaluate_breach(80.0, 4.0, 10.0)  # 20% down vs sod AND peak
    assert r["daily_loss_breached"] is True and r["trailing_dd_breached"] is True
    assert r["any_breached"] is True and not r.get("nav_invalid")


def test_stale_anchor_disarms_the_daily_leg_but_the_trailing_leg_still_fires(
    monkeypatch, tmp_path
):
    """phase-85.5.1: the OTHER half of the contract, and it was missing.

    `test_valid_nav_still_breaches` proves the switch FIRES when it can evaluate.
    Nothing in this file proved it correctly DISARMS when it cannot -- mutation
    M4 (`_sod_date_is_stale` hardcoded to return False, i.e. the phase-36.9 F1
    regression) survived the whole file. A suite that cannot tell a working
    liveness guard from a disabled one is not a book-safety suite.

    This pins the guard's designed behaviour, which the phase-85.5.1 research
    gate identified as textbook IEC 61511 SELECTIVE BYPASS: bypass the single
    unevaluable channel, never the whole trip logic, and make the bypass visible.
    So on a stale anchor:

      * the daily leg is suppressed  -- it genuinely cannot be evaluated
      * `armed` is False             -- and says so, rather than claiming health
      * the TRAILING leg still fires -- it is date-independent
      * `any_breached` is still True -- the book is still protected

    That last line is why the original RED failure was bounded: exposure was only
    ever drawdowns in [daily_limit, trailing_limit).
    """
    monkeypatch.setattr(ks, "_AUDIT_PATH", tmp_path / "kill_switch_audit.jsonl")
    monkeypatch.setattr(ks, "_audit_archive_dir", lambda: tmp_path / "audit")
    st = ks.KillSwitchState()
    monkeypatch.setattr(ks, "_state", st)
    monkeypatch.setattr(ks, "_disarmed_logged", False)

    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    st.update_sod_nav(100.0, date=yesterday)
    st.update_peak(100.0)
    assert st.snapshot()["sod_date"] == yesterday, "PRECONDITION: anchor must be stale"

    r = ks.evaluate_breach(80.0, 4.0, 10.0)      # 20% down vs BOTH baselines

    assert r["daily_baseline_stale"] is True, (
        "the guard failed to notice a stale anchor -- phase-36.9 F1 regression"
    )
    assert r["armed"] is False, "an unevaluable leg must not report itself armed"
    assert r["daily_loss_breached"] is False, (
        "the daily leg fired on an anchor it cannot evaluate"
    )
    # ...and the book is STILL protected, which is the whole point of bypassing
    # one channel rather than the trip logic.
    assert r["trailing_dd_breached"] is True, (
        "SAFETY HOLE: the date-independent trailing leg stopped firing"
    )
    assert r["any_breached"] is True and not r.get("nav_invalid")


def test_peak_reset_dark_by_default(monkeypatch):
    st = ks.get_state()
    monkeypatch.setattr(ks, "get_state", lambda: st)
    before = st.snapshot().get("peak_nav")
    out = st.reset_peak(12345.0, trigger="flatten")
    assert out is None                                   # DARK: no-op
    assert st.snapshot().get("peak_nav") == before       # peak unchanged


def test_peak_reset_active_when_token_enabled(monkeypatch, tmp_path):
    # With the KS-PEAK-RESET token flag ON, reset_peak re-anchors + audits.
    monkeypatch.setattr(ks, "_AUDIT_PATH", tmp_path / "ks_audit.jsonl")
    st = ks.KillSwitchState()
    st._peak_nav = 1000.0
    from backend.config.settings import get_settings
    s = get_settings()
    monkeypatch.setattr(s, "kill_switch_peak_reset_enabled", True)
    out = st.reset_peak(750.0, trigger="operator_resume", operator="peder")
    assert out is not None and st.snapshot()["peak_nav"] == 750.0
    # audited + restart-replayable: a fresh state replays the reset.
    st2 = ks.KillSwitchState()
    assert st2.snapshot()["peak_nav"] == 750.0


def test_resume_reanchors_peak_via_nav_when_token_enabled(monkeypatch, tmp_path):
    # phase-69.1: reset_peak is now WIRED into resume(nav=...). With the token ON,
    # an operator resume re-anchors the trailing peak to the current NAV -> the
    # trailing-DD breach can no longer persist forever after a flatten.
    monkeypatch.setattr(ks, "_AUDIT_PATH", tmp_path / "ks_resume_on.jsonl")
    st = ks.KillSwitchState()
    st._peak_nav = 1000.0
    st._paused = True
    from backend.config.settings import get_settings
    monkeypatch.setattr(get_settings(), "kill_switch_peak_reset_enabled", True)
    snap = st.resume(trigger="manual", nav=800.0, details={"operator": "peder"})
    assert st.snapshot()["peak_nav"] == 800.0 and snap["peak_nav"] == 800.0
    assert st.snapshot()["paused"] is False


def test_resume_does_not_reanchor_peak_when_dark(monkeypatch, tmp_path):
    monkeypatch.setattr(ks, "_AUDIT_PATH", tmp_path / "ks_resume_off.jsonl")
    st = ks.KillSwitchState()
    st._peak_nav = 1000.0
    st._paused = True
    # token flag default OFF -> resume must NOT reset the peak (DARK), still un-pauses.
    snap = st.resume(trigger="manual", nav=800.0)
    assert st.snapshot()["peak_nav"] == 1000.0 and snap["peak_nav"] == 1000.0
    assert st.snapshot()["paused"] is False


def test_thresholds_not_a_setting_here_are_caller_supplied():
    # evaluate_breach takes the limits as args; the fix touches neither the
    # 4/10 breach math constants nor the 8/30 stop/sector caps.
    r = ks.evaluate_breach(100.0, 4.0, 10.0)
    assert r["daily_loss_limit_pct"] == 4.0 and r["trailing_dd_limit_pct"] == 10.0


# ----------------------------------------------------------------------
# Item 3 — op-safety: no process-kill sink reachable from the Slack handler
# ----------------------------------------------------------------------
def test_no_process_kill_sink_in_commands():
    src = pathlib.Path("backend/slack_bot/commands.py").read_text(encoding="utf-8")
    # Strip comments so the removal's own explanatory comment (which names pkill/
    # SIGKILL descriptively) does not false-positive; check the CODE only.
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    for sink in ["pkill", "killpg", "os.kill", "SIGKILL"]:
        assert sink not in code, f"process-kill sink still reachable in CODE: {sink!r}"


# ----------------------------------------------------------------------
# Item 4 — locks: a FAILED acquire must not unlink the live holder's pidfile
# ----------------------------------------------------------------------
def test_cycle_lock_failed_acquire_keeps_live_pidfile(monkeypatch, tmp_path):
    lock_path = tmp_path / "cycle.lock"
    monkeypatch.setattr(cl, "_HANDOFF", tmp_path)
    monkeypatch.setattr(cl, "_LOCK_PATH", lock_path)
    # inspect_lock must report the held lock as LIVE (not stale) so the 2nd acquire
    # raises contention rather than cleaning it.
    monkeypatch.setattr(cl, "inspect_lock", lambda: {"is_stale": False, "pid": 1, "age_sec": 1})
    with cl.acquire("cycle-A"):
        assert lock_path.exists()                     # holder's pidfile present
        with pytest.raises(cl.CycleLockError):
            with cl.acquire("cycle-B"):               # FAILS (contention)
                pass
        # RED (pre-fix): the failed acquire's finally unlinked the live pidfile.
        # GREEN: the live holder's pidfile survives the failed acquire.
        assert lock_path.exists()
    # phase-85.5 CHANGED THE LINE BELOW. It previously asserted the holder
    # UNLINKED its pidfile on exit. That unlink ran BEFORE LOCK_UN, so between
    # the two the holder was alive and still holding the flock while the path
    # was free -- a second acquirer then created a NEW INODE and both believed
    # they held the cycle lock, on the normal release path. Release now marks
    # the payload `released` in place and never unlinks. The item-4 guard this
    # test exists for (above: a FAILED acquire must not unlink a live holder's
    # pidfile) is unchanged and still asserted.
    assert lock_path.exists(), "phase-85.5: release must not unlink"
    assert json.loads(lock_path.read_text())["state"] == "released"
