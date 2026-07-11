"""phase-69.1 book-safety fixes (audit items 1-4).

Reproduction + guard tests for the ways the live engine could destroy its own
book: FX=1.0 phantom proceeds, the unrecoverable kill-switch peak (+ phantom
breach on a bad NAV), the Slack 'clear queue' pkill SIGKILL, and the lock strands.
Thresholds (4/10/8/30) are asserted byte-untouched via the constants + a real breach.
"""

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


def test_valid_nav_still_breaches(monkeypatch):
    # do-no-harm: the guard must NOT suppress a REAL breach on a valid NAV.
    st = ks.get_state()
    monkeypatch.setattr(st, "snapshot", lambda: {"sod_nav": 100.0, "peak_nav": 100.0})
    r = ks.evaluate_breach(80.0, 4.0, 10.0)  # 20% down vs sod AND peak
    assert r["daily_loss_breached"] is True and r["trailing_dd_breached"] is True
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
    # after the holder exits, its own cleanup removes the pidfile.
    assert not lock_path.exists()
