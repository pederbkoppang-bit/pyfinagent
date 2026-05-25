"""phase-38.1 (OPEN-10) -- kill-switch auto-resume hysteresis tests.

Per masterplan 38.1 criteria:
  1. kill_switch_auto_resume_on_no_breach_mode_added
  2. paused_with_no_breach_for_2h_triggers_resume
  3. paused_with_breach_stays_paused
  4. pager_alert_at_plus_1h_prior_to_auto_resume
  5. default_off_feature_flag_owner_approval_recorded

Operator-approval criterion: settings.kill_switch_auto_resume_enabled
defaults False; opt-in is the explicit operator-approval surface.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest


def _fresh_state(monkeypatch, tmp_path):
    from backend.services import kill_switch
    audit_path = tmp_path / "ks_audit.jsonl"
    monkeypatch.setattr(kill_switch, "_AUDIT_PATH", audit_path)
    state = kill_switch.KillSwitchState()
    monkeypatch.setattr(kill_switch, "_state", state)
    return kill_switch, state, audit_path


# Criterion 1
def test_phase_38_1_check_auto_resume_function_added(monkeypatch, tmp_path):
    ks, _, _ = _fresh_state(monkeypatch, tmp_path)
    assert hasattr(ks, "check_auto_resume"), "kill_switch.check_auto_resume must exist"
    # Default-OFF: enabled=False returns no_op
    out = ks.check_auto_resume(current_nav=100_000.0,
                               daily_loss_limit_pct=4.0,
                               trailing_dd_limit_pct=10.0,
                               enabled=False)
    assert out["action"] == "no_op"
    assert "auto_resume_disabled" in out["reason"]


# Criterion 2
def test_phase_38_1_paused_with_no_breach_for_2h_triggers_resume(monkeypatch, tmp_path):
    ks, state, _ = _fresh_state(monkeypatch, tmp_path)
    # Pause 2.5h ago, peak 100K, current 99K (no breach)
    state.update_sod_nav(100_000.0)
    state.update_peak(100_000.0)
    state.pause(trigger="test_breach_then_recover")
    # Backdate the pause to 2.5h ago
    with state._lock:
        state._paused_at = (datetime.now(timezone.utc) - timedelta(hours=2, minutes=30)).isoformat()
    # Sanity: still paused
    assert state.is_paused()
    out = ks.check_auto_resume(
        current_nav=99_000.0,    # only 1% loss vs sod=100K; under 4% limit
        daily_loss_limit_pct=4.0,
        trailing_dd_limit_pct=10.0,
        enabled=True,
    )
    assert out["action"] == "resume"
    assert "no_breach_for_2h" in out["reason"]
    assert state.is_paused() is False


# Criterion 3
def test_phase_38_1_paused_with_breach_stays_paused(monkeypatch, tmp_path):
    ks, state, _ = _fresh_state(monkeypatch, tmp_path)
    state.update_sod_nav(100_000.0)
    state.update_peak(100_000.0)
    state.pause(trigger="test")
    with state._lock:
        state._paused_at = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    # NAV down 10% from SOD -> daily-loss breach still active
    out = ks.check_auto_resume(
        current_nav=90_000.0,
        daily_loss_limit_pct=4.0,
        trailing_dd_limit_pct=10.0,
        enabled=True,
    )
    assert out["action"] == "no_op"
    assert "breach_still_active" in out["reason"]
    assert state.is_paused() is True


# Criterion 4
def test_phase_38_1_pager_alert_at_plus_1h_prior_to_auto_resume(monkeypatch, tmp_path):
    ks, state, audit_path = _fresh_state(monkeypatch, tmp_path)
    state.update_sod_nav(100_000.0)
    state.update_peak(100_000.0)
    state.pause(trigger="test")
    # Pause 1.5h ago (between 1h-alert and 2h-resume thresholds)
    with state._lock:
        state._paused_at = (datetime.now(timezone.utc) - timedelta(hours=1, minutes=30)).isoformat()
    with patch("backend.services.observability.alerting.raise_cron_alert_sync") as alert:
        out = ks.check_auto_resume(
            current_nav=99_000.0,
            daily_loss_limit_pct=4.0,
            trailing_dd_limit_pct=10.0,
            enabled=True,
        )
    assert out["action"] == "alert"
    assert "1h_pager_fired" in out["reason"]
    # State still paused (T+1h is alert-only; resume fires at T+2h)
    assert state.is_paused() is True
    # Audit log carries the alert event
    rows = audit_path.read_text(encoding="utf-8").splitlines()
    assert any('"event": "auto_resume_alert"' in r for r in rows)


def test_phase_38_1_pager_alert_one_shot_no_re_fire(monkeypatch, tmp_path):
    ks, state, _ = _fresh_state(monkeypatch, tmp_path)
    state.update_sod_nav(100_000.0)
    state.update_peak(100_000.0)
    state.pause(trigger="test")
    with state._lock:
        state._paused_at = (datetime.now(timezone.utc) - timedelta(hours=1, minutes=30)).isoformat()
    # First call: alert fires
    with patch("backend.services.observability.alerting.raise_cron_alert_sync"):
        out1 = ks.check_auto_resume(99_000.0, 4.0, 10.0, enabled=True)
    assert out1["action"] == "alert"
    # Second call (same pause cycle, still <2h): should NOT re-alert
    with patch("backend.services.observability.alerting.raise_cron_alert_sync"):
        out2 = ks.check_auto_resume(99_000.0, 4.0, 10.0, enabled=True)
    assert out2["action"] == "no_op"
    assert "under_hysteresis_threshold" in out2["reason"] or "alerted" in out2["reason"]


# Criterion 5
def test_phase_38_1_default_off_settings_flag():
    from backend.config.settings import Settings
    s = Settings()
    assert s.kill_switch_auto_resume_enabled is False, (
        "Settings.kill_switch_auto_resume_enabled MUST default to False -- operator opt-in"
    )


def test_phase_38_1_settings_flag_documents_owner_approval():
    """The settings field description must explicitly cite that flipping
    True is the operator opt-in / approval surface."""
    import inspect
    from backend.config.settings import Settings
    src = inspect.getsource(Settings)
    assert "kill_switch_auto_resume_enabled" in src
    assert "operator opt-in" in src.lower() or "operator approval" in src.lower(), (
        "settings field must document that True is operator opt-in"
    )


def test_phase_38_1_no_pause_timestamp_returns_no_op(monkeypatch, tmp_path):
    """Legacy audit rows lack `ts`; check_auto_resume must fail-open (no_op)
    rather than misfire on legacy paused state."""
    ks, state, _ = _fresh_state(monkeypatch, tmp_path)
    state.pause(trigger="legacy")
    with state._lock:
        state._paused_at = None
    out = ks.check_auto_resume(99_000.0, 4.0, 10.0, enabled=True)
    assert out["action"] == "no_op"
    assert "no_paused_at" in out["reason"]


# ============================================================
# phase-38.1.1 -- paper_trader wires check_auto_resume into the cycle
# ============================================================


def test_phase_38_1_1_paper_trader_imports_check_auto_resume():
    from pathlib import Path
    pt = (Path(__file__).resolve().parents[2] / "backend" / "services" / "paper_trader.py").read_text(encoding="utf-8")
    assert "check_auto_resume" in pt, (
        "paper_trader.py must import check_auto_resume from kill_switch"
    )
    assert "phase-38.1.1" in pt, "paper_trader.py must mark the wire site"


def test_phase_38_1_1_check_and_enforce_kill_switch_invokes_auto_resume(monkeypatch, tmp_path):
    """When flag is OFF, the wire is invoked but returns no_op (default-OFF)."""
    from backend.services import kill_switch, paper_trader
    from backend.config.settings import Settings
    monkeypatch.setattr(kill_switch, "_AUDIT_PATH", tmp_path / "ks_audit.jsonl")
    fresh = kill_switch.KillSwitchState()
    monkeypatch.setattr(kill_switch, "_state", fresh)
    monkeypatch.setattr(kill_switch, "get_state", lambda: fresh)
    s = Settings()
    s.paper_daily_loss_limit_pct = 4.0
    s.paper_trailing_dd_limit_pct = 10.0
    # Flag OFF (default)
    s.kill_switch_auto_resume_enabled = False
    bq = MagicMock()
    bq.get_paper_portfolio.return_value = {
        "current_cash": 100_000.0, "starting_capital": 100_000.0,
        "total_nav": 100_000.0, "portfolio_id": "p1",
    }
    trader = paper_trader.PaperTrader(settings=s, bq_client=bq)
    result = trader.check_and_enforce_kill_switch()
    # Wire fires but is no_op (flag OFF)
    assert "auto_resume" in result
    assert result["auto_resume"]["action"] == "no_op"
    assert "disabled" in result["auto_resume"]["reason"]


def test_phase_38_1_resume_clears_pause_cycle_state(monkeypatch, tmp_path):
    ks, state, _ = _fresh_state(monkeypatch, tmp_path)
    state.pause(trigger="test")
    assert state.snapshot()["paused_at"] is not None
    state.resume(trigger="manual")
    assert state.snapshot()["paused_at"] is None
    assert state.snapshot()["auto_resume_alerted_at"] is None
