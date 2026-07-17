"""phase-64.3: kill-switch state-machine gap tests (pure; no live net/BQ).

Criterion 2: the kill-switch stays-paused policy that RAIL 5 depends on
(docs/runbooks/away-ops-rules.md:17-18 verbatim: "Kill-switch stays paused
after any breach; auto-resume hysteresis stays OFF"). evaluate_breach +
check_auto_resume(enabled=False) are pure functions over a fresh, tmp-backed
KillSwitchState (mirrors test_phase_38_1_kill_switch_auto_resume.py:26).
"""
from __future__ import annotations


def _fresh_state(monkeypatch, tmp_path):
    from backend.services import kill_switch
    audit_path = tmp_path / "ks_audit_64_3.jsonl"
    monkeypatch.setattr(kill_switch, "_AUDIT_PATH", audit_path)
    state = kill_switch.KillSwitchState()
    monkeypatch.setattr(kill_switch, "_state", state)
    return kill_switch, state, audit_path


def test_64_3_kill_switch_machine_pause_sets_paused(monkeypatch, tmp_path):
    _ks, state, _ = _fresh_state(monkeypatch, tmp_path)
    assert state.is_paused() is False
    state.pause(trigger="test")
    assert state.is_paused() is True


def test_64_3_kill_switch_machine_stays_paused_auto_resume_off(monkeypatch, tmp_path):
    """RAIL 5: with auto-resume OFF, a paused switch NEVER auto-resumes."""
    ks, state, _ = _fresh_state(monkeypatch, tmp_path)
    state.update_sod_nav(100_000.0)
    state.update_peak(100_000.0)
    state.pause(trigger="test_breach")
    out = ks.check_auto_resume(
        current_nav=99_000.0,  # only 1% loss vs sod=100K -> healthy, no breach
        daily_loss_limit_pct=4.0,
        trailing_dd_limit_pct=10.0,
        enabled=False,
    )
    assert out["action"] == "no_op"
    assert "auto_resume_disabled" in out["reason"]
    # The invariant rail 5 depends on: STILL paused despite a healthy NAV.
    assert state.is_paused() is True


def test_64_3_kill_switch_machine_active_breach_stays_paused(monkeypatch, tmp_path):
    """Even with auto-resume ENABLED, an active breach must NOT resume."""
    ks, state, _ = _fresh_state(monkeypatch, tmp_path)
    state.update_sod_nav(100_000.0)
    state.update_peak(100_000.0)
    state.pause(trigger="test_breach")
    out = ks.check_auto_resume(
        current_nav=90_000.0,  # 10% loss vs sod -> breaches the 4% daily limit
        daily_loss_limit_pct=4.0,
        trailing_dd_limit_pct=10.0,
        enabled=True,
    )
    assert out["action"] != "resume"
    assert state.is_paused() is True


def test_64_3_kill_switch_machine_evaluate_breach_invalid_nav(monkeypatch, tmp_path):
    """phase-69.1: NAV<=0 (a BQ-timeout `or 0.0` fallback) is no-data, NOT a
    phantom 100% breach (fail-safe: no breach)."""
    ks, state, _ = _fresh_state(monkeypatch, tmp_path)
    state.update_sod_nav(100_000.0)
    state.update_peak(100_000.0)
    out = ks.evaluate_breach(0.0, 4.0, 10.0)
    assert out["nav_invalid"] is True
    assert out["any_breached"] is False


def test_64_3_kill_switch_machine_evaluate_breach_clean(monkeypatch, tmp_path):
    ks, state, _ = _fresh_state(monkeypatch, tmp_path)
    state.update_sod_nav(100_000.0)
    state.update_peak(100_000.0)
    out = ks.evaluate_breach(99_500.0, 4.0, 10.0)  # 0.5% loss -> under limits
    assert out["any_breached"] is False
    assert out.get("nav_invalid") is not True
