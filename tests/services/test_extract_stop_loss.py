"""phase-23.1.8: settings-driven default in _extract_stop_loss closes the
lite-Claude-analyzer gap (risk_assessment={"reason": ...} has no risk_limits)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.services.portfolio_manager import _extract_stop_loss


def _settings(default_pct: float = 8.0) -> SimpleNamespace:
    return SimpleNamespace(paper_default_stop_loss_pct=default_pct)


def test_explicit_stop_loss_wins_over_default():
    risk = {"risk_limits": {"stop_loss": 87.5}}
    analysis = {"price_at_analysis": 100.0}
    assert _extract_stop_loss(risk, analysis, settings=_settings()) == 87.5


def test_stop_loss_pct_in_risk_limits_takes_precedence_over_settings_default():
    risk = {"risk_limits": {"stop_loss_pct": 10.0}}
    analysis = {"price_at_analysis": 100.0}
    out = _extract_stop_loss(risk, analysis, settings=_settings())
    assert out == pytest.approx(90.0)


def test_lite_path_no_risk_limits_uses_settings_default():
    """The bug fix: lite Claude returns risk_assessment={'reason': ...}, no risk_limits."""
    risk = {"reason": "strong momentum + reasonable valuation"}
    analysis = {"price_at_analysis": 100.0}
    out = _extract_stop_loss(risk, analysis, settings=_settings())
    assert out == pytest.approx(92.0)  # 100 * (1 - 8/100)


def test_settings_default_override():
    risk = {"reason": "x"}
    analysis = {"price_at_analysis": 100.0}
    out = _extract_stop_loss(risk, analysis, settings=_settings(default_pct=15.0))
    assert out == pytest.approx(85.0)


def test_no_settings_preserves_old_behavior_returns_none():
    """settings=None → no fallback fires → None (backward compat)."""
    risk = {"reason": "x"}
    analysis = {"price_at_analysis": 100.0}
    assert _extract_stop_loss(risk, analysis) is None
    assert _extract_stop_loss(risk, analysis, settings=None) is None


def test_no_price_returns_none_even_with_settings():
    """No price_at_analysis → cannot derive any stop, even with settings default."""
    risk = {"reason": "x"}
    analysis = {}
    assert _extract_stop_loss(risk, analysis, settings=_settings()) is None


def test_empty_risk_assessment_with_price_uses_default():
    risk = {}
    analysis = {"price_at_analysis": 200.0}
    out = _extract_stop_loss(risk, analysis, settings=_settings())
    assert out == pytest.approx(184.0)  # 200 * 0.92


def test_explicit_zero_stop_loss_falls_through_to_pct():
    """A 0 / falsy explicit stop_loss should not block the next chains."""
    risk = {"risk_limits": {"stop_loss": 0, "stop_loss_pct": 5.0}}
    analysis = {"price_at_analysis": 100.0}
    out = _extract_stop_loss(risk, analysis, settings=_settings())
    assert out == pytest.approx(95.0)


def test_settings_without_attribute_returns_none():
    """Settings object missing the attr → fallback to None (graceful)."""
    bare = SimpleNamespace()  # no paper_default_stop_loss_pct attr
    risk = {"reason": "x"}
    analysis = {"price_at_analysis": 100.0}
    assert _extract_stop_loss(risk, analysis, settings=bare) is None


def test_invalid_default_pct_returns_none():
    """Non-numeric setting → graceful None."""
    bad = SimpleNamespace(paper_default_stop_loss_pct="not a number")
    risk = {"reason": "x"}
    analysis = {"price_at_analysis": 100.0}
    assert _extract_stop_loss(risk, analysis, settings=bad) is None
