"""phase-23.2.A-fix: Risk Judge lite-path duplicate detection."""

from __future__ import annotations

from backend.services.signal_attribution import extract_signals_from_analysis


def test_lite_path_duplicate_riskjudge_marked():
    """When risk_assessment lacks recommended_position_pct (lite-path) AND
    its reasoning equals the Trader reasoning, the RiskJudge entry must be
    relabeled with `lite_path: True` and a clear operator-facing rationale."""
    analysis = {
        "recommendation": "BUY",
        "final_score": 7.0,
        "trader_note": "Strong momentum with industrial sector strength.",
        "risk_assessment": {
            "decision": "APPROVE",
            "reason": "Strong momentum with industrial sector strength.",  # IDENTICAL
            # NO recommended_position_pct -> weight=0.0 in lite-path
        },
    }
    signals = extract_signals_from_analysis(analysis)
    risk_rows = [s for s in signals if s["agent"] == "RiskJudge"]
    assert len(risk_rows) == 1
    rj = risk_rows[0]
    assert rj["weight"] == 0.0
    assert rj.get("lite_path") is True
    assert "Lite-path" in rj["rationale"]
    assert "no independent risk debate" in rj["rationale"]


def test_full_path_riskjudge_unchanged():
    """When risk_assessment HAS recommended_position_pct (full path), the
    RiskJudge entry retains its original reasoning and weight; no lite_path
    flag."""
    analysis = {
        "recommendation": "BUY",
        "final_score": 7.0,
        "trader_note": "Strong momentum.",
        "risk_assessment": {
            "decision": "APPROVE",
            "reasoning": "Volatility manageable; suggest 8% position size.",
            "recommended_position_pct": 8.0,
        },
    }
    signals = extract_signals_from_analysis(analysis)
    risk_rows = [s for s in signals if s["agent"] == "RiskJudge"]
    assert len(risk_rows) == 1
    rj = risk_rows[0]
    assert rj["weight"] == 8.0
    assert rj.get("lite_path") is None or rj.get("lite_path") is False
    assert "Lite-path" not in rj["rationale"]
    assert "Volatility manageable" in rj["rationale"]


def test_distinct_lite_reasoning_not_marked():
    """When the lite-path RiskJudge reasoning happens to differ from
    Trader's, do NOT flag as a duplicate."""
    analysis = {
        "recommendation": "BUY",
        "final_score": 7.0,
        "trader_note": "Strong momentum.",
        "risk_assessment": {
            "decision": "APPROVE",
            "reason": "Acceptable volatility for current portfolio mix.",  # different
        },
    }
    signals = extract_signals_from_analysis(analysis)
    risk_rows = [s for s in signals if s["agent"] == "RiskJudge"]
    assert len(risk_rows) == 1
    rj = risk_rows[0]
    assert rj.get("lite_path") is None or rj.get("lite_path") is False
    assert "Acceptable volatility" in rj["rationale"]
