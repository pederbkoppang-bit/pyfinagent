"""phase-23.1.7: extractor extracts the lite-Claude shape's reason fields and
produces Quant + SignalStack signals from a screener candidate dict."""

from __future__ import annotations

import pytest

from backend.services.signal_attribution import (
    extract_signals_from_analysis,
    extract_quant_signals,
    extract_all_signals,
    group_signals_for_drawer,
)


def _lite_analysis(reason: str = "Q1 beat with margin expansion",
                   risk_reason: str = "Strong momentum + reasonable valuation",
                   rec: str = "BUY", score: float = 7) -> dict:
    """Mirrors what `_run_claude_analysis` returns in autonomous_loop.py."""
    return {
        "ticker": "ON",
        "recommendation": rec,
        "final_score": score,
        "risk_assessment": {"reason": risk_reason},
        "price_at_analysis": 175.89,
        "analysis_date": "2026-04-26T23:43:56+00:00",
        "total_cost_usd": 0.01,
        "full_report": {
            "source": "claude-sonnet-4",
            "analysis": {"action": rec, "confidence": 75, "score": score, "reason": reason},
            "market_data": {"sector": "Technology", "momentum_20d": 6.1, "momentum_60d": 14.2},
        },
    }


def _screener_candidate() -> dict:
    return {
        "ticker": "ON",
        "sector": "Technology",
        "momentum_1m": 4.2,
        "momentum_3m": 11.8,
        "momentum_6m": 24.0,
        "rsi_14": 58.3,
        "volatility_ann": 0.28,
        "composite_score": 8.45,
        "conviction_score": 8,
        "conviction_reason": "strong momentum + positive PEAD",
        "regime_tag": "risk_on",
        "pead_tag": "positive_surprise",
        "news_event_type": "earnings_beat",
        "news_rationale": "Q1 beat consensus",
        "sector_event_type": "earnings",
        "source": "news_only",
    }


# ── Trader extraction ───────────────────────────────────────────

def test_trader_extracts_lite_full_report_reason():
    """phase-23.1.7 fix: full_report.analysis.reason must surface as Trader rationale.
    phase-25.D: Trader weight is now normalized to 0-1 (was 0-10)."""
    sigs = extract_signals_from_analysis(_lite_analysis(reason="Q1 beat with margin expansion"))
    trader = next(s for s in sigs if s["agent"] == "Trader")
    assert "Q1 beat" in trader["rationale"]
    assert trader["weight"] == 0.7  # phase-25.D: final_score=7 -> 0.7


def test_trader_falls_back_to_recommendation_when_no_reason_anywhere():
    a = {"recommendation": "HOLD", "final_score": 5}
    sigs = extract_signals_from_analysis(a)
    trader = next(s for s in sigs if s["agent"] == "Trader")
    assert trader["rationale"] == "Recommendation: HOLD"


def test_trader_prefers_explicit_trader_note_over_lite_path():
    """If both keys are present, trader_note wins (full Gemini path takes priority)."""
    a = _lite_analysis(reason="lite reason")
    a["trader_note"] = "explicit trader note"
    sigs = extract_signals_from_analysis(a)
    trader = next(s for s in sigs if s["agent"] == "Trader")
    assert "explicit trader note" in trader["rationale"]


# ── Risk Judge extraction ───────────────────────────────────────

def test_risk_extracts_reason_key_lite_shape():
    """phase-23.1.7 fix: risk_assessment.reason must surface as RiskJudge rationale."""
    sigs = extract_signals_from_analysis(_lite_analysis(risk_reason="Strong momentum"))
    risk = next(s for s in sigs if s["agent"] == "RiskJudge")
    assert "Strong momentum" in risk["rationale"]


def test_risk_prefers_reasoning_over_reason():
    """Full-shape `reasoning` should take priority over lite-shape `reason`."""
    a = {
        "recommendation": "BUY", "final_score": 6,
        "risk_assessment": {"reasoning": "explicit reasoning", "reason": "lite reason"},
    }
    sigs = extract_signals_from_analysis(a)
    risk = next(s for s in sigs if s["agent"] == "RiskJudge")
    assert "explicit reasoning" in risk["rationale"]


def test_risk_skipped_when_neither_decision_nor_reasoning():
    a = {"recommendation": "BUY", "final_score": 6, "risk_assessment": {}}
    sigs = extract_signals_from_analysis(a)
    assert not any(s["agent"] == "RiskJudge" for s in sigs)


# ── phase-25.F regression: lock 25.B aliasing-cleanup ───────────

def test_lite_path_byte_identical_flagged():
    """phase-25.F: byte-identical RiskJudge/Trader rationale must NOT carry
    a `lite_path` field. Cycle 85 (25.B) removed that aliasing-detection
    branch; this test prevents silent reintroduction.
    """
    identical = "Strong momentum, position 2% of NAV"
    a = {
        "recommendation": "BUY",
        "final_score": 7,
        "trader_note": identical,
        "risk_assessment": {
            "reasoning": identical,
            "recommended_position_pct": 0.02,
        },
    }
    sigs = extract_signals_from_analysis(a)
    risk = next((s for s in sigs if s["agent"] == "RiskJudge"), None)
    assert risk is not None, "RiskJudge entry must be present"
    assert "lite_path" not in risk, (
        f"RiskJudge must NOT carry lite_path key (got keys={list(risk.keys())}). "
        "25.B removed the byte-identical aliasing-detection branch; reintroducing it is a regression."
    )
    assert set(risk.keys()) == {"agent", "role", "rationale", "weight"}, (
        f"RiskJudge must have exactly the canonical 4 keys; got {set(risk.keys())}"
    )


def test_full_path_distinct_rationale():
    """phase-25.F: post-25.A full path -- RiskJudge `reasoning` is distinct
    from the Trader rationale and surfaces verbatim with weight from
    `recommended_position_pct`.
    """
    a = {
        "recommendation": "BUY",
        "final_score": 7,
        "trader_note": "Q1 beat, momentum +12% over 3m",
        "risk_assessment": {
            "reasoning": "Volatility within acceptable band; position cap 3% of NAV",
            "recommended_position_pct": 0.03,
        },
    }
    sigs = extract_signals_from_analysis(a)
    risk = next((s for s in sigs if s["agent"] == "RiskJudge"), None)
    assert risk is not None
    assert "Volatility within acceptable band" in risk["rationale"]
    assert "position cap 3% of NAV" in risk["rationale"]
    assert risk["weight"] == 0.03
    assert risk["role"] == "gate"
    assert "lite_path" not in risk


# ── Quant signals ───────────────────────────────────────────────

def test_extract_quant_signals_full_candidate():
    sigs = extract_quant_signals(_screener_candidate())
    quant = next(s for s in sigs if s["agent"] == "Quant")
    r = quant["rationale"]
    assert "1m momentum +4.2%" in r
    assert "3m momentum +11.8%" in r
    assert "6m momentum +24.0%" in r
    assert "RSI14 58.3" in r
    assert "ann_vol 0.28" in r
    assert "sector Technology" in r
    assert "composite_score 8.450" in r
    assert quant["weight"] == 0.845  # phase-25.D: composite_score=8.45 -> 0.845


def test_extract_quant_signals_empty_candidate():
    assert extract_quant_signals({}) == []


def test_extract_quant_signals_handles_missing_fields():
    sigs = extract_quant_signals({"ticker": "ABC", "momentum_1m": 5.0})
    quant = next(s for s in sigs if s["agent"] == "Quant")
    assert "1m momentum +5.0%" in quant["rationale"]
    # No SignalStack signal when no overlay fields present
    assert not any(s["agent"] == "SignalStack" for s in sigs)


def test_extract_quant_signals_non_dict_returns_empty():
    assert extract_quant_signals(None) == []
    assert extract_quant_signals("not a dict") == []


# ── SignalStack overlay ─────────────────────────────────────────

def test_signalstack_includes_all_overlays():
    sigs = extract_quant_signals(_screener_candidate())
    stack = next(s for s in sigs if s["agent"] == "SignalStack")
    r = stack["rationale"]
    assert "regime:risk_on" in r
    assert "pead:positive_surprise" in r
    assert "conviction 8.00" in r
    assert "strong momentum" in r
    assert "news:earnings_beat" in r
    assert "Q1 beat consensus" in r
    assert "sector_event:earnings" in r
    assert "source:news_only" in r
    assert stack["weight"] == 0.8  # phase-25.D: conviction_score=8 -> 0.8


def test_signalstack_only_with_partial_overlay_fields():
    cand = {"ticker": "X", "regime_tag": "risk_off", "conviction_score": 3}
    sigs = extract_quant_signals(cand)
    stack = next(s for s in sigs if s["agent"] == "SignalStack")
    assert "regime:risk_off" in stack["rationale"]
    assert "conviction 3.00" in stack["rationale"]


def test_signalstack_skipped_when_no_overlay_fields():
    cand = {"ticker": "X", "momentum_1m": 5.0, "composite_score": 5.0}
    sigs = extract_quant_signals(cand)
    # Quant signal yes, SignalStack no
    assert any(s["agent"] == "Quant" for s in sigs)
    assert not any(s["agent"] == "SignalStack" for s in sigs)


# ── extract_all_signals ordering ────────────────────────────────

def test_extract_all_signals_inserts_quant_before_trader():
    sigs = extract_all_signals(_lite_analysis(), candidate=_screener_candidate())
    agents = [s["agent"] for s in sigs]
    # Quant + SignalStack must come before Trader
    assert agents.index("Quant") < agents.index("Trader")
    assert agents.index("SignalStack") < agents.index("Trader")
    # Risk after Trader
    assert agents.index("Trader") < agents.index("RiskJudge")


def test_extract_all_signals_no_candidate_skips_quant():
    sigs = extract_all_signals(_lite_analysis(), candidate=None)
    assert not any(s["agent"] in ("Quant", "SignalStack") for s in sigs)
    # But Trader + Risk still present (from analysis extraction)
    assert any(s["agent"] == "Trader" for s in sigs)
    assert any(s["agent"] == "RiskJudge" for s in sigs)


def test_extract_all_signals_empty_candidate_skips_quant():
    sigs = extract_all_signals(_lite_analysis(), candidate={})
    assert not any(s["agent"] in ("Quant", "SignalStack") for s in sigs)


# ── group_signals_for_drawer routing ────────────────────────────

def test_group_signals_routes_quant_and_signal_stack():
    sigs = extract_all_signals(_lite_analysis(), candidate=_screener_candidate())
    tree = group_signals_for_drawer(sigs)
    assert "quant" in tree
    assert "signal_stack" in tree
    assert len(tree["quant"]) == 1
    assert len(tree["signal_stack"]) == 1
    assert tree["quant"][0]["agent"] == "Quant"
    assert tree["signal_stack"][0]["agent"] == "SignalStack"


def test_group_signals_preserves_existing_layers():
    sigs = extract_all_signals(_lite_analysis(), candidate=_screener_candidate())
    tree = group_signals_for_drawer(sigs)
    # Trader + Risk still routed to original buckets
    assert len(tree["trader"]) == 1
    assert len(tree["risk"]) == 1


def test_group_signals_empty_input():
    tree = group_signals_for_drawer([])
    assert tree == {
        "analyst": [],
        "debate": {"bull": [], "bear": []},
        "quant": [],
        "signal_stack": [],
        "trader": [],
        "risk": [],
    }


# ── End-to-end: drawer-ready JSON shape ─────────────────────────

def test_drawer_json_shape_matches_typescript_interface():
    sigs = extract_all_signals(_lite_analysis(), candidate=_screener_candidate())
    tree = group_signals_for_drawer(sigs)
    # TypeScript Rationale.tree expects these keys
    expected_keys = {"analyst", "debate", "quant", "signal_stack", "trader", "risk"}
    assert set(tree.keys()) == expected_keys
    # Each signal in each bucket has the {agent, role, rationale, weight} shape
    for bucket in ("analyst", "quant", "signal_stack", "trader", "risk"):
        for s in tree[bucket]:
            assert set(s.keys()) >= {"agent", "role", "rationale", "weight"}
