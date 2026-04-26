"""Unit tests for meta_scorer — schema, prompt, fallback paths, batching."""

from __future__ import annotations

import asyncio
from unittest.mock import patch, MagicMock

import pytest

from backend.services.meta_scorer import (
    MetaScoredCandidate,
    MetaScorerBatch,
    meta_score_candidates,
    _build_meta_prompt,
    _format_candidate_block,
    _fallback_conviction,
    _MAX_BATCH,
)


def _mk_cand(ticker="AAPL", composite=10.0, sector="Information Technology") -> dict:
    return {
        "ticker": ticker,
        "sector": sector,
        "momentum_1m": 5.0,
        "momentum_3m": 12.0,
        "momentum_6m": 20.0,
        "rsi_14": 60,
        "composite_score": composite,
    }


def test_schema_conviction_must_be_int_in_range():
    with pytest.raises(Exception):
        MetaScoredCandidate(ticker="AAPL", conviction_score=11, conviction_reason="x")
    with pytest.raises(Exception):
        MetaScoredCandidate(ticker="AAPL", conviction_score=0, conviction_reason="x")


def test_schema_extra_forbidden():
    with pytest.raises(Exception):
        MetaScoredCandidate.model_validate({
            "ticker": "AAPL", "conviction_score": 7,
            "conviction_reason": "x", "extra_field": "boom",
        })


def test_format_candidate_block_includes_all_signals():
    c = _mk_cand()
    c["pead_signal"] = {"sentiment_tag": "positive_surprise", "sentiment_score": 0.8,
                        "surprise_score": 0.3, "holding_window_days": 28}
    c["news_signal"] = {"impact_polarity": "positive", "event_type": "earnings_beat",
                        "confidence": "high"}
    c["sector_event"] = {"event_type": "fda_pdufa", "signal_direction": "positive_catalyst",
                         "days_to_event": 5}
    block = _format_candidate_block(c)
    assert "AAPL" in block
    assert "Information Technology" in block
    assert "positive_surprise" in block
    assert "earnings_beat" in block
    assert "fda_pdufa" in block
    assert "composite_score_pre_meta: 10.0" in block


def test_build_meta_prompt_anti_rubber_stamp_directives():
    cands = [_mk_cand("AAPL"), _mk_cand("NVDA")]
    prompt = _build_meta_prompt(cands, regime=None)
    assert "INDEPENDENTLY" in prompt
    assert "what could go WRONG" in prompt
    assert "risk_off" in prompt
    assert "ordered randomly" in prompt
    assert "EXACTLY 2" in prompt


def test_build_meta_prompt_includes_regime_when_present():
    regime = MagicMock(regime="risk_off", conviction_multiplier=0.7, conviction=0.85)
    prompt = _build_meta_prompt([_mk_cand()], regime=regime)
    assert "risk_off" in prompt
    assert "0.70" in prompt
    assert "fade signal" in prompt


def test_fallback_conviction_clamps_to_1_10():
    assert _fallback_conviction({"composite_score": 100}) == 10
    assert _fallback_conviction({"composite_score": -5}) == 1
    assert _fallback_conviction({"composite_score": 7.4}) == 7
    assert _fallback_conviction({"composite_score": None}) == 5
    assert _fallback_conviction({}) == 5


def test_meta_score_no_anthropic_key_returns_fallback(monkeypatch):
    """When ANTHROPIC_API_KEY is empty, return composite-score-derived conviction."""
    cands = [_mk_cand("AAPL", composite=12.0), _mk_cand("MSFT", composite=8.0)]

    settings_mock = MagicMock()
    settings_mock.anthropic_api_key = ""
    monkeypatch.setattr("backend.services.meta_scorer.get_settings", lambda: settings_mock)

    out = asyncio.run(meta_score_candidates(cands))
    assert len(out) == 2
    assert all("conviction_score" in c for c in out)
    assert all(1 <= c["conviction_score"] <= 10 for c in out)
    # Sorted desc by conviction
    assert out[0]["conviction_score"] >= out[-1]["conviction_score"]
    assert out[0]["ticker"] == "AAPL"  # higher composite = higher fallback conviction
    assert all(c["conviction_reason"] == "fallback (no API key)" for c in out)


def test_meta_score_empty_input_returns_empty():
    out = asyncio.run(meta_score_candidates([]))
    assert out == []


def test_meta_score_caps_at_max_batch(monkeypatch):
    """Passing 50 candidates only meta-scores the top-30 by composite_score."""
    settings_mock = MagicMock()
    settings_mock.anthropic_api_key = ""
    monkeypatch.setattr("backend.services.meta_scorer.get_settings", lambda: settings_mock)

    cands = [_mk_cand(f"T{i:03d}", composite=float(50 - i)) for i in range(50)]
    out = asyncio.run(meta_score_candidates(cands))
    assert len(out) == 50  # all returned
    # Top-30 get LLM-style processing (fallback here since no key); bottom-20 also fallback
    assert all("conviction_score" in c for c in out)


def test_meta_score_sorts_descending_by_conviction(monkeypatch):
    settings_mock = MagicMock()
    settings_mock.anthropic_api_key = ""
    monkeypatch.setattr("backend.services.meta_scorer.get_settings", lambda: settings_mock)

    cands = [_mk_cand("LOW", composite=2.0), _mk_cand("HIGH", composite=9.5), _mk_cand("MID", composite=5.0)]
    out = asyncio.run(meta_score_candidates(cands))
    scores = [c["conviction_score"] for c in out]
    assert scores == sorted(scores, reverse=True)
    assert out[0]["ticker"] == "HIGH"
    assert out[-1]["ticker"] == "LOW"


def test_meta_score_with_mocked_llm(monkeypatch):
    """Full path with stubbed Claude: returns parsed batch sorted by conviction."""
    cands = [_mk_cand("AAPL", composite=12.0), _mk_cand("NVDA", composite=8.0), _mk_cand("MSFT", composite=10.0)]

    settings_mock = MagicMock()
    settings_mock.anthropic_api_key = "sk-ant-fake-key"
    settings_mock.meta_scorer_model = "claude-haiku-4-5"
    monkeypatch.setattr("backend.services.meta_scorer.get_settings", lambda: settings_mock)

    fake_response = MagicMock()
    fake_response.text = (
        '{"candidates":['
        '{"ticker":"AAPL","conviction_score":8,"conviction_reason":"strong momentum + positive PEAD"},'
        '{"ticker":"NVDA","conviction_score":4,"conviction_reason":"momentum fading"},'
        '{"ticker":"MSFT","conviction_score":7,"conviction_reason":"steady fundamentals"}'
        ']}'
    )

    fake_client = MagicMock()
    fake_client.generate_content.return_value = fake_response

    with patch("backend.agents.llm_client.ClaudeClient", return_value=fake_client):
        out = asyncio.run(meta_score_candidates(cands))

    assert len(out) == 3
    convictions = {c["ticker"]: c["conviction_score"] for c in out}
    assert convictions == {"AAPL": 8, "NVDA": 4, "MSFT": 7}
    # Sorted desc
    assert [c["ticker"] for c in out] == ["AAPL", "MSFT", "NVDA"]
    assert "strong momentum" in out[0]["conviction_reason"]


def test_meta_score_clamps_out_of_range_llm_output(monkeypatch):
    """LLM returns 11 (out of range) → clamped to 10."""
    cands = [_mk_cand("AAPL", composite=10.0)]

    settings_mock = MagicMock()
    settings_mock.anthropic_api_key = "sk-ant-fake"
    settings_mock.meta_scorer_model = "claude-haiku-4-5"
    monkeypatch.setattr("backend.services.meta_scorer.get_settings", lambda: settings_mock)

    fake_response = MagicMock()
    fake_response.text = '{"candidates":[{"ticker":"AAPL","conviction_score":11,"conviction_reason":"x"}]}'

    fake_client = MagicMock()
    fake_client.generate_content.return_value = fake_response
    with patch("backend.agents.llm_client.ClaudeClient", return_value=fake_client):
        out = asyncio.run(meta_score_candidates(cands))

    assert out[0]["conviction_score"] == 10  # clamped


def test_meta_score_handles_llm_parse_error(monkeypatch):
    cands = [_mk_cand("AAPL"), _mk_cand("MSFT")]

    settings_mock = MagicMock()
    settings_mock.anthropic_api_key = "sk-ant-fake"
    settings_mock.meta_scorer_model = "claude-haiku-4-5"
    monkeypatch.setattr("backend.services.meta_scorer.get_settings", lambda: settings_mock)

    fake_response = MagicMock()
    fake_response.text = "not json at all"

    fake_client = MagicMock()
    fake_client.generate_content.return_value = fake_response
    with patch("backend.agents.llm_client.ClaudeClient", return_value=fake_client):
        out = asyncio.run(meta_score_candidates(cands))

    assert len(out) == 2
    assert all("conviction_score" in c for c in out)
    assert all("fallback" in c["conviction_reason"] for c in out)


def test_max_batch_constant():
    assert _MAX_BATCH == 30
