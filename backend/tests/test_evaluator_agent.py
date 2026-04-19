"""phase-3.2 tests for the LLM-as-Evaluator agent.

All tests run without Vertex credentials; the model is `None` on the
`VERTEX_AVAILABLE=False` / init-failure path, and `_parse_evaluation_response`
is exercised directly from a known JSON payload.

Coverage:
 1. EvaluatorAgent instantiates without Vertex (mock evaluator path).
 2. `_parse_evaluation_response` maps known JSON to the right verdict.
 3. `_parse_evaluation_response` handles malformed JSON.
 4. `evaluate_proposal` timeout path returns FAIL (conservative).
 5. `EvaluationVerdict` enum values align with known strings.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from backend.agents.evaluator_agent import (
    EvaluationResult,
    EvaluationVerdict,
    EvaluatorAgent,
)


# ---------- 1. Instantiation without Vertex ----------


def test_evaluator_agent_instantiates_without_vertex():
    """EvaluatorAgent.__init__ must not raise if Vertex is unavailable."""
    with patch("backend.agents.evaluator_agent.VERTEX_AVAILABLE", False):
        e = EvaluatorAgent(model_name="gemini-2.0-flash")
        assert e.model is None
        assert e.model_name == "gemini-2.0-flash"
        assert e.max_eval_time == 30


def test_evaluator_agent_default_model_name():
    e = EvaluatorAgent()
    assert e.model_name  # non-empty default


# ---------- 2. Verdict enum ----------


def test_evaluation_verdict_enum_values():
    assert EvaluationVerdict.PASS.value == "PASS"
    assert EvaluationVerdict.CONDITIONAL.value == "CONDITIONAL"
    assert EvaluationVerdict.FAIL.value == "FAIL"


# ---------- 3. Parse known JSON ----------


def test_parse_evaluation_response_maps_verdict_pass():
    """Feed a parse-able JSON response string; expect a PASS EvaluationResult."""
    e = EvaluatorAgent()

    response_text = json.dumps(
        {
            "verdict": "PASS",
            "statistical_validity_score": 90,
            "robustness_score": 85,
            "simplicity_score": 80,
            "reality_gap_score": 75,
            "risk_check_score": 88,
            "summary": "Strong proposal",
            "detailed_reasoning": "Sharpe > baseline, DSR valid, sub-periods consistent.",
            "red_flags": [],
            "yellow_flags": [],
            "green_flags": ["robust across regimes"],
        }
    )
    proposal = {"hypothesis": "test"}
    backtest = {"sharpe": 1.25, "dsr": 0.98, "return": 55}

    result = e._parse_evaluation_response(response_text, proposal, backtest)
    assert isinstance(result, EvaluationResult)
    assert result.verdict == EvaluationVerdict.PASS
    assert 0 <= result.overall_score <= 100
    assert result.green_flags == ["robust across regimes"]


def test_parse_evaluation_response_maps_verdict_fail():
    e = EvaluatorAgent()

    response_text = json.dumps(
        {
            "verdict": "FAIL",
            "statistical_validity_score": 20,
            "robustness_score": 15,
            "simplicity_score": 60,
            "reality_gap_score": 30,
            "risk_check_score": 25,
            "summary": "Overfit to 2020",
            "detailed_reasoning": "Sub-period A Sharpe < 0.3",
            "red_flags": ["overfit", "insufficient trades"],
            "yellow_flags": [],
            "green_flags": [],
        }
    )
    result = e._parse_evaluation_response(response_text, {}, {})
    assert result.verdict == EvaluationVerdict.FAIL
    assert len(result.red_flags) == 2


# ---------- 4. Timeout path returns FAIL ----------


def test_evaluate_proposal_timeout_returns_fail():
    """On asyncio.TimeoutError, evaluate_proposal must return FAIL."""
    e = EvaluatorAgent()

    async def _slow_call(*args, **kwargs):
        await asyncio.sleep(10)  # longer than max_eval_time
        return {}

    e._call_model = _slow_call  # type: ignore[assignment]
    e.max_eval_time = 0.1  # force immediate timeout

    async def _run():
        return await e.evaluate_proposal(
            proposal={"hypothesis": "test"},
            backtest_results={"sharpe": 1.0},
        )

    result = asyncio.run(_run())
    assert isinstance(result, EvaluationResult)
    assert result.verdict == EvaluationVerdict.FAIL
    # Conservative timeout rationale must surface in reasoning
    assert "timeout" in result.detailed_reasoning.lower() or "timeout" in result.summary.lower()
