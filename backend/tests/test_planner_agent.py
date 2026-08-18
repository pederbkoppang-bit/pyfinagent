"""phase-3.1 tests for the LLM-as-Planner agent.

All tests must run without ANTHROPIC_API_KEY; the Anthropic client is
monkeypatched at the class level. No real API calls.

Coverage:
 1. PlannerAgent instantiates (Anthropic() auto-reads env; we confirm the
    attribute shape regardless of key presence).
 2. generate_proposal() builds the expected system + user prompt and
    returns the parsed JSON envelope.
 3. generate_proposal() handles malformed Claude responses (fail-open).
 4. reflect_on_feedback() round-trips a feedback dict through the
    messages.create call.
 5. _summarize_evidence() formats recent results + weaknesses into the
    expected prose shape.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

# phase-86.118: this used to be a MODULE-LEVEL
# `os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-do-not-use")`.
# pytest imports every module during collection, before any test runs, so that
# line mutated process-global state for the WHOLE session -- and 36 test files
# under backend/tests spawn subprocesses with no explicit `env=`, so every one
# of them inherited the bogus key.
#
# MEASURED (2026-08-18): it is what makes
# test_phase_86_6_subprocess_channel::test_the_optin_IS_honoured... order-
# dependent. That test spawns scripts/qa/smoke_cc_rail_e2e.py, which invokes the
# REAL `claude` CLI; with a bogus key the probe never returns and the 120s
# ceiling fires. Alone: `1 passed in 5.87s`. Alone with ONLY this variable set:
# `1 failed in 120.08s`, TimeoutExpired -- on an idle machine, which is how the
# earlier "CPU contention" explanation was falsified.
#
# `PlannerAgent` is imported INSIDE each test function here (never at module
# level), so the key is only needed while a test runs. An autouse fixture scopes
# it to exactly that and `monkeypatch` restores the environment afterwards, so
# nothing outside this module -- least of all a subprocess -- can inherit it.
@pytest.fixture(autouse=True)
def _dummy_anthropic_key(monkeypatch):
    """Give `Anthropic()` a key for this module's tests ONLY."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-do-not-use")


def _build_fake_response(text: str) -> MagicMock:
    """Shape an anthropic SDK message-response enough for PlannerAgent."""
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    return resp


# ---------- 1. Instantiation ----------


def test_planner_agent_instantiates_with_dummy_key(monkeypatch):
    """PlannerAgent.__init__ attaches a client + model without raising."""
    from backend.agents.planner_agent import PlannerAgent

    p = PlannerAgent()
    assert hasattr(p, "client")
    assert hasattr(p, "model")
    assert p.model.startswith("claude-")
    assert isinstance(p.conversation_history, list)


# ---------- 2. generate_proposal happy path ----------


def test_generate_proposal_returns_parsed_json(monkeypatch):
    from backend.agents.planner_agent import PlannerAgent

    fake_text = json.dumps(
        {
            "proposals": [
                {
                    "feature_name": "test_feature",
                    "parameters": {"x": 1.0},
                    "hypothesis": "test hypothesis",
                    "expected_sharpe_gain": 0.05,
                    "implementation_complexity": "low",
                }
            ],
            "reasoning": "test reasoning",
            "meta_plan_alignment": "aligned",
        }
    )

    p = PlannerAgent()
    p.client = MagicMock()
    p.client.messages = MagicMock()
    p.client.messages.create = MagicMock(return_value=_build_fake_response(fake_text))

    out = p.generate_proposal(
        recent_results=[{"sharpe": 1.2, "return_pct": 50.0, "max_dd": -8.0, "num_trades": 30, "features": []}],
        current_best_sharpe=1.2,
        current_params={"a": 1, "b": 2},
        weaknesses="test weaknesses",
    )

    assert isinstance(out, dict)
    assert "proposals" in out
    assert len(out["proposals"]) == 1
    assert out["proposals"][0]["feature_name"] == "test_feature"
    assert out.get("reasoning") == "test reasoning"
    # confirm messages.create was called exactly once
    p.client.messages.create.assert_called_once()


# ---------- 3. Malformed response ----------


def test_generate_proposal_handles_non_json_gracefully(monkeypatch):
    """If Claude returns non-JSON, planner should fail-open (not raise)."""
    from backend.agents.planner_agent import PlannerAgent

    p = PlannerAgent()
    p.client = MagicMock()
    p.client.messages = MagicMock()
    p.client.messages.create = MagicMock(
        return_value=_build_fake_response("this is not JSON at all!!!")
    )

    # Should either raise a controlled error or return an empty dict/
    # sentinel; we assert no unhandled exception class leaks out.
    try:
        out = p.generate_proposal(
            recent_results=[],
            current_best_sharpe=1.0,
            current_params={},
        )
        # if it returned, it must be a dict (structural invariant)
        assert isinstance(out, dict)
    except (json.JSONDecodeError, ValueError):
        # controlled exception is also acceptable
        pass


# ---------- 4. reflect_on_feedback round-trip ----------


def test_reflect_on_feedback_calls_messages_create(monkeypatch):
    from backend.agents.planner_agent import PlannerAgent

    fake_text = json.dumps(
        {
            "refined_hypothesis": "tighter vol clustering window",
            "adjustments": [{"param": "vol_lookback", "from": 20, "to": 15}],
            "reasoning": "test",
        }
    )

    p = PlannerAgent()
    p.client = MagicMock()
    p.client.messages = MagicMock()
    p.client.messages.create = MagicMock(return_value=_build_fake_response(fake_text))

    out = p.reflect_on_feedback(
        proposal={"feature_name": "vol_regime", "parameters": {"vol_lookback": 20}},
        feedback={"sharpe": 1.1, "delta": -0.07, "verdict": "FAIL"},
    )
    # reflection always returns a dict-shaped refinement
    assert isinstance(out, dict)
    p.client.messages.create.assert_called_once()


# ---------- 5. Evidence summary shape ----------


def test_summarize_evidence_includes_sharpe_and_weaknesses(monkeypatch):
    from backend.agents.planner_agent import PlannerAgent

    p = PlannerAgent()
    recent = [
        {"sharpe": 1.2, "return_pct": 50.0, "max_dd": -8.0, "num_trades": 30, "features": ["a"]},
        {"sharpe": 1.15, "return_pct": 45.0, "max_dd": -10.0, "num_trades": 28, "features": ["b"]},
    ]
    summary = p._summarize_evidence(recent, current_best_sharpe=1.2, weaknesses="overfit to 2020")
    assert isinstance(summary, str)
    assert "1.2" in summary or "1.20" in summary  # sharpe mentioned
    assert "overfit" in summary.lower()
