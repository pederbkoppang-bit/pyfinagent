"""phase-71.2 -- Layer-2 honesty: guaranteed structured outputs + kill silent-failure classes.

Covers the four immutable criteria:
  C1 -- the two Claude JSON sites (quality gate + classifier) use constrained-decoding
        structured output (output_config.format json_schema), with a fail-safe fallback.
  C2 -- the quality-gate CLOBBER bug is fixed: an unparseable gate response preserves the
        ORIGINAL analyst answer (returns None), never substitutes the raw gate text.
  C3 -- the fabricated spot-check stub (evaluate_with_spot_checks + _run_spot_checks) is
        DELETED; the evaluator can no longer flip CONDITIONAL->PASS on hardcoded numbers.
  C4 -- no risk-threshold VALUE changed: LOOSE_DSR_MIN is still 0.95 (relocated, not moved).
"""
import ast
import asyncio
import json
import os
from unittest.mock import MagicMock

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_SRC = os.path.join(REPO, "backend", "agents", "evaluator_agent.py")


def _make_orch():
    from backend.agents.multi_agent_orchestrator import (
        MultiAgentOrchestrator,
        reset_anthropic_client,
    )
    reset_anthropic_client()
    return MultiAgentOrchestrator()


def _classification():
    c = MagicMock()
    c.agent_type.value = "main"
    return c


# ---- C2: clobber fix -------------------------------------------------------

def test_c2_unparseable_gate_response_preserves_original_answer():
    """RED->GREEN: an unparseable gate response must return None (keep original),
    NEVER the raw gate text. Pre-fix this returned `gate_response` (the clobber)."""
    orch = _make_orch()
    garbage = "totally unparseable gate output with no scores and no verdict"
    orch._call_agent_json = MagicMock(return_value=(garbage, {"input": 1, "output": 1}))

    improved, _usage = asyncio.run(
        orch._quality_gate("q?", "original analyst answer", _classification())
    )
    assert improved is None, "clobber bug: unparseable gate must keep the original answer"
    assert improved != garbage, "must never substitute the raw gate text as the answer"


# ---- C1: structured path decision semantics preserved ----------------------

def test_c1_structured_pass_keeps_original():
    orch = _make_orch()
    payload = json.dumps({
        "accuracy": 0.9, "completeness": 0.9, "groundedness": 0.9,
        "conciseness": 0.9, "verdict": "PASS", "improved_response": "",
    })
    orch._call_agent_json = MagicMock(return_value=(payload, {"input": 1, "output": 1}))
    improved, _ = asyncio.run(orch._quality_gate("q?", "orig", _classification()))
    assert improved is None  # PASS -> keep original


def test_c1_structured_fail_returns_improved():
    orch = _make_orch()
    payload = json.dumps({
        "accuracy": 0.2, "completeness": 0.3, "groundedness": 0.1,
        "conciseness": 0.9, "verdict": "FAIL", "improved_response": "A much better, grounded answer.",
    })
    orch._call_agent_json = MagicMock(return_value=(payload, {"input": 1, "output": 1}))
    improved, _ = asyncio.run(orch._quality_gate("q?", "orig", _classification()))
    assert improved == "A much better, grounded answer."


def test_c1_structured_fenced_json_still_parses():
    """Gemini-fallback (Anthropic down) may wrap the JSON in ```json fences.
    Fence-stripping keeps the gate's answer-improve ability on that path too."""
    orch = _make_orch()
    body = json.dumps({
        "accuracy": 0.2, "completeness": 0.3, "groundedness": 0.1,
        "conciseness": 0.9, "verdict": "FAIL", "improved_response": "Fenced but better.",
    })
    payload = "```json\n" + body + "\n```"
    orch._call_agent_json = MagicMock(return_value=(payload, {"input": 1, "output": 1}))
    improved, _ = asyncio.run(orch._quality_gate("q?", "orig", _classification()))
    assert improved == "Fenced but better."


def test_c1_structured_fail_no_improvement_keeps_original():
    orch = _make_orch()
    payload = json.dumps({
        "accuracy": 0.2, "completeness": 0.3, "groundedness": 0.1,
        "conciseness": 0.9, "verdict": "FAIL", "improved_response": "",
    })
    orch._call_agent_json = MagicMock(return_value=(payload, {"input": 1, "output": 1}))
    improved, _ = asyncio.run(orch._quality_gate("q?", "orig", _classification()))
    assert improved is None  # FAIL but no replacement -> keep original (fail-safe)


# ---- C1: schemas + helper present + subset-compliant -----------------------

def test_c1_schemas_present_and_subset_compliant():
    import backend.agents.multi_agent_orchestrator as m
    for schema in (m.QUALITY_VERDICT_SCHEMA, m.CLASSIFY_SCHEMA):
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        # strict subset: every property is required
        assert set(schema["required"]) == set(schema["properties"].keys())
    assert hasattr(m.MultiAgentOrchestrator, "_call_agent_json")


def test_c1_call_agent_json_fail_safe_falls_back_to_plain_text():
    """A non-auth error in the structured call must degrade to the plain _call_agent
    text path -- never break the live gate/classifier."""
    orch = _make_orch()
    orch._anthropic_unavailable = False
    bad_client = MagicMock()
    bad_client.messages.create.side_effect = TypeError("output_config unsupported by SDK")
    orch._get_client = MagicMock(return_value=bad_client)
    orch._call_agent = MagicMock(return_value=("FELLBACK_TEXT", {"input": 0, "output": 0}))

    text, _ = orch._call_agent_json(
        MagicMock(model="claude-sonnet-4-6", max_tokens=100, system_prompt="s", name="QG"),
        "task", {"type": "object"},
    )
    assert text == "FELLBACK_TEXT"
    orch._call_agent.assert_called_once()


# ---- C3: fabricated spot-check deleted -------------------------------------

def test_c3_spot_check_methods_deleted():
    from backend.agents.evaluator_agent import EvaluatorAgent
    assert not hasattr(EvaluatorAgent, "_run_spot_checks")
    assert not hasattr(EvaluatorAgent, "evaluate_with_spot_checks")


def test_c3_no_fabricated_literals_in_source():
    src = open(EVAL_SRC).read()
    for lit in ("1.02", "0.95", "0.99"):
        assert lit not in src, f"fabricated/hardcoded literal {lit!r} still present in evaluator_agent.py"


def test_c3_source_parses():
    ast.parse(open(EVAL_SRC).read())


# ---- C4: no risk-threshold VALUE change ------------------------------------

def test_c4_dsr_threshold_value_unchanged():
    from backend.agents.evaluator_agent import LOOSE_DSR_MIN
    assert LOOSE_DSR_MIN == 0.95, "DSR promotion threshold VALUE must be byte-identical (relocated, not moved)"
