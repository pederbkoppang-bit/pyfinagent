"""phase-67.2: behavioral tests for parse_llm_classification's fallback path.

Pins the fix for the live NameError found by the phase-67 audit (2026-07-09):
agent_definitions.py's except tuple referenced `json.JSONDecodeError` while the
module never imported `json`, so ANY malformed Communication-agent routing
output raised `NameError: name 'json' is not defined` instead of degrading to
the designed Main-routed default (sole caller multi_agent_orchestrator.py
_classify_via_llm is un-wrapped -- the error propagated into the Slack/iMessage
classify path). The fix adds `import json` and broadens the tuple with
`AttributeError` (a bare JSON scalar/array passes json_io.loads then explodes
on `.get`).

Pure string->dataclass parsing under test. No mocks (an over-mocked test here
would trip the code-review skill's over-mocked-test BLOCK). All offline.
"""
from __future__ import annotations

from backend.agents.agent_definitions import (
    AgentType,
    QueryComplexity,
    parse_llm_classification,
)


# -- the documented repro: non-JSON garbage must degrade, not raise ---------
def test_malformed_non_json_defaults_to_main():
    result = parse_llm_classification("this is not json at all {broken")
    assert result.agent_type == AgentType.MAIN
    assert result.complexity == QueryComplexity.SIMPLE
    assert result.confidence == 0.5
    assert "Parse failed" in result.reasoning
    # a NameError raise (the pre-fix behavior) fails this test outright


# -- valid JSON, wrong shape: the AttributeError leg of the broadened tuple --
def test_valid_json_scalar_defaults_to_main():
    result = parse_llm_classification("5")
    assert result.agent_type == AgentType.MAIN
    assert "Parse failed" in result.reasoning


def test_valid_json_array_defaults_to_main():
    result = parse_llm_classification("[1, 2]")
    assert result.agent_type == AgentType.MAIN
    assert "Parse failed" in result.reasoning


# -- happy paths must survive the fix (mutation resistance) ------------------
def test_fenced_json_block_parses():
    fenced = (
        "```json\n"
        '{"primary": "qa", "secondary": null, "reasoning": "quant question",'
        ' "complexity": "complex", "triggers_harness": false}\n'
        "```"
    )
    result = parse_llm_classification(fenced)
    assert result.agent_type == AgentType.QA
    assert result.complexity == QueryComplexity.COMPLEX
    assert result.triggers_harness is False


def test_bare_valid_json_parses():
    raw = (
        '{"primary": "research", "secondary": null, "reasoning": "lit scan",'
        ' "complexity": "moderate", "triggers_harness": true}'
    )
    result = parse_llm_classification(raw)
    assert result.agent_type == AgentType.RESEARCH
    assert result.complexity == QueryComplexity.MODERATE
    assert result.triggers_harness is True
