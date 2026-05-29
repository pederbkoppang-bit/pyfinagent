"""phase-47.10: guard for the generate_content max_tokens floor.

Symmetric close of the Opus-4.8 max_tokens-at-xhigh audit. 47.9 floored the
ALWAYS-adaptive orchestrator Opus path; this floors the SECOND Opus path,
`llm_client.generate_content`, which shares `max_tokens` (default 2048) with
adaptive thinking + xhigh effort for Opus-4.8/4.7 with no floor.

Reachability is operator-override-only (ENABLE_THINKING=true AND
DEEP_THINK_MODEL=claude-opus-4-8, both non-default) -- a defensive symmetry fix.

The floor must gate on `thinking_requested` (NOT effort): per Anthropic's effort
doc, "without [the adaptive thinking arg], requests run without thinking", so
effort alone creates no hidden thinking tokens sharing the ceiling.

These guards fail if the floor regresses, fires when it shouldn't (thinking off
or non-Opus), or lowers a caller's larger budget.
"""
from __future__ import annotations


def test_floor_value_matches_orchestrator():
    from backend.agents.llm_client import _OPUS_ADAPTIVE_MIN_MAX_TOKENS

    assert _OPUS_ADAPTIVE_MIN_MAX_TOKENS == 16384
    # symmetric with the orchestrator twin from 47.9
    from backend.agents.multi_agent_orchestrator import (
        _OPUS_ADAPTIVE_MIN_MAX_TOKENS as ORCH_FLOOR,
    )
    assert _OPUS_ADAPTIVE_MIN_MAX_TOKENS == ORCH_FLOOR


def test_floors_opus_with_thinking():
    from backend.agents.llm_client import _opus_adaptive_max_tokens

    # the starvation case: thinking on, Opus, small budget -> floored
    assert _opus_adaptive_max_tokens(2048, "claude-opus-4-8", True) == 16384
    assert _opus_adaptive_max_tokens(1024, "claude-opus-4-7", True) == 16384


def test_noop_when_thinking_off():
    from backend.agents.llm_client import _opus_adaptive_max_tokens

    # thinking off -> max_tokens is pure output budget; effort alone is NOT
    # floored (Anthropic effort doc) -> no-op
    assert _opus_adaptive_max_tokens(2048, "claude-opus-4-8", False) == 2048


def test_noop_when_not_opus():
    from backend.agents.llm_client import _opus_adaptive_max_tokens

    # non-Opus models are not on the adaptive-only path the floor targets
    assert _opus_adaptive_max_tokens(2048, "claude-sonnet-4-6", True) == 2048
    assert _opus_adaptive_max_tokens(2048, "claude-haiku-4-5", True) == 2048
    assert _opus_adaptive_max_tokens(2048, "gemini-2.5-pro", True) == 2048
    assert _opus_adaptive_max_tokens(2048, "", True) == 2048


def test_respects_larger_caller_budget():
    from backend.agents.llm_client import _opus_adaptive_max_tokens

    # the floor is a max() -- it never lowers a caller's higher budget
    assert _opus_adaptive_max_tokens(30000, "claude-opus-4-8", True) == 30000
    assert _opus_adaptive_max_tokens(16384, "claude-opus-4-8", True) == 16384  # boundary


def test_generate_content_applies_the_floor_at_source():
    """The generate_content body must route max_tokens through the helper after
    the effort block (structural guard against silently dropping the call)."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "backend/agents/llm_client.py").read_text(
        encoding="utf-8"
    )
    assert "kwargs[\"max_tokens\"] = _opus_adaptive_max_tokens(" in src
    assert "kwargs[\"max_tokens\"], model_id, thinking_requested" in src
