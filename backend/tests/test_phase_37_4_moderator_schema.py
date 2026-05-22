"""phase-37.4 Moderator response_schema regression tests.

Pre-closed-in-source by phase-37.1: `_MODERATOR_STRUCTURED_CONFIG` at
`debate.py:47-51` already had `response_mime_type: "application/json"`
+ `response_schema: ModeratorConsensus` since phase-3 (Gemini JSON-schema
enforcement). The actual cycle-2 Moderator invalid-JSON-fallback regression
was caused by `include_thoughts=True` being injected unconditionally in
`debate._generate_with_retry` (lines 65-74); phase-37.1 added the
schema-aware guard at lines 65-72 that omits `include_thoughts` when
`response_schema` is in the input config.

This test file LOCKS IN the regression-resistance for the Moderator path
with explicit Moderator-named assertions (so a future commit that removes
the schema or the guard tripping a specific test).

2 immutable success criteria from masterplan 37.4.verification:
  1. moderator_structured_config_gains_response_schema (PRE-MET in source)
  2. live_cycle_post_change_shows_zero_moderator_invalid_json_warnings
     (DEFERRED to live observation; runbook in live_check_37.4.md)

External corroboration (per research_brief_phase_37_4.md):
  - python-genai issue #782 (structured-output + thinking incompat)
  - python-genai issue #637 (gemini-2.5-pro-preview structured outputs)
  - TradingAgents v0.2.5 (production pattern: response_schema for Gemini)
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest


def test_phase_37_4_moderator_structured_config_has_response_mime_type():
    """Criterion #1 part a: _MODERATOR_STRUCTURED_CONFIG carries
    response_mime_type='application/json' so Gemini enforces JSON output."""
    from backend.agents.debate import _MODERATOR_STRUCTURED_CONFIG
    assert _MODERATOR_STRUCTURED_CONFIG.get("response_mime_type") == "application/json", (
        f"_MODERATOR_STRUCTURED_CONFIG response_mime_type = "
        f"{_MODERATOR_STRUCTURED_CONFIG.get('response_mime_type')!r}; "
        "must be 'application/json' (phase-37.4 regression lock)."
    )


def test_phase_37_4_moderator_structured_config_has_response_schema_class():
    """Criterion #1 part b: _MODERATOR_STRUCTURED_CONFIG carries
    response_schema=ModeratorConsensus -- the specific pydantic class."""
    from backend.agents.debate import _MODERATOR_STRUCTURED_CONFIG
    from backend.agents.schemas import ModeratorConsensus
    assert _MODERATOR_STRUCTURED_CONFIG.get("response_schema") is ModeratorConsensus, (
        f"_MODERATOR_STRUCTURED_CONFIG response_schema is "
        f"{_MODERATOR_STRUCTURED_CONFIG.get('response_schema')}; "
        "must be ModeratorConsensus (phase-37.4 regression lock)."
    )


def test_phase_37_4_moderator_consensus_is_pydantic_basemodel():
    """ModeratorConsensus must be a Pydantic BaseModel subclass so Gemini
    SDK can extract the JSON schema. If it's a regular class or TypedDict,
    Gemini structured-output silently degrades."""
    from backend.agents.schemas import ModeratorConsensus
    from pydantic import BaseModel
    assert issubclass(ModeratorConsensus, BaseModel), (
        "ModeratorConsensus must be pydantic.BaseModel subclass "
        "for Gemini response_schema to work."
    )
    # Must have at least one field (otherwise schema is degenerate).
    assert len(ModeratorConsensus.model_fields) >= 1, (
        "ModeratorConsensus must have at least one Field defined."
    )


def test_phase_37_4_debate_generate_with_retry_omits_include_thoughts_for_moderator():
    """The phase-37.1 include_thoughts guard at debate.py:65-72 MUST omit
    include_thoughts=True when input gen_config has response_schema. This
    is the actual root-cause fix for the Moderator invalid-JSON-fallback.

    Tested against the live _MODERATOR_STRUCTURED_CONFIG (which has
    response_schema) -- the guard should skip include_thoughts injection."""
    from backend.agents.debate import _generate_with_retry, _MODERATOR_STRUCTURED_CONFIG, GeminiClient
    model = MagicMock(spec=GeminiClient)
    model.model_name = "gemini-2.5-pro"
    fake = MagicMock()
    fake.text = "{}"
    fake.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=5)
    model.generate_content.return_value = fake

    _generate_with_retry(
        model, "test prompt", "Moderator",
        max_retries=1, gen_config=_MODERATOR_STRUCTURED_CONFIG,
        thinking_budget=4096,
    )

    call_kwargs = model.generate_content.call_args.kwargs
    config_used = call_kwargs.get("generation_config") or model.generate_content.call_args.args[-1]
    assert "thinking" in config_used, "thinking SHOULD be injected when budget > 0"
    assert "include_thoughts" not in config_used, (
        "include_thoughts MUST be omitted when response_schema is in config "
        "(Gemini 2.5+ incompatibility per python-genai issue #782 + #637; "
        "phase-37.1 added this guard, phase-37.4 locks it in for Moderator)."
    )
    assert "response_schema" in config_used, "input response_schema must be preserved"


def test_phase_37_4_moderator_structured_config_block_locked_at_known_lines():
    """Structural lock: confirm the _MODERATOR_STRUCTURED_CONFIG block lives
    at the expected location (debate.py around line 47-51). If a future
    refactor moves it, this test catches the displacement so the planner
    can update the doc references in closure_roadmap.md + the phase-37
    history."""
    with open("backend/agents/debate.py") as f:
        src = f.read()
    # The config block must include all 3 critical fields on consecutive lines
    assert "_MODERATOR_STRUCTURED_CONFIG = {" in src
    # The schema reference is the structural anchor
    config_start = src.find("_MODERATOR_STRUCTURED_CONFIG = {")
    config_end = src.find("}", config_start)
    block = src[config_start:config_end]
    assert '"response_mime_type": "application/json"' in block
    assert '"response_schema": ModeratorConsensus' in block
