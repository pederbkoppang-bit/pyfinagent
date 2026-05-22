"""phase-37.1 RiskJudge response_schema tests.

Fixes phase-34.2 cycle 3 observation: 8 of 10 RiskJudge invocations dropped
to raw-text fallback. Root cause per closure_roadmap §3:
  (a) orchestrator.py:107 _THINKING_RISK_JUDGE_CONFIG was missing
      response_mime_type + response_schema (cosmetic gap; live callsite
      uses _JUDGE_STRUCTURED_CONFIG in risk_debate.py which DID have schema)
  (b) the real bug: _generate_with_retry in risk_debate.py:62-72 + debate.py:65-72
      unconditionally injected include_thoughts=True when thinking_budget>0.
      Gemini 2.5+ structured-output (response_schema) is incompatible with
      include_thoughts=True -- the response includes thoughts pollution that
      breaks _parse_json downstream.

Tests cover the 4 immutable success criteria from masterplan 37.1:
  1. thinking_risk_judge_config_gains_response_mime_type_and_response_schema
  2. pydantic_RiskJudgeVerdict_model_defined_in_schemas_py
  3. live_cycle_post_change_shows_zero_risk_judge_invalid_json_warnings
     (deferred to live observation; this test verifies the code path is fixed)
  4. live_check_quotes_the_zero_warning_count (lives in live_check_37.1.md)

No real Gemini calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest


def test_phase_37_1_thinking_risk_judge_config_has_schema():
    """Criterion 1: _THINKING_RISK_JUDGE_CONFIG gains response_mime_type + response_schema."""
    from backend.agents.orchestrator import _THINKING_RISK_JUDGE_CONFIG
    from backend.agents.schemas import RiskJudgeVerdict
    assert _THINKING_RISK_JUDGE_CONFIG.get("response_mime_type") == "application/json"
    assert _THINKING_RISK_JUDGE_CONFIG.get("response_schema") is RiskJudgeVerdict


def test_phase_37_1_thinking_risk_judge_config_omits_include_thoughts():
    """Criterion 1 (corollary): include_thoughts must NOT be in the config alongside
    response_schema (Gemini 2.5+ incompatibility per closure_roadmap §3)."""
    from backend.agents.orchestrator import _THINKING_RISK_JUDGE_CONFIG
    assert "include_thoughts" not in _THINKING_RISK_JUDGE_CONFIG


def test_phase_37_1_risk_judge_verdict_schema_defined():
    """Criterion 2: RiskJudgeVerdict pydantic model is defined in schemas.py."""
    from backend.agents.schemas import RiskJudgeVerdict
    from pydantic import BaseModel
    assert issubclass(RiskJudgeVerdict, BaseModel)
    # Required fields per risk_debate.py:283-293 fallback dict shape
    fields = RiskJudgeVerdict.model_fields
    expected = {"decision", "risk_adjusted_confidence", "recommended_position_pct",
                "risk_level", "reasoning"}
    missing = expected - set(fields.keys())
    assert not missing, f"RiskJudgeVerdict missing required fields: {missing}"


def test_phase_37_1_judge_structured_config_has_schema():
    """The LIVE callsite (_JUDGE_STRUCTURED_CONFIG in risk_debate.py) has been
    schema-configured since phase-3; verify it still does."""
    from backend.agents.risk_debate import _JUDGE_STRUCTURED_CONFIG
    from backend.agents.schemas import RiskJudgeVerdict
    assert _JUDGE_STRUCTURED_CONFIG["response_mime_type"] == "application/json"
    assert _JUDGE_STRUCTURED_CONFIG["response_schema"] is RiskJudgeVerdict


def _mock_gemini_model(thinking_supported: bool = True):
    """Build a mock LLMClient that simulates a thinking-capable Gemini model."""
    model = MagicMock()
    model.supports_thinking = thinking_supported
    model.model_name = "gemini-2.5-pro"
    fake_response = MagicMock()
    fake_response.text = '{"decision": "APPROVE", "risk_adjusted_confidence": 0.75, "recommended_position_pct": 5, "risk_level": "LOW", "reasoning": "ok"}'
    fake_response.usage_metadata = MagicMock(prompt_token_count=100, candidates_token_count=50)
    model.generate_content.return_value = fake_response
    return model


def test_phase_37_1_generate_with_retry_omits_include_thoughts_when_schema_present():
    """The REAL fix: when response_schema is in the input config, the
    thinking-injection branch in risk_debate._generate_with_retry must NOT add
    include_thoughts=True (Gemini 2.5+ incompatibility)."""
    from backend.agents.risk_debate import _generate_with_retry, _JUDGE_STRUCTURED_CONFIG
    model = _mock_gemini_model(thinking_supported=True)

    # Invoke with thinking_budget=4096; expect include_thoughts NOT to land in config
    _generate_with_retry(
        model, "test prompt", "Risk Judge",
        max_retries=1,
        gen_config=_JUDGE_STRUCTURED_CONFIG,
        thinking_budget=4096,
    )

    # The generate_content call should have received a config WITH thinking but
    # WITHOUT include_thoughts (because response_schema was set in input).
    assert model.generate_content.called
    call_kwargs = model.generate_content.call_args.kwargs
    config_used = call_kwargs.get("generation_config") or model.generate_content.call_args.args[-1]
    assert "thinking" in config_used, "thinking should be injected when budget>0"
    assert "include_thoughts" not in config_used, \
        "include_thoughts must NOT be injected when response_schema is set (Gemini 2.5+ incompat)"
    assert "response_schema" in config_used, "input response_schema preserved"


def test_phase_37_1_generate_with_retry_still_adds_thoughts_when_no_schema():
    """Backward-compat: when response_schema is NOT in the input config (e.g.
    free-form text agents), include_thoughts=True IS injected as before."""
    from backend.agents.risk_debate import _generate_with_retry, _RISK_GEN_CONFIG
    model = _mock_gemini_model(thinking_supported=True)

    # _RISK_GEN_CONFIG has NO response_schema -- thoughts should land
    _generate_with_retry(
        model, "test prompt", "Aggressive R1",
        max_retries=1,
        gen_config=_RISK_GEN_CONFIG,
        thinking_budget=4096,
    )

    call_kwargs = model.generate_content.call_args.kwargs
    config_used = call_kwargs.get("generation_config") or model.generate_content.call_args.args[-1]
    assert "thinking" in config_used
    assert config_used.get("include_thoughts") is True, \
        "include_thoughts SHOULD be added when no response_schema is set"


def test_phase_37_1_debate_generate_with_retry_same_guard():
    """Same fix applied in debate.py:_generate_with_retry. Verify the structured
    Moderator config also gets the include_thoughts guard."""
    from backend.agents.debate import _generate_with_retry as debate_generate
    from backend.agents.debate import _MODERATOR_STRUCTURED_CONFIG, GeminiClient

    if "response_schema" not in _MODERATOR_STRUCTURED_CONFIG:
        pytest.skip("Moderator config does not use response_schema; phase-37.4 territory")

    model = MagicMock(spec=GeminiClient)
    model.model_name = "gemini-2.5-pro"
    fake = MagicMock()
    fake.text = '{}'
    fake.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=5)
    model.generate_content.return_value = fake

    debate_generate(
        model, "test", "Moderator",
        max_retries=1, gen_config=_MODERATOR_STRUCTURED_CONFIG,
        thinking_budget=4096,
    )

    call_kwargs = model.generate_content.call_args.kwargs
    config_used = call_kwargs.get("generation_config") or model.generate_content.call_args.args[-1]
    assert "include_thoughts" not in config_used, \
        "debate._generate_with_retry must also omit include_thoughts under response_schema"
