"""phase-16.36 (#45) regression tests for the Anthropic AuthError fallback path.

Verifies that:
1. AuthenticationError on `_call_agent` triggers `_gemini_text_call` fallback
2. AuthenticationError on `_call_agent_with_tools` triggers fallback
3. `_anthropic_unavailable` flag persists across calls after first 401
4. `reset_anthropic_client()` clears the flag + client + settings cache
5. Gemini fallback returns non-zero token usage (the #46 fix)

Approach: patch `anthropic.AuthenticationError = Exception` so any raised
Exception passes the `isinstance(e, anthropic.AuthenticationError)` check
in the orchestrator. Real anthropic.AuthenticationError requires
httpx.Response in its constructor; sys.modules patching is the canonical
pytest pattern for this case (see Anthropic SDK 0.79.0 source).

No live network calls. No real Anthropic key needed.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---- Fixtures ---------------------------------------------------------

@pytest.fixture
def fake_anthropic(monkeypatch):
    """Replace `anthropic.AuthenticationError` with plain Exception so any
    raised Exception trips the orchestrator's typed catch. Restored on
    fixture teardown via monkeypatch."""
    fake_mod = types.ModuleType("anthropic")
    fake_mod.AuthenticationError = Exception  # type: ignore[attr-defined]

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.messages = MagicMock()

    fake_mod.Anthropic = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)
    yield fake_mod


@pytest.fixture
def fake_llm_response():
    """Construct an LLMResponse with non-zero usage_metadata (mimics
    GeminiClient.generate_content return value). Used to verify #46
    token-usage extraction."""
    from backend.agents.llm_client import LLMResponse, UsageMeta

    return LLMResponse(
        text="Mocked Gemini fallback response.",
        thoughts="",
        usage_metadata=UsageMeta(
            prompt_token_count=137,
            candidates_token_count=42,
        ),
    )


@pytest.fixture
def orchestrator_with_mocks(fake_anthropic, fake_llm_response):
    """Fresh orchestrator instance + mocked Gemini client.

    Returns (orchestrator, mock_anthropic_client, mock_gemini_client).
    Resets module-level singleton via reset_anthropic_client() before/after
    so test isolation is guaranteed.
    """
    from backend.agents.multi_agent_orchestrator import (
        MultiAgentOrchestrator,
        reset_anthropic_client,
    )

    reset_anthropic_client()
    orch = MultiAgentOrchestrator()

    # Mock Anthropic client: messages.create raises immediately
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.side_effect = Exception("mock 401 invalid api key")
    orch._client = mock_anthropic

    # Mock Gemini client: generate_content returns fake LLMResponse
    mock_gemini = MagicMock()
    mock_gemini.generate_content.return_value = fake_llm_response
    orch._gemini_mas_client = mock_gemini

    yield orch, mock_anthropic, mock_gemini

    reset_anthropic_client()


# ---- Tests ------------------------------------------------------------

def test_call_agent_auth_error_triggers_gemini_fallback(orchestrator_with_mocks):
    """When _call_agent gets AuthError, it should switch to _gemini_text_call."""
    orch, mock_anthropic, mock_gemini = orchestrator_with_mocks
    config = MagicMock(model="claude-opus", max_tokens=1024,
                       system_prompt="sys")
    config.name = "TestAgent"

    text, usage = orch._call_agent(config, "test task")

    # Anthropic was called once, raised, then Gemini was called as fallback
    assert mock_anthropic.messages.create.call_count == 1
    assert mock_gemini.generate_content.call_count == 1
    # Fallback flag is now set
    assert orch._anthropic_unavailable is True
    # Returned text + usage from Gemini fallback (the #46 fix)
    assert text == "Mocked Gemini fallback response."
    assert usage["input"] == 137
    assert usage["output"] == 42


def test_call_agent_with_tools_auth_error_triggers_gemini_fallback(
    orchestrator_with_mocks,
):
    """Same fallback for the tools-enabled call path."""
    orch, mock_anthropic, mock_gemini = orchestrator_with_mocks
    config = MagicMock(model="claude-opus", max_tokens=1024,
                       system_prompt="sys")
    config.name = "TestAgent"

    text, usage = orch._call_agent_with_tools(config, "test task with tools")

    assert orch._anthropic_unavailable is True
    assert mock_gemini.generate_content.call_count == 1
    assert "Gemini fallback" in text
    assert usage["input"] == 137
    assert usage["output"] == 42


def test_anthropic_unavailable_flag_persists_after_first_401(orchestrator_with_mocks):
    """After first AuthError, subsequent calls skip Anthropic entirely."""
    orch, mock_anthropic, mock_gemini = orchestrator_with_mocks
    config = MagicMock(model="claude-opus", max_tokens=1024,
                       system_prompt="sys")
    config.name = "TestAgent"

    # First call: trips the AuthError catch, sets the flag
    orch._call_agent(config, "task 1")
    assert orch._anthropic_unavailable is True
    first_anthropic_calls = mock_anthropic.messages.create.call_count

    # Second call: should short-circuit to Gemini WITHOUT touching Anthropic again
    orch._call_agent(config, "task 2")
    assert mock_anthropic.messages.create.call_count == first_anthropic_calls, (
        "Anthropic should not be re-tried after _anthropic_unavailable=True"
    )
    assert mock_gemini.generate_content.call_count == 2  # both calls hit Gemini


def test_reset_anthropic_client_clears_flags_and_settings_cache():
    """reset_anthropic_client() must clear singleton state for key rotation."""
    from backend.agents import multi_agent_orchestrator as mao

    # Force a singleton with the unavailable flag tripped
    orch = mao.get_orchestrator()
    orch._anthropic_unavailable = True
    orch._client = MagicMock()

    # Spy on get_settings.cache_clear (it's an lru_cache wrapper)
    from backend.config.settings import get_settings
    with patch.object(get_settings, "cache_clear") as mock_clear:
        mao.reset_anthropic_client()
        mock_clear.assert_called_once()

    # Both flags cleared
    assert orch._anthropic_unavailable is False
    assert orch._client is None


def test_gemini_usage_dict_populated(orchestrator_with_mocks):
    """The #46 fix: _gemini_text_call must extract real token counts from
    LLMResponse.usage_metadata, not return hardcoded zeros."""
    orch, _mock_anthropic, mock_gemini = orchestrator_with_mocks
    config = MagicMock(model="claude-opus", max_tokens=1024,
                       system_prompt="sys")
    config.name = "TestAgent"

    text, usage = orch._gemini_text_call(config, "any task")

    assert mock_gemini.generate_content.call_count == 1
    assert text == "Mocked Gemini fallback response."
    # CRITICAL: these were both 0 before the #46 fix
    assert usage["input"] == 137, "input tokens not extracted from usage_metadata"
    assert usage["output"] == 42, "output tokens not extracted from usage_metadata"


def test_reset_is_safe_when_no_orchestrator_exists():
    """reset_anthropic_client() must not raise when called before any
    get_orchestrator() invocation."""
    from backend.agents import multi_agent_orchestrator as mao

    # Force the global to None (simulating a fresh import)
    mao._orchestrator = None

    # Should not raise
    mao.reset_anthropic_client()
    assert mao._orchestrator is None
