"""phase-cycle-3 (2026-05-26): tests for the Claude Code CLI subprocess client.

Mocks `subprocess.run` so the tests don't require an actual `claude` install
in CI. Asserts the envelope-parsing + error-detection contracts that the
autonomous-loop relies on when settings.paper_use_claude_code_route=True.
"""
from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from backend.agents.claude_code_client import (
    ClaudeCodeError,
    claude_code_invoke,
    extract_result_text,
)


def _mock_completed(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["claude"], returncode=returncode, stdout=stdout, stderr=stderr,
    )


def test_claude_code_invoke_returns_envelope():
    """Happy path: subprocess returns success envelope; function returns parsed dict."""
    envelope = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "BUY with confidence 80",
        "session_id": "abc123",
        "usage": {"input_tokens": 120, "output_tokens": 45},
        "total_cost_usd": 0.001,
        "duration_ms": 850,
    }
    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        run.return_value = _mock_completed(stdout=json.dumps(envelope))
        result = claude_code_invoke("analyze AAPL")
    assert result["subtype"] == "success"
    assert result["result"] == "BUY with confidence 80"
    assert result["usage"]["input_tokens"] == 120


def test_claude_code_invoke_raises_on_error_subtype():
    """When subtype != 'success', the function raises ClaudeCodeError."""
    envelope = {
        "type": "result",
        "subtype": "error_max_turns",
        "is_error": True,
        "stop_reason": "max_turns",
    }
    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        run.return_value = _mock_completed(stdout=json.dumps(envelope))
        with pytest.raises(ClaudeCodeError, match="subtype='error_max_turns'"):
            claude_code_invoke("analyze AAPL")


def test_claude_code_invoke_handles_timeout():
    """When the subprocess times out, the function raises ClaudeCodeError cleanly."""
    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=120)
        with pytest.raises(ClaudeCodeError, match="timeout after 120s"):
            claude_code_invoke("slow analysis", timeout_s=120)


def test_claude_code_invoke_handles_non_zero_exit():
    """When the CLI exits non-zero, the function raises ClaudeCodeError."""
    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        run.return_value = _mock_completed(stdout="", returncode=2, stderr="boom")
        with pytest.raises(ClaudeCodeError, match="exited with code 2"):
            claude_code_invoke("analyze AAPL")


def test_claude_code_invoke_handles_missing_binary():
    """When `claude` is not on PATH, raise ClaudeCodeError with a useful message."""
    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        run.side_effect = FileNotFoundError("claude not found")
        with pytest.raises(ClaudeCodeError, match="claude CLI not found"):
            claude_code_invoke("analyze AAPL", binary="claude")


def test_claude_code_invoke_handles_invalid_json():
    """When stdout is not valid JSON, raise ClaudeCodeError."""
    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        run.return_value = _mock_completed(stdout="not json at all")
        with pytest.raises(ClaudeCodeError, match="invalid JSON"):
            claude_code_invoke("analyze AAPL")


def test_extract_result_text_prefers_structured_output():
    envelope = {
        "subtype": "success",
        "result": "long result text",
        "structured_output": {"action": "BUY", "confidence": 80},
    }
    assert "BUY" in extract_result_text(envelope)


def test_extract_result_text_falls_back_to_result():
    envelope = {"subtype": "success", "result": "fallback text"}
    assert extract_result_text(envelope) == "fallback text"


def test_extract_result_text_returns_empty_when_missing():
    envelope = {"subtype": "success"}
    assert extract_result_text(envelope) == ""


def test_claude_code_client_class_adapts_to_llm_client_interface():
    """ClaudeCodeClient should provide generate_content() returning LLMResponse."""
    from backend.agents.claude_code_client import ClaudeCodeClient
    from backend.agents.llm_client import LLMResponse

    envelope = {
        "subtype": "success",
        "result": "HOLD with confidence 50",
        "usage": {"input_tokens": 100, "output_tokens": 30},
    }
    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        run.return_value = _mock_completed(stdout=json.dumps(envelope))
        client = ClaudeCodeClient(model_name="claude-sonnet-4-6")
        resp = client.generate_content("analyze MSFT")

    assert isinstance(resp, LLMResponse)
    assert resp.text == "HOLD with confidence 50"
    assert resp.input_tokens == 100
    assert resp.output_tokens == 30


def test_claude_code_client_class_returns_empty_on_error():
    """When the CLI returns an error envelope, generate_content() returns an
    empty-text LLMResponse with thoughts='errored: ...' so downstream callers
    can continue without crashing."""
    from backend.agents.claude_code_client import ClaudeCodeClient

    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=120)
        client = ClaudeCodeClient(model_name="claude-sonnet-4-6")
        resp = client.generate_content("slow prompt")

    assert resp.text == ""
    assert "errored" in resp.thoughts
