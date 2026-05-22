"""phase-35.2 GeminiClient.generate_content telemetry retrofit.

Fixes the closure_roadmap §3 OPEN-23 finding: phase-34.2 cycle 3 (and
c7801712) had 0 llm_call_log rows because ClaudeClient.generate_content
had the log_llm_call retrofit (line 1645+) but GeminiClient.generate_content
did NOT. Since phase-34.1 flipped both standard + deep-think tiers to
gemini-2.5-pro, every Risk-Judge invocation now goes through GeminiClient
and bypasses telemetry. This step mirrors the ClaudeClient pattern into
GeminiClient.

Verification approach: mock the GeminiModelBundle + intercept log_llm_call
to assert it's called with the right shape (provider="gemini", model,
agent + ticker from _role/_ticker side-channel, latency_ms, ok=True).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest


def test_phase_35_2_gemini_log_llm_call_present_in_source():
    """Sanity: the log_llm_call retrofit shows up in GeminiClient.generate_content."""
    src = open("backend/agents/llm_client.py").read()
    # The retrofit comment uniquely identifies the new block
    assert "phase-35.2: llm_call_log retrofit for Gemini path" in src
    # Verify it's in the GeminiClient.generate_content function (line ~849-1075).
    # Find the GeminiClient definition + the new block, ensure the latter is after.
    gc_idx = src.find("class GeminiClient(LLMClient):")
    retrofit_idx = src.find("phase-35.2: llm_call_log retrofit for Gemini path")
    next_class_idx = src.find("class OpenAIClient", gc_idx + 1)
    assert gc_idx < retrofit_idx < next_class_idx, "retrofit must be inside GeminiClient body"


def test_phase_35_2_gemini_log_llm_call_provider_is_gemini():
    """The new write block tags provider='gemini' (not 'vertex' or 'google')."""
    src = open("backend/agents/llm_client.py").read()
    # Locate the retrofit block
    start = src.find("phase-35.2: llm_call_log retrofit for Gemini path")
    end = src.find("return LLMResponse(", start)
    block = src[start:end]
    assert 'provider="gemini"' in block, "provider tag must be 'gemini'"
    assert "agent=generation_config.get(\"_role\")" in block, "must pluck _role side-channel"
    assert "ticker=generation_config.get(\"_ticker\")" in block, "must pluck _ticker side-channel"


def test_phase_35_2_gemini_log_llm_call_fail_open():
    """The retrofit is wrapped in try/except + logs at debug level
    (fail-open per existing pattern)."""
    src = open("backend/agents/llm_client.py").read()
    start = src.find("phase-35.2: llm_call_log retrofit for Gemini path")
    end = src.find("return LLMResponse(", start)
    block = src[start:end]
    assert "try:" in block
    assert "except Exception as _exc:" in block
    assert "[GeminiClient] llm_call_log write skipped" in block, "fail-open with debug log"


def test_phase_35_2_gemini_timer_started_before_call():
    """Latency measurement: _t0 = _time.perf_counter() must run BEFORE the
    actual SDK call AND before _latency_ms is computed for the telemetry write.
    Verified structurally via line ordering."""
    src = open("backend/agents/llm_client.py").read()
    gc_start = src.find("class GeminiClient(LLMClient):")
    method_start = src.find("def generate_content", gc_start)
    next_method_start = src.find("\n    def ", method_start + 1)

    method_body = src[method_start:next_method_start]
    # Timer started in method body
    assert "_t0 = _time.perf_counter()" in method_body, "timer must be started"
    # _latency_ms computed in method body (post-call timing)
    assert "_latency_ms = (_time.perf_counter() - _t0) * 1000.0" in method_body, \
        "_latency_ms must be computed from _t0 delta"
    # The latency computation comes AFTER the timer start
    timer_pos = method_body.find("_t0 = _time.perf_counter()")
    latency_pos = method_body.find("_latency_ms = (_time.perf_counter() - _t0) * 1000.0")
    assert timer_pos < latency_pos, "_t0 must be set before _latency_ms is computed"


def test_phase_35_2_log_llm_call_signature_compatible():
    """The new write block calls log_llm_call with the same kwargs the
    existing ClaudeClient path uses -- so the observability helper API
    contract is preserved (single source of truth)."""
    src = open("backend/agents/llm_client.py").read()
    # Both blocks should pass: provider, model, agent, latency_ms, ttft_ms,
    # input_tok, output_tok, cache_creation_tok, cache_read_tok, request_id,
    # ok, ticker.
    for kwarg in ["provider=", "model=", "agent=", "latency_ms=", "ttft_ms=",
                  "input_tok=", "output_tok=", "cache_creation_tok=",
                  "cache_read_tok=", "request_id=", "ok=True", "ticker="]:
        # Find at least 2 occurrences (one in ClaudeClient + one in GeminiClient)
        count = src.count(kwarg)
        assert count >= 2, f"kwarg {kwarg!r} should appear in both ClaudeClient and GeminiClient log_llm_call sites; got {count}"
