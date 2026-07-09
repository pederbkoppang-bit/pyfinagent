"""phase-67.6: builder-level request-shape tests per Claude model family.

Pins the exact kwargs ClaudeClient.generate_content (and the Layer-2
orchestrator tool-loop) pass to the Anthropic SDK, per model family:

- claude-fable-5: thinking key OMITTED entirely (always-on; ANY explicit
  config -- even {"type": "disabled"} -- 400s), NO sampling params, effort
  in output_config (xhigh no longer spuriously downgraded).
- claude-sonnet-5: adaptive thinking, NO sampling params, effort present
  (new EFFORT_SUPPORTED_MODELS + MODEL_EFFORT_FALLBACK entries).
- claude-opus-4-8 / 4-7: adaptive thinking, NO sampling params (preserved).
- claude-opus-4-6 / sonnet-4-6 / haiku-4-5: existing shapes preserved
  (adaptive + temperature=1 when thinking; temperature present otherwise;
  haiku carries no output_config).
- older models: legacy {"type": "enabled", "budget_tokens": N} survives.

The SDK client is the ONLY faked boundary (ClaudeClient._get_client /
orchestrator _get_client seam); the entire request-construction path under
test runs for real. Hermetic: no network, cost gate disabled via the
documented COST_BUDGET_HARD_BLOCK_DISABLED hatch.
"""
from __future__ import annotations

import os

os.environ.setdefault("COST_BUDGET_HARD_BLOCK_DISABLED", "1")

from types import SimpleNamespace

from backend.agents.llm_client import ClaudeClient
from backend.agents.agent_definitions import AgentConfig, AgentType
from backend.agents import multi_agent_orchestrator as mao


# ── SDK-boundary fake ────────────────────────────────────────────────────
def _fake_response():
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok", citations=None)],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        model="stub",
        _request_id="req_test",
    )


class _CaptureMessages:
    def __init__(self, captured: list):
        self.captured = captured

    def create(self, **kwargs):
        self.captured.append(kwargs)
        return _fake_response()


def _fake_client(captured: list):
    return SimpleNamespace(
        messages=_CaptureMessages(captured),
        beta=SimpleNamespace(messages=_CaptureMessages(captured)),
    )


def _shape(monkeypatch, model: str, config: dict) -> dict:
    """Run generate_content against the faked SDK; return captured kwargs."""
    captured: list = []
    monkeypatch.setattr(
        ClaudeClient, "_get_client", lambda self: _fake_client(captured)
    )
    import backend.services.observability as obs

    monkeypatch.setattr(obs, "log_llm_call", lambda *a, **k: None, raising=False)
    client = ClaudeClient(model_name=model, api_key="test-key")
    client.generate_content("hello", config)
    assert captured, "no request captured at the SDK boundary"
    return captured[0]


THINK = {"thinking": {"budget_tokens": 1024}}


# ── fable-5: the always-on family ────────────────────────────────────────
def test_fable5_omits_thinking_and_sampling(monkeypatch):
    k = _shape(monkeypatch, "claude-fable-5", {**THINK, "effort": "xhigh"})
    assert "thinking" not in k          # ANY explicit config 400s on fable
    assert "temperature" not in k
    assert "top_p" not in k and "top_k" not in k
    # xhigh survives (the old opus-only guard downgraded it to high)
    assert k["output_config"]["effort"] == "xhigh"


def test_fable5_no_thinking_request_still_clean(monkeypatch):
    k = _shape(monkeypatch, "claude-fable-5", {})
    assert "thinking" not in k
    assert "temperature" not in k
    # MODEL_EFFORT_FALLBACK fable row -> xhigh, passes the extended guard
    assert k["output_config"]["effort"] == "xhigh"


# ── sonnet-5: adaptive-only, no sampling ─────────────────────────────────
def test_sonnet5_adaptive_no_sampling(monkeypatch):
    k = _shape(monkeypatch, "claude-sonnet-5", {**THINK, "effort": "high"})
    assert k["thinking"] == {"type": "adaptive"}
    assert "temperature" not in k
    assert k["output_config"]["effort"] == "high"


def test_sonnet5_effort_fallback_without_explicit(monkeypatch):
    k = _shape(monkeypatch, "claude-sonnet-5", {})
    # New MODEL_EFFORT_FALLBACK row: sonnet-5 -> high (doc default)
    assert k["output_config"]["effort"] == "high"
    assert "temperature" not in k


# ── current families: shapes preserved (criterion 4) ─────────────────────
def test_opus48_shape_preserved(monkeypatch):
    k = _shape(monkeypatch, "claude-opus-4-8", dict(THINK))
    assert k["thinking"] == {"type": "adaptive"}
    assert "temperature" not in k


def test_opus47_shape_preserved(monkeypatch):
    k = _shape(monkeypatch, "claude-opus-4-7", dict(THINK))
    assert k["thinking"] == {"type": "adaptive"}
    assert "temperature" not in k


def test_opus46_keeps_temperature_one_with_thinking(monkeypatch):
    k = _shape(monkeypatch, "claude-opus-4-6", dict(THINK))
    assert k["thinking"] == {"type": "adaptive"}
    assert k["temperature"] == 1        # 4.6 accepts sampling; unchanged


def test_sonnet46_no_thinking_keeps_default_temperature(monkeypatch):
    k = _shape(monkeypatch, "claude-sonnet-4-6", {})
    assert "thinking" not in k
    assert k["temperature"] == 0.0      # base default; unchanged
    assert k["output_config"]["effort"] == "medium"


def test_haiku45_adaptive_temp1_no_output_config(monkeypatch):
    k = _shape(monkeypatch, "claude-haiku-4-5", dict(THINK))
    assert k["thinking"] == {"type": "adaptive"}
    assert k["temperature"] == 1
    assert "output_config" not in k     # haiku not in EFFORT_SUPPORTED_MODELS


def test_legacy_opus45_enabled_branch_survives(monkeypatch):
    k = _shape(monkeypatch, "claude-opus-4-5", dict(THINK))
    assert k["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert k["temperature"] == 1


# ── Layer-2 orchestrator tool-loop (declared additional scope) ───────────
def _loop_shape(monkeypatch, model: str) -> dict:
    captured: list = []
    import backend.services.observability as obs

    monkeypatch.setattr(obs, "log_llm_call", lambda *a, **k: None, raising=False)
    cfg = AgentConfig(
        agent_type=AgentType.QA,
        name="shape-test",
        model=model,
        system_prompt="sys",
        max_tokens=128,
    )
    fake_self = SimpleNamespace(
        _anthropic_unavailable=False,
        _get_client=lambda: _fake_client(captured),
    )
    mao.MultiAgentOrchestrator._call_agent_with_tools(fake_self, cfg, "task")
    assert captured, "no tool-loop request captured"
    return captured[0]


def test_orchestrator_loop_fable5_omits_thinking(monkeypatch):
    k = _loop_shape(monkeypatch, "claude-fable-5")
    assert "thinking" not in k
    assert "temperature" not in k


def test_orchestrator_loop_sonnet5_adaptive(monkeypatch):
    k = _loop_shape(monkeypatch, "claude-sonnet-5")
    assert k["thinking"] == {"type": "adaptive"}
    assert "temperature" not in k


def test_orchestrator_loop_opus48_preserved(monkeypatch):
    k = _loop_shape(monkeypatch, "claude-opus-4-8")
    assert k["thinking"] == {"type": "adaptive"}
    assert "temperature" not in k


def test_orchestrator_loop_legacy_family_unchanged(monkeypatch):
    k = _loop_shape(monkeypatch, "claude-sonnet-4-6")
    assert k["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert k["temperature"] == 1
