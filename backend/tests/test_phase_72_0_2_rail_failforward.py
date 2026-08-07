"""phase-72.0.2: standard-tier fail-forward on rail-dead (FLAG-GATED DARK).

All tests are $0 and network-free: rail state is driven through the PUBLIC
rail_guard_disable/rail_guard_reset seam (66.1 precedent), the genai client is
a stub injected at backend.agents._genai_client.get_genai_client, and the
analysis wrapper is exercised with a monkeypatched _run_gemini_analysis.
Criterion 1's flag-OFF byte-identity is asserted on client TYPE and the
timeout_s constructor kwarg (phase-78.16 proved silent kwargs drift is a real
regression class on make_client).
"""
from __future__ import annotations

import asyncio
import copy

import pytest

from backend.agents import claude_code_client as ccc
from backend.agents.claude_code_client import (
    ClaudeCodeClient,
    rail_guard_disable,
    rail_guard_reset,
    rail_guard_status,
)
from backend.agents.llm_client import GeminiClient, make_client
from backend.services import autonomous_loop as al

WORKHORSE = "gemini-2.5-flash"


class _S:
    """Minimal settings stand-in (78.1 idiom): a real-shaped key + the rail
    flag, so branches differ only by the flags under test."""

    anthropic_api_key = "sk-ant-api-test-not-real"
    github_token = ""
    gemini_api_key = ""  # forces the Vertex leg, never AI Studio
    openai_api_key = ""
    claude_code_timeout_s = 60
    paper_use_claude_code_route = True
    paper_rail_failforward_enabled = False
    paper_failforward_model = WORKHORSE
    gemini_model = "claude-sonnet-4-6"


class _StubGenai:
    """Stands in for genai.Client -- constructor-only usage in the seam."""


@pytest.fixture(autouse=True)
def _clean_rail_state():
    rail_guard_reset("test-cycle-72-0-2")
    yield
    rail_guard_reset("test-cycle-72-0-2-teardown")


@pytest.fixture
def _stub_genai(monkeypatch):
    import backend.agents._genai_client as gc

    stub = _StubGenai()
    monkeypatch.setattr(gc, "get_genai_client", lambda: stub)
    return stub


def _settings(**kw) -> _S:
    s = _S()
    for k, v in kw.items():
        setattr(s, k, v)
    return s


# ── Seam A: make_client (llm_client.py) ──────────────────────────────


def test_flag_off_rail_dead_returns_claude_code_client(_stub_genai):
    """M1: flag OFF is byte-identical legacy even on a dead rail -- type AND
    the timeout_s kwarg (78.16 kwargs-drift guard)."""
    rail_guard_disable("test probe dead")
    c = make_client("claude-sonnet-4-6", None, _settings())
    assert isinstance(c, ClaudeCodeClient)
    assert c._timeout_s == 60  # 78.16 kwargs-drift guard (private, test-only read)


def test_flag_on_rail_healthy_returns_claude_code_client(_stub_genai):
    """M2: the fail-forward must never fire on a healthy rail."""
    c = make_client(
        "claude-sonnet-4-6", None, _settings(paper_rail_failforward_enabled=True)
    )
    assert isinstance(c, ClaudeCodeClient)


def test_flag_on_rail_dead_returns_vertex_gemini(_stub_genai):
    """C1 core: flag ON + probe dead -> a REAL Vertex GeminiClient over the
    in-seam ADC bundle (not GeminiClient(model=None) -- the None-trap, M5)."""
    rail_guard_disable("test probe dead")
    c = make_client(
        "claude-sonnet-4-6", None, _settings(paper_rail_failforward_enabled=True)
    )
    assert isinstance(c, GeminiClient)
    bundle = c._model
    assert bundle is not None, "the None-trap: bundle must be built in-seam"
    assert bundle.model_name == WORKHORSE
    assert bundle.client is _stub_genai, "bundle must carry the ADC client"
    assert bundle.base_config.get("temperature") == 0.0
    assert bundle.base_config.get("top_k") == 1


def test_flag_on_breaker_tripped_returns_vertex_gemini(_stub_genai, monkeypatch):
    """The second rail-dead leg: breaker_tripped (driven via the status
    reader, since tripping needs threshold consecutive failures)."""
    real = rail_guard_status()
    fake = {**real, "rail_skipped": False, "breaker_tripped": True,
            "last_error": "test breaker open"}
    monkeypatch.setattr(ccc, "rail_guard_status", lambda: fake)
    c = make_client(
        "claude-sonnet-4-6", None, _settings(paper_rail_failforward_enabled=True)
    )
    assert isinstance(c, GeminiClient)


def test_seam_reads_but_never_mutates_rail_state(_stub_genai):
    """M3 / criterion 2: the fail-forward is a strict READER -- the guard
    state must be deep-equal before and after a fail-forward routing."""
    rail_guard_disable("test probe dead")
    before = copy.deepcopy(rail_guard_status())
    c = make_client(
        "claude-sonnet-4-6", None, _settings(paper_rail_failforward_enabled=True)
    )
    assert isinstance(c, GeminiClient)
    assert rail_guard_status() == before, "fail-forward mutated the rail guard"


def test_no_genai_client_falls_through_to_rail(monkeypatch):
    """Fail-open: ADC unavailable (get_genai_client None) -> today's rail path."""
    import backend.agents._genai_client as gc

    monkeypatch.setattr(gc, "get_genai_client", lambda: None)
    rail_guard_disable("test probe dead")
    c = make_client(
        "claude-sonnet-4-6", None, _settings(paper_rail_failforward_enabled=True)
    )
    assert isinstance(c, ClaudeCodeClient)


def test_non_gemini_substitute_falls_through_to_rail(_stub_genai):
    """Misconfigured paper_failforward_model disables the branch fail-open."""
    rail_guard_disable("test probe dead")
    c = make_client(
        "claude-sonnet-4-6",
        None,
        _settings(
            paper_rail_failforward_enabled=True,
            paper_failforward_model="claude-haiku-4-5",
        ),
    )
    assert isinstance(c, ClaudeCodeClient)


# ── Seam B: _select_lite_analyzer (autonomous_loop.py) ───────────────


def test_selector_legacy_signature_unchanged():
    """Pre-72.0.2 callers (settings omitted) get today's routing exactly."""
    assert al._select_lite_analyzer("claude-sonnet-4-6") is al._run_claude_analysis
    assert al._select_lite_analyzer("gemini-2.5-flash") is al._run_gemini_analysis


def test_selector_flag_off_rail_dead_returns_claude():
    rail_guard_disable("test probe dead")
    got = al._select_lite_analyzer("claude-sonnet-4-6", _settings())
    assert got is al._run_claude_analysis


def test_selector_flag_on_rail_healthy_returns_claude():
    got = al._select_lite_analyzer(
        "claude-sonnet-4-6", _settings(paper_rail_failforward_enabled=True)
    )
    assert got is al._run_claude_analysis


def test_selector_flag_on_rail_dead_returns_failforward():
    rail_guard_disable("test probe dead")
    got = al._select_lite_analyzer(
        "claude-sonnet-4-6", _settings(paper_rail_failforward_enabled=True)
    )
    assert got is al._run_failforward_analysis


def test_selector_gemini_standard_never_failforwards():
    rail_guard_disable("test probe dead")
    got = al._select_lite_analyzer(
        "gemini-2.5-flash", _settings(paper_rail_failforward_enabled=True)
    )
    assert got is al._run_gemini_analysis


# ── model resolution (D3) ────────────────────────────────────────────


def test_resolve_override_wins_over_settings():
    assert (
        al._resolve_lite_gemini_model(_settings(gemini_model="gemini-2.5-pro"),
                                      model_override="gemini-2.5-flash")
        == "gemini-2.5-flash"
    )


def test_resolve_default_is_settings_model():
    s = _settings(gemini_model="gemini-2.5-pro")
    assert al._resolve_lite_gemini_model(s) == "gemini-2.5-pro"


def test_resolve_rejects_non_gemini_either_way():
    with pytest.raises(ValueError, match="is not a Gemini model"):
        al._resolve_lite_gemini_model(_settings(gemini_model="claude-sonnet-4-6"))
    with pytest.raises(ValueError, match="is not a Gemini model"):
        al._resolve_lite_gemini_model(
            _settings(gemini_model="gemini-2.5-pro"),
            model_override="claude-haiku-4-5",
        )


# ── quality floor (D4) ───────────────────────────────────────────────

_GOOD = {"action": "BUY", "confidence": 72, "score": 7, "reason": "momentum"}


def test_floor_accepts_real_payload():
    assert al._failforward_floor_ok(_GOOD) is True


@pytest.mark.parametrize(
    "mutant",
    [
        None,
        "not a dict",
        {**_GOOD, "action": "buy"},
        {**_GOOD, "action": "PANIC"},
        {**_GOOD, "confidence": None},
        {**_GOOD, "confidence": 0},  # the fabrication tell (M4)
        {**_GOOD, "confidence": 101},
        {**_GOOD, "score": 0},
        {**_GOOD, "score": 11},
        {**_GOOD, "score": None},
        {**_GOOD, "reason": "  "},
        {**_GOOD, "_parse_failed": True},
        {"action": "HOLD", "confidence": 0, "score": 5,
         "reason": "Could not parse analysis", "_parse_failed": True},
    ],
)
def test_floor_rejects_degenerate_payloads(mutant):
    assert al._failforward_floor_ok(mutant) is False


# ── _run_failforward_analysis (D5 provenance + 61.2 interaction) ─────


def _ff_result(inner):
    return {
        "ticker": "TEST",
        "_path": "lite",
        "recommendation": inner.get("action"),
        "final_score": inner.get("score"),
        "full_report": {"source": WORKHORSE, "analysis": dict(inner)},
    }


def test_failforward_stamps_provenance_on_floor_pass(monkeypatch, _stub_genai):
    """M7: a floor-passing substituted answer carries the provenance stamps
    and is NOT degraded -- auditable, never masquerading as rail-served."""
    rail_guard_disable("test probe dead")

    async def fake_gemini(ticker, settings, portfolio_context="", model_override=None,
                          client_override=None):
        assert model_override == WORKHORSE, "override must reach the Gemini path"
        assert client_override is not None, "the Vertex client must be injected"
        return _ff_result(_GOOD)

    monkeypatch.setattr(al, "_run_gemini_analysis", fake_gemini)
    out = asyncio.run(al._run_failforward_analysis(
        "TEST", _settings(paper_rail_failforward_enabled=True)
    ))
    assert out["_failforward"] is True
    assert out["_failforward_provider"] == WORKHORSE
    assert "test probe dead" in out["_failforward_reason"]
    assert "_degraded" not in out
    # D9 diagonal cell: a floor-passing fail-forward result survives the
    # 61.2 integrity fold into decide_trades.
    assert al._fold_degraded_for_trading(out) is out


def test_failforward_floor_fail_is_honest_degraded(monkeypatch, _stub_genai):
    """M4/M6: the degenerate fabrication shape must NOT pass -- it becomes an
    honest _degraded row the 61.2 fold drops, never a tradeable value."""
    rail_guard_disable("test probe dead")
    degenerate = {"action": "HOLD", "confidence": 0, "score": 5,
                  "reason": "No JSON in trader response"}

    async def fake_gemini(ticker, settings, portfolio_context="", model_override=None,
                          client_override=None):
        return _ff_result(degenerate)

    monkeypatch.setattr(al, "_run_gemini_analysis", fake_gemini)
    out = asyncio.run(al._run_failforward_analysis(
        "TEST", _settings(paper_rail_failforward_enabled=True)
    ))
    assert out["_degraded"] is True
    assert out["_failforward_reason"].startswith("floor_reject:")
    # D9 diagonal cell: the 61.2 fold drops it from decide_trades input.
    assert al._fold_degraded_for_trading(out) is None


def test_failforward_misconfigured_model_falls_to_legacy(monkeypatch):
    """A non-gemini substitute fails open to the legacy claude lite path."""
    rail_guard_disable("test probe dead")
    sentinel = {"ticker": "TEST", "_path": "lite", "recommendation": "HOLD"}

    async def fake_claude(ticker, settings, portfolio_context=""):
        return dict(sentinel)

    monkeypatch.setattr(al, "_run_claude_analysis", fake_claude)
    out = asyncio.run(al._run_failforward_analysis(
        "TEST",
        _settings(
            paper_rail_failforward_enabled=True,
            paper_failforward_model="claude-haiku-4-5",
        ),
    ))
    assert out == sentinel
    assert "_failforward" not in out


# ── Seam-B transport: Vertex-only, key-independent (Q/A F1/F2) ───────


def test_failforward_client_is_vertex_adc(_stub_genai):
    """The transport guard the cycle-1 Q/A named: the fail-forward client must
    carry the in-seam ADC bundle -- NOT the AI-Studio key branch (which builds
    its own genai.Client from GEMINI_API_KEY) and NOT GeminiClient(model=None)
    (the None-trap)."""
    c = al._build_failforward_client(WORKHORSE)
    assert isinstance(c, GeminiClient)
    bundle = c._model
    assert bundle is not None, "the None-trap: the fail-forward bundle is None"
    assert bundle.client is _stub_genai, (
        "not the ADC singleton -- the AI-Studio leg builds its own client"
    )
    assert bundle.model_name == WORKHORSE


def test_failforward_client_none_without_adc(monkeypatch):
    import backend.agents._genai_client as gc

    monkeypatch.setattr(gc, "get_genai_client", lambda: None)
    assert al._build_failforward_client(WORKHORSE) is None


def test_failforward_no_adc_falls_to_legacy(monkeypatch):
    """ADC unavailable -> fail-open to the legacy lite path (never a broken
    client, never a fabricated score)."""
    import backend.agents._genai_client as gc

    monkeypatch.setattr(gc, "get_genai_client", lambda: None)
    rail_guard_disable("test probe dead")
    sentinel = {"ticker": "TEST", "_path": "lite", "recommendation": "HOLD"}

    async def fake_claude(ticker, settings, portfolio_context=""):
        return dict(sentinel)

    monkeypatch.setattr(al, "_run_claude_analysis", fake_claude)
    out = asyncio.run(al._run_failforward_analysis(
        "TEST", _settings(paper_rail_failforward_enabled=True)
    ))
    assert out == sentinel


class _RecordingClient:
    """Fake LLM client recording generate_content calls, returning valid JSON
    so the REAL _run_gemini_analysis parse path exercises end-to-end."""

    def __init__(self):
        self.calls = []

    def generate_content(self, prompt, config=None):
        self.calls.append(config or {})

        class _R:
            text = ('{"action": "BUY", "confidence": 70, "score": 7, '
                    '"reason": "test"}')
            thoughts = ""

        return _R()


class _FakeHist:
    def __len__(self):
        return 0


def test_seam_b_full_path_uses_injected_client(monkeypatch):
    """FULL-PATH plumb (closes the cycle-1 vacuity finding): the REAL
    _run_gemini_analysis runs end-to-end with fake market data and the
    injected client -- proving client_override carries the transport (a
    mutant that re-routes to make_client leaves the recorder uncalled)."""
    rec = _RecordingClient()
    monkeypatch.setattr(al, "_fetch_yf_market_data",
                        lambda ticker: ({"shortName": "Test Co"}, _FakeHist()))
    out = asyncio.run(al._run_gemini_analysis(
        "TEST", _settings(), model_override=WORKHORSE, client_override=rec,
    ))
    assert len(rec.calls) == 2, "trader + risk-judge must use the injected client"
    assert out["full_report"]["analysis"]["action"] == "BUY"
    assert out["full_report"]["source"] == WORKHORSE


def test_failforward_wires_built_client_into_gemini_path(monkeypatch, _stub_genai):
    """Wiring: _run_failforward_analysis passes EXACTLY the client built by
    _build_failforward_client into _run_gemini_analysis."""
    rail_guard_disable("test probe dead")
    sentinel_client = object()
    monkeypatch.setattr(al, "_build_failforward_client",
                        lambda m: sentinel_client)

    async def fake_gemini(ticker, settings, portfolio_context="", model_override=None,
                          client_override=None):
        assert client_override is sentinel_client
        return _ff_result(_GOOD)

    monkeypatch.setattr(al, "_run_gemini_analysis", fake_gemini)
    out = asyncio.run(al._run_failforward_analysis(
        "TEST", _settings(paper_rail_failforward_enabled=True)
    ))
    assert out["_failforward"] is True
