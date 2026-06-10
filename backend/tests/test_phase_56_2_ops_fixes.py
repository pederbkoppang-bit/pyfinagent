"""phase-56.2 ops fixes -- regression tests for the 55.3 finding map.

Covers:
- F-4 claude-CLI rail health probe (free, never-raises, alert-on-failure wiring)
- F-5 degraded-scoring guard predicate (all-zero / >=3-zero cycle detection)
- F-7 conviction-fallback detection (loud, ordering byte-identical)
- F-6 claude-code rail llm_call_log metering (envelope mapping, ok=False on rail error)
- criterion-2: ticket processor honors paper_use_claude_code_route (CLI rail, not direct SDK)

All offline: subprocess/SDK/logger sinks mocked; no network, BQ, or tokens.
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.agents import claude_code_client as ccc
from backend.services import autonomous_loop as al


# ── F-4: health probe ────────────────────────────────────────────────
def test_rail_probe_false_on_nonzero_exit():
    fake = SimpleNamespace(returncode=1, stdout="", stderr="not logged in")
    with patch.object(ccc.subprocess, "run", return_value=fake):
        ok, detail = ccc.claude_code_health_probe()
    assert ok is False
    assert "exit=1" in detail


def test_rail_probe_true_on_exit_zero_logged_in():
    fake = SimpleNamespace(returncode=0, stdout='{"loggedIn": true}', stderr="")
    with patch.object(ccc.subprocess, "run", return_value=fake):
        ok, detail = ccc.claude_code_health_probe()
    assert ok is True and detail == "ok"


def test_rail_probe_false_on_logged_out_json_even_with_exit_zero():
    fake = SimpleNamespace(returncode=0, stdout='{"loggedIn": false}', stderr="")
    with patch.object(ccc.subprocess, "run", return_value=fake):
        ok, detail = ccc.claude_code_health_probe()
    assert ok is False


def test_rail_probe_never_raises_on_timeout_or_missing_binary():
    with patch.object(
        ccc.subprocess, "run",
        side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=15),
    ):
        ok, _ = ccc.claude_code_health_probe()
    assert ok is False
    with patch.object(ccc.subprocess, "run", side_effect=FileNotFoundError()):
        ok, _ = ccc.claude_code_health_probe()
    assert ok is False


def test_rail_probe_scrubs_api_key_from_env(monkeypatch):
    """The probe must test the OAuth rail, not the direct API key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    captured = {}

    def _spy(args, **kwargs):
        captured["env"] = kwargs.get("env")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch.object(ccc.subprocess, "run", side_effect=_spy):
        ccc.claude_code_health_probe()
    assert "ANTHROPIC_API_KEY" not in (captured["env"] or {})


# ── F-5: degraded-scoring guard predicate ────────────────────────────
def _mk(score, conf=50, rec="Buy"):
    return {"final_score": score, "confidence": conf, "recommendation": rec}


def test_degraded_guard_fires_when_all_zero():
    fire, n_deg, n_tot = al._degraded_scoring_check([_mk(0.0), _mk(0.0), _mk(0.0)])
    assert fire is True and n_deg == 3 and n_tot == 3


def test_degraded_guard_fires_at_three_zeros_of_six():
    analyses = [_mk(0.0), _mk(0.0), _mk(0.0), _mk(7.0), _mk(6.0), _mk(4.0)]
    fire, n_deg, n_tot = al._degraded_scoring_check(analyses)
    assert fire is True and n_deg == 3 and n_tot == 6


def test_degraded_guard_quiet_at_two_zeros_of_six():
    analyses = [_mk(0.0), _mk(0.0), _mk(7.0), _mk(6.0), _mk(4.0), _mk(5.0)]
    fire, n_deg, n_tot = al._degraded_scoring_check(analyses)
    assert fire is False and n_deg == 2


def test_degraded_guard_counts_confidence_zero_uppercase_tell():
    """The 05-27 incident signature: confidence=0 + UPPERCASE rec (rail-down
    fallback) counts as degraded even when the score defaulted to 5."""
    rail_down = {"final_score": 5, "confidence": 0, "recommendation": "HOLD"}
    healthy = {"final_score": 5, "confidence": 60, "recommendation": "Hold"}
    fire, n_deg, _ = al._degraded_scoring_check([rail_down] * 3 + [healthy])
    assert fire is True and n_deg == 3


def test_degraded_guard_quiet_on_empty_cycle():
    fire, _, n_tot = al._degraded_scoring_check([])
    assert fire is False and n_tot == 0


# ── F-7: conviction-fallback detection (loud; ordering untouched) ────
def test_all_conviction_fallback_detects_dead_overlay():
    cands = [
        {"ticker": "MU", "conviction_score": 10,
         "conviction_reason": "conviction 10.00; fallback (LLM unavailable)"},
        {"ticker": "DELL", "conviction_score": 8,
         "conviction_reason": "fallback (LLM unavailable)"},
    ]
    assert al._all_conviction_fallback(cands) is True


def test_all_conviction_fallback_quiet_when_any_llm_scored():
    cands = [
        {"ticker": "MU", "conviction_reason": "fallback (LLM unavailable)"},
        {"ticker": "DELL", "conviction_reason": "HBM supercycle leader; sized for blow-off risk"},
    ]
    assert al._all_conviction_fallback(cands) is False
    assert al._all_conviction_fallback([]) is False


def test_fallback_ordering_byte_identical():
    """Do-no-harm invariant (F-7): the no-LLM fallback still ranks by the
    composite-derived conviction -- detection adds NO reordering."""
    from backend.services.meta_scorer import _fallback_all
    cands = [
        {"ticker": "A", "composite_score": 3.2},
        {"ticker": "B", "composite_score": 9.7},
        {"ticker": "C", "composite_score": 6.1},
    ]
    out = _fallback_all(cands)
    assert [c["ticker"] for c in out] == ["B", "C", "A"]
    assert all("fallback (LLM unavailable)" in c["conviction_reason"] for c in out)


# ── F-6: claude-code rail metering ───────────────────────────────────
def test_log_claude_code_call_maps_envelope_fields():
    envelope = {
        "model": "claude-haiku-4-5",
        "duration_ms": 1234.0,
        "usage": {"input_tokens": 100, "output_tokens": 50,
                  "cache_read_input_tokens": 7, "cache_creation_input_tokens": 3},
    }
    with patch("backend.services.observability.api_call_log.log_llm_call") as spy:
        al._log_claude_code_call(envelope, agent="lite_trader", ticker="MU", ok=True)
    spy.assert_called_once()
    kw = spy.call_args.kwargs
    assert kw["provider"] == "claude-code"
    assert kw["model"] == "claude-haiku-4-5"
    assert kw["agent"] == "lite_trader"
    assert kw["ticker"] == "MU"
    assert kw["input_tok"] == 100 and kw["output_tok"] == 50
    assert kw["ok"] is True


def test_log_claude_code_call_records_rail_failure_as_not_ok():
    with patch("backend.services.observability.api_call_log.log_llm_call") as spy:
        al._log_claude_code_call(None, agent="lite_risk_judge", ticker="SNDK", ok=False)
    kw = spy.call_args.kwargs
    assert kw["ok"] is False and kw["agent"] == "lite_risk_judge"
    assert kw["model"] == "claude-code-cli"


def test_log_claude_code_call_never_raises():
    with patch(
        "backend.services.observability.api_call_log.log_llm_call",
        side_effect=RuntimeError("boom"),
    ):
        al._log_claude_code_call({}, agent="x", ticker="Y", ok=True)  # no raise


# ── criterion 2: ticket processor honors the CLI rail ────────────────
def _mk_processor():
    from backend.services.ticket_queue_processor import TicketQueueProcessor
    proc = TicketQueueProcessor.__new__(TicketQueueProcessor)
    return proc


def test_ticket_agent_uses_cli_rail_when_route_flag_set():
    proc = _mk_processor()
    fake_settings = SimpleNamespace(
        paper_use_claude_code_route=True,
        anthropic_api_key=SimpleNamespace(get_secret_value=lambda: "sk-unused"),
    )
    fake_envelope = {"result": "Approved and acknowledged."}
    with patch("backend.config.settings.get_settings", return_value=fake_settings), \
         patch("backend.agents.claude_code_client.claude_code_invoke",
               return_value=fake_envelope) as cli_spy, \
         patch("anthropic.Anthropic") as sdk_spy:
        out = proc._spawn_real_agent("main", "Approve", 1, "T-1")
    assert out == "Approved and acknowledged."
    cli_spy.assert_called_once()
    sdk_spy.assert_not_called()


def test_ticket_agent_uses_direct_sdk_when_route_flag_off():
    proc = _mk_processor()
    fake_settings = SimpleNamespace(
        paper_use_claude_code_route=False,
        anthropic_api_key=SimpleNamespace(get_secret_value=lambda: "sk-test-not-real"),
    )
    fake_msg = SimpleNamespace(content=[SimpleNamespace(text="SDK reply")])
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_msg
    with patch("backend.config.settings.get_settings", return_value=fake_settings), \
         patch("backend.agents.claude_code_client.claude_code_invoke") as cli_spy, \
         patch("anthropic.Anthropic", return_value=fake_client):
        out = proc._spawn_real_agent("main", "Approve", 1, "T-2")
    assert out == "SDK reply"
    cli_spy.assert_not_called()
