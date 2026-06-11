"""phase-60.1 (AW-4) tests: deep-pipeline restoration + honest-degradation alarm.

Covers the four 60.1 criteria surfaces:
1. model_pin -- the retired gemini-2.0-flash is gone from every LIVE config
   surface and the workhorse pin (gemini-2.5-flash, live-smoke-proven
   2026-06-11) is in place.
2. fallback_alarm -- the cycle-level full->lite fallback-rate predicate
   reproduces the away-week 100%-fallback case, respects the strict->
   threshold semantics, and never fires on deliberate lite_mode.
3. KR-aware skip -- non-SEC tickers are detected and the yfinance-only quant
   substitute has the CF-compatible shape with explicit skip provenance.
4. provenance -- _persist_analysis stamps `_path` into the persisted JSON and
   the morning digest renders the [lite]/[full] marker.

File name carries `60_1` so the immutable verification selector
(`-k 'fallback_alarm or model_pin or 60_1'`) collects everything here.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.config.model_tiers import _BUILD_TIER, GEMINI_WORKHORSE  # noqa: E402
from backend.services.autonomous_loop import (  # noqa: E402
    _fallback_rate_check,
    _persist_analysis,
)

RETIRED = "gemini-2.0-flash"


# ── 1. model_pin ─────────────────────────────────────────────────────────


def test_model_pin_workhorse_is_not_the_retired_model():
    assert GEMINI_WORKHORSE != RETIRED
    assert GEMINI_WORKHORSE.startswith("gemini-")


def test_model_pin_no_retired_gemini_in_build_tier():
    offenders = {role: m for role, m in _BUILD_TIER.items() if m == RETIRED}
    assert not offenders, f"retired pin still in _BUILD_TIER: {offenders}"
    assert _BUILD_TIER["gemini_enrichment"] == GEMINI_WORKHORSE
    assert _BUILD_TIER["layer1_swappable"] == GEMINI_WORKHORSE


def test_model_pin_orchestrator_fallback_repinned():
    from backend.agents.orchestrator import AnalysisOrchestrator

    assert AnalysisOrchestrator._GEMINI_FALLBACK == GEMINI_WORKHORSE
    # Non-Gemini selection must resolve to the workhorse, never the retired id.
    assert AnalysisOrchestrator._resolve_gemini("claude-sonnet-4-6") == GEMINI_WORKHORSE
    # Gemini selections pass through untouched.
    assert AnalysisOrchestrator._resolve_gemini("gemini-2.5-pro") == "gemini-2.5-pro"


def test_model_pin_settings_api_whitelist_drops_retired():
    from backend.api.settings_api import _VALID_MODELS, AVAILABLE_MODELS

    assert RETIRED not in _VALID_MODELS, "selecting a discontinued model guarantees 404s"
    assert GEMINI_WORKHORSE in _VALID_MODELS
    listed = {m["model"] for m in AVAILABLE_MODELS}
    assert RETIRED not in listed
    assert GEMINI_WORKHORSE in listed


def test_model_pin_inventory_repinned():
    inv_path = REPO_ROOT / "backend" / "agents" / "_inventory.json"
    inv = json.loads(inv_path.read_text(encoding="utf-8"))
    offenders = [
        n.get("id") for n in inv.get("agents", inv if isinstance(inv, list) else [])
        if isinstance(n, dict) and n.get("model") == RETIRED
    ]
    assert not offenders, f"_inventory.json nodes still on retired pin: {offenders}"


def test_model_pin_cost_tracker_has_workhorse_pricing():
    from backend.agents.cost_tracker import MODEL_PRICING

    # The workhorse must be priceable (cost rows would otherwise log $0)...
    assert GEMINI_WORKHORSE in MODEL_PRICING
    # ...at the 2026-06-11 re-verified rate.
    assert MODEL_PRICING[GEMINI_WORKHORSE] == (0.30, 2.50)
    # The retired row is KEPT deliberately: historical BQ rows still
    # reference it for cost recomputation.
    assert RETIRED in MODEL_PRICING


# ── 2. fallback_alarm ────────────────────────────────────────────────────


def _mk(ticker: str, fb: str | None = None, path: str = "lite") -> dict:
    a = {"ticker": ticker, "_path": path, "final_score": 7.0, "recommendation": "BUY"}
    if fb:
        a["_fallback_reason"] = fb
    return a


def test_fallback_alarm_fires_on_away_week_100pct():
    # The away-week reproduction: every analysis intended full, landed lite
    # (retired-pin 404s for US, SEC-CIK aborts for KR).
    analyses = [
        _mk("NVDA", "ClientError: 404 Publisher Model gemini-2.0-flash was not found"),
        _mk("MU", "ClientError: 404 Publisher Model gemini-2.0-flash was not found"),
        _mk("STX", "TimeoutError: agent exceeded 90s"),
        _mk("005930.KS", "RuntimeError: ERROR: Ticker 005930.KS not found in SEC CIK mapping."),
        _mk("066570.KS", "RuntimeError: ERROR: Ticker 066570.KS not found in SEC CIK mapping."),
    ]
    fire, n_fb, n_total, reasons = _fallback_rate_check(analyses, 0.5)
    assert fire is True
    assert (n_fb, n_total) == (5, 5)
    # Per-ticker failure reasons must be named -- the alarm's whole point.
    assert "005930.KS" in reasons and "SEC CIK" in reasons["005930.KS"]
    assert "NVDA" in reasons and "404" in reasons["NVDA"]


def test_fallback_alarm_threshold_is_strictly_greater_than():
    # 2/4 = 0.5 is NOT > 0.5 -> quiet.
    quiet = [_mk("A", "x"), _mk("B", "x"), _mk("C"), _mk("D")]
    fire, n_fb, n_total, _ = _fallback_rate_check(quiet, 0.5)
    assert fire is False and (n_fb, n_total) == (2, 4)
    # 3/5 = 0.6 IS > 0.5 -> fires.
    loud = [_mk("A", "x"), _mk("B", "x"), _mk("C", "x"), _mk("D"), _mk("E")]
    fire, n_fb, n_total, _ = _fallback_rate_check(loud, 0.5)
    assert fire is True and (n_fb, n_total) == (3, 5)


def test_fallback_alarm_quiet_on_deliberate_lite_mode():
    # Operator chose lite_mode=True: lite rows carry NO _fallback_reason.
    analyses = [_mk("AAPL"), _mk("MSFT"), _mk("NVDA")]
    fire, n_fb, n_total, reasons = _fallback_rate_check(analyses, 0.5)
    assert fire is False and n_fb == 0 and n_total == 3 and reasons == {}


def test_fallback_alarm_empty_cycle_never_fires():
    fire, n_fb, n_total, _ = _fallback_rate_check([], 0.5)
    assert fire is False and (n_fb, n_total) == (0, 0)


# ── 3. KR-aware skip (60_1) ──────────────────────────────────────────────


def test_60_1_sec_coverage_by_suffix():
    from backend.agents.orchestrator import AnalysisOrchestrator

    assert AnalysisOrchestrator._is_sec_covered("AAPL") is True
    assert AnalysisOrchestrator._is_sec_covered("005930.KS") is False
    assert AnalysisOrchestrator._is_sec_covered("066570.KS") is False
    assert AnalysisOrchestrator._is_sec_covered("SAP.DE") is False


def test_60_1_quant_from_yfinance_shape():
    from backend.agents.orchestrator import _quant_from_yfinance

    yf = {
        "company_name": "Samsung Electronics",
        "sector": "Technology",
        "industry": "Semiconductors",
        "valuation": {"Current Price": 61000.0, "Market Cap": 4.1e14, "P/E Ratio": 12.3},
    }
    q = _quant_from_yfinance("005930.ks", yf)
    # CF-shape compatibility: the keys downstream consumers read.
    assert q["ticker"] == "005930.KS"
    assert q["company_name"] == "Samsung Electronics"
    assert q["sector"] == "Technology"
    assert q["yf_data"] is yf
    assert q["part_5_valuation"]["market_price"] == 61000.0
    # SEC-only fields are explicit Nones with an honest source string --
    # never silently invented.
    assert q["cik"] is None
    assert q["part_1_financials"]["latest_revenue"] is None
    assert "SEC EDGAR skipped" in q["part_1_financials"]["source"]


def test_60_1_quant_from_yfinance_tolerates_none():
    from backend.agents.orchestrator import _quant_from_yfinance

    q = _quant_from_yfinance("005930.KS", None)
    assert q["company_name"] == "005930.KS"
    assert q["part_5_valuation"]["market_price"] is None


# ── 4. provenance (60_1) ─────────────────────────────────────────────────


class _CaptureBQ:
    def __init__(self):
        self.kwargs = None

    def save_report(self, **kwargs):
        self.kwargs = kwargs


def test_60_1_persist_stamps_path_into_full_report_json():
    bq = _CaptureBQ()
    analysis = {
        "ticker": "MU",
        "_path": "lite",
        "_fallback_reason": "ClientError: 404 Publisher Model not found",
        "final_score": 7.0,
        "recommendation": "BUY",
        "risk_assessment": {"reason": "r"},
        "full_report": {"source": "claude-sonnet-4-6", "market_data": {}},
    }
    asyncio.run(_persist_analysis(analysis, bq))
    assert bq.kwargs is not None, "save_report was not called"
    fr = bq.kwargs["full_report"]
    assert fr["_path"] == "lite"
    assert "404" in fr["_fallback_reason"]
    # The in-memory analysis dict's own full_report is not mutated.
    assert "_path" not in analysis["full_report"]


def test_60_1_persist_full_path_row_tagged_full():
    bq = _CaptureBQ()
    analysis = {
        "ticker": "NVDA",
        "_path": "full",
        "final_score": 8.2,
        "recommendation": "BUY",
        "risk_assessment": {"reason": "r"},
        "full_report": {"source": "claude-sonnet-4-6", "rail": "anthropic_direct", "market_data": {}},
    }
    asyncio.run(_persist_analysis(analysis, bq))
    fr = bq.kwargs["full_report"]
    assert fr["_path"] == "full"
    assert "_fallback_reason" not in fr


def test_60_1_report_summary_model_accepts_analysis_path():
    from backend.api.models import ReportSummary

    row = ReportSummary(
        ticker="MU", analysis_date="2026-06-11T00:00:00Z", final_score=7.0,
        recommendation="BUY", summary="s", analysis_path="lite",
    )
    assert row.analysis_path == "lite"
    legacy = ReportSummary(
        ticker="MU", analysis_date="2026-06-11T00:00:00Z", final_score=7.0,
        recommendation="BUY", summary="s",
    )
    assert legacy.analysis_path is None


class _CaptureGenaiClient:
    """Duck-typed genai.Client capturing the assembled GenerateContentConfig."""

    def __init__(self):
        self.captured_config = None
        outer = self

        class _Models:
            def generate_content(self, *, model, contents, config):
                outer.captured_config = config
                from types import SimpleNamespace

                return SimpleNamespace(
                    text="ok", candidates=[], usage_metadata=None,
                )

        self.models = _Models()


def _gemini_call_config(model_name: str, generation_config: dict | None = None):
    from backend.agents.llm_client import GeminiClient, GeminiModelBundle

    fake = _CaptureGenaiClient()
    bundle = GeminiModelBundle(client=fake, model_name=model_name)
    client = GeminiClient(bundle, model_name)
    client.generate_content("hello", generation_config=dict(generation_config or {}))
    return fake.captured_config


def test_model_pin_25_flash_thinking_disabled_by_default():
    # The repin's latency regression: 2.5-flash thinks by default and blew
    # the 90s grounded-step timeout (MU market step, live 2026-06-11).
    # Without an explicit opt-in the budget must be pinned to 0.
    cfg = _gemini_call_config("gemini-2.5-flash", {"temperature": 0.0})
    assert cfg is not None and cfg.thinking_config is not None
    assert cfg.thinking_config.thinking_budget == 0


def test_model_pin_25_pro_thinking_left_at_model_default():
    # 2.5-pro rejects thinking_budget=0 (min 128) -- must NOT be forced off.
    cfg = _gemini_call_config("gemini-2.5-pro", {"temperature": 0.0})
    assert cfg is None or cfg.thinking_config is None


def test_model_pin_thinking_opt_in_still_wins():
    # The enable_thinking path (legacy dict form) keeps its budget.
    cfg = _gemini_call_config(
        "gemini-2.5-flash",
        {"thinking": {"type": "enabled", "budget_tokens": 4096}},
    )
    assert cfg.thinking_config.thinking_budget == 4096


def test_60_1_step_timeout_resolution():
    from types import SimpleNamespace

    from backend.agents.orchestrator import _resolve_step_timeout

    plain = SimpleNamespace()
    cc_rail = SimpleNamespace(recommended_step_timeout=150)

    # Defaults untouched for plain non-grounded calls.
    assert _resolve_step_timeout(plain, 90, False) == 90
    # Grounded calls at the default get the 2.5-flash-appropriate budget.
    assert _resolve_step_timeout(plain, 90, True) == 180
    # The CLI rail lifts the budget above its own 120s subprocess timeout
    # (88.9s round-trips observed live racing the old 90s step budget).
    assert _resolve_step_timeout(cc_rail, 90, False) == 150
    # Explicit caller budgets are only ever raised, never lowered.
    assert _resolve_step_timeout(plain, 300, True) == 300
    assert _resolve_step_timeout(cc_rail, 300, False) == 300


def test_60_1_claude_code_rail_declares_latency_profile():
    pytest.importorskip("backend.agents.claude_code_client")
    from backend.agents.claude_code_client import ClaudeCodeClient

    # The declared step budget must sit ABOVE the rail's own subprocess
    # timeout, or the step gives up while the CLI call is still in flight.
    client = ClaudeCodeClient("claude-sonnet-4-6")
    assert ClaudeCodeClient.recommended_step_timeout > client._timeout_s


def test_60_1_digest_renders_lite_full_markers():
    from backend.slack_bot.formatters import format_morning_digest

    reports = [
        {"ticker": "MU", "final_score": 7.0, "recommendation": "BUY", "analysis_path": "lite"},
        {"ticker": "NVDA", "final_score": 8.2, "recommendation": "BUY", "analysis_path": "full"},
        {"ticker": "OLD", "final_score": 6.0, "recommendation": "HOLD"},  # pre-tag row
    ]
    blocks = format_morning_digest({}, reports)
    section = next(
        b["text"]["text"] for b in blocks
        if b.get("type") == "section" and "Recent Analyses" in str(b.get("text", {}).get("text", ""))
    )
    lines = {ln.split("*")[1]: ln for ln in section.split("\n") if ln.startswith("•")}
    assert "`[lite]`" in lines["MU"]
    assert "`[full]`" in lines["NVDA"]
    # The pre-tag row renders unchanged -- no invented provenance.
    assert "[" not in lines["OLD"]
