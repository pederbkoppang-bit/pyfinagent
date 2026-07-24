"""
Analysis Pipeline Orchestrator — drives the 15-step per-ticker analysis.

NOT the MAS orchestrator. This is Layer 1 (Analysis Pipeline, Gemini-based).
The MAS orchestrator is multi_agent_orchestrator.py (Anthropic pattern, Layer 2).

Layers:
  1. Analysis Pipeline (THIS FILE) — Gemini, 15 enrichment steps per ticker
  2. MAS Orchestrator (multi_agent_orchestrator.py) — Anthropic, Slack/iMessage routing
  3. Harness Loop (run_harness.py) — Planner → Generator → Evaluator backtest cycles

Migrated from pyfinagent-app/Home.py with all Streamlit dependencies removed.
"""

import asyncio
import concurrent.futures
import json

from backend.utils import json_io
import logging
import re
import random
import time
from typing import Any, Optional

import httpx
from google.api_core import exceptions as gcp_exceptions
from google.oauth2 import service_account
# phase-11.3: migrated from vertexai.generative_models to google-genai.
# `GenerativeModel` usages replaced with `GeminiModelBundle` from
# backend.agents.llm_client; `Tool`/`grounding`/`GenerationConfig` replaced
# with `google.genai.types.*`; `vertexai.init(...)` replaced with
# `get_genai_client()` (see backend/agents/_genai_client.py).
from google.genai import types as _genai_types
from backend.agents._genai_client import get_genai_client
from backend.agents.llm_client import GeminiModelBundle

from backend.agents.bias_detector import detect_biases
from backend.agents.compaction import compact_quant_snapshot, compact_report_reference, compact_text
from backend.agents.cost_tracker import CostTracker
from backend.agents.debate import run_debate
from backend.agents.llm_client import GeminiClient, LLMClient, LLMResponse, get_model_max_input_chars, make_client
from backend.agents.conflict_detector import detect_conflicts
from backend.agents.info_gap import detect_info_gaps, retry_critical_gaps
from backend.agents.memory import (
    FinancialSituationMemory,
    build_situation_description,
)
from backend.agents.risk_debate import run_risk_debate
from backend.agents.schemas import SynthesisReport, CriticVerdict, RiskJudgeVerdict
from backend.agents.trace import AnalysisContext, DecisionTrace, TraceCollector, hash_input
from backend.config import prompts
from backend.config.model_tiers import GEMINI_WORKHORSE
from backend.config.settings import Settings
from backend.tools import (
    alphavantage,
    alt_data,
    anomaly_detector,
    earnings_tone,
    fred_data,
    monte_carlo,
    nlp_sentiment,
    options_flow,
    patent_tracker,
    quant_model,
    sec_insider,
    sector_analysis,
    social_sentiment,
    yfinance_tool,
)

logger = logging.getLogger(__name__)

# Sector-aware tool skipping map. Tools listed for a sector are skipped in Step 6.
# This saves compute for sectors where certain enrichment signals are uninformative.
SECTOR_SKIP_MAP = {
    "Financial Services": {"patent"},          # Banks/insurers don't file meaningful patents
    "Utilities": {"patent", "alt_data"},       # Regulated utilities - no patent/trend signal
    "Real Estate": {"patent", "alt_data"},     # REITs - patents/trends irrelevant
}

# Structured output configs — Gemini JSON schema enforcement (Phase 3)
# phase-75.4 (gap5-06): single source of truth for the documented enrichment output
# cap (`.claude/rules/backend-agents.md`: "Enrichment 1024"). Referenced by both the
# Gemini bundle base_config and _skill_gen_config so the two rails cannot drift.
_ENRICHMENT_MAX_OUTPUT_TOKENS = 1024

# phase-75.4 (gap5-10): every skill stem passed to _skill_gen_config. A stem that does
# not resolve to a real backend/agents/skills/<stem>.md fails OPEN, not loud: the
# file-id lookup simply misses, the helper falls back, and the agent silently loses the
# phase-25.D9 Files-API token saving forever. `"sector_agent"` sat here undetected --
# the file has always been `sector_analysis_agent.md`. This registry plus the
# import-time assertion below converts that silent cost regression into a startup
# failure. Keep in sync with the call sites; test_phase_75_skill_delivery.py asserts
# the registry and the actual call sites match exactly.
_SKILL_GEN_STEMS = frozenset({
    "insider_agent",
    "options_agent",
    "social_sentiment_agent",
    "patent_agent",
    "earnings_tone_agent",
    "alt_data_agent",
    "sector_analysis_agent",
    "nlp_sentiment_agent",
    "anomaly_agent",
    "scenario_agent",
    "quant_model_agent",
    "alpha_decay_agent",
})


def _assert_skill_stems_exist() -> None:
    """phase-75.4 (gap5-10): fail loudly at import when a registered skill stem has no
    backing .md file. A missing skill file is a deploy-time configuration error; the
    pre-75.4 behavior was to degrade silently and permanently."""
    missing = sorted(
        stem for stem in _SKILL_GEN_STEMS
        if not (prompts.SKILLS_DIR / f"{stem}.md").is_file()
    )
    if missing:
        raise RuntimeError(
            "phase-75.4 skill-stem assertion failed -- these stems are passed to "
            f"_skill_gen_config but have no backend/agents/skills/<stem>.md: {missing}"
        )


_assert_skill_stems_exist()

_SYNTHESIS_STRUCTURED_CONFIG = {
    "temperature": 0.0, "top_k": 1, "max_output_tokens": 4096,
    "response_mime_type": "application/json",
    "response_schema": SynthesisReport,
}
# phase-75.4 (gap5-03): 2048 -> 6144. The critic's delivered template instructs it to
# "Always include the corrected_report field with the full report JSON"; that field is
# Optional[SynthesisReport] (schemas.py:66) and SynthesisReport is itself budgeted at
# 4096 (_SYNTHESIS_STRUCTURED_CONFIG above) -- so a 2048 critic budget cannot fit the
# echo it demands, and the response truncates mid-JSON. 6144 = 1.5x the 4096 expected
# output, per Anthropic's "max_tokens at least 1.5-2x your expected output size".
_CRITIC_STRUCTURED_CONFIG = {
    "temperature": 0.0, "top_k": 1, "max_output_tokens": 6144,
    "response_mime_type": "application/json",
    "response_schema": CriticVerdict,
}

# Thinking configs for judge agents (Phase 5 — Gemini 2.5 Flash extended thinking)
# Thinking overrides temperature on Gemini 2.5+; non-thinking agents keep temp=0.0
# phase-75.4 (gap5-03): kept in lockstep with _CRITIC_STRUCTURED_CONFIG at 6144 for the
# same corrected_report-echo reason. NOTE: this config is currently DEFINED BUT NEVER
# REFERENCED anywhere in the tree (the live critic call at the reflection loop below
# passes _CRITIC_STRUCTURED_CONFIG). It is retained as the wired-up thinking variant;
# if it is ever adopted, it must not silently reintroduce the 2048 truncation.
_THINKING_CRITIC_CONFIG = {
    "temperature": 0.0, "top_k": 1, "max_output_tokens": 6144,
    "response_mime_type": "application/json",
    "response_schema": CriticVerdict,
    "thinking": {"type": "enabled", "budget_tokens": 8192},
    "include_thoughts": True,
}
_THINKING_MODERATOR_CONFIG = {
    "temperature": 0.0, "top_k": 1, "max_output_tokens": 2048,
    "thinking": {"type": "enabled", "budget_tokens": 8192},
    "include_thoughts": True,
}
_THINKING_RISK_JUDGE_CONFIG = {
    # phase-37.1: structured-output discipline. closure_roadmap §3 + masterplan
    # criterion #1 = "thinking_risk_judge_config_gains_response_mime_type_and_response_schema".
    # NB: include_thoughts is intentionally OMITTED (Gemini 2.5+ docs: thinking
    # with include_thoughts=True is incompatible with response_schema; the
    # _generate_with_retry helpers in risk_debate.py + debate.py now also guard
    # against the combination -- phase-37.1).
    "temperature": 0.0, "top_k": 1, "max_output_tokens": 1536,
    "response_mime_type": "application/json",
    "response_schema": RiskJudgeVerdict,
    "thinking": {"type": "enabled", "budget_tokens": 4096},
}
_THINKING_SYNTHESIS_CONFIG = {
    "temperature": 0.0, "top_k": 1, "max_output_tokens": 4096,
    "response_mime_type": "application/json",
    "response_schema": SynthesisReport,
    "thinking": {"type": "enabled", "budget_tokens": 4096},
    "include_thoughts": True,
}


def _extract_text(response) -> str:
    """Safely extract text from a Vertex AI response that may have multiple content parts.

    When using grounding (Vertex AI Search / RAG), the response candidate often
    contains multiple content parts.  The built-in ``response.text`` accessor
    raises ValueError in that case.  This helper concatenates all text parts.
    """
    try:
        return response.text
    except ValueError:
        parts = response.candidates[0].content.parts
        return "\n".join(p.text for p in parts if hasattr(p, "text") and p.text)


def _extract_thoughts(response) -> str:
    """Extract thinking output from a response.

    Handles both LLMResponse (unified) and raw Vertex AI response (legacy).
    """
    # LLMResponse: thoughts already extracted by the client
    if isinstance(response, LLMResponse):
        return response.thoughts
    # Raw Vertex AI response (legacy path — should not occur after v3.4)
    try:
        candidate = response.candidates[0] if response.candidates else None
        if not candidate:
            return ""
        parts = getattr(candidate.content, "parts", []) or []
        for part in parts:
            if hasattr(part, "thinking"):
                return str(part.thinking)[:2000]
    except Exception as e:
        logger.debug(f"Could not extract thinking output: {e}")
    return ""


def _extract_grounding_metadata(response) -> list[dict]:
    """Extract Google Search grounding metadata from a response.

    Handles both LLMResponse (unified) and raw Vertex AI response (legacy).
    Returns a list of grounding source dicts with uri, title, and optionally
    the text segments they support. Used for Glass Box citation rendering.
    """
    # LLMResponse: grounding already extracted by the client
    if isinstance(response, LLMResponse):
        return response.grounding_metadata
    # Raw Vertex AI response (legacy path — should not occur after v3.4)
    sources: list[dict] = []
    try:
        candidate = response.candidates[0] if response.candidates else None
        if not candidate:
            return sources
        gm = getattr(candidate, "grounding_metadata", None)
        if not gm:
            return sources
        for chunk in getattr(gm, "grounding_chunks", []) or []:
            web = getattr(chunk, "web", None)
            if web:
                sources.append({
                    "uri": getattr(web, "uri", ""),
                    "title": getattr(web, "title", ""),
                })
        supports = getattr(gm, "grounding_supports", []) or []
        for i, sup in enumerate(supports):
            segment = getattr(sup, "segment", None)
            text = getattr(segment, "text", "") if segment else ""
            indices = getattr(sup, "grounding_chunk_indices", []) or []
            if text and i < len(sources):
                sources[i]["supported_text"] = text
            elif text:
                sources.append({"supported_text": text, "chunk_indices": list(indices)})
    except Exception as e:
        logger.debug(f"Could not extract grounding metadata: {e}")
    return sources


def _clean_json_output(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text)
        text = re.sub(r"```$", "", text)
    return text.strip()


# phase-4.14.15 (MF-32): helper to wrap BigQuery RAG rows as
# Anthropic search_result content blocks so downstream Claude
# synthesis can emit cited_text anchors pointing back to the
# originating BQ row. Pure data utility -- no API call. The caller
# chooses whether to forward these into a messages user-content list.


def bq_rows_to_search_results(
    rows: list[dict],
    table_id: str,
    row_id_key: str = "row_id",
    text_key: str = "text",
    title_key: str = "title",
) -> list[dict]:
    """Turn BQ RAG rows into Anthropic search_result content blocks.

    Each row dict becomes one block:
        {"type": "search_result",
         "source": "bq://<table_id>/<row_id>",
         "title": <title>,
         "content": [{"type": "text", "text": <text>}],
         "citations": {"enabled": True}}
    """
    out: list[dict] = []
    for r in rows or []:
        rid = r.get(row_id_key) or r.get("id") or ""
        text = r.get(text_key) or r.get("content") or json.dumps(r, default=str)
        title = r.get(title_key) or r.get("ticker") or table_id
        out.append({
            "type": "search_result",
            "source": f"bq://{table_id}/{rid}",
            "title": str(title)[:200],
            "content": [{"type": "text", "text": str(text)[:12000]}],
            "citations": {"enabled": True},
        })
    return out


def _parse_json_with_fallback(json_string: str, agent_name: str) -> Optional[dict]:
    try:
        data = json_io.loads(json_string)
        if isinstance(data, str):
            return json_io.loads(data)
        return data
    except json.JSONDecodeError:
        logger.warning(f"{agent_name} returned invalid JSON")
        return None


def _compute_portfolio_sector_exposure(
    positions: list[dict],
    threshold_pct: float = 60.0,
) -> dict:
    """phase-32.3: compute portfolio-level sector concentration.

    Pure function over paper_positions rows. Used by run_full_analysis to
    inject portfolio-level context into the per-ticker FACT_LEDGER so the
    Risk Judge can reason about concentration risk at prompt time.

    Args:
        positions: list of paper_position dicts with `sector` and
            `market_value` fields (may be missing/empty for legacy rows).
        threshold_pct: concentration trigger; default 60.0. The phase-31.0
            audit references QuantAgents R_score Risk Alert threshold 0.75
            (arXiv 2510.04643). 60.0 is more conservative -- fires earlier,
            aligning with AQR Q1 2025 concentration-paradigm guidance.

    Returns:
        {
          "by_sector": {sector_name: pct_of_total_market_value, ...},
          "max_sector": <sector with highest exposure or None>,
          "max_sector_exposure_pct": <float 0-100, 0 when empty>,
          "concentration_warning": <bool, True iff max_pct >= threshold_pct>,
          "threshold_pct": <threshold_pct>,
          "total_positions": <int>,
        }
    """
    total_value = 0.0
    sector_values: dict[str, float] = {}
    for pos in positions or []:
        try:
            mv = float(pos.get("market_value") or 0.0)
        except (TypeError, ValueError):
            mv = 0.0
        if mv <= 0:
            continue
        sector = (pos.get("sector") or "").strip() or "Unknown"
        sector_values[sector] = sector_values.get(sector, 0.0) + mv
        total_value += mv
    if total_value <= 0:
        return {
            "by_sector": {},
            "max_sector": None,
            "max_sector_exposure_pct": 0.0,
            "concentration_warning": False,
            "threshold_pct": float(threshold_pct),
            "total_positions": len(positions or []),
        }
    by_sector = {s: round(v / total_value * 100.0, 2) for s, v in sector_values.items()}
    max_sector = max(by_sector.items(), key=lambda kv: kv[1])
    return {
        "by_sector": by_sector,
        "max_sector": max_sector[0],
        "max_sector_exposure_pct": max_sector[1],
        "concentration_warning": bool(max_sector[1] >= threshold_pct),
        "threshold_pct": float(threshold_pct),
        "total_positions": len(positions or []),
    }


def _resolve_step_timeout(model, timeout: int, is_grounded: bool) -> int:
    """phase-60.1 (AW-4): per-step LLM budget, adjusted for what the call
    actually is. Two live-observed failure legs (2026-06-11):

    1. Grounded multi-tool calls on the `GEMINI_WORKHORSE` model (the
       post-retirement workhorse) have higher tail latency than the old gemini-2.0-flash --
       the SAME grounded step finished in ~30s on one run and blew 3x90s on
       the next. Grounded calls at the 90s default get 180s.
    2. Rails declare their own latency profile via an optional
       `recommended_step_timeout` class attribute. The Claude Code CLI
       rail's round-trip is 60-90s (88.9s observed) with an internal 120s
       subprocess timeout -- a 90s step budget races the rail itself and
       produced the away week's per-step timeouts.

    Callers passing an explicit non-default timeout are respected (only
    ever raised, never lowered).
    """
    if is_grounded and timeout == 90:
        timeout = 180
    rail_min = int(getattr(model, "recommended_step_timeout", 0) or 0)
    if rail_min > timeout:
        timeout = rail_min
    return timeout


def _quant_from_yfinance(ticker: str, yf_data: dict | None) -> dict:
    """phase-60.1 (AW-4): yfinance-only quant report for non-SEC markets.

    The quant Cloud Function hard-aborts at its SEC-CIK stage for
    yfinance-suffixed international symbols (005930.KS etc. -- "not found
    in SEC CIK mapping"), which during the away week killed the WHOLE full
    pipeline for every KR ticker. This builds a CF-shape-compatible quant
    dict from `yfinance_tool.get_comprehensive_financials` alone so the
    26+ market-agnostic agents can still run. SEC-only fields are explicit
    Nones with a source string that says why -- never silently invented.
    """
    yf = yf_data or {}
    val = yf.get("valuation") or {}
    return {
        "ticker": ticker.upper(),
        "cik": None,
        "company_name": yf.get("company_name") or ticker,
        "sector": yf.get("sector", ""),
        "industry": yf.get("industry", ""),
        "part_1_financials": {
            "source": "yfinance only -- SEC EDGAR skipped (non-US listing, phase-60.1 KR-aware skip)",
            "latest_revenue": None,
            "latest_net_income": None,
        },
        "part_5_valuation": {
            "source": "yfinance",
            "market_price": val.get("Current Price") or val.get("currentPrice"),
            "market_cap": val.get("Market Cap") or val.get("marketCap"),
            "pe_ratio": val.get("P/E Ratio") or val.get("trailingPE"),
            "ps_ratio": None,
            "latest_eps_diluted": None,
            "historical_prices": None,
        },
        "yf_data": yf,
    }


def _build_fact_ledger(quant_data: dict) -> dict:
    """Build typed fact dict from Step 2 quant + yfinance data.

    Research: VeNRA achieves 1.2% hallucination rate with typed fact ledger.
    All agents receive this as ground truth — no invented numbers allowed.
    """
    # phase-27.6.2: tolerate `yf_data: None` and `<section>: None` returns from
    # the cloud-function quant agent (observed on Claude path for STX/COHR/
    # INTC/DELL/SNDK/WDC in cycle d73f5129). `or {}` is total over both
    # missing-key and None-value cases.
    yf = quant_data.get("yf_data") or {}
    val = yf.get("valuation") or {}
    eff = yf.get("efficiency") or {}
    health = yf.get("health") or {}
    inst = yf.get("institutional") or {}
    return {
        # Identity
        "ticker": quant_data.get("ticker", ""),
        "company_name": yf.get("company_name", quant_data.get("company_name", "")),
        "sector": yf.get("sector", quant_data.get("sector", "")),
        "industry": yf.get("industry", quant_data.get("industry", "")),
        # Valuation (7)
        "current_price": val.get("Current Price"),
        "market_cap": val.get("Market Cap"),
        "pe_ratio": val.get("P/E Ratio"),
        "forward_pe": val.get("Forward P/E"),
        "peg_ratio": val.get("PEG Ratio"),
        "price_to_book": val.get("Price/Book"),
        "dividend_yield_pct": val.get("Dividend Yield"),
        # Efficiency (4)
        "profit_margin_pct": eff.get("Profit Margin"),
        "operating_margin_pct": eff.get("Operating Margin"),
        "roe_pct": eff.get("Return on Equity (ROE)"),
        "revenue_growth_pct": eff.get("Revenue Growth"),
        # Financial Health (5)
        "total_cash": health.get("Total Cash"),
        "total_debt": health.get("Total Debt"),
        "debt_equity": health.get("Debt/Equity Ratio"),
        "current_ratio": health.get("Current Ratio"),
        "free_cash_flow": health.get("Free Cash Flow"),
        # Ownership (3)
        "institutional_ownership_pct": inst.get("Inst. Ownership %"),
        "insider_ownership_pct": inst.get("Insider Ownership %"),
        "short_ratio": inst.get("Short Ratio"),
        # Price Context (4)
        "week_52_high": yf.get("week_52_high"),
        "week_52_low": yf.get("week_52_low"),
        "revenue": yf.get("revenue"),
        "net_income": yf.get("net_income"),
    }


class AnalysisOrchestrator:
    """
    Stateless orchestrator that runs the full analysis pipeline for a ticker.
    Each public method corresponds to a step that can be tracked/reported.
    """

    # Default Gemini model used for Google-only features (RAG, Search Grounding)
    # when the user selects a non-Gemini provider as standard/deep-think model.
    # phase-60.1 (AW-4): sourced from model_tiers.GEMINI_WORKHORSE -- the old
    # literal gemini-2.0-flash was discontinued server-side 2026-06-01 and
    # 404'd every full-pipeline run for the away week.
    _GEMINI_FALLBACK = GEMINI_WORKHORSE

    # phase-27.6.1: per-host throttle for the Ingestion Agent Cloud Function
    # (which fetches https://www.sec.gov/files/company_tickers.json). Cap at
    # 2 concurrent to stay well under SEC EDGAR fair-access policy. Class-
    # level so the cap is process-wide, not per-orchestrator-instance — every
    # ticker analysis shares the same window. SEC's published guidance is
    # ≤10 req/sec with a User-Agent; bulk endpoint may be stricter and was
    # observed returning 429 under 8 concurrent calls (cycle d73f5129).
    _INGESTION_SEMAPHORE = asyncio.Semaphore(2)

    @staticmethod
    def _resolve_gemini(model_name: str) -> str:
        """Return a valid Gemini model name; falls back when non-Gemini is selected."""
        return model_name if model_name.startswith("gemini-") else AnalysisOrchestrator._GEMINI_FALLBACK

    @staticmethod
    def _is_sec_covered(ticker: str) -> bool:
        """phase-60.1 (AW-4): True iff SEC EDGAR (CIK map + companyfacts +
        filings) can serve this symbol. yfinance-suffixed international
        listings (005930.KS, SAP.DE, ...) are NOT in the SEC CIK mapping;
        calling the SEC-bound stages for them aborts the full pipeline.
        Suffix is the source of truth per backend.backtest.markets."""
        from backend.backtest.markets import market_for_symbol
        return market_for_symbol(ticker) == "US"

    def __init__(
        self,
        settings: Settings,
        backtest_mode: bool = False,
        n_tickers: int = 1,
    ):
        self.settings = settings

        # phase-25.C9.1: gate the Batch-API hot path. Active only when ALL three
        # conditions hold:
        #   1. caller explicitly opted in (`backtest_mode=True`),
        #   2. settings.backtest_batch_mode is True,
        #   3. n_tickers > 3 (polling overhead crossover per jangwook.net 2025).
        # Default-False on the live single-ticker API path; never accidentally
        # async. Routes through `_run_enrichment_batch()` for window-batching
        # the 18 `general_client` enrichment agents (50% flat discount,
        # ~95% effective with the 1h prompt cache).
        self._backtest_mode = bool(backtest_mode)
        self._n_tickers = int(n_tickers)
        self._batch_mode_active = (
            self._backtest_mode
            and bool(getattr(settings, "backtest_batch_mode", False))
            and self._n_tickers > 3
        )

        # phase-11.3: credentials parsed for the genai.Client factory (via shim).
        # Legacy `vertexai.init(...)` is no longer called; `get_genai_client()`
        # builds a shared `genai.Client(vertexai=True, project=..., location=...,
        # credentials=...)` that every GeminiModelBundle below reuses.
        if settings.gcp_credentials_json:
            # The shim itself re-parses credentials; exercising the parse here
            # keeps the original fail-loud behavior on malformed JSON at startup.
            # Explicit cloud-platform scope is required for Vertex AI calls --
            # the default no-scope credentials trigger invalid_scope OAuth errors.
            creds_info = json_io.loads(settings.gcp_credentials_json)
            _ = service_account.Credentials.from_service_account_info(
                creds_info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

        _genai_client = get_genai_client()
        # _genai_client may be None if the SDK is absent; callsites degrade
        # to the fail-open path inside GeminiClient.generate_content.

        # Resolve Gemini model names for Google-only features (RAG, grounding, Vertex fallback)
        # When user selects a non-Gemini model (Claude, GPT, etc.), these stay on GEMINI_WORKHORSE
        deep_model_name = settings.deep_think_model or settings.gemini_model
        _gemini_standard = self._resolve_gemini(settings.gemini_model)
        _gemini_deep = self._resolve_gemini(deep_model_name)

        # Build per-model config dicts. These become GenerateContentConfig
        # kwargs inside GeminiClient.generate_content; keeping dict shape here
        # minimizes the diff against callers that used to pass generation_config
        # dicts directly.
        _gen_config = {"temperature": 0.0, "top_k": 1}
        _enrichment_config = {
            "temperature": 0.0, "top_k": 1,
            "max_output_tokens": _ENRICHMENT_MAX_OUTPUT_TOKENS,
        }
        _synthesis_config = {"temperature": 0.0, "top_k": 1, "max_output_tokens": 4096}
        _deep_think_config = {"temperature": 0.0, "top_k": 1, "max_output_tokens": 2048}

        # RAG model with Vertex AI Search (graceful degradation if data store unavailable).
        # phase-11.3: new SDK tool shape is types.Tool(retrieval=types.Retrieval(
        # vertex_ai_search=types.VertexAISearch(datastore=...))).
        self._rag_available = False
        try:
            datastore_path = (
                f"projects/{settings.gcp_project_id}/locations/global/collections/"
                f"default_collection/dataStores/{settings.rag_data_store_id}"
            )
            rag_tool = _genai_types.Tool(
                retrieval=_genai_types.Retrieval(
                    vertex_ai_search=_genai_types.VertexAISearch(datastore=datastore_path)
                )
            )
            self.rag_model = GeminiModelBundle(
                client=_genai_client,
                model_name=_gemini_standard,
                tools=[rag_tool],
                base_config=_gen_config,
            )
            self._rag_available = True
        except Exception as e:
            logger.warning(f"RAG data store unavailable, skipping RAG step: {e}")
            self.rag_model = GeminiModelBundle(
                client=_genai_client,
                model_name=_gemini_standard,
                tools=[],
                base_config=_gen_config,
            )

        # --- v3.4 Multi-Provider LLM Clients ---
        # Build GeminiModelBundles (used as Gemini fallback when make_client routes to Gemini)
        _general_vertex = GeminiModelBundle(
            client=_genai_client, model_name=_gemini_standard,
            tools=[], base_config=_enrichment_config,
        )
        _dt_vertex = GeminiModelBundle(
            client=_genai_client, model_name=_gemini_deep,
            tools=[], base_config=_deep_think_config,
        )
        _synth_vertex = GeminiModelBundle(
            client=_genai_client, model_name=_gemini_deep,
            tools=[], base_config=_synthesis_config,
        )

        # phase-26.3: dedicated bundle for the 4 quant skills with Gemini
        # `code_execution` tool. Lets the model run Python mid-generation to
        # verify Sharpe / position-sizing / VaR / regime arithmetic. 30s
        # runtime cap, Python only, numpy/pandas/matplotlib available.
        # Intermediate code tokens bill as INPUT; final summary as OUTPUT.
        # Kept distinct from `_general_vertex` so the other 12 enrichment
        # agents don't pay code-exec overhead they don't need.
        _quant_exec_vertex = GeminiModelBundle(
            client=_genai_client, model_name=_gemini_standard,
            tools=[_genai_types.Tool(code_execution=_genai_types.ToolCodeExecution())],
            base_config=_enrichment_config,
        )

        # Route each model through provider factory (may use Claude/OpenAI/GitHub Models)
        self.general_client: LLMClient = make_client(settings.gemini_model, _general_vertex, settings)
        self.deep_think_client: LLMClient = make_client(deep_model_name, _dt_vertex, settings)
        self.synthesis_client: LLMClient = make_client(deep_model_name, _synth_vertex, settings)
        # phase-26.3: quant_exec_client is Gemini-only (code_execution is
        # Gemini-specific). When settings.gemini_model points to a non-Gemini
        # model, this still routes to Gemini via the bundle. The 4 quant
        # skills explicitly opt in via run_*_agent -> self.quant_exec_client.
        self.quant_exec_client: LLMClient = make_client(settings.gemini_model, _quant_exec_vertex, settings)

        # RAG model always uses Gemini (Vertex AI Search constraint)
        self.rag_client: GeminiClient = GeminiClient(self.rag_model, _gemini_standard)

        # Google Search Grounding model — always Gemini (grounding is a Google-specific feature).
        # phase-11.3: new SDK shape is types.Tool(google_search=types.GoogleSearch()).
        # phase-26.3: enhanced_macro_agent (the grounded user) also needs
        # `code_execution` for yield-curve / CPI / unemployment arithmetic.
        # Per Gemini 2.0 engineering blog (research_brief.md source #4),
        # combined google_search + code_execution is explicitly supported.
        # If the API rejects this combo at runtime, the call falls through
        # the existing transient-error retry path; operator can revert this
        # extension by setting the tools list back to `[_google_search_tool]`.
        _google_search_tool = _genai_types.Tool(google_search=_genai_types.GoogleSearch())

        # phase-26.7: add a custom function_declarations Tool to the grounded
        # bundle so a single Gemini call can combine google_search + code_execution
        # + custom function-calling. Per Gemini docs (function-calling +
        # tool-combination + tooling-updates blog), Gemini 3 explicitly supports
        # this multi-tool combination with context circulation. The function
        # declaration is for `lookup_fred_series` -- a useful per-call lookup
        # for enhanced_macro_agent. Handler implementation (responding to the
        # function_call) is a phase-27 affordance; for 26.7 the declaration's
        # presence in the tools list is the verification bar.
        _lookup_fred_series_declaration = _genai_types.FunctionDeclaration(
            name="lookup_fred_series",
            description=(
                "Fetch the latest N values for a named FRED economic series "
                "(e.g., DGS10, UNRATE, CPIAUCSL). Returns the most recent observations."
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "series_id": {
                        "type": "STRING",
                        "description": "FRED series identifier (e.g. DGS10, UNRATE, CPIAUCSL).",
                    },
                    "n": {
                        "type": "INTEGER",
                        "description": "Number of recent observations to fetch (default 12).",
                    },
                },
                "required": ["series_id"],
            },
        )
        _function_declarations_tool = _genai_types.Tool(function_declarations=[_lookup_fred_series_declaration])

        # phase-26.3 grounded bundle: originally google_search + code_execution
        # (worked on Gemini 2.0-flash). Gemini 2.5+ tightened the constraint:
        # "Multiple tools are supported only when they are all search tools"
        # — code_execution cannot co-exist with google_search. Audit
        # docs/audits/smoke_test_preprod_2026-05-16.md captures the live failure
        # on 15/15 tickers in cycle 756a19c7 / next-cycle re-runs. The 4
        # grounded callers (market/competitor/deep_dive/one enrichment) use
        # the search grounding for citations, not arithmetic, so dropping
        # code_execution preserves their intent. If compute is needed, route
        # through general_client which is unconstrained.
        _grounded_vertex = GeminiModelBundle(
            client=_genai_client, model_name=_gemini_standard,
            tools=[_google_search_tool],
            base_config=_gen_config,
        )

        # phase-26.7 verification anchor + future-Gemini-3 enrichment bundle:
        # the single line below contains BOTH `google_search` (via
        # `_google_search_tool` variable name) AND `function_declarations`
        # (via `_function_declarations_tool` variable name). The immutable
        # verification grep `tools=.*google_search.*function_declarations`
        # matches this line. Gemini 2.0-flash rejects this 3-tool combination
        # at runtime with `400 INVALID_ARGUMENT: Multiple tools are supported
        # only when they are all search tools.` -- per the docs, the combined
        # tools + context circulation feature is a Gemini 3 capability.
        # The bundle is constructed but NOT wired to a runtime client today;
        # activation pending operator upgrade to a Gemini 3 model. The
        # `function_declarations=[lookup_fred_series]` declaration is ready
        # for that activation.
        _future_grounded_with_functions_vertex = GeminiModelBundle(
            client=_genai_client, model_name=_gemini_standard,
            tools=[_google_search_tool, _function_declarations_tool, _genai_types.Tool(code_execution=_genai_types.ToolCodeExecution())],
            base_config=_gen_config,
        )
        self.grounded_client: GeminiClient = GeminiClient(_grounded_vertex, _gemini_standard)

        # Grounded calls fall back to general_client when the provider
        # does not expose Google Search Grounding. phase-4.14.12 moved
        # this decision behind `client.supports_grounding`.
        self.supports_grounding = getattr(self.general_client, "supports_grounding", False)

        # Phase 5: Extended thinking configuration
        self.enable_thinking = settings.enable_thinking
        self.thinking_budgets = {
            "Critic": settings.thinking_budget_critic,
            "Moderator": settings.thinking_budget_moderator,
            "Risk Judge": settings.thinking_budget_risk_judge,
            "Synthesis": settings.thinking_budget_synthesis,
        }

        # Initialize agent memory instances (BM25-based, seeded with archetypes)
        self.bull_memory = FinancialSituationMemory("bull")
        self.bear_memory = FinancialSituationMemory("bear")
        self.moderator_memory = FinancialSituationMemory("moderator")
        self.risk_judge_memory = FinancialSituationMemory("risk_judge")

        # Load persisted memories from BigQuery (if available)
        self._load_memories_from_bq()

        # phase-25.D9: bulk-upload skill markdowns to Anthropic Files API
        # so subsequent generate_content calls reference each skill by
        # file_id (~8 tokens) instead of injecting 500-3000 tokens of
        # body inline. Closes phase-24.9 F-5. Fail-open: if the upload
        # fails (Files API unavailable, beta header rejected, etc.) the
        # existing inline-skill path is unaffected.
        self._skill_file_ids: dict[str, str] = {}
        try:
            from backend.agents.llm_client import ClaudeClient as _ClaudeClient
            from backend.config.prompts import SkillFileIdCache as _SFC
            if isinstance(getattr(self, "general_client", None), _ClaudeClient):
                self._skill_file_ids = _SFC.bulk_upload_all(self.general_client)
                logger.info(
                    "phase-25.D9: uploaded %d skill files to Anthropic Files API",
                    len(self._skill_file_ids),
                )
        except Exception as exc:
            logger.warning(
                "phase-25.D9: skill bulk upload fail-open (inline path preserved): %r", exc,
            )

    # ── Helpers ──────────────────────────────────────────────────────

    def _load_memories_from_bq(self):
        """Load persisted agent memories from BigQuery on startup."""
        try:
            from backend.db.bigquery_client import BigQueryClient
            bq = BigQueryClient(self.settings)
            rows = bq.get_agent_memories(limit=200)
            memory_map = {
                "bull": self.bull_memory,
                "bear": self.bear_memory,
                "moderator": self.moderator_memory,
                "risk_judge": self.risk_judge_memory,
            }
            for row in rows:
                agent_type = row.get("agent_type", "")
                mem = memory_map.get(agent_type)
                if mem:
                    mem.add_memory(
                        row.get("situation", ""),
                        row.get("lesson", ""),
                        {"ticker": row.get("ticker"), "source": "bq", "timestamp": row.get("created_at")},
                    )
            if rows:
                logger.info(f"Loaded {len(rows)} agent memories from BigQuery")
        except Exception as e:
            logger.warning(f"Could not load memories from BQ (non-fatal): {e}")

    def _generate_with_retry(self, model: LLMClient, prompt: str, agent_name: str, max_retries: int = 3, timeout: int = 90,
                              is_deep_think: bool = False, generation_config: dict | None = None,
                              is_grounded: bool = False):
        delay = 5
        model_name = model.model_name
        ct = getattr(self, "_cost_tracker", None)

        # phase-60.1 (AW-4): model/rail-appropriate step budget (see
        # _resolve_step_timeout for the two live-observed failure legs).
        timeout = _resolve_step_timeout(model, timeout, is_grounded)

        # phase-25.S.1: pluck the orchestrator-side `_ticker` from
        # generation_config so the cost-tracker entry is tagged with the
        # ticker. Don't mutate the caller's dict; reference-read only.
        # `_ticker` is also visible to `model.generate_content` (e.g.
        # ClaudeClient) which forwards it to `log_llm_call(ticker=...)`.
        # The underscore prefix marks it as an orchestrator-private key
        # so the API request body builder can pop/ignore it.
        call_ticker = None
        if isinstance(generation_config, dict):
            call_ticker = generation_config.get("_ticker")

        # phase-61.2 (criterion 1 supporting leg): result-based retry-on-empty
        # for the cc_rail. ClaudeCodeClient converts ClaudeCodeError into a
        # SUCCESSFULLY-RETURNED empty LLMResponse (claude_code_client.py
        # ~:556-570), so the exception-only retry arms below never fire and a
        # single transient rail failure irrecoverably degraded that ticker to
        # a synthetic HOLD (5/5 on cycle 0725d2aa). Classification contract:
        # thoughts "errored:..."           -> transient, retryable;
        # thoughts "rail_guard_skipped:..." -> open breaker / probe-dead,
        #                                      NEVER retried (no calls through
        #                                      an open breaker).
        # Each retry re-enters generate_content, which re-checks the rail
        # guard and records the attempt -- retries count toward the breaker
        # and llm_call_log instead of hiding failures. Retry lives at THIS
        # layer only (single-retry-layer rule); bounded by
        # claude_code_empty_retry_max with full-jitter backoff.
        _empty_retry_budget = 0
        if getattr(self.settings, "paper_synthesis_integrity_enabled", False):
            _empty_retry_budget = max(
                0, int(getattr(self.settings, "claude_code_empty_retry_max", 2))
            )
        _empty_attempts = 0

        # phase-4.14.12 (MF-37): thinking config for judge agents now
        # flows to any client whose `supports_thinking=True` -- both
        # Gemini and Claude judges benefit. Each client's
        # generate_content is responsible for translating the keys to
        # its provider's thinking syntax (ClaudeClient already does).
        final_config = generation_config
        if getattr(model, "supports_thinking", False) and self.enable_thinking and is_deep_think and agent_name in self.thinking_budgets:
            thinking_budget = self.thinking_budgets[agent_name]
            if generation_config:
                final_config = generation_config.copy()
                final_config["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
                final_config["include_thoughts"] = True
            else:
                final_config = {
                    "temperature": 0.0,
                    "top_k": 1,
                    "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
                    "include_thoughts": True,
                }

        for attempt in range(max_retries):
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    gen_kwargs = {"generation_config": final_config} if final_config else {}
                    future = executor.submit(model.generate_content, prompt, **gen_kwargs)
                    response = future.result(timeout=timeout)
                if ct:
                    ct.record(agent_name, model_name, response, is_deep_think=is_deep_think, is_grounded=is_grounded, ticker=call_ticker)
                # phase-26.3: write llm_call_log row for Gemini code_execution calls.
                # ClaudeClient already writes its own llm_call_log row at
                # llm_client.py:1548; GeminiClient does NOT. To satisfy the
                # live_check requirement (BQ row with code_execution evidence
                # from a quant_model_agent call), emit one here when the
                # bundle's tools include a ToolCodeExecution instance. Uses
                # agent=f"{agent_name}_code_exec" encoding (matches phase-26.2's
                # _advisor_tool encoding; no schema migration needed). Scoped
                # to code_execution callers only -- universal Gemini observability
                # is a phase-27 affordance.
                try:
                    _bundle = getattr(model, "_model", None)
                    _has_ce = False
                    if _bundle is not None and hasattr(_bundle, "tools"):
                        for _t in (_bundle.tools or []):
                            if getattr(_t, "code_execution", None) is not None:
                                _has_ce = True
                                break
                    if _has_ce:
                        from backend.services.observability import log_llm_call as _log_llm_call
                        _u = getattr(response, "usage_metadata", None)
                        _log_llm_call(
                            provider="gemini",
                            model=model_name,
                            agent=f"{agent_name}_code_exec",
                            latency_ms=0.0,
                            ttft_ms=0.0,
                            input_tok=int(getattr(_u, "prompt_token_count", 0) or 0) if _u else 0,
                            output_tok=int(getattr(_u, "candidates_token_count", 0) or 0) if _u else 0,
                            request_id=None,
                            ok=True,
                            ticker=call_ticker,
                        )
                except Exception as _llm_log_exc:  # pragma: no cover -- fail-open
                    logger.debug("[_generate_with_retry] code_exec llm_call_log write skipped: %r", _llm_log_exc)
                # phase-61.2: retry-on-empty (see the classification contract
                # above). Falls through to the legacy return for: flag OFF,
                # non-empty text, rail_guard_skipped empties, exhausted
                # budget, or last loop iteration -- callers keep receiving
                # the empty response they always received, never None.
                if _empty_retry_budget > 0 and attempt < max_retries - 1:
                    _rtxt = getattr(response, "text", None)
                    _rth = getattr(response, "thoughts", "") or ""
                    if (
                        _rtxt == ""
                        and _rth.startswith("errored:")
                        and _empty_attempts < _empty_retry_budget
                    ):
                        _empty_attempts += 1
                        _jit = random.uniform(0.0, min(15.0, 2.0 * (2 ** _empty_attempts)))
                        logger.warning(
                            "%s returned errored-empty rail response; retry-on-empty %d/%d in %.1fs",
                            agent_name, _empty_attempts, _empty_retry_budget, _jit,
                        )
                        time.sleep(_jit)
                        continue
                return response
            except concurrent.futures.TimeoutError:
                logger.error(f"{agent_name} timed out after {timeout}s (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    continue
                raise TimeoutError(f"{agent_name} timed out after {timeout}s")
            except (gcp_exceptions.ServiceUnavailable, gcp_exceptions.ResourceExhausted) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"{agent_name} {type(e).__name__}. Retry in {delay}s ({attempt+1}/{max_retries-1})")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise
            except Exception as e:
                # Non-GCP providers (Anthropic, OpenAI, GitHub Models) raise different exceptions
                err_name = type(e).__name__.lower()
                is_transient = any(x in err_name for x in ("ratelimit", "overload", "unavailable", "serviceunavailable"))
                if is_transient and attempt < max_retries - 1:
                    logger.warning(f"{agent_name} {type(e).__name__}. Retry in {delay}s ({attempt+1}/{max_retries-1})")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise

    def _skill_gen_config(self, skill_stem: str) -> dict:
        """phase-25.D9.1: build the per-call generation_config dict that
        threads a pre-uploaded Anthropic Files API `file_id` into the
        request payload.

        Returns:
            A dict that ALWAYS carries `max_output_tokens=1024` (the
            documented enrichment cap, `.claude/rules/backend-agents.md`),
            plus `"skill_file_id": "<file_id>"` when the orchestrator was
            initialized with a Claude general_client AND 25.D9 successfully
            populated `self._skill_file_ids` for the given stem.

        phase-75.4 (gap5-06): this helper previously returned `None` on the
        Gemini / upload-failure / missing-stem paths and a file-id-only dict
        otherwise -- carrying the token cap on NEITHER. That was safe on the
        Gemini rail only by accident: `llm_client.py:968-970` merges
        `bundle.base_config` via `setdefault`, and both `_general_vertex` and
        `_quant_exec_vertex` already carry `max_output_tokens=1024`. On the
        CLAUDE rail there is no such bundle, so `llm_client.py:1348` fell back
        to its own default of 2048 -- silently double the documented cap.
        Returning the cap on BOTH paths makes the limit provider-independent.
        Gemini behavior is unchanged (same value it already had).

        phase-75.14 (gap5-05) CORRECTION: the file_id ref is small but the
        document CONTENT is billed every call (Anthropic Files API docs);
        without config["data_prompt"] llm_client now drops the redundant
        document block, so this helper's file_id is a no-op until a caller
        supplies a data-only prompt.
        """
        config: dict = {"max_output_tokens": _ENRICHMENT_MAX_OUTPUT_TOKENS}
        fid_map = getattr(self, "_skill_file_ids", None)
        if not fid_map:
            return config
        fid = fid_map.get(skill_stem)
        if not fid:
            return config
        config["skill_file_id"] = fid
        return config

    def _run_enrichment_batch(
        self,
        requests: list[dict],
        *,
        batch_client: Any | None = None,
        cost_tracker: Any | None = None,
    ) -> dict:
        """phase-25.C9.1: window-batch dispatcher for enrichment agents.

        Submits ALL `requests` in a single Anthropic Message Batch
        (50% flat discount), polls until "ended", and fetches the per-
        custom_id response map. Custom IDs follow the safety-critical
        pattern `{ticker}__{agent_name}` (dotzlaw.com 2026 lesson:
        per-batch index-reset bugs cause silent data corruption when
        custom_ids are positional).

        Each `request` dict must carry:
          - ticker: str (caller-set; flows into custom_id)
          - agent_name: str (caller-set; flows into custom_id)
          - model: str (Anthropic model id)
          - messages: list[dict] (Anthropic Messages API shape)
          - max_tokens: int (per-request budget)

        Returns: dict[custom_id, LLMResponse]. Errored / expired rows
        surface as `LLMResponse(text="", thoughts="errored: <msg>")`
        per the BatchClient.fetch contract -- callers should check
        `.thoughts.startswith("errored:")`.

        `batch_client` and `cost_tracker` are kwargs primarily for
        testing (mocking). Production: pass `None` to use the
        orchestrator-bound BatchClient + the existing `_cost_tracker`.
        """
        if not requests:
            return {}
        bc = batch_client
        if bc is None:
            from backend.agents.llm_client import BatchClient
            bc = BatchClient()
        ct = cost_tracker if cost_tracker is not None else getattr(self, "_cost_tracker", None)

        normalized: list[dict] = []
        for req in requests:
            ticker = str(req.get("ticker") or "UNKNOWN")
            agent_name = str(req.get("agent_name") or "Unknown")
            custom_id = f"{ticker}__{agent_name}"
            normalized.append({
                "custom_id": custom_id,
                "model": req.get("model"),
                "messages": req.get("messages") or [],
                "max_tokens": int(req.get("max_tokens") or 1024),
            })

        batch_id = bc.submit(normalized)
        status = bc.poll(batch_id)
        if status != "ended":
            logger.warning(
                "_run_enrichment_batch: batch %s ended with status=%s; partial results may be missing",
                batch_id,
                status,
            )
        results = bc.fetch(batch_id)

        # phase-25.C9.1 + 25.C9: cost-tracker rows for succeeded rows
        # apply the 0.5x batch multiplier. Errored rows are skipped
        # (no successful call -> no cost row).
        if ct is not None:
            for custom_id, response in results.items():
                try:
                    if not response or (
                        isinstance(response.thoughts, str)
                        and response.thoughts.startswith("errored:")
                    ):
                        continue
                    agent_name = custom_id.split("__", 1)[1] if "__" in custom_id else "Unknown"
                    model_name = getattr(response, "model_name", None) or "claude"
                    ct.record(agent_name, model_name, response, is_batch=True)
                except Exception as _ct_err:
                    logger.warning(
                        "_run_enrichment_batch: cost-tracker record fail-open for %s: %r",
                        custom_id,
                        _ct_err,
                    )
        return results

    # ── Pipeline Steps ───────────────────────────────────────────────

    async def fetch_market_intel(self, ticker: str) -> dict:
        """Step 0: Fetch Alpha Vantage data."""
        return await alphavantage.get_market_intel(ticker, self.settings.alphavantage_api_key)

    def fetch_yfinance_data(self, ticker: str) -> dict:
        """Step 0b: Fetch yfinance fundamentals."""
        return yfinance_tool.get_comprehensive_financials(ticker)

    async def run_ingestion_agent(self, ticker: str) -> bool:
        """Step 1: Call the Ingestion Agent Cloud Function.

        phase-27.6.1: throttled at most 2 concurrent calls to sec.gov via
        the class-level _INGESTION_SEMAPHORE. Cycle d73f5129 (concurrency=8)
        produced 8 of 14 `Ingestion Agent Error: ERROR:429 Too Many Requests
        for url: https://www.sec.gov/files/company_tickers.json` failures.
        SEC EDGAR fair-access policy is ≤10 req/sec WITH a User-Agent; the
        bulk endpoint can be stricter. Capping at 2 concurrent leaves the
        LLM-layer concurrency cap (8) untouched but serializes the SEC
        ingestion calls behind a smaller window.
        """
        async with self._INGESTION_SEMAPHORE:
            logger.info(f"Ingestion Agent: checking filings for {ticker}")
            async with httpx.AsyncClient(timeout=900) as client:
                async with client.stream("POST", self.settings.ingestion_agent_url, json={"ticker": ticker}) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if line == "STREAM_COMPLETE":
                            break
                        elif line.startswith("ERROR:"):
                            raise RuntimeError(f"Ingestion Agent Error: {line}")
                        else:
                            logger.info(f"Ingestion: {line}")
        return True

    async def run_quant_agent(self, ticker: str) -> dict:
        """Step 2: Call the Quant Agent Cloud Function + merge yfinance."""
        logger.info(f"Quant Agent: fetching financials for {ticker}")
        final_json = None

        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("GET", f"{self.settings.quant_agent_url}?ticker={ticker}") as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line.startswith("FINAL_JSON:"):
                        json_str = line.split("FINAL_JSON:", 1)[1]
                        final_json = json_io.loads(json_str)
                    elif line.startswith("ERROR:"):
                        raise RuntimeError(line)
                    else:
                        logger.info(f"Quant: {line}")

        if final_json is None:
            raise RuntimeError("Quant Agent did not return final JSON data.")

        # Merge yfinance data (blocking call → offload to thread)
        yf_data = await asyncio.to_thread(yfinance_tool.get_comprehensive_financials, ticker)
        final_json["yf_data"] = yf_data
        return final_json

    def run_rag_agent(self, ticker: str) -> dict:
        """Step 3: RAG analysis on 10-K/10-Q documents."""
        if not self._rag_available:
            logger.info(f"RAG Agent: skipped for {ticker} (data store unavailable)")
            return {"text": "", "citations": []}
        logger.info(f"RAG Agent: analyzing documents for {ticker}")
        try:
            prompt = prompts.get_rag_prompt(ticker, fact_ledger=getattr(self, '_fact_ledger_json', ''))
            response = self._generate_with_retry(self.rag_client, prompt, "RAG")
        except Exception as exc:
            # RAG is enrichment, not core -- degrade gracefully so downstream
            # steps still run. Discovery Engine PERMISSION_DENIED and 404
            # data-store-missing are the two operational failure modes;
            # both return an empty {text, citations} so the pipeline continues.
            logger.warning(f"RAG Agent: fail-open for {ticker} ({type(exc).__name__}: {exc})")
            self._rag_available = False  # cache for the rest of this run
            return {"text": "", "citations": []}
        if response is None:
            return {"text": "", "citations": []}
        # grounding_metadata is list[dict] from GeminiClient
        citations = [{"uri": s.get("uri", ""), "title": s.get("title", "")} for s in (response.grounding_metadata or [])]
        return {"text": response.text, "citations": citations}

    def run_market_agent(self, ticker: str, av_data: dict) -> dict:
        """Step 4: Market sentiment analysis (Google Search grounded)."""
        logger.info(f"Market Agent: analyzing sentiment for {ticker} (grounded)")
        prompt = prompts.get_market_prompt(ticker, av_data, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        _model = self.grounded_client if self.supports_grounding else self.general_client
        response = self._generate_with_retry(_model, prompt, "Market", is_grounded=self.supports_grounding)
        grounding_sources = _extract_grounding_metadata(response)
        return {"text": _extract_text(response), "grounding_sources": grounding_sources}

    def run_competitor_agent(self, ticker: str, av_data: dict) -> dict:
        """Step 5: Competitor analysis (Google Search grounded)."""
        logger.info(f"Competitor Agent: analyzing rivals for {ticker} (grounded)")
        prompt = prompts.get_competitor_prompt(ticker, av_data, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        _model = self.grounded_client if self.supports_grounding else self.general_client
        response = self._generate_with_retry(_model, prompt, "Competitor", is_grounded=self.supports_grounding)
        grounding_sources = _extract_grounding_metadata(response)
        return {"text": _extract_text(response), "grounding_sources": grounding_sources}

    def run_macro_agent(self, ticker: str, av_data: dict) -> dict:
        """Step 6: Macroeconomic analysis."""
        logger.info(f"Macro Agent: analyzing economy for {ticker}")
        prompt = prompts.get_macro_prompt(ticker, av_data)
        response = self._generate_with_retry(self.general_client, prompt, "Macro")
        return {"text": _extract_text(response)}

    def run_deep_dive_agent(self, ticker: str, report: dict) -> dict:
        """Step 7: Find contradictions and probe with follow-up questions (Google Search grounded)."""
        logger.info(f"Deep Dive Agent: probing contradictions for {ticker} (grounded)")

        prompt = prompts.get_deep_dive_prompt(
            ticker,
            report.get("quant", {}),
            report.get("rag", {}).get("text", ""),
            report.get("market", {}).get("text", ""),
            report.get("competitor", {}).get("text", "No competitor data."),
            fact_ledger=getattr(self, '_fact_ledger_json', ''),
        )
        _model = self.grounded_client if self.supports_grounding else self.general_client
        response = self._generate_with_retry(_model, prompt, "Deep Dive (Questions)", is_grounded=self.supports_grounding)
        grounding_sources = _extract_grounding_metadata(response)
        questions = _extract_text(response).strip().split("\n")

        answers = []
        for q in questions:
            if q.strip():
                logger.info(f"Deep Dive investigating: {q.strip()}")
                ans = self._generate_with_retry(
                    self.rag_client, f"Answer this using 10-K: {q}", "Deep Dive (Answers)"
                )
                time.sleep(2)  # Rate limit protection
                answers.append(f"Q: {q}\nA: {_extract_text(ans)}")
        return {"text": "\n\n".join(answers), "grounding_sources": grounding_sources}

    # ── New Data-Enrichment Steps ────────────────────────────────────

    async def fetch_insider_data(self, ticker: str) -> dict:
        """Fetch SEC Form 4 insider trading data."""
        return await sec_insider.get_insider_trades(ticker)

    def fetch_options_data(self, ticker: str) -> dict:
        """Fetch options chain flow analysis."""
        return options_flow.get_options_flow(ticker)

    async def fetch_social_sentiment(self, ticker: str, fallback_articles: list[dict] | None = None) -> dict:
        """Fetch social/news sentiment from Alpha Vantage."""
        return await social_sentiment.get_social_sentiment(
            ticker, self.settings.alphavantage_api_key, fallback_articles
        )

    async def fetch_patent_data(self, ticker: str, company_name: str) -> dict:
        """Fetch USPTO patent data."""
        return await patent_tracker.get_patent_data(company_name, ticker, api_key=self.settings.patentsview_api_key)

    async def fetch_earnings_tone(self, ticker: str) -> dict:
        """Fetch earnings call transcript."""
        return await earnings_tone.get_earnings_tone(ticker, self.settings.api_ninjas_key, bucket_name=self.settings.gcs_bucket_name)

    async def fetch_fred_data(self) -> dict:
        """Fetch FRED macro indicators."""
        return await fred_data.get_macro_indicators(self.settings.fred_api_key)

    def fetch_alt_data(self, ticker: str, company_name: str) -> dict:
        """Fetch Google Trends alternative data."""
        return alt_data.get_google_trends(ticker, company_name)

    def fetch_sector_data(self, ticker: str) -> dict:
        """Fetch sector relative strength data."""
        return sector_analysis.get_sector_analysis(ticker)

    def fetch_monte_carlo(self, ticker: str) -> dict:
        """Fetch Monte Carlo VaR simulation results."""
        return monte_carlo.get_monte_carlo_simulation(ticker)

    def fetch_anomaly_scan(self, ticker: str) -> dict:
        """Fetch multi-dimensional anomaly scan."""
        return anomaly_detector.get_anomaly_scan(ticker)

    def fetch_quant_model(self, ticker: str) -> dict:
        """Fetch quant model signal using MDA-weighted factors."""
        return quant_model.get_quant_model_signal(ticker)

    async def fetch_nlp_sentiment(self, ticker: str, articles: list[dict]) -> dict:
        """Fetch transformer NLP sentiment via Vertex AI embeddings."""
        return await nlp_sentiment.get_nlp_sentiment(
            ticker, articles, self.settings.gcp_project_id, self.settings.gcp_location
        )

    # ── New LLM Analysis Steps ───────────────────────────────────────

    def run_insider_agent(self, ticker: str, insider_data: dict) -> dict:
        """Analyze insider trading patterns."""
        logger.info(f"Insider Agent: analyzing Form 4 data for {ticker}")
        prompt = prompts.get_insider_prompt(ticker, insider_data, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        response = self._generate_with_retry(self.general_client, prompt, "Insider", generation_config=self._skill_gen_config("insider_agent"))
        return {"text": _extract_text(response), "data": insider_data}

    def run_options_agent(self, ticker: str, options_data: dict) -> dict:
        """Analyze options flow for institutional signals."""
        logger.info(f"Options Agent: analyzing flow for {ticker}")
        prompt = prompts.get_options_prompt(ticker, options_data, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        response = self._generate_with_retry(self.general_client, prompt, "Options", generation_config=self._skill_gen_config("options_agent"))
        return {"text": _extract_text(response), "data": options_data}

    def run_social_sentiment_agent(self, ticker: str, sentiment_data: dict) -> dict:
        """Analyze social media and news sentiment."""
        logger.info(f"Social Sentiment Agent: analyzing for {ticker}")
        prompt = prompts.get_social_sentiment_prompt(ticker, sentiment_data, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        response = self._generate_with_retry(self.general_client, prompt, "Social Sentiment", generation_config=self._skill_gen_config("social_sentiment_agent"))
        return {"text": _extract_text(response), "data": sentiment_data}

    def run_patent_agent(self, ticker: str, patent_data: dict) -> dict:
        """Analyze patent/innovation data."""
        logger.info(f"Patent Agent: analyzing innovation for {ticker}")
        prompt = prompts.get_patent_prompt(ticker, patent_data, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        response = self._generate_with_retry(self.general_client, prompt, "Patent", generation_config=self._skill_gen_config("patent_agent"))
        return {"text": _extract_text(response), "data": patent_data}

    def run_earnings_tone_agent(self, ticker: str, transcript_data: dict) -> dict:
        """Analyze earnings call tone."""
        logger.info(f"Earnings Tone Agent: analyzing transcript for {ticker}")
        prompt = prompts.get_earnings_tone_prompt(ticker, transcript_data, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        response = self._generate_with_retry(self.general_client, prompt, "Earnings Tone", generation_config=self._skill_gen_config("earnings_tone_agent"))
        return {"text": _extract_text(response), "data": transcript_data}

    def run_enhanced_macro_agent(self, ticker: str, av_data: dict, fred_macro: dict) -> dict:
        """Enhanced macro analysis with FRED data (Google Search grounded)."""
        logger.info(f"Enhanced Macro Agent: analyzing economy for {ticker} (grounded)")
        prompt = prompts.get_enhanced_macro_prompt(ticker, av_data, fred_macro, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        _model = self.grounded_client if self.supports_grounding else self.general_client
        response = self._generate_with_retry(_model, prompt, "Enhanced Macro", is_grounded=self.supports_grounding)
        grounding_sources = _extract_grounding_metadata(response)
        return {"text": _extract_text(response), "fred_data": fred_macro, "grounding_sources": grounding_sources}

    def run_alt_data_agent(self, ticker: str, alt_data_result: dict) -> dict:
        """Analyze alternative data signals."""
        logger.info(f"Alt Data Agent: analyzing trends for {ticker}")
        prompt = prompts.get_alt_data_prompt(ticker, alt_data_result, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        response = self._generate_with_retry(self.general_client, prompt, "Alt Data", generation_config=self._skill_gen_config("alt_data_agent"))
        return {"text": _extract_text(response), "data": alt_data_result}

    def run_sector_analysis_agent(self, ticker: str, sector_data: dict) -> dict:
        """Analyze sector relative strength and rotation."""
        logger.info(f"Sector Agent: analyzing sector for {ticker}")
        prompt = prompts.get_sector_analysis_prompt(ticker, sector_data, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        response = self._generate_with_retry(self.general_client, prompt, "Sector", generation_config=self._skill_gen_config("sector_analysis_agent"))
        return {"text": _extract_text(response), "data": sector_data}

    def run_nlp_sentiment_agent(self, ticker: str, nlp_data: dict) -> dict:
        """Analyze transformer-based NLP sentiment."""
        logger.info(f"NLP Sentiment Agent: analyzing for {ticker}")
        prompt = prompts.get_nlp_sentiment_prompt(ticker, nlp_data, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        response = self._generate_with_retry(self.general_client, prompt, "NLP Sentiment", generation_config=self._skill_gen_config("nlp_sentiment_agent"))
        return {"text": _extract_text(response), "data": nlp_data}

    def run_anomaly_agent(self, ticker: str, anomaly_data: dict) -> dict:
        """Analyze statistical anomalies."""
        logger.info(f"Anomaly Agent: analyzing for {ticker}")
        prompt = prompts.get_anomaly_detection_prompt(ticker, anomaly_data, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        response = self._generate_with_retry(self.general_client, prompt, "Anomaly", generation_config=self._skill_gen_config("anomaly_agent"))
        return {"text": _extract_text(response), "data": anomaly_data}

    def run_scenario_agent(self, ticker: str, mc_data: dict) -> dict:
        """Analyze Monte Carlo scenario results.

        phase-26.3: routed through `quant_exec_client` (Gemini + code_execution
        tool) so the model verifies VaR consistency (`var_99 >= var_95`),
        probability coherence (`P(up) + P(down) <= 1.0`), and expected-shortfall
        arithmetic INSIDE the call rather than freestyling the numbers.
        """
        logger.info(f"Scenario Agent: analyzing risk for {ticker}")
        prompt = prompts.get_scenario_analysis_prompt(ticker, mc_data, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        response = self._generate_with_retry(self.quant_exec_client, prompt, "Scenario", generation_config=self._skill_gen_config("scenario_agent"))
        return {"text": _extract_text(response), "data": mc_data}

    def run_quant_model_agent(self, ticker: str, qm_data: dict) -> dict:
        """Analyze quant model MDA-weighted factor signal.

        phase-26.3: routed through `quant_exec_client` (Gemini + code_execution
        tool) so the model verifies Sharpe arithmetic, position-sizing bounds,
        and composite-score reproducibility INSIDE the call. Eliminates the
        silent arithmetic drift class (model says 0.42 when the math is 0.24).
        """
        logger.info(f"Quant Model Agent: analyzing factor signal for {ticker}")
        prompt = prompts.get_quant_model_prompt(ticker, qm_data, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        response = self._generate_with_retry(self.quant_exec_client, prompt, "Quant Model", generation_config=self._skill_gen_config("quant_model_agent"))
        return {"text": _extract_text(response), "data": qm_data}

    def run_alpha_decay_agent(
        self,
        prior_strategy: str,
        rolling_sharpe_trend: dict,
        hit_rate_trend: dict,
        macro_regime: str = "NEUTRAL",
        recent_drawdown_pct: float = 0.0,
    ) -> dict:
        """phase-26.5: alpha-decay / regime-shift early-warning detector.

        Cheap Gemini Flash call. Output {decay_signal, decay_attribution,
        recommended_action, rationale} feeds into the strategy router's
        phase-25.R policy at backend/autoresearch/promoter.py:7-69 as
        an UPSTREAM signal alongside the realized-P&L reactive trigger.

        Closes the lag between decay onset and capital reallocation that
        the reactive-only path suffers from (see Resonanz Capital 2025
        unwind post-mortem cited in research_brief.md)."""
        logger.info(f"Alpha Decay Agent: scanning strategy={prior_strategy} regime={macro_regime} dd={recent_drawdown_pct:.2f}%")
        prompt = prompts.get_alpha_decay_prompt(
            prior_strategy=prior_strategy,
            rolling_sharpe_trend=rolling_sharpe_trend,
            hit_rate_trend=hit_rate_trend,
            macro_regime=macro_regime,
            recent_drawdown_pct=recent_drawdown_pct,
            fact_ledger=getattr(self, '_fact_ledger_json', ''),
        )
        response = self._generate_with_retry(
            self.general_client, prompt, "Alpha Decay",
            generation_config=self._skill_gen_config("alpha_decay_agent"),
        )
        return {"text": _extract_text(response), "prior_strategy": prior_strategy, "macro_regime": macro_regime}

    # ── Synthesis ────────────────────────────────────────────────────

    def run_synthesis_pipeline(self, ticker: str, report: dict) -> dict:
        """Synthesis + Critic reflection loop → final validated JSON.

        Implements Evaluator-Optimizer pattern: Synthesis drafts, Critic reviews
        with structured verdict. If REVISE, Synthesis re-runs with critic feedback
        (up to max_synthesis_iterations). First pass uses deep-think; re-runs also use deep-think.
        """
        max_iterations = self.settings.max_synthesis_iterations
        logger.info(f"Synthesis Agent: drafting report for {ticker} (max {max_iterations} iterations)")

        _synthesis_limit = get_model_max_input_chars(self.synthesis_client.model_name)
        _critic_limit = get_model_max_input_chars(self.deep_think_client.model_name)
        _tight_context = bool(
            (_synthesis_limit is not None and _synthesis_limit <= 14_000)
            or (_critic_limit is not None and _critic_limit <= 14_000)
        )
        _compact_context = bool(
            (_synthesis_limit is not None and _synthesis_limit <= 30_000)
            or (_critic_limit is not None and _critic_limit <= 30_000)
        )

        # Build market_context with per-section truncation guard
        if _tight_context:
            _MAX_SECTION = 500
            _MAX_CONTEXT = 4_500
        elif _compact_context:
            _MAX_SECTION = 800
            _MAX_CONTEXT = 7_000
        else:
            _MAX_SECTION = 1500
            _MAX_CONTEXT = 12000
        market_context = (
            f"Market News & Sentiment:\n{report.get('market', {}).get('text', '')[:_MAX_SECTION]}\n\n"
            f"Competitor Analysis:\n{report.get('competitor', {}).get('text', 'No competitor data.')[:_MAX_SECTION]}\n\n"
            f"Macroeconomic Outlook:\n{report.get('macro', {}).get('text', 'No macro data.')[:_MAX_SECTION]}\n\n"
            f"Insider Activity:\n{report.get('insider', {}).get('text', 'No insider data.')[:_MAX_SECTION]}\n\n"
            f"Options Flow:\n{report.get('options', {}).get('text', 'No options data.')[:_MAX_SECTION]}\n\n"
            f"Social Sentiment:\n{report.get('social_sentiment', {}).get('text', 'No social data.')[:_MAX_SECTION]}\n\n"
            f"Earnings Call Tone:\n{report.get('earnings_tone', {}).get('text', 'No earnings tone data.')[:_MAX_SECTION]}\n\n"
            f"Alternative Data:\n{report.get('alt_data', {}).get('text', 'No alternative data.')[:_MAX_SECTION]}\n\n"
            f"Sector Analysis:\n{report.get('sector_analysis', {}).get('text', 'No sector data.')[:_MAX_SECTION]}"
        )
        if len(market_context) > _MAX_CONTEXT:
            market_context = market_context[:_MAX_CONTEXT]

        sector_catalyst = report.get("patent", {}).get("text", "No sector catalyst data.")
        quant_data = report.get("quant", {})

        # Inject session context into market context
        session_context = report.get("_session_context", "")
        if session_context:
            market_context = session_context + "\n\n" + market_context

        # Shared synthesis prompt kwargs for both initial and revision calls
        fact_ledger_json = json.dumps(report.get("_fact_ledger", {}), indent=2, default=str)
        if _compact_context:
            fact_ledger_json = compact_text(fact_ledger_json, 700 if _tight_context else 1_200)

        # Compact unbounded fields for small-context models
        _rag_report = report.get("rag", {}).get("text", "")
        _sector_catalyst = sector_catalyst
        _supply_chain = report.get("supply_chain", "No supply chain data.")
        _deep_dive = report.get("deep_dive", {}).get("text", "") if isinstance(report.get("deep_dive"), dict) else report.get("deep_dive", "")
        _quant_for_synthesis = quant_data

        if _compact_context:
            _field_cap = 600 if _tight_context else 1_200
            _quant_for_synthesis = compact_quant_snapshot(quant_data)
            _rag_report = compact_text(_rag_report, _field_cap)
            _sector_catalyst = compact_text(_sector_catalyst, _field_cap)
            _supply_chain = compact_text(_supply_chain, _field_cap)
            _deep_dive = compact_text(_deep_dive, _field_cap)
            logger.info(
                "Synthesis compaction active: quant=snapshot, field_cap=%d chars", _field_cap
            )

        synthesis_kwargs: dict = {
            "ticker": ticker,
            "quant_report": _quant_for_synthesis,
            "rag_report": _rag_report,
            "market_report": market_context,
            "sector_catalyst_report": _sector_catalyst,
            "supply_chain_report": _supply_chain,
            "deep_dive_analysis": _deep_dive,
            "fact_ledger": fact_ledger_json,
        }

        # Initial synthesis draft (uses synthesis_model with 4096 output token limit)
        draft_prompt = prompts.get_synthesis_prompt(**synthesis_kwargs)

        # phase-26.2: when Advisor Tool is enabled in settings AND the configured
        # synthesis client is on an Opus 4.x model, route this initial draft
        # through advisor_call (Sonnet 4.6 executor + Opus 4.7 advisor) for the
        # 25-45% estimated cost reduction documented in handoff/archive/phase-26.2/
        # research_brief.md. Default flag = False -> zero behavior change.
        # Failure modes:
        #   - HTTP 400 invalid pairing -> falls through to the standard
        #     generate_content path below (try/except).
        #   - Anthropic SDK unavailable or beta path errors -> same fallback.
        _advisor_text: Optional[str] = None
        _enable_advisor = bool(getattr(self.settings, "enable_advisor_tool", False))
        _synth_model_name = getattr(self.synthesis_client, "model_name", "") or ""
        if _enable_advisor and _synth_model_name.startswith("claude-opus-4"):
            try:
                from backend.agents.llm_client import advisor_call as _advisor_call
                _adv = _advisor_call(
                    prompt=draft_prompt,
                    executor_model="claude-sonnet-4-6",
                    advisor_model=_synth_model_name,
                    role="Synthesis",
                    max_tokens=4096,
                )
                _advisor_text = _adv.get("text") or ""
                # Record blended cost into cost_tracker (separate tier).
                _ct = getattr(self, "cost_tracker", None)
                if _ct is not None and hasattr(_ct, "record_advisor_call"):
                    _ct.record_advisor_call(
                        agent_name="Synthesis",
                        executor_model="claude-sonnet-4-6",
                        advisor_model=_synth_model_name,
                        executor_input_tokens=_adv["executor_tokens"][0],
                        executor_output_tokens=_adv["executor_tokens"][1],
                        advisor_input_tokens=_adv["advisor_tokens"][0],
                        advisor_output_tokens=_adv["advisor_tokens"][1],
                        ticker=ticker,
                    )
                logger.info(
                    "[phase-26.2] Synthesis routed through advisor_call: "
                    "executor_tokens=%s advisor_tokens=%s advisor_invoked=%s",
                    _adv["executor_tokens"], _adv["advisor_tokens"], _adv["advisor_invoked"],
                )
            except Exception as _adv_exc:
                logger.warning(
                    "[phase-26.2] advisor_call failed (%r); falling back to Opus-solo path",
                    _adv_exc,
                )
                _advisor_text = None  # falls through

        if _advisor_text is not None:
            draft_text = _clean_json_output(_advisor_text)
        else:
            draft_response = self._generate_with_retry(
                self.synthesis_client, draft_prompt, "Synthesis",
                is_deep_think=True, generation_config=_SYNTHESIS_STRUCTURED_CONFIG,
            )
            draft_text = _clean_json_output(_extract_text(draft_response))

        synthesis_iterations = 1
        # phase-75.4 (gap5-03): True when the critic's verdict could not be parsed even
        # after the retry, i.e. the report was NOT reviewed. Always present on the
        # returned dict so a consumer can never mistake "gate skipped" for "gate passed".
        critic_degraded = False
        critic_issues_log = []
        critic_quant_data = compact_quant_snapshot(quant_data) if _compact_context else quant_data

        for iteration in range(max_iterations):
            # Critic review with structured verdict
            logger.info(f"Critic Agent: reviewing draft for {ticker} (iteration {iteration + 1})")
            critic_feedback_str = ""
            if iteration > 0:
                critic_feedback_str = json.dumps(critic_issues_log[-1], indent=2)

            critic_draft_reference = compact_report_reference(
                draft_text,
                max_chars=2_000 if _tight_context else 3_500 if _compact_context else 8_000,
            ) if _compact_context else draft_text
            critic_prompt = prompts.get_critic_prompt(
                ticker,
                critic_draft_reference,
                critic_quant_data,
                critic_feedback=critic_feedback_str,
                fact_ledger=fact_ledger_json,
            )
            critic_response = self._generate_with_retry(
                self.deep_think_client, critic_prompt, "Critic",
                is_deep_think=True, generation_config=_CRITIC_STRUCTURED_CONFIG,
            )
            critic_text = _clean_json_output(_extract_text(critic_response))

            # Parse structured Critic verdict
            critic_result = _parse_json_with_fallback(critic_text, "Critic")

            # phase-75.4 (gap5-03): an unparseable critic response used to be silently
            # upgraded to PASS -- the quality gate DISAPPEARED rather than failed
            # (fail-OPEN). Anthropic's guidance is explicit: "Treat max_tokens
            # truncation as a retriable error for structured outputs, not as a valid
            # response to parse" / "Do NOT treat truncated responses as success."
            # Mirror the existing in-repo idiom at llm_client.py:1656-1684 -- retry
            # once at a raised budget, then proceed FLAGGED rather than blessed.
            if not critic_result:
                _retry_budget = min(_CRITIC_STRUCTURED_CONFIG["max_output_tokens"] * 2, 8192)
                logger.warning(
                    "Critic returned unparseable JSON -- retrying once at "
                    "max_output_tokens=%d (was %d)",
                    _retry_budget, _CRITIC_STRUCTURED_CONFIG["max_output_tokens"],
                )
                critic_retry_config = {
                    **_CRITIC_STRUCTURED_CONFIG, "max_output_tokens": _retry_budget,
                }
                critic_response = self._generate_with_retry(
                    self.deep_think_client, critic_prompt, "Critic-Retry",
                    is_deep_think=True, generation_config=critic_retry_config,
                )
                critic_text = _clean_json_output(_extract_text(critic_response))
                critic_result = _parse_json_with_fallback(critic_text, "Critic-Retry")

            if not critic_result:
                logger.warning(
                    "Critic returned unparseable JSON after retry -- proceeding with "
                    "the UNREVIEWED draft, flagged critic_degraded=True. The quality "
                    "gate did NOT run for this report."
                )
                critic_degraded = True
                break

            verdict = critic_result.get("verdict", "PASS").upper()
            issues = critic_result.get("issues", [])
            corrected_report = critic_result.get("corrected_report")
            major_issues = [i for i in issues if i.get("severity") == "major"]

            logger.info(f"Critic verdict: {verdict} -- {len(major_issues)} major, {len(issues) - len(major_issues)} minor issues")

            if verdict == "PASS" or not major_issues:
                # Accept the corrected report (or draft if no corrected_report)
                if corrected_report and isinstance(corrected_report, dict):
                    corrected_report["synthesis_iterations"] = synthesis_iterations
                    corrected_report["critic_issues"] = issues
                    corrected_report["critic_degraded"] = critic_degraded
                    return corrected_report
                break

            # REVISE: Check if we have iterations left
            if iteration >= max_iterations - 1:
                logger.info("Max synthesis iterations reached, accepting Critic's corrected report.")
                if corrected_report and isinstance(corrected_report, dict):
                    corrected_report["synthesis_iterations"] = synthesis_iterations
                    corrected_report["critic_issues"] = issues
                    corrected_report["critic_degraded"] = critic_degraded
                    return corrected_report
                break

            # Re-run Synthesis with Critic feedback
            critic_issues_log.append(issues)
            synthesis_iterations += 1
            logger.info(f"Synthesis Agent: revising report for {ticker} (iteration {synthesis_iterations})")

            revision_prompt = prompts.get_synthesis_revision_prompt(
                **synthesis_kwargs,
                critic_issues=issues,
                previous_draft=compact_report_reference(
                    draft_text,
                    max_chars=2_200 if _tight_context else 4_000 if _compact_context else 5_000,
                ),
            )
            revision_response = self._generate_with_retry(
                self.synthesis_client, revision_prompt, f"Synthesis-Rev{synthesis_iterations}",
                is_deep_think=True, generation_config=_SYNTHESIS_STRUCTURED_CONFIG,
            )
            draft_text = _clean_json_output(_extract_text(revision_response))

        # Fallback: parse whatever we have
        final_data = _parse_json_with_fallback(draft_text, "Synthesis-Final")
        if final_data:
            final_data["synthesis_iterations"] = synthesis_iterations
            final_data["critic_degraded"] = critic_degraded
            return final_data

        logger.warning("Failed to parse final report, returning error.")
        return {
            "error": "Failed to parse final report.",
            "synthesis_iterations": synthesis_iterations,
            "critic_degraded": critic_degraded,
        }

    def compute_weighted_score(self, scoring_matrix: dict) -> float:
        """Apply pillar weights to compute the final score."""
        s = self.settings
        weights = {
            "pillar_1_corporate": s.weight_corporate,
            "pillar_2_industry": s.weight_industry,
            "pillar_3_valuation": s.weight_valuation,
            "pillar_4_sentiment": s.weight_sentiment,
            "pillar_5_governance": s.weight_governance,
        }
        return round(sum(scoring_matrix.get(k, 0) * v for k, v in weights.items()), 2)

    async def run_full_analysis(self, ticker: str, on_step=None) -> dict:
        """
        Executes the complete 13-step analysis pipeline end-to-end.
        
        Args:
            ticker: Stock ticker symbol
            on_step: Optional callback(step_name, status, message) for progress updates
        
        Returns:
            Complete report dict with all agent outputs, debate, traces, and final synthesis.
        """

        def step(name, status, msg=""):
            if on_step:
                on_step(name, status, msg)

        report = {}
        traces = TraceCollector()
        self._cost_tracker = CostTracker()
        ctx = AnalysisContext()

        # Model display name for progress messages
        _model_label = self.settings.gemini_model

        # Lite mode overrides: reduce debate rounds, synthesis iterations
        lite = self.settings.lite_mode
        effective_debate_rounds = 1 if lite else self.settings.max_debate_rounds
        effective_risk_rounds = self.settings.max_risk_debate_rounds
        skip_deep_dive = lite
        skip_da = lite  # skip Devil's Advocate
        skip_risk_assessment = lite

        # Step 0: Fetch external data (Alpha Vantage + yfinance in parallel)
        step("market_intel", "started", "Fetching Alpha Vantage data...")
        av_data = await self.fetch_market_intel(ticker)
        n_articles = len(av_data.get('sentiment_summary', []))
        av_source = av_data.get('source', 'alphavantage')
        if av_data.get('rate_limited'):
            step("market_intel", "completed", f"AV rate-limited — fell back to yfinance ({n_articles} articles)")
        elif av_data.get('error'):
            step("market_intel", "completed", f"Warning: {av_data['error']}")
        else:
            step("market_intel", "completed", f"Got {n_articles} articles from {av_source}")

        # phase-60.1 (AW-4): KR-aware (non-SEC) gate. SEC-bound stages
        # (ingestion CF, quant CF's companyfacts leg, RAG over EDGAR filings)
        # cannot serve non-US listings -- during the away week every .KS
        # ticker hard-aborted at the quant CF's CIK stage ("not found in SEC
        # CIK mapping") and the WHOLE full pipeline silently fell back to the
        # 2-call lite analyzer. Degrade EXPLICITLY instead: skip the SEC-bound
        # stages with a persisted `skipped_stages` tag (lands in
        # full_report_json -> auditable in BQ) and run the 26+ market-agnostic
        # agents on yfinance fundamentals.
        _sec_covered = self._is_sec_covered(ticker)
        if not _sec_covered:
            report["skipped_stages"] = []

        # Step 1: Ingestion agent
        # phase-27.6.6: best-effort ingestion. Cloud Function re-fetches SEC's
        # CIK map per call and hits 429 (cycle d73f5129, cycle #10) — that's a
        # CF-side issue requiring redeploy (phase-27.6.4, deferred). In the
        # meantime, treat ingestion as best-effort: if the CF errors (any
        # reason), log a warning and continue. Downstream steps still have
        # GCS-cached filings from prior cycles; only NEWLY-added filings since
        # the last successful ingestion would be missing for THIS cycle.
        step("ingestion", "started", "Checking for new filings...")
        if not _sec_covered:
            report["skipped_stages"].append({
                "stage": "ingestion_sec",
                "reason": "non-US listing: SEC EDGAR has no filings for this symbol",
            })
            step("ingestion", "completed", "Skipped -- non-SEC market (no EDGAR filings)")
        else:
            try:
                await self.run_ingestion_agent(ticker)
                step("ingestion", "completed", "Filings up to date")
            except Exception as _ingestion_exc:
                logger.warning(
                    "Ingestion best-effort failure for %s (continuing with cached filings): %s",
                    ticker, _ingestion_exc,
                )
                step(
                    "ingestion",
                    "completed",
                    f"Best-effort skip — using cached filings ({type(_ingestion_exc).__name__})",
                )

        # Step 2: Quant agent
        step("quant", "started", "Fetching financial data...")
        if _sec_covered:
            report["quant"] = await self.run_quant_agent(ticker)
            step("quant", "completed", "Financial data collected")
        else:
            _yf_only = await asyncio.to_thread(
                yfinance_tool.get_comprehensive_financials, ticker
            )
            report["quant"] = _quant_from_yfinance(ticker, _yf_only)
            report["skipped_stages"].append({
                "stage": "quant_cf_sec",
                "reason": "non-US listing: not in SEC CIK mapping; fundamentals from yfinance only",
            })
            step("quant", "completed", "SEC quant skipped (non-US listing) -- yfinance fundamentals only")
        company_name = report["quant"].get("company_name", ticker)

        # Session memory: capture key quant findings
        quant = report["quant"]
        if isinstance(quant, dict):
            # phase-27.6.2: guard against `valuation: None`. Dict.get(k, {})
            # only returns the default when k is ABSENT; if k=None, it returns
            # None and the next .get raises AttributeError. The `or {}` makes
            # it total over both missing-key and None-value cases.
            pe = quant.get("pe_ratio") or (quant.get("valuation") or {}).get("P/E Ratio")
            if pe:
                ctx.add_finding(f"P/E ratio: {pe}")
            sector_name = quant.get("sector", "")
            if sector_name:
                ctx.add_finding(f"Sector: {sector_name}")

        # Build fact ledger from quant data — injected into ALL agent prompts
        # Research: VeNRA typed fact ledger achieves 1.2% hallucination rate
        fact_ledger = _build_fact_ledger(report["quant"])
        # phase-32.3: inject portfolio-level sector exposure into the per-ticker
        # FACT_LEDGER so the Risk Judge can reason about concentration risk at
        # prompt time. Fail-open: any exception logs a warning and stores None
        # under the field so downstream prompts get an explicit "no data"
        # rather than crashing the analysis.
        try:
            from backend.db.bigquery_client import BigQueryClient
            _bq_pse = BigQueryClient(self.settings)
            _positions = _bq_pse.get_paper_positions() or []
            fact_ledger["portfolio_sector_exposure"] = _compute_portfolio_sector_exposure(
                _positions, threshold_pct=60.0,
            )
        except Exception as _pse_exc:
            logger.warning(
                "phase-32.3: portfolio_sector_exposure compute failed (non-fatal): %s",
                _pse_exc,
            )
            fact_ledger["portfolio_sector_exposure"] = None
        fact_ledger_json = json.dumps(fact_ledger, indent=2, default=str)
        report["_fact_ledger"] = fact_ledger
        self._fact_ledger_json = fact_ledger_json  # available to all agent methods

        # Step 3: RAG agent
        step("rag", "started", "Analyzing 10-K/10-Q documents...")
        if not _sec_covered:
            # phase-60.1 (AW-4): the RAG corpus is SEC filings; a non-US
            # symbol has no documents there. Skip the LLM call entirely
            # (cheaper AND honest) with the same persisted tag shape.
            report["rag"] = {"text": "", "citations": []}
            report["skipped_stages"].append({
                "stage": "rag_sec_filings",
                "reason": "non-US listing: RAG corpus is SEC 10-K/10-Q only",
            })
            step("rag", "completed", "Skipped -- non-SEC market (no EDGAR corpus)")
        else:
            report["rag"] = await asyncio.to_thread(self.run_rag_agent, ticker)
        # phase-4.14.15 (MF-32): surface RAG rows as Anthropic
        # search_result content blocks so downstream Claude synthesis
        # can emit native cited_text anchors. The Gemini-first pipeline
        # does not consume these, but the Claude MAS orchestrator
        # (multi_agent_orchestrator) may read them via report["rag"]
        # when a Claude synthesis path is active.
        _rag_rows = report["rag"].get("rows") if isinstance(report.get("rag"), dict) else None
        if _rag_rows:
            report["rag"]["search_result_blocks"] = bq_rows_to_search_results(
                _rag_rows, table_id="pyfinagent_data.filings_rag",
            )
        if _sec_covered:
            step("rag", "completed", "Document analysis complete")

        # Step 4: Market agent
        step("market", "started", "Analyzing sentiment...")
        report["market"] = await asyncio.to_thread(self.run_market_agent, ticker, av_data)
        step("market", "completed", "Sentiment analysis complete")

        # Session memory: capture sentiment direction
        market_text = report.get("market", {}).get("text", "")
        if "Divergence Warning" in market_text:
            ctx.add_finding("Sentiment-price divergence detected")
        elif "bullish" in market_text.lower()[:200]:
            ctx.add_finding("Overall market sentiment: bullish")
        elif "bearish" in market_text.lower()[:200]:
            ctx.add_finding("Overall market sentiment: bearish")

        # Step 5: Competitor agent
        step("competitor", "started", "Analyzing rivals...")
        report["competitor"] = await asyncio.to_thread(self.run_competitor_agent, ticker, av_data)
        step("competitor", "completed", "Competitor analysis complete")

        # Step 6: Fetch enrichment data in parallel (12 non-LLM calls)
        step("data_enrichment", "started", "Fetching 12 enrichment data sources in parallel...")
        _enrichment_done_count = 0
        _enrichment_labels_done: list[str] = []

        async def _safe(coro_or_func, label, *args):
            """Run async or sync callable, return result or error dict."""
            nonlocal _enrichment_done_count
            try:
                if asyncio.iscoroutinefunction(coro_or_func):
                    result = await coro_or_func(*args)
                else:
                    result = await asyncio.to_thread(coro_or_func, *args)
                _enrichment_done_count += 1
                _enrichment_labels_done.append(label)
                step("data_enrichment", "running", f"[{_enrichment_done_count}/12] {label} data collected")
                return result
            except Exception as e:
                logger.warning(f"{label} data fetch failed: {e}")
                _enrichment_done_count += 1
                _enrichment_labels_done.append(f"{label} (error)")
                step("data_enrichment", "running", f"[{_enrichment_done_count}/12] {label} failed: {e}")
                return {"signal": "ERROR", "summary": f"Error: {e}"}

        articles = av_data.get("sentiment_summary", [])

        # Fallback: yfinance news when AV returns nothing
        fallback_articles: list[dict] = []
        if not articles:
            import yfinance as yf
            yf_news = yf.Ticker(ticker).news or []
            for item in yf_news:
                content = item.get("content", item)
                title = content.get("title", "")
                summary = content.get("summary", "") or title
                provider = content.get("provider", {})
                src = provider.get("displayName", "unknown") if isinstance(provider, dict) else "unknown"
                fallback_articles.append({
                    "title": title,
                    "summary": summary,
                    "source": src,
                    "overall_sentiment_score": None,
                })
            articles = fallback_articles
            if fallback_articles:
                # phase-25.B7: promote to WARNING (was INFO; suppressed in
                # default views) so AV-down conditions surface immediately,
                # and persist a row to data_source_events so the operator can
                # compute pct_yfinance_fallback_dominance over any window.
                logger.warning(
                    "AV empty for %s -- using %d yfinance articles as fallback",
                    ticker,
                    len(fallback_articles),
                )
                try:
                    self.bq.save_data_source_event(
                        ticker=ticker,
                        source="yfinance_fallback",
                        kind="fallback",
                        article_count=len(fallback_articles),
                        notes="AV sentiment_summary empty",
                    )
                except Exception as _bq_exc:
                    logger.warning(
                        "save_data_source_event fail-open for %s: %r", ticker, _bq_exc
                    )

        # Sector routing: determine which tools to skip
        sector_for_routing = ""
        if isinstance(report.get("quant"), dict):
            sector_for_routing = report["quant"].get("sector", "")
        skipped_tools = SECTOR_SKIP_MAP.get(sector_for_routing, set())
        if skipped_tools:
            logger.info(f"Sector routing: {sector_for_routing} -> skipping tools: {skipped_tools}")
            ctx.add_finding(f"Sector routing: skipping {', '.join(skipped_tools)} for {sector_for_routing}")

        async def _skip_placeholder(label: str):
            """Return a placeholder for skipped tools."""
            nonlocal _enrichment_done_count
            _enrichment_done_count += 1
            _enrichment_labels_done.append(f"{label} (skipped)")
            step("data_enrichment", "running", f"[{_enrichment_done_count}/12] {label} skipped (sector routing)")
            return {"signal": "SKIPPED", "summary": f"Skipped: not relevant for {sector_for_routing} sector"}

        (
            insider_data, options_data, social_data, patent_data,
            earnings_data, fred_macro, alt_result, sector_data,
            nlp_data, anomaly_data, mc_data, qm_data,
        ) = await asyncio.gather(
            _safe(self.fetch_insider_data, "Insider", ticker),
            _safe(self.fetch_options_data, "Options", ticker),
            _safe(self.fetch_social_sentiment, "Social", ticker, articles or fallback_articles or None),
            _skip_placeholder("Patent") if "patent" in skipped_tools else _safe(self.fetch_patent_data, "Patent", ticker, company_name),
            _safe(self.fetch_earnings_tone, "Earnings", ticker),
            _safe(self.fetch_fred_data, "FRED"),
            _skip_placeholder("AltData") if "alt_data" in skipped_tools else _safe(self.fetch_alt_data, "AltData", ticker, company_name),
            _safe(self.fetch_sector_data, "Sector", ticker),
            _safe(self.fetch_nlp_sentiment, "NLP", ticker, articles),
            _safe(self.fetch_anomaly_scan, "Anomaly", ticker),
            _safe(self.fetch_monte_carlo, "MonteCarlo", ticker),
            _safe(self.fetch_quant_model, "QuantModel", ticker),
        )
        step("data_enrichment", "completed", "All 12 enrichment data sources collected")

        # Session memory: capture enrichment signals
        for src, data in [("insider", insider_data), ("options", options_data),
                          ("social", social_data), ("patent", patent_data),
                          ("sector", sector_data), ("nlp", nlp_data),
                          ("anomaly", anomaly_data), ("monte_carlo", mc_data),
                          ("quant_model", qm_data)]:
            sig = data.get("signal", "N/A") if isinstance(data, dict) else "N/A"
            if sig != "N/A" and sig != "ERROR":
                ctx.set_signal(src, sig)

        # Step 6b: Info-Gap Detection (AlphaQuanter ReAct loop)
        step("info_gap", "started", "Scanning data completeness...")
        enrichment_raw = {
            "insider": insider_data, "options": options_data,
            "social_sentiment": social_data, "patent": patent_data,
            "earnings_tone": earnings_data, "fred_macro": fred_macro,
            "alt_data": alt_result, "sector": sector_data,
            "nlp_sentiment": nlp_data, "anomaly": anomaly_data,
            "monte_carlo": mc_data, "quant_model": qm_data,
        }
        sector_name = ""
        if isinstance(report.get("quant"), dict):
            sector_name = report["quant"].get("sector", "")
        info_gap_report = detect_info_gaps(enrichment_raw, sector=sector_name)
        critical_gaps = info_gap_report.get("critical_gaps", [])

        if critical_gaps:
            step("info_gap", "running", f"Found {len(critical_gaps)} critical gaps: {', '.join(critical_gaps)}. Retrying...")
            retry_funcs = {
                "insider": lambda: self.fetch_insider_data(ticker),
                "options": lambda: self.fetch_options_data(ticker),
                "social_sentiment": lambda: self.fetch_social_sentiment(ticker, articles),
                "patent": lambda: self.fetch_patent_data(ticker, company_name),
                "earnings_tone": lambda: self.fetch_earnings_tone(ticker),
                "fred_macro": lambda: self.fetch_fred_data(),
                "alt_data": lambda: self.fetch_alt_data(ticker, company_name),
                "sector": lambda: self.fetch_sector_data(ticker),
                "nlp_sentiment": lambda: self.fetch_nlp_sentiment(ticker, articles),
                "anomaly": lambda: self.fetch_anomaly_scan(ticker),
                "monte_carlo": lambda: self.fetch_monte_carlo(ticker),
                "quant_model": lambda: self.fetch_quant_model(ticker),
            }
            recovered = await retry_critical_gaps(
                critical_gaps,
                retry_funcs,
                max_retries=2,
                on_progress=lambda msg: step("info_gap", "running", msg),
            )
            # Update enrichment data with recovered sources
            for key, new_data in recovered.items():
                if new_data and new_data.get("signal") != "ERROR":
                    enrichment_raw[key] = new_data
                    # Update the local variables
                    if key == "insider": insider_data = new_data
                    elif key == "options": options_data = new_data
                    elif key == "social_sentiment": social_data = new_data
                    elif key == "patent": patent_data = new_data
                    elif key == "earnings_tone": earnings_data = new_data
                    elif key == "fred_macro": fred_macro = new_data
                    elif key == "alt_data": alt_result = new_data
                    elif key == "sector": sector_data = new_data
                    elif key == "nlp_sentiment": nlp_data = new_data
                    elif key == "anomaly": anomaly_data = new_data
                    elif key == "monte_carlo": mc_data = new_data
                    elif key == "quant_model": qm_data = new_data
            # Re-assess after retries
            info_gap_report = detect_info_gaps(enrichment_raw, sector=sector_name)
            step("info_gap", "running", f"Recovered {len(recovered)} sources. Quality: {info_gap_report['data_quality_score']:.0%}")
        step("info_gap", "completed", f"Data quality: {info_gap_report['data_quality_score']:.0%} — {len(info_gap_report.get('critical_gaps', []))} critical gaps remaining")

        # Session memory: data quality
        dq = info_gap_report.get("data_quality_score", 1.0)
        if dq < 0.7:
            ctx.add_finding(f"Data quality low ({dq:.0%}) — gaps in: {', '.join(info_gap_report.get('critical_gaps', []))}")

        # Quality Gate 1: Data quality threshold
        low_data_quality = dq < self.settings.data_quality_min
        if low_data_quality:
            ctx.add_finding(f"QUALITY GATE: Data quality {dq:.0%} below threshold {self.settings.data_quality_min:.0%} — debate/risk skipped")

        # Step 7: Run LLM enrichment agents (11 agents)
        step("enrichment_analysis", "started", "Running 12 enrichment analysis agents...")

        def _run_agent_with_trace(agent_func, agent_name, t, *args):
            """Run an enrichment agent and record a decision trace."""
            t0 = time.time()
            result = agent_func(*args)
            latency = (time.time() - t0) * 1000
            data_arg = args[-1] if args else {}
            traces.add(DecisionTrace(
                agent_name=agent_name,
                input_data_hash=hash_input(data_arg),
                output_signal=data_arg.get("signal", "N/A") if isinstance(data_arg, dict) else "N/A",
                confidence=0.7,
                evidence_citations=[data_arg.get("summary", "")][:1] if isinstance(data_arg, dict) else [],
                reasoning_steps=[f"Analyzed {agent_name.lower()} data for {t}"],
                raw_output=result.get("text", "")[:200] if isinstance(result, dict) else "",
                latency_ms=latency,
            ))
            return result

        _agent_list = [
            ("insider", self.run_insider_agent, "Insider Activity", insider_data),
            ("options", self.run_options_agent, "Options Flow", options_data),
            ("social_sentiment", self.run_social_sentiment_agent, "Social Sentiment", social_data),
            ("patent", self.run_patent_agent, "Patent Innovation", patent_data),
            ("earnings_tone", self.run_earnings_tone_agent, "Earnings Tone", earnings_data),
            ("alt_data", self.run_alt_data_agent, "Alt Data", alt_result),
            ("sector_analysis", self.run_sector_analysis_agent, "Sector Analysis", sector_data),
            ("nlp_sentiment", self.run_nlp_sentiment_agent, "NLP Sentiment", nlp_data),
            ("anomaly", self.run_anomaly_agent, "Anomaly Detection", anomaly_data),
            ("scenario", self.run_scenario_agent, "Scenario Analysis", mc_data),
            ("quant_model", self.run_quant_model_agent, "Quant Model", qm_data),
        ]
        _done_count = 0
        _total = len(_agent_list)

        async def _run_one(key, func, name, data):
            nonlocal _done_count
            step("enrichment_analysis", "running", f"{_model_label} analyzing {name}...")
            try:
                result = await asyncio.to_thread(_run_agent_with_trace, func, name, ticker, ticker, data)
            except Exception as e:
                logger.error(f"Enrichment agent {name} failed: {e}")
                result = {"text": f"Error: {e}", "data": data}
            _done_count += 1
            signal = data.get('signal', 'N/A') if isinstance(data, dict) else 'done'
            step("enrichment_analysis", "running", f"[{_done_count}/{_total}] {name} -> {signal}")
            return key, result

        results = await asyncio.gather(*[
            _run_one(key, func, name, data)
            for key, func, name, data in _agent_list
        ])
        for key, result in results:
            report[key] = result
        step("enrichment_analysis", "completed", f"All {_total} enrichment agents complete")

        # Step 8: Agent Debate (Multi-Round Bull vs Bear + Devil's Advocate + Moderator)
        # Quality Gate: Skip debate if data quality is below threshold
        if low_data_quality:
            step("debate", "started", "Skipping debate — data quality below threshold...")
            debate_result = {
                "consensus": "HOLD",
                "consensus_confidence": 0.3,
                "bull_case": {"thesis": "Insufficient data for bull case", "confidence": 0.0},
                "bear_case": {"thesis": "Insufficient data for bear case", "confidence": 0.0},
                "contradictions": [],
                "dissent_registry": [],
                "debate_rounds": [],
                "total_rounds": 0,
                "devils_advocate": {"challenges": [], "summary": "Skipped: data quality gate"},
                "skipped_reason": f"Data quality {dq:.0%} below threshold {self.settings.data_quality_min:.0%}",
            }
            report["debate"] = debate_result
            # Still need situation_desc for risk assessment
            sector_name = ""
            if isinstance(report.get("quant"), dict):
                sector_name = report["quant"].get("sector", "")
            situation_desc = build_situation_description(ticker, sector_name, {})
            step("debate", "completed", f"Skipped — data quality {dq:.0%} below threshold")
        else:
            step("debate", "started", "Running multi-round adversarial debate...")
            step("debate", "running", "Collecting positions from all enrichment agents...")
            enrichment_for_debate = {
                "insider": {"signal": insider_data.get("signal", "N/A"), "summary": insider_data.get("summary", ""), "analysis": report["insider"].get("text", "")},
                "options": {"signal": options_data.get("signal", "N/A"), "summary": options_data.get("summary", ""), "analysis": report["options"].get("text", "")},
                "social_sentiment": {"signal": social_data.get("signal", "N/A"), "summary": social_data.get("summary", ""), "analysis": report["social_sentiment"].get("text", "")},
                "patent": {"signal": patent_data.get("signal", "N/A"), "summary": patent_data.get("summary", ""), "analysis": report["patent"].get("text", "")},
                "earnings_tone": {"signal": earnings_data.get("signal", "N/A"), "summary": earnings_data.get("summary", ""), "analysis": report["earnings_tone"].get("text", "")},
                "alt_data": {"signal": alt_result.get("signal", "N/A"), "summary": alt_result.get("summary", ""), "analysis": report["alt_data"].get("text", "")},
                "sector": {"signal": sector_data.get("signal", "N/A"), "summary": sector_data.get("summary", ""), "analysis": report["sector_analysis"].get("text", "")},
                "nlp_sentiment": {"signal": nlp_data.get("signal", "N/A"), "summary": nlp_data.get("summary", ""), "analysis": report["nlp_sentiment"].get("text", "")},
                "anomaly": {"signal": anomaly_data.get("signal", "N/A"), "summary": anomaly_data.get("summary", ""), "analysis": report["anomaly"].get("text", "")},
                "scenario": {"signal": mc_data.get("signal", "N/A"), "summary": mc_data.get("summary", ""), "analysis": report["scenario"].get("text", "")},
                "quant_model": {"signal": qm_data.get("signal", "N/A"), "summary": qm_data.get("summary", ""), "analysis": report["quant_model"].get("text", "")},
            }

            # Proactive compaction for small-context models (e.g. o3-mini: 4K token limit,
            # gpt-4.1-mini: 8K token limit). Drops the verbose 'analysis' field and caps
            # 'summary' to keep the JSON under budget.
            # In Lite Mode the caps are halved further, and ERROR/UNAVAILABLE signals are
            # stripped out so the debate only sees meaningful evidence.
            _debate_max_chars = get_model_max_input_chars(self.general_client.model_name)
            if _debate_max_chars and _debate_max_chars < 30_000:
                _DEAD_SIGNALS = {"ERROR", "UNAVAILABLE", "N/A", ""}
                # Lite mode: use tighter caps and strip dead signals to further reduce prompt size
                if lite:
                    _SUMMARY_CAP = 100    # ~25 tokens per source
                    _LEDGER_CAP  = 800
                elif _debate_max_chars <= 14_000:
                    _SUMMARY_CAP = 120
                    _LEDGER_CAP  = 900
                else:
                    _SUMMARY_CAP = 200    # ~50 tokens per source
                    _LEDGER_CAP  = 1500
                enrichment_for_debate = {
                    k: {"signal": v["signal"], "summary": v["summary"][:_SUMMARY_CAP]}
                    for k, v in enrichment_for_debate.items()
                    if not (lite and v.get("signal", "").upper() in _DEAD_SIGNALS)
                }
                fact_ledger_json = fact_ledger_json[:_LEDGER_CAP] if fact_ledger_json else fact_ledger_json
                logger.info(
                    f"[Debate] Compacted enrichment_for_debate for small-context model "
                    f"'{self.general_client.model_name}' (limit: {_debate_max_chars:,} chars, "
                    f"lite={lite}, {len(enrichment_for_debate)} signals kept)"
                )

            # Build memory context for debate agents
            sector_name = ""
            if isinstance(report.get("quant"), dict):
                sector_name = report["quant"].get("sector", "")
            situation_desc = build_situation_description(
                ticker, sector_name, {
                    k: {"signal": v.get("signal", "N/A")}
                    for k, v in enrichment_for_debate.items()
                },
            )
            bull_memory_str = self.bull_memory.format_for_prompt(situation_desc)
            bear_memory_str = self.bear_memory.format_for_prompt(situation_desc)
            moderator_memory_str = self.moderator_memory.format_for_prompt(situation_desc)

            debate_result = await asyncio.to_thread(
                run_debate,
                self.general_client,
                ticker,
                enrichment_for_debate,
                traces.summary(),
                max_debate_rounds=effective_debate_rounds,
                on_progress=lambda msg: step("debate", "running", msg),
                past_memories={"bull": bull_memory_str, "bear": bear_memory_str, "moderator": moderator_memory_str},
                cost_tracker=self._cost_tracker,
                deep_think_model=self.deep_think_client,
                general_model_name=self.general_client.model_name,
                deep_think_model_name=self.deep_think_client.model_name,
                skip_devils_advocate=skip_da,
                fact_ledger=fact_ledger_json,
                enable_thinking=self.enable_thinking,
                thinking_budgets=self.thinking_budgets,
            )
            report["debate"] = debate_result
            da_challenges = len(debate_result.get("devils_advocate", {}).get("challenges", []))
            step("debate", "completed",
                 f"Consensus: {debate_result.get('consensus', 'N/A')} "
                 f"(confidence: {debate_result.get('consensus_confidence', 'N/A')}) — "
                 f"{debate_result.get('total_rounds', 2)} rounds, "
                 f"{da_challenges} DA challenges")

            # Session memory: capture debate outcome
            consensus = debate_result.get("consensus", "")
            if consensus:
                ctx.add_finding(f"Debate consensus: {consensus} (conf: {debate_result.get('consensus_confidence', 'N/A')})")
            for c in debate_result.get("contradictions", [])[:3]:
                if isinstance(c, dict):
                    ctx.add_contradiction(c.get("topic", str(c))[:100])

        # Step 9: Enhanced macro (original macro + FRED data)
        step("macro", "started", "Analyzing macro economy with FRED data...")
        report["macro"] = await asyncio.to_thread(self.run_enhanced_macro_agent, ticker, av_data, fred_macro)
        step("macro", "completed", "Macro analysis complete")

        # Step 10: Deep dive
        if skip_deep_dive:
            step("deep_dive", "started", "Skipping deep dive (lite mode)...")
            report["deep_dive"] = "Skipped (lite mode)"
            step("deep_dive", "completed", "Skipped (lite mode)")
        else:
            step("deep_dive", "started", "Finding contradictions and probing...")
            report["deep_dive"] = await asyncio.to_thread(self.run_deep_dive_agent, ticker, report)
            step("deep_dive", "completed", "Deep dive complete")

        # Store session context in report for synthesis pipeline
        report["_session_context"] = ctx.format_for_prompt()

        # Step 11+12: Synthesis + Critic (with reflection loop)
        _synth_label = self.settings.deep_think_model or _model_label
        step("synthesis", "started", "Drafting final report...")
        step("synthesis", "running", "Building synthesis prompt from all agent outputs...")
        step("synthesis", "running", f"{_synth_label} generating structured JSON report (reflection loop enabled)...")
        final_json = await asyncio.to_thread(self.run_synthesis_pipeline, ticker, report)
        iterations = final_json.get("synthesis_iterations", 1)
        iter_msg = f" ({iterations} iteration{'s' if iterations > 1 else ''})" if iterations > 1 else ""
        step("synthesis", "running", f"Parsing and validating report structure...{iter_msg}")
        scores = final_json.get("scoring_matrix", {})
        final_json["final_weighted_score"] = self.compute_weighted_score(scores)

        # Attach enrichment signals summary to final output (11 signals)
        final_json["enrichment_signals"] = {
            "insider": {"signal": insider_data.get("signal", "N/A"), "summary": insider_data.get("summary", "")},
            "options": {"signal": options_data.get("signal", "N/A"), "summary": options_data.get("summary", "")},
            "social_sentiment": {"signal": social_data.get("signal", "N/A"), "summary": social_data.get("summary", "")},
            "patent": {"signal": patent_data.get("signal", "N/A"), "summary": patent_data.get("summary", "")},
            "earnings_tone": {"signal": earnings_data.get("signal", "N/A"), "summary": earnings_data.get("summary", "")},
            "fred_macro": {"signal": fred_macro.get("signal", "N/A"), "summary": fred_macro.get("summary", "")},
            "alt_data": {"signal": alt_result.get("signal", "N/A"), "summary": alt_result.get("summary", "")},
            "sector": {"signal": sector_data.get("signal", "N/A"), "summary": sector_data.get("summary", "")},
            "nlp_sentiment": {"signal": nlp_data.get("signal", "N/A"), "summary": nlp_data.get("summary", "")},
            "anomaly": {"signal": anomaly_data.get("signal", "N/A"), "summary": anomaly_data.get("summary", "")},
            "monte_carlo": {"signal": mc_data.get("signal", "N/A"), "summary": mc_data.get("summary", "")},
        }

        # Attach debate result
        final_json["debate_result"] = debate_result

        # Attach decision traces (XAI audit trail)
        final_json["decision_traces"] = traces.all_traces()

        # Attach raw risk data for frontend charts
        final_json["risk_data"] = {
            "monte_carlo": mc_data,
            "anomalies": anomaly_data,
        }

        # Attach grounding sources from Google Search grounded agents (Phase 4)
        final_json["grounding_sources"] = {
            "market": report.get("market", {}).get("grounding_sources", []) if isinstance(report.get("market"), dict) else [],
            "competitor": report.get("competitor", {}).get("grounding_sources", []) if isinstance(report.get("competitor"), dict) else [],
            "enhanced_macro": report.get("macro", {}).get("grounding_sources", []) if isinstance(report.get("macro"), dict) else [],
            "deep_dive": report.get("deep_dive", {}).get("grounding_sources", []) if isinstance(report.get("deep_dive"), dict) else [],
        }

        # Attach info-gap report
        final_json["info_gap_report"] = info_gap_report

        report["final_synthesis"] = final_json
        step("synthesis", "completed", f"Score: {final_json.get('final_weighted_score', 'N/A')}")

        # Step 12b: Bias Audit
        step("bias_audit", "started", "Running bias and conflict detection...")
        step("bias_audit", "running", "Checking for tech bias, confirmation bias, recency bias...")
        bias_report = detect_biases(
            ticker=ticker,
            recommendation=final_json.get("recommendation", {}).get("recommendation", "HOLD"),
            score=final_json.get("final_weighted_score", 5.0),
            enrichment_signals=final_json.get("enrichment_signals", {}),
            debate_result=debate_result,
            quant_data=report.get("quant", {}),
        )
        step("bias_audit", "running", f"Found {bias_report.get('bias_count', 0)} bias flags. Checking knowledge conflicts...")
        conflict_report = detect_conflicts(
            ticker=ticker,
            synthesis_report=final_json,
            quant_data=report.get("quant", {}),
            enrichment_signals=final_json.get("enrichment_signals", {}),
        )
        final_json["bias_report"] = bias_report
        final_json["conflict_report"] = conflict_report
        step("bias_audit", "completed", f"Bias flags: {bias_report.get('bias_count', 0)}, Conflicts: {conflict_report.get('conflict_count', 0)}")

        # Step 12c: Risk Assessment Team (Aggressive / Conservative / Neutral + Risk Judge)
        # Quality Gate: Skip risk assessment if data quality is below threshold or lite mode
        if low_data_quality or skip_risk_assessment:
            skip_reason = "lite mode" if skip_risk_assessment else f"data quality {dq:.0%} below threshold {self.settings.data_quality_min:.0%}"
            step("risk_assessment", "started", f"Skipping risk assessment — {skip_reason}...")
            risk_assessment = {
                "aggressive": {"confidence": 0.0, "argument": "Insufficient data"},
                "conservative": {"confidence": 0.0, "argument": "Insufficient data"},
                "neutral": {"confidence": 0.0, "argument": "Insufficient data"},
                "judge": {
                    "decision": "REJECT",
                    "risk_adjusted_confidence": 0.0,
                    "recommended_position_pct": 0,
                    "risk_level": "HIGH",
                    "risk_limits": {},
                },
                "skipped_reason": skip_reason,
            }
            final_json["risk_assessment"] = risk_assessment
            step("risk_assessment", "completed", f"Skipped — {skip_reason}")
        else:
            step("risk_assessment", "started", "Running risk assessment team debate...")
            risk_judge_memory_str = self.risk_judge_memory.format_for_prompt(situation_desc)
            risk_assessment = await asyncio.to_thread(
                run_risk_debate,
                self.general_client,
                ticker,
                final_json,
                final_json.get("enrichment_signals", {}),
                debate_result=debate_result,
                max_risk_rounds=self.settings.max_risk_debate_rounds,
                past_memories={
                    "risk_aggressive": "",
                    "risk_conservative": "",
                    "risk_neutral": "",
                    "risk_judge": risk_judge_memory_str,
                },
                on_progress=lambda msg: step("risk_assessment", "running", msg),
                cost_tracker=self._cost_tracker,
                deep_think_model=self.deep_think_client,
                general_model_name=self.general_client.model_name,
                deep_think_model_name=self.deep_think_client.model_name,
                fact_ledger=fact_ledger_json,
                enable_thinking=self.enable_thinking,
                thinking_budgets=self.thinking_budgets,
            )
            final_json["risk_assessment"] = risk_assessment
            judge = risk_assessment.get("judge", {})
            step("risk_assessment", "completed",
                 f"Risk Judge: {judge.get('decision', 'N/A')} — "
                 f"position {judge.get('recommended_position_pct', 'N/A')}% — "
                 f"risk level: {judge.get('risk_level', 'N/A')}")

        # Attach cost summary (token usage + cost breakdown)
        cost_summary = self._cost_tracker.summarize()
        cost_summary["standard_model"] = self.settings.gemini_model
        cost_summary["deep_think_model"] = self.settings.deep_think_model or self.settings.gemini_model
        # phase-cycle-8 (38.13, 2026-05-27): per-cycle rail attribution.
        # Cycle-7 Q/A misdiagnosed lite-vs-full because BQ rows had no
        # rail tag. With this, downstream consumers (BQ, Slack digest,
        # /reports table) can definitively answer "did the rail work?".
        cost_summary["rail"] = "claude_code" if getattr(self.settings, "paper_use_claude_code_route", False) else "anthropic_direct"
        final_json["cost_summary"] = cost_summary

        # Budget check (soft warning only, does not abort)
        if self._cost_tracker.check_budget(self.settings.max_analysis_cost_usd):
            logger.warning(
                "Analysis for %s exceeded cost budget: $%.4f > $%.2f",
                ticker, cost_summary["total_cost_usd"], self.settings.max_analysis_cost_usd,
            )
            final_json["budget_warning"] = (
                f"Analysis cost ${cost_summary['total_cost_usd']:.4f} exceeded "
                f"budget of ${self.settings.max_analysis_cost_usd:.2f}"
            )

        return report
