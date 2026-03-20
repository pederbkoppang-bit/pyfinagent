"""
Agent orchestrator — drives the full analysis pipeline.
Migrated from pyfinagent-app/Home.py with all Streamlit dependencies removed.
"""

import asyncio
import concurrent.futures
import json
import logging
import re
import time
from typing import Optional

import httpx
import vertexai
from google.api_core import exceptions as gcp_exceptions
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel, Tool, grounding, GenerationConfig

from backend.agents.bias_detector import detect_biases
from backend.agents.cost_tracker import CostTracker
from backend.agents.debate import run_debate
from backend.agents.llm_client import GeminiClient, LLMClient, LLMResponse, make_client
from backend.agents.conflict_detector import detect_conflicts
from backend.agents.info_gap import detect_info_gaps, retry_critical_gaps
from backend.agents.memory import (
    FinancialSituationMemory,
    build_situation_description,
    generate_reflection,
)
from backend.agents.risk_debate import run_risk_debate
from backend.agents.schemas import SynthesisReport, CriticVerdict
from backend.agents.trace import AnalysisContext, DecisionTrace, TraceCollector, hash_input
from backend.config import prompts
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
_SYNTHESIS_STRUCTURED_CONFIG = {
    "temperature": 0.0, "top_k": 1, "max_output_tokens": 4096,
    "response_mime_type": "application/json",
    "response_schema": SynthesisReport,
}
_CRITIC_STRUCTURED_CONFIG = {
    "temperature": 0.0, "top_k": 1, "max_output_tokens": 2048,
    "response_mime_type": "application/json",
    "response_schema": CriticVerdict,
}

# Thinking configs for judge agents (Phase 5 — Gemini 2.5 Flash extended thinking)
# Thinking overrides temperature on Gemini 2.5+; non-thinking agents keep temp=0.0
_THINKING_CRITIC_CONFIG = {
    "temperature": 0.0, "top_k": 1, "max_output_tokens": 2048,
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
    "temperature": 0.0, "top_k": 1, "max_output_tokens": 1536,
    "thinking": {"type": "enabled", "budget_tokens": 4096},
    "include_thoughts": True,
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


def _parse_json_with_fallback(json_string: str, agent_name: str) -> Optional[dict]:
    try:
        data = json.loads(json_string)
        if isinstance(data, str):
            return json.loads(data)
        return data
    except json.JSONDecodeError:
        logger.warning(f"{agent_name} returned invalid JSON")
        return None


def _build_fact_ledger(quant_data: dict) -> dict:
    """Build typed fact dict from Step 2 quant + yfinance data.

    Research: VeNRA achieves 1.2% hallucination rate with typed fact ledger.
    All agents receive this as ground truth — no invented numbers allowed.
    """
    yf = quant_data.get("yf_data", {})
    val = yf.get("valuation", {})
    eff = yf.get("efficiency", {})
    health = yf.get("health", {})
    inst = yf.get("institutional", {})
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

    def __init__(self, settings: Settings):
        self.settings = settings

        # Build GCP credentials
        credentials = None
        if settings.gcp_credentials_json:
            creds_info = json.loads(settings.gcp_credentials_json)
            credentials = service_account.Credentials.from_service_account_info(creds_info)

        # Initialize Vertex AI
        vertexai.init(
            project=settings.gcp_project_id,
            location=settings.gcp_location,
            credentials=credentials,
        )

        # Build models
        datastore_path = (
            f"projects/{settings.gcp_project_id}/locations/global/collections/"
            f"default_collection/dataStores/{settings.rag_data_store_id}"
        )
        rag_tool = Tool.from_retrieval(
            grounding.Retrieval(grounding.VertexAISearch(datastore=datastore_path))
        )

        _gen_config = {"temperature": 0.0, "top_k": 1}
        _enrichment_config = {"temperature": 0.0, "top_k": 1, "max_output_tokens": 1024}
        _synthesis_config = {"temperature": 0.0, "top_k": 1, "max_output_tokens": 4096}
        _deep_think_config = {"temperature": 0.0, "top_k": 1, "max_output_tokens": 2048}
        self.rag_model = GenerativeModel(settings.gemini_model, tools=[rag_tool], generation_config=_gen_config)

        # --- v3.4 Multi-Provider LLM Clients ---
        # Build Vertex AI base models (used as Gemini fallback or the actual model)
        deep_model_name = settings.deep_think_model or settings.gemini_model
        _general_vertex = GenerativeModel(settings.gemini_model, generation_config=_enrichment_config)
        _dt_vertex = GenerativeModel(deep_model_name, generation_config=_deep_think_config)
        _synth_vertex = GenerativeModel(deep_model_name, generation_config=_synthesis_config)

        # Route each model through provider factory (may use Claude/OpenAI/GitHub Models)
        self.general_client: LLMClient = make_client(settings.gemini_model, _general_vertex, settings)
        self.deep_think_client: LLMClient = make_client(deep_model_name, _dt_vertex, settings)
        self.synthesis_client: LLMClient = make_client(deep_model_name, _synth_vertex, settings)

        # RAG model always uses Gemini (Vertex AI Search constraint)
        self.rag_client: GeminiClient = GeminiClient(self.rag_model, settings.gemini_model)

        # Google Search Grounding model — always Gemini (grounding is a Google-specific feature)
        # Constraint: Schema + Grounding cannot combine on Gemini 2.0 — separate instance
        search_tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())
        _grounded_vertex = GenerativeModel(settings.gemini_model, tools=[search_tool], generation_config=_gen_config)
        self.grounded_client: GeminiClient = GeminiClient(_grounded_vertex, settings.gemini_model)

        # Grounded calls fall back to general_client when non-Gemini provider is selected
        self.supports_grounding = isinstance(self.general_client, GeminiClient)

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

        # Phase 5: Add thinking config for judge agents if enabled (Gemini-specific)
        final_config = generation_config
        if isinstance(model, GeminiClient) and self.enable_thinking and is_deep_think and agent_name in self.thinking_budgets:
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
                    ct.record(agent_name, model_name, response, is_deep_think=is_deep_think, is_grounded=is_grounded)
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

    # ── Pipeline Steps ───────────────────────────────────────────────

    async def fetch_market_intel(self, ticker: str) -> dict:
        """Step 0: Fetch Alpha Vantage data."""
        return await alphavantage.get_market_intel(ticker, self.settings.alphavantage_api_key)

    def fetch_yfinance_data(self, ticker: str) -> dict:
        """Step 0b: Fetch yfinance fundamentals."""
        return yfinance_tool.get_comprehensive_financials(ticker)

    async def run_ingestion_agent(self, ticker: str) -> bool:
        """Step 1: Call the Ingestion Agent Cloud Function."""
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
                        final_json = json.loads(json_str)
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
        logger.info(f"RAG Agent: analyzing documents for {ticker}")
        prompt = prompts.get_rag_prompt(ticker, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        response = self._generate_with_retry(self.rag_client, prompt, "RAG")
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
        response = self._generate_with_retry(self.general_client, prompt, "Insider")
        return {"text": _extract_text(response), "data": insider_data}

    def run_options_agent(self, ticker: str, options_data: dict) -> dict:
        """Analyze options flow for institutional signals."""
        logger.info(f"Options Agent: analyzing flow for {ticker}")
        prompt = prompts.get_options_prompt(ticker, options_data, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        response = self._generate_with_retry(self.general_client, prompt, "Options")
        return {"text": _extract_text(response), "data": options_data}

    def run_social_sentiment_agent(self, ticker: str, sentiment_data: dict) -> dict:
        """Analyze social media and news sentiment."""
        logger.info(f"Social Sentiment Agent: analyzing for {ticker}")
        prompt = prompts.get_social_sentiment_prompt(ticker, sentiment_data, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        response = self._generate_with_retry(self.general_client, prompt, "Social Sentiment")
        return {"text": _extract_text(response), "data": sentiment_data}

    def run_patent_agent(self, ticker: str, patent_data: dict) -> dict:
        """Analyze patent/innovation data."""
        logger.info(f"Patent Agent: analyzing innovation for {ticker}")
        prompt = prompts.get_patent_prompt(ticker, patent_data, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        response = self._generate_with_retry(self.general_client, prompt, "Patent")
        return {"text": _extract_text(response), "data": patent_data}

    def run_earnings_tone_agent(self, ticker: str, transcript_data: dict) -> dict:
        """Analyze earnings call tone."""
        logger.info(f"Earnings Tone Agent: analyzing transcript for {ticker}")
        prompt = prompts.get_earnings_tone_prompt(ticker, transcript_data, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        response = self._generate_with_retry(self.general_client, prompt, "Earnings Tone")
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
        response = self._generate_with_retry(self.general_client, prompt, "Alt Data")
        return {"text": _extract_text(response), "data": alt_data_result}

    def run_sector_analysis_agent(self, ticker: str, sector_data: dict) -> dict:
        """Analyze sector relative strength and rotation."""
        logger.info(f"Sector Agent: analyzing sector for {ticker}")
        prompt = prompts.get_sector_analysis_prompt(ticker, sector_data, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        response = self._generate_with_retry(self.general_client, prompt, "Sector")
        return {"text": _extract_text(response), "data": sector_data}

    def run_nlp_sentiment_agent(self, ticker: str, nlp_data: dict) -> dict:
        """Analyze transformer-based NLP sentiment."""
        logger.info(f"NLP Sentiment Agent: analyzing for {ticker}")
        prompt = prompts.get_nlp_sentiment_prompt(ticker, nlp_data, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        response = self._generate_with_retry(self.general_client, prompt, "NLP Sentiment")
        return {"text": _extract_text(response), "data": nlp_data}

    def run_anomaly_agent(self, ticker: str, anomaly_data: dict) -> dict:
        """Analyze statistical anomalies."""
        logger.info(f"Anomaly Agent: analyzing for {ticker}")
        prompt = prompts.get_anomaly_detection_prompt(ticker, anomaly_data, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        response = self._generate_with_retry(self.general_client, prompt, "Anomaly")
        return {"text": _extract_text(response), "data": anomaly_data}

    def run_scenario_agent(self, ticker: str, mc_data: dict) -> dict:
        """Analyze Monte Carlo scenario results."""
        logger.info(f"Scenario Agent: analyzing risk for {ticker}")
        prompt = prompts.get_scenario_analysis_prompt(ticker, mc_data, fact_ledger=getattr(self, '_fact_ledger_json', ''))
        response = self._generate_with_retry(self.general_client, prompt, "Scenario")
        return {"text": _extract_text(response), "data": mc_data}

    # ── Synthesis ────────────────────────────────────────────────────

    def run_synthesis_pipeline(self, ticker: str, report: dict) -> dict:
        """Synthesis + Critic reflection loop → final validated JSON.

        Implements Evaluator-Optimizer pattern: Synthesis drafts, Critic reviews
        with structured verdict. If REVISE, Synthesis re-runs with critic feedback
        (up to max_synthesis_iterations). First pass uses deep-think; re-runs also use deep-think.
        """
        max_iterations = self.settings.max_synthesis_iterations
        logger.info(f"Synthesis Agent: drafting report for {ticker} (max {max_iterations} iterations)")

        # Build market_context with per-section truncation guard
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
        synthesis_kwargs: dict = {
            "ticker": ticker,
            "quant_report": quant_data,
            "rag_report": report.get("rag", {}).get("text", ""),
            "market_report": market_context,
            "sector_catalyst_report": sector_catalyst,
            "supply_chain_report": report.get("supply_chain", "No supply chain data."),
            "deep_dive_analysis": report.get("deep_dive", {}).get("text", "") if isinstance(report.get("deep_dive"), dict) else report.get("deep_dive", ""),
            "fact_ledger": fact_ledger_json,
        }

        # Initial synthesis draft (uses synthesis_model with 4096 output token limit)
        draft_prompt = prompts.get_synthesis_prompt(**synthesis_kwargs)
        draft_response = self._generate_with_retry(
            self.synthesis_client, draft_prompt, "Synthesis",
            is_deep_think=True, generation_config=_SYNTHESIS_STRUCTURED_CONFIG,
        )
        draft_text = _clean_json_output(_extract_text(draft_response))

        synthesis_iterations = 1
        critic_issues_log = []

        for iteration in range(max_iterations):
            # Critic review with structured verdict
            logger.info(f"Critic Agent: reviewing draft for {ticker} (iteration {iteration + 1})")
            critic_feedback_str = ""
            if iteration > 0:
                critic_feedback_str = json.dumps(critic_issues_log[-1], indent=2)

            critic_prompt = prompts.get_critic_prompt(ticker, draft_text, quant_data, critic_feedback=critic_feedback_str, fact_ledger=fact_ledger_json)
            critic_response = self._generate_with_retry(
                self.deep_think_client, critic_prompt, "Critic",
                is_deep_think=True, generation_config=_CRITIC_STRUCTURED_CONFIG,
            )
            critic_text = _clean_json_output(_extract_text(critic_response))

            # Parse structured Critic verdict
            critic_result = _parse_json_with_fallback(critic_text, "Critic")
            if not critic_result:
                logger.warning("Critic returned invalid JSON, treating as PASS with draft.")
                break

            verdict = critic_result.get("verdict", "PASS").upper()
            issues = critic_result.get("issues", [])
            corrected_report = critic_result.get("corrected_report")
            major_issues = [i for i in issues if i.get("severity") == "major"]

            logger.info(f"Critic verdict: {verdict} — {len(major_issues)} major, {len(issues) - len(major_issues)} minor issues")

            if verdict == "PASS" or not major_issues:
                # Accept the corrected report (or draft if no corrected_report)
                if corrected_report and isinstance(corrected_report, dict):
                    corrected_report["synthesis_iterations"] = synthesis_iterations
                    corrected_report["critic_issues"] = issues
                    return corrected_report
                break

            # REVISE: Check if we have iterations left
            if iteration >= max_iterations - 1:
                logger.info("Max synthesis iterations reached, accepting Critic's corrected report.")
                if corrected_report and isinstance(corrected_report, dict):
                    corrected_report["synthesis_iterations"] = synthesis_iterations
                    corrected_report["critic_issues"] = issues
                    return corrected_report
                break

            # Re-run Synthesis with Critic feedback
            critic_issues_log.append(issues)
            synthesis_iterations += 1
            logger.info(f"Synthesis Agent: revising report for {ticker} (iteration {synthesis_iterations})")

            revision_prompt = prompts.get_synthesis_revision_prompt(
                **synthesis_kwargs,
                critic_issues=issues,
                previous_draft=draft_text,
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
            return final_data

        logger.warning("Failed to parse final report, returning error.")
        return {"error": "Failed to parse final report.", "synthesis_iterations": synthesis_iterations}

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

        # Step 1: Ingestion agent
        step("ingestion", "started", "Checking for new filings...")
        await self.run_ingestion_agent(ticker)
        step("ingestion", "completed", "Filings up to date")

        # Step 2: Quant agent
        step("quant", "started", "Fetching financial data...")
        report["quant"] = await self.run_quant_agent(ticker)
        company_name = report["quant"].get("company_name", ticker)
        step("quant", "completed", "Financial data collected")

        # Session memory: capture key quant findings
        quant = report["quant"]
        if isinstance(quant, dict):
            pe = quant.get("pe_ratio") or quant.get("valuation", {}).get("P/E Ratio")
            if pe:
                ctx.add_finding(f"P/E ratio: {pe}")
            sector_name = quant.get("sector", "")
            if sector_name:
                ctx.add_finding(f"Sector: {sector_name}")

        # Build fact ledger from quant data — injected into ALL agent prompts
        # Research: VeNRA typed fact ledger achieves 1.2% hallucination rate
        fact_ledger = _build_fact_ledger(report["quant"])
        fact_ledger_json = json.dumps(fact_ledger, indent=2, default=str)
        report["_fact_ledger"] = fact_ledger
        self._fact_ledger_json = fact_ledger_json  # available to all agent methods

        # Step 3: RAG agent
        step("rag", "started", "Analyzing 10-K/10-Q documents...")
        report["rag"] = await asyncio.to_thread(self.run_rag_agent, ticker)
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

        # Step 6: Fetch enrichment data in parallel (11 non-LLM calls)
        step("data_enrichment", "started", "Fetching 11 enrichment data sources in parallel...")
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
                step("data_enrichment", "running", f"[{_enrichment_done_count}/11] {label} data collected")
                return result
            except Exception as e:
                logger.warning(f"{label} data fetch failed: {e}")
                _enrichment_done_count += 1
                _enrichment_labels_done.append(f"{label} (error)")
                step("data_enrichment", "running", f"[{_enrichment_done_count}/11] {label} failed: {e}")
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
                logger.info("AV empty for %s — using %d yfinance articles as fallback", ticker, len(fallback_articles))

        # Sector routing: determine which tools to skip
        sector_for_routing = ""
        if isinstance(report.get("quant"), dict):
            sector_for_routing = report["quant"].get("sector", "")
        skipped_tools = SECTOR_SKIP_MAP.get(sector_for_routing, set())
        if skipped_tools:
            logger.info(f"Sector routing: {sector_for_routing} → skipping tools: {skipped_tools}")
            ctx.add_finding(f"Sector routing: skipping {', '.join(skipped_tools)} for {sector_for_routing}")

        async def _skip_placeholder(label: str):
            """Return a placeholder for skipped tools."""
            nonlocal _enrichment_done_count
            _enrichment_done_count += 1
            _enrichment_labels_done.append(f"{label} (skipped)")
            step("data_enrichment", "running", f"[{_enrichment_done_count}/11] {label} skipped (sector routing)")
            return {"signal": "SKIPPED", "summary": f"Skipped: not relevant for {sector_for_routing} sector"}

        (
            insider_data, options_data, social_data, patent_data,
            earnings_data, fred_macro, alt_result, sector_data,
            nlp_data, anomaly_data, mc_data,
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
        )
        step("data_enrichment", "completed", "All 11 enrichment data sources collected")

        # Session memory: capture enrichment signals
        for src, data in [("insider", insider_data), ("options", options_data),
                          ("social", social_data), ("patent", patent_data),
                          ("sector", sector_data), ("nlp", nlp_data),
                          ("anomaly", anomaly_data), ("monte_carlo", mc_data)]:
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
            "monte_carlo": mc_data,
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
        step("enrichment_analysis", "started", "Running 11 enrichment analysis agents...")

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
        ]
        _done_count = 0
        _total = len(_agent_list)

        async def _run_one(key, func, name, data):
            nonlocal _done_count
            step("enrichment_analysis", "running", f"Gemini analyzing {name}...")
            try:
                result = await asyncio.to_thread(_run_agent_with_trace, func, name, ticker, ticker, data)
            except Exception as e:
                logger.error(f"Enrichment agent {name} failed: {e}")
                result = {"text": f"Error: {e}", "data": data}
            _done_count += 1
            signal = data.get('signal', 'N/A') if isinstance(data, dict) else 'done'
            step("enrichment_analysis", "running", f"[{_done_count}/{_total}] {name} → {signal}")
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
            }
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
        step("synthesis", "started", "Drafting final report...")
        step("synthesis", "running", "Building synthesis prompt from all agent outputs...")
        step("synthesis", "running", "Gemini generating structured JSON report (reflection loop enabled)...")
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
