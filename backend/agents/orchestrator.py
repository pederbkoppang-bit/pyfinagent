"""
Agent orchestrator — drives the full analysis pipeline.
Migrated from pyfinagent-app/Home.py with all Streamlit dependencies removed.
"""

import asyncio
import json
import logging
import re
import time
from typing import Optional

import httpx
import vertexai
from google.api_core import exceptions as gcp_exceptions
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel, Tool, grounding

from backend.config import prompts
from backend.config.settings import Settings
from backend.tools import alphavantage, yfinance_tool

logger = logging.getLogger(__name__)


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

        self.rag_model = GenerativeModel(settings.gemini_model, tools=[rag_tool])
        self.general_model = GenerativeModel(settings.gemini_model)

    # ── Helpers ──────────────────────────────────────────────────────

    def _generate_with_retry(self, model, prompt: str, agent_name: str, max_retries: int = 3):
        delay = 5
        for attempt in range(max_retries):
            try:
                return model.generate_content(prompt)
            except (gcp_exceptions.ServiceUnavailable, gcp_exceptions.ResourceExhausted) as e:
                if attempt < max_retries - 1:
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
        prompt = prompts.get_rag_prompt(ticker)
        response = self._generate_with_retry(self.rag_model, prompt, "RAG")

        citations = []
        if hasattr(response, "grounding_metadata") and response.grounding_metadata:
            for cit in response.grounding_metadata.grounding_attributions:
                citations.append({"uri": cit.web.uri, "title": cit.web.title})

        return {"text": response.text, "citations": citations}

    def run_market_agent(self, ticker: str, av_data: dict) -> dict:
        """Step 4: Market sentiment analysis."""
        logger.info(f"Market Agent: analyzing sentiment for {ticker}")
        prompt = prompts.get_market_prompt(ticker, av_data)
        response = self._generate_with_retry(self.general_model, prompt, "Market")
        return {"text": response.text}

    def run_competitor_agent(self, ticker: str, av_data: dict) -> dict:
        """Step 5: Competitor analysis."""
        logger.info(f"Competitor Agent: analyzing rivals for {ticker}")
        prompt = prompts.get_competitor_prompt(ticker, av_data)
        response = self._generate_with_retry(self.general_model, prompt, "Competitor")
        return {"text": response.text}

    def run_macro_agent(self, ticker: str, av_data: dict) -> dict:
        """Step 6: Macroeconomic analysis."""
        logger.info(f"Macro Agent: analyzing economy for {ticker}")
        prompt = prompts.get_macro_prompt(ticker, av_data)
        response = self._generate_with_retry(self.general_model, prompt, "Macro")
        return {"text": response.text}

    def run_deep_dive_agent(self, ticker: str, report: dict) -> str:
        """Step 7: Find contradictions and probe with follow-up questions."""
        logger.info(f"Deep Dive Agent: probing contradictions for {ticker}")

        prompt = prompts.get_deep_dive_prompt(
            ticker,
            report.get("quant", {}),
            report.get("rag", {}).get("text", ""),
            report.get("market", {}).get("text", ""),
            report.get("competitor", {}).get("text", "No competitor data."),
        )
        response = self._generate_with_retry(self.general_model, prompt, "Deep Dive (Questions)")
        questions = response.text.strip().split("\n")

        answers = []
        for q in questions:
            if q.strip():
                logger.info(f"Deep Dive investigating: {q.strip()}")
                ans = self._generate_with_retry(
                    self.rag_model, f"Answer this using 10-K: {q}", "Deep Dive (Answers)"
                )
                time.sleep(2)  # Rate limit protection
                answers.append(f"Q: {q}\nA: {ans.text}")
        return "\n\n".join(answers)

    def run_synthesis_pipeline(self, ticker: str, report: dict) -> dict:
        """Step 8+9: Synthesis + Critic pipeline → final validated JSON."""
        logger.info(f"Synthesis Agent: drafting report for {ticker}")

        market_context = (
            f"Market News & Sentiment:\n{report.get('market', {}).get('text', '')}\n\n"
            f"Competitor Analysis:\n{report.get('competitor', {}).get('text', 'No competitor data.')}\n\n"
            f"Macroeconomic Outlook:\n{report.get('macro', {}).get('text', 'No macro data.')}"
        )

        draft_prompt = prompts.get_synthesis_prompt(
            ticker,
            report.get("quant", {}),
            report.get("rag", {}).get("text", ""),
            market_context,
            report.get("sector_catalyst", "No sector catalyst data."),
            report.get("supply_chain", "No supply chain data."),
            report.get("deep_dive", ""),
        )
        draft_response = self._generate_with_retry(self.general_model, draft_prompt, "Synthesis")
        draft_text = _clean_json_output(draft_response.text)

        # Critic pass
        logger.info(f"Critic Agent: reviewing draft for {ticker}")
        critic_prompt = prompts.get_critic_prompt(ticker, draft_text, report.get("quant", {}))
        final_response = self._generate_with_retry(self.general_model, critic_prompt, "Critic")
        final_text = _clean_json_output(final_response.text)

        final_data = _parse_json_with_fallback(final_text, "Critic")
        if final_data:
            return final_data

        logger.warning("Critic returned invalid JSON, falling back to draft.")
        draft_data = _parse_json_with_fallback(draft_text, "Synthesis")
        if draft_data:
            return draft_data

        return {"error": "Failed to parse final report from both agents."}

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
        Executes the complete analysis pipeline end-to-end.
        
        Args:
            ticker: Stock ticker symbol
            on_step: Optional callback(step_name, status, message) for progress updates
        
        Returns:
            Complete report dict with all agent outputs and final synthesis.
        """

        def step(name, status, msg=""):
            if on_step:
                on_step(name, status, msg)

        report = {}

        # Step 0: Fetch external data
        step("market_intel", "started", "Fetching Alpha Vantage data...")
        av_data = await self.fetch_market_intel(ticker)
        step("market_intel", "completed", f"Got {len(av_data.get('sentiment_summary', []))} articles")

        # Step 1: Ingestion agent
        step("ingestion", "started", "Checking for new filings...")
        await self.run_ingestion_agent(ticker)
        step("ingestion", "completed", "Filings up to date")

        # Step 2: Quant agent
        step("quant", "started", "Fetching financial data...")
        report["quant"] = await self.run_quant_agent(ticker)
        step("quant", "completed", "Financial data collected")

        # Step 3: RAG agent
        step("rag", "started", "Analyzing 10-K/10-Q documents...")
        report["rag"] = await asyncio.to_thread(self.run_rag_agent, ticker)
        step("rag", "completed", "Document analysis complete")

        # Step 4: Market agent
        step("market", "started", "Analyzing sentiment...")
        report["market"] = await asyncio.to_thread(self.run_market_agent, ticker, av_data)
        step("market", "completed", "Sentiment analysis complete")

        # Step 5: Competitor agent
        step("competitor", "started", "Analyzing rivals...")
        report["competitor"] = await asyncio.to_thread(self.run_competitor_agent, ticker, av_data)
        step("competitor", "completed", "Competitor analysis complete")

        # Step 6: Macro agent
        step("macro", "started", "Analyzing macro economy...")
        report["macro"] = await asyncio.to_thread(self.run_macro_agent, ticker, av_data)
        step("macro", "completed", "Macro analysis complete")

        # Step 7: Deep dive
        step("deep_dive", "started", "Finding contradictions and probing...")
        report["deep_dive"] = await asyncio.to_thread(self.run_deep_dive_agent, ticker, report)
        step("deep_dive", "completed", "Deep dive complete")

        # Step 8+9: Synthesis + Critic
        step("synthesis", "started", "Drafting final report...")
        final_json = await asyncio.to_thread(self.run_synthesis_pipeline, ticker, report)
        scores = final_json.get("scoring_matrix", {})
        final_json["final_weighted_score"] = self.compute_weighted_score(scores)
        report["final_synthesis"] = final_json
        step("synthesis", "completed", f"Score: {final_json.get('final_weighted_score', 'N/A')}")

        return report
