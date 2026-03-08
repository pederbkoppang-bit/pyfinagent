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

from backend.agents.bias_detector import detect_biases
from backend.agents.debate import run_debate
from backend.agents.conflict_detector import detect_conflicts
from backend.agents.trace import DecisionTrace, TraceCollector, hash_input
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
        prompt = prompts.get_insider_prompt(ticker, insider_data)
        response = self._generate_with_retry(self.general_model, prompt, "Insider")
        return {"text": response.text, "data": insider_data}

    def run_options_agent(self, ticker: str, options_data: dict) -> dict:
        """Analyze options flow for institutional signals."""
        logger.info(f"Options Agent: analyzing flow for {ticker}")
        prompt = prompts.get_options_prompt(ticker, options_data)
        response = self._generate_with_retry(self.general_model, prompt, "Options")
        return {"text": response.text, "data": options_data}

    def run_social_sentiment_agent(self, ticker: str, sentiment_data: dict) -> dict:
        """Analyze social media and news sentiment."""
        logger.info(f"Social Sentiment Agent: analyzing for {ticker}")
        prompt = prompts.get_social_sentiment_prompt(ticker, sentiment_data)
        response = self._generate_with_retry(self.general_model, prompt, "Social Sentiment")
        return {"text": response.text, "data": sentiment_data}

    def run_patent_agent(self, ticker: str, patent_data: dict) -> dict:
        """Analyze patent/innovation data."""
        logger.info(f"Patent Agent: analyzing innovation for {ticker}")
        prompt = prompts.get_patent_prompt(ticker, patent_data)
        response = self._generate_with_retry(self.general_model, prompt, "Patent")
        return {"text": response.text, "data": patent_data}

    def run_earnings_tone_agent(self, ticker: str, transcript_data: dict) -> dict:
        """Analyze earnings call tone."""
        logger.info(f"Earnings Tone Agent: analyzing transcript for {ticker}")
        prompt = prompts.get_earnings_tone_prompt(ticker, transcript_data)
        response = self._generate_with_retry(self.general_model, prompt, "Earnings Tone")
        return {"text": response.text, "data": transcript_data}

    def run_enhanced_macro_agent(self, ticker: str, av_data: dict, fred_macro: dict) -> dict:
        """Enhanced macro analysis with FRED data."""
        logger.info(f"Enhanced Macro Agent: analyzing economy for {ticker}")
        prompt = prompts.get_enhanced_macro_prompt(ticker, av_data, fred_macro)
        response = self._generate_with_retry(self.general_model, prompt, "Enhanced Macro")
        return {"text": response.text, "fred_data": fred_macro}

    def run_alt_data_agent(self, ticker: str, alt_data_result: dict) -> dict:
        """Analyze alternative data signals."""
        logger.info(f"Alt Data Agent: analyzing trends for {ticker}")
        prompt = prompts.get_alt_data_prompt(ticker, alt_data_result)
        response = self._generate_with_retry(self.general_model, prompt, "Alt Data")
        return {"text": response.text, "data": alt_data_result}

    def run_sector_analysis_agent(self, ticker: str, sector_data: dict) -> dict:
        """Analyze sector relative strength and rotation."""
        logger.info(f"Sector Agent: analyzing sector for {ticker}")
        prompt = prompts.get_sector_analysis_prompt(ticker, sector_data)
        response = self._generate_with_retry(self.general_model, prompt, "Sector")
        return {"text": response.text, "data": sector_data}

    def run_nlp_sentiment_agent(self, ticker: str, nlp_data: dict) -> dict:
        """Analyze transformer-based NLP sentiment."""
        logger.info(f"NLP Sentiment Agent: analyzing for {ticker}")
        prompt = prompts.get_nlp_sentiment_prompt(ticker, nlp_data)
        response = self._generate_with_retry(self.general_model, prompt, "NLP Sentiment")
        return {"text": response.text, "data": nlp_data}

    def run_anomaly_agent(self, ticker: str, anomaly_data: dict) -> dict:
        """Analyze statistical anomalies."""
        logger.info(f"Anomaly Agent: analyzing for {ticker}")
        prompt = prompts.get_anomaly_detection_prompt(ticker, anomaly_data)
        response = self._generate_with_retry(self.general_model, prompt, "Anomaly")
        return {"text": response.text, "data": anomaly_data}

    def run_scenario_agent(self, ticker: str, mc_data: dict) -> dict:
        """Analyze Monte Carlo scenario results."""
        logger.info(f"Scenario Agent: analyzing risk for {ticker}")
        prompt = prompts.get_scenario_analysis_prompt(ticker, mc_data)
        response = self._generate_with_retry(self.general_model, prompt, "Scenario")
        return {"text": response.text, "data": mc_data}

    # ── Synthesis ────────────────────────────────────────────────────

    def run_synthesis_pipeline(self, ticker: str, report: dict) -> dict:
        """Synthesis + Critic pipeline → final validated JSON."""
        logger.info(f"Synthesis Agent: drafting report for {ticker}")

        market_context = (
            f"Market News & Sentiment:\n{report.get('market', {}).get('text', '')}\n\n"
            f"Competitor Analysis:\n{report.get('competitor', {}).get('text', 'No competitor data.')}\n\n"
            f"Macroeconomic Outlook:\n{report.get('macro', {}).get('text', 'No macro data.')}\n\n"
            f"Insider Activity:\n{report.get('insider', {}).get('text', 'No insider data.')}\n\n"
            f"Options Flow:\n{report.get('options', {}).get('text', 'No options data.')}\n\n"
            f"Social Sentiment:\n{report.get('social_sentiment', {}).get('text', 'No social data.')}\n\n"
            f"Earnings Call Tone:\n{report.get('earnings_tone', {}).get('text', 'No earnings tone data.')}\n\n"
            f"Alternative Data:\n{report.get('alt_data', {}).get('text', 'No alternative data.')}\n\n"
            f"Sector Analysis:\n{report.get('sector_analysis', {}).get('text', 'No sector data.')}"
        )

        # Build sector catalyst data from patent + innovation signals
        sector_catalyst = report.get("patent", {}).get("text", "No sector catalyst data.")

        draft_prompt = prompts.get_synthesis_prompt(
            ticker,
            report.get("quant", {}),
            report.get("rag", {}).get("text", ""),
            market_context,
            sector_catalyst,
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

        # Step 0: Fetch external data (Alpha Vantage + yfinance in parallel)
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
        company_name = report["quant"].get("company_name", ticker)
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

        # Step 6: Fetch enrichment data in parallel (11 non-LLM calls)
        step("data_enrichment", "started", "Fetching 11 enrichment data sources in parallel...")

        async def _safe(coro_or_func, label, *args):
            """Run async or sync callable, return result or error dict."""
            try:
                if asyncio.iscoroutinefunction(coro_or_func):
                    return await coro_or_func(*args)
                else:
                    return await asyncio.to_thread(coro_or_func, *args)
            except Exception as e:
                logger.warning(f"{label} data fetch failed: {e}")
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

        (
            insider_data, options_data, social_data, patent_data,
            earnings_data, fred_macro, alt_result, sector_data,
            nlp_data, anomaly_data, mc_data,
        ) = await asyncio.gather(
            _safe(self.fetch_insider_data, "Insider", ticker),
            _safe(self.fetch_options_data, "Options", ticker),
            _safe(self.fetch_social_sentiment, "Social", ticker, fallback_articles or None),
            _safe(self.fetch_patent_data, "Patent", ticker, company_name),
            _safe(self.fetch_earnings_tone, "Earnings", ticker),
            _safe(self.fetch_fred_data, "FRED"),
            _safe(self.fetch_alt_data, "AltData", ticker, company_name),
            _safe(self.fetch_sector_data, "Sector", ticker),
            _safe(self.fetch_nlp_sentiment, "NLP", ticker, articles),
            _safe(self.fetch_anomaly_scan, "Anomaly", ticker),
            _safe(self.fetch_monte_carlo, "MonteCarlo", ticker),
        )
        step("data_enrichment", "completed", "All 11 enrichment data sources collected")

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

        report["insider"] = await asyncio.to_thread(_run_agent_with_trace, self.run_insider_agent, "Insider Activity", ticker, ticker, insider_data)
        report["options"] = await asyncio.to_thread(_run_agent_with_trace, self.run_options_agent, "Options Flow", ticker, ticker, options_data)
        report["social_sentiment"] = await asyncio.to_thread(_run_agent_with_trace, self.run_social_sentiment_agent, "Social Sentiment", ticker, ticker, social_data)
        report["patent"] = await asyncio.to_thread(_run_agent_with_trace, self.run_patent_agent, "Patent Innovation", ticker, ticker, patent_data)
        report["earnings_tone"] = await asyncio.to_thread(_run_agent_with_trace, self.run_earnings_tone_agent, "Earnings Tone", ticker, ticker, earnings_data)
        report["alt_data"] = await asyncio.to_thread(_run_agent_with_trace, self.run_alt_data_agent, "Alt Data", ticker, ticker, alt_result)
        report["sector_analysis"] = await asyncio.to_thread(_run_agent_with_trace, self.run_sector_analysis_agent, "Sector Analysis", ticker, ticker, sector_data)
        report["nlp_sentiment"] = await asyncio.to_thread(_run_agent_with_trace, self.run_nlp_sentiment_agent, "NLP Sentiment", ticker, ticker, nlp_data)
        report["anomaly"] = await asyncio.to_thread(_run_agent_with_trace, self.run_anomaly_agent, "Anomaly Detection", ticker, ticker, anomaly_data)
        report["scenario"] = await asyncio.to_thread(_run_agent_with_trace, self.run_scenario_agent, "Scenario Analysis", ticker, ticker, mc_data)
        step("enrichment_analysis", "completed", "All 11 enrichment agents complete")

        # Step 8: Agent Debate (Bull vs Bear vs Moderator)
        step("debate", "started", "Running adversarial debate (Bull vs Bear)...")
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
        debate_result = await asyncio.to_thread(
            run_debate, self.general_model, ticker, enrichment_for_debate, traces.summary()
        )
        report["debate"] = debate_result
        step("debate", "completed", f"Consensus: {debate_result.get('consensus', 'N/A')}")

        # Step 9: Enhanced macro (original macro + FRED data)
        step("macro", "started", "Analyzing macro economy with FRED data...")
        report["macro"] = await asyncio.to_thread(self.run_enhanced_macro_agent, ticker, av_data, fred_macro)
        step("macro", "completed", "Macro analysis complete")

        # Step 10: Deep dive
        step("deep_dive", "started", "Finding contradictions and probing...")
        report["deep_dive"] = await asyncio.to_thread(self.run_deep_dive_agent, ticker, report)
        step("deep_dive", "completed", "Deep dive complete")

        # Step 11+12: Synthesis + Critic
        step("synthesis", "started", "Drafting final report...")
        final_json = await asyncio.to_thread(self.run_synthesis_pipeline, ticker, report)
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

        report["final_synthesis"] = final_json
        step("synthesis", "completed", f"Score: {final_json.get('final_weighted_score', 'N/A')}")

        # Step 12b: Bias Audit
        step("bias_audit", "started", "Running bias and conflict detection...")
        bias_report = detect_biases(
            ticker=ticker,
            recommendation=final_json.get("recommendation", {}).get("recommendation", "HOLD"),
            score=final_json.get("final_weighted_score", 5.0),
            enrichment_signals=final_json.get("enrichment_signals", {}),
            debate_result=debate_result,
            quant_data=report.get("quant", {}),
        )
        conflict_report = detect_conflicts(
            ticker=ticker,
            synthesis_report=final_json,
            quant_data=report.get("quant", {}),
            enrichment_signals=final_json.get("enrichment_signals", {}),
        )
        final_json["bias_report"] = bias_report
        final_json["conflict_report"] = conflict_report
        step("bias_audit", "completed", f"Bias flags: {bias_report.get('bias_count', 0)}, Conflicts: {conflict_report.get('conflict_count', 0)}")

        return report
