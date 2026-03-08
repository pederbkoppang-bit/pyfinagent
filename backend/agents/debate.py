"""
Multi-agent debate framework — Bull vs Bear with Moderator consensus.
Implements a 4-round adversarial debate before synthesis.

Research basis: TradingAgents (arXiv ref 32).
"""

import json
import logging
import re
import time
from typing import Optional

from vertexai.generative_models import GenerativeModel
from google.api_core import exceptions as gcp_exceptions

from backend.config import prompts

logger = logging.getLogger(__name__)


def _generate_with_retry(model: GenerativeModel, prompt: str, agent_name: str, max_retries: int = 3):
    delay = 5
    for attempt in range(max_retries):
        try:
            return model.generate_content(prompt)
        except (gcp_exceptions.ServiceUnavailable, gcp_exceptions.ResourceExhausted) as e:
            if attempt < max_retries - 1:
                logger.warning(f"{agent_name} {type(e).__name__}. Retry in {delay}s")
                time.sleep(delay)
                delay *= 2
            else:
                raise


def _clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text)
        text = re.sub(r"```$", "", text)
    return text.strip()


def _parse_json(text: str, label: str) -> Optional[dict]:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        logger.warning(f"{label} returned invalid JSON, using raw text")
        return None


def run_debate(
    model: GenerativeModel,
    ticker: str,
    enrichment_signals: dict,
    trace_summary: dict,
) -> dict:
    """
    Execute 4-round adversarial debate.

    Args:
        model: Vertex AI GenerativeModel instance
        ticker: Stock ticker
        enrichment_signals: Dict of signal_name -> {signal, summary, text} from enrichment agents
        trace_summary: Summary from TraceCollector.summary()

    Returns:
        Debate result dict with bull_case, bear_case, consensus, contradictions, dissent_registry
    """
    logger.info(f"Debate: starting 4-round debate for {ticker}")

    signals_json = json.dumps(enrichment_signals, indent=2, default=str)
    trace_json = json.dumps(trace_summary, indent=2, default=str)

    # ── Round 1: Position statements ────────────────────────────
    # (Positions come from the trace_summary which has each agent's signal + confidence)
    logger.info("Debate Round 1: Collecting agent positions")

    # ── Round 2: Bull Agent ─────────────────────────────────────
    logger.info("Debate Round 2: Bull Agent synthesizing bullish case")
    bull_prompt = prompts.get_bull_agent_prompt(ticker, signals_json, trace_json)
    bull_response = _generate_with_retry(model, bull_prompt, "Bull Agent")
    bull_text = bull_response.text.strip()

    # ── Round 3: Bear Agent ─────────────────────────────────────
    logger.info("Debate Round 3: Bear Agent synthesizing bearish case")
    bear_prompt = prompts.get_bear_agent_prompt(ticker, signals_json, trace_json)
    bear_response = _generate_with_retry(model, bear_prompt, "Bear Agent")
    bear_text = bear_response.text.strip()

    # ── Round 4: Moderator ─────────────────────────────────────
    logger.info("Debate Round 4: Moderator resolving contradictions")
    moderator_prompt = prompts.get_moderator_prompt(ticker, bull_text, bear_text, signals_json)
    moderator_response = _generate_with_retry(model, moderator_prompt, "Moderator")
    moderator_text = _clean_json(moderator_response.text)

    # Try to parse moderator output as structured JSON
    debate_result = _parse_json(moderator_text, "Moderator")

    if debate_result:
        # Ensure all expected keys exist
        debate_result.setdefault("bull_case", {"thesis": bull_text, "confidence": 0.5})
        debate_result.setdefault("bear_case", {"thesis": bear_text, "confidence": 0.5})
        debate_result.setdefault("consensus", "HOLD")
        debate_result.setdefault("consensus_confidence", 0.5)
        debate_result.setdefault("contradictions", [])
        debate_result.setdefault("dissent_registry", [])
    else:
        # Fallback: build structured result from raw text
        debate_result = {
            "bull_case": {
                "thesis": bull_text[:2000],
                "confidence": 0.5,
                "key_catalysts": [],
                "evidence": [],
            },
            "bear_case": {
                "thesis": bear_text[:2000],
                "confidence": 0.5,
                "key_threats": [],
                "evidence": [],
            },
            "consensus": "HOLD",
            "consensus_confidence": 0.5,
            "moderator_analysis": moderator_text[:2000],
            "contradictions": [],
            "dissent_registry": [],
        }

    logger.info(f"Debate complete: consensus={debate_result.get('consensus')}, "
                f"confidence={debate_result.get('consensus_confidence')}")
    return debate_result
