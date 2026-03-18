"""
Multi-agent debate framework — Bull vs Bear with Moderator consensus.
Implements multi-round adversarial debate with Devil's Advocate before synthesis.

Research basis: TradingAgents (arXiv ref 32) — multi-round conversational debate
where each agent sees the opponent's prior argument, plus Devil's Advocate challenge.
"""

import json
import logging
import re
import time
from typing import Optional

from vertexai.generative_models import GenerativeModel
from google.api_core import exceptions as gcp_exceptions

from backend.agents.cost_tracker import CostTracker

_DEBATE_GEN_CONFIG = {"temperature": 0.2, "max_output_tokens": 1536}
_MODERATOR_GEN_CONFIG = {"temperature": 0.2, "max_output_tokens": 2048}

from backend.config import prompts

logger = logging.getLogger(__name__)


def _generate_with_retry(model: GenerativeModel, prompt: str, agent_name: str, max_retries: int = 3,
                          cost_tracker: CostTracker | None = None, model_name: str = "",
                          is_deep_think: bool = False, gen_config: dict | None = None):
    delay = 5
    config = gen_config or _DEBATE_GEN_CONFIG
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt, generation_config=config)
            if cost_tracker and model_name:
                cost_tracker.record(agent_name, model_name, response, is_deep_think=is_deep_think)
            return response
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
    max_debate_rounds: int = 2,
    on_progress: Optional[callable] = None,
    past_memories: dict = None,
    cost_tracker: CostTracker | None = None,
    deep_think_model: GenerativeModel | None = None,
    general_model_name: str = "",
    deep_think_model_name: str = "",
    skip_devils_advocate: bool = False,
) -> dict:
    """
    Execute multi-round adversarial debate with Devil's Advocate.

    Implements TradingAgents-style conversational debate where Bull and Bear
    agents iteratively refine arguments by seeing the opponent's prior case.

    Flow:
      1. Position collection (from trace_summary)
      2. N rounds of Bull ↔ Bear debate (each sees opponent's last argument)
      3. Devil's Advocate challenge (stress-tests the emerging consensus)
      4. Moderator resolves contradictions with DA input

    Args:
        model: Vertex AI GenerativeModel instance
        ticker: Stock ticker
        enrichment_signals: Dict of signal_name -> {signal, summary, text}
        trace_summary: Summary from TraceCollector.summary()
        max_debate_rounds: Number of Bull↔Bear exchange rounds (default: 2)
        on_progress: Optional callback(message: str) for real-time progress updates
        past_memories: Dict of agent_type -> memory_str for past lessons

    Returns:
        Debate result dict with bull_case, bear_case, debate_rounds,
        devils_advocate, consensus, contradictions, dissent_registry
    """
    def _progress(msg: str):
        if on_progress:
            on_progress(msg)

    memories = past_memories or {}

    logger.info(f"Debate: starting {max_debate_rounds}-round debate for {ticker}")

    signals_json = json.dumps(enrichment_signals, indent=2, default=str)
    trace_json = json.dumps(trace_summary, indent=2, default=str)

    # ── Round 1: Position statements ────────────────────────────
    logger.info("Debate Round 1: Collecting agent positions")
    _progress("Round 1: Collecting positions from all enrichment agents...")

    # ── Rounds 2..N+1: Iterative Bull ↔ Bear debate ────────────
    debate_rounds = []
    bull_text = ""
    bear_text = ""

    for round_num in range(1, max_debate_rounds + 1):
        # Bull Agent turn
        logger.info(f"Debate Round {round_num}: Bull Agent (round {round_num}/{max_debate_rounds})")
        _progress(f"Round {round_num}/{max_debate_rounds}: Bull agent building case...")
        bull_prompt = prompts.get_bull_agent_prompt(
            ticker, signals_json, trace_json,
            opponent_argument=bear_text if bear_text else None,
            round_number=round_num,
            max_rounds=max_debate_rounds,
            past_memory=memories.get("bull", ""),
        )
        bull_response = _generate_with_retry(model, bull_prompt, f"Bull Agent R{round_num}",
                                               cost_tracker=cost_tracker, model_name=general_model_name)
        bull_text = bull_response.text.strip()

        # Bear Agent turn
        logger.info(f"Debate Round {round_num}: Bear Agent (round {round_num}/{max_debate_rounds})")
        _progress(f"Round {round_num}/{max_debate_rounds}: Bear agent challenging with risk evidence...")
        bear_prompt = prompts.get_bear_agent_prompt(
            ticker, signals_json, trace_json,
            opponent_argument=bull_text,
            round_number=round_num,
            max_rounds=max_debate_rounds,
            past_memory=memories.get("bear", ""),
        )
        bear_response = _generate_with_retry(model, bear_prompt, f"Bear Agent R{round_num}",
                                               cost_tracker=cost_tracker, model_name=general_model_name)
        bear_text = bear_response.text.strip()

        debate_rounds.append({
            "round": round_num,
            "bull_argument": bull_text[:3000],
            "bear_argument": bear_text[:3000],
        })

    # ── Devil's Advocate ────────────────────────────────────────
    if skip_devils_advocate:
        logger.info("Debate: Devil's Advocate skipped (lite mode)")
        _progress("Devil's Advocate skipped (lite mode)")
        da_result = {
            "challenges": [],
            "hidden_risks": [],
            "confidence_adjustment": 0.0,
            "summary": "Skipped (lite mode)",
        }
    else:
        logger.info("Debate: Devil's Advocate challenging consensus")
        _progress("Devil's Advocate stress-testing the emerging consensus...")
        da_prompt = prompts.get_devils_advocate_prompt(
            ticker, bull_text, bear_text, signals_json
        )
        da_response = _generate_with_retry(model, da_prompt, "Devil's Advocate",
                                             cost_tracker=cost_tracker, model_name=general_model_name)
        da_text = _clean_json(da_response.text)
        da_result = _parse_json(da_text, "Devil's Advocate")
        if not da_result:
            da_result = {
                "challenges": [da_text[:1500]],
                "hidden_risks": [],
                "confidence_adjustment": 0.0,
                "summary": da_text[:1000],
            }

    # ── Moderator ───────────────────────────────────────────────
    logger.info("Debate: Moderator resolving contradictions")
    _progress("Moderator resolving contradictions and assigning consensus...")

    # Build debate history for moderator context
    debate_history = "\n".join(
        f"--- ROUND {r['round']} ---\nBull: {r['bull_argument'][:1500]}\nBear: {r['bear_argument'][:1500]}"
        for r in debate_rounds
    )

    moderator_prompt = prompts.get_moderator_prompt(
        ticker, bull_text, bear_text, signals_json,
        devils_advocate=json.dumps(da_result, default=str),
        debate_history=debate_history,
        past_memory=memories.get("moderator", ""),
    )
    _moderator_model = deep_think_model or model
    moderator_response = _generate_with_retry(_moderator_model, moderator_prompt, "Moderator",
                                               cost_tracker=cost_tracker,
                                               model_name=deep_think_model_name or general_model_name,
                                               is_deep_think=deep_think_model is not None,
                                               gen_config=_MODERATOR_GEN_CONFIG)
    moderator_text = _clean_json(moderator_response.text)

    # Try to parse moderator output as structured JSON
    debate_result = _parse_json(moderator_text, "Moderator")

    if debate_result:
        debate_result.setdefault("bull_case", {"thesis": bull_text, "confidence": 0.5})
        debate_result.setdefault("bear_case", {"thesis": bear_text, "confidence": 0.5})
        debate_result.setdefault("consensus", "HOLD")
        debate_result.setdefault("consensus_confidence", 0.5)
        debate_result.setdefault("contradictions", [])
        debate_result.setdefault("dissent_registry", [])
    else:
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

    # Attach multi-round metadata (backward-compatible additions)
    debate_result["debate_rounds"] = debate_rounds
    debate_result["total_rounds"] = max_debate_rounds
    debate_result["devils_advocate"] = da_result

    logger.info(f"Debate complete: consensus={debate_result.get('consensus')}, "
                f"confidence={debate_result.get('consensus_confidence')}, "
                f"rounds={max_debate_rounds}, da_challenges={len(da_result.get('challenges', []))}")
    return debate_result
