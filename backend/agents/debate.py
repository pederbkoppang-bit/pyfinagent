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
from typing import Any
from typing import Callable, Optional

from vertexai.generative_models import GenerativeModel
from google.api_core import exceptions as gcp_exceptions

from backend.agents.cost_tracker import CostTracker
from backend.agents.compaction import (
    build_compact_debate_history,
    compact_argument,
    compact_da_result,
    compact_text,
    compact_trace_summary,
)
from backend.agents.llm_client import GeminiClient, LLMClient, get_model_max_input_chars
from backend.agents.schemas import DevilsAdvocateResult, ModeratorConsensus

_DEBATE_GEN_CONFIG = {"temperature": 0.0, "top_k": 1, "max_output_tokens": 1536}
_MODERATOR_GEN_CONFIG = {"temperature": 0.0, "top_k": 1, "max_output_tokens": 2048}

# Structured output configs — Gemini JSON schema enforcement (Phase 3)
_DA_STRUCTURED_CONFIG = {
    "temperature": 0.0, "top_k": 1, "max_output_tokens": 1536,
    "response_mime_type": "application/json",
    "response_schema": DevilsAdvocateResult,
}
_MODERATOR_STRUCTURED_CONFIG = {
    "temperature": 0.0, "top_k": 1, "max_output_tokens": 2048,
    "response_mime_type": "application/json",
    "response_schema": ModeratorConsensus,
}

from backend.config import prompts

logger = logging.getLogger(__name__)


def _generate_with_retry(model: LLMClient, prompt: str, agent_name: str, max_retries: int = 3,
                          cost_tracker: CostTracker | None = None, model_name: str = "",
                          is_deep_think: bool = False, gen_config: dict | None = None,
                          thinking_budget: int = 0):
    delay = 5
    effective_model_name = model_name or model.model_name
    config = gen_config or _DEBATE_GEN_CONFIG
    # Phase 5: Inject thinking config for Gemini 2.5 judge agents (Gemini-specific)
    if isinstance(model, GeminiClient) and thinking_budget > 0:
        config = {**config, "thinking": {"type": "enabled", "budget_tokens": thinking_budget}, "include_thoughts": True}
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt, generation_config=config)
            if cost_tracker and effective_model_name:
                cost_tracker.record(agent_name, effective_model_name, response, is_deep_think=is_deep_think)
            return response
        except (gcp_exceptions.ServiceUnavailable, gcp_exceptions.ResourceExhausted) as e:
            if attempt < max_retries - 1:
                logger.warning(f"{agent_name} {type(e).__name__}. Retry in {delay}s")
                time.sleep(delay)
                delay *= 2
            else:
                raise
        except Exception as e:
            # Non-GCP providers raise different exceptions
            err_name = type(e).__name__.lower()
            is_transient = any(x in err_name for x in ("ratelimit", "overload", "unavailable"))
            if is_transient and attempt < max_retries - 1:
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
    model: LLMClient,
    ticker: str,
    enrichment_signals: dict,
    trace_summary: dict,
    max_debate_rounds: int = 2,
    on_progress=None,
    past_memories: dict | None = None,
    cost_tracker: CostTracker | None = None,
    deep_think_model: LLMClient | None = None,
    general_model_name: str = "",
    deep_think_model_name: str = "",
    skip_devils_advocate: bool = False,
    fact_ledger: str = "",
    enable_thinking: bool = False,
    thinking_budgets: dict | None = None,
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

    _model_limit = get_model_max_input_chars(getattr(model, "model_name", "") or general_model_name)
    _tight_context = bool(_model_limit and _model_limit <= 14_000)
    _compact_context = bool(_model_limit and _model_limit <= 30_000)
    _json_kwargs: dict[str, Any] = {"default": str}
    if _compact_context:
        _json_kwargs["separators"] = (",", ":")
    else:
        _json_kwargs["indent"] = 2

    trace_payload = trace_summary
    if _compact_context:
        trace_payload = compact_trace_summary(
            trace_summary,
            max_chars=700 if _tight_context else 1_600,
            max_evidence_per_agent=1,
            evidence_chars=90 if _tight_context else 120,
        )
        fact_ledger = compact_text(fact_ledger, 700 if _tight_context else 1_200)
        logger.info(
            f"Debate: compact mode enabled for {getattr(model, 'model_name', general_model_name)} "
            f"(limit={_model_limit:,} chars, tight={_tight_context})"
        )

    signals_json = json.dumps(enrichment_signals, **_json_kwargs)
    trace_json = json.dumps(trace_payload, **_json_kwargs)

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
            opponent_argument=compact_argument(bear_text, 900 if _tight_context else 1_400) if bear_text and _compact_context else bear_text or None,
            round_number=round_num,
            max_rounds=max_debate_rounds,
            past_memory=memories.get("bull", ""),
            fact_ledger=fact_ledger,
        )
        bull_response = _generate_with_retry(model, bull_prompt, f"Bull Agent R{round_num}",
                                               cost_tracker=cost_tracker, model_name=general_model_name)
        if bull_response:
            bull_text = bull_response.text.strip()

        # Bear Agent turn
        logger.info(f"Debate Round {round_num}: Bear Agent (round {round_num}/{max_debate_rounds})")
        _progress(f"Round {round_num}/{max_debate_rounds}: Bear agent challenging with risk evidence...")
        bear_prompt = prompts.get_bear_agent_prompt(
            ticker, signals_json, trace_json,
            opponent_argument=compact_argument(bull_text, 900 if _tight_context else 1_400) if bull_text and _compact_context else bull_text,
            round_number=round_num,
            max_rounds=max_debate_rounds,
            past_memory=memories.get("bear", ""),
            fact_ledger=fact_ledger,
        )
        bear_response = _generate_with_retry(model, bear_prompt, f"Bear Agent R{round_num}",
                                               cost_tracker=cost_tracker, model_name=general_model_name)
        if bear_response:
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
            ticker,
            compact_argument(bull_text, 1_000 if _tight_context else 1_800) if _compact_context else bull_text,
            compact_argument(bear_text, 1_000 if _tight_context else 1_800) if _compact_context else bear_text,
            signals_json,
            fact_ledger=fact_ledger,
        )
        da_response = _generate_with_retry(model, da_prompt, "Devil's Advocate",
                                             cost_tracker=cost_tracker, model_name=general_model_name,
                                             gen_config=_DA_STRUCTURED_CONFIG)
        da_text = _clean_json(da_response.text) if da_response else ""
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
    debate_history = build_compact_debate_history(
        debate_rounds,
        max_chars=1_200 if _tight_context else 2_600 if _compact_context else 6_000,
        per_argument_chars=220 if _tight_context else 420 if _compact_context else 1_500,
    )

    bull_for_moderator = compact_argument(bull_text, 1_000 if _tight_context else 1_800) if _compact_context else bull_text
    bear_for_moderator = compact_argument(bear_text, 1_000 if _tight_context else 1_800) if _compact_context else bear_text
    da_for_moderator = (
        compact_da_result(da_result, max_chars=800 if _tight_context else 1_400)
        if _compact_context else json.dumps(da_result, default=str)
    )

    moderator_prompt = prompts.get_moderator_prompt(
        ticker, bull_for_moderator, bear_for_moderator, signals_json,
        devils_advocate=da_for_moderator,
        debate_history=debate_history,
        past_memory=memories.get("moderator", ""),
        fact_ledger=fact_ledger,
    )
    _moderator_model = deep_think_model or model
    _moderator_thinking_budget = (thinking_budgets or {}).get("Moderator", 0) if enable_thinking else 0
    moderator_response = _generate_with_retry(_moderator_model, moderator_prompt, "Moderator",
                                               cost_tracker=cost_tracker,
                                               model_name=deep_think_model_name or general_model_name,
                                               is_deep_think=deep_think_model is not None,
                                               gen_config=_MODERATOR_STRUCTURED_CONFIG,
                                               thinking_budget=_moderator_thinking_budget)
    moderator_text = _clean_json(moderator_response.text) if moderator_response else ""

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
