"""
Risk Assessment Team — Aggressive / Conservative / Neutral + Risk Judge.

Implements TradingAgents-style round-robin risk debate where three risk analysts
with different risk philosophies evaluate the synthesis result through multi-round
adversarial debate, followed by a Risk Judge who renders the final risk verdict.

Each analyst sees the other two analysts' prior arguments and must directly
address/counter them. The Judge sees the full debate history.

Research basis: TradingAgents (arXiv ref 32) — RiskDebateState with
aggressive/conservative/neutral round-robin + Risk Judge.
"""

import json
import logging
import re
import time
from typing import Callable, Optional

from vertexai.generative_models import GenerativeModel
from google.api_core import exceptions as gcp_exceptions

from backend.agents.cost_tracker import CostTracker
from backend.agents.llm_client import GeminiClient, LLMClient
from backend.agents.schemas import RiskAnalystArgument, RiskJudgeVerdict
from backend.config import prompts

logger = logging.getLogger(__name__)

_RISK_GEN_CONFIG = {"temperature": 0.0, "top_k": 1, "max_output_tokens": 1024}
_JUDGE_GEN_CONFIG = {"temperature": 0.0, "top_k": 1, "max_output_tokens": 1536}

# Structured output configs — Gemini JSON schema enforcement (Phase 3)
_RISK_STRUCTURED_CONFIG = {
    "temperature": 0.0, "top_k": 1, "max_output_tokens": 1024,
    "response_mime_type": "application/json",
    "response_schema": RiskAnalystArgument,
}
_JUDGE_STRUCTURED_CONFIG = {
    "temperature": 0.0, "top_k": 1, "max_output_tokens": 1536,
    "response_mime_type": "application/json",
    "response_schema": RiskJudgeVerdict,
}


def _generate_with_retry(model: LLMClient, prompt: str, agent_name: str, max_retries: int = 3,
                          cost_tracker: CostTracker | None = None, model_name: str = "",
                          is_deep_think: bool = False, gen_config: dict | None = None,
                          thinking_budget: int = 0):
    delay = 5
    effective_model_name = model_name or model.model_name
    config = gen_config or _RISK_GEN_CONFIG
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


def run_risk_debate(
    model: LLMClient,
    ticker: str,
    synthesis_result: dict,
    enrichment_signals: dict,
    debate_result: dict | None = None,
    max_risk_rounds: int = 1,
    past_memories: dict | None = None,
    on_progress=None,
    cost_tracker: CostTracker | None = None,
    deep_think_model: LLMClient | None = None,
    general_model_name: str = "",
    deep_think_model_name: str = "",
    fact_ledger: str = "",
    enable_thinking: bool = False,
    thinking_budgets: dict | None = None,
) -> dict:
    """
    Execute multi-round round-robin risk assessment:
    N rounds of (Aggressive → Conservative → Neutral), then Risk Judge.

    Each analyst sees the other two's prior arguments from round 2 onward,
    enabling true adversarial cross-debate.

    Args:
        model: Vertex AI GenerativeModel instance
        ticker: Stock ticker
        synthesis_result: Output from Synthesis Agent (the draft report JSON)
        enrichment_signals: Dict of signal_name -> {signal, summary}
        debate_result: Output from Bull/Bear debate (for richer context)
        max_risk_rounds: Number of Agg/Con/Neu rounds before Judge (default: 1)
        past_memories: Dict of agent_type -> memory_str for past lessons
        on_progress: Optional callback(message: str) for progress updates

    Returns:
        Risk assessment dict with aggressive/conservative/neutral cases,
        judge verdict, debate_rounds, and total_rounds
    """
    def _progress(msg: str):
        if on_progress:
            on_progress(msg)

    logger.info(f"Risk debate: starting {max_risk_rounds}-round risk assessment for {ticker}")

    synthesis_json = json.dumps(synthesis_result, indent=2, default=str)[:6000]
    signals_json = json.dumps(enrichment_signals, indent=2, default=str)
    debate_context = ""
    if debate_result:
        debate_context = json.dumps({
            "consensus": debate_result.get("consensus"),
            "consensus_confidence": debate_result.get("consensus_confidence"),
            "bull_thesis": str(debate_result.get("bull_case", {}).get("thesis", ""))[:500],
            "bear_thesis": str(debate_result.get("bear_case", {}).get("thesis", ""))[:500],
            "contradictions": debate_result.get("contradictions", [])[:3],
        }, default=str)

    memories = past_memories or {}

    # Track arguments across rounds
    aggressive_text = ""
    conservative_text = ""
    neutral_text = ""
    risk_rounds = []

    for round_num in range(1, max_risk_rounds + 1):
        round_label = f"R{round_num}/{max_risk_rounds}"

        # ── Aggressive Analyst ──────────────────────────────────
        logger.info(f"Risk debate {round_label}: Aggressive Analyst")
        _progress(f"Round {round_num}/{max_risk_rounds}: Aggressive analyst evaluating upside...")
        aggressive_prompt = prompts.get_aggressive_analyst_prompt(
            ticker, synthesis_json, signals_json,
            conservative_arg=conservative_text[:2000] if conservative_text else "",
            neutral_arg=neutral_text[:2000] if neutral_text else "",
            debate_context=debate_context,
            past_memory=memories.get("risk_aggressive", ""),
            fact_ledger=fact_ledger,
        )
        aggressive_response = _generate_with_retry(model, aggressive_prompt, f"Aggressive {round_label}",
                                                     cost_tracker=cost_tracker, model_name=general_model_name,
                                                     gen_config=_RISK_STRUCTURED_CONFIG)
        if aggressive_response:
            aggressive_text = aggressive_response.text.strip()

        # ── Conservative Analyst ────────────────────────────────
        logger.info(f"Risk debate {round_label}: Conservative Analyst")
        _progress(f"Round {round_num}/{max_risk_rounds}: Conservative analyst assessing risks...")
        conservative_prompt = prompts.get_conservative_analyst_prompt(
            ticker, synthesis_json, signals_json,
            aggressive_arg=aggressive_text[:2000],
            neutral_arg=neutral_text[:2000] if neutral_text else "",
            debate_context=debate_context,
            past_memory=memories.get("risk_conservative", ""),
            fact_ledger=fact_ledger,
        )
        conservative_response = _generate_with_retry(model, conservative_prompt, f"Conservative {round_label}",
                                                       cost_tracker=cost_tracker, model_name=general_model_name,
                                                       gen_config=_RISK_STRUCTURED_CONFIG)
        if conservative_response:
            conservative_text = conservative_response.text.strip()

        # ── Neutral Analyst ─────────────────────────────────────
        logger.info(f"Risk debate {round_label}: Neutral Analyst")
        _progress(f"Round {round_num}/{max_risk_rounds}: Neutral analyst balancing views...")
        neutral_prompt = prompts.get_neutral_analyst_prompt(
            ticker, synthesis_json, signals_json,
            aggressive_text[:2000], conservative_text[:2000],
            debate_context=debate_context,
            past_memory=memories.get("risk_neutral", ""),
            fact_ledger=fact_ledger,
        )
        neutral_response = _generate_with_retry(model, neutral_prompt, f"Neutral {round_label}",
                                                   cost_tracker=cost_tracker, model_name=general_model_name,
                                                   gen_config=_RISK_STRUCTURED_CONFIG)
        if neutral_response:
            neutral_text = neutral_response.text.strip()

        risk_rounds.append({
            "round": round_num,
            "aggressive": aggressive_text[:2000],
            "conservative": conservative_text[:2000],
            "neutral": neutral_text[:2000],
        })

    # Parse final round results
    aggressive_result = _parse_json(_clean_json(aggressive_text), "Aggressive Analyst")
    if not aggressive_result:
        aggressive_result = {"position": aggressive_text[:1500], "confidence": 0.5, "max_position_pct": 5}

    conservative_result = _parse_json(_clean_json(conservative_text), "Conservative Analyst")
    if not conservative_result:
        conservative_result = {"position": conservative_text[:1500], "confidence": 0.5, "max_position_pct": 1}

    neutral_result = _parse_json(_clean_json(neutral_text), "Neutral Analyst")
    if not neutral_result:
        neutral_result = {"position": neutral_text[:1500], "confidence": 0.5, "max_position_pct": 3}

    # ── Risk Judge ──────────────────────────────────────────────
    logger.info("Risk debate: Risk Judge rendering verdict")
    _progress("Risk Judge rendering final risk verdict...")

    # Build debate history for judge
    debate_history = "\n".join(
        f"--- ROUND {r['round']} ---\n"
        f"Aggressive: {r['aggressive'][:800]}\n"
        f"Conservative: {r['conservative'][:800]}\n"
        f"Neutral: {r['neutral'][:800]}"
        for r in risk_rounds
    )

    judge_prompt = prompts.get_risk_judge_prompt(
        ticker, synthesis_json,
        aggressive_text[:2000], conservative_text[:2000], neutral_text[:2000],
        debate_history=debate_history if max_risk_rounds > 1 else "",
        past_memory=memories.get("risk_judge", ""),
        fact_ledger=fact_ledger,
    )
    _judge_model = deep_think_model or model
    _judge_thinking_budget = (thinking_budgets or {}).get("Risk Judge", 0) if enable_thinking else 0
    judge_response = _generate_with_retry(_judge_model, judge_prompt, "Risk Judge",
                                           cost_tracker=cost_tracker,
                                           model_name=deep_think_model_name or general_model_name,
                                           is_deep_think=deep_think_model is not None,
                                           gen_config=_JUDGE_STRUCTURED_CONFIG,
                                           thinking_budget=_judge_thinking_budget)
    judge_text = _clean_json(judge_response.text) if judge_response else ""
    judge_result = _parse_json(judge_text, "Risk Judge")
    if not judge_result:
        judge_result = {
            "decision": "APPROVE_REDUCED",
            "risk_adjusted_confidence": 0.5,
            "recommended_position_pct": 3,
            "risk_level": "MODERATE",
            "reasoning": judge_text[:1500],
            "risk_limits": {"stop_loss_pct": 10, "max_drawdown_pct": 15},
            "unresolved_risks": [],
            "summary": judge_text[:500],
        }

    risk_assessment = {
        "aggressive": aggressive_result,
        "conservative": conservative_result,
        "neutral": neutral_result,
        "judge": judge_result,
        "risk_debate_rounds": risk_rounds,
        "total_risk_rounds": max_risk_rounds,
    }

    logger.info(
        f"Risk debate complete: decision={judge_result.get('decision')}, "
        f"risk_level={judge_result.get('risk_level')}, "
        f"position={judge_result.get('recommended_position_pct')}%, "
        f"rounds={max_risk_rounds}"
    )
    return risk_assessment
