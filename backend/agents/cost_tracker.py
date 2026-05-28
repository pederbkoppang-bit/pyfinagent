"""
Cost tracker for LLM token usage and cost estimation.

Records per-agent token counts and computes costs based on model pricing.
Used by orchestrator, debate, and risk_debate to produce a cost summary
that is surfaced in the frontend Cost tab after analysis completes.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (input, output) — verified 2026-04-18
# Sources:
#   https://platform.claude.com/docs/en/about-claude/pricing
#   https://platform.claude.com/docs/en/about-claude/models/overview
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Gemini (Vertex AI)
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.00),
    # Anthropic Claude — current GA (Opus 4.x, Sonnet 4.6, Haiku 4.5)
    "claude-opus-4-8": (5.00, 25.00),  # phase-47.3: launched 2026-05-28, $5/$25 (same as 4.7)
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-opus-4-5": (5.00, 25.00),
    "claude-opus-4-1": (15.00, 75.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # Anthropic Claude — legacy (still live, deprecated 2026-06-15 for Sonnet 4 / Opus 4)
    "claude-sonnet-4": (3.00, 15.00),
    "claude-opus-4": (15.00, 75.00),
    # OpenAI (direct or via GitHub Models)
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5": (10.00, 40.00),
    "gpt-5-chat": (5.00, 20.00),
    "gpt-5-mini": (2.00, 8.00),
    "gpt-5-nano": (0.50, 2.00),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o1-preview": (15.00, 60.00),
    "o3": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    # Meta Llama (via GitHub Models)
    "meta-llama-3.1-405b-instruct": (5.00, 15.00),
    "meta-llama-3.1-8b-instruct": (0.18, 0.18),
    "llama-3.3-70b-instruct": (0.23, 0.70),
    "llama-4-maverick": (0.19, 0.85),
    "llama-4-scout": (0.11, 0.40),
    # DeepSeek (via GitHub Models)
    "deepseek-r1": (0.55, 2.19),
    "deepseek-r1-0528": (0.55, 2.19),
    "deepseek-v3-0324": (0.27, 1.10),
    # xAI (via GitHub Models)
    "grok-3": (3.00, 15.00),
    "grok-3-mini": (0.30, 0.50),
    # Microsoft (via GitHub Models)
    "phi-4": (0.07, 0.14),
    "mai-ds-r1": (0.55, 2.19),
    "phi-4-mini-instruct": (0.07, 0.14),
    "phi-4-mini-reasoning": (0.10, 0.20),
    "phi-4-reasoning": (0.10, 0.40),
    # Mistral (via GitHub Models)
    "ministral-3b": (0.10, 0.10),
    "codestral-2501": (0.30, 0.90),
    "mistral-medium-2505": (2.00, 6.00),
    "mistral-small-2503": (0.10, 0.30),
}

# Fallback pricing if model not in table
_DEFAULT_PRICING = (0.10, 0.40)


@dataclass
class AgentCostEntry:
    """Token usage for a single LLM call."""
    agent_name: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    is_deep_think: bool = False
    is_grounded: bool = False
    # Phase 2.12: Prompt caching metrics
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    # phase-25.C9: when True, the recorded cost_usd reflects the 50% flat
    # discount from Anthropic's Batch API. Set by callers that route a
    # non-interactive request through `BatchClient` (n>3 tickers in
    # backtest mode). Closes phase-24.9 F-4.
    is_batch: bool = False
    # phase-25.S.1: optional per-call ticker tag enabling exact per-ticker
    # cost attribution. None when the call isn't ticker-scoped (e.g., a
    # MetaCoordinator decision call). When set, flows into llm_call_log
    # for downstream `profit_per_llm_dollar` aggregation at the ticker
    # granularity (north-star goal-c rendered per-ticker).
    ticker: Optional[str] = None
    # phase-26.2: Advisor Tool tier (Sonnet executor + Opus advisor).
    # When True, this entry's cost_usd reflects a BLENDED cost across the
    # executor model (charged at its rates) and the advisor model (charged
    # at advisor rates). advisor_input_tokens + advisor_output_tokens are
    # the OPUS-priced portion; input_tokens + output_tokens are the
    # executor-priced portion. Total = (input * exec_rate_in + output *
    # exec_rate_out + advisor_input * opus_rate_in + advisor_output *
    # opus_rate_out) / 1e6. See record_advisor_call().
    is_advisor: bool = False
    advisor_model: Optional[str] = None
    advisor_input_tokens: int = 0
    advisor_output_tokens: int = 0


@dataclass
class CostTracker:
    """
    Thread-safe accumulator for per-agent token usage across an analysis run.
    Create one instance per analysis; call record() after each LLM call.
    """
    entries: list[AgentCostEntry] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        agent_name: str,
        model: str,
        response: object,
        is_deep_think: bool = False,
        is_grounded: bool = False,
        is_batch: bool = False,
        ticker: Optional[str] = None,
    ) -> Optional[AgentCostEntry]:
        """
        Extract token counts from a Vertex AI response and record the entry.

        Args:
            agent_name: Human-readable agent label (e.g. "Synthesis Agent")
            model: Model name string (e.g. "gemini-2.0-flash")
            response: The GenerateContentResponse from Vertex AI
            is_deep_think: Whether this agent uses the deep-think model
        """
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            return None

        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or (input_tokens + output_tokens)

        # Phase 2.12: Extract prompt caching metrics
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

        pricing = MODEL_PRICING.get(model, _DEFAULT_PRICING)
        # Anthropic prompt caching pricing:
        #   cache reads  = 0.1x base input (90% discount)
        #   cache writes = 1.25x base input (5-min TTL, default)
        #                = 2.00x base input (1-hour TTL, extended-cache-ttl beta)
        # phase-25.A9 (2026-05-12): bumped to 2.0x. llm_client.py:773-779 already
        # passes `"ttl": "1h"` on every caching call, so the actual Anthropic charge
        # is the 1h-TTL 2.0x rate (not the 5-min 1.25x). Prior 1.25x under-reported
        # cache-write cost by ~60% (e.g., $0.026 reported vs $0.041 actual for a
        # 4096-token Opus 4.7 system prompt). Closes phase-24.9 audit finding F-1.
        if cache_read > 0 or cache_creation > 0:
            regular_input = max(0, input_tokens - cache_read - cache_creation)
            cached_read_cost = cache_read * pricing[0] * 0.1 / 1_000_000
            cache_write_cost = cache_creation * pricing[0] * 2.0 / 1_000_000
            regular_cost = regular_input * pricing[0] / 1_000_000
            output_cost = output_tokens * pricing[1] / 1_000_000
            cost = cached_read_cost + cache_write_cost + regular_cost + output_cost
        else:
            cost = (input_tokens * pricing[0] + output_tokens * pricing[1]) / 1_000_000

        # phase-25.C9: Batch API 50% flat discount stacks with caching.
        # When `is_batch=True`, halve the computed cost before recording.
        # Closes phase-24.9 F-4.
        if is_batch:
            cost *= 0.5

        entry = AgentCostEntry(
            agent_name=agent_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=round(cost, 6),
            is_deep_think=is_deep_think,
            is_grounded=is_grounded,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            is_batch=is_batch,
            ticker=ticker,
        )

        with self._lock:
            self.entries.append(entry)

        return entry

    def record_advisor_call(
        self,
        agent_name: str,
        executor_model: str,
        advisor_model: str,
        executor_input_tokens: int,
        executor_output_tokens: int,
        advisor_input_tokens: int,
        advisor_output_tokens: int,
        ticker: Optional[str] = None,
    ) -> AgentCostEntry:
        """phase-26.2: record an Anthropic Advisor Tool call. The blended
        cost is executor_tokens at executor_model rates + advisor_tokens at
        advisor_model rates. Both sides are tracked separately on the entry
        so downstream attribution can split them out.

        When advisor was NOT invoked, pass advisor_input_tokens=0 and
        advisor_output_tokens=0 -- the entry is recorded with is_advisor=True
        but the advisor portion contributes $0.
        """
        exec_pricing = MODEL_PRICING.get(executor_model, _DEFAULT_PRICING)
        adv_pricing = MODEL_PRICING.get(advisor_model, _DEFAULT_PRICING)

        executor_cost = (
            executor_input_tokens * exec_pricing[0]
            + executor_output_tokens * exec_pricing[1]
        ) / 1_000_000
        advisor_cost = (
            advisor_input_tokens * adv_pricing[0]
            + advisor_output_tokens * adv_pricing[1]
        ) / 1_000_000
        total_cost = round(executor_cost + advisor_cost, 6)

        entry = AgentCostEntry(
            agent_name=agent_name,
            model=executor_model,  # executor is the "primary" model for the entry
            input_tokens=executor_input_tokens,
            output_tokens=executor_output_tokens,
            total_tokens=(executor_input_tokens + executor_output_tokens
                          + advisor_input_tokens + advisor_output_tokens),
            cost_usd=total_cost,
            is_deep_think=True,  # advisor sub-inference is Opus 4.7 -> deep-think tier
            is_grounded=False,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            is_batch=False,
            ticker=ticker,
            is_advisor=True,
            advisor_model=advisor_model,
            advisor_input_tokens=advisor_input_tokens,
            advisor_output_tokens=advisor_output_tokens,
        )

        with self._lock:
            self.entries.append(entry)

        return entry

    @property
    def total_cost(self) -> float:
        """Current total cost in USD across all recorded entries."""
        with self._lock:
            return sum(e.cost_usd for e in self.entries)

    def check_budget(self, max_cost_usd: float) -> bool:
        """Return True if total cost has exceeded the budget. Logs a warning on first breach."""
        exceeded = self.total_cost > max_cost_usd
        if exceeded:
            logger.warning("Cost budget exceeded: $%.4f > $%.2f limit", self.total_cost, max_cost_usd)
        return exceeded

    def summarize(self) -> dict:
        """
        Produce a JSON-serializable cost summary for the frontend.

        Returns dict with:
          - total_tokens, total_input_tokens, total_output_tokens
          - total_cost_usd
          - total_calls, deep_think_calls
          - model_breakdown: {model: {calls, tokens, cost}}
          - agents: list of per-agent entries
        """
        with self._lock:
            entries = list(self.entries)

        total_input = sum(e.input_tokens for e in entries)
        total_output = sum(e.output_tokens for e in entries)
        total_tokens = sum(e.total_tokens for e in entries)
        total_cost = sum(e.cost_usd for e in entries)
        deep_think_calls = sum(1 for e in entries if e.is_deep_think)

        # Group by model
        model_breakdown: dict[str, dict] = {}
        for e in entries:
            bucket = model_breakdown.setdefault(e.model, {"calls": 0, "tokens": 0, "cost_usd": 0.0})
            bucket["calls"] += 1
            bucket["tokens"] += e.total_tokens
            bucket["cost_usd"] = round(bucket["cost_usd"] + e.cost_usd, 6)

        grounded_calls = sum(1 for e in entries if e.is_grounded)

        # Phase 2.12: Prompt caching aggregate stats
        total_cache_creation = sum(e.cache_creation_input_tokens for e in entries)
        total_cache_read = sum(e.cache_read_input_tokens for e in entries)
        cache_hit_entries = sum(1 for e in entries if e.cache_read_input_tokens > 0)
        cache_eligible = sum(1 for e in entries if e.cache_creation_input_tokens > 0 or e.cache_read_input_tokens > 0)

        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "total_calls": len(entries),
            "deep_think_calls": deep_think_calls,
            "grounded_calls": grounded_calls,
            "model_breakdown": model_breakdown,
            # Phase 2.12: Prompt caching summary
            "prompt_caching": {
                "total_cache_creation_tokens": total_cache_creation,
                "total_cache_read_tokens": total_cache_read,
                "cache_hit_entries": cache_hit_entries,
                "cache_eligible_entries": cache_eligible,
                "cache_hit_rate": round(cache_hit_entries / cache_eligible, 3) if cache_eligible > 0 else 0.0,
                "estimated_savings_pct": round(total_cache_read / total_input * 90, 1) if total_input > 0 and total_cache_read > 0 else 0.0,
            },
            "agents": [
                {
                    "agent_name": e.agent_name,
                    "model": e.model,
                    "input_tokens": e.input_tokens,
                    "output_tokens": e.output_tokens,
                    "total_tokens": e.total_tokens,
                    "cost_usd": e.cost_usd,
                    "is_deep_think": e.is_deep_think,
                    "is_grounded": e.is_grounded,
                    "cache_creation_input_tokens": e.cache_creation_input_tokens,
                    "cache_read_input_tokens": e.cache_read_input_tokens,
                }
                for e in entries
            ],
        }
