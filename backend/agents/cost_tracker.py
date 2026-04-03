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

# Pricing per 1M tokens (input, output) — June 2026
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Gemini (Vertex AI)
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.00),
    # Anthropic Claude (direct or via GitHub Models)
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-7-sonnet-20250219": (3.00, 15.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-opus-4": (15.00, 75.00),
    "claude-sonnet-4-6": (3.00, 15.00),
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
        # Anthropic prompt caching pricing: cache reads cost 90% less than regular input
        if cache_read > 0:
            regular_input = max(0, input_tokens - cache_read)
            cached_cost = cache_read * pricing[0] * 0.1 / 1_000_000  # 90% discount
            regular_cost = regular_input * pricing[0] / 1_000_000
            output_cost = output_tokens * pricing[1] / 1_000_000
            cost = cached_cost + regular_cost + output_cost
        else:
            cost = (input_tokens * pricing[0] + output_tokens * pricing[1]) / 1_000_000

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
