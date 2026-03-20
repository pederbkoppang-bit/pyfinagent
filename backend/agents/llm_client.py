"""
Unified LLM client abstraction — v3.4 Multi-Provider support.

Wraps Gemini (Vertex AI), Claude (Anthropic), and OpenAI (direct or GitHub Models)
behind a single LLMClient interface so orchestrator, debate, and risk_debate are
provider-agnostic.

Provider routing (factory priority order):
  1. Model in GITHUB_MODELS_CATALOG + github_token set  → OpenAIClient via GitHub Models
  2. Model starts with "claude-" + anthropic_api_key set → ClaudeClient (direct Anthropic)
  3. Model starts with "gpt-"/"o1"/"o3" + openai_api_key → OpenAIClient (direct OpenAI)
  4. Default → GeminiClient (Vertex AI, always available)

Constraint: Structured output schemas (Phase 3) and Google Search Grounding (Phase 4)
are Gemini-specific. When a non-Gemini model is selected, those features degrade
gracefully: structured output is injected as a JSON system prompt, and grounded
agents fall back to the general (non-grounded) client.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.config.settings import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GitHub Models catalog — models available via models.inference.ai.azure.com
# with a GitHub PAT (Copilot Pro subscription).
# ---------------------------------------------------------------------------
GITHUB_MODELS_CATALOG: set[str] = {
    # OpenAI models
    "gpt-4o",
    "gpt-4o-mini",
    # Anthropic models available through GitHub Models
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-6",
    # Meta
    "meta-llama-3.1-405b-instruct",
    "meta-llama-3.1-70b-instruct",
    "meta-llama-3.1-8b-instruct",
    # Microsoft
    "phi-4",
    "phi-3.5-moe-instruct",
    "phi-3.5-mini-instruct",
    "phi-3-medium-128k-instruct",
    # Mistral
    "mistral-large-2407",
    "mistral-nemo",
    "ministral-3b",
    "mistral-small",
}

# ---------------------------------------------------------------------------
# Common response/usage types
# ---------------------------------------------------------------------------

@dataclass
class UsageMeta:
    """Normalized token counts across all providers."""
    prompt_token_count: int = 0
    candidates_token_count: int = 0
    total_token_count: int = 0


@dataclass
class LLMResponse:
    """Provider-agnostic response container."""
    text: str
    thoughts: str = ""
    usage_metadata: UsageMeta = field(default_factory=UsageMeta)
    grounding_metadata: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Provider-agnostic LLM interface."""

    model_name: str

    @abstractmethod
    def generate_content(self, prompt: str, generation_config: dict | None = None) -> LLMResponse:
        """Generate content from a prompt.

        Args:
            prompt: The text prompt
            generation_config: Provider-specific config dict. Keys understood:
                - max_output_tokens (int)
                - temperature (float)
                - top_k (int)
                - response_mime_type ("application/json") — triggers JSON mode
                - response_schema (Pydantic model) — schema hint injected as system prompt
                - thinking (dict) — Gemini 2.5+ extended thinking config
                - include_thoughts (bool)

        Returns:
            LLMResponse with normalized text, thoughts, and usage_metadata
        """
        ...


# ---------------------------------------------------------------------------
# GeminiClient — wraps existing Vertex AI GenerativeModel
# ---------------------------------------------------------------------------

class GeminiClient(LLMClient):
    """Thin wrapper around a Vertex AI GenerativeModel.

    Preserves full Phase 3 (structured output), Phase 4 (grounding), and
    Phase 5 (extended thinking) compatibility since the underlying model is
    already wired for those features.
    """

    def __init__(self, model, model_name: str):
        """
        Args:
            model: A vertexai.generative_models.GenerativeModel instance
            model_name: String name for cost tracking (e.g. "gemini-2.0-flash")
        """
        self._model = model
        self.model_name = model_name

    def generate_content(self, prompt: str, generation_config: dict | None = None) -> LLMResponse:
        import concurrent.futures
        gen_kwargs = {"generation_config": generation_config} if generation_config else {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._model.generate_content, prompt, **gen_kwargs)
            response = future.result(timeout=120)

        # Extract text
        try:
            text = response.text
        except ValueError:
            parts = response.candidates[0].content.parts
            text = "\n".join(p.text for p in parts if hasattr(p, "text") and p.text)

        # Extract thoughts (Phase 5)
        thoughts = ""
        try:
            candidate = response.candidates[0] if response.candidates else None
            if candidate:
                for part in getattr(candidate.content, "parts", []) or []:
                    if hasattr(part, "thinking"):
                        thoughts = str(part.thinking)[:2000]
                        break
        except Exception:
            pass

        # Extract grounding (Phase 4)
        grounding_sources: list[dict] = []
        try:
            candidate = response.candidates[0] if response.candidates else None
            if candidate:
                gm = getattr(candidate, "grounding_metadata", None)
                if gm:
                    for chunk in getattr(gm, "grounding_chunks", []) or []:
                        web = getattr(chunk, "web", None)
                        if web:
                            grounding_sources.append({
                                "uri": getattr(web, "uri", ""),
                                "title": getattr(web, "title", ""),
                            })
        except Exception:
            pass

        # Normalize usage
        usage = getattr(response, "usage_metadata", None)
        umeta = UsageMeta(
            prompt_token_count=getattr(usage, "prompt_token_count", 0) or 0,
            candidates_token_count=getattr(usage, "candidates_token_count", 0) or 0,
            total_token_count=getattr(usage, "total_token_count", 0) or 0,
        ) if usage else UsageMeta()

        return LLMResponse(
            text=text,
            thoughts=thoughts,
            usage_metadata=umeta,
            grounding_metadata=grounding_sources,
        )


# ---------------------------------------------------------------------------
# OpenAIClient — covers both direct OpenAI and GitHub Models
# ---------------------------------------------------------------------------

class OpenAIClient(LLMClient):
    """Client for OpenAI-compatible APIs.

    Setting base_url="https://models.inference.ai.azure.com" routes calls
    through GitHub Models, which serves both OpenAI and Claude models
    under a Copilot Pro subscription — no per-provider key needed.
    """

    def __init__(self, model_name: str, api_key: str, base_url: str | None = None):
        self.model_name = model_name
        self._api_key = api_key
        self._base_url = base_url

    def _get_client(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai>=1.50.0")
        kwargs: dict = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return OpenAI(**kwargs)

    def generate_content(self, prompt: str, generation_config: dict | None = None) -> LLMResponse:
        config = generation_config or {}
        max_tokens = config.get("max_output_tokens", 2048)
        temperature = config.get("temperature", 0.0)

        # Build messages
        messages = [{"role": "user", "content": prompt}]

        # JSON mode — inject schema hint as system prompt if schema provided
        schema = config.get("response_schema")
        mime = config.get("response_mime_type", "")
        if mime == "application/json" or schema:
            schema_hint = ""
            if schema and hasattr(schema, "model_json_schema"):
                try:
                    schema_hint = f"\n\nYou MUST respond with valid JSON matching this exact schema:\n{json.dumps(schema.model_json_schema(), indent=2)}\n\nDo not include any text outside the JSON object."
                except Exception:
                    schema_hint = "\n\nYou MUST respond with a valid JSON object only. No prose outside the JSON."
            else:
                schema_hint = "\n\nYou MUST respond with a valid JSON object only. No prose outside the JSON."
            messages = [{"role": "system", "content": "You are a financial analysis AI." + schema_hint}] + messages

        client = self._get_client()
        kwargs: dict = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # JSON response format for OpenAI native models
        if (mime == "application/json" or schema) and not self._base_url:
            # GitHub Models doesn't always support response_format — skip for them
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)

        text = response.choices[0].message.content or ""
        usage = response.usage
        umeta = UsageMeta(
            prompt_token_count=getattr(usage, "prompt_tokens", 0) or 0,
            candidates_token_count=getattr(usage, "completion_tokens", 0) or 0,
            total_token_count=getattr(usage, "total_tokens", 0) or 0,
        ) if usage else UsageMeta()

        return LLMResponse(text=text, usage_metadata=umeta)


# ---------------------------------------------------------------------------
# ClaudeClient — direct Anthropic API
# ---------------------------------------------------------------------------

class ClaudeClient(LLMClient):
    """Client for Anthropic Claude models via direct API."""

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self._api_key = api_key

    def _get_client(self):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic>=0.49.0")
        return anthropic.Anthropic(api_key=self._api_key)

    def generate_content(self, prompt: str, generation_config: dict | None = None) -> LLMResponse:
        config = generation_config or {}
        max_tokens = config.get("max_output_tokens", 2048)
        temperature = config.get("temperature", 0.0)

        # JSON schema injection as system prompt
        schema = config.get("response_schema")
        mime = config.get("response_mime_type", "")
        system_prompt = "You are a financial analysis AI."
        if mime == "application/json" or schema:
            if schema and hasattr(schema, "model_json_schema"):
                try:
                    system_prompt += f"\n\nYou MUST respond with valid JSON matching this exact schema:\n{json.dumps(schema.model_json_schema(), indent=2)}\n\nDo not include any text outside the JSON object."
                except Exception:
                    system_prompt += "\n\nYou MUST respond with a valid JSON object only. No prose outside the JSON."
            else:
                system_prompt += "\n\nYou MUST respond with a valid JSON object only."

        client = self._get_client()
        kwargs: dict = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Extended thinking for Claude 3.7+ (when thinking_budget > 0)
        thinking_cfg = config.get("thinking")
        if thinking_cfg and isinstance(thinking_cfg, dict):
            budget = thinking_cfg.get("budget_tokens", 0)
            if budget > 0:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
                # Claude requires temperature=1 when thinking is enabled
                kwargs["temperature"] = 1

        response = client.messages.create(**kwargs)

        # Parse content blocks
        text = ""
        thoughts = ""
        for block in response.content:
            block_type = getattr(block, "type", "")
            if block_type == "thinking":
                thoughts = str(getattr(block, "thinking", ""))[:2000]
            elif block_type == "text":
                text += getattr(block, "text", "")

        usage = response.usage
        umeta = UsageMeta(
            prompt_token_count=getattr(usage, "input_tokens", 0) or 0,
            candidates_token_count=getattr(usage, "output_tokens", 0) or 0,
            total_token_count=(getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0),
        ) if usage else UsageMeta()

        return LLMResponse(text=text, thoughts=thoughts, usage_metadata=umeta)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_client(model_name: str, vertex_model, settings: "Settings") -> LLMClient:
    """Create the appropriate LLMClient for a model name.

    Priority:
      1. GitHub Models (model in catalog + GITHUB_TOKEN set)
      2. Direct Claude (model starts with "claude-" + ANTHROPIC_API_KEY set)
      3. Direct OpenAI (model starts with "gpt-"/"o1"/"o3" + OPENAI_API_KEY set)
      4. Gemini (default — always available, uses the pre-built GenerativeModel)

    Args:
        model_name: The model identifier string
        vertex_model: A pre-built vertexai.GenerativeModel (used for Gemini fallback)
        settings: App settings (for API keys)

    Returns:
        An LLMClient instance ready for generate_content() calls

    Raises:
        ValueError: If a non-Gemini model is selected but the required key is missing
    """
    github_token = getattr(settings, "github_token", "")
    anthropic_key = getattr(settings, "anthropic_api_key", "")
    openai_key = getattr(settings, "openai_api_key", "")

    # 1. GitHub Models — check catalog first (preferred for testing)
    if model_name in GITHUB_MODELS_CATALOG:
        if github_token:
            logger.info(f"[LLMClient] Routing {model_name} → GitHub Models (OpenAI-compatible)")
            return OpenAIClient(
                model_name=model_name,
                api_key=github_token,
                base_url="https://models.inference.ai.azure.com",
            )
        else:
            raise ValueError(
                f"Model '{model_name}' requires a GitHub Token (GITHUB_TOKEN) but none is set. "
                "Add GITHUB_TOKEN=ghp_... to backend/.env"
            )

    # 2. Direct Anthropic
    if model_name.startswith("claude-"):
        if anthropic_key:
            logger.info(f"[LLMClient] Routing {model_name} → Anthropic direct")
            return ClaudeClient(model_name=model_name, api_key=anthropic_key)
        else:
            raise ValueError(
                f"Model '{model_name}' requires ANTHROPIC_API_KEY but none is set. "
                "Add ANTHROPIC_API_KEY=sk-ant-... to backend/.env"
            )

    # 3. Direct OpenAI
    if model_name.startswith(("gpt-", "o1", "o3")):
        if openai_key:
            logger.info(f"[LLMClient] Routing {model_name} → OpenAI direct")
            return OpenAIClient(model_name=model_name, api_key=openai_key)
        else:
            raise ValueError(
                f"Model '{model_name}' requires OPENAI_API_KEY but none is set. "
                "Add OPENAI_API_KEY=sk-... to backend/.env"
            )

    # 4. Default: Gemini via Vertex AI
    logger.debug(f"[LLMClient] Routing {model_name} → Gemini (Vertex AI)")
    return GeminiClient(model=vertex_model, model_name=model_name)
