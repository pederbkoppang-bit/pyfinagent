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


def _normalize_model_name(model_name: str) -> str:
    """Collapse namespaced GitHub Models IDs back to canonical model keys."""
    if not model_name:
        return ""
    if "/" not in model_name:
        return model_name
    return model_name.split("/", 1)[1]

# ---------------------------------------------------------------------------
# GitHub Models catalog — models available via models.github.ai/inference
# with a GitHub PAT (Copilot Pro subscription).
# API uses namespaced model IDs: {publisher}/{model_name}  e.g. openai/gpt-4.1
# ---------------------------------------------------------------------------
GITHUB_MODELS_CATALOG: set[str] = {
    # OpenAI models
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-5",
    "gpt-5-chat",
    "gpt-5-mini",
    "gpt-5-nano",
    "o1",
    "o1-mini",
    "o1-preview",
    "o3",
    "o3-mini",
    "o4-mini",
    # Anthropic models (available on GitHub Models)
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4",
    "claude-opus-4",
    # Meta
    "meta-llama-3.1-405b-instruct",
    "meta-llama-3.1-8b-instruct",
    "llama-3.3-70b-instruct",
    "llama-4-maverick",
    "llama-4-scout",
    # Microsoft
    "phi-4",
    "mai-ds-r1",
    "phi-4-mini-instruct",
    "phi-4-mini-reasoning",
    "phi-4-reasoning",
    # Mistral
    "ministral-3b",
    "codestral-2501",
    "mistral-medium-2505",
    "mistral-small-2503",
    # DeepSeek
    "deepseek-r1",
    "deepseek-r1-0528",
    "deepseek-v3-0324",
    # xAI
    "grok-3",
    "grok-3-mini",
}

# ---------------------------------------------------------------------------
# Per-model input size limits (approximate character caps).
# GitHub Models enforces hard request-body limits on smaller/cheaper models.
# 1 token ≈ 3.5–4 chars; we use conservative limits to leave headroom.
# Models NOT in this dict are treated as unconstrained.
# ---------------------------------------------------------------------------
_MODEL_MAX_INPUT_CHARS: dict[str, int] = {
    # Standard GitHub low/high-tier 8K input models
    "gpt-4.1":          26_000,
    "gpt-4o":           26_000,
    # GitHub Models o-series — un-gated large context
    "o1":         500_000,   # ~128K tokens — generous, no hard cap known
    "o3":         500_000,   # ~200K tokens — generous
    # Custom tier: 4,000 token in / 4,000 token out limit (from GitHub Models rate table)
    # 4,000 tokens × 3.5 chars = ~14K chars; use 13K for safety headroom
    "o1-mini":          13_000,
    "o1-preview":       13_000,
    "o3-mini":          13_000,
    "o4-mini":          56_000,   # confirmed ~16K tokens
    "gpt-5":            13_000,
    "gpt-5-chat":       13_000,
    "gpt-5-mini":       13_000,
    "gpt-5-nano":       13_000,
    "deepseek-r1":      13_000,
    "deepseek-r1-0528": 13_000,
    "grok-3":           13_000,
    "grok-3-mini":      13_000,
    "mai-ds-r1":        13_000,
    # Low tier: 8,000 token in limit (~26K chars)
    "gpt-4.1-mini":     26_000,
    "gpt-4.1-nano":     26_000,
    "gpt-4o-mini":      26_000,
    # Small models with limited context
    "ministral-3b":              14_000,
    "meta-llama-3.1-8b-instruct": 14_000,
    "phi-4-mini-instruct":        14_000,
    "phi-4-mini-reasoning":       14_000,
}


def get_model_max_input_chars(model_name: str) -> int | None:
    """Return the maximum prompt character count for a model, or None if unconstrained.

    The lookup checks the resolved API model name (after GitHub Models ID mapping),
    so callers should pass the name exactly as it will be sent to the API.
    """
    canonical_name = _normalize_model_name(model_name)
    explicit_limit = _MODEL_MAX_INPUT_CHARS.get(canonical_name)
    if explicit_limit is not None:
        return explicit_limit
    if canonical_name in GITHUB_MODELS_CATALOG:
        return 26_000
    return None


# Map our canonical model names → GitHub Models namespaced API identifiers.
# New endpoint (models.github.ai/inference) requires {publisher}/{model_name} format.
# See: https://docs.github.com/en/rest/models/inference
_GITHUB_MODELS_ID_MAP: dict[str, str] = {
    # OpenAI — openai/{model_name}
    "gpt-4o":           "openai/gpt-4o",
    "gpt-4o-mini":      "openai/gpt-4o-mini",
    "gpt-4.1":          "openai/gpt-4.1",
    "gpt-4.1-mini":     "openai/gpt-4.1-mini",
    "gpt-4.1-nano":     "openai/gpt-4.1-nano",
    "gpt-5":            "openai/gpt-5",
    "gpt-5-chat":       "openai/gpt-5-chat",
    "gpt-5-mini":       "openai/gpt-5-mini",
    "gpt-5-nano":       "openai/gpt-5-nano",
    "o1":               "openai/o1",
    "o1-mini":          "openai/o1-mini",
    "o1-preview":       "openai/o1-preview",
    "o3":               "openai/o3",
    "o3-mini":          "openai/o3-mini",
    "o4-mini":          "openai/o4-mini",
    # Anthropic — anthropic/{model_name}
    "claude-3-5-sonnet-20241022": "anthropic/claude-3.5-sonnet",
    "claude-3-5-haiku-20241022":  "anthropic/claude-3.5-haiku",
    "claude-3-7-sonnet-20250219": "anthropic/claude-3.7-sonnet",
    "claude-sonnet-4":            "anthropic/claude-sonnet-4",
    "claude-opus-4":              "anthropic/claude-opus-4",
    # Meta — meta/{model_name}
    "meta-llama-3.1-405b-instruct": "meta/meta-llama-3.1-405b-instruct",
    "meta-llama-3.1-8b-instruct":   "meta/meta-llama-3.1-8b-instruct",
    "llama-3.3-70b-instruct":       "meta/llama-3.3-70b-instruct",
    "llama-4-maverick":             "meta/llama-4-maverick-17b-128e-instruct-fp8",
    "llama-4-scout":                "meta/llama-4-scout-17b-16e-instruct",
    # Microsoft — microsoft/{model_name}
    "phi-4":                  "microsoft/phi-4",
    "mai-ds-r1":              "microsoft/mai-ds-r1",
    "phi-4-mini-instruct":    "microsoft/phi-4-mini-instruct",
    "phi-4-mini-reasoning":   "microsoft/phi-4-mini-reasoning",
    "phi-4-reasoning":        "microsoft/phi-4-reasoning",
    # Mistral — mistral-ai/{model_name}
    "ministral-3b":       "mistral-ai/ministral-3b",
    "codestral-2501":     "mistral-ai/codestral-2501",
    "mistral-medium-2505": "mistral-ai/mistral-medium-2505",
    "mistral-small-2503": "mistral-ai/mistral-small-2503",
    # DeepSeek — deepseek/{model_name}
    "deepseek-r1":      "deepseek/deepseek-r1",
    "deepseek-r1-0528": "deepseek/deepseek-r1-0528",
    "deepseek-v3-0324": "deepseek/deepseek-v3-0324",
    # xAI — xai/{model_name}
    "grok-3":      "xai/grok-3",
    "grok-3-mini": "xai/grok-3-mini",
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

    @staticmethod
    def _flatten_schema(schema: dict) -> dict:
        """Convert Pydantic JSON Schema to the Vertex AI OpenAPI subset.

        Uses a WHITELIST approach — only the keys Vertex AI's Schema proto
        explicitly supports are kept. Everything else is dropped automatically,
        so this never needs updating when Pydantic emits new keywords.

        Vertex AI supported keys (OpenAPI 3.0 subset):
            type, format, description, nullable, enum,
            properties, required, items

        Also handles:
          - $ref / $defs  → inlined recursively
          - anyOf: [T, null]  → {T, nullable: true}  (Pydantic Optional[T])

        IMPORTANT: The whitelist only applies to schema objects (where keys are
        schema keywords). The value of a "properties" key is a field_name →
        schema_object mapping, so field names must not be filtered.
        """
        _ALLOWED = frozenset({
            "type", "format", "description", "nullable",
            "enum", "properties", "required", "items",
        })
        defs = schema.get("$defs", {})

        def _resolve(obj: object) -> object:
            if isinstance(obj, dict):
                # 1. Resolve $ref — inline the referenced definition
                if "$ref" in obj:
                    ref_name = obj["$ref"].split("/")[-1]
                    return _resolve(defs.get(ref_name, {}))

                # 2. Collapse anyOf: [T, {"type": "null"}]  →  {...T, nullable: true}
                #    Pydantic v2 emits this for every Optional[T] / T | None field.
                if "anyOf" in obj:
                    variants = obj["anyOf"]
                    non_null = [v for v in variants if v != {"type": "null"}]
                    if len(non_null) == 1:
                        resolved = _resolve(non_null[0])
                        if isinstance(resolved, dict):
                            result = dict(resolved)
                            result["nullable"] = True
                            # Carry over description from the wrapper if not already set
                            if "description" in obj and "description" not in result:
                                result["description"] = obj["description"]
                            return result
                    # Multi-variant anyOf (Union of non-null types) → generic object
                    return {"type": "object", "nullable": True}

                # 3. Whitelist — keep only Vertex AI-supported schema keys.
                #    Special case: the value of "properties" is a {field_name: schema}
                #    mapping — field names are NOT schema keywords, so we preserve them
                #    and only apply whitelist/resolve to the field schema values.
                result = {}
                for k, v in obj.items():
                    if k not in _ALLOWED:
                        continue
                    if k == "properties" and isinstance(v, dict):
                        # v is {field_name: schema_object} — keep all field names,
                        # but recurse into each field's schema object normally
                        result[k] = {field: _resolve(field_schema)
                                     for field, field_schema in v.items()}
                    elif k == "type" and isinstance(v, str):
                        # Vertex AI protobuf Type enum expects UPPERCASE
                        # (STRING, NUMBER, INTEGER, BOOLEAN, ARRAY, OBJECT)
                        result[k] = v.upper()
                    else:
                        result[k] = _resolve(v)
                return result

            if isinstance(obj, list):
                return [_resolve(item) for item in obj]
            return obj

        resolved = _resolve({k: v for k, v in schema.items() if k != "$defs"})
        return resolved if isinstance(resolved, dict) else {"type": "object"}

    def generate_content(self, prompt: str, generation_config: dict | None = None) -> LLMResponse:
        import concurrent.futures
        # Convert Pydantic model class → flattened JSON schema dict for response_schema.
        # Vertex AI SDK v1.141+ no longer auto-converts Pydantic classes to proto Schema,
        # and it also rejects $defs/$ref — so we must fully inline all nested definitions.
        if generation_config and "response_schema" in generation_config:
            schema = generation_config["response_schema"]
            if isinstance(schema, type) and hasattr(schema, "model_json_schema"):
                raw = schema.model_json_schema()
                logger.debug(f"[GeminiClient] Converting Pydantic class {schema.__name__} to dict (has $defs: {'$defs' in raw})")
                schema = raw
            if isinstance(schema, dict):
                schema = self._flatten_schema(schema)
                # Verify no banned fields remain
                _dump = str(schema)
                if "$defs" in _dump or "$ref" in _dump or "anyOf" in _dump:
                    logger.error(f"[GeminiClient] SCHEMA STILL HAS BANNED FIELDS after flatten: {_dump[:500]}")
            generation_config = {**generation_config, "response_schema": schema}
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

    Setting base_url="https://models.github.ai/inference" routes calls
    through GitHub Models (new endpoint), which serves OpenAI, Anthropic, Meta,
    Microsoft, and Mistral models under a Copilot Pro subscription.
    Model IDs must use namespaced format: {publisher}/{model_name}.
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

        # Safety-net: truncate prompt if model has a known input character limit.
        # This catches any call (debate, risk_debate, synthesis, etc.) that would
        # otherwise get a 413 / tokens_limit_reached from GitHub Models.
        _max_chars = _MODEL_MAX_INPUT_CHARS.get(self.model_name)
        if _max_chars:
            # Total chars across all messages (system + user)
            _total_chars = sum(len(m.get("content", "") or "") for m in messages)
            if _total_chars > _max_chars:
                # Truncate the user message (last in list) to fit within the budget
                _overhead = sum(len(m.get("content", "") or "") for m in messages[:-1])
                _budget = _max_chars - _overhead - 200  # 200 chars for suffix
                if _budget > 0:
                    _orig = messages[-1]["content"] or ""
                    messages[-1]["content"] = _orig[:_budget] + "\n\n[Context truncated — model input limit]"
                logger.warning(
                    f"[{self.model_name}] Prompt ({_total_chars:,} chars) exceeds "
                    f"limit ({_max_chars:,} chars). Truncated to fit."
                )

        client = self._get_client()

        # o-series reasoning models require max_completion_tokens (not max_tokens)
        # and do not support the temperature parameter.
        _is_reasoning = self.model_name.startswith(("o1", "o3", "o4"))

        kwargs: dict = {
            "model": self.model_name,
            "messages": messages,
        }
        if _is_reasoning:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature

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
            api_model_id = _GITHUB_MODELS_ID_MAP.get(model_name, model_name)
            logger.info(f"[LLMClient] Routing {model_name} → GitHub Models as '{api_model_id}'")
            return OpenAIClient(
                model_name=api_model_id,
                api_key=github_token,
                base_url="https://models.github.ai/inference",
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
    if model_name.startswith(("gpt-", "o1", "o3", "o4")):
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
