"""
Centralized model ID registry for the MAS + autoresearch stack.

Why this exists:
  Until this module, every Claude model ID was a hardcoded string in one of
  four places — agent_definitions.py (4 lines), run_memo.py env_defaults,
  and settings.py for Gemini. A launch-time cost switch required touching
  every one of them. Now resolve_model("role") is the single lookup point
  and `COST_TIER=live` flips the whole stack at once.

Why "build" is a verbatim snapshot:
  The build tier MUST remain byte-identical to the mapping that was live
  before this refactor. That is an explicit user directive — the
  refactor itself must be runtime-invisible. Any divergence is a bug
  and the byte-identical test in the plan's verification section will
  catch it.

Why "live" is a placeholder:
  The live tier is decided at May launch, not now. Every role resolves
  to a TODO sentinel that raises a clear exception if anyone accidentally
  flips COST_TIER=live before the mapping is decided. This prevents a
  silent misroute to a wrong model in production.

References:
  - .claude/context/research-gate.md (citations baked into the plan at
    /Users/ford/.claude/plans/parsed-tinkering-stallman.md)
  - Anthropic Multi-Agent blog: "Opus lead + Sonnet subagents outperforms
    single Opus by 90.2%" — validates the current build mapping.
  - C3PO (arXiv 2511.07396): reasoning roles drop at most one tier; the
    live mapping, when decided, must respect this cap.
"""

from __future__ import annotations

from typing import Literal

CostTier = Literal["build", "live"]

# Build tier: verbatim snapshot of the mapping that existed before this
# refactor. Lines cited for each role so a future auditor can git-blame
# back to the source.
_BUILD_TIER: dict[str, str] = {
    # agent_definitions.py:127
    "mas_communication": "claude-sonnet-4-6",
    # agent_definitions.py:177
    "mas_main": "claude-opus-4-6",
    # agent_definitions.py:225
    "mas_qa": "claude-opus-4-6",
    # agent_definitions.py:271
    "mas_research": "claude-sonnet-4-6",
    # scripts/autoresearch/run_memo.py env_defaults
    # Fixed 2026-04-18: removed "anthropic:" prefix which broke
    # make_client() routing (startswith("claude-") failed, silently
    # fell through to Gemini). MF-47.
    "autoresearch_fast": "claude-haiku-4-5",
    "autoresearch_smart": "claude-sonnet-4-6",
    "autoresearch_strategic": "claude-opus-4-6",
    # settings.py:28
    "gemini_enrichment": "gemini-2.0-flash",
    # settings.py:29
    "gemini_deep_think": "gemini-2.5-flash",
}

# Live tier: every value is the sentinel. resolve_model() raises if it
# sees this, so an accidental COST_TIER=live flip fails loud instead of
# silently routing to a wrong model. Replace entries one-by-one at May
# launch, guided by the C3PO "reasoning roles drop at most one tier"
# rule (see research-gate section of the plan file).
_LIVE_SENTINEL = "TODO_DECIDE_AT_LAUNCH"
_LIVE_TIER: dict[str, str] = {role: _LIVE_SENTINEL for role in _BUILD_TIER}

MODEL_TIER_MAP: dict[CostTier, dict[str, str]] = {
    "build": _BUILD_TIER,
    "live": _LIVE_TIER,
}


def resolve_model(role: str, tier: CostTier | None = None) -> str:
    """Look up the model ID for a role in the active cost tier.

    Args:
        role: one of the keys in _BUILD_TIER (e.g. "mas_main",
            "autoresearch_fast", "gemini_enrichment").
        tier: override the tier; if None, reads settings.cost_tier.

    Returns:
        The model ID string (e.g. "claude-opus-4-6").

    Raises:
        KeyError: unknown role.
        RuntimeError: tier is "live" and the role has not been mapped
            (still the TODO sentinel). Message includes the role so
            the operator knows exactly what to decide.
    """
    if tier is None:
        from backend.config.settings import get_settings
        tier = get_settings().cost_tier  # type: ignore[assignment]

    if tier not in MODEL_TIER_MAP:
        raise KeyError(f"unknown cost_tier {tier!r}; expected 'build' or 'live'")

    mapping = MODEL_TIER_MAP[tier]
    if role not in mapping:
        raise KeyError(
            f"unknown model role {role!r}; valid roles: {sorted(_BUILD_TIER)}"
        )

    model = mapping[role]
    if model == _LIVE_SENTINEL:
        raise RuntimeError(
            f"cost_tier=live but role {role!r} is still {_LIVE_SENTINEL}. "
            "Edit backend/config/model_tiers.py::_LIVE_TIER[{role!r}] "
            "before launching with COST_TIER=live."
        )
    return model


def build_tier_snapshot() -> dict[str, str]:
    """Return a copy of the build-tier map for the byte-identical test."""
    return dict(_BUILD_TIER)


# -----------------------------------------------------------------------
# phase-4.14.3 (MF-28) - output_config.effort defaults per agent class.
#
# Anthropic effort docs
# (https://platform.claude.com/docs/en/build-with-claude/effort) list
# the supported levels as low / medium / high / xhigh / max. Two hard
# constraints we have to respect:
#   - "xhigh" is accepted only by claude-opus-4-7. Sending it to any
#     other model returns a 400. We downgrade xhigh to high in
#     llm_client when the target is not Opus 4.7.
#   - Haiku 4.5 is not listed as a supported model for output_config.
#     We omit effort entirely on Haiku routes (value = None).
# API implicit default when effort is omitted = "high".
# Sonnet 4.6 recommended default in the docs is "medium" (the docs say
# to "explicitly set" it to avoid unexpected latency at high).
# Opus 4.7 recommended default for coding/agentic = "xhigh".
# -----------------------------------------------------------------------

Effort = Literal["low", "medium", "high", "xhigh", "max"]

EFFORT_SUPPORTED_MODELS: tuple[str, ...] = (
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-opus-4-1",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
)

EFFORT_DEFAULTS: dict[str, Effort | None] = {
    "mas_communication":       "low",
    "mas_main":                "high",
    "mas_qa":                  "high",
    "mas_research":            "medium",
    "autoresearch_fast":       None,
    "autoresearch_smart":      "medium",
    "autoresearch_strategic":  "high",
    "gemini_enrichment":       None,
    "gemini_deep_think":       None,
}

MODEL_EFFORT_FALLBACK: tuple[tuple[str, Effort | None], ...] = (
    ("claude-opus-4-7",   "xhigh"),
    ("claude-opus-4-6",   "high"),
    ("claude-opus-4-5",   "high"),
    ("claude-opus-4-1",   "high"),
    ("claude-sonnet-4-6", "medium"),
    ("claude-sonnet-4-5", "medium"),
    ("claude-haiku-4-5",  None),
)


def resolve_effort(role: str) -> Effort | None:
    """Look up the effort default for a role.

    Args:
        role: one of the keys in EFFORT_DEFAULTS. Must also appear in
            _BUILD_TIER so we don't silently accept typos.

    Returns:
        The effort level string, or None to mean omit output_config.

    Raises:
        KeyError: unknown role.
    """
    if role not in _BUILD_TIER:
        raise KeyError(
            f"unknown model role {role!r}; valid roles: {sorted(_BUILD_TIER)}"
        )
    return EFFORT_DEFAULTS.get(role)


def resolve_effort_by_model(model_id: str | None) -> Effort | None:
    """Fallback: derive effort default from a bare model ID prefix.

    Used when a caller does not supply a role (e.g. direct
    ClaudeClient instantiation outside the role-routed MAS path).
    """
    if not model_id:
        return None
    for prefix, effort in MODEL_EFFORT_FALLBACK:
        if model_id.startswith(prefix):
            return effort
    return None


def model_supports_effort(model_id: str | None) -> bool:
    """True if the model ID accepts output_config.effort per Anthropic docs."""
    if not model_id:
        return False
    return model_id.startswith(EFFORT_SUPPORTED_MODELS)
