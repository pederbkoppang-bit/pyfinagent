"""phase-21.1 tests for the apply_model_to_all_agents settings override.

Verifies that resolve_model() honors the operator's "Apply to all agents"
toggle but skips Gemini-locked roles (RAG / Search Grounding / Vertex
structured output).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.config.model_tiers import (  # noqa: E402
    _BUILD_TIER,
    _GEMINI_LOCKED_ROLES,
    resolve_model,
)


class _StubSettings:
    """Minimal stub mimicking pydantic Settings for resolve_model()."""

    def __init__(
        self,
        *,
        cost_tier: str = "build",
        apply_model_to_all_agents: bool = False,
        gemini_model: str = "claude-sonnet-4-6",
    ) -> None:
        self.cost_tier = cost_tier
        self.apply_model_to_all_agents = apply_model_to_all_agents
        self.gemini_model = gemini_model


def _patched(settings: _StubSettings):
    return patch("backend.config.settings.get_settings", return_value=settings)


# ----------------------
# Default behavior (override OFF) -- existing roles unchanged
# ----------------------

def test_default_off_returns_per_role_models():
    s = _StubSettings(apply_model_to_all_agents=False)
    with _patched(s):
        assert resolve_model("mas_main") == _BUILD_TIER["mas_main"]
        assert resolve_model("mas_communication") == _BUILD_TIER["mas_communication"]
        assert resolve_model("autoresearch_strategic") == _BUILD_TIER["autoresearch_strategic"]


def test_default_off_gemini_roles_unchanged():
    s = _StubSettings(apply_model_to_all_agents=False)
    with _patched(s):
        assert resolve_model("gemini_enrichment") == _BUILD_TIER["gemini_enrichment"]
        assert resolve_model("gemini_deep_think") == _BUILD_TIER["gemini_deep_think"]


# ----------------------
# Override ON -- non-Gemini roles take the override
# ----------------------

def test_override_applies_to_mas_main():
    s = _StubSettings(apply_model_to_all_agents=True, gemini_model="claude-haiku-4-5")
    with _patched(s):
        assert resolve_model("mas_main") == "claude-haiku-4-5"


def test_override_applies_to_all_anthropic_roles():
    s = _StubSettings(apply_model_to_all_agents=True, gemini_model="claude-opus-4-7")
    with _patched(s):
        for role in ("mas_main", "mas_qa", "mas_communication", "mas_research",
                     "autoresearch_fast", "autoresearch_smart", "autoresearch_strategic"):
            assert resolve_model(role) == "claude-opus-4-7", f"override should apply to {role}"


# ----------------------
# Override ON -- Gemini-locked roles still return their hardcoded model
# ----------------------

def test_override_skips_gemini_enrichment():
    s = _StubSettings(apply_model_to_all_agents=True, gemini_model="claude-opus-4-7")
    with _patched(s):
        # gemini_enrichment is in _GEMINI_LOCKED_ROLES; override is bypassed
        assert resolve_model("gemini_enrichment") == _BUILD_TIER["gemini_enrichment"]
        assert resolve_model("gemini_enrichment").startswith("gemini-")


def test_override_skips_gemini_deep_think():
    s = _StubSettings(apply_model_to_all_agents=True, gemini_model="claude-opus-4-7")
    with _patched(s):
        assert resolve_model("gemini_deep_think") == _BUILD_TIER["gemini_deep_think"]


def test_gemini_locked_roles_set_is_correct():
    """Both Gemini roles in _BUILD_TIER must be in _GEMINI_LOCKED_ROLES."""
    gemini_roles_in_build = {r for r in _BUILD_TIER if _BUILD_TIER[r].startswith("gemini-")}
    assert gemini_roles_in_build == set(_GEMINI_LOCKED_ROLES)


# ----------------------
# Edge cases
# ----------------------

def test_override_on_but_no_gemini_model_value_falls_through():
    """If override flag is set but gemini_model is empty string, fall back to per-role mapping."""
    s = _StubSettings(apply_model_to_all_agents=True, gemini_model="")
    with _patched(s):
        assert resolve_model("mas_main") == _BUILD_TIER["mas_main"]


def test_explicit_tier_arg_bypasses_settings():
    """When `tier` is passed explicitly, settings (and override) are ignored."""
    s = _StubSettings(apply_model_to_all_agents=True, gemini_model="claude-haiku-4-5")
    with _patched(s):
        # explicit tier="build" -> direct mapping lookup, no settings read
        assert resolve_model("mas_main", tier="build") == _BUILD_TIER["mas_main"]


def test_unknown_role_still_raises():
    s = _StubSettings(apply_model_to_all_agents=True, gemini_model="claude-haiku-4-5")
    with _patched(s):
        with pytest.raises(KeyError):
            resolve_model("nonexistent_role")
