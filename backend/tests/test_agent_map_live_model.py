"""phase-22.1 tests for live model resolution in /api/agent-map.

Verifies:
- The endpoint injects a `live_model` field per node by calling resolve_model()
- When apply_model_to_all_agents=True, swappable nodes get the operator's gemini_model override
- gemini_locked nodes (RAGAgent) ALWAYS show their static Gemini workhorse pin regardless
- Backward-compat: existing static `model` field is preserved
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.api.agent_map import (  # noqa: E402
    _NODE_ID_TO_ROLE,
    _inject_live_model,
    get_agent_map,
)


class _StubSettings:
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
# Default endpoint behavior (no override)
# ----------------------

def test_endpoint_injects_live_model_field():
    """Every node with a mapped role gets a live_model field on the endpoint response."""
    s = _StubSettings(apply_model_to_all_agents=False)
    with _patched(s):
        out = get_agent_map()
    nodes = out["nodes"]
    # mas_main role -> claude-fable-5 (phase-59.1: Fable 5 adoption on the
    # rare-event orchestrator role; was claude-opus-4-8 since 2026-05-28,
    # claude-opus-4-7 before that)
    main_node = next(n for n in nodes if n["id"] == "main")
    assert main_node.get("live_model") == "claude-fable-5"


def test_swappable_nodes_get_default_when_override_off():
    """Swappable Layer-1 skills get the gemini_enrichment default."""
    s = _StubSettings(apply_model_to_all_agents=False)
    with _patched(s):
        out = get_agent_map()
    bull = next(n for n in out["nodes"] if n["id"] == "bull_agent")
    assert bull["live_model"] == "gemini-2.5-flash"


# ----------------------
# Override propagation (apply_model_to_all_agents=True)
# ----------------------

def test_override_propagates_to_swappable_layer1_skills():
    """When operator picks Claude in Settings, Layer-1 swappable skills get Claude."""
    s = _StubSettings(apply_model_to_all_agents=True, gemini_model="claude-haiku-4-5")
    with _patched(s):
        out = get_agent_map()
    by_id = {n["id"]: n for n in out["nodes"]}
    # bull_agent is swappable -> picks up the override
    assert by_id["bull_agent"]["live_model"] == "claude-haiku-4-5"
    # bear_agent is swappable
    assert by_id["bear_agent"]["live_model"] == "claude-haiku-4-5"
    # mas_main is not Gemini-locked, so override applies
    assert by_id["main"]["live_model"] == "claude-haiku-4-5"


def test_locked_node_bypasses_override():
    """RAGAgent ALWAYS shows its static Gemini pin regardless of operator's choice."""
    s = _StubSettings(apply_model_to_all_agents=True, gemini_model="claude-opus-4-7")
    with _patched(s):
        out = get_agent_map()
    rag = next(n for n in out["nodes"] if n["id"] == "rag_agent")
    assert rag.get("gemini_locked") is True
    assert rag["live_model"] == "gemini-2.5-flash"


# ----------------------
# Static model field preserved
# ----------------------

def test_static_model_field_preserved():
    """The original `model` field stays intact for backward compat."""
    s = _StubSettings(apply_model_to_all_agents=True, gemini_model="claude-haiku-4-5")
    with _patched(s):
        out = get_agent_map()
    bull = next(n for n in out["nodes"] if n["id"] == "bull_agent")
    assert bull.get("model") == "gemini-2.5-flash"  # static unchanged
    assert bull.get("live_model") == "claude-haiku-4-5"  # live reflects override


# ----------------------
# Defensive
# ----------------------

def test_inject_live_model_no_role_falls_through():
    """A node with no role mapping returns unchanged (no live_model added)."""
    out = _inject_live_model({"id": "nonexistent_xyz", "model": "static-x"})
    assert out["model"] == "static-x"
    assert "live_model" not in out


def test_node_id_to_role_map_covers_known_agents():
    """Sanity: the role map references real model_tiers roles."""
    from backend.config.model_tiers import _BUILD_TIER
    valid_roles = set(_BUILD_TIER.keys())
    for node_id, role in _NODE_ID_TO_ROLE.items():
        assert role in valid_roles, f"node {node_id!r} maps to unknown role {role!r}"
