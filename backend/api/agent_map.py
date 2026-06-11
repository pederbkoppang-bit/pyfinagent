"""phase-18.1 GET /api/agent-map -- agent topology inventory.

Serves the static `backend/agents/_inventory.json` that catalogues all
~54 agents in pyfinagent (Layer 1 Gemini analysis pipeline + Layer 2
in-app Claude agents + Layer 3 Harness MAS + services + meta-evolution).
The frontend AgentMap component (phase-18.2+) renders this as an
interactive React Flow topology.

Returns the JSON inventory verbatim. No parameters, no auth -- this is
read-only metadata about the system itself, not user data.

Edges are derived from `parents`/`children` for visualization
convenience (avoids duplicate computation in the frontend).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api", tags=["agent-map"])

INVENTORY_PATH = (
    Path(__file__).resolve().parents[1] / "agents" / "_inventory.json"
)


def _derive_edges(nodes: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Compute edges from parents/children. Deduped by (from, to) tuple."""
    seen: set[tuple[str, str]] = set()
    edges: list[dict[str, str]] = []
    by_id = {n["id"] for n in nodes}
    for n in nodes:
        nid = n["id"]
        for child in n.get("children") or []:
            if child not in by_id:
                continue
            key = (nid, child)
            if key in seen:
                continue
            seen.add(key)
            edges.append({"from": nid, "to": child})
        for parent in n.get("parents") or []:
            if parent not in by_id:
                continue
            key = (parent, nid)
            if key in seen:
                continue
            seen.add(key)
            edges.append({"from": parent, "to": nid})
    return edges


# phase-22.1: map node ids -> resolve_model() roles. Only nodes that
# correspond to a real model_tiers role get a live_model injected.
# Nodes outside this map (e.g. layer1_pipeline group, non-LLM services)
# keep their static `model` field as-is.
_NODE_ID_TO_ROLE: dict[str, str] = {
    "main": "mas_main",
    "researcher": "mas_research",
    "qa": "mas_qa",
    "communication_agent": "mas_communication",
    "skill_optimizer": "autoresearch_strategic",
    "directive_rewriter": "autoresearch_strategic",
    "directive_review": "autoresearch_smart",
    # Layer-2 + Layer-1: these don't have explicit model_tiers roles today,
    # so we use mas_main (deep-think) for orchestrator/planner/analyst and
    # gemini_enrichment for the swappable Layer-1 skills. The override flag
    # in resolve_model() will propagate the operator's Standard Model
    # selection (gemini_model field) when apply_model_to_all_agents=true.
    "multi_agent_orchestrator": "mas_main",
    "planner_agent": "mas_main",
    "analyst_agent": "mas_communication",
    "evaluator_agent": "layer1_swappable",
    "moderator_agent": "mas_communication",
    "synthesis_agent": "layer1_swappable",
    # Layer-1 swappable skills -- use the new layer1_swappable role so the
    # apply_model_to_all_agents override propagates (gemini_enrichment is
    # _GEMINI_LOCKED_ROLES-locked and would NOT propagate).
    "bull_agent": "layer1_swappable",
    "bear_agent": "layer1_swappable",
    "devils_advocate_agent": "layer1_swappable",
    "bias_detector_skill": "layer1_swappable",
    "risk_judge_skill": "layer1_swappable",
    "earnings_tone_agent": "layer1_swappable",
    "nlp_sentiment_agent": "layer1_swappable",
    "social_sentiment_agent": "layer1_swappable",
    "insider_agent": "layer1_swappable",
    "options_agent": "layer1_swappable",
    "patent_agent": "layer1_swappable",
    "supply_chain_agent": "layer1_swappable",
    "alt_data_agent": "layer1_swappable",
    "anomaly_agent": "layer1_swappable",
    "sector_analysis_agent": "layer1_swappable",
    "sector_catalyst_agent": "layer1_swappable",
    "scenario_agent": "layer1_swappable",
    "info_gap_agent": "layer1_swappable",
    "critic_agent": "layer1_swappable",
    "quant_model_agent": "layer1_swappable",
    "quant_strategy": "layer1_swappable",
    "aggressive_analyst": "layer1_swappable",
    "conservative_analyst": "layer1_swappable",
    "neutral_analyst": "layer1_swappable",
    # Locked nodes (kept here so live_model still resolves; the
    # gemini_locked flag in inventory tells the UI to badge them)
    "rag_agent": "gemini_enrichment",
    "competitor_agent": "gemini_enrichment",
    "market_agent": "gemini_enrichment",
    "deep_dive_agent": "gemini_enrichment",
    "enhanced_macro_agent": "gemini_enrichment",
}


def _inject_live_model(node: dict[str, Any]) -> dict[str, Any]:
    """Add `live_model` to the node by resolving the operator's runtime model.

    Locked nodes (gemini_locked=true) always return their static Gemini
    workhorse pin regardless of operator override (phase-60.1: gemini-2.5-flash). Non-LLM nodes and nodes not in the
    role map fall through with no live_model field added.

    Returns the node dict (mutated in place is also fine; we return a copy
    for clarity).
    """
    node_id = node.get("id", "")
    out = dict(node)

    # Hard-locked nodes always show their static gemini model
    if node.get("gemini_locked"):
        # phase-60.1 (AW-4): fallback repinned from the discontinued gemini-2.0-flash.
        out["live_model"] = node.get("model") or "gemini-2.5-flash"
        return out

    role = _NODE_ID_TO_ROLE.get(node_id)
    if role is None:
        # No mapped role -- keep static model as-is, no live_model
        return out

    try:
        # Local import: avoid pulling settings into module-load time
        from backend.config.model_tiers import resolve_model
        out["live_model"] = resolve_model(role)
    except Exception:
        # Fail open: any resolver failure falls back to the static model
        out["live_model"] = node.get("model")
    return out


@router.get("/agent-map")
def get_agent_map() -> dict[str, Any]:
    """Return the agent-topology inventory + derived edges + live model resolution.

    phase-22.1: each node now carries a `live_model` field reflecting the
    operator's actual runtime model (respecting Settings overrides via
    resolve_model()). Locked nodes (gemini_locked=true) always show their
    static Gemini workhorse pin. The original static `model` field is preserved
    for backward compat.
    """
    if not INVENTORY_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"agent inventory missing at {INVENTORY_PATH}",
        )
    try:
        data = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"agent inventory is not valid JSON: {exc!r}",
        ) from exc

    nodes = data.get("nodes") or []
    data["nodes"] = [_inject_live_model(n) for n in nodes]
    data["edges"] = _derive_edges(nodes)
    return data


__all__ = ["router", "get_agent_map", "INVENTORY_PATH"]
