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


@router.get("/agent-map")
def get_agent_map() -> dict[str, Any]:
    """Return the static agent-topology inventory + derived edges."""
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
    data["edges"] = _derive_edges(nodes)
    return data


__all__ = ["router", "get_agent_map", "INVENTORY_PATH"]
