"""
MAS Events SSE API — Real-time event stream for dashboard observability.

Endpoints:
  GET /api/mas/events     — SSE stream of live MAS events
  GET /api/mas/events/buffer — Get buffered events (last 200, optional ?run_id=)
  GET /api/mas/events/stats  — Event bus statistics
"""

import asyncio
import logging

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from backend.agents.mas_events import get_event_bus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/mas", tags=["MAS Observability"])


@router.get("/events")
async def stream_events(include_buffer: bool = Query(True)):
    """SSE endpoint — streams MAS events in real-time.

    Connect from frontend:
        const es = new EventSource('/api/mas/events');
        es.onmessage = (e) => { const event = JSON.parse(e.data); ... };
    """
    bus = get_event_bus()

    async def event_generator():
        async for event in bus.subscribe(include_buffer=include_buffer):
            yield event.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/events/buffer")
async def get_buffer(run_id: str = Query(None)):
    """Get buffered events (last 200). Optionally filter by run_id."""
    bus = get_event_bus()
    return {"events": bus.get_buffer(run_id=run_id)}


@router.get("/events/stats")
async def get_stats():
    """Event bus statistics."""
    bus = get_event_bus()
    return bus.stats


@router.get("/dashboard")
async def get_dashboard():
    """Full MAS dashboard data for Slack App Home + frontend /agents page.

    Returns: agent configs, system health, event bus stats, cost summary.
    """
    from backend.agents.agent_definitions import AGENT_CONFIGS, AgentType
    import httpx

    # Agent configs (current models)
    agents = {}
    for agent_type, config in AGENT_CONFIGS.items():
        agents[agent_type.value] = {
            "name": config.name,
            "model": config.model,
            "max_tokens": config.max_tokens,
            "description": config.description,
            "can_delegate_to": config.can_delegate_to,
        }

    # Event bus stats
    bus = get_event_bus()
    event_stats = bus.stats

    # System health
    health = {"backend": "unknown", "frontend": "unknown"}
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get("http://localhost:8000/api/health")
            health["backend"] = "ok" if r.status_code == 200 else "error"
    except Exception:
        health["backend"] = "down"
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get("http://localhost:3000/")
            health["frontend"] = "ok" if r.status_code == 200 else "error"
    except Exception:
        health["frontend"] = "down"

    # Cost tracker (if available)
    cost_summary = None
    try:
        from backend.agents.cost_tracker import CostTracker
        tracker = CostTracker()
        cost_summary = tracker.summarize()
    except Exception:
        pass

    # Recent events
    recent_events = bus.get_buffer()[-10:]

    return {
        "agents": agents,
        "health": health,
        "event_stats": event_stats,
        "cost_summary": cost_summary,
        "recent_events": recent_events,
    }


@router.post("/agents/{agent_type}/model")
async def update_agent_model(agent_type: str, body: dict):
    """Update an agent's model at runtime.

    Body: {"model": "claude-sonnet-4-6"}
    """
    from backend.agents.agent_definitions import AGENT_CONFIGS, AgentType

    type_map = {t.value: t for t in AgentType}
    at = type_map.get(agent_type)
    if not at or at not in AGENT_CONFIGS:
        return {"error": f"Unknown agent: {agent_type}"}

    new_model = body.get("model", "")
    if not new_model:
        return {"error": "model is required"}

    old_model = AGENT_CONFIGS[at].model
    AGENT_CONFIGS[at].model = new_model

    logger.info(f"🔄 Agent model changed: {agent_type} {old_model} → {new_model}")
    return {"ok": True, "agent": agent_type, "old_model": old_model, "new_model": new_model}
