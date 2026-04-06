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
