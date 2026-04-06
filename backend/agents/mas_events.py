"""
MAS Event Emitter — Real-time observability for the dashboard.

Emits structured events from multi_agent_orchestrator.py so the frontend
can render a live node graph of agent activity.

Architecture:
  - EventBus holds an asyncio.Queue per subscriber
  - Orchestrator calls emit() at each step
  - FastAPI SSE endpoint streams events to frontend
  - Events are also buffered in-memory (last 200) for late joiners

Event types (map to dashboard nodes):
  classify    — Communication Agent routing decision
  plan        — Ford's planning step
  delegate    — Subagent spawned with 4-component delegation
  tool_call   — Subagent called a harness tool
  tool_result — Tool returned data
  thinking    — Interleaved thinking block (summarized)
  synthesize  — Ford merging subagent findings
  loop_check  — "More research needed?" decision
  quality_gate — Quality Gate scoring result
  citation    — CitationAgent processing
  complete    — Final response ready
  memory_save — Plan/result persisted to memory
  mask        — Observation masking applied
  error       — Error in any step

References:
  https://www.anthropic.com/engineering/multi-agent-research-system
  (Observability section: "debugging benefits from...")
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional
from collections import deque

logger = logging.getLogger(__name__)

# ── Event Schema ────────────────────────────────────────────────


@dataclass
class MASEvent:
    """Single event from the MAS orchestrator."""
    event_type: str               # classify, plan, delegate, tool_call, etc.
    agent: str                    # Which agent emitted this
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_id: str = ""              # Groups events from one user query
    iteration: int = 0            # Research iteration number (0 = first)
    data: dict = field(default_factory=dict)  # Event-specific payload
    duration_ms: float = 0.0      # Time taken for this step
    tokens: dict = field(default_factory=dict)  # {"input": N, "output": N}

    def to_dict(self) -> dict:
        return asdict(self)

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        return f"data: {json.dumps(self.to_dict(), default=str)}\n\n"


# ── Event Bus ───────────────────────────────────────────────────

# Buffer size for late joiners
_BUFFER_SIZE = 200


class MASEventBus:
    """Pub/sub event bus for MAS observability.

    Usage:
        bus = get_event_bus()

        # Producer (orchestrator):
        bus.emit(MASEvent(event_type="classify", agent="Communication", ...))

        # Consumer (SSE endpoint):
        async for event in bus.subscribe():
            yield event.to_sse()
    """

    def __init__(self):
        self._subscribers: list[asyncio.Queue] = []
        self._buffer: deque[MASEvent] = deque(maxlen=_BUFFER_SIZE)
        self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None
        self._total_events = 0

    def emit(self, event: MASEvent) -> None:
        """Emit an event to all subscribers + buffer."""
        self._buffer.append(event)
        self._total_events += 1

        # Non-blocking push to all subscriber queues
        dead = []
        for i, queue in enumerate(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(f"[MASEventBus] Subscriber {i} queue full, dropping event")
            except Exception:
                dead.append(i)

        # Clean up dead subscribers
        for i in reversed(dead):
            self._subscribers.pop(i)

        # Log significant events
        if event.event_type in ("classify", "complete", "quality_gate", "error"):
            logger.info(
                f"[MAS] {event.event_type} | {event.agent} | "
                f"run={event.run_id[:8] if event.run_id else '?'} | "
                f"{event.duration_ms:.0f}ms"
            )

    async def subscribe(self, include_buffer: bool = True):
        """Subscribe to events. Yields MASEvent objects.

        If include_buffer=True, replays recent events first (for late joiners).
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._subscribers.append(queue)

        try:
            # Replay buffer for late joiners
            if include_buffer:
                for event in self._buffer:
                    yield event

            # Stream live events
            while True:
                event = await queue.get()
                yield event
        finally:
            # Unsubscribe on disconnect
            try:
                self._subscribers.remove(queue)
            except ValueError:
                pass

    def get_buffer(self, run_id: Optional[str] = None) -> list[dict]:
        """Get buffered events, optionally filtered by run_id."""
        events = list(self._buffer)
        if run_id:
            events = [e for e in events if e.run_id == run_id]
        return [e.to_dict() for e in events]

    @property
    def stats(self) -> dict:
        return {
            "total_events": self._total_events,
            "buffer_size": len(self._buffer),
            "subscribers": len(self._subscribers),
        }


# ── Singleton ───────────────────────────────────────────────────

_event_bus: Optional[MASEventBus] = None


def get_event_bus() -> MASEventBus:
    """Get the global MAS event bus singleton."""
    global _event_bus
    if _event_bus is None:
        _event_bus = MASEventBus()
    return _event_bus


# ── Helper: generate run IDs ───────────────────────────────────

def make_run_id() -> str:
    """Generate a short unique run ID for grouping events."""
    import hashlib
    raw = f"{time.time()}-{id(asyncio.get_event_loop)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
