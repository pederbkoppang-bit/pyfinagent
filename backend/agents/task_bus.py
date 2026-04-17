"""Intra-process MAS task-delegation bus (phase-3.7 step 3.7.4).

Asyncio.Queue-backed message bus for the Data / Strategy / Risk
three-node MAS. Each agent owns one Queue; the bus provides a
single `delegate()` helper that enforces per-hop timeouts (expiry),
an explicit retry loop for transient failures, and a cancellation
path that does not leak (per Python 3.14 asyncio best practice).

Why not the A2A SDK here: A2A mandates HTTP/JSON-RPC transport and
has no in-process binding. For a fixed 3-node local topology the
researcher-recommended path is stdlib asyncio.Queue (see ADR 0002
and phase-3.7 step 3.7.0). A2A belongs at the external service
boundary, not between two coroutines in the same process.

Envelope mirrors the A2A Task shape (id / contextId / status /
artifacts / history) so that the bus can be replaced with an A2A
transport later without changing caller code.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable


AgentHandler = Callable[[dict], Awaitable[dict]]


@dataclass
class TaskEnvelope:
    """A2A-shape task envelope for intra-process delegation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context_id: str = ""
    status: str = "submitted"  # submitted / working / completed / failed / canceled
    artifacts: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    expires_at: float | None = None  # monotonic deadline
    payload: dict = field(default_factory=dict)

    def record(self, hop: str, kind: str, detail: dict | None = None) -> None:
        self.history.append({
            "hop": hop,
            "kind": kind,
            "at": datetime.now(timezone.utc).isoformat(),
            "detail": detail or {},
        })


class AsyncTaskBus:
    """Per-agent asyncio.Queue bus with retry + expiry + cancellation.

    A single AsyncTaskBus instance holds one Queue per registered
    agent. Callers `delegate(target_agent, envelope, timeout)` and
    receive the downstream agent's response or a TimeoutError after
    the configured retry budget is exhausted.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentHandler] = {}
        self._workers: list[asyncio.Task] = []
        self._queues: dict[str, asyncio.Queue] = {}
        self._started = False

    def register(self, name: str, handler: AgentHandler) -> None:
        """Register an agent handler. Raises if the bus is already started."""
        if self._started:
            raise RuntimeError("cannot register after start()")
        self._agents[name] = handler
        self._queues[name] = asyncio.Queue()

    async def start(self) -> None:
        """Spawn one worker per registered agent."""
        if self._started:
            return
        self._started = True
        for name, handler in self._agents.items():
            self._workers.append(
                asyncio.create_task(self._worker(name, handler),
                                     name=f"task_bus.{name}")
            )

    async def stop(self) -> None:
        """Cancel all workers cleanly."""
        for w in self._workers:
            w.cancel()
        for w in self._workers:
            try:
                await w
            except asyncio.CancelledError:
                pass
        self._workers.clear()
        self._started = False

    async def _worker(self, name: str, handler: AgentHandler) -> None:
        queue = self._queues[name]
        while True:
            msg: tuple[TaskEnvelope, asyncio.Future] = await queue.get()
            envelope, response_fut = msg
            if response_fut.done():
                continue  # caller already timed out
            envelope.status = "working"
            envelope.record(name, "start")
            try:
                result = await handler(envelope.payload)
                envelope.status = "completed"
                envelope.artifacts.append({"hop": name, "result": result})
                envelope.record(name, "done")
                if not response_fut.done():
                    response_fut.set_result(result)
            except Exception as e:
                envelope.status = "failed"
                envelope.record(name, "error",
                                 {"type": type(e).__name__, "msg": str(e)})
                if not response_fut.done():
                    response_fut.set_exception(e)

    async def delegate(self, target: str, envelope: TaskEnvelope,
                        timeout: float = 0.5,
                        max_retries: int = 2) -> dict:
        """Send envelope to target agent, await response with retry+expiry.

        - Retries on asyncio.TimeoutError AND on transient
          exceptions flagged via a `transient=True` attribute on
          the exception.
        - Cancellation is explicit (task.cancel + await) so no
          leaked tasks. See researcher notes on Python 3.14.
        """
        if target not in self._queues:
            raise KeyError(f"unknown agent: {target}")

        last_exc: BaseException | None = None
        for attempt in range(max_retries + 1):
            envelope.record(target, f"delegate_attempt_{attempt + 1}")
            response_fut: asyncio.Future = asyncio.get_event_loop().create_future()
            await self._queues[target].put((envelope, response_fut))
            try:
                return await asyncio.wait_for(
                    asyncio.shield(response_fut), timeout=timeout,
                )
            except asyncio.TimeoutError as e:
                last_exc = e
                envelope.record(target, "timeout", {"attempt": attempt + 1})
                if not response_fut.done():
                    response_fut.cancel()
                continue
            except Exception as e:
                last_exc = e
                if getattr(e, "transient", False):
                    envelope.record(target, "transient_retry",
                                     {"type": type(e).__name__})
                    continue
                raise

        # Exhausted retries
        envelope.status = "failed"
        envelope.record(target, "exhausted_retries",
                         {"max_retries": max_retries})
        if isinstance(last_exc, Exception):
            raise last_exc
        raise RuntimeError(f"delegate to {target} failed after "
                            f"{max_retries + 1} attempts")


class TransientFailure(Exception):
    """Signals a retriable failure; bus.delegate will re-queue."""
    transient = True
