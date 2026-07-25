"""phase-80.4 -- an idle SSE stream must be distinguishable from a dead one.

`GET /api/mas/events` returned 200 + `text/event-stream` and then, on a system
with no MAS run in flight, sent NOTHING. Measured live: the bus has published
0 events since process start (its emit sites fire only on MAS orchestration
runs, which the trading cycle never triggers), so "healthy but quiet" is the
NORMAL case here -- and it was byte-identical to "hung".

These tests drive the real `event_generator` against the real event bus.

House convention (see `backend/tests/test_autonomous_loop_step_5_6.py`): drive
async reproducers with `asyncio.run()` -- pytest-asyncio is not installed.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.agents.mas_events import MASEvent, get_event_bus
from backend.api import mas_events as api


def _make_event(**kw):
    """Build a MASEvent through whatever constructor this build exposes."""
    import inspect

    params = inspect.signature(MASEvent).parameters
    payload = {}
    for name in params:
        if name in kw:
            payload[name] = kw[name]
    return MASEvent(**payload) if payload else MASEvent(**kw)


async def _drain(gen, n, timeout=2.0):
    """Pull up to n chunks, stopping early on timeout."""
    out = []
    for _ in range(n):
        try:
            out.append(await asyncio.wait_for(gen.__anext__(), timeout))
        except (asyncio.TimeoutError, StopAsyncIteration):
            break
    return out


# ── the heartbeat itself ───────────────────────────────────────────────


def test_stream_announces_itself_immediately(monkeypatch):
    """Criterion 2, first half: the stream proves itself on CONNECT.

    Pre-fix the first byte only arrived with the first real event -- which,
    on an idle bus, never came.
    """

    async def _run():
        monkeypatch.setattr(api, "_HEARTBEAT_SECONDS", 0.05)
        resp = await api.stream_events(include_buffer=False)
        gen = resp.body_iterator
        try:
            first = await asyncio.wait_for(gen.__anext__(), 1.0)
            assert first == ": connected\n\n", repr(first)
        finally:
            await gen.aclose()

    asyncio.run(_run())


def test_idle_stream_emits_periodic_keepalive(monkeypatch):
    """Criterion 2, second half: an idle stream keeps proving itself."""

    async def _run():
        monkeypatch.setattr(api, "_HEARTBEAT_SECONDS", 0.05)
        resp = await api.stream_events(include_buffer=False)
        gen = resp.body_iterator
        try:
            chunks = await _drain(gen, 4, timeout=1.0)
            assert chunks[0] == ": connected\n\n"
            pings = [c for c in chunks[1:] if c == ": ping\n\n"]
            assert pings, f"idle stream sent no keepalive: {chunks!r}"
        finally:
            await gen.aclose()

    asyncio.run(_run())


def test_keepalive_is_a_comment_so_it_cannot_reach_onmessage(monkeypatch):
    """Every heartbeat byte must start with ':'.

    Per the WHATWG spec a comment line is ignored by EventSource and never
    surfaces to onmessage -- that is what stops the heartbeat inflating
    /agents' event counters or firing the data path.
    """

    async def _run():
        monkeypatch.setattr(api, "_HEARTBEAT_SECONDS", 0.05)
        resp = await api.stream_events(include_buffer=False)
        gen = resp.body_iterator
        try:
            for chunk in await _drain(gen, 4, timeout=1.0):
                assert chunk.startswith(":"), f"non-comment chunk on an idle bus: {chunk!r}"
                assert "data:" not in chunk
        finally:
            await gen.aclose()

    asyncio.run(_run())


# ── the trap: heartbeats must not kill the subscription ────────────────


def test_subscription_survives_many_heartbeats(monkeypatch):
    """THE REGRESSION THIS DESIGN EXISTS TO PREVENT.

    The obvious implementation -- `asyncio.wait_for(agen.__anext__(), t)` --
    CANCELS its inner awaitable on timeout, and cancelling `__anext__` throws
    CancelledError *into* `MASEventBus.subscribe`, running its `finally` and
    silently unsubscribing the client on the FIRST idle heartbeat. The stream
    would then look alive (pings keep coming) while delivering nothing ever
    again -- strictly worse than the bug being fixed.

    So: after several heartbeats the subscriber must still be registered.
    """

    async def _run():
        monkeypatch.setattr(api, "_HEARTBEAT_SECONDS", 0.05)
        bus = get_event_bus()
        before = len(bus._subscribers)

        resp = await api.stream_events(include_buffer=False)
        gen = resp.body_iterator
        try:
            chunks = await _drain(gen, 4, timeout=1.0)
            assert sum(1 for c in chunks if c == ": ping\n\n") >= 1, chunks
            assert len(bus._subscribers) == before + 1, (
                "the subscription was torn down by the heartbeat loop -- the client "
                "would receive pings forever and never an event"
            )
        finally:
            await gen.aclose()

    asyncio.run(_run())


def test_event_published_after_heartbeats_is_still_delivered(monkeypatch):
    """The subscription must be FUNCTIONAL after idling, not merely present."""

    async def _run():
        monkeypatch.setattr(api, "_HEARTBEAT_SECONDS", 0.05)
        bus = get_event_bus()

        resp = await api.stream_events(include_buffer=False)
        gen = resp.body_iterator
        try:
            await _drain(gen, 3, timeout=1.0)  # idle through several heartbeats

            ev = _make_event(
                event_type="test", agent="phase-80-4", run_id="t", message="hello",
                data={}, ticker="TEST",
            )
            for q in list(bus._subscribers):
                q.put_nowait(ev)

            got = None
            for _ in range(20):
                chunk = await asyncio.wait_for(gen.__anext__(), 1.0)
                if not chunk.startswith(":"):
                    got = chunk
                    break
            assert got is not None, "no real event arrived after idling"
            assert "data:" in got
        finally:
            await gen.aclose()

    asyncio.run(_run())


def test_generator_unsubscribes_on_client_disconnect(monkeypatch):
    """Closing the stream must not leak a subscriber or a pending task."""

    async def _run():
        monkeypatch.setattr(api, "_HEARTBEAT_SECONDS", 0.05)
        bus = get_event_bus()
        before = len(bus._subscribers)

        resp = await api.stream_events(include_buffer=False)
        gen = resp.body_iterator
        await _drain(gen, 2, timeout=1.0)
        assert len(bus._subscribers) == before + 1
        await gen.aclose()
        await asyncio.sleep(0.05)
        assert len(bus._subscribers) == before, "subscriber leaked after disconnect"

    asyncio.run(_run())


def test_heartbeat_interval_is_configured_and_sane():
    """Fixture pin bound to the subject, not a library fact."""
    assert isinstance(api._HEARTBEAT_SECONDS, (int, float))
    assert 1.0 <= api._HEARTBEAT_SECONDS <= 60.0, api._HEARTBEAT_SECONDS
