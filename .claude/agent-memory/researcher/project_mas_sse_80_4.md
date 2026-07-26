---
name: project-mas-sse-80-4
description: MAS SSE facts measured for step 80.4 — bus is idle-by-design, onopen fires on headers, Starlette disconnect handling is uvicorn-spec_version-gated, the immutable grep forces es.onopen, and SIGTERM does NOT kill an established SSE stream (use SIGKILL to test death detection)
metadata:
  type: project
---

Measured 2026-07-25 while researching phase-80 step 80.4 (`/agents` shows a
permanent red "Disconnected" on a healthy SSE endpoint).

**The MAS event bus is idle BY DESIGN, not broken.** `total_events: 0` /
`buffer_size: 0` on `/api/mas/events/stats` is the correct reading of a
freshly-restarted backend. The ~11 `bus.emit()` sites all live in
`backend/agents/multi_agent_orchestrator.py` and fire only during a MAS
orchestration run (Slack-chat class). The autonomous trading cycle never
publishes. The buffer is a process-local `deque(maxlen=200)` that empties on
every restart. Do not treat a silent MAS stream as an incident before
checking whether a MAS run has happened since the last restart.

**`EventSource.onopen` fires on validated response HEADERS, before body
interpretation** (WHATWG: "announce the connection *and* interpret res's body
line by line"; announce sets `readyState = OPEN` and fires `open`). MDN never
states this — only the normative spec does. Consequence: a connection-state
indicator derived from `onopen` works against a server that streams zero
bytes, so a frontend fix does not have to wait on a backend heartbeat.
Comment lines (`: ping`) are ignored and never reach `onmessage`, so a
heartbeat can never substitute for `onopen`.

**Starlette 1.0.0 client-disconnect handling is gated on the ASGI
`spec_version`.** Installed uvicorn 0.42.0 advertises `2.3` (both `h11_impl`
and `httptools_impl`) → the `anyio` task-group + `listen_for_disconnect`
branch is live, so a generator blocked forever on `await queue.get()` is
still cancelled on disconnect and its `finally:` unsubscribes (verified:
subscribers returned to 0 after curl disconnects). If uvicorn ever advertises
`>= 2.4`, that branch disappears and a never-sending generator would stop
noticing disconnects — a periodic heartbeat becomes the only thing that
surfaces the `OSError`. Re-measure the spec_version before trusting either
branch.

**Harness trap worth remembering:** 80.4's immutable verification command
greps `useEventSource.ts` for the literal `onopen`, so the implementation
must use `es.onopen = ...`; the equivalent `addEventListener("open", ...)`
would pass review and fail the gate. Also `jsdom` 29 ships no `EventSource`
at all, so every test of this hook is necessarily mock-driven.

**SIGTERM does NOT kill an established SSE stream — never use plain `kill` /
`pkill` to test SSE death detection** (measured 2026-07-26, uvicorn 0.42.0,
throwaway :8009 rig). `Server.shutdown()` closes the LISTENING socket first,
then `await asyncio.wait_for(self._wait_tasks_to_complete(), timeout=
self.config.timeout_graceful_shutdown)` — and that default is **`None`**, i.e.
wait forever. A never-returning SSE generator is a task that never completes.
Measured: post-SIGTERM the process stayed alive (`STAT=SN`, log "Waiting for
connections to close"), a NEW `curl` got `000`, and the OPEN stream kept
receiving `: ping` (61 → 93 bytes over 10s). `kill -9` ended it instantly.
So `curl → 000` + "0 listeners" proves the listener is gone, **not** that
established connections died — it is the wrong death oracle for SSE. This
artifact cost step 80.4 a false "criterion 4 FAILS" verdict: the indicator
stayed green because the stream really was alive. Test with `kill -9` or
`--timeout-graceful-shutdown 1`. (Corroborated: Kludex/uvicorn#451 — with a
stream open the first SIGTERM "will not stop the responses".)

**Comment heartbeats can never drive client-side staleness detection, but a
named event is filtered for free.** `whatwg/html#7571` (still OPEN, no
implementer interest) exists precisely because `:`-prefixed lines are
"not currently distinguishable" by clients. Conversely, an `event: heartbeat`
frame is dispatched with `type: "heartbeat"` and MDN is explicit that a
`"message"` listener "will not trigger on any other event type" — so adding a
named heartbeat cannot inflate an event counter fed by
`addEventListener("message", …)`. No string-comparison filter is needed; DOM
event-type dispatch is the filter.

Related: [[feedback-measure-dont-assert-claims]],
[[feedback-queue-discovered-defects-in-masterplan]].
