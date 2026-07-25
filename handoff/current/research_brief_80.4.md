# Research Brief — step 80.4 (SSE "Disconnected" false alarm on /agents)

Tier: **moderate**. `coverage.audit_class = false`. Date: 2026-07-25.
Researcher: Layer-3 harness researcher. Write-first, incremental.

---

## HEADLINE — the two questions that decide "one fix or two"

**A2 — `onopen` fires on RESPONSE HEADERS, not on the first body byte.**
WHATWG HTML §server-sent-events, "process the fetch response": after
validating `status == 200` and `Content-Type: text/event-stream`, the spec
says *"announce the connection **and** interpret res's body line by line."*
Announce comes **before** body interpretation, and the announce task
*"sets the `readyState` attribute to `OPEN` and fires an event named
`open`."*
⇒ **Criterion 1 and criterion 2 are genuinely independent.** The frontend
`onopen` handler alone fixes the false "Disconnected" against the CURRENT,
heartbeat-less backend. The heartbeat is a separate (still-correct) fix for
proxy idle-timeout + dead-vs-idle discrimination. This matters operationally
— see §C4 (the backend half is inert until a restart; the frontend half is
not).

**B4 — the bus is IDLE, not broken. This is one defect, not two.**
Measured live 2026-07-25 22:19 UTC against the running backend:

```
$ curl -s localhost:8000/api/mas/events/stats
{"total_events":0,"buffer_size":0,"subscribers":0}
```

`total_events: 0` since process start. Publishers exist and are correctly
wired — `backend/agents/multi_agent_orchestrator.py` calls `bus.emit(...)`
at **10 sites** (`:434, :455, :464, :495, :512, :527, :603, :622, :644,
:665, :1337`) plus the `/api/mas/events/ingest` relay at
`backend/api/mas_events.py:81`. They fire only during a **MAS orchestration
run** (Slack-chat class work), which is *not* part of the autonomous
trading cycle. The buffer is a process-local `deque` (`mas_events.py:97`,
`maxlen=200`) that empties on every backend restart.
⇒ "0 events | 1 sub" on a freshly-restarted backend is the **correct**
reading of an idle system. Nothing is broken upstream. But see §C2 — a
heartbeat *does* create a new honesty gap that the UI should close.

---

## Search queries run (3-variant discipline)

| Variant | Query |
|---|---|
| Current-year frontier | `SSE keepalive heartbeat interval best practice 2026 EventSource proxy timeout` |
| Last-2-year window | `EventSource onopen not firing until first byte browser behavior 2025 2026` |
| Year-less canonical | `FastAPI StreamingResponse SSE heartbeat asyncio queue wait_for timeout pattern` (+ direct spec/MDN fetches, no year term) |

---

## Read in full (8; gate floor is 5)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|---|
| 1 | https://html.spec.whatwg.org/multipage/server-sent-events.html | 2026-07-25 | **spec (normative)** | WebFetch, full | *"announce the connection and interpret res's body line by line"*; announce *"sets the readyState attribute to OPEN and fires an event named open"*. Comment ABNF: `comment = colon *any-char end-of-line`; *"If the line starts with a U+003A COLON character (:) … Ignore the line"* — **no message event**. Keepalive: *"Legacy proxy servers are known to, in certain cases, drop HTTP connections after a short timeout. To protect against such proxy servers, authors can include a comment line (one starting with a ':' character) **every 15 seconds or so**."* Reconnect: *"Set the readyState attribute to CONNECTING"* then *"Fire an event named error"*. Terminal: *"Once the user agent has failed the connection, it does **not** attempt to reconnect."* |
| 2 | https://developer.mozilla.org/en-US/docs/Web/API/EventSource | 2026-07-25 | official docs | WebFetch, full | `open`: *"Fired when a connection to an event source has opened."* `error`: *"Fired when a connection to an event source failed to open."* readyState *"CONNECTING (0), OPEN (1), or CLOSED (2)"*. `close()` *"Closes the connection, if any, and sets the readyState attribute to CLOSED."* `withCredentials` read-only, default `false`. |
| 3 | https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events | 2026-07-25 | official docs | WebFetch, full | *"By default, if the connection between the client and server closes, the connection is restarted."* *"A colon as the first character of a line is in essence a comment, and is ignored."* *"The comment line can be used to prevent connections from timing out; a server can send a comment periodically to keep the connection alive."* Stop reconnection: `evtSource.close()`. |
| 4 | https://developer.mozilla.org/en-US/docs/Web/API/EventSource/open_event | 2026-07-25 | official docs | WebFetch, full | *"The `open` event … is fired when a connection with an event source is opened."* Both `addEventListener("open", …)` and `evtSource.onopen = …` are documented, equivalent forms. Generic `Event`, non-bubbling, non-cancelable. |
| 5 | https://github.com/sysid/sse-starlette | 2026-07-25 | official lib docs (the de-facto FastAPI SSE reference impl) | WebFetch, full | `ping` parameter, **default 15 seconds**, *"Ping interval in seconds (0 to disable)"*. Pings are sent as **comment lines** (`:` prefix); `ping_message_factory` for custom (`ServerSentEvent(comment="…")`). Disconnect: *"Always check `await request.is_disconnected()` in loops"* to prevent hanging connections and task leaks. `AppStatus` + `shutdown_grace_period` for graceful shutdown. |
| 6 | https://www.starlette.io/responses/ | 2026-07-25 | official docs | WebFetch, full | `StreamingResponse(generator, media_type=None)`; async generators supported. **Docs are silent on client-disconnect detection** — resolved by reading the installed source instead (see §B3a). |
| 7 | https://tigerabrodi.blog/server-sent-events-a-practical-guide-for-the-real-world | 2026-07-25 | practitioner blog | WebFetch, full | *"Comment lines (starting with a colon) are ignored by the EventSource API but still keep the connection alive through proxies"*, `": heartbeat\n\n"` every **30 seconds**. `onerror` branches on `readyState`: `CONNECTING → 'Reconnecting…'`, `CLOSED → setError(...)` — i.e. onerror can fire **repeatedly** across retry attempts. `onopen` used to set status directly. |
| 8 | https://dev.to/napster_rj/what-are-server-sent-events-sse-a-developers-guide-for-2026-4jb6 | 2026-07-25 | practitioner blog (**pub. 2026-05-24** — recency-scan anchor) | WebFetch, full | *"Send a comment line every 15-30 seconds"*; `setInterval(() => res.write(':keepalive\n\n'), 15000)`. *"Lines starting with `:` are spec-compliant comments that maintain connection warmth without triggering client-side `onmessage` events."* *"When the connection drops, the browser waits ~3 seconds and reconnects. You don't write any code for this."* Infra: AWS ALB 50s idle timeout; nginx `proxy_read_timeout 24h` + `X-Accel-Buffering: no`. |

## Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://oneuptime.com/blog/post/2026-01-27-sse-different-frameworks/view | blog | Recency data point (Jan 2026); framework survey, no new mechanism |
| https://www.evlune.com/2026/04/how-to-handle-sse-connection-drops-in.html | blog | Spring WebFlux/Nginx-specific; corroborates heartbeat-merge pattern |
| https://niteagent.com/blog/2026-07-09-streaming-agent-responses-production-guide/ | blog | Jul-2026 recency data point; agent-streaming focus |
| https://stackoverflow.com/questions/19778231/ (SSE times out after 1h22m) | community | Canonical prior-art hit from year-less query; lowest tier |
| https://github.com/r3labs/sse/issues/101 | community | Go lib; keepalive-ping design discussion |
| https://www.ioriver.io/terms/sse | vendor glossary | Marketing-tier |
| https://medium.com/@bhagyarana80/fastapi-streaming-responses-real-time-without-websockets-… | blog | Medium paywall-class; pattern already covered by #5 |
| https://ranjankumar.in/building-chatgpt-style-streaming-in-react-fastapi-next-js-production-guide | blog | Same pattern, lower authority than #5 |
| https://www.w3.org/Bugs/Public/show_bug.cgi?id=14120 | spec bug tracker | Historical (2011) onopen-timing discussion; superseded by current spec text |
| https://lists.w3.org/Archives/Public/public-webapps/2014OctDec/0400.html | mailing list | Historical |
| https://developer.mozilla.org/en-US/docs/Web/API/EventSource/message_event | official docs | Covered by #2/#3 |
| https://codingtechroom.com/question/eventsource-onmessage-not-working | community | Low tier |

**URLs collected: 20 unique.**

## Recency scan (2024–2026) — PERFORMED

Searched the 2024–2026 window explicitly (two of the three query variants
were year-scoped). **Result: no new findings that supersede the canonical
WHATWG guidance.** Four independent 2026 sources (DEV 2026-05-24, oneuptime
2026-01-27, evlune 2026-04, niteagent 2026-07-09) all re-recommend the
*same* comment-line keepalive at **15–30 s**, which is the interval the
2011-era spec text already prescribes ("every 15 seconds or so"). No
EventSource API change, no `onopen`-semantics change, no new spec revision
surfaced. The one genuinely *new* material is infrastructure-side (AWS ALB
50 s idle default; nginx `proxy_read_timeout`), which is **not applicable
here** — pyfinagent is local-only (`localhost:8000` → `localhost:3000`, no
proxy in front). The heartbeat's value on this deployment is therefore
**dead-vs-idle discrimination and generator liveness**, *not* proxy
survival. Say that honestly in the contract rather than importing the
proxy rationale.

---

## §B — Internal code inventory (measured; anchors verified, NOT drifted)

| File | Lines | Role | Status |
|---|---|---|---|
| `frontend/src/lib/hooks/useEventSource.ts` | 1–164 (whole file read) | shared SSE hook | **zero occurrences of `onopen`/`"open"`** |
| `frontend/src/app/agents/page.tsx` | 194–222, 306–316 | the ONLY consumer | binary label |
| `backend/api/mas_events.py` | 22–44 | SSE endpoint | no heartbeat |
| `backend/agents/mas_events.py` | 111–140, 180–203 | bus emit + subscribe | blocks forever on `queue.get()` |
| `frontend/src/lib/hooks/useEventSource.test.ts` | 1–51 | existing tests | 2 cases, `withCredentials` only |
| `frontend/src/lib/hooks/index.ts` | 9–10 | barrel export | unchanged surface |
| `frontend/vitest.config.ts` / `vitest.setup.ts` | — | test env | jsdom 29.0.2, no EventSource |
| `.venv/…/starlette/responses.py` | `StreamingResponse` | disconnect handling | version-gated (see §B3a) |

### B1 — `useEventSource.ts`: every status transition

Four-value union at `:20`:
`status: "connecting" | "connected" | "disconnected" | "error"`.

| Line | Transition | Trigger |
|---|---|---|
| `:81` | initial `"connecting"` | `useState` |
| `:99` | `"connecting"` | top of `connect()` |
| **`:110`** | **`"connected"`** | **inside `onMessage` ONLY** ← the defect |
| `:127` | `"error"` | `es.onerror` |
| `:136` | `"disconnected"` | failure count reached `maxFailures` |
| `:142` | `"error"` | synchronous throw from `new EventSource(...)` |
| `:150` | `"disconnected"` | effect sees `!enabled \|\| !url` |

Verbatim, the defect (`:109-122`):

```ts
const onMessage = (event: MessageEvent) => {
  setStatus("connected");
  setFailures(0);
  backoffRef.current = 1000;
  setLastEventAt(Date.now());
  …
};

es.addEventListener(eventType, onMessage as EventListener);   // :124
```

`eventType` defaults to `"message"` (`:70`). Two consequences:
1. **Connection state is derived from data arrival.** An open, healthy,
   idle stream stays `"connecting"` forever → `/agents` paints red.
2. A spec-comment heartbeat (`: ping`) will **never** reach this listener
   (spec source #1: comment lines are ignored). So the heartbeat **cannot**
   substitute for `onopen`, and equally cannot accidentally pollute
   `/agents`' event list. Both halves are needed; neither breaks the other.

**`maxFailures` / reconnect (`:126-140`), verbatim:**

```ts
es.onerror = () => {
  setStatus("error");
  cleanup();                       // ← es.close(): kills the BROWSER's own retry
  setFailures((prev) => {
    const next = prev + 1;
    if (next < maxFailures) {
      const delay = Math.min(backoffRef.current, 30_000);
      backoffRef.current = Math.min(backoffRef.current * 2, 30_000);
      window.setTimeout(connect, delay);
    } else {
      setStatus("disconnected");
    }
    return next;
  });
};
```

**Load-bearing fact for criterion 4:** because `cleanup()` calls
`es.close()`, `readyState` goes to `CLOSED` and the browser's *native*
auto-reconnect is disabled. Reconnection is **entirely manual** here. So
the ambiguity in source #7 ("onerror can fire repeatedly across retries")
does **not** apply — in this hook `onerror` fires **exactly once per
connection attempt**, and `failures` increments exactly once per attempt.
With `/agents`' `maxFailures: 5` (`page.tsx:200`) and backoff 1s→2s→4s→8s,
a dead backend reaches `"disconnected"` in **~15 s**. Criterion 4 is
satisfiable and cheaply mutation-testable.

Cleanup/unmount: `cleanup` at `:88-93` closes + nulls the ref; returned as
the effect teardown at `:154`. **Not covered:** a pending
`window.setTimeout(connect, delay)` from `:134` is *never* cleared on
unmount — a late reconnect can fire after teardown. Pre-existing, out of
scope, queue it (§D).

### B2 — `/agents/page.tsx`: the derivation and the label

```ts
// :198-200
const { status: sseStatus, failures: sseFailures, reconnect: sseReconnect } =
  useEventSource<MASEvent>(sseUrl, {
    maxFailures: 5,
// :217
const connected = sseStatus === "connected";
```

```tsx
// :306-316
<span className={`… ${connected ? "text-emerald-400" : "text-rose-400"}`}>
  <span className={`h-2 w-2 rounded-full ${connected ? "bg-emerald-400 animate-pulse" : "bg-rose-400"}`} />
  {connected ? "Connected" : "Disconnected"}
</span>
{stats && (<span …>{stats.total_events} events | {stats.subscribers} sub</span>)}
```

**Answering the lead's question: YES, a third state exists and the UI
collapses it.** The hook exposes four states; `page.tsx:217` flattens them
to a boolean, so `connecting` **and** `error` both render as the red
"Disconnected". That is a second, independent honesty bug on the same
line: during the first ~50 ms of every page load, and during every
1s/2s/4s backoff window, the operator is told "Disconnected" when the
truth is "Connecting".

**Recommendation (goes beyond the letter of criterion 3, still inside its
spirit):** render three states — emerald "Connected" (`connected`), amber
"Connecting…" (`connecting` | `error`), rose "Disconnected"
(`disconnected`). This is *more* honest than forcing "Connected", and it
does not weaken criterion 4 (the rose state is still reached, and only
reached, via `maxFailures`). If Main prefers a minimal diff, the binary
label with `onopen` satisfies criteria 1/3/4 as written — but then say so
explicitly rather than leaving the collapse undisclosed.

Note `stats.subscribers` comes from the **15 s poll** of
`/api/mas/events/stats`, not from the SSE stream — so "1 sub" is real and
independent evidence the connection was live while the label said
"Disconnected". That is the cleanest existing proof of the false alarm.

### B3 — `backend/api/mas_events.py`: the endpoint, verbatim

```python
# :22-44
@router.get("/events")
async def stream_events(include_buffer: bool = Query(True)):
    bus = get_event_bus()

    async def event_generator():
        async for event in bus.subscribe(include_buffer=include_buffer):
            yield event.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )
```

And the bus side (`backend/agents/mas_events.py:180-203`):

```python
async def subscribe(self, include_buffer: bool = True):
    queue: asyncio.Queue = asyncio.Queue(maxsize=500)
    self._subscribers.append(queue)
    try:
        if include_buffer:
            for event in self._buffer:
                yield event
        while True:
            event = await queue.get()      # ← blocks forever when idle
            yield event
    finally:
        try:
            self._subscribers.remove(queue)
        except ValueError:
            pass
```

**This is exactly the "blocks forever on the bus queue" hazard the lead
asked about.** The idiomatic fix is `asyncio.wait_for` (or 3.11+
`asyncio.timeout`) around `queue.get()`, converting the timeout into a
heartbeat rather than an error:

```python
while True:
    try:
        event = await asyncio.wait_for(queue.get(), timeout=15.0)
    except (asyncio.TimeoutError, TimeoutError):
        yield None            # sentinel → endpoint yields ": ping\n\n"
        continue
    yield event
```

Why `wait_for` and **not** a racing `asyncio.sleep` task: `wait_for`
cancels the pending `get()` cleanly on timeout and creates no orphan
task; a `asyncio.wait([get_task, sleep_task], FIRST_COMPLETED)` race
leaves the un-awaited `get_task` holding a place in the queue's getter
deque unless you explicitly re-use it across iterations — the classic
source of dropped events. **Interleaving is safe and order-preserving in
either design**: heartbeats are only emitted on the *timeout* branch,
i.e. exactly when the queue was empty, so a heartbeat can never be
interleaved *between* a queued event and its delivery, and can never
reorder or drop one. (`asyncio.TimeoutError` is an alias of built-in
`TimeoutError` on 3.11+ — this repo is Python 3.14, so catching
`TimeoutError` alone is sufficient; catching both is harmless.)

The **initial** flush is separate and simpler: yield `": connected\n\n"`
(or `":ok\n\n"`) as the very first statement of `event_generator()`,
before touching the bus. It costs nothing, is spec-ignored by the client,
and makes `curl` immediately distinguishable from a hang — which is
precisely the evidence the live_check needs.

**Where to put the heartbeat.** Prefer keeping `MASEventBus.subscribe`
unchanged and doing the timeout in `event_generator()` (a second
subscriber-side loop reading `bus`'s queue is not exposed). The minimal
shape that avoids touching the bus contract is to add an optional
`heartbeat_seconds` parameter to `subscribe()` that yields a `None`
sentinel, and have `event_generator()` translate `None → ": ping\n\n"`.
Either is fine; state the choice in the contract.

Also note `MASEvent.to_sse()` (`mas_events.py:66-68`) already emits a
correct `data: …\n\n` frame — the heartbeat must be a *sibling* string,
not routed through `to_sse()`.

### B3a — client disconnect + task leak: MEASURED, no leak today

Starlette **1.0.0** (installed) branches on the ASGI `spec_version`:

```python
spec_version = tuple(map(int, scope.get("asgi", {}).get("spec_version", "2.0").split(".")))
if spec_version >= (2, 4):
    try: await self.stream_response(send)
    except OSError: raise ClientDisconnect()
else:
    with collapse_excgroups():
        async with anyio.create_task_group() as task_group:
            async def wrap(func): await func(); task_group.cancel_scope.cancel()
            task_group.start_soon(wrap, partial(self.stream_response, send))
            await wrap(partial(self.listen_for_disconnect, receive))
```

Installed **uvicorn 0.42.0** advertises `spec_version: "2.3"` in *both*
`h11_impl` and `httptools_impl` → **the `else` branch is live**, so
`listen_for_disconnect` runs concurrently and cancels the generator on
`http.disconnect`. The `finally:` in `subscribe()` therefore runs and the
subscriber is removed.

Verified empirically — after two `curl` SSE connections were opened and
dropped, the bus reported:

```
{"total_events":0,"buffer_size":0,"subscribers":0}
```

⇒ **no subscriber leak, no task leak today, and the heartbeat will not
introduce one.** (Forward-looking caveat: if uvicorn ever advertises
`spec_version >= 2.4`, the `else` branch disappears and a never-sending
generator would *stop* noticing disconnects — at which point the heartbeat
becomes the *only* thing that surfaces `OSError`. The heartbeat is thus
also a hedge against that upgrade. `sse-starlette`'s
`await request.is_disconnected()` advice (#5) is the belt-and-braces
option; not required today.)

### B4 — the MAS event bus: who publishes, how often (MEASURED)

Publishers (grep over `backend/`, `scripts/`):

| Site | Trigger |
|---|---|
| `backend/agents/multi_agent_orchestrator.py:434,455,464,495,512,527,603,622,644,665,1337` | a MAS orchestration run (classify → plan → delegate → … → complete) |
| `backend/api/mas_events.py:81` (`POST /api/mas/events/ingest`) | relay from the Slack-bot process |
| `backend/slack_bot/app_home.py:41,455,543,559` | reads the bus (`get_event_bus()`), does not emit |
| `backend/tests/test_phase_75_event_loop.py:83` | test only |

Live measurement (2026-07-25 22:19 UTC, backend healthy, `/api/health`
200): `total_events: 0`, `buffer_size: 0`, `subscribers: 0`;
`curl --max-time 6` on the SSE stream returned **0 bytes** with correct
headers (`content-type: text/event-stream; charset=utf-8`,
`x-accel-buffering: no`, `transfer-encoding: chunked`) — the lead's
observation reproduced exactly.

**Verdict: idle, not broken.** The autonomous trading cycle does not run
the MAS orchestrator, so on a backend that has not served a Slack MAS
query since its last restart, `total_events` is legitimately 0. **This
step is not papering over a dead bus.** (Confirmable at any time by the
operator: run one MAS query, then re-poll `/api/mas/events/stats` and
watch `total_events` rise. Worth including in the live_check as the
positive control — it also exercises criterion 3's "0 events" claim by
contrast.)

### B5 — `useEventSource.test.ts` and what jsdom can do

Existing file: 51 lines, one `describe`, two `it`s, both about
`withCredentials` (phase-75.12). It stubs a hand-rolled `MockEventSource`
(`:11-27`) via `vi.stubGlobal` and asserts the constructor argument.

**Measured: jsdom 29.0.2 does NOT implement `EventSource`** —
`typeof window.EventSource === "undefined"`. There is no shim in
`vitest.setup.ts` (which only shims `ResizeObserver`). So *every*
EventSource test here is necessarily mock-driven; that is a constraint,
not a shortcut. A mock CAN meaningfully exercise `onopen`/`onerror`
because the hook assigns/registers real handlers on the instance — the
test just drives them itself.

The existing mock is **missing an `onopen` field** (`:16` declares only
`onerror`). Two implementation shapes, and they are **not
interchangeable**:

- `es.onopen = () => setStatus("connected")` → the mock needs
  `onopen: (() => void) | null = null`, and the test calls
  `instances[0].onopen?.()`.
- `es.addEventListener("open", …)` → the existing `addEventListener`
  already captures it into `listeners["open"]`; the test calls
  `listeners.open[0](new Event("open"))`.

**⚠ The immutable verification command greps `grep -n 'onopen'
src/lib/hooks/useEventSource.ts`.** The `addEventListener("open", …)`
form would contain no literal `onopen` and would **fail the immutable
command**. Use the `es.onopen = …` property form (MDN #4 documents both as
equivalent). Flagging because this is exactly the kind of
criterion/implementation coupling that costs a cycle.

Criterion 5's new case, minimally: stub the mock, render the hook, assert
`result.current.status === "connecting"`, fire `onopen()`, assert
`"connected"` — **with zero message events delivered**. That is the
open-but-no-events state, and it is a real discriminating guard: revert
the `onopen` line and it fails. Mutation-test it exactly that way
(auto-memory: a guard that can't fail doesn't count).

### B6 — other consumers of `useEventSource`

`grep -rn "useEventSource" frontend/src` →
`frontend/src/app/agents/page.tsx` (the only call site, `:199`),
`frontend/src/lib/hooks/index.ts:9-10` (barrel re-export of the function
and `UseEventSourceState`), and the test. **`/agents` is the sole
consumer**, so a status-semantics change has exactly one blast radius.
The `UseEventSourceState` type is exported publicly, so *widening*
behaviour (adding `onopen`) is source-compatible; **do not** change the
`status` union's member set or the `lastEventAt` semantics — that would be
a consumer-contract break for any future importer.

Design note on `lastEventAt`: it should stay tied to **real events only**.
If a heartbeat ever updated it, the field would stop meaning "when did the
MAS last do something", which is the field's only use.

---

## §C — Risk / do-no-harm

**C1 — criterion 4 is the trap, and here is the exact trip-wire.**
A fully-dead backend is safe: connection is refused, `onopen` never fires,
`onerror` fires, `failures` climbs to 5, label → "Disconnected" in ~15 s.
The dangerous case is a **half-dead / flapping** backend (accepts TCP,
returns 200 + `text/event-stream` headers, then dies — e.g. a proxy that
accepts and holds, or uvicorn mid-reload). There, `onopen` **does** fire on
a dead-in-practice connection.

> **Therefore: `onopen` must set `status` and NOTHING else. Do NOT
> `setFailures(0)` and do NOT reset `backoffRef.current` inside `onopen`.**

If `onopen` reset the failure counter, every reconnect attempt would zero
the budget and the indicator would be **permanently green on a flapping
backend** — the always-green regression criterion 4 exists to prevent.
Leaving the reset in `onMessage` (real data) preserves the budget:
connected → error → connect → connected → error … still increments
`failures` each cycle and still reaches "Disconnected" at 5. Mutation-test
both directions: (a) delete `onopen` → criterion 3 must fail; (b) add
`setFailures(0)` to `onopen` and simulate open/error flapping → the
indicator must still reach "Disconnected", and if it doesn't, that is the
proof the reset is forbidden.

Residual, accept-and-disclose: with the reset confined to `onMessage`, a
long-lived session on a genuinely idle bus accumulates blips forever and
could show "Disconnected" after 5 *lifetime* transient failures despite
being connected. On a localhost-only deployment this is negligible. If
Main wants to close it, the correct mechanism is **not** resetting on
`onopen` — it is a named `event: ping` frame that the hook subscribes to
separately (`addEventListener("ping", …)`) so real liveness, not mere
connection establishment, clears the counter. That is strictly more work
and it changes the backend frame from a spec comment to a named event
(which stays invisible to `/agents`, since its listener is on `"message"`).
**Recommendation: ship the comment heartbeat + `onopen`-sets-status-only
now; queue the named-ping liveness counter as a follow-up step if the
false-red ever recurs.**

**C2 — a heartbeat CAN mask a dead bus. Close the gap in the UI, not the
protocol.** After this fix, "Connected" means "the HTTP stream is alive",
which is a strictly weaker claim than "MAS observability is working".
An operator must still be able to tell that nothing is publishing. Good
news: the page **already** renders `{stats.total_events} events |
{stats.subscribers} sub` right beside the indicator (`page.tsx:311-315`),
sourced from a *separate* 15 s poll. So "Connected · 0 events | 1 sub" is
already an honest, complete readout — the two facts are orthogonal and both
visible. **Do not remove or fold that counter.** A minimal strengthening
(optional): render "Connected · idle" when `lastEventAt === null` and
"Connected · live" once an event has arrived — one ternary, no new state,
and it makes the distinction explicit rather than inferred. Explicitly
**out of scope**: do not add a "bus health" alarm here; if the operator
wants "MAS hasn't run in N days" that is a different step.

**C3 — trading blast radius: none.** `/api/mas/events*` is pure
observability. `MASEventBus` is a standalone singleton
(`backend/agents/mas_events.py:223-231`) touched only by the MAS
orchestrator, the Slack App Home, and this router; no import path reaches
`paper_trader`, `portfolio_manager`, `screener`, or the autonomous loop.
`emit()` is non-blocking (`put_nowait`, `:124`) and already drops on
`QueueFull` rather than back-pressuring producers. A heartbeat adds one
timer wake-up per connected browser tab per 15 s — with `subscribers: 0-1`
in practice, that is unmeasurable. It cannot hold the event loop
(`asyncio.wait_for` yields), and per §B3a it cannot leak a task.
One genuine caution: `emit()` is **sync** and calls `asyncio.Queue.
put_nowait` (`:124`), which is *not* thread-safe if a producer ever emits
from a non-event-loop thread. Pre-existing, unrelated to this step, and it
does not interact with the heartbeat (which only ever runs inside the
loop). Do not "fix" it here; queue it (§D).

**C4 — split-deploy reality: which criteria the restart blocks.**
The backend runs **without `--reload`** (measured: `uvicorn backend.main:app
--host 0.0.0.0 --port 8000`, no reload flag, PID 70791, started 11:39). So:

| Criterion | Half | In effect when? |
|---|---|---|
| 1 (`onopen` registered) | frontend | next `npm run build` / dev rebuild — **no restart needed** |
| 2 (initial comment + keepalive) | **backend** | **INERT until the operator restarts uvicorn** |
| 3 (`/agents` shows Connected, 0 events) | frontend | **satisfiable TODAY against the current heartbeat-less backend** — this is the whole point of A2 |
| 4 (kill backend → Disconnected) | frontend | testable today |
| 5 (tests pass + new case) | frontend | testable today |

`.claude/masterplan.json` step **79.55 is a pending RESTART BLOCKER**
("answer BEFORE the next backend restart"). ⇒ **Do not restart the backend
to satisfy this step.** The honest live_check is: capture criteria 1/3/4/5
now against the running backend, and record criterion 2 as *code-shipped,
runtime-verification deferred to the next restart* — with the `curl`
heartbeat-bytes capture explicitly marked pending. Alternatively bundle the
restart with 79.55's resolution and capture both at once. Either way, **say
which it is in the live_check**; do not present a pre-restart curl as proof
of criterion 2, and do not silently trigger a restart that jumps 79.55.

---

## §D — Out-of-scope defects found (queue as their own masterplan steps)

Per the standing rule (any out-of-scope defect gets its own research-gated
step; written for an executor with no memory of the discovery):

1. **`useEventSource.ts:134` — reconnect timer is never cleared on
   unmount.** `window.setTimeout(connect, delay)` inside `es.onerror`
   returns a handle that is discarded; `cleanup()` (`:88-93`) only closes
   the EventSource. Navigating away from `/agents` during a backoff window
   leaves a pending timer that calls `connect()` after teardown. Fix: hold
   the handle in a ref and `clearTimeout` it in `cleanup`.
2. **`useEventSource.ts:129-139` — side effects inside a `setState`
   updater.** `setFailures((prev) => { … window.setTimeout(...);
   setStatus(...) … })` performs scheduling and a second state update
   inside the reducer. React does not guarantee updater purity; under
   StrictMode double-invocation this schedules two reconnects.
   (Measured mitigating fact: **StrictMode is not enabled** — no
   `reactStrictMode` in `next.config.*` and no `<StrictMode>` in `src/`.
   So this is latent, not live. Still wrong.)
3. **`backend/agents/mas_events.py:124` — `asyncio.Queue.put_nowait` called
   from a sync `emit()`.** Safe only while every producer runs on the event
   loop. A producer on a worker thread would corrupt the queue's waiter
   bookkeeping. Also `:122` shadows the stdlib `queue` module imported at
   `:38` with the loop variable name `queue`.
4. **`useEventSource.ts:69` — `options?.parser` is a raw `useCallback`
   dependency (`:145`).** A consumer passing an inline arrow `parser`
   would get a new `connect` identity every render → the effect at
   `:147-155` tears down and reopens the SSE connection on every render.
   `/agents` does not pass `parser`, so it is latent; ref it like
   `onEvent` already is (`:75-78`).

---

## §E — Consensus vs debate (external)

**Consensus (unanimous across #1, #3, #5, #7, #8):** the keepalive is a
comment line beginning with `:`; it is ignored by `EventSource` and never
reaches `onmessage`; it exists to defeat idle-connection timeouts.

**Debate — only on the interval.** Spec #1 and reference impl #5 both say
**15 s**; #7 says 30 s; #8 says "15–30 s". No source argues for anything
outside that band. **Recommendation: 15 s**, matching both the normative
text and `sse-starlette`'s default — the tightest of the recommendations,
and the cost on localhost is nil.

**Under-documented (a real gap, worth naming):** MDN (#2, #4) never states
*when* `open` fires relative to the response body. Only the WHATWG spec
(#1) settles it. A search specifically targeting the "does `onopen` need a
first byte?" question (last-2-year variant) returned only historical
(2011–2014) W3C bug/mailing-list threads and low-tier community pages —
i.e. **the question is answerable only from the normative text**, which is
why this brief leans on the spec quote rather than on any blog. One MDN
snippet from that search adds a *handler-registration* caveat (if you
attach `onopen` long after construction the event may already have fired);
it does **not** apply here — `useEventSource` assigns handlers
synchronously in the same tick as `new EventSource(...)` (`:103-126`), and
DOM events are queued as tasks, so the handler cannot miss `open`.

---

## §F — Application to pyfinagent (external → internal anchors)

| Finding | Source | Anchor + action |
|---|---|---|
| `open` fires on validated headers, before body interpretation | #1 | `useEventSource.ts:126` — add `es.onopen = () => setStatus("connected");` immediately before the `onerror` assignment. Property form (not `addEventListener`) so the immutable `grep -n 'onopen'` matches. |
| Comment lines never reach `onmessage` | #1, #3, #8 | Heartbeat cannot fix the label by itself, and cannot pollute `/agents`' event list (`page.tsx:201-215`). Both halves required; neither interferes. |
| Comment keepalive every ~15 s | #1, #5 | `mas_events.py:32-34` — initial `": connected\n\n"` then `": ping\n\n"` per 15 s idle timeout. |
| `asyncio.wait_for` around the queue read | #5, year-less search | `backend/agents/mas_events.py:196` `await queue.get()` → timeout-wrapped; timeout branch emits the heartbeat, so ordering is preserved and no event is dropped. |
| Disconnect must terminate the generator | #5 | Already satisfied via Starlette's `listen_for_disconnect` on uvicorn 0.42.0 / spec_version 2.3 — **measured**, subscribers returned to 0. Heartbeat is an added hedge if that branch ever changes. |
| `close()` disables native retry | #2 | `useEventSource.ts:128` `cleanup()` — this is why `onerror` fires once per attempt and `maxFailures` is a clean, testable budget (criterion 4). |
| Proxy rationale does not apply locally | recency scan | State the real rationale in the contract: dead-vs-idle discrimination + generator liveness, not proxy survival. |

---

## Research Gate Checklist

Hard blockers:
- [x] ≥5 authoritative external sources READ IN FULL via WebFetch — **8**
- [x] 10+ unique URLs total (incl. snippet-only) — **20**
- [x] Recency scan (2024–2026) performed + reported (incl. the empty-result honesty)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (hook, page, endpoint, bus, tests, barrel, vitest config, installed Starlette/uvicorn)
- [x] Contradictions / consensus noted (§E — 15 s vs 30 s; MDN's silence on `open` timing)
- [x] All claims cited per-claim
- [x] Live measurements taken rather than inferred (stats, SSE byte count, headers, jsdom `EventSource`, uvicorn spec_version, uvicorn reload flag)

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 12,
  "urls_collected": 20,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "onopen fires on validated response HEADERS, before body interpretation (WHATWG: 'announce the connection and interpret res's body line by line'; announce sets readyState OPEN and fires open). So criterion 1 is independent of criterion 2 and fixes the false Disconnected against today's heartbeat-less backend. Comment lines (':' prefix) are ignored by EventSource and never reach onmessage, so the heartbeat can neither fix the label alone nor pollute /agents. Bus MEASURED idle-not-broken: total_events 0, buffer 0, subscribers 0; 11 emit sites in multi_agent_orchestrator.py fire only on MAS runs, which the trading cycle never triggers. No task/subscriber leak today (uvicorn 0.42.0 spec_version 2.3 -> Starlette 1.0.0 listen_for_disconnect branch; subscribers returned to 0 after curl disconnects). CRITICAL TRAP: onopen must set status ONLY -- resetting failures/backoff there makes a flapping backend permanently green and breaks criterion 4. Use es.onopen = ... (property form): addEventListener('open') would fail the immutable grep. jsdom 29 has no EventSource, so criterion 5's case must extend the existing MockEventSource with an onopen field. Backend half is INERT until restart and 79.55 is a pending RESTART BLOCKER -- criteria 1/3/4/5 are verifiable today, criterion 2's runtime evidence is not.",
  "brief_path": "handoff/current/research_brief_80.4.md",
  "gate_passed": true
}
```
