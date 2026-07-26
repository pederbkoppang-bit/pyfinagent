# Research Brief 80.4b -- SSE death detection (follow-up)

Tier: **moderate**. Caller: Main (phase-80 step 80.4, criterion 4 reported FAILED).
Written 2026-07-26. Follow-up to `handoff/current/research_brief_80.4.md` (main gate
already PASSED -- not redone here).

> ## HEADLINE
>
> **Criterion 4 has not been shown to fail. It was not tested.** The `:8001` backend
> was stopped with a default `kill`/`pkill` (SIGTERM), and uvicorn's graceful shutdown
> **closes the listening socket while keeping the established SSE stream open and still
> pinging, forever** (`timeout_graceful_shutdown` default is `None`). Reproduced from
> scratch below. So: `curl` -> `000` and the stats poll failing are both correct
> (those are NEW connections, refused), and the MAS indicator staying green is ALSO
> correct -- **the stream really was alive**. There was no disconnection for `onerror`
> to report, and no retry for Playwright to log.
>
> `onerror` **does** fire on a genuine drop (WHATWG "reestablish the connection" fires
> `error` before every retry), and the hook's budget path is intact end-to-end.
> **Recommended action: re-measure with `kill -9`. Zero code change.**

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://html.spec.whatwg.org/multipage/server-sent-events.html | 2026-07-26 | spec (normative) | WebFetch | "Set the `readyState` attribute to `CONNECTING`. **Fire an event named `error`**" -- error fires on EVERY reestablish. Comment line: "If the line starts with a U+003A COLON character (:) **Ignore the line.**" |
| https://developer.mozilla.org/en-US/docs/Web/API/EventSource | 2026-07-26 | official doc | WebFetch | "The event `message` is a special case, as it will capture events without an event field as well as events that have the specific type `event: message`. **It will not trigger on any other event type.**" |
| https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events | 2026-07-26 | official doc | WebFetch | "A colon as the first character of a line is in essence a comment, and **is ignored**." / "By default, if the connection between the client and server closes, the connection is restarted." |
| https://github.com/whatwg/html/issues/7571 | 2026-07-26 | spec issue (OPEN) `[ADVERSARIAL]` | WebFetch | Comment keepalives are "**not currently distinguishable** by SSE clients ... clients cannot determine if the connection remains alive during silence" -- the spec itself concedes a comment heartbeat cannot drive client liveness. Still open, "needs implementer interest". |
| https://javascript.info/server-sent-events | 2026-07-26 | tutorial (canonical) | WebFetch | "During reconnection attempts, `readyState` equals `EventSource.CONNECTING (=0)`"; distinguishing fatal-vs-retry is done by reading `readyState` inside `onerror`. |
| https://github.com/Kludex/uvicorn/issues/451 | 2026-07-26 | vendor issue | WebFetch | With a streaming response open, the **first** SIGINT/SIGTERM "will **not stop the responses**"; the second does. Corroborates the root cause. |
| https://oneuptime.com/blog/post/2026-01-15-server-sent-events-sse-react/view | 2026-07-26 | engineering blog (2026-01) | WebFetch | Uses `res.write(": heartbeat\n\n")` -- "**Comment line (ignored by EventSource)**". Warns: "`onerror` fires during reconnection attempts, not permanent failure. Only `readyState === CLOSED` indicates actual connection loss." |
| https://thebackenddevelopers.substack.com/p/server-sent-events-in-2026-streaming | 2026-07-26 | engineering blog (2026) | WebFetch | "Design ... to assume that connections will break. Because they will. Not if. Will." Heartbeats "often enough to keep connections alive, but not so often that they become noise". |
| https://timetobuildbob.com/blog/stale-event-sse-reconnect/ | 2026-07-26 | engineering blog (2025/26) | WebFetch | Timer hygiene: "The fix is to store timer handles and clear them explicitly on close" -- reconnect timers that fire after close "open several concurrent streams". Generation-counter pattern for stale-event suppression. Notably **does not** recommend a watchdog. |
| https://react.dev/reference/react/useState | 2026-07-26 | official doc | WebFetch | "In Strict Mode, React will **call your updater function twice** ... If your updater function is pure (as it should be), this should not affect the behavior." + "It **must be pure**". |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://issues.chromium.org/issues/41003167 | browser bug (sleep-mode: request cancelled, close never fired) | Fetch returns the Google sign-in wall; only the search snippet is quotable |
| https://github.com/Yaffle/EventSource/issues/88 | polyfill issue "eventsource not always reconnecting" | corroborating only |
| https://github.com/SignalR/SignalR/issues/779 | "Chrome+SSE stops reconnecting if server loses network connectivity" | corroborating only |
| https://developer.mozilla.org/en-US/docs/Web/API/EventSource/readyState | doc | covered by the two MDN pages read in full |
| https://developer.mozilla.org/docs/Web/API/EventSource/close | doc | covered |
| https://websocket.org/guides/heartbeat/ | practitioner | "silent disconnect ... the connection object reports as open but nothing can pass through it"; "3 missed heartbeats is a reasonable default" |
| https://www.rabbitmq.com/docs/heartbeats | vendor doc | cross-domain heartbeat-threshold prior art |
| https://supabase.com/docs/guides/troubleshooting/realtime-heartbeat-messages | vendor doc | cross-domain |
| https://www.speakeasy.com/openapi/content/server-sent-events/ | practitioner | SSE heartbeat guidance |
| https://github.com/sysid/sse-starlette/issues/167 | library issue | "no easy way for the SSE content generator to know about the impending shutdown" |
| https://github.com/Kludex/uvicorn/discussions/2257 | vendor discussion | graceful-shutdown semantics in k8s |
| https://github.com/Kludex/uvicorn/pull/853 | vendor PR | SIGTERM graceful shutdown w/ multiple workers |
| https://markaicode.com/errors/uvicorn-timeout-error-fix-production/ | blog | `--timeout-graceful-shutdown` tuning |
| https://reactuse.com/browser/useeventsource/ | library doc | prior-art hook shape |
| https://github.com/nklswbr/react-eventsource + https://www.npmjs.com/package/react-eventsource | library | prior-art hook shape |
| https://www.w3.org/TR/2011/WD-eventsource-20110310/ | superseded spec | superseded by WHATWG living standard |
| https://en.wikipedia.org/wiki/Server-sent_events | encyclopedia | lowest tier |
| https://www.uvicorn.org/settings/ + https://www.uvicorn.org/server-behavior/ | official doc | **DNS failed twice** (`getaddrinfo ENOTFOUND`) from this sandbox -- replaced with a stronger primary source: the installed uvicorn source itself (see Key finding #6) |

**Queries run (3-variant discipline):** current-year (`... 2026 best practice`), last-2-year
(`... 2025 SSE reliability laptop sleep ...`), and four year-less canonical searches
(WHATWG reestablish; `onerror` not firing; keepalive-comment-vs-heartbeat-event; React hook
readyState polling; uvicorn graceful shutdown).

## Recency scan (2024-2026)

Performed. **Two relevant new findings; nothing supersedes the spec.**

1. The 2026 practitioner sources ([oneuptime 2026-01-15](https://oneuptime.com/blog/post/2026-01-15-server-sent-events-sse-react/view),
   [thebackenddevelopers 2026](https://thebackenddevelopers.substack.com/p/server-sent-events-in-2026-streaming))
   still teach the **comment** keepalive (`res.write(": heartbeat\n\n")`) and explicitly
   annotate it "ignored by EventSource". So the phase-80.4 backend implementation is the
   current idiom, not a mistake -- its purpose is transport-keepalive and server-side
   liveness, never client-side staleness detection.
2. `whatwg/html#7571` (open, no implementer interest as of this scan) is the live attempt
   to make keepalives client-observable via a standardized `:keepalive` token. **It is
   still open**, which means: as of 2026 there is *no* standard way for JS to see a
   comment heartbeat. This is the authoritative answer to question A3.

No 2024-2026 change to the `error`-event / reconnection algorithm was found. The WHATWG
text quoted above is current.

---

## Key findings

1. **`onerror` DOES fire on a genuine drop -- once per reconnection attempt.** WHATWG,
   *reestablish the connection*: "If the `readyState` attribute is set to `CLOSED`, abort
   the task. Set the `readyState` attribute to `CONNECTING`. **Fire an event named
   `error`**." So on server death the sequence is: EOF -> `readyState = CONNECTING` ->
   `error` -> wait -> retry -> (refused) -> `error` -> ... The failure budget in
   `useEventSource.ts:157-167` therefore **does** get exercised, one increment per drop.
   The distinct *fail the connection* path (`readyState = CLOSED`, fire `error`, "**does
   _not_ attempt to reconnect**") applies to non-2xx / wrong-content-type responses.
   (Source: WHATWG, accessed 2026-07-26.)

2. **`readyState` inside `onerror` is the documented discriminator.** `CONNECTING (0)` =
   the UA will retry; `CLOSED (2)` = terminal. (javascript.info; oneuptime 2026.) The hook
   currently reads neither -- it treats every `error` identically and runs its own retry.
   That is a defensible design (it owns backoff), but it means a **terminal** failure
   (e.g. a 500 or a `text/html` response) is retried 5 times anyway. Not a criterion-4
   issue; noted.

3. **Comment lines are invisible to JS -- confirmed normatively, and the gap is a known
   OPEN spec issue.** WHATWG stream interpretation: "If the line starts with a U+003A
   COLON character (:) **Ignore the line.**" No event of any type is dispatched -- not
   `message`, not a named type. MDN: "A colon as the first character of a line is in
   essence a comment, and is ignored." `whatwg/html#7571` states the consequence
   explicitly: comment keepalives are "not currently distinguishable" by clients, so
   "clients cannot determine if the connection remains alive during silence".
   **The caller's hypothesis on this point is CORRECT: a `: ping` heartbeat cannot drive
   client-side staleness detection.** (What it *is* good for -- and why it should stay --
   is proving liveness to `curl`/operators and defeating intermediary idle timeouts.)

4. **A named event does NOT reach a `"message"` listener -- so the `0 events` readout is
   safe BY SPEC, with no string comparison needed.** MDN, verbatim: "The event `message`
   is a special case, as it will capture events without an event field as well as events
   that have the specific type `event: message`. **It will not trigger on any other event
   type.**" WHATWG: the dispatched event's `type` is set to the event-type buffer. The
   hook registers exactly one listener, `addEventListener(eventType, onMessage)` with
   `eventType` defaulting to `"message"` (`useEventSource.ts:70,124`), and `/agents`
   counts only what arrives through `onEvent` (`agents/page.tsx:201-215`). So an
   `event: heartbeat` frame is **filtered by DOM event-type dispatch itself** -- it never
   reaches `onMessage`, never touches `data`/`lastEventAt`/`onEvent`, never increments the
   counter. The answer to "how is that filtering done without a fragile string
   comparison?" is: *you do not write a filter at all*; you register a second, separate
   `addEventListener("heartbeat", ...)` that only stamps a ref. The single guard needed is
   that a consumer must not configure `eventType: "heartbeat"`.

5. **The industry pattern for silent-drop detection is a client watchdog with a
   missed-beat threshold -- and it requires a REAL event, not a comment.** websocket.org:
   "a connection that's dead but nobody knows it -- the TCP stack hasn't detected the
   failure ... the connection object reports as open ... The only reliable detection is
   application-level"; "3 missed heartbeats is a reasonable default". RabbitMQ and
   Supabase use the same threshold idiom cross-domain. Chromium
   [41003167](https://issues.chromium.org/issues/41003167) (sleep/wake: request cancelled,
   no event fired) is the browser-side instance of this class.

6. **ROOT CAUSE of the §D measurement (primary source: the installed uvicorn).**
   `uvicorn 0.42.0`, `Config.__init__` -> `timeout_graceful_shutdown` **default `None`**,
   and `Server.shutdown()`:
   ```python
   logger.info("Shutting down")
   # Stop accepting new connections.
   for server in self.servers: server.close()
   ...
   await asyncio.wait_for(self._wait_tasks_to_complete(),
                          timeout=self.config.timeout_graceful_shutdown)   # None = forever
   ```
   `timeout=None` on `asyncio.wait_for` means **wait indefinitely**. An SSE generator that
   never returns is a task that never completes. Corroborated externally by
   [uvicorn#451](https://github.com/Kludex/uvicorn/issues/451) ("the first
   SIGINT/SIGTERM **will not stop the responses**").

---

## B1 -- ROOT CAUSE: the backend was NOT dead. Measured, not inferred.

**The criterion-4 measurement is an artifact of `SIGTERM` + uvicorn graceful shutdown.
The SSE connection was still alive and still pinging when the screenshot was taken, so
green "Connected" was the CORRECT reading, and the absence of retry attempts was correct
too.**

Reproduced on a throwaway port (`:8009`) with a minimal FastAPI app whose generator is
shaped like `backend/api/mas_events.py:40-77` (`yield ": connected\n\n"`, then
`": ping\n\n"` on a timer). Same venv, same uvicorn 0.42.0, same Python 3.14. Probe
source: `/private/tmp/claude-501/-Users-ford--openclaw-workspace-pyfinagent/df87839b-b9ab-4177-abf5-a397a5e2dc58/scratchpad/sse_kill_probe.py`

```
$ kill -TERM <uvicorn pid>          # what plain `kill` / `pkill` send BY DEFAULT
uvicorn log:
  INFO:     Shutting down
  INFO:     Waiting for connections to close. (CTRL+C to force quit)

  server process 69997      -> STILL ALIVE  (ps STAT=SN)
  NEW connection curl       -> code=000       <-- "0 listeners"; looks dead
  EXISTING SSE stream       -> STILL RECEIVING ": ping"
                               61 bytes -> 93 bytes over the following 10s

$ kill -9 <uvicorn pid>             # real death
  server process            -> dead
  EXISTING SSE stream       -> curl exited immediately (EOF delivered)
  byte count frozen at 109
```

Every §D symptom maps 1:1:

| Observed in `live_check_80.4.md` §D | Explained by |
|---|---|
| `curl` -> `000`, "0 listeners" | uvicorn closed the LISTENING socket on SIGTERM |
| stats poll -> "stopped after 5 consecutive failures" | those are NEW connections -- refused |
| MAS indicator stayed green "Connected" | the ESTABLISHED stream was never closed -- **correct** |
| Playwright network log: **no** `/api/mas/events` retries | nothing to retry; the connection never dropped |
| 70s with no state change | `timeout_graceful_shutdown` default `None` = wait forever |

`--lifespan off` (used by the `:8001` rig, §A) does not change this -- lifespan governs
startup/shutdown *events*, not connection draining. If anything it made the hang more
likely to go unnoticed, since the "waiting for connections to close" line is the only
signal and it goes to the rig's own log.

**Consequence:** criterion 4 has not failed; it has not been exercised. The retest is
`kill -9` (or start the rig with `--timeout-graceful-shutdown 1`), not a code change.

### B1b -- what the code does when the server DOES die

Trace of `frontend/src/lib/hooks/useEventSource.ts` on a genuine drop while OPEN:

1. Browser sees EOF/FIN. Per WHATWG this is *reestablish the connection*: set
   `readyState = CONNECTING`, **fire `error`**. So `onerror` fires -- once per drop.
2. `:155` `setStatus("error")` -> `/agents:217` `connected` goes false -> `:309` renders
   **Disconnected**. **The indicator leaves green on the FIRST error**, long before the
   budget is spent. Criterion 4's "within the existing maxFailures budget" is satisfied
   with ~30s of headroom to spare.
3. `:156` `cleanup()` -> `es.close()` -> `readyState = CLOSED`. This deliberately
   pre-empts the browser's own retry (the spec's reestablish algorithm re-checks
   `readyState` after the delay and returns if it is no longer `CONNECTING`). **Answering
   the caller's suspicion directly: yes, `close()` suppresses further events -- but only
   on that now-dead object, which is exactly the intent.** Each retry constructs a NEW
   `EventSource` at `:103` with its own `onerror`, so no failure is lost.
4. `:157-167` `setFailures(prev => ...)` is reached unconditionally -- plain synchronous
   statements, no early return, no throw between `:155` and `:157`. `cleanup()` cannot
   prevent it.
5. Under the cap: `window.setTimeout(connect, delay)` with 1/2/4/8/16s backoff. At the
   cap: `setStatus("disconnected")` and `/agents:218-221` raises the banner + Retry
   button.

**There is no bug on the `onerror` path** of the kind option 3 assumed.

**The one genuinely uncovered case** is narrower: a *silently* blackholed TCP connection
(sleep/wake, Wi-Fi drop, NAT rebind, an intermediary holding the socket). No FIN -> no
`error` -> the hook has no timer to notice. Chromium 41003167 and the websocket.org
"silent disconnect" write-up are this class. It is real, it is worth a step of its own,
and **it is not what §D measured**.

### B1c -- two real defects found in passing (neither implicated in §D)

- **`useEventSource.ts:162` -- reconnect timer handle is never stored and never cleared.**
  `window.setTimeout(connect, delay)` returns an id that is discarded; the unmount path
  (`:182 return cleanup`) closes the EventSource but cannot cancel a pending reconnect. A
  timer that fires post-unmount calls `connect()`, which builds a NEW EventSource on a
  dead component -- a leaked stream plus `setState`-after-unmount. Exactly the failure
  [timetobuildbob](https://timetobuildbob.com/blog/stale-event-sse-reconnect/) describes:
  "store timer handles and clear them explicitly on close ... otherwise you open several
  concurrent streams". Carried over from the first 80.4 brief; **confirmed NOT implicated
  in §D** (no timer ever fired there).
- **`useEventSource.ts:157-167` -- the `setFailures` updater is impure.** It calls
  `window.setTimeout(...)` and `setStatus(...)` from inside the updater. react.dev: an
  updater "**must be pure**" and "In Strict Mode, React will **call your updater function
  twice**". Next 15 defaults `reactStrictMode: true` and `next.config.js` does not
  override it, so in dev **every error schedules two reconnects**, corrupting the backoff
  schedule and doubling connection attempts. It cannot produce a false green (it errs
  toward *more* failures, sooner), so it is not implicated in §D either. Fix shape: read
  the count from a ref, do the branching in the handler, keep the updater as
  `prev => prev + 1`.

Per `feedback_queue_discovered_defects_in_masterplan`, both belong in their own
research-gated masterplan steps, not folded into 80.4.

---

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `frontend/src/lib/hooks/useEventSource.ts` | 88-93 | `cleanup()` -- closes ES, nulls ref | correct; suppresses only the *closed* object's events |
| " | 70, 124 | `eventType` default `"message"`; single `addEventListener` | the only path into the event counter |
| " | 109-122 | `onMessage` -- connected + reset budget/backoff + `onEvent` | untouched by any option below except R2's listener |
| " | 150-152 | `es.onopen` (phase-80.4) -- status only, no budget reset | correct (F2 mutation guards it) |
| " | 154-168 | `es.onerror` -- status/cleanup/budget/backoff | **intact for a real drop** (B1b) |
| " | 162 | reconnect `setTimeout` -- handle discarded, never cleared | pre-existing defect (B1c), NOT implicated in §D |
| " | 157-167 | impure `setFailures` updater | defect (B1c), NOT implicated in §D |
| " | 175-183 | mount effect; `return cleanup` | fine |
| " | 185-189 | `reconnect()` -- resets budget + backoff | reachable only via the >=5-failure banner (below) |
| `frontend/src/app/agents/page.tsx` | 5, 194-216 | the **only** consumer; `maxFailures: 5`; `onEvent` pushes EVERY event | blast radius of a semantics change = 1 page |
| " | 217, 309 | `connected = sseStatus === "connected"` -> `{connected ? "Connected" : "Disconnected"}` | binary label; any non-`connected` status reads Disconnected |
| " | 218-221, 374 | banner at `sseFailures >= 5` + its Retry button (`connect` = `sseReconnect`, `:222`) | **the intermediate failure count is never rendered, and manual reconnect is only reachable AFTER the budget is spent** |
| `frontend/src/lib/hooks/index.ts` | 9-10 | re-export | no other importers (grepped) |
| `frontend/src/lib/hooks/useEventSource.test.ts` | 11-30 | `MockEventSource` -- has `onopen`/`onerror`, **no `readyState`** | any `readyState`-based option needs the mock extended |
| " | 62-105 | phase-80.4 cases incl. the F2 flapping trap | passes; see the missing assertion in R1 |
| `backend/api/mas_events.py` | 54, 70 | `": connected"` + `": ping"` comment frames | invisible to JS **by design and by spec** (finding #3) |
| " | 68 | `asyncio.wait({pending}, timeout=...)`, not `wait_for` | correct; B3 mutation documented in the live_check |
| " | 94-102 | `StreamingResponse` -- no `request.is_disconnected()` / shutdown awareness | why SIGTERM hangs forever (finding #6) |

**Other consumers of `useEventSource`: none.** One import, one page.

---

## Ranked recommendation

### R1 -- RE-MEASURE with `kill -9`. Zero code change. **RECOMMENDED.**

The premise of options 1-4 is that the indicator is stuck green over a dead backend.
**It is not.** Re-run the §D capture killing `:8001` with `SIGKILL`:

```bash
kill -9 $(lsof -ti tcp:8001)      # or: --timeout-graceful-shutdown 1 on the rig
```

- **Criterion 4:** satisfied. The FIRST `error` flips `status` to `"error"`, so the label
  reads Disconnected within ~1 network round-trip -- comfortably inside the 5-failure /
  ~31s budget (B1b step 2). The subsequent budget exhaustion adds the banner.
- **Criterion 3 (`0 events`):** unaffected -- nothing changes.
- **New false states introduced:** none.
- **Size:** smallest possible.

Two things to ship alongside it so this cannot silently rot:

1. **Close the assertion gap in the unit suite.** `useEventSource.test.ts:81-104` asserts
   the END state (`disconnected` after the budget). It does **not** assert the criterion-4
   semantic that matters: *the first error leaves green immediately*. Add a case: drive
   `onopen` then a single `onerror` with `maxFailures: 5`, assert
   `status !== "connected"`. Mutation to kill it: make `onerror` skip `setStatus("error")`
   (leaving the reconnect machinery in place) -- the existing tests still pass, the new
   one fails. That is the guard §D actually needed.
2. **Record the kill discipline in the live_check.** Document verbatim that SIGTERM on
   uvicorn drains listeners but not established streams, cite the byte-growth measurement,
   and state that the capture used SIGKILL. This is the durable artifact -- the next
   auditor will otherwise repeat the artifact exactly.

**Honest caveat to state in the live_check, not to fix here:** after a SIGTERM the backend
is a half-dead process that serves only the one stream and refuses everything else. In
that state the MAS indicator reading "Connected" is *literally* correct but arguably
unhelpful. That is a UX question about what the indicator should mean, not a correctness
bug, and it is out of criterion 4's scope -- the stats-poll banner already covers it.

### R2 -- Named `heartbeat` event + client staleness timer. **Correct fix for the REAL gap; own step.**

Addresses the silent-blackhole case (sleep/wake, Wi-Fi drop) that `onerror` genuinely
cannot see. **Not** needed for criterion 4.

- **Does it preserve the `0 events` readout? YES, by spec, with no string comparison.**
  Per finding #4, an `event: heartbeat` frame is dispatched as type `"heartbeat"` and MDN
  states a `"message"` listener "will not trigger on any other event type". The hook's one
  listener is `addEventListener("message", onMessage)`. So the heartbeat is invisible to
  `onEvent`, `data`, `lastEventAt` and the `/agents` counter **without any filter code**.
  Implementation: a *second* listener, `es.addEventListener("heartbeat", () => {
  lastTrafficRef.current = Date.now(); })`, which touches a ref only -- no state, no
  re-render, no counter. Guard: refuse/skip if a consumer sets `eventType === "heartbeat"`.
- **Backend:** keep `": connected"` and `": ping"` exactly as they are (criterion 2's
  immutable byte-evidence stays valid) and emit an additional `event: heartbeat\ndata:
  {"ts": ...}\n\n` at the same cadence in `mas_events.py:70`. It composes with
  `include_buffer` and `to_sse()` -- both are separate `yield`s on the same generator; the
  heartbeat frame is emitted only on the timeout branch, so it can never interleave inside
  a real event's frame.
- **Failure mode to bound:** a watchdog can invent a **false RED** if a heartbeat is late
  (event-loop stall, GC). Use a >=2.5-3x threshold (the "3 missed heartbeats" idiom from
  websocket.org/RabbitMQ/Supabase) -> ~40-45s at a 15s cadence. A false red is the safe
  direction on an observability surface; a watchdog can only move *toward* disconnected,
  so **it cannot create a false green**.
- **Cost:** backend + hook + tests + a new mutation matrix. Real regression surface.
  Queue as its own research-gated step.

### R3 -- `readyState` polling backstop. **Do not ship.**

Polling `readyState` catches only the case where `error` fired but React state failed to
update -- for which there is no evidence. In the silent-blackhole case `readyState` stays
`OPEN` (the browser doesn't know either), so it does **not** cover the gap R2 covers. It
also needs a second ref (`cleanup()` nulls `sourceRef.current` at `:91`) and a
`readyState` property added to `MockEventSource`. Cost without a covered failure mode.

Narrow exception worth keeping in mind: reading `readyState` **inside `onerror`** (finding
#2) to stop retrying on a terminal `CLOSED` is a small, genuine improvement -- but it is
about retry economics, not death detection, and it belongs with R2's step if at all.

### R4 -- "Client watchdog on any traffic" (as literally proposed). **Impossible today. HARD NO.**

Answering the caller's question plainly: **yes, it collapses.** On an idle stream there is
no observable traffic at all -- comment frames dispatch nothing (finding #3), and
`whatwg/html#7571` is open precisely because clients cannot see them. A traffic watchdog
on today's stream would time out on every healthy idle stream and paint it red after one
interval -- i.e. it would *reintroduce the original phase-80.4 bug*, permanently. It only
becomes viable once R2 supplies a real event to observe, at which point it IS R2.

### Summary table

| Option | Fixes criterion 4 | Keeps `0 events` | New false-GREEN risk | New false-RED risk | Size |
|---|---|---|---|---|---|
| **R1 re-measure with `kill -9`** | **yes (already correct)** | yes | none | none | zero code |
| R2 named heartbeat + watchdog | n/a (fixes a different, real gap) | **yes -- by spec, no filter** | none | bounded (use 3x threshold) | backend + hook + tests |
| R3 readyState polling | no | yes | none | low | medium, no covered failure mode |
| R4 traffic watchdog on comments | no | yes | none | **certain** (breaks healthy idle) | n/a -- not implementable |

**Anything that could make the indicator green while the backend is dead: none of the
above does.** The hook goes non-green on the first `error`, and the only proposal that
touches the green path (R2's watchdog) can only take green away.

---

## Application to pyfinagent

1. Re-run `live_check_80.4.md` §D with `kill -9 $(lsof -ti tcp:8001)`; expect the label at
   `frontend/src/app/agents/page.tsx:309` to read **Disconnected** within seconds, then
   the `sseFailures >= 5` banner (`:218-221`) after ~31s of backoff.
2. Add the first-error assertion to `frontend/src/lib/hooks/useEventSource.test.ts` and its
   mutation (strip `setStatus("error")` from `useEventSource.ts:155`).
3. Record the SIGTERM-vs-SIGKILL finding in the live_check, with the byte-growth evidence.
4. Queue three follow-up steps (do not fold into 80.4): **R2** silent-drop watchdog;
   **B1c-i** uncleared reconnect timer at `useEventSource.ts:162`; **B1c-ii** impure
   `setFailures` updater at `useEventSource.ts:157-167`.

## Consensus vs debate (external)

**Consensus:** comment keepalives are the standard SSE idiom and are ignored by clients
(WHATWG, MDN, oneuptime 2026, javascript.info). `EventSource` auto-reconnects and fires
`error` per attempt. Named events require `addEventListener` and never reach `onmessage`.

**Debate / open:** whether clients *should* be able to observe keepalives at all
(`whatwg/html#7571`, open, `[ADVERSARIAL]` to the "just use a watchdog" instinct -- the
spec editors have not accepted it). And whether an application-level watchdog is warranted
at all: [timetobuildbob](https://timetobuildbob.com/blog/stale-event-sse-reconnect/), the
most recent hands-on SSE-reliability write-up found, **declines to recommend a watchdog**
and invests in generation counters + timer hygiene instead -- which is evidence for R1+B1c
over R2 as the higher-yield work.

## Pitfalls (from literature + measurement)

- Stopping a uvicorn SSE server with SIGTERM does **not** close established streams
  (measured; uvicorn#451). Any "kill the backend" test must use SIGKILL or set
  `--timeout-graceful-shutdown`.
- `curl -> 000` proves the *listener* is gone, **not** that established connections died.
  Do not use it as the death oracle for an SSE test.
- Comment heartbeats prove liveness to `curl`, never to JS (`whatwg/html#7571`).
- `onerror` also fires on transient reconnects; treating every `error` as fatal is the
  documented over-reaction (oneuptime 2026).
- Reconnect timers must be stored and cleared or you get concurrent streams
  (timetobuildbob).
- React updater functions must be pure; Strict Mode double-invokes them (react.dev).

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (**10**)
- [x] 10+ unique URLs total (**~29** incl. snippet-only)
- [x] Recency scan (2024-2026) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (hook, page, test, backend
      endpoint, hooks index; consumer grep run)
- [x] Contradictions / consensus noted (whatwg#7571 open; timetobuildbob declines the
      watchdog)
- [x] All claims cited per-claim
- Gap: `https://issues.chromium.org/issues/41003167` is behind a sign-in wall, so the
  sleep-mode browser bug is cited from its search snippet only, not read in full. It is
  corroborating context for R2, not load-bearing for the R1 recommendation.
- Gap: `www.uvicorn.org` did not resolve from this sandbox (twice). Substituted a
  **stronger** primary source -- `inspect.getsource(uvicorn.server.Server.shutdown)` on
  the installed 0.42.0 plus a live SIGTERM/SIGKILL experiment.

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 19,
  "urls_collected": 29,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 1,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Criterion 4 was never tested. The :8001 backend was stopped with SIGTERM; uvicorn 0.42.0 closes the LISTENING socket but keeps established streams open forever (timeout_graceful_shutdown default None). Reproduced live: post-SIGTERM the process stayed alive, new curls got 000, and the open SSE stream kept receiving ': ping' (61->93 bytes/10s); SIGKILL ended it instantly. So curl=000, the failing stats poll, the green indicator and the absent retries are ALL correct simultaneously. Per WHATWG, a genuine drop fires 'error' on every reestablish, and the hook flips status to 'error' on the FIRST one (useEventSource.ts:155) -- green is lost immediately, well inside the budget. cleanup() suppressing later events on the closed object is intended; each retry builds a new EventSource. Comment frames are invisible to JS (spec + open whatwg/html#7571), so a traffic watchdog on ': ping' is impossible. A named 'heartbeat' event would NOT reach the 'message' listener (MDN), so the '0 events' readout is safe by spec with no string filter. Ranked: R1 re-measure with kill -9, zero code change; R2 named heartbeat + 3x watchdog as its own step for the silent-blackhole gap; R3 readyState polling no; R4 traffic watchdog impossible. Two real defects queued: uncleared reconnect timer (:162), impure setFailures updater (:157-167).",
  "brief_path": "handoff/current/research_brief_80.4b_death_detection.md",
  "gate_passed": true
}
```
