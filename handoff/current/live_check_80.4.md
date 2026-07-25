# live_check — phase-80.4 — **NOT COMPLETE. CRITERION 4 FAILS.**

**Required (masterplan, verbatim):** *Playwright screenshot of `/agents` showing 'Connected'
with 0 events, curl showing the heartbeat bytes on the SSE stream, and a second capture
after stopping the backend showing it correctly reads Disconnected.*

Captured 2026-07-26. **The third clause is NOT satisfied, and I am recording that rather
than shipping around it.** Criteria 1, 2, 3 and 5 are met; **criterion 4 is not.**

---

## §A. Method

Isolated `:8001` backend (`--lifespan off`, so no scheduler and no second trading loop) +
isolated skip-auth `:3100` frontend with `NEXT_PUBLIC_API_URL=http://localhost:8001`. The
operator's `:8000` was never restarted (`phase-79.55` is an open RESTART BLOCKER) and their
`:3000` was never driven — `302` throughout, `:8000` pid `70791` unchanged.

The `:8001` rig exists specifically so criterion 4 could be tested by **killing a backend I
own** rather than the operator's.

## §B. Criterion 2 — the heartbeat, live

```
$ curl -s -N -m 20 -H 'Accept: text/event-stream' \
    'http://localhost:8001/api/mas/events?include_buffer=true'      # phase-80.4 code
: connected$
$
: ping$
$
  bytes received: 21
  ": connected" lines: 1
  ": ping"      lines: 1
  "data:"       lines: 0      <- idle bus, so no real events; correct

$ same call against the operator's UN-RESTARTED :8000 (old code)
  bytes received: 0
```

**21 bytes vs 0.** An idle stream is now distinguishable from a dead one. Every heartbeat
byte is an SSE **comment** (`:` prefix), so it cannot reach `onmessage` or inflate the
event counters.

## §C. Criterion 3 — MET

`captures_80.4/80.4_agents_CONNECTED_zero_events.png`: green dot, **"Connected"**, and
**"0 events | 1 sub"** beside it. That is the exact failing case from the audit — an open,
healthy stream with zero events — now reading correctly.

## §D. Criterion 4 — **FAILS. This is the blocker.**

`captures_80.4/80.4_agents_DISCONNECTED_after_full_budget.png`.

`:8001` was killed (`curl` → `000`, 0 listeners). I then waited **~70 seconds** — well past
the page's `maxFailures: 5` budget with its 1+2+4+8+16 ≈ 31s of exponential backoff.

**The MAS indicator was still green "Connected".**

The same capture shows the *stats poll* correctly failing right below it:

> *"Agent stats polling stopped after 5 consecutive failures. Last error: Cannot reach
> backend at http://localhost:8001..."*

So the page knows the backend is gone. The SSE indicator does not.

### Why this matters more than the bug it was meant to fix

Pre-fix, `status` only became `"connected"` when an event arrived. With an idle bus it
stayed `"connecting"` → red. That was wrong on a healthy backend, but **accidentally right
on a dead one**.

Post-fix, `onopen` sets `"connected"` — correct on a healthy backend, but the transition
back out depends entirely on `onerror` firing. On this evidence it does not fire reliably,
so the indicator is now **stuck green over a dead backend**. A false green on an
observability surface is worse than a false red: it is the exact "always-green" failure
criterion 4 was written to prevent, and the criterion did its job.

### The likely root cause — and it is a design problem, not a typo

**EventSource never surfaces comment lines to JavaScript.** The `: ping` heartbeat keeps
the transport alive and proves liveness to `curl`, but it is **invisible to the client**,
so the hook cannot use it for staleness detection. Client-side death detection therefore
rests entirely on `onerror`.

Two candidate fixes, neither of which I am shipping without a research gate:

1. **Emit a real named event** (`event: heartbeat\ndata: {...}`) instead of / alongside the
   comment, and have the hook mark the stream stale when none has arrived within
   ~2.5× the interval. This gives the client a positive liveness signal instead of relying
   on an error that may not fire. Cost: it *does* reach `onmessage`, so the event-counter
   and buffer paths must explicitly filter it — a real regression risk for the `0 events`
   readout that criterion 3 depends on.
2. **Make the hook's death detection independent of `onerror`** — e.g. treat
   `readyState === CLOSED` or a `connecting` state persisting beyond a deadline as
   disconnected.

Both change client semantics and need their own gate. Guessing between them here is
exactly what this session has been correcting.

## §E. What IS verified

| # | Criterion | Status |
|---|---|---|
| 1 | `onopen` handler set, status on establishment | **MET** — `useEventSource.ts:150`, `es.onopen = …` (property form, so the immutable `grep -n 'onopen'` sees it) |
| 2 | initial comment + periodic keepalive | **MET** — §B, 21 bytes vs 0 |
| 3 | `/agents` shows Connected with 0 events | **MET** — §C |
| 4 | killing the backend still flips to Disconnected | **FAILS** — §D |
| 5 | existing test passes + gains an open-but-no-events case | **MET** — 4 passed, incl. the flapping case |

**Backend + frontend suites:** `pytest test_phase_80_4_sse_heartbeat.py` → **7 passed**;
`vitest useEventSource.test.ts` → **4 passed**. Immutable command exits **0**.

**Mutation matrices — 7/7 killed:**

```
backend   B1 remove ': connected'                       KILLED
          B2 remove ': ping'                            KILLED
          B3 THE TRAP: wait_for(__anext__) not wait()   KILLED  (5 of 7 tests fail)
          B4 heartbeat emits data: not a comment        KILLED
frontend  F1 delete the onopen handler                  KILLED
          F2 THE TRAP: reset the failure budget in onopen  KILLED
          F3 onopen sets the wrong status               KILLED
```

**B3 is worth keeping in the record:** the obvious implementation,
`asyncio.wait_for(agen.__anext__(), timeout)`, cancels its inner awaitable, which throws
`CancelledError` **into** `MASEventBus.subscribe` and runs its `finally` — silently
unsubscribing the client on the **first idle heartbeat**. The stream would keep pinging
while never delivering another event. `asyncio.wait({pending}, timeout=…)` leaves the task
alive across timeouts instead.

**F2 is the criterion-4 trap in unit form** and it *is* guarded: resetting `failures` in
`onopen` would let a flapping backend stay green forever. The unit test proves the budget
still exhausts. **But the live browser evidence in §D shows the budget is not being reached
at all** — so the unit guard is necessary and not sufficient, which is precisely why the
masterplan demanded a live capture.

## §F. Teardown

```
:3100 listeners: 0
:8001 listeners: 0
operator :3000/ -> 302
operator :8000 pid -> 70791   (never restarted)
```

`frontend/tsconfig.json` + `next-env.d.ts` restored from HEAD, `git status` clean on both.

## §G. Disposition

**80.4 stays `pending`.** The code is written, mutation-tested and 4 of 5 criteria are met,
but shipping it would replace a false-red with a **false-green on a dead backend** — a
strictly worse failure on an observability surface. The remaining work needs its own
research gate on EventSource death-detection semantics.
