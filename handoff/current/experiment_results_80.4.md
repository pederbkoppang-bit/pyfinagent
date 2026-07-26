# Experiment Results — phase-80.4

**Step:** `80.4` (P0) — `/agents` reported "Disconnected" over a working SSE endpoint.
Date 2026-07-26. Contract: `handoff/current/contract_80.4.md`.
Gates: `research_brief_80.4.md` + `research_brief_80.4b_death_detection.md`
(both `gate_passed: true`).

**Written in cycle 2.** Cycle 1 never produced this file — disclosed in the contract's
PROTOCOL DISCLOSURE section rather than backdated.

---

## What was built

### Cycle 1 — product code (committed WIP `4bcd60ad`, unchanged in cycle 2)

**`backend/api/mas_events.py`** — `_HEARTBEAT_SECONDS = 15.0`. `event_generator` yields
`": connected\n\n"` immediately, then loops on
`asyncio.wait({pending}, timeout=_HEARTBEAT_SECONDS)`, yielding `": ping\n\n"` on timeout.
`finally` cancels the pending task, awaits it, then `aclose()`s the generator.

The `asyncio.wait` choice is load-bearing: `asyncio.wait_for(agen.__anext__(), t)` cancels
its inner awaitable on timeout, which throws `CancelledError` **into**
`MASEventBus.subscribe` and runs its `finally` — silently unsubscribing the client on the
**first idle heartbeat**. The stream would keep pinging while never delivering another
event: strictly worse than the bug being fixed.

**`frontend/src/lib/hooks/useEventSource.ts`** — `es.onopen` sets `status` on connection
establishment. Property form (`es.onopen = …`), so the immutable `grep -n 'onopen'` sees
it. It deliberately sets status **only** — resetting `failures`/`backoffRef` there would
let a flapping backend stay permanently green.

### Cycle 2 — no product-code change; one test added

**`frontend/src/lib/hooks/useEventSource.test.ts`** — added
`stops being green on the FIRST error, not only after the budget exhausts`.

The pre-existing tests pinned only the **end** state (`disconnected` after the budget
spends). The whole pre-exhaustion window was unguarded — **~15s** on `/agents`
(`maxFailures: 5` at `agents/page.tsx:200`; the guard is `if (next < maxFailures)`, so only
FOUR reconnects are ever scheduled: 1+2+4+8s). Measured **9.0s** in the actual capture run
(first error at 25631ms, fifth at 34665ms in
`.playwright-mcp/console-2026-07-25T23-40-06-809Z.log`).

> **CORRECTION (cycle 4, Q/A WARN 1).** Earlier revisions said *"~31s of backoff at the
> default budget"*. Both halves were wrong, and the error was **inherited from cycle 1**
> (`git show 4bcd60ad:handoff/current/live_check_80.4.md:55`), not introduced by a
> remediation. (a) Off-by-one: the `1+2+4+8+16 ≈ 31s` sum includes a `+16` term that never
> executes, because `next < maxFailures` stops scheduling at the 4th reconnect. (b) *"the
> default budget"* is wrong twice — the hook default is `maxFailures ?? 3`
> (`useEventSource.ts:71`), a 3s window; `/agents` overrides it to 5. Derived, not
> asserted: delays 1+2+4+8 = **15s**. Masterplan step `80.33`, authored during the cycle-2
> remediation, already carried the correct figure — so the artifacts had been contradicting
> a step from the same edit. Since `agents/page.tsx:217` derives green from the single condition
`sseStatus === "connected"`, that first transition is what death detection rests on.

## Files changed

| File | Cycle | Change |
|---|---|---|
| `backend/api/mas_events.py` | 1 | heartbeat + announce; `asyncio.wait` loop; explicit teardown |
| `frontend/src/lib/hooks/useEventSource.ts` | 1 | `es.onopen` sets status only |
| `backend/tests/test_phase_80_4_sse_heartbeat.py` | 1 | new, 7 tests |
| `frontend/src/lib/hooks/useEventSource.test.ts` | 1→2 | +2 cases in c1, **+1 in c2** (5 total) |
| `handoff/current/live_check_80.4.md` | 2 | rewritten — criterion 4 now MET |
| `handoff/current/contract_80.4.md` | 2 | new (disclosing the cycle-1 gap) |

## Verbatim verification output

**Immutable command** (venv active, per the project's own "always `source .venv/bin/activate`" rule):
```
$ cd frontend && npx tsc --noEmit -p tsconfig.json && grep -n 'onopen' src/lib/hooks/useEventSource.ts && python -c "import ast; ast.parse(open('../backend/api/mas_events.py').read())"
138:      // `onopen` fires on the validated response HEADERS, before the body is
150:      es.onopen = () => {
EXIT=0
```
(Note: run **without** the venv it exits 127 — `python` is not on the bare PATH. Recorded
because the first attempt did exactly that.)

**Suites:**
```
$ python -m pytest backend/tests/test_phase_80_4_sse_heartbeat.py -q
7 passed in 0.78s

$ npx vitest run src/lib/hooks/useEventSource.test.ts
 Test Files  1 passed (1)
      Tests  5 passed (5)
```

## The correction — cycle 1's criterion-4 FAIL was a measurement error

Cycle 1 stopped the rig with SIGTERM. Uvicorn's graceful shutdown closes the *listener*
but keeps *established* connections open, and an SSE generator never completes — so the
process stayed alive serving the open stream. The indicator was green because **the stream
was genuinely alive**.

Main re-measured from scratch:

```
baseline (healthy)       stream = 21 bytes, curl alive
── kill -TERM ─────────────────────────────────────────
  uvicorn alive:  YES      NEW conn: 000   <- what fooled cycle 1
  bytes: 29 -> 37 (+20s)   GREW: YES       curl alive: YES
── kill -9 ────────────────────────────────────────────
  uvicorn alive:  no                       curl alive: NO (EOF)
  bytes: 69 -> 69 (+18s)   GREW: NO        frozen -- dead
```

`kill -9` is the correct death oracle for SSE. Under it, criterion 4 passes: the page
reads **"Disconnected"** and *"Lost connection to MAS event stream after 5 failures."*

**Correction (Q/A finding 3).** An earlier revision of this paragraph blamed cycle 1 for a
401 rig. Wrong attribution: cycle 1's rig streamed fine (`scratchpad/sse804.txt`, 21 bytes,
00:46). The 401 was **cycle 2's** first rig — Main's own error today, ~01:33 — and it is
still worth recording because it is the same rig-auth trap that produced the 80.2 false
pass, now hit twice in two cycles. Every rig used for measurement here was verified `200`
on the SSE route before any bytes were counted.

## Mutation matrix

**Mutation matrix — 10 DISTINCT mutations, all killed** (4 backend + 6 frontend).

> **CORRECTION (cycle 3, Q/A finding).** An earlier revision of this line claimed
> "13/13 killed (7 backend from cycle 1 + 6 frontend in cycle 2)". **Both halves were
> wrong**, and the error was introduced BY the cycle-2 remediation — the same
> assert-a-count-you-did-not-derive defect that remediation was fixing.
> (a) **Composition:** cycle 1 was **4 backend + 3 frontend**, never 7 backend — see its
> own committed artifact, `git show 4bcd60ad:handoff/current/live_check_80.4.md`.
> (b) **Double-count:** cycle-2's M3/M4/M6 are the SAME mutations as cycle-1's F1/F2/F3,
> re-run independently by the Q/A. A re-run is **re-verification, not a new mutation**.
> Summing 7+6 counted them twice. The distinct set is enumerated below and is **10**.

```
BACKEND (4) -- authored cycle 1
  B1  remove ': connected'                            KILLED   2 failed | 5 passed
  B2  remove ': ping'                                 KILLED   4 failed | 3 passed
  B3  THE TRAP: wait_for(__anext__) not wait()        KILLED   5 failed | 2 passed
  B4  heartbeat emits data: not a comment             KILLED   3 failed | 4 passed

FRONTEND (6)
  F1  delete the es.onopen handler                    KILLED   3 failed | 2 passed
        authored c1; re-run independently by c2 Q/A as M3, and again by c3 Q/A
  F2  THE TRAP: reset the failure budget in onopen    KILLED   1 failed | 4 passed
        authored c1; re-run as M4 (c2 Q/A), again by c3 Q/A
  F3  onopen sets the wrong status ("connecting")     KILLED   2 failed | 3 passed
        authored c1; re-run as M6 (c2 Q/A), again by c3 Q/A
  M1  delete setStatus("error") from onerror          KILLED   1 failed | 4 passed
        NEW in c2 (Main); re-run independently by c3 Q/A
  M2  onerror sets "connected" (the false-green shape) KILLED  1 failed | 4 passed
        NEW in c2 (Main)
  M5  drop the terminal setStatus("disconnected")     KILLED   1 failed | 4 passed
        NEW in c2 (Q/A)
```

**Every one of the 10 rows has now been reproduced from scratch by an independent
evaluator** — so these are measured, not transcribed. The cycle-3 Q/A re-ran 8 (B1, B2, B3,
B4, F1, F2, F3, M1); the cycle-4 Q/A re-ran B2, **M2 and M5** — deliberately choosing the
two rows no prior evaluator had touched — and reproduced the matrix exactly.

> **Precision note (cycle 4, Q/A WARN 2).** An earlier revision said the cycle-3 Q/A's 8
> matched "with exactly matching pass/fail counts". The *set* is right, but that phrasing
> cannot hold for **B1, B2 and B4**: cycle 1 recorded only `KILLED` for those three with no
> counts (only B3 carried "5 of 7 tests fail"), so the cycle-3 Q/A **originated** those
> numbers rather than matching a prior claim. The operative point — that all 8 were
> measured, not transcribed — is true; the word "matching" was not.

M1 and M2 leave **all four pre-existing tests passing** and are caught **only** by the new
first-error test: that is the proof the guard gap was real and that this assertion closes
it. File restored from `HEAD` after every mutation; final md5 matches baseline.

This is what satisfies criterion 4's *"MUTATION-TEST the guard both directions"*:
**must go green when healthy** — F1, F3; **must leave green when dead** — M1, M2, F2, M5.

## Artifact shape

`/api/mas/events` on an idle bus:
```
: connected

: ping

```
Comments only — per WHATWG they are ignored by `EventSource`, so they cannot reach
`onmessage` or inflate the `0 events` readout that criterion 3 depends on.

## Scope honesty

- **No product code changed in cycle 2.** What changed is the measurement and the record.
- Two genuine hook defects were found and **not** fixed here — out of scope, and now
  genuinely queued as masterplan steps **`80.33`** (discarded reconnect `setTimeout`
  handle, `:162`) and **`80.34`** (impure `setFailures` updater, `:157-167`). Neither can
  produce a false green — both err toward more failures, sooner. `80.34` explicitly flags
  that this step never measured whether `reactStrictMode` is actually enabled, so the
  "double-schedules in dev" consequence is unverified and must be measured there.
- The named-`heartbeat`-event design (R2 in the cycle-2 brief) is **not** adopted here. It
  addresses a different gap — a silent blackhole (sleep/wake, Wi-Fi drop) where `onerror`
  never fires — and belongs in its own step.
- Operator's `:8000` never restarted, `:3000` never driven; verified `200`/`302`/`200` at
  teardown, pid `70791` unchanged.
- **Owed:** `rm -rf frontend/.next-audit-3100` (leftover rig build dir).
