# live_check — phase-80.4 — ALL FIVE CRITERIA MET (on the rig; NOT yet live for the operator)

> **The operator's own `/agents` on `:3000` will keep reading "Disconnected" until `:8000`
> is restarted.** Measured by the cycle-3 Q/A against the live backend: the SSE route
> returned **zero bytes in 6s** and `/api/mas/events/stats` returned
> `{"total_events":0,"buffer_size":0,"subscribers":0}`. pid `70791` predates this fix and
> the restart is blocked by the open **phase-79.55 RESTART BLOCKER**. No criterion requires
> the fix to be live on `:8000`, and every measurement below is attributed to the `:8001`
> rig — but the operator should not expect their surface to change before the restart.
>
> That stats reading also **positively confirms** a premise these artifacts had previously
> only asserted: the MAS bus really has published 0 events since process start.

**Required (masterplan, verbatim):** *Playwright screenshot of `/agents` showing 'Connected'
with 0 events, curl showing the heartbeat bytes on the SSE stream, and a second capture
after stopping the backend showing it correctly reads Disconnected.*

Captured 2026-07-26.

---

## §0. CORRECTION — cycle 1's criterion-4 FAIL was a MEASUREMENT ERROR, not a defect

The previous revision of this file reported criterion 4 as FAILING and concluded the fix
had produced "a false green on a dead backend." **That conclusion was wrong, and the
product code was never at fault.** It is recorded here rather than quietly overwritten.

Cycle 1 stopped the `:8001` rig with a default `kill` / `pkill` — **SIGTERM**. Uvicorn's
graceful shutdown closes the *listening socket* but keeps *established* connections open,
waiting for their tasks to finish. An SSE generator never finishes. So after SIGTERM the
process stays alive serving exactly the open stream, and `curl → 000` reports only that
the listener is gone — **it is not a death oracle for an already-open stream.**

The indicator was green because **the stream was genuinely alive.** Green was correct.

### The measurement, re-run by Main from scratch

One `curl -N` stream held open against the `:8001` rig, byte count sampled over time:

```
baseline (healthy)            stream = 21 bytes, curl alive
── kill -TERM ──────────────────────────────────────────────
  uvicorn process alive:      YES
  NEW connection:             000          <- what fooled cycle 1
  stream bytes:               29
  stream bytes +20s:          37           -> GREW: YES  (still pinging)
  curl alive:                 YES
── kill -9 ─────────────────────────────────────────────────
  uvicorn process alive:      no
  curl alive:                 NO  -- stream hit EOF
  stream bytes:               69
  stream bytes +18s:          69           -> GREW: NO   (frozen, dead)
```

`SIGTERM` → stream keeps growing. `SIGKILL` → EOF. **`kill -9` is the correct death
oracle for SSE**, and criterion 4 is tested with it below.

Root cause in the library, read from source rather than docs:
`uvicorn.server.Server.shutdown` closes the listeners, then awaits
`_wait_tasks_to_complete()` with `timeout_graceful_shutdown` defaulting to **`None`** —
wait forever. Corroborated by Kludex/uvicorn#451: with a stream open, the first SIGTERM
"will not stop the responses". Full sourcing in
`handoff/current/research_brief_80.4b_death_detection.md` (10 sources read in full,
29 URLs, `gate_passed: true`).

**CORRECTION to an earlier revision of this section (Q/A finding 3).** A previous draft
said "a second *cycle-1* error compounded it: the first `:8001` rig ran without
`DEV_LOCALHOST_BYPASS=1`, so the SSE curl got 401 and no stream was ever held open."
**That attribution was wrong.** Cycle 1's rig streamed fine — its own capture
`scratchpad/sse804.txt` holds 21 bytes (`: connected` + `: ping`) written 00:46. The 401
was **cycle 2's** first rig, started ~01:33 by Main *today*, and it was Main's own error,
not an additional cycle-1 fault. Cycle 1's single error was the SIGTERM oracle.

The 401 is still worth recording, because it is the same rig-auth trap that produced the
80.2 false pass — hit twice now, in two different cycles. Every rig used for measurement
in this revision was verified with a live `200` on the SSE route **before** any bytes were
counted.

## §A. Method

Isolated `:8001` backend (`DEV_LOCALHOST_BYPASS=1`, `--lifespan off` — no scheduler, no
second trading loop) + isolated skip-auth `:3100` frontend
(`LIGHTHOUSE_SKIP_AUTH=1`, `PLAYWRIGHT_DIST_DIR=.next-audit-3100`,
`NEXT_PUBLIC_API_URL=http://localhost:8001`).

The operator's `:8000` was never restarted (`phase-79.55` is an open RESTART BLOCKER) and
their `:3000` was never driven. The isolated `distDir` is required by
`feedback_second_next_dev_breaks_operator_3000` — a second `next dev` sharing `.next`
breaks the operator's `:3000`. Verified healthy at every stage: `:3000/` → `302`,
`:3000/login` → `200`, `:8000/api/health` → `200`, pid `70791` unchanged.

The `:8001` rig exists specifically so criterion 4 could be tested by **killing a backend
I own**. `kill -9` was issued against pid `79781` after an explicit guard that it was not
`70791`.

## §B. Criterion 2 — the heartbeat, live

```
$ curl -s -N -m 20 -H 'Accept: text/event-stream' \
    'http://localhost:8001/api/mas/events?include_buffer=true'
: connected$
$
: ping$
$
  ": connected" lines: 1
  ": ping"      lines: 1
  "data:"       lines: 0      <- idle bus, so no real events; correct
```

An idle stream is now distinguishable from a dead one. Every heartbeat byte is an SSE
**comment** (`:` prefix), so per WHATWG it is ignored by `EventSource` and can never reach
`onmessage` or inflate the event counters — which is what keeps criterion 3's `0 events`
readout honest.

## §C. Criterion 3 — MET

`captures_80.4/80.4_CONNECTED_before_sigkill.png` (01:40:25) is the primary artifact. The
a11y text below came from a **live `browser_snapshot` tool response** during this session,
not from a persisted `.yml`:

> `- generic [ref=e95]: Connected`
> `- generic [ref=e138]: 0 events | 1 sub`

**Provenance caveat (cycle-3 Q/A finding, WARN).** The only persisted cycle-2 snapshot,
`.playwright-mcp/page-2026-07-25T23-40-07-259Z.yml`, was written at **navigation** time and
therefore captures the **pre-connect** state, reading `Disconnected`. An earlier revision
quoted these refs as if from a persisted cycle-2 file; they happen to coincide with cycle
1's `22-47-06` snapshot. The *substance* is independently confirmed — the cycle-3 Q/A read
both PNGs itself and verified green "Connected · 0 events | 1 sub" and red "Disconnected".
Read the PNGs, not the refs.

Green dot, **"Connected"**, **0 events**, 1 subscriber. That is the exact failing case from
the audit — an open, healthy stream with zero events — now reading correctly.

## §D. Criterion 4 — MET (with the correct death oracle)

`captures_80.4/80.4_DISCONNECTED_after_sigkill.png` (01:41:20) is the primary artifact; the
a11y text below is from a live `browser_snapshot` response (same provenance caveat as §C).

Independent corroboration from `.playwright-mcp/console-2026-07-25T23-40-06-809Z.log`:
a single **`ERR_INCOMPLETE_CHUNKED_ENCODING`** on `/api/mas/events` — the SIGKILL cutting
the open stream mid-body — followed by repeated **`ERR_CONNECTION_REFUSED`** reconnect
attempts. That is the byte-level signature of a genuinely dead backend, and it is exactly
what SIGTERM did *not* produce.

`kill -9 79781` → `:8001` → `000`, 0 listeners. After ~40s the same page reads:

> `- generic [ref=e95]: Disconnected`
> `- paragraph [ref=e150]: Lost connection to MAS event stream after 5 failures. Backend may be down.`
> `- button "Retry" [ref=e151]`

The indicator flipped to **Disconnected** and the failure budget **exhausted at exactly 5**,
which is the designed behaviour — `onerror` fires, `useEventSource.ts:154-168` counts the
failure and backs off, and `agents/page.tsx:217` turns the dot red. 20 console errors
recorded on the page across the retries, consistent with 5 failed reconnects.

## §E. What IS verified

| # | Criterion | Status |
|---|---|---|
| 1 | `onopen` handler set, status on establishment | **MET** — `useEventSource.ts:150`, `es.onopen = …` (property form, so the immutable `grep -n 'onopen'` sees it) |
| 2 | initial comment + periodic keepalive | **MET** — §B |
| 3 | `/agents` shows Connected with 0 events | **MET** — §C |
| 4 | killing the backend flips to Disconnected | **MET** — §D, `kill -9` |
| 5 | existing test passes + gains an open-but-no-events case | **MET** — 5 passed |

**Suites:** `pytest test_phase_80_4_sse_heartbeat.py` → **7 passed**;
`vitest useEventSource.test.ts` → **5 passed**. Immutable command exits **0**.

## §F. The assertion this suite was missing

Cycle 1's frontend tests pinned only the **end** state (`disconnected` once the budget is
spent). Nothing asserted that the indicator stops being green on the **first** error —
the entire pre-exhaustion window was unguarded — **~15s** on `/agents`
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
> a step from the same edit.
Since `/agents:217` derives green from the single condition `sseStatus === "connected"`,
that transition is exactly what death detection rests on.

Added `stops being green on the FIRST error, not only after the budget exhausts`.

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

## §G. Teardown

```
:3100 listeners: 0
:8001 listeners: 0
operator :8000 -> 200, pid 70791 (never restarted)
operator :3000/ -> 302 ; :3000/login -> 200
frontend/tsconfig.json + next-env.d.ts restored from HEAD (git diff clean)
```

`next dev` rewrote both TS config files to point at `.next-audit-3100`; both were restored
from `HEAD` after inspecting the diff (rig-generated only, no authored content discarded).

**Owed to the operator:** `rm -rf frontend/.next-audit-3100` (leftover rig build dir).

## §H. Two real defects found in the hook — NOT implicated here, queued separately

Neither can produce a false green (both err toward *more* failures, sooner), so neither
blocks this step. **Queued as masterplan steps `80.33` and `80.34`** (created 2026-07-26,
both `status: pending`, `harness_required: true`, each with a failing-first test criterion
and a mutation criterion) per the queue-discovered-defects rule:

1. **`80.33`** — `useEventSource.ts:162`: the reconnect `setTimeout` handle is discarded and
   never cleared on unmount, so a post-unmount `connect()` can leak a stream.
2. **`80.34`** — `useEventSource.ts:157-167`: the `setFailures` updater is **impure** (it
   calls `setTimeout` and `setStatus` inside the updater). React requires updaters to be
   pure and Strict Mode invokes them twice.

   **Scope-honesty note:** an earlier revision asserted "Next 15 defaults
   `reactStrictMode: true` and `next.config.js` does not override it, so every error
   schedules two reconnects in dev." **That was never measured in this step.** The
   impurity is real and read directly from the source; the *consequence* is conditional on
   a Strict Mode setting I did not verify. Step `80.34` carries an explicit criterion to
   measure and quote the resolved value rather than inherit the claim.

## §I. Disposition

**All five criteria met. 80.4 is ready to close.** The product code is unchanged from the
cycle-1 WIP commit (`4bcd60ad`) except for the added test; what changed is the
measurement, and the honest record of why cycle 1's was invalid.
