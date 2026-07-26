# Contract — phase-80.4

**Step id:** `80.4` (phase-80, priority **P0**, `harness_required: true`)
**Title (masterplan):** *`/agents` reports "Disconnected" over a working SSE endpoint.*

---

## PROTOCOL DISCLOSURE — read this first

**This contract was authored at the start of cycle 2, not before cycle 1's GENERATE.**

Cycle 1 (2026-07-25/26) ran RESEARCH (`research_brief_80.4.md`, `gate_passed: true`) and
GENERATE, and produced `live_check_80.4.md`, but **never wrote `contract_80.4.md` or
`experiment_results_80.4.md`**. That is a breach of the contract-before-GENERATE rule
(`feedback_contract_before_generate`). It is recorded here rather than backdated. Cycle 1
did not close the step and did not append to `harness_log.md`, so nothing was certified on
the missing artifacts — but the gap is real and this disclosure is the correction.

Cycle 2 ran a fresh research gate before any new work:
`research_brief_80.4b_death_detection.md` (`gate_passed: true`, 10 sources read in full,
29 URLs, recency scan performed).

---

## Research-gate summary

**Cycle-1 brief** (`research_brief_80.4.md`): WHATWG EventSource semantics — `onopen`
fires on validated response *headers*, before the body is interpreted; comment lines
(`:` prefix) are ignored and never surface to `onmessage`. Established that connection
state must derive from connection establishment, not data arrival.

**Cycle-2 brief** (`research_brief_80.4b_death_detection.md`) — commissioned specifically
because cycle 1 reported criterion 4 as failing. Its finding **overturned cycle 1's
conclusion**:

- Uvicorn's `Server.shutdown` closes the listening socket, then awaits
  `_wait_tasks_to_complete()` with `timeout_graceful_shutdown` defaulting to **`None`**
  (wait forever). An SSE generator never completes, so **after SIGTERM the process stays
  alive serving the already-open stream**. Corroborated by Kludex/uvicorn#451.
- Therefore `curl → 000` proves only that the *listener* is gone. It is **not** a death
  oracle for an already-open stream. Cycle 1's criterion-4 test was invalid.
- On a genuine drop, WHATWG's *reestablish the connection* fires `error` before each
  retry, so `onerror` is a sound death signal; there was never a bug on that path.
- A `"message"` listener does not fire for other event types (MDN), so a named
  `heartbeat` event could not pollute the `0 events` counter — recorded for a future
  step, **not** adopted here.

Main independently re-measured SIGTERM vs SIGKILL from scratch before accepting this
(see `live_check_80.4.md` §0).

## Hypothesis

`/agents` read "Disconnected" over a healthy endpoint because `status` only became
`"connected"` inside `onMessage` — so an open stream that had not yet delivered an event
stayed `"connecting"` forever. On this system the MAS bus has published **0 events since
process start**, so open-but-silent is the *normal* case, not an edge case.

Setting status from `onopen` fixes the readout. Adding an SSE keepalive makes an idle
stream distinguishable from a dead one at the byte level. Neither may weaken the
transition *out* of green when the backend actually dies.

## Immutable success criteria (verbatim from `.claude/masterplan.json`)

Copied byte-for-byte from `.claude/masterplan.json` → step `80.4` →
`verification.success_criteria`:

1. `useEventSource registers an onopen handler that sets status='connected' on connection establishment, independent of whether any event has arrived`
2. `backend/api/mas_events.py emits an initial comment/heartbeat and a periodic keepalive so an idle stream is distinguishable from a dead one`
3. `/agents shows 'Connected' on a live backend with ZERO MAS events flowing (this is the exact failing case -- prove it with an idle system)`
4. `Killing the backend still flips the indicator to Disconnected within the existing maxFailures budget -- the fix must not make the indicator always-green. MUTATION-TEST the guard both directions.`
5. `The existing useEventSource.test.ts still passes and gains a case for the open-but-no-events state`

> **CORRECTION (cycle 2, Q/A finding 1).** An earlier revision of this block was labelled
> "verbatim" but was in fact a paraphrase. Criterion 4 was rendered as *"Stopping the
> backend still flips the indicator to Disconnected"* — which substituted **"Stopping"**
> for the masterplan's **"Killing"** and silently dropped three binding sub-clauses. The
> softening ran in the direction that eased this cycle's narrative: the entire question
> *"is `kill -9` a legitimate reading of the criterion?"* existed **only** because of the
> paraphrase. The masterplan word is **"Killing"**, so `kill -9` is the literal reading,
> not a liberty taken with it.

### Criterion 4, sub-clause by sub-clause

| Sub-clause | Evidence |
|---|---|
| *"Killing the backend still flips the indicator to Disconnected"* | `live_check_80.4.md` §D — `kill -9 79781`; page reads **"Disconnected"**. Capture `80.4_DISCONNECTED_after_sigkill.png`. |
| *"within the existing maxFailures budget"* | Same capture: *"Lost connection to MAS event stream after **5** failures."* `maxFailures` is unchanged at its existing value; the budget exhausted rather than being widened. |
| *"the fix must not make the indicator always-green"* | Mutation **M4** (`onopen` also resets `failures`+backoff) is KILLED by the flapping test — the always-green shape is guarded. Plus M1/M2/M5. |
| *"MUTATION-TEST the guard both directions"* | **Must go green when healthy:** M3 (delete `onopen`) `3 failed \| 2 passed`; M6 (`onopen` sets `"connecting"`) `2 failed \| 3 passed`. **Must leave green when dead:** M1, M2, M4, M5. Both directions covered. |

**Verification command (immutable):**
```
cd frontend && npx tsc --noEmit -p tsconfig.json && grep -n 'onopen' src/lib/hooks/useEventSource.ts && python -c "import ast; ast.parse(open('../backend/api/mas_events.py').read())"
```

**live_check (immutable):** *Playwright screenshot of `/agents` showing 'Connected' with 0
events, curl showing the heartbeat bytes on the SSE stream, and a second capture after
stopping the backend showing it correctly reads Disconnected.*

## Plan

Cycle 1 (already committed as WIP `4bcd60ad`):
1. `mas_events.py` — yield `": connected"` on connect, `": ping"` every 15s while idle,
   using `asyncio.wait({pending}, timeout=)` **not** `wait_for` (which would cancel the
   inner awaitable and silently unsubscribe the client on the first heartbeat).
2. `useEventSource.ts` — `es.onopen` sets status only; deliberately does **not** reset the
   failure budget (that would let a flapping backend stay green forever).

Cycle 2 (this cycle):
3. Re-measure criterion 4 with `kill -9`, the correct death oracle. **No product-code
   change.**
4. Add the assertion the suite was missing: the indicator must stop being green on the
   **first** error, not only after the budget exhausts. Mutation-test it.
5. Rewrite `live_check_80.4.md` with the corrected finding and an explicit record of why
   cycle 1's measurement was invalid.
6. Queue the two genuine hook defects found as their own masterplan steps — neither can
   cause a false green. **Done: `80.33`** (unmanaged reconnect timer) and **`80.34`**
   (impure `setFailures` updater), both created 2026-07-26 as `pending`.

## Do-no-harm constraints

Paper only. No `.env` edits, no flag flips, `historical_macro` frozen. Kill-switch, stops,
sector caps, DSR and PBO byte-untouched. The operator's `:8000` must not be restarted
(`phase-79.55` is an open RESTART BLOCKER) and their `:3000` must not be driven — all UI
evidence comes from the isolated skip-auth `:3100` rig with its own `distDir`.

**Hard stop:** any change that could leave the indicator green over a genuinely dead
backend.

## References

- `handoff/current/research_brief_80.4.md` (cycle 1)
- `handoff/current/research_brief_80.4b_death_detection.md` (cycle 2)
- WHATWG HTML — Server-sent events; MDN — `EventSource`, `addEventListener`
- `uvicorn.server.Server.shutdown` (read from source); Kludex/uvicorn#451
- `feedback_second_next_dev_breaks_operator_3000`, `feedback_mutation_test_guards_and_fixtures`
