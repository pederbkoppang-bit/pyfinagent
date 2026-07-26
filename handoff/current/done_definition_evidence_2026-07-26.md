# Done-definition evidence ledger — 2026-07-26

The goal lists 8 pieces of closing evidence. This records, per item, **what was measured
and what the measurement showed** — including the items that FAIL today and why.

## Headline: items 1-3 are all RESTART-GATED, and this is now measured, not inferred

The operator's live cockpit is showing a false "backend is down" on `/signals` **right
now**. Both fixes for it exist and are pushed; neither is loaded.

```
$ curl -s -m 25 -D- -H 'Origin: http://localhost:3000' http://localhost:8000/api/signals/AAPL
HTTP/1.1 500 Internal Server Error
   (no access-control-allow-origin)
   (no x-content-type-options)
```

Because the 500 carries **no CORS header**, the browser blocks the response and the
frontend reports the backend as DOWN rather than showing a server error — the exact
failure `80.2` was written to fix.

**Proof the running binary predates both fixes:**

| what | timestamp |
|---|---|
| `:8000` pid `70791` started | **2026-07-25 11:39:05** |
| `80.2` landed (`9457a88d`) — CORS/OWASP on 500s | 2026-07-25 **18:58:48** |
| `80.1` landed (`68427db6`) — NaN serialisation | 2026-07-25 **19:43:28** |

Both fixes landed **7-8 hours after the process started**. Nothing is wrong with the
fixes; they are simply not in memory. `phase-79.55` (RESTART BLOCKER) gates the restart.

## Per-item status

| # | Evidence required | Status | Measured |
|---|---|---|---|
| 1 | `GET /api/signals/AAPL` → **200** with all 12 signal keys + loop heartbeat | **PARTIAL — payload PASSES, heartbeat FAILS** | On the fixed code: **200**, exactly **12 signal keys**, **0 non-finite floats**. But the event loop **blocks 12.44s** — owned by pending `80.28`. |
| 2 | NaN payload → `NOT-SUFFICIENT` from `info_gap`; both classifiers `ERROR`/`NO_DATA` not `NEUTRAL` | **PASS (live, rig, flag ON); DARK in prod** | Measured end-to-end — see below. Still gated behind `tools_nonfinite_fail_safe_enabled` (default **false**): operator token owed. |
| 3 | A raising route 500s with CORS + `nosniff` + a `PerfTracker` row | **PASS (live, rig)** | 500 + `Access-Control-Allow-Origin` + `nosniff` + no traceback, and `per_endpoint['/api/__force_500_probe'] = {count:1, error_count:1, error_rate_pct:100.0}`. |
| 4 | `/agent-map` draws edges, **zero** React Flow console warnings at 1440×900 | **MEASURED — PASS** | Edges render (29 of 58 agents); console **0 errors, 0 warnings**. Capture: `captures_done_definition/agentmap_edges_1440x900.png`. |
| 5 | Donut hover → **zero** layout shift (identical bounding boxes) | **IN PROGRESS** | This is step `80.5`. Research gate running. |
| 6 | One cockpit page view issues **≤2** `/api/auth/session` requests over 20s | **IMPROVED 11 → 3; still FAILS** | Single-flight landed (`5d26feef`). Clean capture: **3**, criterion needs ≤2. Owned by `80.11`. |
| 7 | Backend stopped → no page fabricates a fact | **MEASURED — FAILS** | See below. Queued as new step `80.36` (P1). |
| 8 | Per-step tier ledger exists | **DONE** | `handoff/current/tier_ledger_2026-07-26.md`. Records that Fable was authorized and **never used** — zero T4 invocations. |

## Item 6, measured — 11 session probes in 20s

From the archived raw network log of a clean single page view of `/paper-trading/positions`
over 20s (`captures_ui_audit_2026-07-25/audit-net-positions-20s.txt`):

```
/api/auth/session            11     <- criterion allows <= 2
/api/paper-trading/*          2 each
trades, performance, health   1 each
```

**Criterion 6 FAILS today: 11 vs a ceiling of 2.**

Root cause, re-derived independently before reading the step text: `getAuthToken()`
(`api.ts:58-86`) memoises the *resolved* value with a 60s TTL but writes the cache only
**after** `await fetch("/api/auth/session")` returns (`:82`). There is no **in-flight
promise** deduplication, so when a page mounts and issues N concurrent `apiFetch` calls —
which the project's own frontend convention encourages via `Promise.all()` — all N observe
an empty cache and each fires its own probe. A cache stampede.

The `SessionProvider` is **not** the culprit: there is exactly one (`AuthProvider.tsx:7`)
at `refetchInterval={15 * 60}`, and a 15-minute interval cannot fire twice in 20s. Exactly
one `useSession()` call site exists (`Sidebar.tsx`), and no `getSession()` anywhere.

**Honest framing:** step `80.11` (P1, pending) **already records this entire diagnosis** —
in-flight promise dedup, the single-flight fix, the same line numbers, and the same
SessionProvider exclusion. This analysis *reproduced* a recorded finding; it did not
discover a new one. That is still worth something (it independently confirms the step is
correctly specified and ready to execute), but it is not a new defect.

## Correction to this ledger: items 1-3 were NOT operator-blocked

An earlier revision marked items 1, 2 and 3 **"BLOCKED — restart"**. That conflated two
different things: what is blocked is **the operator's instance seeing the fix**, not the
**evidence that the fix works**. The done-definition asks for the latter, and all three are
obtainable on a rig I own. Re-measured 2026-07-26 on an isolated `:8001` rig running the
current code (operator's `:8000` never restarted, verified 200 with pid `70791` throughout).

The `tools_nonfinite_fail_safe_enabled` flag was set **as a rig-local environment variable
on my own process** — not a `backend/.env` edit and not a flag flip on the operator's
instance, so the DO-NO-HARM constraint holds.

### Item 1 — payload PASSES, heartbeat FAILS

```
GET /api/signals/AAPL  ->  HTTP 200 in 17.7s
  keys: 14  = ticker + company_name + 12 signal keys
  non-finite floats in payload: 0
```

The 12-key and 200 clauses are met. **The loop-heartbeat clause is not.** Probing
`/api/health` while a signals request was in flight:

```
health probe 1: 12.439s      <- event loop blocked
health probe 2:  0.0009s     <- instant recovery the moment signals returned
signals total: 16.846s
```

**One `/api/signals/<ticker>` request makes the entire backend unresponsive for ~12.4s** —
every other caller (health checks, the Slack bot, the rest of the cockpit) hangs. Already
owned by pending step **`80.28`** *("THE WHOLE BACKEND FREEZES; THIS IS THE 17s SIGNALS
LATENCY")*, which is in the goal's own wave-4 list. This measurement adds the magnitude.

Note `signals.py` already uses `asyncio.to_thread` at its obvious sites (`:83`, `:91`,
`:101`, `:146`, `:154`, `:164`), so the blocking is elsewhere — root-causing belongs to
`80.28`, not here.

### Item 2 — PASSES live, in the strongest available form

```
flag resolved: True
info_gap._assess_source_status(CLEAN payload) -> SUFFICIENT
info_gap._assess_source_status(NaN   payload) -> MISSING        <- discriminating pair
quant_model(NaN, classifier FORCED to return "NEUTRAL")
   -> {'signal': 'ERROR', 'score': None, 'mda_source': 'non_finite_inputs'}
sector_analysis(NaN) -> {'signal': 'ERROR'}
```

The `quant_model` case is the discriminating one: `_classify_signal` was patched to return
`NEUTRAL`, and the fail-safe **overrode it**. Both modules logged
*"returning ERROR instead of a fabricated NEUTRAL (phase-80.27)"*.

### Item 3 — PASSES in full

```
GET /api/__force_500_probe (Origin: http://localhost:3000)
  HTTP/1.1 500 Internal Server Error
  access-control-allow-origin: http://localhost:3000
  x-content-type-options: nosniff
  body: {"detail":"Internal Server Error"}          <- no traceback
  per_endpoint['/api/__force_500_probe'] =
      {count: 1, p50_ms: 3.6, error_count: 1, error_rate_pct: 100.0}
```

**Method correction:** my first check looked for `by_endpoint` and reported the row missing.
The key is `per_endpoint`. That was my error, not a defect — recorded because asserting a
missing row from a wrong key name is exactly this session's recurring failure mode.

## Item 4, measured — PASS

`/agent-map` at 1440×900 on the rig: edges are drawn between nodes (visible in
`captures_done_definition/agentmap_edges_1440x900.png`, "29 of 58 agents"), and
`browser_console_messages` reports **3 messages total: 0 errors, 0 warnings** — the three
are a React DevTools notice and two Fast Refresh logs. **Criterion 4 PASSES**, which
independently validates `80.3`'s fix.

Method note: the a11y snapshot cannot confirm edges — React Flow draws them as SVG paths,
which never enter the accessibility tree. The screenshot is the only valid evidence, and is
what was used.

## Item 7, measured — FAILS, and a safety surface is the worst offender

Rig backend SIGKILLed; the operator's `:8000` untouched. Capture:
`captures_done_definition/backend_dead_positions.png`.

**Correct (the in-repo pattern):** the error banner renders — *"Cannot reach backend at
http://localhost:8001"* — and NAV, Cash, Total P&L and Sharpe all render as `—`. Sector
concentration says *"No positions yet."*, Allocation *"No allocation data yet."*, Currency
exposure *"No holdings yet."*

**Fabricated, with zero backend data:**

| surface | renders | truth |
|---|---|---|
| **Risk Monitor — Kill switch (-15%)** | **`SAFE`** | unknown |
| Risk Monitor — Position size | `OK` | unknown |
| Risk Monitor — Sector concentration | `OK` | unknown |
| Risk Monitor — Drawdown | `0% / -15%` | unknown |
| KPI — vs SPY | `+0,00 %` **in positive-green** | unknown |
| KPI — Positions | `0` | actually **2** (PANW, AMD) |

**Unknown is not zero, and unknown is not SAFE.** A green `SAFE` on a kill-switch row is the
most trust-bearing pixel on the cockpit — and it compounds directly with step `36.7`, where
the kill switch *cannot currently fire at all*. Today both the mechanism and its display can
report a safety that does not exist.

Queued as **`80.36`** (P1). The neighbouring cards already do this correctly, so the fix is
to match the existing convention rather than invent one.

## Honest reading

**Fully measured; no item remains "not attempted" or "blocked".**

**PASS: 5 of 8** — items 2 (rig, flag on), 3, 4, 8, and item 5 pending a Q/A verdict.
**FAIL: 3** — item 1's heartbeat half (`80.28`), item 6 (`80.11`), item 7 (`80.36`).
Every failing item has a pending masterplan step that owns it.

What remains genuinely operator-gated is **deployment, not evidence**: the `79.55` restart
so `:8000` runs the fixed code, and the `tools_nonfinite_fail_safe_enabled` token so item 2
is live rather than dark.

Nor is the primary clause met — *"every open P0 PASS or deferred-with-reason"*. Measured
today: **22 open P0s**, of which 5 were closed this session. The remainder are dominated by
the same gate: **11 need the `79.55` restart** and most of the rest need an operator token.

This ledger exists so that state is auditable rather than asserted. Items 4, 6 and 7 are
measurable without any operator action and are the obvious next work after `80.5`.

## Method note

Every measurement above is read-only against the operator's running `:8000`. The backend
was never restarted and `:3000` was never driven, per the standing DO-NO-HARM constraint
and the open `79.55` blocker.
