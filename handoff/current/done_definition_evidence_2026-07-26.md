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
| 1 | `GET /api/signals/AAPL` → **200** with all 12 signal keys + loop heartbeat | **BLOCKED — restart** | Returns **500** in 19.1s on the live backend (pre-`80.1` binary). Verified 200 on the `80.1` rig at close of that step. |
| 2 | NaN payload → `NOT-SUFFICIENT` from `info_gap`; both classifiers `ERROR`/`NO_DATA` not `NEUTRAL` | **PASS on rig; DARK in prod** | 30 tests in `test_phase_80_27_nonfinite_fail_safe.py`. Gated behind `tools_nonfinite_fail_safe_enabled` (default **false**) — operator flag token owed. |
| 3 | A raising route 500s with CORS + `nosniff` + a `PerfTracker` row | **BLOCKED — restart** | Live 500 above carries **neither** header. Verified on the `80.2` rig via `/api/__force_500_probe` at close of that step. |
| 4 | `/agent-map` draws edges, **zero** React Flow console warnings at 1440×900 | **MEASURED — PASS** | Edges render (29 of 58 agents); console **0 errors, 0 warnings**. Capture: `captures_done_definition/agentmap_edges_1440x900.png`. |
| 5 | Donut hover → **zero** layout shift (identical bounding boxes) | **IN PROGRESS** | This is step `80.5`. Research gate running. |
| 6 | One cockpit page view issues **≤2** `/api/auth/session` requests over 20s | **MEASURED — FAILS (11)** | See below. Owned by pending step `80.11`. |
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

**The done-definition is NOT satisfied**, but the picture is now fully measured rather than
partly unknown. **3 of 8 pass** (4, 5-pending-verdict, 8). Unmet: items 1 and 3 blocked on
the operator's restart; item 2 on a flag token; items 6 and 7 **measured and failing**, each
with a queued step (`80.11`, `80.36`).

No item remains "not attempted".

Nor is the primary clause met — *"every open P0 PASS or deferred-with-reason"*. Measured
today: **22 open P0s**, of which 5 were closed this session. The remainder are dominated by
the same gate: **11 need the `79.55` restart** and most of the rest need an operator token.

This ledger exists so that state is auditable rather than asserted. Items 4, 6 and 7 are
measurable without any operator action and are the obvious next work after `80.5`.

## Method note

Every measurement above is read-only against the operator's running `:8000`. The backend
was never restarted and `:3000` was never driven, per the standing DO-NO-HARM constraint
and the open `79.55` blocker.
