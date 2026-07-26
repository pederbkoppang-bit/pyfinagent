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
| 4 | `/agent-map` draws edges, **zero** React Flow console warnings at 1440×900 | **not re-measured this session** | `80.3` closed with a Playwright BEFORE/AFTER capture. The zero-warnings clause was not independently re-measured today. |
| 5 | Donut hover → **zero** layout shift (identical bounding boxes) | **IN PROGRESS** | This is step `80.5`. Research gate running. |
| 6 | One cockpit page view issues **≤2** `/api/auth/session` requests over 20s | **not attempted** | No step in this drain targeted it. |
| 7 | Backend stopped → no page fabricates a fact | **not attempted** | Requires driving a rig with its backend killed; not run. |
| 8 | Per-step tier ledger exists | **DONE** | `handoff/current/tier_ledger_2026-07-26.md`. Records that Fable was authorized and **never used** — zero T4 invocations. |

## Honest reading

**The done-definition is NOT satisfied.** 5 of 8 evidence items are unmet: two blocked on
the operator's restart, one on an operator flag token, and three (4, 6, 7) simply were not
attempted in this drain.

Nor is the primary clause met — *"every open P0 PASS or deferred-with-reason"*. Measured
today: **22 open P0s**, of which 5 were closed this session. The remainder are dominated by
the same gate: **11 need the `79.55` restart** and most of the rest need an operator token.

This ledger exists so that state is auditable rather than asserted. Items 4, 6 and 7 are
measurable without any operator action and are the obvious next work after `80.5`.

## Method note

Every measurement above is read-only against the operator's running `:8000`. The backend
was never restarted and `:3000` was never driven, per the standing DO-NO-HARM constraint
and the open `79.55` blocker.
