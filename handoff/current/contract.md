# Contract — phase-80.2

**Step id:** `80.2` (phase-80, priority **P0**, `harness_required: true`)
**Title (masterplan, abbreviated):** *[P0 — MASKS EVERY BACKEND ERROR APP-WIDE] An
unhandled backend 500 carries NO `Access-Control-Allow-Origin` header, so the browser
blocks the response and the frontend tells the operator the backend is DOWN when it is
actually running and returning a server error.*

Date: 2026-07-25 | Wave 0 of the masterplan drain
(`handoff/current/goal_masterplan_drain_2026-07-25_DRAFT.md`).

This step goes **first** because until a 500 reaches `PerfTracker` and carries CORS, every
other defect in the drain is misreported as "backend is down" and is invisible in
`/api/observability/latency` — i.e. **I cannot measure my own subsequent fixes.**

**Tier (operator tiering directive):** **T3** — Opus 5 @ `xhigh`.
Rationale for not spending T4/Fable here: the design question was *decidable from source*
(Starlette's `build_middleware_stack`) and I resolved it with a measured 3-variant
executable probe rather than model judgment, so extra model capability buys nothing.
Fable quota is reserved for `80.27`, where the failure mode is a wrong trading verdict.
Full ledger in `experiment_results.md`.

---

## 1. Research gate summary (MANDATORY — gate must pass before this contract)

**Researcher spawn:** Agent-tool `researcher` (T3, `model: opus`), 2026-07-25.
**Artifact:** `handoff/current/research_brief.md` (686 lines, written incrementally).
**Envelope:** `gate_passed: **true**` — `external_sources_read_in_full: 8` (floor is 5),
`urls_collected: 23` (floor is 10), `recency_scan_performed: true`,
`internal_files_inspected: 14`, `coverage.audit_class: false`.
Three-query-variant discipline satisfied (current-year 2026, year-less canonical ×2,
last-2-year 2025/2026) — queries listed in the brief.

Load-bearing findings that **changed the plan**:

- **A1/A2 (from the installed source, not inferred).** `starlette 1.0.0`
  `applications.py:57-77` hardcodes `[ServerErrorMiddleware] + user_middleware +
  [ExceptionMiddleware]`, and `:62-66` routes any handler keyed on `500` **or**
  `Exception` into `ServerErrorMiddleware` — the **outermost** layer. Therefore
  **`@app.exception_handler(Exception)` cannot fix this**: its response is produced
  outside `CORSMiddleware` *and* outside `auth_and_security_middleware`. FastAPI 0.135.2
  overrides `build_middleware_stack` (`fastapi/applications.py:1018-1060`) only to add
  `AsyncExitStackMiddleware` innermost; ordering is byte-identical.
- **A3/A4 — upstream position.** Starlette docs state ServerErrorMiddleware *"is always
  the outermost middleware layer"* and name the exact defect. `Kludex/starlette#1175`
  CLOSED with no fix; `fastapi#13398` (2025-02) Kludex: **"No plans"** to document;
  `fastapi#14313` (2025-11) re-reports it with no accepted answer. The workaround
  endorsed in-thread is a middleware `try/except` around `call_next`.
  **The snippet in #13398 hardcodes `access-control-allow-origin: *` — exactly what
  criterion 4 forbids. Do not copy it.**
- **A5 — recency scan (2024-10 → 2026-03).** Starlette 0.40.0 → 1.0.0 release notes carry
  **no change** to the ordering rule. Starlette 1.0.0 (2026-03-22) removed the
  *Starlette-level* `@app.exception_handler`/`@app.middleware` decorators, but **FastAPI
  re-implements both** (`fastapi/applications.py:4600`, `:4646`), so `main.py:509` is safe
  on this pin. Nothing in the window supersedes the analysis.
- **A7 — browser strings (criterion 3).** Safari/WebKit `Load failed`; Chromium
  `Failed to fetch`; Firefox `NetworkError when attempting to fetch resource.`; axios
  `Network Error`. **Not spec-stable** — WHATWG Fetch mandates only a `TypeError` and
  leaves the message implementation-defined. Consequence: match a *set* of substrings and
  keep the generic fallback; never make correctness depend on the string alone.
- **B3.** There are **zero** existing exception handlers anywhere in `backend/` — nothing
  to duplicate or fight, and nothing to piggyback on.
- **B7 — DEBUG is OFF in the running process (measured: `/docs` → 404, `/openapi.json`
  → 404).** A debug-gated probe route would 404 and **the step's own immutable
  verification command would fail**. This directly determines the probe design below.

**Independent measurements by Main (not inherited from the step text):**

- Live middleware nesting dumped off the real `backend.main:app` object:
  `ServerErrorMiddleware → auth_and_security_middleware → CORSMiddleware →
  ExceptionMiddleware → router`. **`CORSMiddleware` is INSIDE the auth middleware**, which
  corrects the natural mental model and is why the 401 path at `:536-539` echoes the origin
  by hand.
- **Executable 3-variant design probe** (`scratchpad/probe_design.py`, mirrors the exact
  nesting) — see §4. It proves the `exception_handler` approach fails *by measurement*,
  not by reading.
- `PerfTracker.summarize()` **discards `status_code` entirely** (`perf_tracker.py:78-89`);
  `per_endpoint[ep]` is `{count, p50_ms, p95_ms}`. See §3.D — this is why an additive
  error surface is in scope.

---

## 2. Hypothesis

Because `ServerErrorMiddleware` is pinned outermost and `auth_and_security_middleware`
is registered *after* `CORSMiddleware` (and `add_middleware` inserts at index 0), an
exception escaping `call_next` at `main.py:547` unwinds past **every** layer that decorates
responses. So a single catch-all that converts the exception into a real `Response`
**at a point nested inside `CORSMiddleware`** will let the response return normally
through all outer layers, and will therefore — with **one** change — restore:

1. the CORS header on 500s (criterion 1), enforced by `CORSMiddleware` itself so the
   allow-list is untouched (criterion 4);
2. the six OWASP headers + `X-Response-Time` (`main.py:559-567`); and
3. the `PerfTracker.record(...)` call at `main.py:552-558`.

Criterion 2 then follows with **no frontend change**, because `api.ts:161-163` already
throws ``Server error on ${path}. Check the backend logs for details.`` — it is merely
unreachable today. Criterion 3 is a separate one-line additive edit at `api.ts:125`.

---

## 3. Immutable success criteria — **copied verbatim from `.claude/masterplan.json`**

> 1. A 500 response from the backend INCLUDES access-control-allow-origin for an allowed origin (verify against a route that genuinely raises, not a synthetic 204) -- so the browser surfaces the status instead of a network failure
> 2. With a deliberately-broken endpoint, the UI shows a server-error message that does NOT claim the backend is unreachable
> 3. api.ts network-error detection also matches Safari's 'Load failed'
> 4. The CORS allow-list behaviour is UNCHANGED for disallowed origins -- a 500 to a non-allowed origin must still omit the header (do not fix this by echoing '*')

**Immutable verification command** (verbatim):

```
curl -s -D - -o /dev/null -H 'Origin: http://localhost:3000' http://localhost:8000/api/__force_500_probe 2>&1 | grep -i 'access-control-allow-origin'
```

**Immutable `live_check`** (verbatim): `handoff/current/live_check_80.2.md`: paired curl
-D- output for a 500 and a 200 on the same origin showing the header now present on both,
plus a Playwright console capture showing no CORS-block error.

### 3.D — the ADDITIONAL success criteria carried in the step's ADDENDUM

The step body (VERIFIED-BY-WORKFLOW addendum) adds, verbatim:

> ADDITIONAL SUCCESS CRITERIA: a deliberately-raising route must yield a 500 that (i)
> carries the CORS header, (ii) carries nosniff, and (iii) produces a `PerfTracker` record
> with status_code 500 that is **visible in** `GET /api/observability/latency`.

**Scope decision, stated explicitly rather than assumed** (research brief §B6 flagged this
as needing an explicit call): `summarize()` does not filter by status, so a 500 record
lands in `total_requests`/`p50`/`p95` **immediately** once the middleware tail runs — but
`status_code` is discarded, so **no error signal is derivable** from
`/api/observability/latency` even after the middleware fix. A bare count bump does not
satisfy "a record **with status_code 500** that is visible". Therefore this step **also**
makes the error count visible, as a strictly **additive** change to `summarize()` and to
the `/api/observability/latency` payload. Blast radius is checked in §6.

**This is an addition to what I deliver, never a relaxation of the four immutable
criteria.** The four above are not amended, reworded, or reinterpreted.

---

## 4. Plan

**Chosen approach: C1** — a **pure-ASGI** catch-all registered *before*
`app.add_middleware(CORSMiddleware, ...)` so it nests **inside** CORS.

Ranked against the alternatives (research brief §C), and then **measured** rather than
argued. `scratchpad/probe_design.py` builds three apps with pyfinagent's exact nesting and
a genuinely-raising route:

| variant | 500 CORS hdr | 500 `nosniff` | recorded by auth mw | disallowed origin |
|---|---|---|---|---|
| baseline (today) | `None` | `None` | *(none)* | `None` ✓ |
| `@app.exception_handler(Exception)` | **`None`** | `None` | *(none)* | `None` ✓ |
| **catch-all nested inside CORS** | `http://localhost:3000` | `nosniff` | `('/boom', 500)` | **`None` ✓** |

C3 (`app = CORSMiddleware(app, ...)`, the official-docs answer) is rejected: it fixes CORS
only, leaves PerfTracker + OWASP still skipped, and rebinds `app` to a non-FastAPI object,
breaking `from backend.main import app` (used by every test module and `authed_test_client`).
C2 (try/except inside the auth middleware) works but needs a manual origin echo — a
**third** copy of `_TAILSCALE_ORIGIN_RE`, which is precisely the drift phase-75.1 closed.
C1 adds **no** copy: the allow-list decision stays entirely inside
`CORSMiddleware.is_allowed_origin`.

### Steps

1. **`backend/middleware/catch_all_errors.py`** (new package — `backend/middleware/` does
   not exist today). A pure-ASGI class mirroring `starlette/middleware/errors.py:154-186`:
   - pass non-`http` scopes straight through;
   - wrap `send` and track `response_started`;
   - `except Exception` → `logger.exception(...)` with method + path, then emit
     `JSONResponse(500, {"detail": ...})` **only if not `response_started`**; if the
     response already started (mid-stream failure, e.g. the SSE route at
     `mas_events.py:36`) **re-raise** — a partially-sent response cannot be rewritten, and
     the code comment says so rather than pretending it is covered;
   - explicitly render `HTTPException`/`StarletteHTTPException` with **its own**
     status code, never 500 (defensive: `ExceptionMiddleware` is nested inside and
     normally handles these first, but this makes the "catch-all downgrades a 401 to a
     500" trap structurally impossible);
   - never place a traceback in the body (phase-75.16 established traceback leakage to
     HTTP callers as a defect class here).
2. **`backend/main.py`** — register it *before* the `CORSMiddleware` call, with an inline
   comment explaining the counter-intuitive ordering (`add_middleware` inserts at index 0,
   so **earlier registration = more inner**). The ordering is invisible by inspection, so
   it also gets an executable assertion (step 7).
3. **`backend/main.py`** — add `GET /api/__force_500_probe`: unconditional (DEBUG is OFF —
   §1/B7), **auth-gated** (deliberately *not* added to `_PUBLIC_PATHS`, which would need a
   `.claude/rules/security.md` row and widen the unauthenticated surface for zero gain),
   `include_in_schema=False`, raising a self-identifying `RuntimeError`. This is what makes
   criterion 1 checkable against "a route that genuinely raises, not a synthetic 204" —
   and it stays valid after 80.1 removes the `/api/signals/AAPL` 500.
4. **`backend/services/perf_tracker.py` + `backend/api/observability_api.py`** — additive
   error visibility (§3.D): count non-2xx/3xx per endpoint and overall, exposed as new
   keys. **No existing key is renamed, removed, or changed in meaning.**
5. **`frontend/src/lib/api.ts:125`** — extend the network-error substring set with Safari's
   `Load failed` (plus Firefox's full string and axios's `Network Error`), keeping the
   generic fallback at `:131` because the strings are not spec-stable (A7).
6. **`frontend/src/lib/types.ts`** — additive optional fields on `EndpointLatency` /
   `PerfSummary` to match step 4.
7. **Tests — `backend/tests/test_phase_80_2_error_response_contract.py`**, end-to-end
   through the real stack (never a source scan — `test_phase_75_deploy_surface.py:397-401`
   is the vacuous-guard shape to avoid):
   - 500 from the probe route + allowed Origin → has `access-control-allow-origin`;
   - **same 500 + `Origin: https://evil.example` → header ABSENT** (criterion 4);
   - 500 carries `x-content-type-options: nosniff`;
   - the 500 produces a `PerfTracker` entry with `status_code == 500`, and that shows as a
     non-zero error count in `/api/observability/latency`;
   - `logger.exception` actually fires (caplog) — the swallowed-traceback mitigation is
     **asserted, not assumed**;
   - a 401 route still returns 401 and a 404 still returns 404 (the inverse trap);
   - the middleware **registration order** is asserted executably.
   Requires `authed_test_client(app, raise_server_exceptions=False)` — `TestClient`
   defaults to re-raising, and sends no `Origin` header unless one is passed.
8. **MUTATION-TEST every guard** (`feedback_mutation_test_guards_and_fixtures`): revert
   each change in turn and confirm the corresponding assertion **FAILS**. A guard that
   cannot fail does not count. Mutating the *stub* is included.
9. **Restart the backend** (middleware is built once at startup —
   `applications.py:88-90`), then capture `live_check_80.2.md` **after** the restart, or
   the evidence shows the old behaviour.
10. **Playwright capture** on the isolated skip-auth `:3100` rig (never the operator's
    `:3000`), showing the console has **no CORS-block error** and the UI surfaces a server
    error rather than "backend unreachable". Restore `tsconfig.json` + `next-env.d.ts`
    afterwards.

---

## 5. Out of scope — queued, not silently fixed

Per `feedback_queue_discovered_defects_in_masterplan`, these get their **own**
research-gated masterplan steps and are **not** drive-by edits inside 80.2:

- **Three components bypass `apiFetch` entirely**, so they get none of this error handling:
  `ResearchInvestigator.tsx:33` (`/api/investigate`), `Sidebar.tsx:155` (`/api/changelog`),
  `StockChart.tsx:94` (`/api/charts/{ticker}`).
- **Doc/code drift:** `.claude/rules/security.md` "OWASP Headers" says
  `X-XSS-Protection: 1; mode=block`; `backend/main.py:564` sets `0` (the modern-correct
  value). Pre-existing, **not caused by this step**. I am not touching that header block,
  so I am not reconciling the doc line here — flagged so the diff cannot be read as
  implying the rule was already aligned.

Also explicitly out of scope: the unconditional
`access-control-allow-credentials: true` on disallowed origins (it lives in
`cors.py` `simple_headers`). Criterion 4 is about `access-control-allow-origin` **only**;
changing the credentials header would be an unrequested behaviour change.

---

## 6. DO-NO-HARM

- **The live book does not move.** No `.env` edits, no flag flips, no optimizer runs
  (`historical_macro` stays FROZEN). Kill-switch limits, stops, sector caps, DSR and PBO
  are byte-untouched. Nothing here touches `paper_trader.py`, the scheduler, or any order
  path.
- **Auth cannot change.** C1 nests *inside* `CORSMiddleware`, which is inside the auth
  middleware, so it runs **after** auth has already decided — it can never convert a
  401/403 into a 500. The inverse trap (a catch-all swallowing `HTTPException`) is closed
  structurally by the explicit `HTTPException` branch **and** regression-tested.
- **Allow-list unchanged.** No new copy of `_TAILSCALE_ORIGIN_RE`; the decision stays in
  `CORSMiddleware.is_allowed_origin`. Never `*`.
- **No exception is silently swallowed.** `logger.exception` replaces the traceback
  `ServerErrorMiddleware`'s `raise exc` currently feeds to `backend.log` — asserted by
  caplog, not assumed.
- **Streaming is not regressed.** `mas_events.py:36` serves SSE; a mid-stream failure has
  already sent headers, so the catch-all re-raises rather than corrupting the stream.
  Verified live after restart.
- **`perf_optimizer` (`:51,:83`) will now see 500-latency samples in its p95.** It tunes
  **cache TTLs only**, never trade parameters — so this is a measurement improvement, not
  a book risk. Stated because it is a real behaviour change, not hidden.
- **Consumer-contract check before flip:** every reader of `summarize()` and of
  `/api/observability/latency` is enumerated and confirmed additive-safe
  (`observability_api.py:75`, `performance_api.py:34,40`, `perf_optimizer.py:51,83`,
  `autonomous_loop.py:1579`, `settings/page.tsx:1360`, `types.ts:958-966`).
- **`git add -An` before the flip** (`feedback_audit_the_commit_not_the_diff`) — the
  auto-commit hook stages the whole tree under this step's name.

---

## 7. Evidence to produce

| Artifact | Content |
|---|---|
| `handoff/current/experiment_results.md` | files changed + verbatim verification output + the mutation matrix + the tier ledger |
| `handoff/current/live_check_80.2.md` | paired `curl -D-` 500/200 on the same origin **after restart**, the disallowed-origin negative, the `/api/observability/latency` row, and the Playwright console capture |
| `handoff/current/evaluator_critique.md` | verbatim Q/A verdict (Main transcribes, never authors) |
| `handoff/harness_log.md` | cycle block, appended **before** the status flip |

## 8. References

- `handoff/current/research_brief.md` (the gate artifact; 8 sources in full, 23 URLs)
- Installed `starlette/applications.py:57-77`, `:62-66`; `starlette/middleware/errors.py:154-186`;
  `starlette/middleware/cors.py:87-105,156-174`; `fastapi/applications.py:1018-1060,4600,4646`
- `https://raw.githubusercontent.com/Kludex/starlette/main/docs/middleware.md`
- `https://github.com/Kludex/starlette/issues/1175`,
  `https://github.com/fastapi/fastapi/discussions/13398`,
  `https://github.com/fastapi/fastapi/discussions/14313`
- `https://raw.githubusercontent.com/Kludex/starlette/main/docs/release-notes.md`
- `https://trackjs.com/javascript-errors/load-failed/`,
  `https://developer.apple.com/forums/thread/771127`
- `.claude/rules/security.md` (OWASP Headers / CORS), `.claude/rules/backend-api.md:53`
- `scratchpad/probe_design.py` (the executable 3-variant design probe)
