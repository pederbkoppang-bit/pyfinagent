# Experiment Results — phase-80.2

**Step:** `80.2` (P0) — an unhandled backend 500 carried no CORS header, so the browser
blocked it and the frontend told the operator the backend was DOWN.
Date 2026-07-25. Contract: `handoff/current/contract.md`. Gate: `research_brief.md`
(`gate_passed: true`, 8 sources in full, 23 URLs).

---

## 1. What was built

**Root cause, confirmed from the installed source rather than inferred:** starlette 1.0.0
`applications.py:57-77` hardcodes `[ServerErrorMiddleware] + user_middleware +
[ExceptionMiddleware]`, and `:62-66` routes any handler keyed on `500`/`Exception` into
that **outermost** layer. So an escaping exception unwinds past `CORSMiddleware` **and**
past `auth_and_security_middleware`, whose entire tail (`main.py:548-569` — PerfTracker
record, `X-Response-Time`, six OWASP headers) is skipped.

**The intuitive fix does not work.** A 3-variant executable probe
(`scratchpad/probe_design.py`) reproduced pyfinagent's exact nesting:

| variant | 500 CORS hdr | 500 `nosniff` | recorded by auth mw | disallowed origin |
|---|---|---|---|---|
| baseline (today) | `None` | `None` | *(none)* | `None` ok |
| `@app.exception_handler(Exception)` | **`None`** | `None` | *(none)* | `None` ok |
| **catch-all nested INSIDE CORS** | `http://localhost:3000` | `nosniff` | `('/boom', 500)` | **`None` ok** |

So the fix is a pure-ASGI catch-all **registered before `add_middleware(CORSMiddleware)`**
— because `add_middleware` inserts at index 0, earlier registration = further in. The 500
becomes an ordinary response that travels back out through every decorating layer, which
closes all three consequences with one change.

### Files

| File | Δ | What |
|---|---|---|
| `backend/middleware/__init__.py` | **new** | new package |
| `backend/middleware/catch_all_errors.py` | **new**, 149 L (`wc -l`, measured) | `CatchAllServerErrorMiddleware`: pure ASGI, wraps `send` + tracks `response_started`, `logger.exception`, JSON body with no traceback, explicit `HTTPException` branch that keeps the original status |
| `backend/main.py` | +48 | import; `add_middleware(CatchAllServerErrorMiddleware)` **above** the CORS block with a comment explaining the counter-intuitive ordering; `GET /api/__force_500_probe` (`include_in_schema=False`, auth-gated, always raises) |
| `backend/services/perf_tracker.py` | +26/−6 | `summarize()` now also reports `error_count` / `error_rate_pct`, overall and per-endpoint. **Additive only** — no existing key renamed, removed, or changed in meaning |
| `backend/api/observability_api.py` | +9 | passes the two new fields through, incl. the fail-open branch |
| `frontend/src/lib/api.ts` | +28/−1 | extracted `NETWORK_FAILURE_MESSAGES` + `isNetworkFailureMessage()`; added Safari `Load failed`, Firefox's full string, axios `Network Error`; generic fallback kept |
| `frontend/src/lib/types.ts` | +6 | optional `error_count` / `error_rate_pct` on `EndpointLatency` + `PerfSummary` |
| `backend/tests/test_phase_80_2_error_response_contract.py` | **new**, 18 tests | end-to-end through the real stack |
| `frontend/src/lib/api.network-errors.test.ts` | **new**, **13 tests** | 7 on the network-string predicate, + 6 added in cycle 2 that bind the real `apiFetch` branch to it (§3.1) |

`git diff --stat`: `5 files changed, 111 insertions(+), 6 deletions(-)` + 4 new files.

### Why the probe route is permanent and not debug-gated

Measured: `DEBUG` is **off** in the running process (`/docs` and `/openapi.json` both
404). A debug-gated probe would 404 — and a 404 **carries the CORS header** (it is an
`HTTPException` handled by the innermost `ExceptionMiddleware`), so the step's own
`grep`-based verification command would have silently passed on the wrong response. It is
auth-gated (deliberately **not** added to `_PUBLIC_PATHS`), reads nothing, writes nothing,
touches no trading state. Kept permanently because `/api/signals/AAPL` stops being a 500
once 80.1 lands, and a contract with no fixture rots.

---

## 2. Verification output (verbatim)

### 2.1 Syntax gate

```
OK  backend/main.py
OK  backend/middleware/catch_all_errors.py
OK  backend/services/perf_tracker.py
OK  backend/api/observability_api.py
```

### 2.2 New backend suite

```
$ .venv/bin/python -m pytest backend/tests/test_phase_80_2_error_response_contract.py -q
..................                                                       [100%]
18 passed, 1 warning in 2.13s
```

### 2.3 Adjacent suites — no regressions

```
$ .venv/bin/python -m pytest tests/api/test_observability.py \
      backend/tests/test_phase_75_deploy_surface.py -q
49 passed, 1 warning in 7.91s
```

### 2.4 Frontend

```
$ npx tsc --noEmit
[tsc exit=0]

$ npx vitest run src/lib/
 Test Files  8 passed (8)
      Tests  49 passed (49)
```

### 2.5 The immutable verification command (live, `:8001` rig — see live_check §A)

```
$ curl -s -D - -o /dev/null -H 'Origin: http://localhost:3000' \
       .../api/__force_500_probe | grep -i 'access-control-allow-origin'
access-control-allow-origin: http://localhost:3000
```

with the status line confirmed **500** (not 404, not 401) in the full capture.

---

## 3. Mutation matrix — 9/9 guards held, 0 vacuous

`feedback_mutation_test_guards_and_fixtures`: a guard that cannot fail does not count.
Each mutation was applied to the real file, the guard run, and the file restored from an
in-memory snapshot (never `git stash` — hooks append to tracked audit logs).
Driver: `scratchpad/mutate_80_2.py`.

| # | Mutation | File | Guard | Result |
|---|---|---|---|---|
| M1 | Register the catch-all **after** `CORSMiddleware` → it nests outside CORS (the silent-revert mode) | `main.py` | cors + owasp + perftracker + order | **FAILED as required** |
| M2 | Remove the catch-all entirely (today's behaviour) | `main.py` | cors + owasp + latency-visibility | **FAILED as required** |
| M3 | Drop `logger.exception` | `catch_all_errors.py` | `test_unhandled_exception_is_logged_with_traceback` | **FAILED as required** |
| M4 | Count every request as an error (`>= 500` → `>= 0`) | `perf_tracker.py` | `test_successful_requests_do_not_count_as_errors` | **FAILED as required** |
| M5 | Stop exposing `error_count` on the latency route | `observability_api.py` | `test_500_is_visible_as_an_error_in_observability_latency` | **FAILED as required** |
| M6 | "Fix" CORS by echoing `*` (criterion 4 violation) | `catch_all_errors.py` | disallowed-origin + no-wildcard | **FAILED as required** |
| M7 | Drop the `HTTPException` branch → a 401 becomes a 500 | `catch_all_errors.py` | `test_http_exception_is_rendered_with_its_own_status_code` | **FAILED as required** |
| **M9** | **STUB MUTATION** — break the *fixture*: probe route returns 200 instead of raising | `main.py` | `test_probe_route_genuinely_raises_a_500` | **FAILED as required** |
| M8 | Remove Safari's `Load failed` from the string set | `api.ts` | `api.network-errors.test.ts` | **FAILED as required** |

`9/9 guards held; 0 vacuous.` Working tree verified byte-identical after the run.

### 3.1 CYCLE 2 — the mutation my matrix did NOT run, authored by Q/A (finding 5.2)

**M8 was vacuous at the defect site and I missed it.** M8 mutates the exported
`NETWORK_FAILURE_MESSAGES` array — which the unit test imports *directly* — so it cannot
distinguish a predicate that is **correct** from one that is **wired in**. Q/A mutated the
wiring instead, re-introducing the exact operator-visible bug:

```diff
-    if (isNetworkFailureMessage(msg)) {
+    if (msg.includes("Failed to fetch") || msg.includes("NetworkError")) {
```

and **everything stayed green**: `api.network-errors.test.ts` 7/7, `src/lib` 49/49,
`tsc --noEmit` exit 0. Sole-coverage vacuity on a behavioural criterion.

**Fix (condition C1):** `apiFetch` is module-private, so the new tests drive it through an
exported caller (`listReports`) with a stubbed `fetch` that rejects
`new TypeError("Load failed")` — exercising the real branch rather than the helper in
isolation. Six new cases: the three engine strings → `Cannot reach backend`; an assertion
that the raw `Network error calling ...` fallback is **not** reached; an unrecognised
rejection that **does** fall through (correctness must not depend on the string set); and
a resolving 500 that must take the `!res.ok` path.

**M10 — re-running Q/A's exact mutation against the new guard:**

```
$ npx vitest run src/lib/api.network-errors.test.ts        # with the call site reverted
- Expected: /Network error calling/
+ Received: "Network error calling /api/reports/?limit=20: Load failed"
  ❯ src/lib/api.network-errors.test.ts:98:32

 Test Files  1 failed (1)
      Tests  2 failed | 11 passed (13)
```

**GUARD NOW HOLDS.** The received value is literally the operator's screenshot text. File
restored; `md5 frontend/src/lib/api.ts` → `a51fe1fc07f6cf106deee69be1121d71`, matching the
value Q/A recorded after its own restore. 13/13 green again.

Running total: **10/10 guards held, 0 vacuous** — with the honest note that one of them
existed only because the evaluator found the hole, which is the evaluator gate working.

**Deliberately avoided anti-pattern:** `backend/tests/test_phase_75_deploy_surface.py:397-401`
is a *source-scan* CORS test (`assert "...'*'" not in src`). A source scan cannot see
middleware ordering — which is the entire bug — so it would have been unfailable here.
Every assertion in the new suite drives a real request through the real stack.

---

## 4. Criteria → evidence

| # | Criterion (verbatim) | Evidence | Status |
|---|---|---|---|
| 1 | 500 INCLUDES access-control-allow-origin for an allowed origin (a route that genuinely raises, not a synthetic 204) | live_check §C2 — status line `500`, header present. `test_probe_route_genuinely_raises_a_500` blocks the 404/401 false pass (§F) | **MET** |
| 2 | With a deliberately-broken endpoint the UI shows a server-error message that does NOT claim the backend is unreachable | live_check §G — rendered: `Server error on /api/signals/AAPL. Check the backend logs for details.` | **MET** |
| 3 | api.ts network-error detection also matches Safari's `Load failed` | `api.network-errors.test.ts` **13/13**, of which the binding evidence is the `apiFetch network-failure branch (defect-site binding)` block + the **wiring mutation M10** (§3.1), which goes red with `"Network error calling /api/reports/?limit=20: Load failed"`. **NOT M8** — cycle-1 Q/A proved M8 vacuous at the defect site (it mutates the array the test imports directly, so it cannot tell a correct predicate from a wired-in one) | **MET** |
| 4 | Allow-list UNCHANGED for disallowed origins; do not echo `*` | live_check §C3 — header **absent** for `https://evil.example` while the 500 still returns; mutation M6. No new copy of `_TAILSCALE_ORIGIN_RE` | **MET** |
| add (i) | 500 carries the CORS header | §C2 | **MET** |
| add (ii) | 500 carries `nosniff` | §C2 (all six OWASP headers + `x-response-time`) | **MET** |
| add (iii) | 500 produces a PerfTracker record with status 500 **visible in** `/api/observability/latency` | live_check §D — `error_count: 5`, probe endpoint at `error_rate_pct: 100.0` | **MET** |

---

## 5. Scope honesty

- **The operator's `:8000` was NOT restarted**, so the fix is **inert in production until
  it is.** The gating step is **`phase-79.55`** — `status: pending`,
  `[RESTART BLOCKER -- answer BEFORE the next backend restart]`. Restarting would have
  silently shipped the phase-78.2 rail re-tiering (the six signal overlays down to
  `claude-haiku-4-5`, the lite trader/risk judge to `settings.gemini_model`) before the
  operator answered. After-fix evidence therefore comes from an isolated `:8001` instance
  running the same code with `--lifespan off` (no scheduler, no second trading loop). Full
  disclosure in live_check §A. **This is the one thing a reviewer should weigh.**

  > **Cycle-2 correction (Q/A C3/5.3).** Cycle 1 cited `phase-79.2` as the gate and said a
  > restart "would have breached both". That was wrong: **`79.2` is `status: done`** — its
  > body records `EXECUTED 2026-07-25 11:39:05 ... new pid 70791`, which is the pid
  > measured live on `:8000`. Only `79.55` is open. The decision was identical under the
  > corrected citation, but the stale reference would have sent a reader to a closed step.
- **One scope addition, declared in the contract before building:** `summarize()` +
  the latency route gained `error_count`/`error_rate_pct`. Without it, addendum criterion
  (iii) ("*visible in* `/api/observability/latency`") could not be met — `status_code` was
  discarded, so only an anonymous `count` bump would have changed. Strictly additive;
  every consumer enumerated below.
- **No immutable criterion was amended, reworded, or reinterpreted.**

### Consumer-contract check (additive-safe)

**Membership rule for this set, written down so the claim is auditable**
(`feedback_measure_dont_assert_claims`): every site that reads the dict returned by
`PerfTracker.summarize()`, derived by `grep -rn '\.summarize(' backend/ scripts/` and then
filtering out the unrelated `_cost_tracker.summarize()` / Slack `tracker.summarize()`
name-collisions (`orchestrator.py:2399`, `app_home.py:50,498`, `mas_events.py:133` — a
different tracker), plus every consumer of `/api/observability/latency`.

| Consumer | Reads | Safe? |
|---|---|---|
| `backend/api/observability_api.py:75` | re-keys `p50/p95/p99`, passes `per_endpoint` | yes |
| `backend/api/performance_api.py:34` | returns `summarize()` raw | yes |
| `backend/api/performance_api.py:40` | `get_slow_endpoints()` | yes |
| `backend/services/perf_optimizer.py:51,83` | `p95_ms`, `cache_hit_rate_pct` | yes |
| `backend/agents/meta_coordinator.py:266` | `p95_ms` | yes — **added cycle 2 (Q/A C5)** |
| `backend/services/perf_tracker.py:121` (`get_slow_endpoints`) | `data["p95_ms"]`, then `{"endpoint": ep, **data}` — so `/api/performance/slow` rows now also carry the two new keys | yes, additive — **added cycle 2 (Q/A C5)** |
| `backend/services/autonomous_loop.py:1579` | passes the tracker in (not a `summarize()` reader) — kept for traceability | n/a |
| `frontend/src/app/settings/page.tsx:1360` | renders `count`/`p50_ms`/`p95_ms` only | yes |
| `frontend/src/lib/types.ts:952-966` | new fields optional | yes |

**Cycle-1 honesty note:** the first version of this list said "every consumer enumerated"
but was not derived from a full grep — Q/A found the two rows now marked *added cycle 2*.
Both are safe; the *conclusion* held, its **completeness** was overstated. Recorded rather
than quietly patched.

No key renamed, removed, or changed in meaning. The `if not recent:` early return makes
both new divisions division-by-zero-safe.
`test_observability_latency_keeps_its_pre_existing_keys` asserts no pre-existing key
disappeared. `tsc --noEmit` clean.

### Out of scope — queued, not silently fixed

Per `feedback_queue_discovered_defects_in_masterplan`, to be installed as their own
research-gated steps:

1. **Three components bypass `apiFetch` entirely** and get none of this error handling:
   `ResearchInvestigator.tsx:33`, `Sidebar.tsx:155`, `StockChart.tsx:94`.
2. **Doc/code drift:** `.claude/rules/security.md` says `X-XSS-Protection: 1; mode=block`;
   `backend/main.py:564` sets `0` (the modern-correct value). Pre-existing; this step did
   not touch that header block, so it is flagged rather than quietly reconciled.

---

## 6. DO-NO-HARM ledger

| Item | Status |
|---|---|
| Live paper-trading book | **Untouched.** No `.env` edit, no flag flip, no optimizer run, `historical_macro` FROZEN. Kill-switch / stops / sector caps / DSR / PBO byte-untouched |
| Operator's `:8000` | Not restarted, same pid `70791` |
| Operator's `:3000` | `/` → 302, `/login` → 200 after teardown |
| Second trading loop | Prevented by `--lifespan off` (the lifespan starts an APScheduler paper-trading scheduler) |
| Auth behaviour | Cannot change: the catch-all runs *inside* CORS which is *inside* the auth middleware. `test_401_is_unchanged_for_an_unauthenticated_caller` + `test_404_still_returns_404_not_500` + M7 |
| Traceback in `backend.log` | Preserved via `logger.exception`; asserted by caplog (M3), verified live (live_check §E1) |
| SSE (`mas_events.py:36`) | Byte-identical to the pre-fix control (live_check §E2) |
| `perf_optimizer` p95 | Now includes 500-latency samples. It tunes **cache TTLs only**, never trade parameters — a measurement improvement, disclosed not hidden |
| `tsconfig.json` / `next-env.d.ts` | Rewritten by `next dev`, restored; md5s back to baseline, `git status` clean |

---

## 7. Tier ledger (operator tiering directive)

| Phase | Role | Model / effort | Why |
|---|---|---|---|
| RESEARCH | Agent-tool `researcher` | **T3** — Opus 5, `effort: max` (agent pin) | Audit-grade source reading; the gate is where a wrong premise is cheapest to catch |
| GENERATE | Main | **T3** — Opus 5 `xhigh` | Design was decidable from source + settled by an executable probe; extra model capability buys nothing over a measurement |
| EVALUATE | fresh Q/A | **T3** — Opus 5, `effort: max` | Independent verdict on a P0 |

**Fable (T4) deliberately NOT spent here.** The goal reserves it for correctness-critical
steps where a wrong answer moves the book; 80.2 changes no trading logic, and its
correctness question was answered by measurement rather than judgment. Quota is preserved
for `80.27` (NaN → live trading verdict), which is the step that does touch decisions.
