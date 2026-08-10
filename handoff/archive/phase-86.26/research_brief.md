# Research Brief — phase-80.2 (backend 500 carries no CORS / OWASP / PerfTracker)

Tier: **moderate**. `coverage.audit_class = false`. Written 2026-07-25,
write-first + incremental. (Previous occupant of this path was the
step-75.18 brief; it is preserved at
`handoff/archive/phase-75.18/research_brief.md`.)

## Immutable success criteria (verbatim — DO NOT AMEND)

1. A 500 response from the backend INCLUDES access-control-allow-origin for an allowed origin (verify against a route that genuinely raises, not a synthetic 204) -- so the browser surfaces the status instead of a network failure
2. With a deliberately-broken endpoint, the UI shows a server-error message that does NOT claim the backend is unreachable
3. api.ts network-error detection also matches Safari's 'Load failed'
4. The CORS allow-list behaviour is UNCHANGED for disallowed origins -- a 500 to a non-allowed origin must still omit the header (do not fix this by echoing '*')

---

## STATUS: COMPLETE — `gate_passed: true` (8 external sources read in full, recency scan performed, all internal claims line-anchored and re-measured against the running backend)

## Installed versions (reason against THESE, not latest-on-PyPI)

Measured from `/Users/ford/.openclaw/workspace/pyfinagent/.venv/bin/python -m importlib.metadata`:

| Package | Installed |
|---|---|
| fastapi | **0.135.2** |
| starlette | **1.0.0** |
| uvicorn | 0.42.0 |
| anyio | 4.13.0 |
| pydantic | 2.12.5 |

## Finding A1 (from installed source, definitive) — the middleware stack order

`.venv/lib/python3.14/site-packages/starlette/applications.py:57-77`:

```python
def build_middleware_stack(self) -> ASGIApp:
    debug = self.debug
    error_handler = None
    exception_handlers: dict[Any, ExceptionHandler] = {}

    for key, value in self.exception_handlers.items():
        if key in (500, Exception):
            error_handler = value
        else:
            exception_handlers[key] = value

    middleware = (
        [Middleware(ServerErrorMiddleware, handler=error_handler, debug=debug)]
        + self.user_middleware
        + [Middleware(ExceptionMiddleware, handlers=exception_handlers, debug=debug)]
    )

    app = self.router
    for cls, args, kwargs in reversed(middleware):
        app = cls(app, *args, **kwargs)
    return app
```

and `applications.py:98-101`:

```python
def add_middleware(self, middleware_class, *args, **kwargs) -> None:
    if self.middleware_stack is not None:  # pragma: no cover
        raise RuntimeError("Cannot add middleware after an application has started")
    self.user_middleware.insert(0, Middleware(middleware_class, *args, **kwargs))
```

Two consequences, both load-bearing for this step:

1. `ServerErrorMiddleware` is **hardcoded as the outermost layer** — it is
   prepended to `user_middleware`, so no `add_middleware` call can ever get
   outside it. `CORSMiddleware` is a *user* middleware and is therefore always
   INSIDE it. The suspected root cause in the step text is **CONFIRMED from
   source**, not inferred.
2. `add_middleware` **inserts at index 0**, so *later* registration = *more
   outer*. In this app CORS is registered first (`backend/main.py:485`) and the
   auth middleware second (`:509`), which makes the real order
   **outermost -> innermost**:
   `ServerErrorMiddleware -> auth_and_security_middleware -> CORSMiddleware -> ExceptionMiddleware -> router`.
   i.e. **CORSMiddleware is INSIDE the auth middleware**, not outside it.

## Finding A2 (from installed source, definitive) — `@app.exception_handler(Exception)` does NOT fix this

`applications.py:62-66` routes any handler keyed on `500` **or** `Exception`
into `error_handler`, which is passed to **`ServerErrorMiddleware`** —
the OUTERMOST layer. So a catch-all `@app.exception_handler(Exception)`
response is produced *outside* `CORSMiddleware` and *outside*
`auth_and_security_middleware`; it gets **no CORS headers, no OWASP headers,
no PerfTracker record**. It changes the body from `Internal Server Error`
to JSON and nothing else. This is the long-standing encode/starlette#1175
behaviour and it is still true on starlette 1.0.0.

`.venv/.../starlette/middleware/errors.py:163-186` (verbatim):

```python
try:
    await self.app(scope, receive, _send)
except Exception as exc:
    request = Request(scope)
    if self.debug:
        response = self.debug_response(request, exc)
    elif self.handler is None:
        response = self.error_response(request, exc)
    else:
        if is_async_callable(self.handler):
            response = await self.handler(request, exc)
        else:
            response = await run_in_threadpool(self.handler, request, exc)

    if not response_started:
        await response(scope, receive, send)

    # We always continue to raise the exception.
    # This allows servers to log the error, or allows test clients
    # to optionally raise the error within the test case.
    raise exc
```

Note the final `raise exc`: **ServerErrorMiddleware always re-raises** so
uvicorn logs the traceback. Any fix that catches the exception lower in the
stack must log it explicitly or `backend.log` loses the traceback
(do-no-harm item, see Risk section).

## Finding A3 — official Starlette docs CONFIRM this is by-design, and name the fix

`https://raw.githubusercontent.com/Kludex/starlette/main/docs/middleware.md`
(read in full 2026-07-25; the Starlette repo now lives under `Kludex/starlette`
and the docs site is `starlette.dev` — `starlette.io` still resolves):

- On `ServerErrorMiddleware`: *"This is **always** the outermost middleware
  layer."*
- Documented stack: `ServerErrorMiddleware -> [user middleware] -> ExceptionMiddleware -> Routing -> Endpoint`.
- The docs explicitly call out our exact defect: *"it's important to ensure
  that CORS headers are applied even to error responses generated by unhandled
  exceptions"*, and the documented remedy is to **wrap the entire application**
  in `CORSMiddleware` from OUTSIDE (i.e. `app = CORSMiddleware(app, ...)`,
  not `app.add_middleware`), because *"This approach guarantees that even if an
  exception is caught by ServerErrorMiddleware (or other outer error-handling
  middleware), the response will still include the proper
  `Access-Control-Allow-Origin` header."*
- Also relevant to any Route/Mount-scoped middleware: *"middleware used in this
  way is not wrapped in exception handling middleware like the middleware
  applied to the `Starlette` application is."*

## Finding A4 — upstream position: WON'T-DOC, WON'T-FIX (as of 2026)

- `https://github.com/Kludex/starlette/issues/1175` (read in full): custom
  500/`Exception` handlers do not run through middleware, because `Exception`
  and `500` are special-cased in `build_middleware_stack`. Four fixes were
  proposed by the reporter; the issue is **CLOSED** with no upstream change.
  The workaround endorsed in-thread is an HTTP middleware that wraps
  `call_next(request)` in `try/except Exception` and returns a `JSONResponse`.
- `https://github.com/fastapi/fastapi/discussions/13398` (read in full;
  opened 2025-02-20 against FastAPI 0.115.8 / Pydantic 2.10.6 / py3.12.7) —
  "Is there any plan to document that CORS headers are not applied in
  ServerErrorMiddleware?" **Kludex (FastAPI collaborator): "No plans."**
  He suggests uvicorn should eventually ship native CORS and welcomes a PR to
  document it in Starlette. The asker's own workaround hardcodes
  `headers={"access-control-allow-origin": "*"}` — **exactly what criterion 4
  forbids**; do not copy that snippet.
- `https://fastapi.tiangolo.com/tutorial/handling-errors/` (read in full):
  documents `@app.exception_handler(...)`, overriding `StarletteHTTPException`
  and `RequestValidationError`, and reusing the default handlers. It contains
  **no mention of CORS or middleware interaction at all** — so the "just add a
  catch-all exception handler" instinct a reader would take from the official
  tutorial does NOT solve this (see Finding A2).

## Finding A5 — recency scan (2024-2026)

Searched the last-2-year window explicitly. **Result: the behaviour is
unchanged and upstream has decided not to fix it; two 2025 threads and the
2026 Starlette 1.0 release confirm it still holds.** Detail:

| When | Source | Finding |
|---|---|---|
| 2024-10-15 -> 2025-10-28 | Starlette release notes 0.40.0 -> 0.49.0 | No change to the `ServerErrorMiddleware`-outermost rule. Middleware-adjacent entries are unrelated: 0.46.0 "Raise exception from background task on BaseHTTPMiddleware"; 0.49.0 "Do not pollute exception context in `Middleware` when using `BaseHTTPMiddleware`". |
| 2025-02-20 | fastapi discussion #13398 | Kludex: **"No plans"** to document it in FastAPI. |
| 2025-11-08..11 | fastapi discussion #14313 (read in full) | Same defect re-reported (`ValueError` no CORS, `AssertionError` with a handler does get CORS). YuriiMotov confirms the ExceptionMiddleware-vs-ServerErrorMiddleware split, offers (1) wrap the app in `CORSMiddleware(app=app, ...)`, (2) register handlers for **specific** exception types rather than bare `Exception`. **No final accepted answer**; "acknowledged the design isn't ideal". |
| 2026-02-23 (rc1) / 2026-03-22 (1.0.0) | Starlette 1.0.0 release notes | 1.0 removed `@app.exception_handler()` and `@app.middleware()` **decorators from Starlette** (FastAPI keeps its own — see A6), removed `on_event`, added "Return explicit origin in CORS response when credentials are allowed". **`build_middleware_stack` still hardcodes `ServerErrorMiddleware` outermost** (verified in the installed 1.0.0 copy, Finding A1). |

Net: nothing in the last two years supersedes the analysis; the newest
(2026-03) Starlette major release preserves the exact ordering.

## Finding A6 — the decorators this app uses still exist (FastAPI, not Starlette)

Starlette 1.0.0 removed `@app.middleware()` / `@app.exception_handler()` from
`starlette.applications.Starlette`, but **FastAPI re-implements both** —
verified in the installed copy:
`fastapi/applications.py:4600 def middleware(...)`,
`fastapi/applications.py:4646 def exception_handler(...)`.
So `backend/main.py:509 @app.middleware("http")` is safe on this pin, and a
new `@app.exception_handler(...)` would also be accepted (it just would not
solve the problem — A2). FastAPI 0.135.2 also **overrides**
`build_middleware_stack` (`fastapi/applications.py:1018-1060`) purely to add
`AsyncExitStackMiddleware` **inside** `ExceptionMiddleware`; the
`ServerErrorMiddleware`-first / user-middleware / `ExceptionMiddleware` order
is byte-identical to Starlette's.

## Finding A7 — browser network-error strings (criterion 3)

| Engine | `TypeError.message` on a failed `fetch()` |
|---|---|
| Safari / WebKit | **`Load failed`** |
| Chrome / Edge / Chromium | `Failed to fetch` |
| Firefox / Gecko | `NetworkError when attempting to fetch resource.` |
| axios (any engine) | `Network Error` |

Source: TrackJS error-reference page for `Load failed` (read in full,
2026-07-25) — *"a connection failure, not a server response"*, i.e. the
request never reached the server **or the response was blocked** (a
CORS-blocked response is indistinguishable from an unreachable host at the JS
layer — which is precisely why the current api.ts message is wrong).
Corroborated by the WebKit-tracking threads found in search (hotwired/stimulus
#782 "Load failed errors when sending fetch requests in Safari on iOS";
Apple Developer Forums thread 771127 "fetch fails with 'Load failed'").
**Stability/localisation:** no authoritative source states the string is
API-stable or localised; the WHATWG Fetch spec only mandates rejecting with a
`TypeError` and leaves the message implementation-defined. Practical
consequence for the fix: match a **set** of substrings and keep the generic
`throw new Error(...)` fallback at the end (never make correctness depend on
the string alone).

---

# B. Internal code inventory (all line numbers MEASURED 2026-07-25, not inherited)

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/main.py` | 465-473 | `app = FastAPI(...)`, docs debug-gated | unchanged |
| `backend/main.py` | 480-482 | `_TAILSCALE_ORIGIN_RE` — the single origin predicate | the allow-list SSOT (criterion 4) |
| `backend/main.py` | 485-491 | `app.add_middleware(CORSMiddleware, ...)` | registered FIRST -> ends up INNER |
| `backend/main.py` | 497-506 | `_PUBLIC_PATHS` (8 entries) | auth-skip list |
| `backend/main.py` | 509-569 | `auth_and_security_middleware` | registered SECOND -> OUTSIDE CORS |
| `backend/main.py` | 609, 647 | only two app-level routes (`/api/health`, `/api/changelog`) | no probe route exists |
| `backend/services/perf_tracker.py` | 29-147 | `PerfTracker` + module singleton | sole latency source |
| `frontend/src/lib/api.ts` | 115-134 | network-error branch | Chromium/Firefox-only strings |

### B1. Ordered middleware registration (the measured truth)

`backend/main.py` registers, in file order:
1. `:485-491` `CORSMiddleware(allow_origin_regex=_TAILSCALE_ORIGIN_RE.pattern, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])`
2. `:509` `@app.middleware("http") auth_and_security_middleware`

Because `add_middleware` **inserts at index 0** (A1), the resulting runtime
stack is **outermost -> innermost**:

```
ServerErrorMiddleware            (Starlette, always outermost)
  auth_and_security_middleware   (BaseHTTPMiddleware, registered LAST)
    CORSMiddleware               (registered FIRST -> inner)
      ExceptionMiddleware        (Starlette)
        AsyncExitStackMiddleware (FastAPI)
          router / endpoint
```

> **This corrects a plausible mental model**: `CORSMiddleware` is *not* the
> outermost user middleware — it sits **inside** the auth middleware. Any fix
> that returns a response *from inside `auth_and_security_middleware`* is
> **outside** `CORSMiddleware` and will therefore get **no** automatic CORS
> headers (this is exactly why the 401 path at `:529-544` has to echo the
> origin by hand).

### B2. `auth_and_security_middleware` body (verbatim anchors)

- `:512` `path = request.url.path`
- `:518` auth skip for `OPTIONS` + `_PUBLIC_PATHS`
- `:519-544` `get_current_user` -> on `HTTPException`, returns a `JSONResponse`
  that **manually** sets `Access-Control-Allow-Origin` / `-Credentials` /
  `Vary` **only when `_TAILSCALE_ORIGIN_RE.match(origin)`** (`:536-539`).
  **This is the in-repo precedent for an allow-list-preserving manual echo.**
- `:546` `start = time.perf_counter()`
- `:547` `response: Response = await call_next(request)`  <- **the exception escapes here**
- `:548` latency computed
- `:551-558` `get_perf_tracker().record(endpoint, method, status_code, latency_ms, cache_hit)`
- `:559` `X-Response-Time`
- `:562-567` the six OWASP headers
- `:569` `return response`

Everything from `:548` to `:569` is skipped when `call_next` raises —
confirming both consequences (a) and (b) in the step text at the measured
line numbers.

### B3. Existing exception handlers: **NONE**

`grep -rn "exception_handler" --include="*.py" backend/` returns **zero
matches**. There is no catch-all, no `add_exception_handler`, no override of
FastAPI's default `HTTPException` / `RequestValidationError` handlers. Nothing
to duplicate or fight — but also nothing to piggyback on.

### B4. CORS config (criterion 4 is entirely about this)

```python
_TAILSCALE_ORIGIN_RE = re.compile(
    r"^http://(localhost|100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+):\d+$"
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=_TAILSCALE_ORIGIN_RE.pattern,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Not from settings — a module-level hardcoded regex (phase-75.1 security-04
deliberately made it the single predicate shared with the 401 echo). Note
`allow_origins` is empty, so in `starlette/middleware/cors.py`:
- `allow_all_origins = False` -> `simple_headers = {"Access-Control-Allow-Credentials": "true"}` only (`cors.py:39-45`);
- `send()` (`cors.py:156-174`) adds `Access-Control-Allow-Origin` **only** when `is_allowed_origin(origin)` (`cors.py:98-105`, `fullmatch` on the regex);
- `__call__` (`cors.py:87-89`) **short-circuits entirely when the request has no `Origin` header** — so header-less curl/CLI callers never see CORS headers (expected).

**Measured baseline (live backend, 2026-07-25 15:53 UTC):**

```
A) GET /api/paper-trading/portfolio  Origin: http://localhost:3000   -> 200
   access-control-allow-credentials: true
   access-control-allow-origin: http://localhost:3000
   vary: Origin
   x-response-time: 4060ms
   x-content-type-options: nosniff

B) GET /api/paper-trading/portfolio  Origin: https://evil.example    -> 200
   access-control-allow-credentials: true
   x-content-type-options: nosniff
   (NO access-control-allow-origin)      <-- criterion-4 BASELINE to preserve

C) GET /api/signals/AAPL             Origin: http://localhost:3000   -> 500
   date / server / content-length: 21 / content-type: text/plain
   (NO access-control-allow-origin, NO OWASP headers, NO x-response-time)
   body: "Internal Server Error"
```

Note (B): a disallowed origin still receives `access-control-allow-credentials:
true` (it lives in `simple_headers` unconditionally). **Criterion 4 is about
`access-control-allow-origin` only** — do not "fix" the credentials header as
part of this step; changing it is out of scope and would be a behaviour change.

Note (C) also proves the defect is *live right now* and that
`/api/signals/AAPL` is a genuinely-raising route today (that is step 80.1's
subject — once 80.1 lands, this route stops being a usable 500 fixture, which
is exactly why criterion 1 needs a deliberate probe).

### B5. `frontend/src/lib/api.ts` — verbatim, measured line numbers

```ts
107      let res: Response;
108      try {
109        res = await fetch(`${API_BASE}${path}`, {
110          ...init,
111          headers,
112          credentials: "include",
113          signal: init?.signal ?? controller.signal,
114        });
115      } catch (err) {
116        // Abort / timeout
117        if (err instanceof DOMException && err.name === "AbortError") {
118          throw new Error(
119            `Request to ${path} timed out after 30 seconds. ` +
120            "The backend may be overloaded or unresponsive."
121          );
122        }
123        // Network-level failure (CORS, DNS, refused, etc.)
124        const msg = err instanceof Error ? err.message : String(err);
125        if (msg.includes("Failed to fetch") || msg.includes("NetworkError")) {
126          throw new Error(
127            `Cannot reach backend at ${API_BASE}. ` +
128            "Make sure the FastAPI server is running (uvicorn backend.main:app --port 8000)."
129          );
130        }
131        throw new Error(`Network error calling ${path}: ${msg}`);
132      } finally {
133        clearTimeout(timeoutId);
134      }
```

and the already-correct server-error branch that criterion 2 depends on:

```ts
161      if (res.status === 500) {
162        throw new Error(`Server error on ${path}. Check the backend logs for details.`);
163      }
```

**So criterion 2 needs NO frontend change** — `:161-163` already produces a
non-"backend is down" message. It is unreachable today only because the
browser never gets a readable response. Fixing the backend (criteria 1) makes
`:161` fire. Criterion 3 is a separate, additive edit at `:125`.

> **Out-of-scope defect found (queue it, don't silently fix):** three
> components bypass `apiFetch` and call `fetch()` directly, so they get none
> of this error handling —
> `frontend/src/components/ResearchInvestigator.tsx:33` (`/api/investigate`),
> `frontend/src/components/Sidebar.tsx:155` (`/api/changelog`),
> `frontend/src/components/StockChart.tsx:94` (`/api/charts/{ticker}`).
> Per `feedback_queue_discovered_defects_in_masterplan` this deserves its own
> masterplan step, not a drive-by edit inside 80.2.

### B6. `PerfTracker` — signature and every consumer

`backend/services/perf_tracker.py`:
```python
def record(self, endpoint: str, method: str, status_code: int,
           latency_ms: float, cache_hit: bool = False) -> None
```
(`:37-57`; appends a `LatencyEntry` dataclass `:19-26` with `timestamp=time.time()`,
FIFO-evicted at `max_entries=10_000`; module singleton `_perf_tracker` at
`:143`, accessor `get_perf_tracker()` at `:146`.)

Consumers (complete list):
| Consumer | Line | Uses |
|---|---|---|
| `backend/main.py` | 552 | the **only writer** |
| `backend/api/observability_api.py` | 75 | `summarize(window_seconds)` -> `/api/observability/latency`, re-keyed to `p50/p95/p99` (`:87-95`), fail-open to zeros on exception (`:76-86`) |
| `backend/api/performance_api.py` | 34 | `/api/perf/summary` |
| `backend/api/performance_api.py` | 40 | `/api/perf/slow` -> `get_slow_endpoints(threshold_ms)` |
| `backend/services/perf_optimizer.py` | 51, 83 | TTL optimizer baseline + keep/discard measurement (`p95_ms`, `cache_hit_rate_pct`) |
| `backend/services/autonomous_loop.py` | 1579-1582 | passes the tracker into the cycle |

**What a 500 record must look like to show up in `/api/observability/latency`:**
just a normal `record(endpoint=path, method=request.method, status_code=500,
latency_ms=<measured>, cache_hit=False)` inside the 300 s default window.
`summarize()` does **not** filter by status code at all (`:59-101`), so the
entry counts toward `total_requests` and the p50/p95/p99 percentiles the
moment it is recorded. There is **no** per-status breakdown today — an error
rate cannot be derived from `/api/observability/latency` even after this fix;
only "the failing endpoint appears in `per_endpoint` with a count" changes.
(Flagging: if the step wants a visible *error* signal rather than merely a
non-blind latency series, that is an additive change to `summarize()` and
should be scoped explicitly, because `perf_optimizer` reads the same dict.)

### B7. Probe route: **does not exist**

`grep -rn "force_500|__force|force-500" --include="*.py" backend/` -> no
matches. `backend/main.py` defines exactly two app-level routes
(`:609 /api/health`, `:647 /api/changelog`); there is no debug/raise route in
any router. **`/api/__force_500_probe` must be created by this step.**
Constraints found:

- **Auth**: `/api/__force_500_probe` would NOT be in `_PUBLIC_PATHS`, so the
  auth middleware runs first. The live capture above shows `/api/signals/AAPL`
  reached its route from a bare localhost curl, i.e. the
  `DEV_LOCALHOST_BYPASS=1` + `client.host in (127.0.0.1, ::1, localhost)` rail
  in `backend/api/auth.py:150-152` is active in the running process. So the
  step's `curl http://localhost:8000/api/__force_500_probe` will reach the
  route **on this machine** without a token. Do **not** add the probe to
  `_PUBLIC_PATHS` — that would need a `.claude/rules/security.md` row and
  widen the unauthenticated surface for zero benefit.
- **DEBUG is OFF in the running process — MEASURED, decisive.**
  `curl -s -o /dev/null -w '%{http_code}' localhost:8000/docs` -> **404**;
  `/openapi.json` -> **404**. So `get_settings().debug is False` live
  (`backend/main.py:464`). **A debug-gated probe route would 404 and the
  step's own immutable verification command would fail.** Register the probe
  **unconditionally**, keep it auth-gated (do NOT add to `_PUBLIC_PATHS`),
  set `include_in_schema=False`, and have it raise a plain `RuntimeError`
  with a self-identifying message. If the contract still prefers a gate, it
  must be a NEW dedicated env flag that is ON in the operator's `.env`, not
  `DEBUG` — and that trades a doc/ops burden for no security gain over the
  existing auth gate.
- **404-vs-500 contrast, measured live** (the cleanest proof of Finding A2):
  `GET /api/__force_500_probe` (nonexistent today) with
  `Origin: http://localhost:3000` returns
  `HTTP/1.1 404 Not Found` **WITH** `access-control-allow-origin:
  http://localhost:3000`. A 404 is an `HTTPException` handled by
  `ExceptionMiddleware` (innermost) so `CORSMiddleware` decorates it; the 500
  bypasses both. Same app, same origin, same curl — only the exception class
  differs. Use this pair as the before/after evidence in the live_check.

### B8. Test conventions

- Runner: **`python -m pytest`** from the repo root with the venv active
  (`source .venv/bin/activate`); `pytest.ini` at repo root only registers the
  `requires_live` marker (`requires_live` tests are skipped unless
  `PYFINAGENT_LIVE_TESTS=1`). Tests live in `backend/tests/` (158 entries),
  named `test_phase_<X>_<Y>_<slug>.py`. Per
  `project_fable5_adoption` the file name must contain the new phase token so
  a `-k` selection can actually reach it: **`backend/tests/test_phase_80_2_*.py`**.
- **`TestClient` gotchas that this step will hit:**
  - `TestClient(app)` defaults to `raise_server_exceptions=True`
    (`starlette/testclient.py:204`), which **re-raises** the endpoint
    exception instead of returning a 500. A pre-fix reproduction test must
    pass `raise_server_exceptions=False`.
  - `TestClient` reports `request.client.host == "testclient"`, so the
    localhost bypass never fires in-process — use
    `backend/tests/auth_helper.py::authed_test_client(app, **kwargs)`
    (`:77-89`; it forwards `**kwargs` to `TestClient`, so
    `authed_test_client(app, raise_server_exceptions=False)` works).
  - The house pattern for importing the real app is a guarded import:
    `try: from backend.main import app / except ...: pytest.skip(...)`
    (`backend/tests/test_phase_23_2_13_governance_watcher.py:165-167`);
    `backend/tests/api/test_sovereign.py:24` imports it unguarded.
  - `TestClient` sends **no `Origin` header** by default — the assertion must
    set `headers={"Origin": "http://localhost:3000"}` explicitly, plus a
    negative case with `Origin: https://evil.example` for criterion 4.
- **Anti-pattern warning (`feedback_mutation_test_guards_and_fixtures`):**
  `backend/tests/test_phase_75_deploy_surface.py:397-401` is a *source-scan*
  CORS test (`assert "Access-Control-Allow-Origin': '*'" not in src`). Do not
  copy that shape for 80.2 — a source scan cannot see middleware ordering. The
  only non-vacuous guard here is an **end-to-end request through the real
  stack** against a route that genuinely raises, asserting header presence for
  an allowed origin AND header absence for a disallowed one.

### B9. Rule text to cite

- `.claude/rules/security.md`, "## OWASP Headers (all responses)":
  `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  **`X-XSS-Protection: 1; mode=block`**, `Referrer-Policy: strict-origin-when-cross-origin`,
  `Cache-Control: no-store`, `Permissions-Policy` (restricted). The heading
  "(all responses)" is the breached requirement.
  *Pre-existing doc/code drift, NOT caused by this step and NOT this step's to
  fix silently:* `backend/main.py:564` sets `X-XSS-Protection: 0` (the modern
  correct value) while the rule still says `1; mode=block`. Flag it; if 80.2
  touches the header block, reconciling the doc line is a one-line honest
  cleanup — but state it explicitly rather than letting the diff imply the
  rule was already aligned.
- `.claude/rules/security.md`, "## CORS": *"Allows `localhost:*` and Tailscale
  CGNAT IPs only (`100.64.0.0/10`, second octet 64-127, RFC 6598) via the
  single module-level `_TAILSCALE_ORIGIN_RE` in `backend/main.py` — shared by
  CORSMiddleware `allow_origin_regex` AND the manual 401 CORS echo so the two
  seams cannot drift (phase-75.1)"*. A third seam would need the same
  treatment — **prefer a fix that adds no fourth copy of the predicate.**
- `.claude/rules/backend-api.md:53`: *"OWASP security headers on all responses"*.
- `.claude/rules/backend-services.md`: *"`perf_tracker.py` — Thread-safe
  per-endpoint latency recorder. Middleware collects timing..."* — the
  middleware IS the documented collection point, so a 500 that skips it is a
  documented-behaviour breach too.

---

# C. Application to pyfinagent — the four candidate fixes, ranked

| # | Approach | Criterion 1 (CORS on 500) | Criterion 4 (allow-list intact) | Fixes PerfTracker + OWASP? | Blast radius |
|---|---|---|---|---|---|
| **C1** | **New catch-all middleware registered BEFORE `add_middleware(CORSMiddleware)` so it lands INSIDE CORS** | YES — `CORSMiddleware.send` decorates it like any normal response | YES — enforced by `CORSMiddleware` itself, **no new copy of the predicate** | YES — the response returns normally through `auth_and_security_middleware`, so `:552` record + `:559`-`:567` headers all run | Low, but depends on a **counter-intuitive registration order** |
| C2 | `try/except` around `:547 call_next` inside `auth_and_security_middleware` | Only with a **manual origin echo** (this response is outside CORS) | Yes if it reuses `_TAILSCALE_ORIGIN_RE` | YES (fall through to the existing tail) | Low, mirrors the `:536-539` 401 idiom, but creates a **third copy** of the CORS-echo logic |
| C3 | `app = CORSMiddleware(app, ...)` wrapping (the official Starlette doc answer) | YES | Yes (same regex) | **NO** — response still bypasses auth middleware | Medium/high: rebinds `app` to a non-FastAPI object; breaks `from backend.main import app` consumers, `TestClient` fixtures, later `include_router`, `app.state` |
| C4 | `@app.exception_handler(Exception)` | **NO** (Finding A2) | n/a | NO | trap — looks right, does nothing |

**Recommendation: C1**, with C2 as the fallback if the ordering proves
fragile. Rationale: C1 is the only option that satisfies all four criteria
*and* both secondary consequences without duplicating the origin predicate
(directly honouring the phase-75.1 "two seams cannot drift" rule). The
resulting stack would be:

```
ServerErrorMiddleware
  auth_and_security_middleware        (records + headers, now always reached)
    CORSMiddleware                    (decorates the 500 -> criterion 1 + 4)
      <new catch-all>                 (converts the raise into a JSONResponse)
        ExceptionMiddleware
          router
```

Implementation notes for C1:
- Write it as a **pure-ASGI** middleware (not another `BaseHTTPMiddleware`) —
  nesting `BaseHTTPMiddleware` adds a task group + contextvar copy per layer,
  and Starlette 0.46/0.49 changelog entries show this area is still being
  patched. Mirror `starlette/middleware/errors.py:154-186`: wrap `send`,
  track `response_started`, and only emit the JSON 500 `if not
  response_started` (a mid-stream failure cannot be rewritten — say so in the
  code comment rather than pretending it is covered).
- **Log the swallowed exception explicitly** (`logger.exception(...)`);
  otherwise `backend.log` loses the traceback that
  `ServerErrorMiddleware`'s `raise exc` (`errors.py:186`) currently produces.
  This is do-no-harm item (iii) and must be asserted in the test, not assumed.
- Give the response the shape the frontend already expects:
  `JSONResponse(status_code=500, content={"detail": "..."})` — `api.ts:151-157`
  reads `body.detail`, and `:161` throws the server-error message. Do **not**
  include the traceback in the body (phase-75.16 established that traceback
  leakage to HTTP callers is a defect class in this repo).
- The registration-order dependency is invisible-by-inspection, so it needs
  (a) an inline comment at the registration site explaining
  `add_middleware` inserts at index 0, and (b) an executable ordering
  assertion, e.g. `[m.cls.__name__ for m in app.user_middleware]`, in
  addition to the end-to-end header test.

---

# D. Risk / do-no-harm

| Risk | Verdict | Detail |
|---|---|---|
| (i) **auth behaviour change** | **Low with C1; a REAL trap with C4/C2-if-sloppy** | The catch-all in C1 sits **inside** `CORSMiddleware`, which is inside `auth_and_security_middleware`, so it runs **after** auth has already decided. It can therefore never convert a 401/403 into a 500. **The inverse trap is real:** `HTTPException(401)` raised *inside a route* is handled by `ExceptionMiddleware` (innermost) and never reaches the catch-all — but a catch-all that naively catches `Exception` and is placed **outside** `ExceptionMiddleware` in a future refactor WOULD swallow `HTTPException` and turn every 401/403/404 into a 500. Mitigation: re-raise `HTTPException` / `StarletteHTTPException` explicitly, and add a regression test that a 401 route still returns 401 and a 404 still returns 404. |
| (i-b) the `:519-544` manual-401 path | untouched by C1 | It returns *before* `call_next`, so it never traverses the new middleware. Its own CORS echo keeps working. Verify with a no-token request from a non-localhost client if possible. |
| (ii) **which origins are allowed** | **Zero change under C1** | The allow-list decision stays 100% inside `CORSMiddleware.is_allowed_origin` (`cors.py:98-105`). Under C2 a third copy of the predicate is introduced — that is the drift risk phase-75.1 explicitly closed. **Never** `access-control-allow-origin: *` (criterion 4, and `test_phase_75_deploy_surface.py:397-401` would catch the literal in Cloud-Function code but NOT in `backend/main.py` — that scan reads only `functions/earnings/main.py`). |
| (iii) **swallowing a useful exception** | **Real; must be actively mitigated** | Today `ServerErrorMiddleware` re-raises (`errors.py:183-186`) so uvicorn prints "Exception in ASGI application" + traceback into `backend.log`. Any lower catcher ends that. Mitigation = `logger.exception` with the path + method, asserted by a caplog test. Also note `DEBUG=true` currently yields the HTML traceback page (`errors.py:167-169`); C1 pre-empts that in debug too — acceptable, but say so. |
| (iv) **response-shape change for existing consumers** | **Low, and a strict improvement** | Today a 500 is `text/plain` `"Internal Server Error"` (21 bytes, captured live). After C1 it becomes `application/json {"detail": ...}`. `api.ts:151-157` already tries `res.json()` first and falls back to `res.text()`, so both shapes work; `:161` short-circuits on status 500 before `detail` is used. No backend consumer parses 500 bodies (there is no internal HTTP client that calls these routes). |
| (v) **live paper-trading book** | **No exposure** | Nothing in this step touches `backend/services/paper_trader.py`, the scheduler, or any order path. The one indirect coupling is `perf_optimizer` (`:51,:83`) now seeing 500-latency samples in its p95 — it tunes **cache TTLs only**, never trade parameters. Worth one sentence in the contract; not a book risk. |
| (vi) **restart required** | YES | Middleware is built once (`applications.py:88-90`, and `add_middleware` raises `RuntimeError` after startup). The fix is inert until the backend is restarted — and `handoff/current/live_check_80.2.md` evidence must be captured **after** that restart, or it will show the old behaviour. |
| (vii) **new probe route** | Small, bounded | Auth-gated, no data, `include_in_schema=False`. Do not add it to `_PUBLIC_PATHS`. Decide debug-gating against the running process's actual `DEBUG` value (B7) so the immutable verification command can pass. |

---

## Search queries run (3-variant discipline, `.claude/rules/research-gate.md`)

| # | Variant | Query |
|---|---|---|
| 1 | current-year (2026) | `FastAPI 500 error missing CORS header ServerErrorMiddleware exception handler 2026` |
| 2 | year-less canonical | `starlette CORSMiddleware exception handler 500 no Access-Control-Allow-Origin` |
| 3 | year-less canonical | `Safari fetch TypeError "Load failed" WebKit error message network failure` |
| 4 | last-2-year (2025/2026) | `Starlette 1.0 release notes 2025 2026 breaking changes middleware exception handling` |

---

## Read in full (8; counts toward the gate)

| # | URL | Accessed | Kind / tier | Fetched how | Key finding |
|---|-----|----------|-------------|-------------|-------------|
| 1 | https://raw.githubusercontent.com/Kludex/starlette/main/docs/middleware.md | 2026-07-25 | Official docs (T2) | WebFetch | *"This is always the outermost middleware layer"* (ServerErrorMiddleware); documented remedy = wrap the whole app in `CORSMiddleware` |
| 2 | https://fastapi.tiangolo.com/tutorial/handling-errors/ | 2026-07-25 | Official docs (T2) | WebFetch | Full exception-handler API; **zero** mention of CORS/middleware interaction -> the tutorial leads readers straight into the C4 trap |
| 3 | https://github.com/Kludex/starlette/issues/1175 | 2026-07-25 | Upstream issue (T2) | WebFetch | Root cause = `Exception`/`500` special-cased in `build_middleware_stack`; CLOSED without fix; endorsed workaround = HTTP middleware `try/except` around `call_next` |
| 4 | https://github.com/fastapi/fastapi/discussions/13398 | 2026-07-25 | Upstream discussion (T2) | WebFetch | 2025-02-20, FastAPI 0.115.8. Kludex: **"No plans"** to document. Asker's snippet uses `allow-origin: *` — criterion-4 violation, do not copy |
| 5 | https://github.com/fastapi/fastapi/discussions/14313 | 2026-07-25 | Upstream discussion (T2) | WebFetch | 2025-11-08..11 recurrence; `ValueError` no CORS vs `AssertionError`+handler yes; two workarounds; no accepted answer |
| 6 | https://raw.githubusercontent.com/Kludex/starlette/main/docs/release-notes.md | 2026-07-25 | Official changelog (T2) | WebFetch | 0.40.0 (2024-10) -> 1.0.0 (2026-03-22): no change to the ordering rule; 1.0 removed the Starlette-level `@app.exception_handler`/`@app.middleware` decorators; "Return explicit origin in CORS response when credentials are allowed" |
| 7 | https://trackjs.com/javascript-errors/load-failed/ | 2026-07-25 | Industry vendor (T4) | WebFetch | Safari = `Load failed`; Chromium = `Failed to fetch`; Firefox = `NetworkError when attempting to fetch resource`; axios = `Network Error`; *"a connection failure, not a server response"* |
| 8 | https://developer.apple.com/forums/thread/771127 | 2026-07-25 | Apple-hosted vendor forum (T2/T4) | WebFetch | Independent confirmation of the literal string: Safari iOS 18 rejects with `TypeError` message **`Load failed`**; thread open, no Apple fix |

## Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://starlette.dev/middleware/ | Official docs | **HTTP 403 to WebFetch** — substituted the identical source markdown (#1) |
| https://github.com/fastapi/fastapi/issues/775 | Upstream issue | Same defect, 2019 origin thread; superseded by #1175 + #14313 |
| https://github.com/fastapi/fastapi/issues/4071 | Upstream issue | Duplicate of the same report |
| https://github.com/fastapi/fastapi/discussions/8658 | Discussion | Duplicate |
| https://github.com/fastapi/fastapi/discussions/7847 | Discussion | Duplicate |
| https://github.com/fastapi/fastapi/discussions/8027 | Discussion | Duplicate ("CORS headers on exception handler responses") |
| https://github.com/encode/starlette/issues/1116 | Upstream issue | "Error are not reported using CORS middleware" — same root cause |
| https://github.com/Kludex/starlette/discussions/2424 | Discussion | Missing ACAO when origin not allowed — the criterion-4 *expected* behaviour |
| https://github.com/Kludex/starlette/discussions/2561 | Discussion | `allow-credentials` emitted regardless of origin — explains live capture (B) |
| https://github.com/Kludex/starlette/issues/810 | Upstream issue | Generic CORS-not-working; not this defect |
| https://github.com/Kludex/starlette/issues/2625 | Upstream issue | BaseHTTPMiddleware silently swallowing exceptions — relevant caution for C1 |
| https://docs.bswen.com/blog/2026-02-27-starlette-100-breaking-changes/ | Blog (T3) | Starlette 1.0 breaking-change summary; primary changelog (#6) preferred |
| https://github.com/hotwired/stimulus/issues/782 | Community issue | Corroborates `Load failed` on iOS Safari |
| https://www.peterp.me/articles/react-native-type-error-load-failed/ | Blog (T3) | RN-specific `Load failed` write-up |
| https://discussions.apple.com/thread/256112607 | Community forum (T5) | Safari 18 network flakiness, anecdotal |

Unique URLs collected: **23** (8 read in full + 15 snippet-only).

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch — **8**
- [x] 10+ unique URLs total — **23**
- [x] Recency scan (last 2 years) performed + reported — Finding A5 (2024-10 -> 2026-03)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim — §B, all re-measured today

Soft checks:
- [x] Internal exploration covered every module the caller named (main.py, api.ts, perf_tracker + all 6 consumers, tests, both rule files) plus the installed starlette/fastapi sources
- [x] Contradictions / consensus noted (official docs recommend app-wrapping = C3; upstream threads recommend middleware try/except = C2; this brief recommends C1 and says why both others fall short here)
- [x] All claims cited per-claim
- [x] Live reproduction captured against the running backend rather than inferred

Deviations to disclose: this brief is **~5.5K words, well over the
`moderate` tier's <=700-word guidance**. The overrun is deliverable-driven,
not padding — the caller requested verbatim quotes of the middleware body,
`api.ts:107-134`, the CORS config, six PerfTracker consumers, two rule files,
and a ranked fix comparison. Compressing would have meant dropping requested
material. Flagging it rather than silently rescoping. Also: source #7 is tier-4 (vendor error-reference) and #8 is
an Apple-hosted developer forum thread — used only for the browser
error-string question, where no tier-1/2 source enumerates per-engine message
text (the WHATWG Fetch spec mandates only `TypeError`). The six sources
carrying the load-bearing middleware findings are all tier-2 (official docs,
official changelog, upstream issue trackers) **plus** the installed source
code itself, which is the strongest evidence available and is quoted verbatim
in Findings A1/A2.

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 15,
  "urls_collected": 23,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Root cause CONFIRMED from the installed source, not inferred: starlette 1.0.0 applications.py:57-77 hardcodes ServerErrorMiddleware as the outermost layer and routes any handler keyed on 500/Exception into it, so a catch-all @app.exception_handler(Exception) is ALSO outside CORSMiddleware and does NOT fix this (the C4 trap). Measured registration order in backend/main.py is ServerError -> auth_and_security_middleware(:509) -> CORSMiddleware(:485) -> ExceptionMiddleware -> router, i.e. CORS is INSIDE the auth middleware, so a fix returning a response from the auth middleware still needs a manual origin echo. Recommended C1: a pure-ASGI catch-all registered BEFORE add_middleware(CORSMiddleware) so it lands inside CORS -- CORSMiddleware then enforces the allow-list itself (criterion 4, no fourth copy of _TAILSCALE_ORIGIN_RE) and the response returns normally through the auth middleware, restoring PerfTracker(:552) and the OWASP headers(:562-567). Live-measured: DEBUG is OFF (docs 404) so the probe route must NOT be debug-gated; no probe route exists yet; no exception handler exists anywhere in backend/; api.ts:161 already emits a correct 500 message so only :125 needs the Safari 'Load failed' string. Do-no-harm: must logger.exception (ServerErrorMiddleware's raise exc currently feeds backend.log) and must re-raise HTTPException so 401/404 cannot become 500.",
  "brief_path": "handoff/current/research_brief.md",
  "gate_passed": true
}
```

