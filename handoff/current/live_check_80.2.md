# live_check — phase-80.2

**Required (masterplan, verbatim):** *paired curl -D- output for a 500 and a 200 on the
same origin showing the header now present on both, plus a Playwright console capture
showing no CORS-block error.*

Captured 2026-07-25. All output below is verbatim.

---

## §A. METHOD, AND THE ONE HONEST LIMITATION — read this first

**The operator's backend on `:8000` was NOT restarted, so it is still running the
pre-fix code.** That is deliberate, not an oversight:

- **`phase-79.55` is `status: pending`** and is an open **`[RESTART BLOCKER -- answer
  BEFORE the next backend restart]`** — the rail-model tier confirmation opened by 78.2.
  Restarting now would silently ship a model-tier **downgrade** on live signal paths (the
  six overlays to `claude-haiku-4-5`, the lite trader and lite risk judge to
  `settings.gemini_model`) before the operator has answered.

> **Correction (cycle 2, Q/A finding 5.3).** The first version of this section also cited
> `phase-79.2` ("BACKEND RESTART") as an open operator action. **It is `status: done`** —
> its body records `EXECUTED 2026-07-25 11:39:05 ... new pid 70791`, and 70791 is the pid
> measured live on `:8000`. That restart already happened; it simply predates this step's
> code. The open gate on the *next* restart is `79.55` alone. The decision not to restart
> is unchanged; only the citation was wrong.

So the after-fix evidence was captured on an **isolated second instance**, exactly
mirroring this project's established `:3100` frontend-rig discipline:

```
DEV_LOCALHOST_BYPASS=1 PYFINAGENT_TEST_NO_BQ=1 \
  .venv/bin/python -m uvicorn backend.main:app --port 8001 --lifespan off --log-level warning
```

Three deliberate choices in that command, each stated so the evidence can be audited
rather than trusted:

1. **`--lifespan off`** — `backend/main.py`'s lifespan starts an APScheduler paper-trading
   scheduler when `settings.paper_trading_enabled`, plus a queue processor, plus a
   `handoff/.autonomous_loop.lock`. A second full instance could therefore have run a
   **second trading loop**. `--lifespan off` skips startup entirely. The middleware stack
   is built at ASGI-app construction, independent of lifespan, so **the thing under test
   is fully exercised** — real uvicorn, real `CORSMiddleware`, real
   `auth_and_security_middleware`, real routes.
2. **`DEV_LOCALHOST_BYPASS=1`** — read from the *process* env at
   `backend/api/auth.py:150`, not from `.env`. The operator's `:8000` process has it
   (proven below: a bare localhost curl reaches the route and 500s instead of 401ing).
   **`backend/.env` was not edited.** Without this the probe returned **401**, and the
   step's own `grep`-based verification command would have "passed" on the 401's CORS
   header — a false pass, caught and recorded in §F.
3. **Port 8001** — the operator's `:8000` and `:3000` were never touched. Confirmed in §G.

**Consequence for the operator:** this fix is **inert on `:8000` until the backend is
restarted** (middleware is built once — `starlette/applications.py:88-90`;
`add_middleware` raises `RuntimeError` after startup). The restart is owed, and `79.55`
must be answered first — both are on the batched operator ask list.

**Do NOT cite the immutable verification command's exit code as evidence today.** Run
verbatim against the un-restarted `:8000`, it emits
`access-control-allow-origin: http://localhost:3000` and exits 0 — **on a 404**, because
`/api/__force_500_probe` does not exist in that running process and a 404 always carried
the CORS header. The command cannot currently distinguish pass from fail on `:8000`; the
binding evidence is the in-process suite (which asserts `status_code == 500` *before* any
CORS assertion) plus the `:8001` captures below.

Playwright: `@playwright/mcp@0.0.76` as connected this session, viewport 1440x900,
isolated skip-auth Next dev server on `:3100` with
`PLAYWRIGHT_DIST_DIR=.next-audit-3100` and `NEXT_PUBLIC_API_URL=http://localhost:8001`.
The operator's `:3000` was never driven.

---

## §B. BEFORE — the operator's live `:8000` (pre-fix code), 15:48 UTC

### B1. 500 path, allowed origin

```
$ curl -s -D - -o /dev/null -m 300 -H 'Origin: http://localhost:3000' \
       http://localhost:8000/api/signals/AAPL
HTTP/1.1 500 Internal Server Error
date: Sat, 25 Jul 2026 15:48:27 GMT
server: uvicorn
content-length: 21
content-type: text/plain; charset=utf-8

[curl] http_code=500 total=18.102917s
```

**NO `access-control-allow-origin`. NO OWASP headers. NO `x-response-time`.**
Body is Starlette's plain-text `Internal Server Error` (21 bytes).

### B2. 200 control, same origin — proves the headers work on the success path

```
HTTP/1.1 200 OK
...
access-control-allow-credentials: true
access-control-allow-origin: http://localhost:3000
vary: Origin
x-response-time: 4970ms
x-content-type-options: nosniff
x-frame-options: DENY
x-xss-protection: 0
referrer-policy: strict-origin-when-cross-origin
cache-control: no-store
permissions-policy: camera=(), microphone=(), geolocation=()

[curl] http_code=200 total=4.971933s
```

### B3. Measured middleware nesting on the live app object

```
user_middleware (index 0 = OUTERMOST):
  [0] BaseHTTPMiddleware  dispatch=auth_and_security_middleware
  [1] CORSMiddleware

exception_handlers registered: {'HTTPException': ..., 'RequestValidationError': ...,
                                'WebSocketRequestValidationError': ...}

=> ServerErrorMiddleware -> auth_and_security_middleware -> CORSMiddleware
   -> ExceptionMiddleware -> router
```

No catch-all `Exception`/`500` handler existed.

---

## §C. AFTER — `:8001` rig, phase-80.2 code, 16:09 UTC

### C1. The immutable verification command (port swapped to the rig)

```
$ curl -s -D - -o /dev/null -H 'Origin: http://localhost:3000' \
       .../api/__force_500_probe | grep -i 'access-control-allow-origin'
access-control-allow-origin: http://localhost:3000
```

### C2. Criterion 1 — the 500, full headers. **Status is 500, not 404 and not 401.**

```
HTTP/1.1 500 Internal Server Error
date: Sat, 25 Jul 2026 16:09:42 GMT
server: uvicorn
content-length: 34
content-type: application/json
access-control-allow-credentials: true
access-control-allow-origin: http://localhost:3000     <-- criterion 1
vary: Origin
x-response-time: 1ms                                   <-- middleware tail now runs
x-content-type-options: nosniff                        <-- addendum (ii)
x-frame-options: DENY
x-xss-protection: 0
referrer-policy: strict-origin-when-cross-origin
cache-control: no-store
permissions-policy: camera=(), microphone=(), geolocation=()

[curl] http_code=500
```

Body — JSON, no traceback (phase-75.16 leak class):

```
{"detail":"Internal Server Error"}
```

### C3. Criterion 4 — the SAME 500 to a **disallowed** origin

```
$ curl -s -D - -o /dev/null -H 'Origin: https://evil.example' .../api/__force_500_probe
HTTP/1.1 500 Internal Server Error
date: Sat, 25 Jul 2026 16:09:42 GMT
server: uvicorn
content-length: 34
content-type: application/json
access-control-allow-credentials: true
x-response-time: 1ms
x-content-type-options: nosniff
x-frame-options: DENY
x-xss-protection: 0
referrer-policy: strict-origin-when-cross-origin
cache-control: no-store
permissions-policy: camera=(), microphone=(), geolocation=()

[curl] http_code=500
```

**`access-control-allow-origin` is ABSENT.** The allow-list is unchanged and nothing
echoes `*`. (`access-control-allow-credentials` is present for both origins — that is
pre-existing Starlette behaviour from `cors.py` `simple_headers`, unchanged by this step
and explicitly out of scope; criterion 4 is about `access-control-allow-origin` only.)

### C4. 200 control on the same rig, same origin

```
HTTP/1.1 200 OK
content-type: application/json
access-control-allow-credentials: true
access-control-allow-origin: http://localhost:3000
vary: Origin
x-response-time: 0ms
x-content-type-options: nosniff
... (all six OWASP headers)
[curl] http_code=200
```

**Paired 500/200 on the same origin, header now present on both — the required artifact.**

---

## §D. Addendum (iii) — the 500 reaches PerfTracker AND is visible

```
$ curl -s ".../api/observability/latency?window=300"
{
    "p50": 0.9,
    "p95": 17.4,
    "p99": 21.2,
    "total_requests": 6,
    "window_seconds": 300,
    "cache_hit_rate_pct": 0.0,
    "error_count": 5,
    "error_rate_pct": 83.3,
    "per_endpoint": {
        "/api/health": {
            "count": 1, "p50_ms": 22.1, "p95_ms": 22.1,
            "error_count": 0, "error_rate_pct": 0.0
        },
        "/api/__force_500_probe": {
            "count": 5, "p50_ms": 0.9, "p95_ms": 2.8,
            "error_count": 5, "error_rate_pct": 100.0
        }
    }
}
```

Before this step the failing endpoint contributed **nothing at all** to this surface. A
bare `count` bump would still not have shown *error-ness*, because `summarize()` discarded
`status_code`; `error_count`/`error_rate_pct` are the additive fields that make the
failure visible rather than merely counted.

## §E. Do-no-harm evidence

### E1. The traceback still reaches the log (`logger.exception`, not swallowed)

```
Unhandled exception serving GET /api/__force_500_probe: RuntimeError('phase-80.2
__force_500_probe: deliberate unhandled exception used to verify that a 500 carries
CORS + OWASP headers and is recorded by PerfTracker. This route always raises by design.')
Traceback (most recent call last):
  File ".../backend/middleware/catch_all_errors.py", line 103, in __call__
    await self.app(scope, receive, _send)
  File ".../starlette/middleware/exceptions.py", line 63, in __call__
    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
  ...
```

5 occurrences for 5 probe requests — one per failure, none silently swallowed.

### E2. SSE is not regressed by the new middleware layer

`backend/api/mas_events.py:36` serves `text/event-stream`. Same call, both processes:

| | `:8001` (phase-80.2 code) | `:8000` (pre-fix control) |
|---|---|---|
| status | `HTTP/1.1 200 OK` | `HTTP/1.1 200 OK` |
| content-type | `text/event-stream; charset=utf-8` | `text/event-stream; charset=utf-8` |
| CORS | `access-control-allow-origin: http://localhost:3000` | same |
| nosniff | present | present |
| body bytes in 5s (idle stream) | 0 | 0 |

Identical. A mid-stream failure still re-raises (headers are already on the wire and
cannot be rewritten) — that case is documented in the middleware, not papered over.

---

## §F. A false pass this check caught

The step's immutable verification command is a bare
`... | grep -i 'access-control-allow-origin'`. On the first rig attempt the probe returned
**401** (the process env lacked `DEV_LOCALHOST_BYPASS`), and the 401 path at
`backend/main.py:536-539` echoes CORS headers by hand — so **the grep matched and the
command "passed" against a response that was never a 500**.

The same trap exists for a 404: measured 2026-07-25, `GET /api/__force_500_probe` while
the route did not exist returned `404` **WITH** `access-control-allow-origin`, because a
404 is an `HTTPException` handled by the innermost `ExceptionMiddleware`.

This is why every capture above prints the **full header block including the status
line**, and why `test_probe_route_genuinely_raises_a_500` asserts `500` before any of the
CORS assertions are allowed to mean anything.

---

## §G. Playwright — criterion 2 + "no CORS-block error"

Skip-auth `:3100` rig, `NEXT_PUBLIC_API_URL=http://localhost:8001`. Navigated to
`/signals`, typed `AAPL`, clicked **Fetch Signals**. `/api/signals/AAPL` still genuinely
500s (that is step 80.1, deliberately not fixed here) — so this is a real broken endpoint,
not a synthetic one.

**Console, verbatim:**

```
[INFO]  %cDownload the React DevTools ...
[ERROR] Failed to load resource: the server responded with a status of 404 (Not Found)
        @ http://localhost:3100/favicon.ico:0
[LOG]   [Fast Refresh] rebuilding
[LOG]   [Fast Refresh] done in 764ms
[ERROR] Failed to load resource: the server responded with a status of 500
        (Internal Server Error) @ http://localhost:8001/api/signals/AAPL:0
```

**No `has been blocked by CORS policy` line. No `net::ERR_FAILED`.** The browser now
surfaces the real HTTP status. (The `favicon.ico` 404 is a separate, already-catalogued
phase-80 P2 defect — not introduced here.)

Compare with what the audit recorded before the fix:

```
Access to fetch at 'http://localhost:8000/api/signals/AAPL' from origin
'http://localhost:3100' has been blocked by CORS policy: No 'Access-Control-Allow-Origin'
header is present on the requested resource.
... net::ERR_FAILED
```

**Criterion 2 — the rendered UI message (accessibility snapshot, verbatim):**

```
- paragraph [ref=e118]: Server error on /api/signals/AAPL. Check the backend logs for details.
```

This replaces the operator's screenshotted
`Network error calling /api/signals/AAPL: Load failed`. It names a **server error** and
does **not** tell the operator to go restart a healthy backend.

Screenshot: `handoff/current/captures_80.2/80.2_signals_server_error_message.png`.

### Teardown + operator-instance integrity

```
:3100 -> 0 listeners
:8001 -> 0 listeners
:3000/       -> 302   (healthy authed-instance signature)
:3000/login  -> 200
:8000/api/health -> 200
:8000 pid -> 70791    (same pid as at session start => NOT restarted)
```

`frontend/tsconfig.json` and `frontend/next-env.d.ts` were rewritten by `next dev` to
point at `.next-audit-3100` (the documented side effect) and were restored to their HEAD
contents; md5s verified back to the pre-run baseline
`cecfaa5d04f97bf443b8750d944606f9` / `ba64ff7d54714a8f64db89b1003207d8`, and
`git status` on both is clean.
