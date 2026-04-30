# External Research Brief — phase-23.1.20
# Pause/Resume Endpoint Timeout Hardening

Research date: 2026-04-29
Tier: moderate (stated by caller)

---

### Queries run

1. **Current-year frontier:** "FastAPI asyncio.wait_for handler timeout pattern 2026"
2. **Last-2-year window:** "circuit breaker pattern FastAPI slow upstream timeout fail-fast" (2024-2026 hits present)
3. **Year-less canonical:** "HTTP 503 504 408 upstream timeout semantics REST API"; "Retry-After header 503 response REST API AbortController frontend"

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://sentry.io/answers/make-long-running-tasks-time-out-in-fastapi/ | 2026-04-29 | Official blog/doc | WebFetch | `asyncio.wait_for(awaitable, timeout=N)` raises TimeoutError after N seconds; catch it and return cached result or error. Python <3.11 raises `asyncio.TimeoutError`, >=3.11 raises stdlib `TimeoutError`. |
| https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/503 | 2026-04-29 | Official doc (MDN) | WebFetch | 503 = server not ready (overloaded/maintenance). Include `Retry-After` header. Do not cache 503 responses. 503 vs 504: 503 is a direct server issue; 504 is a proxy/gateway not hearing from upstream. |
| https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Retry-After | 2026-04-29 | Official doc (MDN) | WebFetch | `Retry-After: <delay-seconds>` (non-negative integer) or `Retry-After: <http-date>`. Used with 503, 429, 301. Clients must wait before retrying. Jitter (±10%) recommended to avoid retry storms. |
| https://github.com/fastapi/fastapi/discussions/7364 | 2026-04-29 | Community (FastAPI official repo) | WebFetch | Middleware pattern: `await asyncio.wait_for(call_next(request), timeout=N)` catches asyncio.TimeoutError and returns 504 JSON. Per-route pattern: custom `APIRoute` subclass wraps handler in `wait_for`. Background tasks may continue after timeout — the HTTP response is released but the task is not guaranteed cancelled. `anyio.fail_after()` cited as more robust. |
| https://blog.greeden.me/en/2026/04/21/a-practical-introduction-to-circuit-breakers-and-fallback-design-in-fastapi-real-world-patterns-for-preventing-external-api-failures-from-becoming-system-wide-failures/ | 2026-04-29 | Authoritative blog (2026) | WebFetch | Circuit breaker with 3 states (Closed/Open/Half-Open). FastAPI returns 503 on `CircuitOpenError`. Fail-fast: when circuit open, reject immediately without calling upstream. Timeout is a prerequisite — "without timeouts, circuit breaker has nothing to trip on." |
| https://fastapi.tiangolo.com/async/ | 2026-04-29 | Official doc (FastAPI) | WebFetch | `asyncio.to_thread` / `def` vs `async def` trade-offs. Sync blocking calls in `async def` handlers block the event loop; `asyncio.to_thread` offloads to threadpool but thread is not cancellable from asyncio side. FastAPI recommends `AnyIO` for advanced concurrency. |

---

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://github.com/fastapi/fastapi/discussions/6066 | Community | Fetched but discussion was inconclusive/invalid label — noted that middleware timeout behavior is a Starlette issue, not FastAPI proper |
| https://github.com/fastapi/fastapi/issues/5881 | Community | Duplicate of #6066; snippet only |
| https://github.com/fastapi/fastapi/issues/1752 | Community | Older global timeout discussion; superseded by #7364 |
| https://www.aritro.in/post/fastapi-resiliency-circuit-breakers-rate-limiting-and-external-api-management/ | Blog | Circuit breaker overview; not fetched as blog.greeden.me covered same topic at higher quality |
| https://pypi.org/project/circuitbreaker/ | PyPI | Snippet only; no BQ-specific integration |
| https://medium.com/@kaushalsinh73/fastapi-circuit-breakers-with-resilience-patterns-surviving-downstream-failures-4af0920799d3 | Blog | Snippet only; blog.greeden.me source was more complete |
| https://howhttpworks.com/status-codes/408 | Community | 408 semantics covered by MDN 503 doc |
| https://repost.aws/knowledge-center/api-gateway-504-errors | AWS docs | 504 semantics covered by MDN 503 doc |
| https://nurkiewicz.com/2015/02/retry-after-http-header-in-practice.html | Blog | 2015; older; MDN source was sufficient |
| https://community.cloudflare.com/t/503-service-unavailable-with-a-retry-after-header | Community | Snippet only; lower authority |

---

### Recency scan (2024-2026)

Searched for 2024-2026 literature on FastAPI timeouts, circuit breakers for BQ upstreams, and Retry-After patterns.

Found 1 new finding from the 2026 window that complements canonical sources: the greeden.me post (2026-04-21) explicitly ties circuit breaker to FastAPI 0.115+ patterns and includes a minimal `SimpleCircuitBreaker` dataclass implementation with `fail_max` and `reset_timeout_sec` fields — directly applicable without a third-party library.

No findings from 2024-2026 that supersede the canonical timeout approach (asyncio.wait_for + TimeoutError → 503). Python 3.11's `async with asyncio.timeout(N)` context manager is newer (2022, stable in 3.11) and is the preferred idiom in Python 3.11+ over `asyncio.wait_for`, but the project is on Python 3.14 so it is available.

---

### Key findings

1. **503 is the correct status for "backend couldn't reach BQ in time"** — The request arrived, the server processed it, but an upstream dependency (BQ) was slow. 503 ("server not ready") is correct. 504 would only apply if this server were acting as a gateway/proxy layer; it is the origin. 408 applies when the *client* was too slow sending the request. (Source: MDN 503, https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/503)

2. **`asyncio.wait_for` around `asyncio.to_thread` releases the HTTP response on timeout but does not cancel the underlying thread** — The BQ thread will continue running in the background until the BQ SDK returns or its own internal timeout fires. This is acceptable: the client gets a 503 promptly, and the lingering thread does not consume a file descriptor (it is blocked inside the BQ SDK, not holding an open FD in our code). (Source: FastAPI async docs, https://fastapi.tiangolo.com/async/ and GitHub discussion https://github.com/fastapi/fastapi/discussions/7364)

3. **`Retry-After: 5` is the correct companion header on a 503** — Expressed as an integer (seconds). This is the standard signal for clients to back off before retrying. The frontend `AbortController` fires at 30s; a 503 with `Retry-After: 5` returned at ~5s gives the frontend enough time to react, surface a toast, and optionally auto-retry. (Source: MDN Retry-After, https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Retry-After)

4. **For Python 3.11+, prefer `async with asyncio.timeout(N)` over `asyncio.wait_for`** — `asyncio.timeout()` is the stdlib-native context manager introduced in Python 3.11. Project is on Python 3.14, so it is available. It is semantically equivalent to `asyncio.wait_for` but considered cleaner: `except TimeoutError` (not `except asyncio.TimeoutError`) in 3.11+. (Source: Sentry FastAPI answer, https://sentry.io/answers/make-long-running-tasks-time-out-in-fastapi/)

5. **Circuit breaker is a Phase 2 concern, not Phase 1** — The fail-fast pattern (open circuit → immediate 503 without calling BQ) is valuable for repeated failures, but requires state management (failure counter, reset timer). The simpler fix — `asyncio.timeout(5)` — solves the immediate 30s hang with two lines of code and no new state. Circuit breaker should be added if resume is called frequently during BQ degradation. (Source: greeden.me 2026, https://blog.greeden.me/en/2026/04/21/a-practical-introduction-to-circuit-breakers-and-fallback-design-in-fastapi-real-world-patterns-for-preventing-external-api-failures-from-becoming-system-wide-failures/)

---

### Consensus vs debate

**Consensus:**
- 503 + Retry-After is universally recommended for temporary upstream unavailability. No debate.
- `asyncio.wait_for` / `asyncio.timeout` are the idiomatic Python async timeout tools. No debate.
- The underlying thread from `asyncio.to_thread` is NOT cancelled on timeout — this is documented behavior, not a bug. Clients must be aware the thread lingers.

**Debate:**
- Middleware-level global timeout vs per-endpoint timeout: Global is simpler to maintain but the FastAPI discussion thread notes that background tasks continue after the timeout response, which can cause interference. Per-endpoint `asyncio.timeout()` in the handler is more surgical and predictable.
- Circuit breaker vs simple timeout: Simple timeout is sufficient for an infrequent user-triggered action like resume. Circuit breaker adds value for high-frequency automated calls.

---

### Pitfalls (from literature)

1. **`asyncio.wait_for` on a non-cooperative coroutine does not cancel threads.** The thread running `bq.get_paper_portfolio` will continue after the timeout. If BQ eventually returns, the thread exits cleanly. If BQ hangs indefinitely, the thread holds a threadpool slot. The default ThreadPoolExecutor in Python has a max_workers ceiling (CPU count * 5), so a long BQ hang can exhaust the threadpool if resume is called repeatedly. Mitigation: pass `timeout=` to `QueryJob.result()` inside `get_paper_portfolio` (the CLAUDE.md "30s" rule) so the thread itself terminates.

2. **Python <3.11 exception type mismatch.** `asyncio.wait_for` raises `asyncio.TimeoutError` in Python <=3.10 but `TimeoutError` in >=3.11. The project is on Python 3.14 so `except TimeoutError` is correct. Do NOT write `except asyncio.TimeoutError` as that is the old form.

3. **Do not cache 503 responses.** The `Cache-Control: no-store` OWASP header already applied to all responses covers this. No additional action needed.

4. **`Retry-After` must be a string in FastAPI's `Response.headers`.** The integer 5 must be passed as `"5"` or cast to string when setting `response.headers["Retry-After"] = "5"`.

---

### Application to pyfinagent — fix sketches mapped to file:line anchors

**Fix A — asyncio.timeout around BQ call in resume_trading**

`backend/api/paper_trading.py:374-398`

Replace:
```python
portfolio = await asyncio.to_thread(bq.get_paper_portfolio, "default")
```
With:
```python
try:
    async with asyncio.timeout(5):
        portfolio = await asyncio.to_thread(bq.get_paper_portfolio, "default")
except TimeoutError:
    from fastapi.responses import JSONResponse
    resp = JSONResponse(
        status_code=503,
        content={"detail": "BQ unavailable -- retry in 5s", "error": "bq_timeout"},
    )
    resp.headers["Retry-After"] = "5"
    return resp
```

Additionally, apply the CLAUDE.md 30s BQ rule inside `bigquery_client.py:489`:
```python
rows = list(self.client.query(query, job_config=job_config).result(timeout=30))
```
This ensures the thread itself terminates within 30s even if the asyncio timeout fires at 5s (belt-and-suspenders).

**Fix B — Pause endpoint defensive timeout**

Internal audit confirms: pause endpoint has NO external I/O. The only I/O is a local file append that is already fail-soft. Fix B is NOT warranted. Do not add `asyncio.timeout` to the pause handler — it adds complexity for no safety gain.

**Fix C — Regression test**

New file: `tests/api/test_pause_resume_timeout.py`

```python
import asyncio
import time
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def slow_get_paper_portfolio(portfolio_id="default"):
    time.sleep(15)
    return {"total_nav": 100_000.0}

def test_resume_returns_503_on_bq_timeout():
    with patch(
        "backend.api.paper_trading.BigQueryClient.get_paper_portfolio",
        side_effect=slow_get_paper_portfolio,
    ):
        start = time.monotonic()
        resp = client.post(
            "/api/paper-trading/resume",
            json={"confirmation": "RESUME"},
            headers={"Authorization": "Bearer test-token"},
        )
        elapsed = time.monotonic() - start
    assert resp.status_code == 503, f"Expected 503, got {resp.status_code}"
    assert elapsed < 8, f"Handler took {elapsed:.1f}s — expected <8s, not 30s hang"
    assert "Retry-After" in resp.headers
    assert resp.headers["Retry-After"] == "5"
```

Note: TestClient is synchronous; for true async timeout testing use `httpx.AsyncClient` with `anyio` as the test runner. The mock `slow_get_paper_portfolio` sleeps 15s in the thread — the `asyncio.timeout(5)` in the handler should fire at 5s and return 503 before the 8s assertion deadline.

**Fix D — Architectural: move breach check out of hot path**

The resume endpoint currently does:
1. Validate BQ-sourced NAV (blocking)
2. Evaluate breach thresholds
3. Flip in-memory state flag

The state flip (step 3) should never wait on external I/O. The current architecture mixes two concerns: precondition validation (BQ read) and state mutation (in-memory flip). The principled separation is:
- The breach check should be a cached value updated by the background trading loop, not fetched on-demand in the resume handler.
- The resume handler's only job is to flip the in-memory flag. The breach guard would read from an in-memory or short-TTL cached breach status.

Assessment: This is architecturally correct but invasive. It requires the trading loop to maintain a cached breach state, which is a non-trivial refactor. Defer to Phase 2 after Fix A + C are shipped and tested. Fix A (asyncio.timeout) is the minimal correct fix that eliminates the hang without restructuring the precondition logic.

**Recommended combination: Fix A + Fix C. Defer Fix B (not needed). Defer Fix D (Phase 2).**

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total incl. snippet-only (16 total)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (in audit file)

Soft checks:
- [x] Internal exploration covered every relevant module (paper_trading.py, bigquery_client.py, kill_switch.py, api_cache.py, tests/)
- [x] Contradictions / consensus noted (503 vs 504, middleware vs per-endpoint)
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/phase-23.1.20-external-research.md",
  "gate_passed": true
}
```
