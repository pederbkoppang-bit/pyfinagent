# Research Brief: phase-23.5.3.1
# Fix Docker-alias hostname in `_send_morning_digest` + `_send_evening_digest`

Tier assumption: `simple` (as specified by caller).
Accessed: 2026-05-09.

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|---------------------|
| https://12factor.net/config | 2026-05-09 | doc | WebFetch | "The twelve-factor app stores config in environment variables ... can be modified between deployments without altering code" |
| https://docs.docker.com/compose/how-tos/networking/ | 2026-05-09 | doc | WebFetch | "Service names are only resolvable within the Docker network, not from the host machine itself." Canonical workaround: expose ports + use `http://localhost:PORT` |
| https://www.python-httpx.org/advanced/clients/ | 2026-05-09 | doc | WebFetch | "`base_url` allows you to prepend a URL to all outgoing requests"; `ConnectError` when DNS resolution fails or connection refused |
| https://apscheduler.readthedocs.io/en/3.x/modules/events.html | 2026-05-09 | doc | WebFetch | `EVENT_JOB_EXECUTED` fires when a job completes without raising to the scheduler; if `except Exception` swallows the error internally, APScheduler sees success |
| https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/ | 2026-05-09 | doc | WebFetch | `BaseSettings` field with default value is overridden by matching env var at instantiation; `env_prefix` applies consistently |
| https://oneuptime.com/blog/post/2026-02-03-python-httpx-async-requests/view | 2026-05-09 | blog | WebFetch | httpx `base_url` pattern for localhost vs Docker; `ConnectError` = DNS failure or connection refused; treat as non-recoverable without retry |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://oneuptime.com/blog/post/2026-02-20-twelve-factor-app-guide/view | blog | Twelve-factor covered by canonical source above |
| https://forums.docker.com/t/precedence-of-dns-entry-vs-compose-service-name/120967 | forum | Docker DNS scope covered by official Docker docs |
| https://github.com/agronholm/apscheduler/discussions/1016 | forum | APScheduler behavior confirmed by official docs |
| https://github.com/agronholm/apscheduler/issues/652 | issue | Supplemental; confirms exceptions swallowed = EVENT_JOB_EXECUTED |
| https://github.com/encode/httpx/discussions/3076 | forum | base_url covered by official httpx docs |
| https://docs.pydantic.dev/2.0/usage/pydantic_settings/ | doc | Older version; current doc fetched in full above |
| https://securebin.ai/blog/docker-environment-variables-guide/ | blog | Docker env covered by official Docker docs |
| https://www.cherryservers.com/blog/set-docker-environment-variables | blog | Supplemental; covered |
| https://vsupalov.com/docker-arg-env-variable-guide/ | blog | Supplemental |
| https://copyprogramming.com/howto/how-do-i-set-environment-variables-in-a-python-script-using-a-dockerfile | blog | Community tier; not load-bearing |

## Recency scan (2024-2026)

Searched: "Docker Compose service name DNS resolution host process outside container" (year-less),
"httpx AsyncClient BASE_URL configuration pattern localhost 2025" (2025 window),
"Python module-level constant localhost URL Docker host override env var pattern 2026" (2026 frontier),
"twelve factor app configuration environment variables URL base 2026" (2026 frontier),
"pydantic-settings BaseSettings environment variable override URL default 2025" (2025 window).

Result: no new findings from 2024-2026 that supersede the canonical Docker Compose DNS scoping rule
(Docker docs remain authoritative). The 2026 oneuptime.com httpx guide reinforces the base_url
pattern with no conceptual changes. Twelve-factor principles unchanged since original publication.
Pydantic-settings env-override behavior stable through v2.x series (2024-2026).

---

## Key findings

1. **Docker Compose service-name DNS is container-scoped, not host-scoped.** "Service names are only resolvable within the Docker network, not from the host machine itself." (Docker Compose Networking docs, 2026-05-09). A host process (the slack-bot running on Mac) hitting `http://backend:8000` will get a DNS resolution failure -- no connection is ever attempted.

2. **When `except Exception` swallows the error, APScheduler fires `EVENT_JOB_EXECUTED`.** "APScheduler only observes what the job returns or whether an unhandled exception propagates to the executor level." (APScheduler docs, 2026-05-09). So the heartbeat listener sees `status="ok"`, the dashboard shows green, and the operator receives no Slack message.

3. **Twelve-factor recommends env vars for all per-deployment URLs**, especially "per-deploy values such as the canonical hostname." (12factor.net/config, 2026-05-09). The `os.environ.get("KEY", "default")` fallback pattern satisfies this while keeping code deployable without mandatory env setup.

4. **Pydantic BaseSettings env-override pattern is well-established.** A field with a default is overridden by the matching env var at instantiation, without code changes. (Pydantic Settings docs, 2026-05-09). This is Option D's foundation.

5. **httpx `base_url` client-level configuration** centralizes URL management: `async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client: r = await client.get("/api/...")`. (httpx docs, 2026-05-09). Reduces per-call string duplication.

6. **`commands.py` already uses `http://localhost:8000`**, confirming localhost is the established pattern for non-container slack-bot code in this repo. (Internal grep, line 22.)

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/slack_bot/scheduler.py` | 1-35 | Module-level constants: `_BACKEND_URL`, `_HEARTBEAT_URL`, `_HEALTH_PROBE_URL` | `_BACKEND_URL` still points to Docker alias; only watchdog uses `_HEALTH_PROBE_URL` |
| `backend/slack_bot/scheduler.py` | 205-228 | `_send_morning_digest` | Two `_BACKEND_URL` call sites: lines 211, 214 |
| `backend/slack_bot/scheduler.py` | 230-252 | `_send_evening_digest` | Two `_BACKEND_URL` call sites: lines 236, 239 |
| `backend/slack_bot/commands.py` | 22 | Separate `_BACKEND_URL = "http://localhost:8000"` | Already correct; already uses localhost |
| `backend/slack_bot/formatters.py` | 309-351 | `format_morning_digest(portfolio_data: dict, recent_reports: list)` | Tolerant of empty dict / empty list inputs |
| `backend/slack_bot/formatters.py` | 354-402 | `format_evening_digest(portfolio_data: dict, trades_today: list)` | Tolerant of empty dict / empty list inputs |
| `tests/slack_bot/test_watchdog_alert_semantics.py` | 1-154 | Watchdog semantics tests from 23.5.2.6 | Template for digest tests; fixtures: `_FakeAsyncClient`, `_fake_response`, `_fake_app` |
| `tests/slack_bot/` | (dir) | 11 test files | No file covering `_send_morning_digest` or `_send_evening_digest` |

---

### `_send_morning_digest` verbatim body (scheduler.py:205-228)

```python
async def _send_morning_digest(app: AsyncApp):
    """Fetch portfolio performance and post morning digest."""
    settings = get_settings()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            portfolio_res = await client.get(f"{_BACKEND_URL}/api/portfolio/performance")
            portfolio_data = portfolio_res.json() if portfolio_res.status_code == 200 else {}

            reports_res = await client.get(f"{_BACKEND_URL}/api/reports/?limit=5")
            reports_data = reports_res.json() if reports_res.status_code == 200 else []

        blocks = format_morning_digest(portfolio_data, reports_data)

        await app.client.chat_postMessage(
            channel=settings.slack_channel_id,
            blocks=blocks,
            text=f"PyFinAgent Morning Digest -- {datetime.now().strftime('%B %d, %Y')}",
        )
        logger.info("Morning digest sent")

    except Exception:
        logger.exception("Failed to send morning digest")
```

Call sites: `_BACKEND_URL` at lines 211 and 214.

### `_send_evening_digest` verbatim body (scheduler.py:230-252)

```python
async def _send_evening_digest(app: AsyncApp):
    """Fetch end-of-day portfolio summary and post evening digest."""
    settings = get_settings()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            portfolio_res = await client.get(f"{_BACKEND_URL}/api/portfolio/performance")
            portfolio_data = portfolio_res.json() if portfolio_res.status_code == 200 else {}

            trades_res = await client.get(f"{_BACKEND_URL}/api/paper-trading/trades?limit=10")
            trades_data = trades_res.json() if trades_res.status_code == 200 else []

        blocks = format_evening_digest(portfolio_data, trades_data)

        await app.client.chat_postMessage(
            channel=settings.slack_channel_id,
            blocks=blocks,
            text=f"PyFinAgent Evening Digest -- {datetime.now().strftime('%B %d, %Y')}",
        )
        logger.info("Evening digest sent")

    except Exception:
        logger.exception("Failed to send evening digest")
```

Call sites: `_BACKEND_URL` at lines 236 and 239.

---

## `_BACKEND_URL` call sites — full repo inventory

**scheduler.py (problematic -- Docker alias):**
- Line 24: definition `_BACKEND_URL = "http://backend:8000"`
- Line 211: `_send_morning_digest` -- `/api/portfolio/performance`
- Line 214: `_send_morning_digest` -- `/api/reports/?limit=5`
- Line 236: `_send_evening_digest` -- `/api/portfolio/performance`
- Line 239: `_send_evening_digest` -- `/api/paper-trading/trades?limit=10`

**commands.py (already correct -- localhost):**
- Line 22: independent definition `_BACKEND_URL = "http://localhost:8000"` -- SEPARATE constant, not shared
- Lines 75, 106, 113, 138, 159: use of commands.py's own `_BACKEND_URL` (already localhost) -- NOT BROKEN

Key observation: `commands.py` defines its own `_BACKEND_URL` at `"http://localhost:8000"` independent of `scheduler.py`'s `_BACKEND_URL`. These are two module-scope constants with the same name in different modules. The commands.py version is already correct; only scheduler.py's constant is broken. The 4 call sites in scheduler.py (lines 211, 214, 236, 239) are the complete blast radius.

---

## Consensus vs debate

Consensus: Docker Compose service-name DNS does not resolve on the host; localhost is the correct target for host processes. (Docker docs, commands.py precedent, 23.5.2.6 watchdog fix.) No debate.

Option A (two named path constants) vs B/C/D: practitioner consensus favors minimal blast radius for a local-only deployment (B or C), with env-var flexibility as a bonus (D). Twelve-factor mandates env-var approach (D) for any URL that varies per deployment. However, since pyfinagent is local-only with no `.env` access in this phase, Option B provides correctness without adding new configuration surface.

## Pitfalls

- **Pitfall 1:** Modifying `_BACKEND_URL = "http://backend:8000"` directly (Option C) without a comment will silently break any future Docker-compose deployment without a visible warning. The 23.5.2.6 approach of keeping the old constant and adding a new one is preferable for documentation.
- **Pitfall 2:** Option D requires `import os` at module top and a comment explaining the env var name -- easy to get wrong.
- **Pitfall 3:** `format_morning_digest` and `format_evening_digest` both accept empty dicts/lists gracefully (confirmed at formatters.py:319, 374). Tests can pass `{}` / `[]` without real BQ data.
- **Pitfall 4:** The `_FakeAsyncClient` in test_watchdog_alert_semantics.py pops responses in FIFO order via `_responses.pop(0)`. For morning/evening digest tests that call `client.get` twice, two responses must be queued.

---

## Application to pyfinagent

The fix maps exactly to the 23.5.2.6 watchdog template:

1. Add `_LOCAL_BACKEND_URL = "http://127.0.0.1:8000"` near line 35 (after `_HEALTH_PROBE_URL`).
2. In `_send_morning_digest` (lines 211, 214): replace `_BACKEND_URL` with `_LOCAL_BACKEND_URL`.
3. In `_send_evening_digest` (lines 236, 239): replace `_BACKEND_URL` with `_LOCAL_BACKEND_URL`.
4. Leave `_BACKEND_URL = "http://backend:8000"` at line 24 with a comment that it is aspirational for Docker-compose future resurrection, but currently unused.

No other files in `backend/` contain `_BACKEND_URL` (commands.py has its own independent constant already at localhost). The blast radius is exactly 4 lines in one file.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total (16 collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (scheduler.py, commands.py, formatters.py, tests/slack_bot/)
- [x] No contradictions; consensus is clear
- [x] All claims cited per-claim

---

## Answers to Main's three decision questions

### Answer 1: Recommended option -- Option B

**Add `_LOCAL_BACKEND_URL = "http://127.0.0.1:8000"` and replace `_BACKEND_URL` in the two digest functions.**

Rationale:
- Minimal blast radius: only 4 line-level changes in one file.
- Mirrors the 23.5.2.6 watchdog pattern exactly (`_HEALTH_PROBE_URL` was added rather than mutating `_BACKEND_URL`).
- Preserves `_BACKEND_URL = "http://backend:8000"` with documentation value for future Docker-compose resurrection.
- Does not add env-var surface area (Option D) which would require `.env` editing and operator friction with no benefit in a local-only deployment.
- Does not require two named path constants (Option A) -- verbose with no gain over a single base constant.
- Option C (mutating `_BACKEND_URL` itself) eliminates the documentation signal and silently breaks Docker if ever resurrected.

Trade-off summary:
| Option | Blast radius | Docker resurrection | Operator friction | Recommended |
|--------|-------------|---------------------|-------------------|-------------|
| A (two path constants) | 4 lines + 2 new consts | Fine | Low | No -- verbose |
| B (one `_LOCAL_BACKEND_URL`) | 4 lines + 1 new const | Fine (`_BACKEND_URL` survives) | None | **YES** |
| C (mutate `_BACKEND_URL`) | 4 lines, no new const | Broken silently | None | No -- loses Docker doc |
| D (env-driven) | 4 lines + import + env var | Flexible | Medium (env var setup) | No -- overkill for local-only |

### Answer 2: Other `_BACKEND_URL` call sites

**Only the 4 lines in `scheduler.py` are broken.** `commands.py` defines its own independent `_BACKEND_URL = "http://localhost:8000"` at line 22 and is already correct. No other files in `backend/` reference `_BACKEND_URL` from `scheduler.py`. All 4 call sites (lines 211, 214, 236, 239) are in scope for phase-23.5.3.1. There are no out-of-scope modules requiring a separate substep.

### Answer 3: Test design

Reuse the fixtures from `tests/slack_bot/test_watchdog_alert_semantics.py` verbatim: `_FakeAsyncClient`, `_fake_response`, `_fake_app`. Add a new file `tests/slack_bot/test_digest_url_semantics.py` with the following test cases:

**For `_send_morning_digest`:**

```python
import asyncio
from types import SimpleNamespace
from unittest.mock import patch
from backend.slack_bot import scheduler as scheduler_mod
# (same imports / helpers as test_watchdog_alert_semantics.py)

def test_morning_digest_uses_localhost_not_docker_alias():
    """Regression guard: digest calls MUST hit 127.0.0.1, not backend:8000."""
    app = _fake_app()
    portfolio_resp = _fake_response(200, {"total_pnl": 100, "total_return_pct": 1.0})
    reports_resp = _fake_response(200, [])
    cm, fake = _patch_client_with([portfolio_resp, reports_resp])
    with patch.object(scheduler_mod, "get_settings",
                      return_value=SimpleNamespace(slack_channel_id="C_TEST")):
        with cm:
            asyncio.run(scheduler_mod._send_morning_digest(app))
    assert len(fake.calls) == 2
    for url in fake.calls:
        assert "127.0.0.1:8000" in url or "localhost:8000" in url, f"URL regressed: {url!r}"
        assert "://backend:8000" not in url, f"Docker alias leaked: {url!r}"

def test_morning_digest_posts_to_slack_on_success():
    """On 200 responses, chat_postMessage must be called exactly once."""
    app = _fake_app()
    portfolio_resp = _fake_response(200, {"total_pnl": 50, "total_return_pct": 0.5})
    reports_resp = _fake_response(200, [])
    cm, _ = _patch_client_with([portfolio_resp, reports_resp])
    with patch.object(scheduler_mod, "get_settings",
                      return_value=SimpleNamespace(slack_channel_id="C_TEST")):
        with cm:
            asyncio.run(scheduler_mod._send_morning_digest(app))
    assert app.client.chat_postMessage.await_count == 1
```

**For `_send_evening_digest`:**

```python
def test_evening_digest_uses_localhost_not_docker_alias():
    """Regression guard: evening digest calls MUST hit 127.0.0.1, not backend:8000."""
    app = _fake_app()
    portfolio_resp = _fake_response(200, {"total_pnl": 200, "total_return_pct": 2.0})
    trades_resp = _fake_response(200, [])
    cm, fake = _patch_client_with([portfolio_resp, trades_resp])
    with patch.object(scheduler_mod, "get_settings",
                      return_value=SimpleNamespace(slack_channel_id="C_TEST")):
        with cm:
            asyncio.run(scheduler_mod._send_evening_digest(app))
    assert len(fake.calls) == 2
    for url in fake.calls:
        assert "127.0.0.1:8000" in url or "localhost:8000" in url, f"URL regressed: {url!r}"
        assert "://backend:8000" not in url, f"Docker alias leaked: {url!r}"

def test_evening_digest_posts_to_slack_on_success():
    """On 200 responses, chat_postMessage must be called exactly once."""
    app = _fake_app()
    portfolio_resp = _fake_response(200, {"total_pnl": 200, "total_return_pct": 2.0})
    trades_resp = _fake_response(200, [])
    cm, _ = _patch_client_with([portfolio_resp, trades_resp])
    with patch.object(scheduler_mod, "get_settings",
                      return_value=SimpleNamespace(slack_channel_id="C_TEST")):
        with cm:
            asyncio.run(scheduler_mod._send_evening_digest(app))
    assert app.client.chat_postMessage.await_count == 1
```

Key design notes:
- `_FakeAsyncClient._responses.pop(0)` -- morning/evening each make 2 `client.get` calls, so 2 responses must be queued in FIFO order (portfolio first, then reports/trades).
- `_patch_client_with` patches `scheduler_mod.httpx.AsyncClient` not `httpx.AsyncClient`, so it intercepts the context manager in the function body.
- The URL-assertion pattern mirrors `test_uses_localhost_probe_url_not_docker_alias` from the watchdog test (line 144-153).
- Empty lists for reports/trades still exercise the success path because `format_morning_digest` and `format_evening_digest` are both tolerant of empty inputs (formatters.py:331, 374).

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "gate_passed": true
}
```
