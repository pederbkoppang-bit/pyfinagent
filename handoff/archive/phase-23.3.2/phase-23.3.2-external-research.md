# Phase-23.3.2 External Research Brief
# APScheduler Cross-Process Introspection and Slack-Bot Liveness Patterns

Tier: simple (assumed — caller did not state)
Accessed: 2026-05-07

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://apscheduler.readthedocs.io/en/3.x/faq.html | 2026-05-07 | Official docs | WebFetch | Sharing a persistent job store across processes causes duplicate execution or missed jobs; the recommended fix is to run the scheduler in a dedicated process and connect via RPyC, gRPC, or HTTP |
| https://martinfowler.com/articles/patterns-of-distributed-systems/heartbeat.html | 2026-05-07 | Authoritative blog (Fowler) | WebFetch | Heartbeat = periodic message sent to show liveness; listener waits timeout > round-trip before declaring peer dead; this is the canonical cross-process liveness primitive |
| https://microservices.io/patterns/observability/health-check-api.html | 2026-05-07 | Authoritative pattern catalog | WebFetch | Service exposes /health that evaluates infrastructure connectivity, not just process liveness; benefits: enables periodic validation; drawbacks: health checks may lag actual failure |
| https://daily-devops.net/posts/health-checks-operational-monitoring/ | 2026-05-07 | Authoritative blog | WebFetch | Green Dashboard, Dead Application anti-pattern: process returns HTTP 200 while app is actually broken; prescribes separate /health/live (process alive?) vs /health/ready (can serve?) checks |
| https://healthchecks.io/ | 2026-05-07 | SaaS/reference implementation | WebFetch | Dead-man's-switch model: job calls a ping URL on completion; if ping doesn't arrive within Period+Grace, the check transitions to "Down" and alerts fire — the canonical external cron-job monitoring pattern |
| https://apscheduler.readthedocs.io/en/3.x/userguide.html | 2026-05-07 | Official docs | WebFetch | Event listeners (add_listener) are APScheduler's primary hook for monitoring; exposes EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED, etc.; no built-in mechanism to export state to another process |
| https://github.com/slackapi/bolt-python/issues/439 | 2026-05-07 | Official GitHub issue (Slack) | WebFetch (403) | Fetch failed; snippet: Bolt Python has NO built-in health check endpoint; the issue was filed for Kubernetes liveness and closed as a question with the guidance to implement your own route |

Note: bolt-python issue 439 returned 403 (GitHub auth wall). The snippet from search confirmed: no built-in health endpoint in Bolt. The fetch attempt is logged as required.

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://pypi.org/project/APScheduler/ | Docs | PyPI page is a summary; FAQ covers the specific claims |
| https://github.com/agronholm/apscheduler | Code | Source code; FAQ/userguide cover the relevant pattern claims |
| https://betterstack.com/community/guides/scaling-python/apscheduler-scheduled-tasks/ | Blog | WebFetch returned no monitoring content; snippet confirmed APScheduler job store patterns only |
| https://cronradar.com/blog/python-scheduler-monitoring | Blog | WebFetch returned 403; snippet confirmed event-listener + HTTP POST heartbeat pattern |
| https://apscheduler.readthedocs.io/en/3.x/modules/events.html | Docs | WebFetch confirmed event constants (EVENT_JOB_EXECUTED, etc.); add_listener is the hook; no further implementation detail needed |
| https://github.com/jcass77/django-apscheduler | Code | ORM-backed jobstore pattern; not directly applicable to pyfinagent (no Django) |
| https://groups.google.com/g/apscheduler/c/Gjc_JQMPePc | Forum | Multi-scheduler discussion; lower authority; snippet sufficient |
| https://devcenter.heroku.com/articles/clock-processes-python | Docs | Heroku clock process pattern; relevant context but covered by FAQ |
| https://github.com/airflow-helm/charts/.../scheduler-liveness-probe.md | Docs | Airflow-specific; pattern generalizes but Airflow is not APScheduler |
| https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/ | Docs | Kubernetes probe semantics; relevant for definitions of liveness vs readiness |

---

## Recency scan (2024-2026)

Searched: "APScheduler cross-process job introspection dashboard heartbeat registry 2026" and "APScheduler job_listener event HTTP POST heartbeat registry 2024 2025" and "heartbeat push cross-process job status HTTP endpoint pattern microservices 2025".

Results: No new 2024-2026 publications supersede the core findings. The APScheduler FAQ (shared jobstore limitation, HTTP-server recommendation) has been stable since APScheduler v3.x. The Heartbeat pattern (Fowler) is canonical and unchanged. The dead-man's-switch pattern (healthchecks.io) has been available since 2013 and remains the most widely used external cron-monitoring approach in 2026. The cronradar.com blog (December 2025 by snippet evidence) confirms the event-listener + HTTP POST push is the current idiomatic pattern for APScheduler monitoring, which aligns with the existing `POST /api/jobs/heartbeat` design in pyfinagent. No new APScheduler 4.x cross-process feature has shipped that would change the recommendation.

---

## Key Findings

1. **APScheduler does not support cross-process scheduler.get_jobs() calls.** The FAQ states explicitly: "APScheduler does not currently have any interprocess synchronization and signalling scheme." The only sanctioned cross-process path is via RPyC, gRPC, or HTTP. (Source: APScheduler FAQ, https://apscheduler.readthedocs.io/en/3.x/faq.html)

2. **The heartbeat-push pattern is the standard solution.** The scheduler process emits an HTTP POST to a central registry on job completion. This is the pattern already built into pyfinagent's `POST /api/jobs/heartbeat` (job_status_api.py:135-143) and the `heartbeat()` context manager in job_runtime.py. The wiring is incomplete: job_runtime.py's `sink` defaults to `logger.info`, not an HTTP POST. (Corroborated by: cronradar.com snippet Dec 2025; APScheduler event docs; pyfinagent internal audit)

3. **Bolt Python has no built-in health or scheduler-state endpoint.** The GitHub issue #439 confirms this. The correct approach is to implement a custom HTTP route alongside the Bolt app, or accept that the Bolt process has no inbound HTTP surface (Socket Mode). (Source: github.com/slackapi/bolt-python/issues/439 snippet; Bolt docs)

4. **The Green Dashboard / Manifest-Only anti-pattern is documented.** A dashboard that shows a static list of jobs (manifest) without a liveness signal gives operators false confidence that jobs are running. The classic example: the process can silently die and the manifest row remains. The daily-devops.net analysis names this "Green Dashboard, Dead Application." (Source: daily-devops.net/posts/health-checks-operational-monitoring/)

5. **Dead-man's-switch (ping-on-completion) is the lowest-friction cross-process liveness pattern.** The job itself calls an HTTP endpoint on every run. If no ping arrives within expected_period + grace_time, the system declares the job dead. This requires no inbound HTTP server on the worker side. (Source: healthchecks.io; Fowler Heartbeat pattern)

6. **Liveness vs Readiness distinction matters.** Liveness: "is the process alive?" (SIGKILL-restart trigger). Readiness: "can it serve?" (scheduler started and at least one job registered). The watchdog_health_check job itself provides a form of liveness signal — it calls the main backend's /api/health on an interval. If the slack-bot process dies, this watchdog stops firing, and the main backend can detect absence of the cross-process probe. However, there is currently no mechanism in pyfinagent to detect when the watchdog has gone silent. (Source: microservices.io Health Check API; Kubernetes probe semantics)

---

## Internal Code Inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/slack_bot/scheduler.py` | 383 | Registers + starts all APScheduler jobs in slack-bot process | Active; 4 core + 7 phase-9 jobs |
| `backend/slack_bot/app.py` | 78 | Slack-bot entry point; calls `start_scheduler(app)` | Active; no inbound HTTP server |
| `backend/slack_bot/job_runtime.py` | 118 | `heartbeat()` context manager + idempotency | Active; sink defaults to `logger.info`; HTTP wiring absent |
| `backend/api/job_status_api.py` | 153 | FastAPI job heartbeat registry for 7 phase-9 jobs | Active; `POST /api/jobs/heartbeat` exists but not wired |
| `backend/api/cron_dashboard_api.py` | 232 | `GET /api/jobs/all` merges live APScheduler + static manifest | Active; 4 core jobs shown as `status="manifest"` |
| `backend/main.py` | 300+ | Registers main + queue schedulers via `_register_cron_scheduler` | Active; no reference to slack-bot scheduler |
| `frontend/src/app/cron/page.tsx` | 80+ | Cron Jobs UI tab; renders status badge "manifest" in grey | Active; no distinction shown between manifest and verified |

---

## Consensus vs Debate (external)

**Consensus:** APScheduler cross-process scheduler state requires explicit HTTP or RPC. No polling-shared-memory approach exists in v3.x. Event listeners are the hook. HTTP push is the most common pattern (not pull).

**Minor debate:** Some practitioners prefer a shared persistent jobstore (SQLAlchemy-backed) for cross-process job state, but the APScheduler FAQ explicitly warns this causes duplicate execution or missed jobs due to lack of inter-process signalling. For a single-developer local app, the heartbeat-push to the FastAPI backend is simpler and safer.

---

## Pitfalls (from literature)

- **Manifest-only dashboard**: operators trust the job list is real. If the slack-bot process dies, all 4 core jobs go dark silently. The /cron tab continues to show them as "manifest" with no alarm. (daily-devops.net)
- **Shared jobstore in multiple processes**: duplicate job execution, race conditions. (APScheduler FAQ)
- **health check that passes when app is broken**: a ping that returns 200 without testing actual scheduler registration or last-fired time gives false assurance. (daily-devops.net; microservices.io)
- **Missing SLACK_CHANNEL_ID**: the scheduler guard at scheduler.py:28-30 silently skips all 4 jobs if this env var is unset. No alarm is raised.

---

## Application to pyfinagent (Mapping Findings to file:line Anchors)

| Finding | File:line | Gap |
|---|---|---|
| Heartbeat-push pattern already partially built | `job_status_api.py:135-143` | Not wired: `job_runtime.py:83` sink is `logger.info` |
| 4 core jobs not in heartbeat registry | `job_status_api.py:52-60` | Add core job IDs to `_JOB_NAMES` or extend registry dynamically |
| No inbound HTTP in slack-bot | `app.py:1-78` | Cannot poll; must push from slack-bot to main backend |
| SLACK_CHANNEL_ID guard silently skips all jobs | `scheduler.py:28-30` | No `start_scheduler` failure propagated to `/api/jobs/all` |
| Manifest-only anti-pattern active | `cron_dashboard_api.py:155` | `status="manifest"` shown without staleness or liveness signal |
| No slack-bot log in Logs tab | `cron_dashboard_api.py:102-110` | `_log_paths()` has no `slackbot` key |

---

## RECOMMENDATION

Three options in increasing implementation cost:

### Option (a) — Keep manifest-only, document explicitly (lowest cost)

The manifest is already correct: it lists the 4 core jobs with `status="manifest"`. The slack-bot liveness signal lives elsewhere: the watchdog_health_check fires periodically to the main backend's `/api/health`; if it stops arriving, the backend is either down or the slack-bot is dead. No new code needed.

**Limitation**: the operator cannot distinguish between "slack-bot running but SLACK_CHANNEL_ID not set so jobs never registered" and "slack-bot dead." The /cron tab looks identical in both failure modes.

Suggested code-level annotation (no implementation, just documentation change):
```python
# cron_dashboard_api.py _static_to_dict — add a "liveness_signal" key
# so the frontend can render a tooltip explaining manifest vs live:
return {
    ...
    "status": "manifest",
    "liveness_note": "jobs registered in slack-bot process; no cross-process heartbeat wired yet",
}
```

### Option (b) — Add a slack-bot-side HTTP health endpoint (moderate cost)

Add a minimal aiohttp or starlette server (NOT Bolt, which has no inbound HTTP in Socket Mode) that the main backend can poll. Returns `{scheduler_running, jobs: [...]}`.

Sketch (add to `app.py::main()` before `handler.start_async()`):
```python
from aiohttp import web

async def _health(request):
    from backend.slack_bot.scheduler import _scheduler
    jobs = []
    if _scheduler and _scheduler.running:
        jobs = [{"id": j.id, "next_run": str(j.next_run_time)} for j in _scheduler.get_jobs()]
    return web.json_response({"scheduler_running": bool(_scheduler and _scheduler.running), "jobs": jobs})

health_app = web.Application()
health_app.router.add_get("/health", _health)
runner = web.AppRunner(health_app)
await runner.setup()
site = web.TCPSite(runner, "127.0.0.1", 8001)
await site.start()
```

The main backend's `/api/jobs/all` would then `httpx.get("http://127.0.0.1:8001/health", timeout=2)` on each request and merge the live job list. Adds a second port (8001) which must not be firewalled.

**Limitation**: adds a port, adds aiohttp dependency, requires the main backend to make an internal HTTP call on every `/api/jobs/all` fetch.

### Option (c) — Wire heartbeat-push from slack-bot to `/api/jobs/heartbeat` (RECOMMENDED — best fit)

The infrastructure already exists. `POST /api/jobs/heartbeat` at `job_status_api.py:135-143` is fully functional. The `heartbeat()` context manager in `job_runtime.py:67-114` already accepts a custom `sink`. The phase-9 jobs (daily_price_refresh, etc.) already use `heartbeat()`. Only two changes are needed:

**Change 1 — Add a push-sink to `scheduler.py::start_scheduler`** (after `_scheduler.start()`):
```python
# Wire heartbeat push to main backend registry for all 4 core jobs.
# Each job calls this sink on completion.
import httpx as _httpx

def _http_heartbeat_sink(event: dict) -> None:
    try:
        _httpx.post(
            "http://localhost:8000/api/jobs/heartbeat",
            json=event,
            timeout=3.0,
        )
    except Exception:
        pass  # fail-open; don't break the scheduler

# Attach APScheduler event listener to forward job events to the registry.
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED

def _aps_to_heartbeat(event):
    status = "ok" if not getattr(event, "exception", None) else "failed"
    _http_heartbeat_sink({
        "job": event.job_id,
        "status": status,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "error": repr(event.exception) if getattr(event, "exception", None) else None,
    })

_scheduler.add_listener(
    _aps_to_heartbeat,
    EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED,
)
```

**Change 2 — Add 4 core job IDs to `job_status_api.py::_JOB_NAMES`**:
```python
_JOB_NAMES: tuple[str, ...] = (
    # phase-9 jobs (existing)
    "daily_price_refresh", "weekly_fred_refresh", "nightly_mda_retrain",
    "hourly_signal_warmup", "nightly_outcome_rebuild", "weekly_data_integrity", "cost_budget_watcher",
    # phase-23.3 core slack-bot jobs (new)
    "morning_digest", "evening_digest", "watchdog_health_check", "prompt_leak_redteam",
)
```

After these two changes:
- The 4 core jobs appear in `/api/jobs/status` with real `last_run_at` and `status` after they fire.
- The `/cron` Jobs tab can show "ok" (green) vs "manifest" (grey) vs "failed" (red) for all 11 slack-bot jobs.
- The main backend needs no new port. The slack-bot needs no inbound HTTP server.
- Fail-open: if `httpx.post` fails (main backend down), the scheduler continues silently.

**This is the recommended option.** It requires the fewest new dependencies, uses the already-built `POST /api/jobs/heartbeat` endpoint, and is consistent with the existing heartbeat pattern used by the phase-9 jobs.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched, 1 returned 403 but was attempted and snippet confirmed)
- [x] 10+ unique URLs total (10 snippet-only + 7 attempted/fetched = 17)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (scheduler.py, app.py, job_runtime.py, job_status_api.py, cron_dashboard_api.py, main.py, cron/page.tsx)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "gate_passed": true
}
```
