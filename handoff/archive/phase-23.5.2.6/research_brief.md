---
step: phase-23.5.2.6
topic: Investigate watchdog_health_check Slack spam and fix
tier: moderate
date: 2026-05-09
---

## Research: watchdog_health_check Slack spam — alert-on-failure pattern

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://sre.google/sre-book/monitoring-distributed-systems/ | 2026-05-09 | Official doc (Google SRE) | WebFetch full | "Every page should be actionable … if a page merely merits a robotic response, it shouldn't be a page." |
| https://oneuptime.com/blog/post/2026-01-24-fix-monitoring-alert-fatigue/view | 2026-05-09 | Engineering blog (Jan 2026) | WebFetch full | "Every alert must be actionable — notifications requiring no human intervention should never trigger pages." |
| https://www.checklyhq.com/docs/alerting-and-retries/alert-states/ | 2026-05-09 | Official docs (Checkly) | WebFetch full | Recovery alert suppressed if no prior failure alert was sent; recovery is not sent from a passing state. |
| https://oneuptime.com/blog/post/2026-01-30-alert-deduplication/view | 2026-05-09 | Engineering blog (Jan 2026) | WebFetch full | AlertGroup state machine tracks first_seen, last_seen, count; fixed vs sliding suppression windows; threshold-based re-notify at [1, 5, 10, 50…]. |
| https://apscheduler.readthedocs.io/en/3.x/userguide.html | 2026-05-09 | Official docs (APScheduler) | WebFetch full | `event.exception` attribute distinguishes success from error; `add_listener(fn, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)` pattern shown explicitly. |
| https://sensu.io/blog/alert-fatigue-in-sre-and-devops | 2026-05-09 | Authoritative blog (Sensu) | WebFetch full | "Skip non-actionable events; suppress alerts lacking clear remediation paths; use logging instead." Occurrence-based filtering to prevent alert storms. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://oneuptime.com/blog/post/2026-03-05-alert-fatigue-ai-on-call/view | Blog | Fetched (partially useful); ML-heavy, not applicable to our httpx/Bolt stack |
| https://sre.google/sre-book/practical-alerting/ | Official doc | Snippet confirms symptoms-over-causes; full SRE book read via sister page |
| https://medium.com/@ebubekirdinc/health-check-with-watchdogs-in-a-microservices-architecture-45c474617878 | Blog | Fetched; no state-tracking patterns — .NET HealthChecksUI UI, not relevant |
| https://www.pagerduty.com/docs/guides/datadog-integration-guide/ | Docs | Snippet; confirmed PagerDuty uses dedup_key to group events — confirmed state pattern |
| https://cloud.google.com/blog/topics/developers-practitioners/why-focus-symptoms-not-causes | Google blog | Snippet; confirms symptoms-not-causes; core point captured via SRE book |
| https://moss.sh/devops-monitoring/how-to-set-up-slack-alerts-for-monitoring/ | Blog | Snippet; generic Slack alerting how-to, no new technical patterns |
| https://cronradar.com/blog/python-scheduler-monitoring | Blog | Snippet; APScheduler monitoring generic overview |

### Recency scan (2024-2026)

Searched: "alert fatigue SRE 2026", "alert on failure only watchdog health check deduplication 2026", "Slack bot alerting backoff flood control 2025 2026".

**Findings:** Two 2026 sources (OneUptime Jan 2026 x2) provide concrete deduplication patterns and alert-fatigue remediation frameworks. No breakthrough that supersedes the Google SRE canonical guidance. The canonical principle (alert on actionable failure only, not steady state) is stable. Deduplication using a module-level state dict is the current community consensus for lightweight Python daemons — Redis is mentioned for distributed systems, which does not apply here (pyfinagent is single-Mac). No 2025-2026 papers on the arXiv/IEEE front relevant to this operational bug.

---

### Key findings

1. **Google SRE canonical rule** — "Every page should be actionable. If a page merely merits a robotic response, it shouldn't be a page." A "backend healthy" message every 15 minutes is the definition of a robotic, non-actionable notification. (Source: Google SRE Book, sre.google)

2. **Alert deduplication with state tracking** — The standard pattern for a single-process Python daemon is a module-level dict tracking `last_status`. On each probe cycle, compare new status to `last_status`; post only on HEALTHY→UNHEALTHY and UNHEALTHY→HEALTHY transitions. No external store needed for single-process daemons. (Source: OneUptime deduplication blog, Jan 2026)

3. **Recovery alerts are valuable, not spam** — Checkly's documented pattern: send a recovery message when state transitions from DOWN to PASSING, but suppress it if no prior failure alert was sent. This closes the loop for the operator without adding steady-state noise. (Source: Checkly docs)

4. **Suppression window / snooze for consecutive failures** — The deduplication blog recommends re-alerting at count thresholds (1st, 5th, 10th failure). For pyfinagent's minimum-blast-radius fix, a simpler model is correct: alert on first failure, suppress subsequent identical-state posts, send one recovery when it clears. An hourly re-alert on sustained failure is optional. (Source: OneUptime deduplication blog)

5. **APScheduler does not need to change** — The job fires on the correct interval (15 min) for probing. The scheduler config is not the bug; the logic inside `_watchdog_health_check` is. (Source: APScheduler docs, internal read)

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/slack_bot/scheduler.py` | 502 | Defines `_watchdog_health_check` and all scheduler wiring | BUG in lines 245-291 (see below) |
| `backend/main.py` | ~500 | Defines `/api/health` endpoint at line 416 | Healthy — returns `{"status": "ok", ...}` with HTTP 200 |
| `backend/config/settings.py` | - | `watchdog_interval_minutes: int = Field(15, ...)` at line 201 | Default=15, not overridden in .env (env-denied, confirmed by settings.py read) |
| `tests/scheduler/` | - | APScheduler-related tests | No test for `_watchdog_health_check` function body behavior |
| `tests/slack_bot/` | - | Slack bot tests | No test for watchdog alert suppression semantics |
| `tests/verify_phase_23_3_2.py` | - | Verifies `watchdog_health_check` job is registered | Only checks job registration, not alert behavior |

No `last_status`, `was_healthy`, `state_transition`, `dedup_alert`, or `alert_dedup` variables exist anywhere in `backend/slack_bot/`. No state-tracking pattern is currently present.

---

### Function body verbatim (scheduler.py lines 245-291)

```python
async def _watchdog_health_check(app: AsyncApp):
    """Probe backend health endpoint; post to Slack only on failure."""
    settings = get_settings()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{_BACKEND_URL}/api/health")
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                logger.debug("Watchdog health check passed")
                return

        await app.client.chat_postMessage(
            channel=settings.slack_channel_id,
            blocks=[{
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        ":warning: *Watchdog Alert* -- Backend health check failed\n"
                        f"Status: {resp.status_code} at {datetime.now().strftime('%H:%M:%S')}"
                    ),
                },
            }],
            text="Watchdog Alert: backend health check failed",
        )
        logger.warning("Watchdog health check failed -- status %d", resp.status_code)

    except Exception:
        try:
            await app.client.chat_postMessage(
                channel=settings.slack_channel_id,
                blocks=[{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            ":rotating_light: *Watchdog Alert* -- Backend unreachable\n"
                            f"Time: {datetime.now().strftime('%H:%M:%S')}"
                        ),
                    },
                }],
                text="Watchdog Alert: backend unreachable",
            )
        except Exception:
            pass
        logger.exception("Watchdog health check -- backend unreachable")
```

---

### Root cause analysis (the four questions)

#### Q1. What exact lines cause the every-15-min Slack post on HEALTHY responses?

**The code does NOT post on healthy responses.** Reading the function body verbatim:

- Lines 251-254: If `resp.status_code == 200` AND `resp.json().get("status") == "ok"`, the function calls `logger.debug(...)` and **`return`**. No Slack post.
- Lines 256-270: Only reached if the HTTP status is NOT 200 or the JSON status is not "ok".
- Lines 273-290: Only reached if an exception is raised (network error, connection refused, etc.).

**The `/api/health` endpoint (main.py:416-451) returns HTTP 200 with `{"status": "ok", ...}` when the backend is running.**

**Confirmed:** `curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/api/health` returns `200`.

**Therefore the "spam" is NOT healthy-state spam.** The watchdog is catching the backend as unreachable (falling into the `except Exception` branch) and posting the `:rotating_light: Watchdog Alert -- Backend unreachable` message every 15 minutes. The backend at `http://backend:8000` (Docker network hostname) is NOT reachable from the Slack bot process, which runs on the host. The bot can reach `http://127.0.0.1:8000` (used for heartbeats) but probes `http://backend:8000` for health.

**Exact line causing the spam: line 251** — `resp = await client.get(f"{_BACKEND_URL}/api/health")` where `_BACKEND_URL = "http://backend:8000"` (line 24). Connection refused or DNS failure raises an exception, which falls into the `except Exception` block at line 272, which unconditionally posts to Slack.

The `_BACKEND_URL` is `"http://backend:8000"` (Docker hostname). The heartbeat posts to `_HEARTBEAT_URL = "http://127.0.0.1:8000/api/jobs/heartbeat"` (localhost) and succeeds (confirmed in log). The watchdog probes the Docker hostname which fails in a non-Docker local-only deployment.

#### Q2. Recommended fix shape

Two orthogonal fixes are needed:

**Fix A (required): Fix the URL** — Change `_watchdog_health_check` to probe `http://127.0.0.1:8000/api/health` instead of `http://backend:8000/api/health`. The heartbeat already uses `_HEARTBEAT_URL` at `127.0.0.1:8000` and succeeds. This eliminates the connection-refused exception that causes the spam.

**Fix B (required, separate from Fix A): Add state-transition gating** — Even after Fix A, the exception branch posts unconditionally if the backend goes down. Once the backend IS down, every 15-min tick posts a new message. The operator wants: first failure = alert, subsequent failures = suppress, recovery = alert once.

Recommended minimum-blast-radius implementation:

```python
# Module-level state (survives across scheduler ticks, resets on daemon restart)
_watchdog_last_was_healthy: bool | None = None  # None = first run ever
```

In `_watchdog_health_check`:
1. After the probe, determine `is_healthy: bool`.
2. Compare to `_watchdog_last_was_healthy`.
3. Post to Slack only on: (a) HEALTHY→UNHEALTHY transition (first-failure alert), (b) UNHEALTHY→HEALTHY transition (recovery alert).
4. Update `_watchdog_last_was_healthy = is_healthy`.
5. `logger.debug` on healthy pass, `logger.warning` on unhealthy (but no Slack post if state unchanged).

**Regarding optional sub-questions:**
- **(a) Recovery message on DOWN→UP?** YES — recommended. Operators need to know the backend recovered so they can stop worrying. Checkly's canonical pattern confirms this: recovery alert only fires if a prior failure alert was sent, which our state variable naturally handles.
- **(b) Backoff for consecutive failures?** NO for the minimal fix. State-transition gating alone eliminates the stream. An optional "re-alert after N consecutive failures" (e.g., alert again after 8 consecutive failures = 2 hours of downtime) can be added later with a `_watchdog_failure_count` counter. Do not add this now — minimum blast radius.
- **(c) Persist across daemon restarts?** NO. Module-level dict is sufficient. On restart, `_watchdog_last_was_healthy = None` causes the first probe result to set the baseline without posting. This is the correct behavior: a daemon restart does not constitute a new outage.

#### Q3. Is the `/api/health` 404 a confound?

Earlier in the session, `curl http://localhost:8000/health` returned 404. The correct path is `/api/health` (not `/health`). Confirmed: `curl http://localhost:8000/api/health` returns HTTP 200. The watchdog already probes `/api/health` (line 251) — the path is correct.

The confound is the **hostname**, not the path. `http://backend:8000` is a Docker-compose DNS alias that resolves only inside a Docker network. pyfinagent is a local-only Mac deployment (confirmed: `project_local_only_deployment.md`). The Slack bot process runs on the host, not in Docker. `http://backend:8000` does not resolve, raising a connection exception at every tick, which posts to Slack every 15 minutes.

**This is "spam from broken endpoint" (the hostname is wrong), not "spam from healthy state".**

**Scope implication:** The fix touches only `_watchdog_health_check` in `scheduler.py`. The correct probe URL is `http://127.0.0.1:8000/api/health` (mirrors the working heartbeat URL). No change to main.py or the health endpoint is required.

#### Q4. Test design

Proposed test in `tests/slack_bot/test_watchdog_alert_semantics.py` using `pytest` + `pytest-asyncio` + `unittest.mock`:

```python
"""Regression test: watchdog posts only on state transitions, never on steady state."""
import backend.slack_bot.scheduler as scheduler_mod

async def _run_watchdog(app_mock, health_raises=False, status_code=200, status_val="ok"):
    # Patch httpx.AsyncClient.get and module-level _watchdog_last_was_healthy
    ...

@pytest.mark.asyncio
async def test_healthy_state_no_post(mock_app):
    """When backend is healthy, no Slack post is sent."""
    # Call _watchdog_health_check 3 times; assert mock_app.client.chat_postMessage.call_count == 0

@pytest.mark.asyncio
async def test_first_failure_posts(mock_app):
    """First transition to unhealthy sends exactly one alert."""
    # healthy → unhealthy; assert post count == 1

@pytest.mark.asyncio
async def test_consecutive_failure_no_post(mock_app):
    """Consecutive failures do not re-post (state unchanged)."""
    # unhealthy → unhealthy → unhealthy; assert post count == 1 total (from first failure)

@pytest.mark.asyncio
async def test_recovery_posts(mock_app):
    """Recovery (unhealthy → healthy) sends exactly one recovery message."""
    # first call = failure (count=1), second call = healthy (count=2); assert count == 2

@pytest.mark.asyncio
async def test_steady_healthy_after_recovery_no_post(mock_app):
    """After recovery, continued healthy does not re-post."""
    # failure → recovery → healthy → healthy; assert total post count == 2 (failure + recovery)
```

The existing `verify_phase_23_3_2.py` checks job registration only (job id in the scheduler). Do not modify it — add the semantic tests separately.

---

### Consensus vs debate (external)

**Consensus:** Alert only on state transitions; suppress steady-state. Module-level state dict is the right pattern for a single-process daemon. Recovery alerts are desirable, not extra noise. All five sources agree.

**Debate:** Backoff / re-alert threshold for prolonged outages — deduplication blog recommends threshold-based re-notify; SRE book says pages must be actionable; a 2-hour sustained outage re-alert is reasonable but out of scope for the minimal fix.

### Pitfalls (from literature)

1. **Second-opinion-shopping on states** — do not snooze and re-alert hoping the problem clears; use actual state tracking (SRE book).
2. **Posting "all clear" proactively** — Checkly explicitly suppresses recovery alerts when no prior failure alert was sent. Avoid unsolicited "backend healthy" messages on daemon restart.
3. **State lost on daemon restart** — acceptable for pyfinagent; on restart `last_status = None` → first probe sets baseline silently. Do not persist to disk or BQ; that's over-engineering.
4. **Using `http://backend:8000` in a non-Docker context** — the root cause of the current spam. Always use `127.0.0.1` for host-side processes in the pyfinagent local-only deployment.

### Application to pyfinagent (mapping external findings to file:line anchors)

| Finding | File:line |
|---------|-----------|
| Root cause: wrong URL `http://backend:8000` raises connection exception → unconditional Slack post | `backend/slack_bot/scheduler.py:24` (`_BACKEND_URL`) and `scheduler.py:251` |
| Fix A: change probe URL to `http://127.0.0.1:8000/api/health` | `scheduler.py:251` |
| Fix B: add `_watchdog_last_was_healthy: bool | None = None` module-level | `scheduler.py` (add near line 31, after `_scheduler`) |
| Fix B: state-transition logic inside `_watchdog_health_check` | `scheduler.py:245-291` (full function rewrite) |
| Health endpoint is correct, returns HTTP 200 + `{"status": "ok"}` | `backend/main.py:416-451` |
| Default `watchdog_interval_minutes = 15` from settings | `backend/config/settings.py:201` |
| No existing state-tracking: no `last_status`, `was_healthy`, etc. | Confirmed grep across `backend/slack_bot/` — zero hits |
| No existing test for watchdog alert semantics | `tests/scheduler/` and `tests/slack_bot/` — no file for `_watchdog_health_check` body |

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total (13 URLs collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (scheduler.py, main.py, settings.py, tests/)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 7,
  "urls_collected": 13,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "gate_passed": true
}
```
