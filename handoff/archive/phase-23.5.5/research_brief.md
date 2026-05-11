# Research Brief: phase-23.5.5 — Cron Job Verification: watchdog_health_check (slack_bot)

**Date:** 2026-05-09
**Tier:** simple
**Step id:** phase-23.5.5

---

## Queries run (three-variant discipline)

1. Current-year frontier: `"APScheduler IntervalTrigger semantics watchdog jobs 2026"`
2. Last-2-year window: `"APScheduler watchdog alive meta-monitoring watchdog 2025 2026"`
3. Year-less canonical: `"APScheduler IntervalTrigger restart behavior first fire immediate or wait interval"`
4. Year-less canonical: `"health check interval 1 5 15 60 minutes SRE monitoring best practices"`
5. Year-less canonical: `"watchdog health check meta-monitoring dead man's switch scheduler 2025"`

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://apscheduler.readthedocs.io/en/3.x/modules/triggers/interval.html | 2026-05-09 | Official docs | WebFetch | "If no start_date is provided, the initial execution occurs one interval duration after the scheduler starts." |
| https://apscheduler.readthedocs.io/en/3.x/userguide.html | 2026-05-09 | Official docs | WebFetch | "if the scheduler is not yet running when the job is added, the job will be scheduled tentatively and its first run time will only be computed when the scheduler starts." misfire_grace_time controls post-restart catch-up. |
| https://jdhao.github.io/2024/11/02/python_apascheduler_start_job_immediately/ | 2026-05-09 | Blog (authoritative practitioner) | WebFetch | "By default, IntervalTrigger does NOT fire immediately. The job waits for the specified interval before its first execution." next_run_time=datetime.now() is the explicit override. |
| https://blog.ediri.io/how-to-set-up-a-dead-mans-switch-in-prometheus | 2026-05-09 | Practitioner blog | WebFetch | "A watchdog alert (dead man's switch) helps you continuously test your entire alerting pipeline from beginning to end." External service monitors incoming pings; silence triggers alert. |
| https://oneuptime.com/blog/post/2026-02-06-heartbeat-dead-man-switch-opentelemetry-pipeline/view | 2026-05-09 | Practitioner blog (2026) | WebFetch | "If the watchdog stops receiving it, the alerting system itself is broken." Recommended heartbeat interval: 30s generation, 1-min repeat, 2-5 min detection window. |
| https://sre.google/sre-book/monitoring-distributed-systems/ | 2026-05-09 | Authoritative (Google SRE book) | WebFetch | "Probing for a 200 (success) status more than once or twice a minute is probably unnecessarily frequent." Emphasizes symptom-based state-transition alerting over continuous polling. |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://microservices.io/patterns/observability/health-check-api.html | Pattern doc | Fetched but contained no interval guidance |
| https://carlmastrangelo.com/blog/health-checking-best-practices | Blog | Fetched; covers keep-alives not interval tradeoffs |
| https://betterstack.com/community/guides/scaling-python/apscheduler-scheduled-tasks/ | Tutorial | Fetched; does not address misfire/restart semantics |
| https://cronradar.com/blog/python-scheduler-monitoring | Blog | 403 blocked |
| https://pypi.org/project/APScheduler/ | PyPI | Snippet; no restart detail |
| https://github.com/agronholm/apscheduler/issues/1095 | GitHub issue | Snippet: "missed jobs run at next fire time instead of immediately" — confirms wait-not-immediate default |
| https://apscheduler.readthedocs.io/en/3.x/faq.html | Official docs | Fetched; FAQ does not cover restart/first-fire |
| https://training.promlabs.com/training/monitoring-and-debugging-prometheus/metrics-based-meta-monitoring/end-to-end-watchdog-alerts/ | Training | Snippet; confirms Prometheus Watchdog pattern |

---

## Recency scan (2024-2026)

Searched explicitly for 2024-2026 literature on APScheduler IntervalTrigger restart behavior and watchdog meta-monitoring patterns.

**Findings:**
- jdhao.github.io (November 2024): confirms IntervalTrigger default waits one full interval before first fire; next_run_time override is the canonical workaround.
- oneuptime.com (February 2026): heartbeat + dead man's switch pattern for OTel pipelines; recommends 30s heartbeat, 1-min repeat, 2-5 min detection window — no material change from pre-2024 SRE doctrine.
- metafunctor.com (February 2026): federated dead man's switch implementation at v0.5 — not relevant to APScheduler.

No findings supersede the canonical APScheduler docs or the pyfinagent 23.5.2.6 design. The 2024-2026 window confirms the interval-wait default and the state-transition-only alerting pattern.

---

## Key findings

1. **IntervalTrigger first-fire waits one full interval.** "If no start_date is provided, the initial execution occurs one interval duration after the scheduler starts." (APScheduler official docs, 2026-05-09). In pyfinagent's case: daemon starts at 10:20:21 CEST → first watchdog fire at 10:35:21 CEST (exactly 15 min later). Confirmed by log.

2. **No retroactive catch-up on restart.** "If the start date is in the past, the trigger will not fire many times retroactively but instead calculates the next run time from the current time." (APScheduler official docs). The watchdog does not storm Slack after a daemon restart.

3. **Misfire grace time applies to interval jobs.** If the scheduler is down during a scheduled tick and restarts within `misfire_grace_time`, the missed tick fires once (if coalesce=True). The watchdog job has NO explicit misfire_grace_time set (only phase-9 jobs have it), so it uses the APScheduler default (typically 1 second). In practice this means a very brief outage could trigger one extra fire, but the state machine handles it correctly: `None -> True` (clean baseline on restart) suppresses the post.

4. **State-transition-only alerting is the established SRE pattern.** Google SRE book: "rules that catch real incidents most often should be as simple, predictable, and reliable as possible." Posting only on transitions (not steady-state) matches this doctrine. (Source: sre.google, 2026-05-09)

5. **15-minute interval is appropriate for a single-backend local deployment.** Google SRE: "probing for a 200 (success) status more than once or twice a minute is probably unnecessarily frequent." For a single Mac host process (not a distributed fleet), 15 min is conservative but well within SRE norms. A 1-min interval would be over-monitoring for this deployment model. (Source: sre.google, 2026-05-09)

6. **Meta-monitoring (is the watchdog alive?).** The canonical pattern (Prometheus Watchdog, healthchecks.io) is an external service that expects a periodic ping from the watchdog and alerts if the ping goes missing. pyfinagent's heartbeat push (`_aps_to_heartbeat` -> `/api/jobs/heartbeat`) is the internal equivalent: the main backend tracks `last_run` and `next_run` for the watchdog job, enabling the `/api/jobs/all` endpoint to surface liveness. This is a valid implementation of meta-monitoring within a single-host system. (Source: blog.ediri.io, 2026-05-09)

---

## Internal code inventory

| File | Lines inspected | Role | Status |
|------|----------------|------|--------|
| `backend/slack_bot/scheduler.py` | 1-544 | APScheduler setup, watchdog function, heartbeat push | Current; post-23.5.2.6 |
| `tests/slack_bot/test_watchdog_alert_semantics.py` | 1-154 | State-machine unit tests (6 tests) | Current; post-23.5.2.6 |
| `backend/api/job_status_api.py` | L55-86 | `_JOB_NAMES` pre-seed list | Current |
| `backend/config/settings.py` | L201 | `watchdog_interval_minutes: int = Field(15, ...)` | Current |
| `handoff/logs/slack_bot.log` | Last 200 lines | Live log evidence | Observed 2026-05-09 |

---

## `_watchdog_health_check` function body (verbatim, post-23.5.2.6)

From `backend/slack_bot/scheduler.py:266-332`:

```python
async def _watchdog_health_check(app: AsyncApp):
    """Probe backend health endpoint; post to Slack only on state transitions.

    phase-23.5.2.6: state-transition gating. Posts to Slack only on:
        None -> False  (first probe failed; alert)
        True -> False  (down: alert)
        False -> True  (recovery: alert)
    Steady-state (None->True, True->True, False->False) logs only.
    """
    global _watchdog_last_was_healthy

    settings = get_settings()
    is_healthy = False
    detail = ""

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(_HEALTH_PROBE_URL)
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                is_healthy = True
                detail = f"HTTP {resp.status_code}"
            else:
                detail = f"HTTP {resp.status_code}, body did not have status=ok"
    except Exception as exc:
        detail = f"unreachable: {type(exc).__name__}"

    prior = _watchdog_last_was_healthy
    _watchdog_last_was_healthy = is_healthy

    # Decide whether to post.
    post: tuple[str, str] | None = None  # (emoji+text, fallback_text)
    if is_healthy:
        if prior is False:
            post = (...)  # recovery message
            logger.info("Watchdog recovery -- %s", detail)
        else:
            # None->True (clean baseline) or True->True (steady) -- silent.
            logger.debug("Watchdog steady-healthy -- %s", detail)
    else:
        if prior is None or prior is True:
            post = (...)  # alert message
            logger.warning("Watchdog unhealthy transition -- %s", detail)
        else:
            # False->False (steady-down) -- log only; do NOT spam.
            logger.warning("Watchdog steady-unhealthy -- %s", detail)

    if post is not None:
        try:
            await app.client.chat_postMessage(...)
        except Exception:
            logger.exception("Watchdog Slack post failed")
```

Key wiring confirmed:
- `_HEALTH_PROBE_URL = "http://127.0.0.1:8000/api/health"` (line 41) — localhost, not Docker alias
- `_watchdog_last_was_healthy: bool | None = None` (line 52) — reset to None on daemon restart
- Probe timeout: 10s (line 282)
- `settings.watchdog_interval_minutes` default: 15 (settings.py:201)

---

## Registration confirmed

`backend/slack_bot/scheduler.py:161-169`:
```python
_scheduler.add_job(
    _watchdog_health_check,
    "interval",
    minutes=settings.watchdog_interval_minutes,
    args=[app],
    id="watchdog_health_check",
    replace_existing=True,
)
```

No `misfire_grace_time` or `coalesce` on the watchdog job (unlike phase-9 jobs which have them at line 514-528). This is intentional: for a 15-min interval on a health probe, missing one tick is not a correctness concern, and the state machine's `None -> True` suppression handles the post-restart baseline cleanly.

---

## Bridge pre-seed confirmed

`backend/api/job_status_api.py:65`:
```
"watchdog_health_check",     # phase-23.3.2
```
Pre-seeded in `_JOB_NAMES`, so `_registry` is initialized with `{"name": "watchdog_health_check"}` at import time, before any heartbeat fires. Status advances from `"manifest"` to `"scheduled"` via `_seed_next_run_registry()` at daemon start, then to `"ok"` on first fire.

---

## Test coverage confirmed

`tests/slack_bot/test_watchdog_alert_semantics.py` — 6 tests covering:
1. `test_steady_healthy_after_clean_start_no_post` — None->True->True->True: 0 posts
2. `test_first_failure_after_clean_start_posts_alert` — None->False: 1 post (alert)
3. `test_consecutive_failures_no_repost` — None->False->False->False: 1 post (the spam-fix regression)
4. `test_recovery_after_failure_posts_recovery` — None->False->True: 2 posts
5. `test_steady_healthy_after_recovery_no_more_posts` — None->False->True->True->True: 2 posts
6. `test_uses_localhost_probe_url_not_docker_alias` — regression guard for Docker-alias bug

All 6 tests cover the state-machine transitions. The Docker-alias regression guard (test 6) directly addresses the phase-23.5.2.6 bug being verified here.

---

## Live in-the-wild evidence (post-daemon-restart)

**Daemon restart time (from log):** 2026-05-09 10:20:21 CEST

**First watchdog fire:** 2026-05-09 10:35:21 CEST (exactly 15 min after start — confirms IntervalTrigger default behavior: waits one full interval)

**Total fires logged today:** 49

**Last fire logged:** 2026-05-09 22:35:21 CEST

**Live API state (verified at research time):**
```
OK watchdog_health_check ok 2026-05-09T22:50:21.067885+02:00
```
- `status = "ok"` (not "manifest")
- `next_run = "2026-05-09T22:50:21.067885+02:00"` (populated, advancing every 15 min)
- Immutable criterion PASSES: `status != "manifest"` AND `next_run is not None`

**Slack posts since restart:** ZERO. The log contains no `Watchdog steady-healthy` (DEBUG, not visible at INFO), no `Watchdog unhealthy transition`, no `Watchdog recovery`, no `Watchdog steady-unhealthy`. All 49 fires completed as `"Job executed successfully"` (APScheduler INFO). The state machine correctly classified all fires as `None->True` then `True->True` (steady-healthy, no post).

**This confirms:** backend has been continuously healthy since 10:35:21 CEST. Zero spurious Slack alerts — the phase-23.5.2.6 spam fix is live and working.

---

## Consensus vs debate

**Consensus:** IntervalTrigger default waits one full interval before first fire. State-transition-only alerting is standard SRE practice. 15-min interval is appropriate for a single-host local deployment. The heartbeat push to `/api/jobs/heartbeat` serves as in-process meta-monitoring.

**No debate:** The state machine design and `_HEALTH_PROBE_URL` localhost fix are both well-established patterns with no competing approaches that would apply here.

---

## Pitfalls (from literature)

1. **APScheduler has no built-in monitoring.** (cronradar.com, snippet): if the scheduler thread dies silently, jobs stop firing with no alert. The `_aps_to_heartbeat` listener partially mitigates this — it won't fire if the scheduler thread is dead.
2. **Misfire on restart without grace time.** If the slack-bot daemon is down for >15 min and the watchdog job has no `misfire_grace_time`, APScheduler drops the missed ticks (default misfire_grace_time is ~1 second). For a health-check watchdog this is correct behavior.
3. **next_run_time=None on job store init.** Without `_seed_next_run_registry()`, `/api/jobs/all` would show `next_run=null` until the first fire. The seed at line 213 of scheduler.py patches this correctly.
4. **Docker alias DNS failure.** Confirmed pre-fix bug (23.5.2.6): `http://backend:8000` does not resolve on Mac host process. Fixed to `http://127.0.0.1:8000/api/health`.

---

## Application to pyfinagent

The immutable verification criterion `status != "manifest" AND next_run is not None` is a TRUE liveness signal for `watchdog_health_check`:

- `status = "ok"` is set only after at least one successful heartbeat POST from `_aps_to_heartbeat`, which fires only after the job executes. A stuck scheduler cannot produce "ok".
- `next_run` populated means the APScheduler computed and published the next fire time, which requires the scheduler to be running with the job registered.
- Both fields advancing on 15-min cadence rules out stale data from the seed-only state.

The criterion is sufficient. The watchdog is verified live in the wild with 49 consecutive healthy fires and zero Slack posts.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total (incl. snippet-only) (14 URLs)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (scheduler.py, test file, job_status_api.py, settings.py, live log)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 8,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "gate_passed": true
}
```
