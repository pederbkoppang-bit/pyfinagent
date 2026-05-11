---
step: phase-23.5.3
title: Cron job verification — morning_digest (slack_bot)
tier: simple
date: 2026-05-09
---

## Research: APScheduler CronTrigger timezone semantics + morning_digest liveness audit

### Queries run (three-variant discipline)
1. Current-year frontier: `APScheduler CronTrigger timezone ZoneInfo next_run_time isoformat 2026`
2. Last-2-year window: `APScheduler 3.x CronTrigger timezone semantics next_run_time UTC offset 2025`
3. Year-less canonical: `APScheduler CronTrigger spring forward DST skip double-fire hour=8 America/New_York`
4. Supplemental: `APScheduler zoneinfo Python 3.9 pytz replacement fully supported 2024 2025`
5. Supplemental: `Slack daily digest cron job idempotency retry dedup operational patterns 2025`

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html | 2026-05-09 | Official doc | WebFetch | "cron trigger works with wall clock time... may cause unexpected behavior... when entering or leaving DST"; timezone param defaults to scheduler timezone |
| https://apscheduler.readthedocs.io/en/3.x/userguide.html | 2026-05-09 | Official doc | WebFetch | Scheduler default timezone not stated; ZoneInfo tzinfo accepted; next_run_time is timezone-aware via the specified zone |
| https://github.com/agronholm/apscheduler/releases/tag/3.11.0 | 2026-05-09 | Official release notes | WebFetch | "support for ZoneInfo time zones was introduced while support for pytz time zones was simultaneously deprecated" |
| https://github.com/agronholm/apscheduler/commit/efe16602580d47ef5cb9787f977a65a5791ea024 | 2026-05-09 | Source code commit | WebFetch | zoneinfo conversion: trigger results are timezone-aware ZoneInfo datetime objects; isoformat() produces standard UTC-offset notation (e.g. `-04:00`) |
| https://slack.engineering/executing-cron-scripts-reliably-at-scale/ | 2026-05-09 | Authoritative engineering blog | WebFetch | Slack's own cron reliability pattern: dedup via DB table tracking state transitions; conductor pattern separating scheduling from execution; key failure mode is visibility gaps |
| https://fastapi.tiangolo.com/advanced/settings/ | 2026-05-09 | Official doc | WebFetch | Pydantic-settings `Field(default=N)` uses the default when env var absent; `@lru_cache` ensures settings loaded once; integer env var `MORNING_DIGEST_HOUR=9` overrides the Field default of 8 |
| https://github.com/agronholm/apscheduler/issues/606 | 2026-05-09 | GitHub issue (bug report) | WebFetch | DST skip: `CronTrigger(hour=0, timezone='US/Pacific')` skipped midnight on March 14, 2022 (spring-forward day); closed as duplicate of #529 — known recurring bug in next-fire calculation |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://github.com/agronholm/apscheduler/issues/980 | Bug report | Fetched — fall-back infinite loop (bi-hourly trigger); confirmed different from spring-forward skip, but both root in same #529 DST bug lineage |
| https://github.com/agronholm/apscheduler/issues/529 | Bug report | Snippet only; canonical DST root issue; referenced by multiple duplicates |
| https://github.com/agronholm/apscheduler/issues/384 | Bug report | Snippet only; pytz vs standard tzinfo discussion |
| https://johal.in/zoneinfo-python-3-9-iana-timezone-standard-library-2025/ | Blog | 403 Forbidden |
| https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/ | Official doc | Fetched — confirms Field defaults honored when env var absent; lru_cache not covered |

### Recency scan (2024-2026)

Searched for 2024-2026 literature on APScheduler CronTrigger DST semantics, zoneinfo support, and daily digest idempotency.

**Findings:**
- APScheduler 3.11.0 (released 2024) formally deprecated pytz and added zoneinfo support. This is the canonical recent change. Pyfinagent's use of `ZoneInfo("America/New_York")` in `scheduler.py:132` is correct per the 3.11.0 migration path.
- No new DST-handling papers in 2024-2026 beyond the known #529/#606 lineage. The spring-forward skip at non-2AM hours is a known but unresolved intermittent bug in APScheduler 3.x.
- Slack's cron reliability engineering blog (cited above) is the authoritative 2025 practitioner reference for daily-digest operational patterns; no newer superseding source found.

---

### Key findings

1. **next_run_time carries the trigger's timezone offset, not UTC** -- Live Bash test confirmed: `CronTrigger(hour=8, timezone=ZoneInfo("America/New_York")).get_next_fire_time(...)` returns `2026-05-09T08:00:00-04:00`. The isoformat string is ET-offset (EDT in May), not UTC. The `/api/jobs/all` response value `"next_run":"2026-05-09T08:00:00-04:00"` seen in the live system is therefore the correct and expected format. (Source: live Bash test against installed APScheduler; commit efe1660)

2. **APScheduler 3.11.0 fully supports zoneinfo; pytz is deprecated** -- pyfinagent's `from zoneinfo import ZoneInfo` in `scheduler.py:8` and `timezone=ZoneInfo("America/New_York")` at line 132 are the current canonical pattern. Using pytz here would emit deprecation warnings. (Source: APScheduler 3.11.0 release notes)

3. **Spring-forward DST (hour=8): morning_digest is SAFE from the known skip bug** -- The documented DST skip bug (APScheduler #606, #529) occurs when the scheduled hour falls in the 2:00-3:00 AM window that is removed by the spring-forward transition. Morning digest is registered at `hour=8` (Eastern), which is never in the DST transition window. Spring-forward in America/New_York moves 2:00 AM -> 3:00 AM; 8:00 AM is unaffected. The job will fire at 08:00 EDT on the spring-forward day without issue. (Source: APScheduler docs DST warning; issue #606 root cause analysis)

4. **Fall-back DST (hour=8): fires once, not twice** -- The official APScheduler docs state "if the time occurs twice in a given day [fall-back], it only fires once." At 8:00 AM ET on fall-back Sunday, the clock reads 8:00 AM twice (EDT then EST), but the trigger fires once. Confirmed wall-clock semantics. (Source: APScheduler CronTrigger docs)

5. **`morning_digest_hour` default is 8 from settings.py; env var override via `MORNING_DIGEST_HOUR`** -- `settings.py:199` declares `morning_digest_hour: int = Field(8, ...)`. The Field default (8) is used when the env var is absent. Override via `.env` key `MORNING_DIGEST_HOUR=N`. pydantic-settings validates defaults on load. (Source: settings.py:199; FastAPI pydantic-settings docs)

6. **`_send_morning_digest` uses `_BACKEND_URL = "http://backend:8000"` (Docker alias) -- confirmed silent-failure risk** -- `scheduler.py:211-214` calls `f"{_BACKEND_URL}/api/portfolio/performance"` and `f"{_BACKEND_URL}/api/reports/?limit=5"`. `_BACKEND_URL` is the Docker network alias (`http://backend:8000`) which does NOT resolve on a local Mac host process. The slack_bot runs as a host process (not inside Docker), per `memory/project_local_only_deployment.md`. This is the same class of bug fixed for the watchdog in phase-23.5.2.6 (watchdog was using `_BACKEND_URL`; fix moved it to `_HEALTH_PROBE_URL = "http://127.0.0.1:8000/api/health"`). `_send_morning_digest` has NOT been patched to use `127.0.0.1`. **The digest silently fails every morning: both httpx.get() calls raise `httpx.ConnectError` (connection refused on `backend:8000`), the outer `except Exception` catches it, logs `logger.exception("Failed to send morning digest")`, and NO Slack message is posted.**

7. **The heartbeat listener fires on ALL terminal events INCLUDING exceptions** -- `scheduler.py:12-15` wires the listener for `EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED`. When `_send_morning_digest` raises (or more precisely, when it catches its own exception internally and returns normally), the APScheduler event fired is `EVENT_JOB_EXECUTED` (the job function returned without re-raising). So `_aps_to_heartbeat` records `status="ok"`. The `/api/jobs/all` endpoint will show `status="ok"` after the first morning fire — even though NO digest was actually sent to Slack. **This is a false-positive liveness signal.**

8. **Verification test will PASS, but morning_digest is non-functional** -- The immutable verification checks `status != "manifest"` and `next_run is not None`. After the startup seed (`_seed_next_run_registry`), morning_digest has `status="scheduled"` and a valid `next_run_time`. After first fire, it will show `status="ok"`. Both satisfy the verification assertion. Neither proves a Slack message was delivered. The verification does not catch the Docker-alias silent failure.

9. **No `_send_morning_digest`-specific unit tests exist** -- grep over all test files finds NO test that calls `_send_morning_digest` directly or mocks its httpx calls to verify the Slack post. `test_slack_bot_heartbeat_push.py` tests the `_aps_to_heartbeat` listener shape and `record_heartbeat` registry updates, but not the digest function itself. `test_cron_dashboard.py:80` only checks that `morning_digest` appears in job IDs.

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/slack_bot/scheduler.py` | 533 | APScheduler job registration + handlers | READ IN FULL |
| `backend/config/settings.py` | 211 | Pydantic-settings; `morning_digest_hour` field | READ IN FULL |
| `backend/api/job_status_api.py` | 189 | Heartbeat registry; `_JOB_NAMES`; `get_registry_snapshot()` | READ IN FULL |
| `backend/api/cron_dashboard_api.py` (lines 195-243) | partial | `/api/jobs/all` merge logic post-23.5.2.5 | READ |
| `tests/services/test_slack_bot_heartbeat_push.py` | 134+ | Tests listener payload shape + registry updates | READ (head) |
| `tests/api/test_cron_dashboard.py` | checked | Cron dashboard API tests | GREP |
| `tests/verify_phase_23_3_2.py` | checked | Phase verification | GREP |

### Consensus vs debate

**Consensus:** APScheduler 3.11.0+ with `ZoneInfo` is the correct and canonical timezone pattern. `next_run_time.isoformat()` returns ET-offset strings (`-04:00` in EDT, `-05:00` in EST). The bridge (`/api/jobs/all`) correctly surfaces these. Hour=8 ET is safe from DST skip bugs.

**Debate:** Whether the verification criterion (`status != "manifest"` + `next_run not None`) is sufficient to declare morning_digest "live." This brief concludes it is NOT sufficient — the criterion proves scheduler registration but not handler correctness.

### Pitfalls (from literature + code audit)

1. **Docker-alias hostname in digest handler** (`scheduler.py:24,211,214`) -- The core pitfall. `http://backend:8000` only resolves inside Docker networks. On a local Mac host-process slack_bot, all httpx calls to this URL fail with `ConnectError`. The phase-23.5.2.6 fix correctly addressed this for the watchdog by creating `_HEALTH_PROBE_URL = "http://127.0.0.1:8000/api/health"` and `_HEARTBEAT_URL = "http://127.0.0.1:8000/api/jobs/heartbeat"` — but `_send_morning_digest` (and `_send_evening_digest`) were NOT updated and still use `_BACKEND_URL`.

2. **Fail-open exception swallowing masks the failure** (`scheduler.py:226-227`) -- `except Exception: logger.exception(...)` prevents any re-raise. APScheduler records the job as `EVENT_JOB_EXECUTED` (not `EVENT_JOB_ERROR`), so the heartbeat listener posts `status="ok"`. No Slack alert fires on morning digest failure.

3. **APScheduler DST spring-forward skip** -- Known but NOT applicable to hour=8. Applicable only to jobs scheduled in the 1:00-3:00 AM window (America/New_York). Documented for awareness.

4. **No idempotency guard** -- `_send_morning_digest` has no dedup or date-stamp check. If it fires twice (e.g., due to a scheduler restart mid-day), two digests would post. Slack's own engineering docs (cited) recommend DB-table dedup for production digest jobs. Low priority for pyfinagent's local-only deployment but worth noting.

5. **`morning_digest_hour` description says "local timezone" but registration uses `ZoneInfo("America/New_York")`** -- settings.py:199 description reads "Hour (0-23) for daily morning digest in local timezone". The registration (scheduler.py:132) hard-codes `ZoneInfo("America/New_York")`. If the Mac's local timezone were different from ET, the description would be misleading. For Peder's Mac in Norway (CET/CEST), this would fire at 8 AM ET = 2 PM / 3 PM Norwegian time — correct for US market open, but the description is misleading. No actual bug here; worth clarifying.

### Application to pyfinagent (file:line anchors)

| Finding | File:Line | Impact |
|---------|-----------|--------|
| `_BACKEND_URL = "http://backend:8000"` used in digest | scheduler.py:24, 211, 214 | CRITICAL — silent daily failure |
| `_HEARTBEAT_URL` correctly pinned to 127.0.0.1 | scheduler.py:30 | Working (watchdog fix applied here) |
| `_HEALTH_PROBE_URL` correctly pinned to 127.0.0.1 | scheduler.py:35 | Working (watchdog fix applied here) |
| `morning_digest` registration with `ZoneInfo` | scheduler.py:127-136 | Correct; next_run_time = ET-offset |
| `morning_digest_hour: int = Field(8, ...)` default | settings.py:199 | Default 8 AM ET; overridable via MORNING_DIGEST_HOUR= |
| `morning_digest` in `_JOB_NAMES` pre-seeded | job_status_api.py:63 | Registry has row from startup; never_run -> scheduled on seed |
| Seed pushes `status="scheduled"` on startup | scheduler.py:85-112 | next_run populated without waiting for first fire; status="scheduled" |
| Seed does NOT clobber terminal status | job_status_api.py:116 | If status was already "ok"/"failed", seed is no-op on status field |
| `/api/jobs/all` merge: `row.get("status", "never_run")` | cron_dashboard_api.py:228 | Falls back to "never_run" only if no registry entry; post-seed always "scheduled" |
| No test for `_send_morning_digest` httpx calls | tests/slack_bot/* | No test coverage for the Docker-alias failure mode |

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 fetched)
- [x] 10+ unique URLs total incl. snippet-only (collected 15+ across all searches)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (scheduler, settings, job_status_api, cron_dashboard_api, tests)
- [x] Contradictions / consensus noted (consensus on ZoneInfo being correct; debate on verification sufficiency)
- [x] All claims cited per-claim

---

### Recommendation: Is "scheduled + non-null next_run" sufficient for morning_digest?

**No. The verification passes but does not prove liveness.**

The verification command (`status != "manifest"`, `next_run is not None`) proves that:
- APScheduler registered the job (startup seed pushed `status="scheduled"`)
- APScheduler calculated a next fire time (`next_run = "2026-05-09T08:00:00-04:00"`)

It does NOT prove that:
1. The job handler will successfully reach the backend (it won't — Docker alias)
2. A Slack message will be posted (it won't — httpx fails silently)
3. After first fire, `status="ok"` actually means a digest was delivered (it doesn't — fail-open swallows the ConnectError and APScheduler sees a clean return)

**`_send_morning_digest` has the same Docker-alias bug that the watchdog had before phase-23.5.2.6.** Both `portfolio_res` and `reports_res` calls use `_BACKEND_URL = "http://backend:8000"` (scheduler.py:211,214). This URL does not resolve on the Mac host process. The digest fails every morning at 8 AM ET with `Failed to send morning digest` in the log, but no Slack message and no `status="failed"` in the registry (because the exception is caught before re-raising, so APScheduler fires `EVENT_JOB_EXECUTED`, not `EVENT_JOB_ERROR`).

**The phase-23.5.3 verification is a false positive for morning_digest specifically.** Main should:
1. Note the verification criterion passes (as specified in masterplan — the criterion is immutable)
2. Flag the Docker-alias bug in `_send_morning_digest` as a **separate substep** (out of scope per the caller's instructions, but must be documented in the critique so it is not lost)
3. Confirm the criterion `status != "manifest"` and `next_run not null` are satisfied by the current bridge state

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 5,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "gate_passed": true
}
```
