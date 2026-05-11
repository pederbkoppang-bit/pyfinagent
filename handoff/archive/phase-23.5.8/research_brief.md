## Research: phase-23.5.8 — Cron job verification: weekly_fred_refresh

Tier assumed: **simple** (as stated by caller).

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html | 2026-05-09 | Official docs | WebFetch | "The cron trigger works with the 'wall clock' time. Thus, if the selected time zone observes DST, you should be aware that it may cause unexpected behavior." |
| https://apscheduler.readthedocs.io/en/3.x/userguide.html | 2026-05-09 | Official docs | WebFetch | "If the execution of a job is delayed due to no threads... you may want to... adjust the misfire_grace_time setting to a higher value." |
| https://pypi.org/project/fredapi/ | 2026-05-09 | Official docs (PyPI) | WebFetch | "fredapi 0.5.2 (May 5 2024). Key methods: get_series(), get_series_latest_release(), search(). Returns pandas Series or DataFrame." |
| https://github.com/mortada/fredapi | 2026-05-09 | Code / authoritative | WebFetch | "fredapi wraps FRED web service; initialize with api_key; call get_series('DGS10') for macro series retrieval." |
| https://medium.com/@nsunder724/access-macroeconomic-data-at-scale-with-fedfred-a-modern-python-client-for-the-fred-api-96745541ef2a | 2026-05-09 | Authoritative blog | WebFetch | "fedfred uses full async I/O, built-in caching, rate limiting, and retry logic; older packages (fredapi) lack async support." |
| https://sugimiyanto.medium.com/idempotency-the-thing-we-should-implement-to-have-a-reliable-data-pipeline-in-batch-processing-b13664be630d | 2026-05-09 | Practitioner blog | WebFetch | "Idempotency: no matter how many times a job or task is executed, it always results in the exact same result. BigQuery partitioned tables enable isolation by partition." |

---

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://fred.stlouisfed.org/docs/api/fred/ | Official docs | 403 returned by WebFetch; known from prior research sessions |
| https://github.com/gw-moore/pyfredapi | Code | Redundant with fredapi/fedfred coverage; snippet sufficient |
| https://datons.com/en/blog/download-and-analyze-fred-data-automatically-with-python | Blog | Fetched but only confirms basic API key + series fetch pattern; no new facts |
| https://www.oreilly.com/library/view/data-engineering-design/9781098165826/ch04.html | Book | Paywalled; snippet covers idempotency chapter title |
| https://medium.com/@yorrr78/ensuring-idempotency-in-bigquery-pipelines-with-apache-airflow-fedf4ab43541 | Blog | Snippet confirms partition overwrite pattern; fetching datons.com covered same topic in full |
| https://coderslegacy.com/python/apscheduler-cron-trigger/ | Tutorial | Snippet confirms day_of_week="sun" valid; cron.html docs are authoritative |
| https://snyk.io/advisor/python/APScheduler/functions/apscheduler.triggers.cron.CronTrigger | Advisory | Snippet confirms usage; docs already fetched in full |
| https://www.kubeblogs.com/how-to-schedule-simple-tasks-using-apscheduler-a-devops-focused-guide/ | Blog | Snippet: "Cron Jobs Failing? Switch to APScheduler" - confirms misfire_grace_time best practice |
| https://dev.to/alexmercedcoder/idempotent-pipelines-build-once-run-safely-forever-2o2o | Blog | Snippet covers stable keys and upserts; covered by sugimiyanto source in full |
| https://thearchitectsnotebook.substack.com/p/advanced-idempotency-in-system-design | Blog | Snippet on idempotency keys in system design; no new weekly-specific content |

---

### Recency scan (2024-2026)

Searched: "FRED API Python macro series fetch 2026", "APScheduler weekly cron DST handling interval trigger 2025", "weekly batch job idempotency design patterns BigQuery 2025", "APScheduler cron trigger day_of_week sun misfire_grace_time best practice 2024 2025".

Result: fredapi's latest release is 0.5.2 (May 2024). A newer library, fedfred, emerged as a modern async alternative (2024-2025 publication). APScheduler 3.x cron DST behavior is unchanged through 2025 (version 3.11.2.post1 is current). No new findings materially change the canonical approach: use UTC or non-DST timezone for weekly cron jobs; iso-week keying is the established idempotency pattern for weekly batch jobs. No superseding papers or breaking changes found in the 2024-2026 window.

---

### Search queries run (three-variant discipline)

1. **Current-year frontier**: "FRED API Python macro series fetch 2026", "APScheduler cron trigger day_of_week sun misfire_grace_time best practice 2024 2025"
2. **Last-2-year window**: "APScheduler weekly cron DST handling interval trigger 2025", "weekly batch job idempotency design patterns BigQuery 2025"
3. **Year-less canonical**: "fredapi", "APScheduler cron trigger", "idempotency batch processing pipeline"

---

### Key findings

1. **heartbeat() has no URL; no HTTP call to backend** -- The `heartbeat()` context manager (`backend/slack_bot/job_runtime.py:66-114`) is a pure Python context manager. Its sink defaults to `logger.info` (line 83). No HTTP, no httpx, no URL. (Source: internal code read, `job_runtime.py:83`)

2. **weekly_fred_refresh has no HTTP calls to backend** -- The `run()` function in `weekly_fred_refresh.py:14-33` calls only `heartbeat()` (no URL), then `_default_fetch()` (returns empty dict stub) or an injected `fetch_fn`. Production wrappers around fredapi are injected via `fetch_fn`; the module itself never imports httpx or calls the backend. No Docker-alias risk. (Source: `weekly_fred_refresh.py:24-31`)

3. **_default_fetch is a no-op stub; production must inject fredapi** -- `_default_fetch()` at line 36-37 returns `{s: [] for s in series}` with the comment "injected in tests; production wraps fredapi". This means production execution requires a non-default `fetch_fn` to actually call FRED. If no `fetch_fn` is injected (APScheduler calls `run()` with no kwargs), the job runs with zero real data. (Source: `weekly_fred_refresh.py:36-37`)

4. **_default_write is also a stub** -- `_default_write()` at line 40-41 returns `len(rows)` (i.e., the number of series keys), never touches BigQuery. Same injection requirement: production needs a real `write_fn`. (Source: `weekly_fred_refresh.py:40-41`)

5. **APScheduler trigger: `day_of_week="sun", hour=2`, no timezone specified** -- `scheduler.py:523` configures the trigger with `{"day_of_week": "sun", "hour": 2, "misfire_grace_time": 7200, "coalesce": True}`. No timezone is passed, so it defaults to the scheduler's timezone. If the scheduler is configured in a DST-observing timezone, Sunday 2am is DST-adjacent (spring forward typically removes 2am entirely). UTC scheduling would eliminate this risk. (Source: `scheduler.py:522-523`, APScheduler docs)

6. **Idempotency is ISO-week keyed, correctly** -- `IdempotencyKey.weekly()` at `job_runtime.py:51-56` computes `f"{job_name}:{iso.year}-W{iso.week:02d}"`. `weekly_fred_refresh.run()` passes this key to `heartbeat()`. On a Sunday fire, the key will be the current ISO week. If the daemon is restarted before midnight Sunday (i.e., the job fires twice in the same ISO week), the second fire is correctly skipped via `skipped_idempotent`. (Source: `job_runtime.py:51-56`, `weekly_fred_refresh.py:22-27`)

7. **misfire_grace_time=7200 (2h) for weekly job** -- Matches phase-23.3.3 intent (per comment at `scheduler.py:515-518`): a daemon restart within 2 hours of the scheduled Sunday 2am tick does NOT immediately re-fire the missed tick. `coalesce=True` collapses any stacked misfires to one. Correct for a weekly job where a missed week can simply wait. (Source: `scheduler.py:515-523`)

8. **fredapi 0.5.2 (May 2024) — no async support** -- Production FRED fetches via fredapi are synchronous. If APScheduler calls `run()` from a thread pool, this is fine; if called from the async event loop directly, it would block. The scheduler's job executor model (thread-based by default) means this is safe for APScheduler 3.x. (Source: pypi.org/project/fredapi)

9. **_JOB_NAMES bridge merge confirmed** -- `backend/api/job_status_api.py:57` contains `"weekly_fred_refresh"` in `_JOB_NAMES`. The registry is pre-seeded at `job_status_api.py:86`. The bridge pushes `next_run_time` via the APScheduler `EVENT_JOB_SCHEDULED` listener. (Source: `job_status_api.py:55-86`)

10. **Live log confirms scheduling** -- `handoff/logs/slack_bot.log` last line for this topic: `2026-05-09 23:24:23,335 INFO backend.slack_bot.scheduler: phase-9 jobs registered: ['daily_price_refresh', 'weekly_fred_refresh', ...]`. Job is live in the scheduler.

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/slack_bot/jobs/weekly_fred_refresh.py` | 45 | Job entry point: heartbeat + idempotency + stub fetch/write | Active; stubs need production injection |
| `backend/slack_bot/job_runtime.py` | 118 | heartbeat() context manager + IdempotencyStore + IdempotencyKey | Active; no HTTP/URL |
| `backend/slack_bot/scheduler.py:491-548` | ~60 (relevant) | register_phase9_jobs(): APScheduler trigger config for weekly_fred_refresh | Active; no timezone arg |
| `backend/api/job_status_api.py:55-86` | ~32 (relevant) | _JOB_NAMES bridge registration; _registry pre-seeded | Active; weekly_fred_refresh present at line 57 |
| `backend/api/cron_dashboard_api.py:79` | 1 | Cron dashboard manifest entry for weekly_fred_refresh | Active |
| `tests/slack_bot/test_weekly_fred_refresh.py` | 43 | 3 tests: write via injection, idempotency, no live fredapi import | Active; all use injected fns |

---

### Consensus vs debate (external)

Consensus: iso-week keying is the canonical idempotency approach for weekly batch jobs. APScheduler's cron trigger aligns to the weekday (not a 7-day interval), so DST transitions on Sundays can theoretically skip or double-fire a 2am job. Using UTC or scheduling at a non-DST-boundary hour (e.g., 6am or 12pm UTC) is the recommended mitigation.

No debate on the main patterns; the only open question is whether the scheduler's default timezone is UTC or local time (a code read of scheduler initialization would confirm, but out of scope per caller).

---

### Pitfalls (from literature and code)

1. **Production stubs**: Both `_default_fetch` and `_default_write` are stubs. APScheduler calls `run()` with no arguments, so production fires execute the stubs. Zero real FRED data is fetched or written. This is the most significant finding.
2. **DST at 2am Sunday**: APScheduler cron triggers fire on wall-clock time. If the scheduler timezone observes DST, the Sunday 2am slot may vanish during spring forward. UTC scheduling eliminates this.
3. **fredapi is synchronous**: If the APScheduler executor is async-only, synchronous FRED API calls would block. Default thread-pool executor is safe; document for future async migration.
4. **In-memory idempotency store**: `IdempotencyStore` is in-memory. A daemon restart resets it. Since `misfire_grace_time=7200` prevents immediate re-fire, the risk window is narrow, but a production BQ-backed store would be fully safe across restarts.

---

### Application to pyfinagent (mapping external findings to file:line anchors)

| Finding | File:line | Action needed? |
|---------|-----------|----------------|
| No HTTP call to backend in job | `weekly_fred_refresh.py:24-33`, `job_runtime.py:83` | No — confirmed safe, no Docker-alias risk |
| heartbeat() wired correctly | `weekly_fred_refresh.py:22-27`, `job_runtime.py:66-114` | No — ISO-week key + heartbeat used correctly |
| Production stubs need injection | `weekly_fred_refresh.py:36-37`, `40-41` | YES — phase-23.5.8 must verify or document that a real fetch_fn/write_fn is injected at call site |
| DST risk at 2am Sunday | `scheduler.py:523` | Low-priority note — scheduler timezone not yet confirmed |
| Idempotency resets on restart | `job_runtime.py:39` (`_GLOBAL_STORE`) | Low — 7200s misfire_grace_time is the guard |
| _JOB_NAMES includes job | `job_status_api.py:57` | No — confirmed present |
| Live scheduler confirms job registered | `handoff/logs/slack_bot.log` | No — verified at 2026-05-09 23:24:23 |

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total (16 collected including snippet-only)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (6 files inspected)
- [x] Contradictions / consensus noted (production stubs are the key finding)
- [x] All claims cited per-claim with URL + file:line

---

### Three answers required by caller

**1. Does `weekly_fred_refresh` (or its called code) have any Docker-alias bug equivalent?**

No. The job has no HTTP calls to the backend. `heartbeat()` uses a logger sink by default (`job_runtime.py:83`) — no URL. The cross-process push via `_HEARTBEAT_URL=127.0.0.1` is in `_aps_to_heartbeat()` (a separate path, not called by this job). `_default_fetch` makes no HTTP calls at all (it returns a stub dict). Any production `fetch_fn` would call the FRED API at `api.stlouisfed.org` — an external URL, not a Docker alias. No Docker-alias risk exists for this job.

**2. Is `heartbeat()` wired correctly for this job?**

Yes. `weekly_fred_refresh.run()` at line 24 calls `with heartbeat(JOB_NAME, idempotency_key=key, store=store)`. The key is an ISO-week string (`weekly_fred_refresh:2026-W17`). The heartbeat context manager emits started/ok/failed events, marks the key on success (`job_runtime.py:112-113`), and skips on re-entry (`job_runtime.py:92-98`). This matches the established pattern for weekly jobs. Wiring is correct.

**3. Is the verification criterion a TRUE liveness signal?**

Partial. The criterion checks that `/api/jobs/all` returns `status != "manifest"` and `next_run != null` for `weekly_fred_refresh`. This is a TRUE liveness signal for the bridge+scheduler integration (confirming APScheduler registered the job and the next_run time is populated). It is NOT a signal that the job has actually executed or that real FRED data flows -- the criterion would pass even if `_default_fetch` (the stub) is the active fetch path. The criterion is a good scheduler-registration check, but a full execution-path check would require either waiting for the Sunday 2am fire or triggering the job manually and verifying BQ writes.

The critical concern for follow-up: APScheduler calls `run()` with no keyword arguments. Both `fetch_fn` and `write_fn` default to stubs that write zero real data. If there is no call-site that injects real implementations (e.g., a wrapper in `scheduler.py` that passes a fredapi-backed `fetch_fn`), the job runs as a no-op in production. The `register_phase9_jobs()` function at `scheduler.py:535-548` calls `scheduler.add_job(func, ...)` where `func = getattr(mod, "run")` with no partial application of `fetch_fn` or `write_fn`. This means production fires call `run()` with zero kwargs -- the stubs are active in production today.

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "gate_passed": true
}
```
