---
step: phase-23.5.2
cycle_date: 2026-05-08
tier: simple
topic: Cron job verification -- ticket_queue_process_batch (main_apscheduler / IntervalTrigger)
---

# Research Brief -- phase-23.5.2

## Queries run (three-variant discipline)

1. **Current-year frontier:** "apscheduler interval trigger every 5 seconds next_run_time recomputed 2026"
2. **Last-2-year window:** "apscheduler BackgroundScheduler interval trigger 5 seconds coalesce max_instances misfire_grace_time production 2025", "apscheduler AsyncIOScheduler vs BackgroundScheduler FastAPI lifespan 2024 2025"
3. **Year-less canonical:** "apscheduler interval trigger next_run_time always populated high frequency job liveness check", "apscheduler AsyncIOScheduler interval trigger misfire coalesce high frequency queue pull"

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://apscheduler.readthedocs.io/en/3.x/modules/triggers/interval.html | 2026-05-08 | official doc | WebFetch | Constructor: `IntervalTrigger(weeks, days, hours, minutes, seconds, start_date, end_date, timezone, jitter)`. `get_next_fire_time()` returns the next datetime to fire on, or None if no such datetime can be calculated (only when end_date is exceeded). Default `start_date` is `datetime.now() + interval` -- job waits one full interval before first fire unless `next_run_time` override is passed. |
| https://apscheduler.readthedocs.io/en/3.x/userguide.html | 2026-05-08 | official doc | WebFetch | "By default, only one instance of each job is allowed to run at the same time." Misfire: "the scheduler will then check each missed execution time against the job's misfire_grace_time option to see if the execution should still be triggered." Coalesce: "if coalescing is enabled and the scheduler sees one or more queued executions for the job, it will only trigger it once." |
| https://apscheduler.readthedocs.io/en/stable/modules/triggers/interval.html | 2026-05-08 | official doc | WebFetch | `get_next_fire_time(previous_fire_time, now)` -- when start_date is in the past, "calculates the next run time from the current time, based on the past start time." Returns None only when end_date is exceeded. No end_date = next_run_time is never None for a running interval job. |
| https://jdhao.github.io/2024/11/02/python_apascheduler_start_job_immediately/ | 2026-05-08 | authoritative blog | WebFetch | "By setting [next_run_time] to the current datetime, the job executes right after the scheduler starts." Confirms that without the override, IntervalTrigger waits one full interval. Documents the `next_run_time=datetime.now()` override as the canonical approach for immediate-start. Accessed 2026-05-08; article dated 2024-11-02 (within recency window). |
| https://sentry.io/answers/schedule-tasks-with-fastapi/ | 2026-05-08 | authoritative doc (Sentry engineering) | WebFetch | AsyncIOScheduler integrates with FastAPI lifespan via `@asynccontextmanager`; `scheduler.shutdown()` called on exit. IntervalTrigger shown with `start_date` param. BackgroundScheduler and AsyncIOScheduler both support lifespan integration. |
| https://apscheduler.readthedocs.io/en/3.x/modules/schedulers/base.html | 2026-05-08 | official doc | WebFetch | `add_job()` accepts `job_defaults` for scheduler-level defaults; individual jobs override per-call. `max_instances` default is 1 (one concurrent instance per job). `coalesce` default is True per scheduler docs. `misfire_grace_time` default is 1 second (documented by omission; APScheduler source sets `_default_misfire_grace_time = 1`). |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://betterstack.com/community/guides/scaling-python/apscheduler-scheduled-tasks/ | practitioner blog | Fetched; returned only basic interval trigger usage (seconds=5 demo), no production-level coalesce/misfire guidance |
| https://github.com/agronholm/apscheduler/issues/348 | GitHub issue | Fetched; user misconfused cron `second=5` (fires once per minute at :05) with interval trigger; resolved as user error -- confirms interval trigger is the correct choice for "every 5 seconds" |
| https://medium.com/@rasifrazak123/fastapi-scheduling-background-tasks-backgroundtasks-vs-apscheduler-vs-celery-complete-guide-ff90d6be524b | blog | Fetched; high-level comparison only; no misfire_grace_time or queue-pull depth |
| https://rajansahu713.medium.com/implementing-background-job-scheduling-in-fastapi-with-apscheduler-6f5fdabf3186 | blog | Snippet; introductory-level; confirms AsyncIOScheduler + lifespan pattern |
| https://snyk.io/advisor/python/APScheduler/functions/apscheduler.triggers.interval.IntervalTrigger | advisor | Snippet; confirms constructor signature; no new depth |
| https://github.com/agronholm/apscheduler/issues/1095 | GitHub issue | Snippet; confirms missed jobs run at next fire time when coalesce=False; reinforces misfire behavior |
| https://apscheduler.readthedocs.io/en/stable/modules/schedulers/base.html | official doc | Second fetch attempt; returned same content as 3.x version without default values in prose |
| https://apscheduler.readthedocs.io/en/master/api.html | official doc | Snippet; APScheduler 4.x API -- not applicable (pyfinagent uses 3.x) |
| https://www.nashruddinamin.com/blog/running-scheduled-jobs-in-fastapi | blog | Fetched; shows AsyncIOScheduler with FastAPI but no comparative or advanced config content |
| https://ahaw021.medium.com/scheduled-jobs-with-fastapi-and-apscheduler-5a4c50580b0e | blog | Snippet; confirms FastAPI lifespan integration pattern only |

## Recency scan (2024-2026)

Searched: "apscheduler interval trigger every 5 seconds next_run_time recomputed 2026", "apscheduler BackgroundScheduler interval trigger 5 seconds coalesce max_instances misfire_grace_time production 2025", "apscheduler AsyncIOScheduler vs BackgroundScheduler FastAPI lifespan 2024 2025".

**Result:** The most relevant 2024-2026 finding is the jdhao.github.io article (2024-11-02) documenting that `next_run_time=datetime.now()` is the canonical pattern for immediate-start interval jobs -- relevant because pyfinagent does NOT use this override (job waits one full 5-second interval before first fire after app restart). No new findings in the 2024-2026 window contradict the 23.5.1 sources or the canonical APScheduler 3.x documentation. APScheduler 4.x pre-release changes (stateful triggers, unified Scheduler class) remain pre-release and do not affect pyfinagent's 3.x API.

## Key findings

1. **IntervalTrigger `next_run_time` is never None for a running job without `end_date`.** `get_next_fire_time()` returns None only when the job's `end_date` is exceeded. `ticket_queue_process_batch` has no `end_date` (confirmed `main.py:220-225` -- `add_job(process_batch, 'interval', seconds=5, id=..., name=..., replace_existing=True)` with no `end_date`). Therefore `next_run_time is not None` is a TAUTOLOGICAL truth for this job as long as the scheduler is running. (Source: APScheduler 3.x IntervalTrigger doc, https://apscheduler.readthedocs.io/en/stable/modules/triggers/interval.html)

2. **`next_run_time` is recomputed by the trigger immediately after each fire, before the next scheduling cycle.** The scheduler calls `get_next_fire_time(previous_fire_time, now)` post-execution, updating the stored `next_run_time` to `previous_fire_time + interval`. For a 5-second job, this means `next_run_time` is always within 5 seconds of current wall time. The live API response confirms: `next_run: "2026-05-08T18:06:36.633914+02:00"` (well-populated). (Source: APScheduler 3.x userguide + stable interval docs)

3. **`coalesce` default is True, `max_instances` default is 1.** APScheduler's scheduler-level default is coalesce=True: if the event loop is busy and multiple 5-second fires are missed, only one fire is executed on catch-up. This is the correct behavior for a queue-pull job -- "pull one batch on catch-up, not N batches." `max_instances=1` prevents concurrent re-entry if a slow batch (e.g., waiting for an Anthropic API response up to 60s) is still running when the next 5-second tick arrives. (Source: APScheduler 3.x userguide, https://apscheduler.readthedocs.io/en/3.x/userguide.html)

4. **`misfire_grace_time` default is 1 second.** For a 5-second interval job, a 1-second grace means: if the scheduler is delayed by more than 1 second past the scheduled fire time (e.g., event loop blocked by a GC pause or the 60-second agent thread), the missed fire is discarded and only the next scheduled tick is executed. Combined with coalesce=True, this means a 60-second agent call causes ~12 missed 5-second fires, all discarded, and the batch resumes at the next tick after the agent call completes. This is safe for the ticket-queue use case -- no business harm from missed polls. (Source: APScheduler 3.x userguide misfire section)

5. **`next_run_time is not None` is a MORE reliable liveness indicator for IntervalTrigger than for CronTrigger.** CronTrigger jobs can exhaust their schedule (e.g., `day_of_week='mon-fri'` yields None over a weekend if the scheduler is paused). IntervalTrigger with no `end_date` can never exhaust; `next_run_time` is always non-null while the scheduler runs. This makes the criterion strictly stronger for the interval case. (Source: APScheduler 3.x docs, cross-reference with 23.5.1 CronTrigger analysis)

6. **AsyncIOScheduler is the correct scheduler type for this job.** `ticket_queue_process_batch` calls `async def process_batch()` which internally calls `await processor.process_queue_batch(batch_size=10)`. `main.py:206` correctly instantiates `AsyncIOScheduler()` (not `BackgroundScheduler`) -- this runs jobs in the FastAPI event loop rather than a thread pool, which is correct for async handlers. (Source: Sentry FastAPI scheduling doc, APScheduler userguide; `backend/main.py:201,206`)

7. **The source key `"main_apscheduler"` is applied to ALL registered schedulers.** `cron_dashboard_api.py:202` hard-codes `source="main_apscheduler"` for every scheduler in `get_registered_schedulers()`, regardless of whether it was registered under the name `"main"` or `"queue"`. Both `paper_trading_daily` (main scheduler) and `ticket_queue_process_batch` (queue scheduler) appear as `source="main_apscheduler"` in the API response. The verification command's probe `x["id"]=="ticket_queue_process_batch"` is sufficient to discriminate -- no ambiguity. (Source: `backend/api/cron_dashboard_api.py:199-206`)

8. **Phase-23.3.1 explicitly verified this job as live.** The 23.3.1 audit found `next_run: "2026-05-07T21:25:01+02:00"`, `status: "scheduled"`, `description: "Ticket queue batch processor"` -- and fixed the anonymous ID/name issue. No regressions since. (Source: `handoff/archive/phase-23.3.1/phase-23.3.1-audit-findings.md:29-34`)

## Internal code inventory

| File | Lines / Anchors | Role | Status |
|------|-----------------|------|--------|
| `backend/main.py` | 197-231 | Registers `ticket_queue_process_batch` via `AsyncIOScheduler`, `interval`, `seconds=5`. No `coalesce`, `max_instances`, `misfire_grace_time` overrides -- all APScheduler defaults apply. | Active, authoritative |
| `backend/services/ticket_queue_processor.py` | 1-576 | Defines `TicketQueueProcessor`. Handler is `process_batch()` which calls `processor.process_queue_batch(batch_size=10)`. Queue backed by SQLite (`tickets_db.py`). Max concurrent = 1 (`__init__:30`). Long-running agent calls use daemon thread with 60s timeout (`_spawn_real_agent:243-251`). | Active, authoritative |
| `backend/api/cron_dashboard_api.py` | 160-176 (`_job_to_dict`), 199-206 (`get_all_jobs`) | Builds `/api/jobs/all`. All registered schedulers emit `source="main_apscheduler"`. `status` derived from `next_run_time is not None` at line 174. | Active |
| `backend/db/tickets_db.py` | 403 (`get_ticket_queue_position`) | SQLite-backed ticket store. Queue is in-process/local SQLite, not Redis/BQ. | Active |
| `tests/db/test_tickets_db_no_fd_leak.py` | 1-69 | Phase-23.1.19 FD-leak regression guard. 100-iteration loop asserts net FD delta <= 5. Tests `TicketsDB` CRUD with `contextlib.closing()`. | Active |
| `tests/test_queue_processor.py` | 1-80+ | Integration test for queue processing and agent routing. Tests FIFO, classification routing (`operational->MAIN`, `analytical->Q&A`, `research->Research`), and batch processing. Does NOT assert APScheduler job registration or `next_run_time`. | Active |
| `handoff/archive/phase-23.3.1/phase-23.3.1-audit-findings.md` | full | Phase-23.3.1 confirmed ticket_queue_process_batch live and scheduled; fixed anonymous ID. | Historical |

## Consensus vs debate (external)

**Consensus:** APScheduler 3.x IntervalTrigger with no `end_date` always produces a non-null `next_run_time` after each fire. No authoritative source disputes this. `coalesce=True` and `max_instances=1` are the correct defaults for a single-instance queue-pull job. `AsyncIOScheduler` is the correct choice for async FastAPI handlers.

**No debate:** whether `next_run_time is not None` is a reliable liveness signal -- it is tautologically true for an interval job without `end_date`. The only liveness concern for this job type is a scheduler that has crashed or been garbage-collected, which would make the entire API response empty (no job entry at all), not `next_run_time=null`.

## Pitfalls (from literature and internal audit)

- **Misfire swallowing fires silently (coalesce=True + 1s grace):** A 60-second agent call causes ~12 missed 5-second fires, all silently discarded. No tickets are lost (they stay OPEN in SQLite), but queue drain is slower during heavy agent load. This is by design and documented in the code (`process_queue_batch` returns 0 when no open tickets). NOT a bug for this verification step.
- **Emojis in logger messages (security.md violation):** `ticket_queue_processor.py` has numerous emoji in logger calls (e.g., lines 204, 219, 226, 323, 342, 374, 402, 465, 491). This violates `.claude/rules/security.md` ("ASCII-only logger messages"). Not blocking for 23.5.2 (out of scope per instructions), but noted.
- **AsyncIOScheduler in a lifespan `try/except` block:** The queue scheduler is started inside a `try/except Exception` block (`main.py:200-231`). If `queue_scheduler.start()` raises (e.g., event loop already closed), the exception is caught and logged as WARNING, and the job is never registered. The `/api/jobs/all` response would then have no `ticket_queue_process_batch` entry -- the verification command would catch this correctly (`assert j is not None, "job missing"`).
- **`next_run_time` always non-null for IntervalTrigger without `end_date`:** The `next_run is not None` check is structurally guaranteed -- it CANNOT be the discriminating signal for an unhealthy scheduler state. The real discriminating signal is the existence of the job entry (`j is not None`) and a sane `next_run` timestamp (within ~10 seconds of current wall time).

## Application to pyfinagent (file:line anchors)

**Does the 23.5.1 structural argument carry over to a 5-second IntervalTrigger?**

YES, and it is STRONGER for the interval case:

1. `status != "manifest"` -- same logic as 23.5.1. `_job_to_dict` at `cron_dashboard_api.py:174` sets `status="scheduled"` for any job with `next_run_time is not None`. `"manifest"` is reserved for `_static_to_dict` (`cron_dashboard_api.py:179-188`). A `main_apscheduler` job (including the queue scheduler) can never produce `status="manifest"`.

2. `next_run is not None` -- TAUTOLOGICALLY true for an IntervalTrigger job with no `end_date`. The only scenario where `next_run_time` becomes None is (a) job is paused (`.pause_job()` called explicitly -- not wired in pyfinagent), (b) job is removed (`.remove_job()` called -- would make `j is None`), or (c) `end_date` is exceeded (no `end_date` set). Therefore `next_run is not None` for this job is a scheduler-liveness proxy, not a trigger-saturation proxy: if `next_run` is None, the scheduler has been stopped, which the verification would not reach anyway.

**What the criterion CATCHES (correctly):**
- Job not registered at all (`j is None`) -- scheduler failed to start or exception was swallowed.
- Job registered but paused (`.pause_job()` called externally) -- would yield `next_run=null`, `status="paused"`.

**What the criterion does NOT catch (by design, acceptable for this step):**
- Whether the batch handler is actually executing successfully (no `last_run` tracking -- same gap documented at `cron_dashboard_api.py:173`).
- Whether the SQLite queue has a backlog (not in scope for this verification step).
- Whether agent calls are timing out at 60s (observable only in logs, not in `/api/jobs/all`).

**Conclusion:** `status != "manifest" AND next_run is not None` is COMPLETE and STRONGER for a 5-second IntervalTrigger than it was for the CronTrigger case in 23.5.1. The live API state already satisfies both: `status="scheduled"`, `next_run="2026-05-08T18:06:36.633914+02:00"`. **Verification command SHOULD PASS without any code changes.**

**One higher-frequency concern (liveness, not a blocker):** For a 5-second job, `next_run` is always within 5 seconds of wall time. If `next_run` were anomalously far in the future (e.g., 2 minutes out), it could indicate the scheduler's internal wakeup loop is jammed. The current live `next_run` is ~5s ahead, which is normal. The verification command does not check the timestamp delta, but Main may optionally confirm the wall-clock proximity as a sanity check.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched in full: APScheduler 3.x IntervalTrigger, APScheduler 3.x userguide, APScheduler stable IntervalTrigger, jdhao 2024 blog, Sentry FastAPI doc, APScheduler stable schedulers/base)
- [x] 10+ unique URLs total (incl. snippet-only) -- 16 unique URLs collected
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (main.py, ticket_queue_processor.py, cron_dashboard_api.py, tickets_db.py, test_tickets_db_no_fd_leak.py, test_queue_processor.py, phase-23.3.1 audit)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

---

## Recommendation to Main

**The immutable criterion is sufficient for phase-23.5.2, and is STRUCTURALLY STRONGER than for 23.5.1.**

For an IntervalTrigger job with no `end_date`:
- `status != "manifest"` is structurally impossible for any APScheduler-registered job (same as 23.5.1).
- `next_run is not None` is TAUTOLOGICALLY true as long as the scheduler is running and the job is not paused -- a stronger guarantee than the CronTrigger case where schedule exhaustion (e.g., `end_date` or exhausted cron fields) could produce `None`.

The higher fire frequency (every 5s vs once per trading day) introduces a nuance: `next_run is not None` is LESS INFORMATIVE as a trigger-saturation signal (it can never be None for this job type) but MORE INFORMATIVE as a scheduler-liveness signal (if it IS None, something is very wrong). The real discriminating check for this job is `j is not None` (job is registered at all).

**No higher-frequency liveness concern requires a new criterion.** The concern about `coalesce=True` swallowing fires is real but is not a liveness defect -- missed fires are intentionally discarded per APScheduler's documented behavior, and ticket data is not lost (stays OPEN in SQLite). This is out of scope for the verification step.

**Verification command WILL PASS against current state** -- live probe shows `status="scheduled"`, `next_run="2026-05-08T18:06:36.633914+02:00"`, satisfying both criterion conditions.

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "gate_passed": true
}
```
