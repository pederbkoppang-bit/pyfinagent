## Research: phase-23.5.10 -- Cron job verification: hourly_signal_warmup (slack_bot, phase-9.5)

Tier assumed: simple (stated by caller).
Date accessed: 2026-05-10.

---

### Search queries run (three-variant discipline)

1. **Current-year frontier**: "cache warmup pattern hourly interval job idempotency 2026"
2. **Last-2-year window**: "cache warmup idempotency duplicate fire safety no-op pattern Python scheduler 2025"
3. **Year-less canonical**: "APScheduler CronTrigger vs IntervalTrigger when to use each scheduling pattern"
4. Supplemental: "APScheduler CronTrigger minute only fires every hour production Python 2025 2026"
5. Supplemental: "production stub pattern background job scheduled task testing Python inject callable 2025"

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://oneuptime.com/blog/post/2026-01-30-cache-warming-strategies/view | 2026-05-10 | blog (2026) | WebFetch full | `TimeBasedWarmer` hourly pattern; existence-check before warm (no-op-when-warm); skip already-cached keys |
| https://aerospike.com/blog/cache-warming-explained/ | 2026-05-10 | vendor doc | WebFetch full | Don't warm everything; write-through preferred over set-and-forget periodic batches; throttle to avoid backend overload |
| https://medium.com/@pinakdatta/why-your-python-data-pipeline-breaks-on-reruns-and-how-idempotency-fixes-it-b9c13082435f | 2026-05-10 | blog (Mar 2026) | WebFetch full | Upsert/unique-key/state-check patterns; "if file_name in processed_files: continue" is the canonical no-op guard |
| https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html | 2026-05-10 | official doc | WebFetch full | CronTrigger with `minute=5` only: hour defaults to `*` (all hours), fires every hour at :05 -- confirmed matches observed log |
| https://betterstack.com/community/guides/scaling-python/apscheduler-scheduled-tasks/ | 2026-05-10 | authoritative guide | WebFetch full | IntervalTrigger for fixed elapsed-time spacing; CronTrigger for calendar/wall-clock; CronTrigger `minute=N` is idiomatic for "every hour at :N" |
| https://medium.com/@ThinkingLoop/7-scheduler-strategies-for-python-jobs-celery-rq-arq-48b1eb5f8f79 | 2026-05-10 | blog (2025) | WebFetch full | Redis SETNX idempotency pattern; "idempotency by design" as strategy #2; no-op-when-key-exists is the canonical guard |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://globaltill.com/warmup-cache-requests/ | blog | Snippet sufficient; overlaps with oneuptime coverage |
| https://onewebcare.com/blog/warmup-cache-request/ | blog | Snippet sufficient |
| https://denebrixai.com/blog/warmup-cache-request/ | blog | Snippet sufficient |
| https://www.ioriver.io/terms/cache-warming | vendor glossary | Snippet sufficient |
| https://github.com/agronholm/apscheduler/issues/606 | GitHub issue | DST edge-case; not relevant to hourly warmup |
| https://snyk.io/advisor/python/APScheduler/functions/apscheduler.triggers.cron.CronTrigger | Snyk | Snippet sufficient |
| https://danielsarney.com/blog/python-background-task-processing-2025-handling-asynchronous-work-modern-applications/ | blog (2025) | Background tasks overview; no new warmup-specific content |
| https://medium.com/@raghavrg09/caching-in-python-applications-from-simple-dict-to-redis-llm-agents-d1b50d97fc17 | blog (Apr 2026) | Snippet sufficient |
| https://oneuptime.com/blog/post/2026-01-25-prevent-duplicate-requests-python/view | blog (2026) | Snippet sufficient; overlaps with idempotency article |
| https://dev.to/leonardkachi/idempotency-keys-your-apis-safety-net-against-chaos-j1b | blog | Snippet sufficient |

---

### Recency scan (2024-2026)

Searched for 2024-2026 literature on cache warmup, hourly interval scheduling, and idempotency for Python scheduled jobs. Result: found 4 new findings from 2025-2026 that complement canonical sources:

1. oneuptime 2026-01-30: `TimeBasedWarmer` pattern with hour-aligned schedules and no-op-when-warm guards (new; not in older APScheduler docs).
2. Pinak Datta March 2026: idempotency-by-design in Python data pipelines -- upsert/skip-if-seen patterns now mainstream (complements but does not supersede classic scheduler docs).
3. ThinkingLoop 2025: Redis SETNX as the production idempotency guard for scheduled jobs (well-established by 2025).
4. Aerospike 2026 cache warming blog: write-through preferred over periodic batch warming for freshness; periodic warmup still valid for pre-population of a watchlist-backed cache.

No findings that supersede the core pattern used by `hourly_signal_warmup`. The injectable-fn + idempotency-key + in-memory store approach is consistent with 2025-2026 best practice.

---

### Key findings

1. **CronTrigger `minute=5` fires every hour at :05, not once per hour starting from registration.** APScheduler docs: "Fields greater than the least significant explicitly defined field default to `*`." With `minute=5` only, `hour` defaults to `*` -- the job fires at HH:05 for every hour. (Source: APScheduler official docs, https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html)

2. **The scheduler.py mapping uses `"cron"` trigger (not `IntervalTrigger`) with `minute=5`.** This is calendar-aligned ("every hour at :05"), which is correct for a cache warmup. IntervalTrigger would give wall-clock elapsed intervals from the last fire, which drifts over time. CronTrigger is appropriate here. (Source: scheduler.py:526-527, internal)

3. **No HTTP calls to backend.** The job is entirely self-contained: it calls `compute_signal_fn(ticker)` (injectable, defaults to `lambda t: {"score": 0.0}`) and writes into `cache_backend` (injectable dict, defaults to `{}`). No `_BACKEND_URL`, no `_LOCAL_BACKEND_URL`, no httpx, no external I/O at all. (Source: hourly_signal_warmup.py:29-31, internal)

4. **`heartbeat()` is correctly wired.** The job uses `with heartbeat(JOB_NAME, idempotency_key=key, store=store) as state:` at line 24. The idempotency key is `IdempotencyKey.hourly(JOB_NAME, iso_hour=iso_hour)` which produces `"hourly_signal_warmup:2026-05-09T22"` style keys -- one per UTC hour. The heartbeat context manager emits `started` on entry and `ok`/`failed`/`skipped_idempotent` on exit. (Source: hourly_signal_warmup.py:21-34; job_runtime.py:66-114)

5. **Production-stub analysis: NOT affected (real work, like nightly_mda_retrain).** The default `compute_signal_fn` is `lambda t: {"score": 0.0}` -- this IS a production stub for the signal computation, but it is different in nature from the daily_price_refresh/weekly_fred_refresh stubs. Those stubs replaced an external API call with a no-op. Here, the stub replaces an in-process signal computation with a trivial placeholder. The cache warmup itself (iterating the watchlist and populating the dict) IS real work -- the job legitimately warms the cache. The signal values it writes are stubbed (score=0.0) if `compute_signal_fn` is not wired in production. This is a partial stub: cache infrastructure is real; signal computation may be a no-op depending on whether the production caller injects a real fn. (Source: hourly_signal_warmup.py:29; internal inspection)

6. **Idempotency is hourly-granular.** The key format `{job_name}:{YYYY-MM-DDTHH}` means a second fire within the same UTC hour (e.g., a restart) is safely skipped via the `skipped_idempotent` path. This matches the 2025-2026 pattern of using a structured key per time-window rather than a mutex lock. (Source: job_runtime.py:59-63; ThinkingLoop 2025 blog)

7. **Live log confirms real execution.** The log at `handoff/logs/slack_bot.log` shows:
   - Registration at 2026-05-09 23:24:23: `['daily_price_refresh', ..., 'hourly_signal_warmup', ...]`
   - Fire at 2026-05-10 00:05:00 UTC: `status: started` then `status: ok`, duration ~0.0002s
   - The 0.0002s duration is consistent with the no-op default fn + a tiny watchlist (likely `["AAPL", "MSFT", "SPY"]` fallback since settings.watchlist may be empty).

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/slack_bot/jobs/hourly_signal_warmup.py` | 49 | Phase-9.5 job: hourly cache warmup | Active, no HTTP calls |
| `backend/slack_bot/job_runtime.py` | 117 | heartbeat() context mgr + IdempotencyStore/Key | Active, correct |
| `backend/slack_bot/scheduler.py` (lines 502-548) | ~47 | register_phase9_jobs() + CronTrigger mapping | Active, trigger is "cron" minute=5 |
| `backend/api/job_status_api.py` (line 59) | ~190 | `_JOB_NAMES` includes `hourly_signal_warmup` at index 9.5 | Active |
| `tests/slack_bot/test_hourly_signal_warmup.py` | 55 | 3 tests: cache population, settings fallback, custom cache | Active |

---

### Consensus vs debate (external)

**Consensus**: CronTrigger `minute=N` for "every hour at :N" is idiomatic and correct; IntervalTrigger is for elapsed-time spacing. Both are appropriate for hourly cadence but CronTrigger gives wall-clock alignment which is better for observability. Idempotency keys scoped to a time-window (hour/day/week) are the 2025-2026 standard pattern.

**Debate**: Whether the default `lambda t: {"score": 0.0}` constitutes a production stub depends on whether the scheduler's `run` is called with an injected `compute_signal_fn` in production. The job structure is correct for DI; whether production wires in a real fn is a separate question not answerable from code inspection alone.

### Pitfalls (from literature)

1. **Set-and-forget warmup without refresh** -- Aerospike blog warns this creates stale data. The hourly refiring mitigates this. The 0.0002s duration suggests the real signal fn is NOT being injected in production, meaning the cache fills with `{"score": 0.0}` placeholders -- potentially a stale-value risk.
2. **CronTrigger restart misfire** -- APScheduler GitHub issue #606: DST transitions can cause skipped fires. The `misfire_grace_time=600` (10 min) at scheduler.py:527 mitigates this.
3. **In-memory idempotency store resets on restart** -- `_GLOBAL_STORE` is an in-process set; a restart clears it, so the first post-restart fire is always un-skipped. This is correct behavior (you want a warmup after restart), but means the idempotency guard only prevents double-fires WITHIN a session.

### Application to pyfinagent

| Finding | File:line | Implication |
|---------|-----------|-------------|
| No HTTP calls | hourly_signal_warmup.py:29-31 | Docker-alias bug (Answer 1) = NOT APPLICABLE |
| heartbeat() correct | hourly_signal_warmup.py:24; job_runtime.py:66-114 | heartbeat wired correctly (Answer 2) = YES |
| compute_signal_fn default is stub | hourly_signal_warmup.py:29 | Production-stub-affected (Answer 3) -- PARTIAL |
| CronTrigger minute=5 | scheduler.py:526-527 | Fires every hour at :05 UTC; NOT IntervalTrigger despite UI showing "in 27m" |
| Live log ok | handoff/logs/slack_bot.log | status="ok", next_run advancing -- verification criterion is TRUE LIVENESS |

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total incl. snippet-only (16 URLs)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (5 files)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

---

## Three answers for the decision

**Answer 1 -- Docker-alias bug?** NO. `hourly_signal_warmup.py` makes zero HTTP calls. It operates entirely in-process: reads watchlist from settings, calls an injectable compute fn, writes results into an injectable dict. There is no `_BACKEND_URL`, no `_LOCAL_BACKEND_URL`, no `httpx` import, no network I/O. Docker-alias bug risk is zero.

**Answer 2 -- `heartbeat()` correctly wired?** YES. The job calls `with heartbeat(JOB_NAME, idempotency_key=key, store=store) as state:` with a properly formed hourly key. The heartbeat emits `started`, runs the body, emits `ok` on success. The log confirms this with actual `status: started` + `status: ok` entries at 00:05:00 UTC. The idempotency guard skips duplicate fires within the same UTC hour.

**Answer 3 -- Criterion is TRUE LIVENESS or production-stub-affected?** PARTIAL. The job IS live and truly running (not a manifest stub). The cache warmup infrastructure (heartbeat, idempotency, watchlist load, dict population) is real. However, the default `compute_signal_fn` is `lambda t: {"score": 0.0}` -- a stub placeholder. Unless the production scheduler call injects a real signal-computation function, the cache fills with `{"score": 0.0}` for each ticker. The `duration_s: 0.0002s` in the live log is consistent with this stub fn (no real BQ or model calls). Conclusion: the job meets the verification criterion (status != "manifest", next_run populated, status="ok") as TRUE LIVENESS -- it is genuinely registered and firing. But the economic value of the warmup is minimal if compute_signal_fn is not injected with a real implementation.

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "gate_passed": true
}
```
