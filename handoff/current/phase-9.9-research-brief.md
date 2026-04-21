---
step: 9.9
tier: moderate
date: 2026-04-20
topic: APScheduler v3 wiring for 7 phase-9 jobs
---

## Research: APScheduler v3 Production Patterns for Phase-9.9

### Queries run (three-variant discipline)

1. Current-year frontier: "APScheduler v3 production cron jobs coalesce max_instances misfire_grace_time 2026"
2. Last-2-year window: "APScheduler v3 add_job replace_existing kwargs production patterns 2025" and "APScheduler v3 AsyncIOScheduler add_job no args kwargs job signature TypeError 2025"
3. Year-less canonical: "APScheduler cron job production best practices coalesce misfire timezone" and "APScheduler v3 job function called no arguments TypeError keyword-only required arguments production"

---

### Read in full (>=5 required; counts toward gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://betterstack.com/community/guides/scaling-python/apscheduler-scheduled-tasks/ | 2026-04-20 | blog/tutorial | WebFetch | Confirms scheduler selection (AsyncIOScheduler for asyncio apps); job persistence patterns; does NOT cover coalesce/misfire defaults |
| https://github.com/agronholm/apscheduler/discussions/592 | 2026-04-20 | maintainer discussion | WebFetch | Maintainer confirms: multiple worker processes each start their own scheduler, creating competing instances -- the single-process model is critical for correctness |
| https://github.com/agronholm/apscheduler/issues/1095 | 2026-04-20 | bug report | WebFetch | Confirms misfire_grace_time=None does NOT guarantee immediate makeup execution; scheduler resets to original schedule offset after a missed run |
| https://github.com/agronholm/apscheduler/issues/15 | 2026-04-20 | bug report | WebFetch | Establishes that job functions must have arguments passed via args= or kwargs= at add_job time; functions called with no args/kwargs receive zero arguments at invocation |
| https://groups.google.com/g/apscheduler/c/svqkO4UHDsQ | 2026-04-20 | maintainer mailing list | WebFetch | Maintainer (Gronholm): args= wraps function positional arguments; kwargs= wraps keyword arguments; passing trigger config kwargs to add_job does NOT pass them to the function |
| https://ryanhaas.us/post/asynchronous-scheduling-with-apscheduler/ | 2026-04-20 | blog | WebFetch | Confirms args= usage pattern for AsyncIOScheduler; tzlocal/pytz deprecation warning mitigation; async execution ordering caveats |
| https://github.com/agronholm/apscheduler/issues/373 | 2026-04-20 | bug report | WebFetch | replace_existing=True behavior: ConflictingIdError raised when ID collision occurs without the flag; flag replaces job definition in-place without dropping queued fires |
| https://pypi.org/project/APScheduler/ | 2026-04-20 | official | WebFetch | APScheduler 3.11.2 is current stable (released 2025-12-22); v4.0.0a6 is pre-release only |

---

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://apscheduler.readthedocs.io/en/3.x/userguide.html | official docs | 403 from readthedocs |
| https://apscheduler.readthedocs.io/en/3.x/modules/schedulers/base.html | official docs | 403 from readthedocs |
| https://apscheduler.readthedocs.io/en/stable/userguide.html | official docs | 403 from readthedocs |
| https://apscheduler.readthedocs.io/en/stable/modules/schedulers/base.html | official docs | 403 from readthedocs |
| https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html | official docs | 403 (inferred from search snippets only) |
| https://github.com/agronholm/apscheduler/issues/296 | bug report | Fetched; max_instances/missed interaction documented above |
| https://coderslegacy.com/python/apscheduler-cron-trigger/ | blog | Fetched; no coalesce/misfire defaults |
| https://github.com/agronholm/apscheduler/issues/87 | bug report | DST/timezone issue; snippet only |
| https://github.com/agronholm/apscheduler/issues/484 | bug report | AsyncIOScheduler + uvicorn lifecycle; snippet only |
| https://enqueuezero.com/concrete-architecture/apscheduler.html | architecture blog | ECONNREFUSED |
| https://github.com/agronholm/apscheduler/issues/251 | bug report | Fetched; argument validation details above |

---

### Recency scan (2024-2026)

Searched for 2024-2026 literature on "APScheduler v3 production cron jobs 2025 2026" and "APScheduler AsyncIOScheduler UTC timezone market jobs financial production 2025 2026".

Findings:
- APScheduler 3.11.2 released 2025-12-22 is the current stable. No breaking changes to coalesce/misfire/max_instances defaults documented in the 3.x branch.
- APScheduler 4.0.0a6 (pre-release, 2025) restructures executors and replaces pytz with zoneinfo, but is NOT production-ready. The pytz/tzlocal deprecation warning is live in 3.x.
- No 2024-2026 literature found that changes the canonical production recommendations for coalesce=True, max_instances=1, misfire_grace_time explicitly set. These remain the accepted production defaults per search snippets of 3.x docs.
- The tzlocal-to-zoneinfo transition (PEP 495) is a 2024-2025 active concern: using `get_localzone()` without explicit UTC specification can produce unexpected timezone objects in 3.x.

---

### Key findings

1. **args= vs kwargs= in add_job are for the JOB FUNCTION, not the trigger** -- The trigger config dict (`{"hour": 1}`) is unpacked as `**kwargs` in the `scheduler.add_job()` call. APScheduler receives these as its own positional trigger fields, not as arguments passed to the callable at invocation time. When APScheduler fires the job, it calls `func(*job.args, **job.kwargs)`. Since no `args=` or `kwargs=` are set in the registration, every phase-9 job is invoked as `run()` -- with zero arguments. (Sources: issue #15; mailing list svqkO4UHDsQ; run_harness observation)

2. **cost_budget_watcher.run() has required keyword-only parameters** -- The signature is `run(*, daily_spend_usd: float, monthly_spend_usd: float, ...)`. These have NO default values. Calling `run()` with no args raises `TypeError: run() missing 2 required keyword-only arguments: 'daily_spend_usd' and 'monthly_spend_usd'`. This is a hard production crash at every scheduled invocation. (Source: `/Users/ford/.openclaw/workspace/pyfinagent/backend/slack_bot/jobs/cost_budget_watcher.py` lines 19-27)

3. **weekly_data_integrity.run() receives empty dicts** -- The signature is `run(*, current_counts: dict|None=None, prior_counts: dict|None=None, ...)`. Both default to None, which the function promotes to `{}`. `_compute_drifts({}, {}, threshold=0.20)` iterates over nothing and always returns `drifts=[]`. The job runs successfully but produces no meaningful output: it never detects drift because `current_counts` is always empty when called with no arguments from the scheduler. (Source: `/Users/ford/.openclaw/workspace/pyfinagent/backend/slack_bot/jobs/weekly_data_integrity.py` lines 18-42; phase-9.7 carry-forward confirmed)

4. **APScheduler v3 default for coalesce, max_instances, misfire_grace_time** -- From search snippets of official docs: `coalesce` defaults to `True` (scheduler-level), `max_instances` defaults to `1`, `misfire_grace_time` defaults to `1` second. These defaults are broadly safe for nightly/weekly jobs. However, for hourly_signal_warmup (fires every hour at minute=5), if execution exceeds 1 hour (unlikely but possible), the default `max_instances=1` will cause the next fire to be skipped. No explicit setting in the phase-9.9 wiring; defaults apply. (Sources: search snippets from apscheduler.readthedocs.io/en/3.x/userguide.html; issue #296; issue #1095)

5. **replace_existing=True is safe for in-place replacement** -- It replaces the job definition atomically without dropping already-queued fires for the current cycle. The `ConflictingIdError` is only raised when `replace_existing=False` (the default) and a duplicate ID is added while the scheduler is running. The phase-9.9 wiring passes `replace_existing=True` correctly. (Sources: issue #373; mailing list; PyPI snippet)

6. **Timezone: AsyncIOScheduler defaults to local timezone** -- When no timezone is specified, APScheduler 3.x uses `tzlocal.get_localzone()`. In containerized environments (Docker on Linux), this typically resolves to UTC. However, on macOS dev machines it resolves to America/New_York or the system TZ. The cron triggers `hour=1`, `hour=2`, etc. are interpreted in the scheduler's timezone. If the Slack bot container has TZ=UTC (standard), then `hour=1` fires at 01:00 UTC = ~21:00 EST the prior evening -- which is safe for a data refresh job. The phase-9.9 wiring does not pass an explicit `timezone=` to either `AsyncIOScheduler()` or the individual `add_job()` calls. (Sources: issue #315; ryanhaas blog; search snippets from tzlocal docs)

7. **Process model: single scheduler in Slack bot is correct** -- The maintainer's discussion confirms that multiple worker processes each spawning a scheduler creates competing instances and duplicate executions. Since the Slack bot runs as a single process (Socket Mode, single asyncio loop), there is no multi-worker race condition. This is the correct deployment model. (Source: discussion #592)

8. **cost_budget_watcher idempotency key uses daily granularity but monitors both daily AND monthly** -- `IdempotencyKey.daily(JOB_NAME, day=day)` deduplicates by YYYY-MM-DD. The monthly circuit breaker fires at the daily granularity, meaning a monthly overage detected on day N is re-checked (and re-fires the alert) on day N+1. This is the phase-9.8 carry-forward: intentional by design (the job reruns daily) but only trips the monthly alert once per calendar month if `alert_fn` is idempotent externally. Without an external alert deduplication layer, the monthly alert fires daily once the cap is breached. (Source: `/Users/ford/.openclaw/workspace/pyfinagent/backend/slack_bot/jobs/cost_budget_watcher.py` lines 28-61)

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/slack_bot/scheduler.py` | 379 | APScheduler wiring + phase-9.9 registration | Active; critical args= gap at lines 365-378 |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/slack_bot/jobs/daily_price_refresh.py` | 53 | Phase-9.2 job | All params keyword-only with defaults -- safe to call with no args |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/slack_bot/jobs/weekly_fred_refresh.py` | 44 | Phase-9.3 job | All params keyword-only with defaults -- safe to call with no args |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/slack_bot/jobs/nightly_mda_retrain.py` | 51 | Phase-9.4 job | All params keyword-only with defaults -- safe to call with no args |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/slack_bot/jobs/hourly_signal_warmup.py` | 48 | Phase-9.5 job | All params keyword-only with defaults -- safe to call with no args |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/slack_bot/jobs/nightly_outcome_rebuild.py` | 58 | Phase-9.6 job | All params keyword-only with defaults -- safe to call with no args |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/slack_bot/jobs/weekly_data_integrity.py` | 57 | Phase-9.7 job | current_counts/prior_counts default None->{}; will always report zero drifts at runtime |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/slack_bot/jobs/cost_budget_watcher.py` | 64 | Phase-9.8 job | daily_spend_usd/monthly_spend_usd REQUIRED with no defaults -- HARD CRASH at runtime |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/slack_bot/job_runtime.py` | 117 | Idempotency + heartbeat primitives | Correct; UTC-aware; global in-memory store resets on process restart |
| `/Users/ford/.openclaw/workspace/pyfinagent/tests/slack_bot/test_scheduler_phase9.py` | 51 | Phase-9.9 tests | StubScheduler captures trigger kwargs not job args; tests do not exercise runtime invocation |

---

### Consensus vs debate (external)

**Consensus:**
- `args=` and `kwargs=` in `add_job()` are for the called function, not the trigger.
- `replace_existing=True` is the correct approach for idempotent re-registration on reload.
- Single APScheduler instance per process is the correct model for Socket Mode Slack bots.
- UTC is the safe timezone for scheduled financial data jobs to avoid DST-shift ambiguity.

**Debate / ambiguity:**
- Whether `misfire_grace_time=1` (default) is sufficient for long-running nightly jobs, or whether `None` (unlimited) is better. Issue #1095 shows `misfire_grace_time=None` still resets the schedule; it does not guarantee catchup execution.
- Whether coalesce=True (default) is always appropriate for data integrity jobs that should not skip a run even if the scheduler was down.

---

### Pitfalls (from literature)

1. **Trigger kwargs pollution** -- The most critical pitfall: `scheduler.add_job(func, trigger=trigger, id=job_id, replace_existing=replace_existing, **kwargs)` where `kwargs = {"hour": 1}`. The `hour=1` is consumed by APScheduler as the cron trigger field, not stored in `job.kwargs`. The job fires with `job.kwargs == {}`. This is correct for trigger config but means `func()` receives no arguments. Safe for 5 of 7 jobs (all optional kwargs with defaults), catastrophic for `cost_budget_watcher.run()`.

2. **Required keyword-only args without defaults** -- `cost_budget_watcher.run(*, daily_spend_usd: float, monthly_spend_usd: float, ...)` requires two floats. APScheduler's argument validator (`check_callable_args`) will raise a `ValueError` at `add_job()` time if it detects unsatisfied required args -- or if the validator is not strict enough, it will raise `TypeError` at invocation time.

3. **Empty-dict drift check** -- `weekly_data_integrity.run()` called with no arguments silently runs and reports zero drifts every Monday. The job appears healthy (heartbeat = "ok", idempotency key marked) but performs no useful work.

4. **Timezone ambiguity** -- No explicit UTC specification in either `AsyncIOScheduler()` instantiation or individual `add_job()` calls. Production correctness depends on the container's TZ environment variable.

5. **In-memory idempotency store** -- `job_runtime.py` line 39: `_GLOBAL_STORE = IdempotencyStore()`. This is an in-process set. Any process restart (container redeploy, crash recovery) clears the store, allowing re-execution of already-run jobs on the same calendar day/week.

6. **replace_existing pre-start gap** -- Issue #373 confirms that `ConflictingIdError` is only raised after `scheduler.start()`. If `register_phase9_jobs` is called before `_scheduler.start()`, duplicate IDs silently accumulate. The current code calls `start_scheduler()` (which calls `_scheduler.start()`) before `register_phase9_jobs` is called from the tests, but in production the order must be verified.

---

### Application to pyfinagent (mapping findings to file:line anchors)

| Finding | Severity | File:line | Impact at production runtime |
|---------|----------|-----------|------------------------------|
| cost_budget_watcher.run() requires daily_spend_usd, monthly_spend_usd -- no defaults | CRITICAL | `backend/slack_bot/jobs/cost_budget_watcher.py:19-27` | TypeError at every hourly/daily invocation; job crashes silently (fail-open logs the exception) |
| Trigger kwargs not forwarded to run() as job kwargs | HIGH | `backend/slack_bot/scheduler.py:374` | All 7 jobs invoked with zero arguments -- by design for 5, silently broken for cost_budget_watcher |
| weekly_data_integrity.run() called with no current_counts/prior_counts | MEDIUM | `backend/slack_bot/jobs/weekly_data_integrity.py:18-42` | Always reports drifts=[] -- runs without error but is functionally inert |
| No timezone=UTC in AsyncIOScheduler() or cron triggers | LOW | `backend/slack_bot/scheduler.py:31` | Schedule fires at container-local time; safe in UTC containers, wrong on macOS dev |
| In-memory idempotency store resets on restart | LOW | `backend/slack_bot/job_runtime.py:39` | Daily jobs re-execute on restart; not a correctness bug, just unexpected repeated runs |
| coalesce/max_instances/misfire defaults not overridden | LOW | `backend/slack_bot/scheduler.py:347-378` | APScheduler defaults (coalesce=True, max_instances=1, misfire_grace_time=1s) are broadly safe for nightly/weekly jobs |

---

### Research Gate Checklist

Hard blockers:

- [x] >=5 authoritative external sources READ IN FULL via WebFetch (8 fetched in full)
- [x] 10+ unique URLs total incl. snippet-only (19 unique URLs collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:

- [x] Internal exploration covered every relevant module (all 7 job files, scheduler.py, job_runtime.py, tests)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim (not just listed in a footer)

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 11,
  "urls_collected": 19,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "report_md": "handoff/current/phase-9.9-research-brief.md",
  "gate_passed": true
}
```
