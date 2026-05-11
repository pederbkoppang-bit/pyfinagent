---
step: phase-23.5.1
cycle_date: 2026-05-08
tier: simple
topic: Cron job verification — paper_trading_daily (main_apscheduler)
---

# Research Brief — phase-23.5.1

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://apscheduler.readthedocs.io/en/3.x/userguide.html | 2026-05-08 | official doc | WebFetch | "Triggers are completely stateless." "When a job is paused, its next run time is cleared and no further run times will be calculated for it until the job is resumed." |
| https://apscheduler.readthedocs.io/en/3.x/modules/events.html | 2026-05-08 | official doc | WebFetch | `JobExecutionEvent` carries `job_id`, `scheduled_run_time`, `retval`. No `last_run_time` field on the Job object; must be captured in listener. |
| https://apscheduler.readthedocs.io/en/3.x/modules/schedulers/base.html | 2026-05-08 | official doc | WebFetch | `get_job()` returns a `Job` object with `next_run_time` as sole temporal field. No `last_run_time` attribute; documentation is explicit by omission. |
| https://apscheduler.readthedocs.io/en/stable/modules/job.html | 2026-05-08 | official doc | WebFetch | Complete Job attribute list: `id`, `name`, `func`, `args`, `kwargs`, `coalesce`, `trigger`, `executor`, `misfire_grace_time`, `max_instances`, `next_run_time`. No `last_run_time`. |
| https://apscheduler.readthedocs.io/en/master/migration.html | 2026-05-08 | official doc | WebFetch | APScheduler 4.x renames `BackgroundScheduler` → unified `Scheduler` with `start_in_background()`. Triggers become stateful. "Event brokers" replace simple listeners. pyfinagent uses 3.x API (`BackgroundScheduler.add_job`); no migration needed. |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html | official doc | Fetched; returned partial page — key facts captured in separate CronTrigger fetch above; snippet confirms `get_next_fire_time()` returns "next datetime to fire on" |
| https://bytegoblin.io/blog/implementing-background-job-scheduling-in-fastapi-with-apscheduler.mdx | blog | Fetched but introductory-only; no production health-check content returned |
| https://rajansahu713.medium.com/implementing-background-job-scheduling-in-fastapi-with-apscheduler-6f5fdabf3186 | blog | Fetched; intro-level only; confirmed event listener pattern but no "is next_run sufficient?" discussion |
| https://dev.to/hexshift/how-to-run-cron-jobs-in-python-the-right-way-using-apscheduler-4pkn | blog | Fetched; graceful-shutdown focus; no production monitoring depth |
| https://pypi.org/project/APScheduler/ | package index | Snippet only; version lineage reference |
| https://apscheduler.readthedocs.io/en/stable/userguide.html | official doc | Snippet; same as 3.x userguide — confirms "paused" = next_run_time cleared |
| https://snyk.io/advisor/python/APScheduler/functions/apscheduler.events.EVENT_JOB_EXECUTED | advisor | Snippet; confirms mask constant name and wiring pattern |
| https://github.com/amisadmin/fastapi-scheduler | GitHub | Snippet; thin wrapper; no new state insight |
| https://healthchecks.io/ | SaaS product | Snippet; heartbeat-ping SLO model; mentioned as external alt pattern |
| https://reintech.io/blog/data-pipeline-orchestration-airflow-dagster-prefect-2026 | blog | Snippet; orchestrator comparison; not APScheduler-specific |

## Recency scan (2024-2026)

Searched: "APScheduler BackgroundScheduler cron trigger next_run_time semantics 2026", "APScheduler EVENT_JOB_EXECUTED listener FastAPI health check production 2025", APScheduler 4.x migration guide (master branch).

**Result:** APScheduler 4.x (currently in pre-release / master) introduces stateful triggers, a unified `Scheduler` class, and event brokers, but these changes do NOT affect pyfinagent because the codebase pins the 3.x API. The `next_run_time` attribute name and `EVENT_JOB_EXECUTED` listener pattern are unchanged in 3.x through 3.11.x (the latest 3.x release as of 2026-05-08). No breaking changes in the 2024-2026 window affect the `status="scheduled"` + non-null `next_run_time` pattern used in `cron_dashboard_api.py`. No new 2025-2026 practitioner findings supersede the canonical APScheduler 3.x documentation.

## Key findings

1. **`next_run_time` is the ONLY temporal field on a Job object (APScheduler 3.x).** The Job class exposes: `id`, `name`, `func`, `args`, `kwargs`, `coalesce`, `trigger`, `executor`, `misfire_grace_time`, `max_instances`, `next_run_time`. There is no `last_run_time` field. (Source: APScheduler 3.x stable Job reference, https://apscheduler.readthedocs.io/en/stable/modules/job.html)

2. **`next_run_time` is recalculated by the trigger after each fire.** The CronTrigger's `get_next_fire_time()` is called post-execution by the scheduler internals; it finds "the earliest possible time satisfying the conditions in every field" after the previous firing. A paused job has `next_run_time` set to `None`. (Source: APScheduler 3.x CronTrigger doc, https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html)

3. **`last_run_time` must be tracked manually via `EVENT_JOB_EXECUTED`.** The `JobExecutionEvent` object carries `scheduled_run_time` (the time the job was *scheduled* to run, not actual wall-clock completion) and `retval`. To persist last-run state an application must implement a listener: `scheduler.add_listener(fn, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)`. (Source: APScheduler events module, https://apscheduler.readthedocs.io/en/3.x/modules/events.html)

4. **`status="scheduled"` in pyfinagent is a derived field, not a native APScheduler concept.** `cron_dashboard_api.py:174` defines `"status": "scheduled" if nrt is not None else "paused"`. This is local logic mapping `next_run_time is not None` → "scheduled". The comment on line 173 reads explicitly: `# APScheduler doesn't expose this; phase-2 if needed`. (Source: `backend/api/cron_dashboard_api.py:160-176`)

5. **The Slack-bot scheduler (a different process) DOES wire `EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED` via `add_listener`.** The main APScheduler instance (which owns `paper_trading_daily`) does NOT. (Source: `backend/slack_bot/scheduler.py:12-14, 122-124` vs `backend/api/paper_trading.py` — no listener call present)

6. **`last_run: null` for `paper_trading_daily` is expected by design, not a bug.** It is hard-coded `None` in `_job_to_dict` (`cron_dashboard_api.py:173`) and is only populated for main_apscheduler jobs via an event listener — which has not been wired for the main scheduler. (Source: `backend/api/cron_dashboard_api.py:173` + phase-23.3.1 audit findings)

7. **Phase-23.3.1 already verified this job as live and healthy.** Audit confirmed `next_run: 2026-05-08T14:00:00-04:00`, `status: scheduled`, `description: Paper trading daily run` — exactly matching the current API state reported by Main. (Source: `handoff/archive/phase-23.3.1/phase-23.3.1-audit-findings.md`)

8. **Phase-23.3.6 consolidation did not contradict the healthy state.** Phase-23.3.6 archive contains only `contract.md`, `evaluator_critique.md`, `experiment_results.md` — no new findings about `paper_trading_daily`. (Source: `handoff/archive/phase-23.3.6/` listing)

## Internal code inventory

| File | Lines / Anchors | Role | Status |
|------|-----------------|------|--------|
| `backend/api/paper_trading.py` | 35 (`_scheduler_job_id`), 911-923 (`_add_scheduler_job`), 927-934 (`_scheduled_run`) | Registers the APScheduler job; cron trigger; no event listener wired | Active, authoritative |
| `backend/api/cron_dashboard_api.py` | 41-47 (`register_scheduler`), 160-176 (`_job_to_dict`), 173 (hardcoded `last_run: None`), 174 (status derivation), 194-218 (`get_all_jobs`) | Builds `/api/jobs/all` response; derives `status` from `next_run_time` | Active, authoritative |
| `backend/main.py` | 26-28 (imports), 172 (`init_scheduler`), 175 (`_register_cron_scheduler("main", ...)`) | Wires scheduler into lifespan; registers with cron dashboard registry | Active |
| `backend/slack_bot/scheduler.py` | 12-14 (imports), 122-124 (`add_listener`) | Reference: only process with EVENT_JOB_EXECUTED wired | Separate process; not relevant to main_apscheduler |
| `tests/verify_phase_23_3_1.py` | 37-43 (paper_trading check), 46-76 (live HTTP probe), 60-63 (id + description asserts) | Prior-art verifier for this job; checks description label only, NOT status/next_run | Active; complementary to 23.5.1 criterion |
| `handoff/archive/phase-23.3.1/phase-23.3.1-audit-findings.md` | Full file | Phase-23.3.1 audit confirming paper_trading_daily live and scheduled | Historical; confirms healthy baseline |

## Consensus vs debate (external)

**Consensus:** APScheduler 3.x does not natively expose `last_run_time`; manual tracking via `EVENT_JOB_EXECUTED` is the universally documented pattern. `status="scheduled"` + non-null `next_run_time` is a standard readiness indicator for a job registered and waiting to fire; it is the established pattern in FastAPI/APScheduler integrations. No debate in authoritative sources.

**Open question (not a debate, an architecture decision):** whether `last_run: null` constitutes a SLO violation depends on whether a freshness SLO has been defined. The masterplan criterion for this step does NOT require `last_run` population — only `status != "manifest"` and `next_run is not None`.

## Pitfalls (from literature)

- **DST boundary behavior:** CronTrigger uses "wall clock" time; ET timezone cron fires can shift by 1 hour on DST transitions. `paper_trading_daily` uses `ZoneInfo("America/New_York")` correctly. (Source: APScheduler 3.x CronTrigger doc)
- **APScheduler 4.x migration confusion:** 4.x renames `BackgroundScheduler` and changes listener semantics. pyfinagent uses 3.x; do not apply 4.x patterns to this codebase without a migration step.
- **`scheduled_run_time` vs wall-clock time in `JobExecutionEvent`:** The listener receives `scheduled_run_time` (the intended fire time), not the actual completion time. Relevant only if a last_run listener is added later — must use `datetime.now()` in the listener for actual completion timestamp.

## Application to pyfinagent (mapping findings to file:line anchors)

**Is the immutable criterion sufficient?**

The criterion (`status != "manifest" AND next_run is not None`) maps exactly to the internal logic at `cron_dashboard_api.py:174`:
```python
"status": "scheduled" if nrt is not None else "paused"
```
`status="scheduled"` iff `next_run_time is not None`. `status="manifest"` is only produced for static (out-of-process) entries via `_static_to_dict` (`cron_dashboard_api.py:179-188`). A `main_apscheduler` job can never have `status="manifest"` — it is either "scheduled" or "paused".

**Conclusion:** `status != "manifest" AND next_run is not None` is a COMPLETE liveness signal for `paper_trading_daily` as defined by this codebase's derivation logic. It confirms:
1. The job is registered in the live APScheduler instance (not a static manifest entry)
2. APScheduler has computed a valid next fire time (trigger is not paused or exhausted)

**What `last_run: null` means:** By design. No `EVENT_JOB_EXECUTED` listener is wired on the main scheduler. `last_run` is hard-coded `None` for all `main_apscheduler` jobs. This is a known gap documented in the code comment (`cron_dashboard_api.py:173`) with a "phase-2 if needed" tag. It is NOT a regression, NOT a bug, and NOT required by the step-23.5.1 criterion. If Main needs to verify that the job fires correctly, the path is: add an `EVENT_JOB_EXECUTED` listener to the main scheduler in `backend/api/paper_trading.py` (or `backend/main.py` post-`init_scheduler`), persist `event.scheduled_run_time` to an in-memory dict keyed by `job_id`, and surface it in `_job_to_dict` — but this is a future enhancement, not a blocker for 23.5.1.

**Phase-23.3.6 consolidation:** No contradictions. The 23.3.6 archive holds only harness lifecycle files; no new findings about `paper_trading_daily`.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5 official APScheduler docs pages)
- [x] 10+ unique URLs total (incl. snippet-only) — 15 unique URLs collected
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set — all 5 are documentation pages read in full
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (paper_trading.py, cron_dashboard_api.py, main.py, slack_bot/scheduler.py, tests/verify_phase_23_3_1.py, phase-23.3.1 audit, phase-23.3.6 archive)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

## Queries run (three-variant discipline)

1. **Current-year frontier:** "APScheduler BackgroundScheduler cron trigger next_run_time semantics 2026"
2. **Last-2-year window:** "APScheduler EVENT_JOB_EXECUTED listener FastAPI health check production 2025", "airflow prefect dagster manifest vs scheduled cron job status dashboard 2025"
3. **Year-less canonical:** "apscheduler cron job health check status scheduled live production"

---

## Recommendation to Main

**The immutable criterion is sufficient for phase-23.5.1.**

`status != "manifest" AND next_run is not None` fully captures liveness for a `main_apscheduler` job because:
- `status="manifest"` is structurally impossible for APScheduler-registered jobs — only static manifest entries produce that value.
- `next_run is not None` maps directly to APScheduler's `next_run_time is not None`, which means the trigger has computed a valid future fire time and the job is active (not paused, not removed).
- The live API response already satisfies both conditions: `status="scheduled"`, `next_run="2026-05-08T14:00:00-04:00"`.

**`last_run: null` does NOT require remediation for this step.** It is an acknowledged design gap (hard-coded `None` at `cron_dashboard_api.py:173`) with no event listener wired on the main scheduler. Validating that `last_run` populates after a fire would require waiting until Friday 14:00 ET and re-probing, or wiring an `EVENT_JOB_EXECUTED` listener — neither of which is in scope for a verification step. The masterplan criterion does not require `last_run` population, and Main must not amend that criterion.

**Verification command should PASS against current state** without any code changes.

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 10,
  "urls_collected": 15,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "gate_passed": true
}
```
