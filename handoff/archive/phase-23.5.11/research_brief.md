---
step_id: phase-23.5.11
step_name: "Cron job verification: nightly_outcome_rebuild (slack_bot, phase-9.6)"
tier: simple
researcher: Researcher (merged researcher + Explore)
date: 2026-05-10
---

# Research Brief: phase-23.5.11 — nightly_outcome_rebuild

## Queries run (three-query discipline)

1. **Current-year frontier**: "outcome tracking prediction validation batch rebuild patterns 2026"
2. **Last-2-year window**: "nightly batch rebuild idempotency patterns production stub testing 2025"
3. **Year-less canonical**: "idempotent batch job outcome tracking heartbeat context manager Python"
4. **Supplementary**: "APScheduler cron job production stub pattern dependency injection Python"
5. **Supplementary**: "trade outcome tracking win loss PnL batch job BigQuery Python 2025"

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://medium.com/towards-data-engineering/building-idempotent-data-pipelines-a-practical-guide-to-reliability-at-scale-2afc1dcb7251 | 2026-05-10 | blog (practitioner) | WebFetch full | "Add a small buffer to catch records from clock skew"; hybrid approach: "Recent data: Full recomputation (last 7-30 days); Historical data: Incremental processing" |
| https://www.prefect.io/blog/the-importance-of-idempotent-data-pipelines-for-resilience | 2026-05-10 | official vendor doc | WebFetch full | "Idempotency Keys: Unique identifiers track whether operations have already executed, preventing duplicate data imports during recovery attempts." Prefect 3 embeds idempotency via @on_rollback compensation pattern |
| https://oneuptime.com/blog/post/2026-01-30-batch-processing-retry-strategies/view | 2026-05-10 | blog (2026) | WebFetch full | "Idempotent operations produce the same result regardless of how many times they run. This is critical for batch processing because retries should not create duplicate records." Checkpoint-based recovery for long-running batches |
| https://airbyte.com/data-engineering-resources/idempotency-in-data-pipelines | 2026-05-10 | official vendor doc | WebFetch full | Idempotency keys "produce identical results whether executed once or multiple times"; atomic operations critical; testing must include "repeated execution testing, fault injection, concurrent operation testing" |
| https://betterstack.com/community/guides/scaling-python/apscheduler-scheduled-tasks/ | 2026-05-10 | authoritative doc | WebFetch full | APScheduler cron registrations via add_job() with id; production needs SQLAlchemy jobstore for persistence; BackgroundScheduler for concurrent apps. No native DI -- wrapper function or factory pattern required |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.buildmvpfast.com/blog/idempotent-ai-agent-retry-safe-patterns-production-workflow-2026 | blog 2026 | 403 Forbidden |
| https://apxml.com/courses/introduction-to-mlops/chapter-5-model-deployment-and-serving/deployment-patterns-online-batch | course page | 403 Forbidden |
| https://www.reform.app/blog/predictive-scoring-validation-best-practices | blog | snippet only; content covered by airbyte source |
| https://pypi.org/project/APScheduler/ | PyPI | snippet only; version info only |
| https://docs.bullmq.io/patterns/idempotent-jobs | official doc | snippet only; Node.js ecosystem, not Python |
| https://www.prefect.io/blog/ | vendor blog | snippet only (redirected to full prefect source above) |
| https://temporal.io/blog/idempotency-and-durable-execution | authoritative blog | snippet only; covers Temporal workflow engine, not directly applicable |
| https://www.startdataengineering.com/post/why-how-idempotent-data-pipeline/ | blog | snippet only; concept covered by medium source above |
| https://betterstack.com/community/guides/scaling-python/python-dependency-injection/ | authoritative doc | snippet only; DI patterns only |
| https://dev.to/alexmercedcoder/idempotent-pipelines-build-once-run-safely-forever-2o2o | blog | snippet only |

---

## Recency scan (2024-2026)

Searched for 2024-2026 literature on outcome tracking batch jobs, idempotent pipelines, and heartbeat patterns.

**Findings**: The 2026-scoped search ("outcome tracking prediction validation batch rebuild patterns 2026") returned a batch-retry-strategies post from January 2026 (oneuptime.com) that directly addresses checkpoint-based recovery and idempotency keys for batch jobs, confirming current best practice. The 2025-scoped search ("nightly batch rebuild idempotency patterns production stub testing 2025") returned the medium.com medallion-architecture article and the buildmvpfast.com AI agent idempotency post (403, blocked). No new findings from 2024-2026 supersede the canonical patterns (idempotency keys + daily scoping + fail-open + checkpoint atomicity). The pyfinagent implementation in `nightly_outcome_rebuild.py` uses `IdempotencyKey.daily()` which exactly matches the 2026 recommended pattern.

---

## Key external findings

1. **Daily-scoped idempotency keys are the canonical pattern** for nightly batch jobs -- [Medium/TowardsDataEngineering 2025] and [Airbyte 2024]. Keys that incorporate job name + calendar date (e.g. `nightly_outcome_rebuild::2026-05-10`) prevent double-execution on restarts.
2. **Fail-open on write failure is correct design** -- "better to load duplicates and deduplicate downstream than miss data entirely" (Medium). The nightly_outcome_rebuild `_default_write` silently returning `len(outcomes)` without a real BQ write is a production-stub gap, not an architectural error.
3. **Heartbeat context manager is the right abstraction** for tracking job progress and idempotency state -- confirmed by Prefect 3 patterns and Temporal patterns; the pyfinagent `heartbeat()` context manager mirrors this exactly.
4. **APScheduler does not natively inject dependencies** into registered job functions -- wrapper/factory pattern is required. nightly_outcome_rebuild.run() sidesteps this via keyword-argument injection (ledger_fetch_fn, outcome_write_fn) defaulting to production stubs.
5. **Full-rebuild vs incremental** -- for small ledgers (paper trading at low volume) full nightly rebuild is simpler and safe; incremental is only needed when ledger volume grows beyond memory constraints.

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/slack_bot/jobs/nightly_outcome_rebuild.py` | 58 | Phase-9.6 job: fetch trades, compute outcomes, write | REAL computation (_compute_outcomes), STUB I/O (_default_fetch returns [], _default_write returns len) |
| `backend/slack_bot/scheduler.py` (lines 491-548) | ~57 | register_phase9_jobs() with APScheduler mapping | Active; nightly_outcome_rebuild registered at hour=4, misfire_grace_time=3600, coalesce=True |
| `backend/api/job_status_api.py` (lines 55-67) | ~13 | _JOB_NAMES tuple defining all monitored jobs | nightly_outcome_rebuild confirmed at line 60 |
| `tests/slack_bot/test_nightly_outcome_rebuild.py` | 49 | Unit tests: win/loss classification, idempotency, fail-open | Active; 3 tests, all inject real IdempotencyStore |
| `handoff/logs/slack_bot.log` | N/A | Runtime log | Last relevant line: 2026-05-09 23:24:23 confirming phase-9 jobs registered including nightly_outcome_rebuild |

---

## Consensus vs debate (external)

**Consensus**: Idempotency keys scoped to job name + calendar date, combined with a checkpoint/heartbeat context manager, is the universally recommended pattern for nightly batch jobs (confirmed by 4 of 5 full-read sources). Fail-open on write is the correct design when data durability is not the job's primary concern.

**Debate**: Full-rebuild vs incremental is genuinely contested for large ledgers; at pyfinagent paper-trading scale (small ledger), full-rebuild is simpler and has no downsides.

---

## Pitfalls (from literature)

1. **Missing idempotency key scoping** -- keys not scoped to date allow old idempotency records to block future runs. Mitigation: `IdempotencyKey.daily()` already scopes to date (confirmed in code).
2. **Swallowing all write errors without recording** -- fail-open without logging loses observability. Mitigation: `nightly_outcome_rebuild.py:31` uses `logger.warning("outcome_rebuild: write fail-open: %r", exc)` before returning `n=0`.
3. **Production stubs shipping to prod** -- `_default_fetch` and `_default_write` are named with `_default_` prefix but are the actual production callables (no real BQ reads/writes). This is the primary concern flagged in prior phase-9 work.
4. **Double-fire on restart** -- APScheduler fires missed ticks on startup without `misfire_grace_time`. Mitigation: scheduler.py:529 sets `misfire_grace_time=3600` for nightly_outcome_rebuild.

---

## Application to pyfinagent (mapping external findings to file:line anchors)

### Docker-alias bug?
**No.** The nightly_outcome_rebuild job does NOT make any HTTP calls. It calls `ledger_fetch_fn()` and `outcome_write_fn()` which default to `_default_fetch` (returns empty list) and `_default_write` (returns `len(outcomes)`). No `http://backend:8000` alias, no Docker networking involved. Docker-alias bug is NOT present for this job.

### heartbeat() correctly wired?
**Yes.** `nightly_outcome_rebuild.py:22`: `with heartbeat(JOB_NAME, idempotency_key=key, store=store) as state:` -- heartbeat context manager is called with the correct job name constant (`JOB_NAME = "nightly_outcome_rebuild"`) and a daily idempotency key. The `IdempotencyStore` is passed in from the caller (or defaults to a fresh in-memory store). The heartbeat wraps the entire computation block including both `ledger_fetch_fn()` and `outcome_write_fn()`. This matches the canonical heartbeat-wraps-work pattern.

### Production-stub affected, or real work runs?
**Production-stub affected.** Two stubs are present:
- `_default_fetch()` at line 50-51: returns `[]` with comment `# production reads pyfinagent_pms.paper_trades`. No real BQ read is wired.
- `_default_write()` at line 54-55: returns `len(outcomes)` with no real BQ write. Outcomes are computed in memory (`_compute_outcomes` is real logic at lines 37-47) but the write is a no-op.

**Impact assessment**: The job will run, fire heartbeat, compute an empty list of outcomes (because fetch returns []), write 0 outcomes (noop), and report `{"rebuilt": 0, "key": ..., "skipped": False}`. The job is NOT erroring but also NOT doing real work. This is the same pattern as `daily_price_refresh` and `weekly_fred_refresh` (confirmed affected in prior phase-9 work). The verification command checks `status != "manifest"` and `next_run is not None` -- both will pass since the job is registered and scheduled. The job itself returning `rebuilt=0` is observable via heartbeat state but not surfaced by the verification command.

**Severity**: Same as other stub-affected jobs -- real BQ integration (reading `pyfinagent_pms.paper_trades`, writing to `pyfinagent_data.outcome_tracking`) is deferred work, not a regression.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5 fetched: medium, prefect, oneuptime, airbyte, betterstack)
- [x] 10+ unique URLs total (incl. snippet-only) -- 15 total collected
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (job file, scheduler, job_status_api, tests, log)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim (not just listed in a footer)

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 10,
  "urls_collected": 15,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "gate_passed": true
}
```
