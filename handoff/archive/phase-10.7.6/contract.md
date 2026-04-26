---
step: phase-10.7.6
title: Weekly APScheduler wiring for the meta-evolution loop
cycle_date: 2026-04-26
harness_required: true
verification: python -m pytest tests/scheduler/test_meta_cron.py -v
research_brief: handoff/current/phase-10.7.6-research-brief.md
---

# Contract -- phase-10.7.6

## Step ID

`phase-10.7.6` -- "Weekly APScheduler wiring" (`.claude/masterplan.json:3259-3267`).

## Research-gate summary

Spawned `researcher` (moderate tier). Brief at
`handoff/current/phase-10.7.6-research-brief.md`. Gate: 6 sources read in
full via WebFetch, 13 unique URLs, recency scan (2024-2026) performed,
11 internal files inspected. `gate_passed: true`.

Key external findings:
- APScheduler 3.x `CronTrigger(day_of_week="sun", hour=2, minute=0, timezone=ZoneInfo("America/New_York"))` is the canonical Sunday-2am-ET form
- `replace_existing=True` + explicit `id=` are mandatory for idempotent registration on restart
- `BackgroundScheduler` (sync) is correct for sync job functions; `AsyncIOScheduler` only for async
- Test pattern: `StubScheduler` with `add_job` recording (mirrors `tests/slack_bot/test_scheduler_phase9.py:14-21`)
- 3.x trigger introspection uses `get_next_fire_time(None, now)` (4.x uses `trigger.next()` -- DO NOT use 4.x API)
- Google SRE: meta-evolution is monitoring-tier -> fail-open per sub-call

Key internal anchors:
- Pattern source: `backend/autoresearch/cron.py:17-41` (separate cron module shim)
- Test pattern: `tests/slack_bot/test_scheduler_phase9.py:14-21` (StubScheduler)
- Existing register signature: `backend/slack_bot/scheduler.py:351` (`register_phase9_jobs(scheduler, replace_existing=True)`)
- Slot 14 in `.claude/cron_budget.yaml` already named `meta_evolution_weekly_reallocation`
- `tests/scheduler/` directory does NOT exist; must create with `__init__.py`

## Hypothesis

A new `backend/meta_evolution/cron.py` module with two top-level
functions -- `register_meta_evolution_cron(scheduler, *, replace_existing=True)`
and `run_meta_evolution_cycle()` -- can wire the weekly meta-evolution
reallocation onto any APScheduler-shaped object (BackgroundScheduler,
AsyncIOScheduler, StubScheduler in tests) without coupling
meta_evolution to any single scheduler instance. Fail-open per sub-call
keeps a single sub-failure from breaking the whole cycle.

## Immutable success criteria (verbatim from masterplan)

```
verification: python -m pytest tests/scheduler/test_meta_cron.py -v
```

The test module exists at `tests/scheduler/test_meta_cron.py` and all
tests pass.

## Plan steps

1. Create `backend/meta_evolution/cron.py` (~120-150 LOC):
   - Module-level constants: `_JOB_ID = "meta_evolution_weekly"`,
     `_CRON_BUDGET_YAML`, `_PROVIDER_BUDGET_YAML`,
     `_TIMEZONE = ZoneInfo("America/New_York")`
   - `register_meta_evolution_cron(scheduler, *, replace_existing=True, day_of_week="sun", hour=2, minute=0) -> str | None`
     -- adds the job with `trigger="cron"`, returns job_id on success or None on fail-open
   - `run_meta_evolution_cycle(*, cron_budget_yaml=None, provider_budget_yaml=None, bq_client=None, now=None) -> dict[str, Any]`
     -- executes one cycle; each of (cron_allocator, provider_rebalancer,
     archetype_library lookup) wrapped in individual try/except with
     warning-log fail-open
   - ASCII-only logger messages per `.claude/rules/security.md`

2. Create `tests/scheduler/__init__.py` (empty marker file).

3. Create `tests/scheduler/test_meta_cron.py` (~150-200 LOC, 8 tests):
   1. `test_register_adds_job_to_scheduler`
   2. `test_job_id_is_meta_evolution_weekly`
   3. `test_register_passes_replace_existing_true`
   4. `test_trigger_fires_sunday_2am_et` (instantiate `CronTrigger` directly + `get_next_fire_time(None, monday_reference)`)
   5. `test_timezone_is_explicitly_new_york`
   6. `test_run_cycle_calls_cron_allocator` (monkeypatch spy)
   7. `test_run_cycle_calls_provider_rebalancer` (monkeypatch spy)
   8. `test_run_cycle_handles_sub_failures_fail_open` (raise inside one sub-call; cycle still returns dict)

4. Verify: `python -m pytest tests/scheduler/test_meta_cron.py -v`

## References

- `.claude/masterplan.json:3259-3267` -- step entry
- `handoff/current/phase-10.7.6-research-brief.md` -- research gate
- `backend/autoresearch/cron.py:17-41` -- structural pattern
- `backend/slack_bot/scheduler.py:340-382` -- register pattern + fail-open
- `tests/slack_bot/test_scheduler_phase9.py:14-52` -- StubScheduler pattern
- `backend/meta_evolution/{cron_allocator,provider_rebalancer,archetype_library,alpha_velocity}.py` -- sub-modules
- `.claude/cron_budget.yaml` slot 14 -- canonical Sunday slot
- APScheduler 3.x CronTrigger docs: https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html
- APScheduler userguide (replace_existing): https://apscheduler.readthedocs.io/en/3.x/userguide.html
- Google SRE distributed periodic scheduling: https://sre.google/sre-book/distributed-periodic-scheduling/

## Out of scope

- Wiring `register_meta_evolution_cron(scheduler)` into the live
  `backend/slack_bot/scheduler.py` `start_scheduler()` function. That
  is a one-line follow-up; this cycle ships the module + tests.
- BQ table migration for any new telemetry. Not introduced this cycle.
- LLM-cost telemetry on cycle execution.
- A real APScheduler dry-run (the immutable verification command does
  not require it; StubScheduler proves wiring).
