---
step: phase-23.3.3
title: Slack-bot phase-9 jobs audit -- activate the 7 dormant jobs
cycle_date: 2026-05-07
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_3_3.py'
research_brief: handoff/current/phase-23.3.3-external-research.md (also see phase-23.3.3-internal-codebase-audit.md)
---

# Contract — phase-23.3.3

## Hypothesis

`register_phase9_jobs(scheduler, replace_existing=True)` is defined at
`backend/slack_bot/scheduler.py:397-428` and maps 7 phase-9 job ids
(daily_price_refresh, weekly_fred_refresh, nightly_mda_retrain,
hourly_signal_warmup, nightly_outcome_rebuild, weekly_data_integrity,
cost_budget_watcher) to their module paths + cron triggers. **The
function has zero callsites in the codebase** -- the 7 jobs have been
dormant since the file was added.

The runbook `docs/runbooks/phase9-cron-runbook.md:23` explicitly
says: "Called once at Slack bot process startup AFTER `start_scheduler`
has set up the morning/evening digest + watchdog jobs." Dormancy is
an oversight, not intentional.

Researcher (a40a5015c28ebd163) confirmed all 7 modules default to
stub fetchers (no real yfinance, no FRED API, no BQ writes) -- the
blast radius of activation is near-zero. Recommended option (a): wire
the call in `start_scheduler()` AND add `misfire_grace_time` +
`coalesce=True` to each job's kwargs to prevent stale-tick fires on
restart and rapid-succession execution.

## Research-gate summary

Researcher (a40a5015c28ebd163) returned `gate_passed: true`:
- 7 sources read in full (APScheduler 3.x User Guide + Job module +
  Stable User Guide, Google SRE Workbook canarying, Google SRE Book
  product launches, Unleash kill-switch, anti-pattern cron blog)
- 17 URLs collected; 10 in snippet-only
- Recency scan 2024-2026 — APScheduler 3.x stable since 2020
- 10 internal files inspected (each phase-9 job module + runbook)
- Concrete recommendation: option (a) with parameter guards

Key findings cited:
- `register_phase9_jobs` never called (`grep` zero hits outside def).
- All 7 modules default to stubs -- safe to activate.
- Phase-23.3.2's APScheduler listener already covers heartbeats for
  these jobs once registered (no per-job HTTP push needed).

## Immutable success criteria (verbatim — DO NOT EDIT)

1. `backend/slack_bot/scheduler.py::start_scheduler` invokes
   `register_phase9_jobs(_scheduler)` after `_scheduler.start()` (or
   safely before — both work). The 7 phase-9 jobs are registered
   alongside the 4 core jobs whenever the slack-bot starts.
2. The mapping in `register_phase9_jobs` (currently at lines 406-413)
   is updated so every job's kwargs include:
   - `misfire_grace_time` per researcher's recommendation:
     - 3600 for daily jobs (daily_price_refresh, nightly_mda_retrain,
       nightly_outcome_rebuild, cost_budget_watcher)
     - 7200 for weekly jobs (weekly_fred_refresh, weekly_data_integrity)
     - 600 for hourly (hourly_signal_warmup)
   - `coalesce=True` for all 7
3. The audit deliverable `handoff/current/phase-23.3.3-audit-findings.md`
   documents: (a) the dormancy was an oversight per the runbook, (b)
   each of the 7 modules verified as a stub-default-safe activation,
   (c) the parameter guards added, (d) the operator-restart caveat
   from phase-23.3.2 still applies (slack-bot daemon must be
   restarted to pick up the new registrations).
4. Regression test `tests/services/test_phase9_registration.py`
   asserts:
   - `register_phase9_jobs` is called from `start_scheduler` source
     (string match in source).
   - Calling `register_phase9_jobs` against a fake scheduler returns
     all 7 ids (subject to module imports succeeding) AND the kwargs
     passed to each `add_job` include `misfire_grace_time` + `coalesce`.
5. `python tests/verify_phase_23_3_3.py` exits 0.
6. `python -c "import ast; ast.parse(...)"` passes for the modified
   file.

## Plan steps

1. Edit `backend/slack_bot/scheduler.py`:
   - In the mapping at lines 406-413, add `misfire_grace_time` +
     `coalesce` per job tier (daily/weekly/hourly).
   - In `start_scheduler` after `_scheduler.start()` add
     `register_phase9_jobs(_scheduler)`. Wrap in try/except (fail-open)
     so a bad import in any single phase-9 module doesn't break the
     core 4 jobs.
   - The `add_job` call at line 424 needs to read the new kwargs out
     of the mapping. Update the function body to pass them through.
2. Add `tests/services/test_phase9_registration.py` (3+ tests).
3. Add `tests/verify_phase_23_3_3.py` (deterministic checks).
4. Write `handoff/current/phase-23.3.3-audit-findings.md`.
5. Append `harness_log.md` AFTER PASS.

## Operator-restart caveat (load-bearing)

Same as phase-23.3.2: the slack-bot daemon (PID 16385, running since
2026-04-08) won't pick up the new registrations until restarted.
Audit findings doc names this prominently; verifier validates the
wiring statically.

## Out of scope

- Wiring real fetch/write implementations (yfinance for
  daily_price_refresh, fredapi for weekly_fred_refresh, BQ writes).
  Stub defaults are safe; per-job upgrades are separate phases.
- Wiring `alert_fn` for `weekly_data_integrity` (drift currently
  computed but alerts silently dropped). P2 follow-up.
- Operator restart of the slack-bot daemon (operator job).
- The `SLACK_CHANNEL_ID` silent-skip guard (deferred from
  phase-23.3.2).

## Backwards compatibility

- `register_phase9_jobs` is fail-open per job (existing behavior).
- New kwargs are additive; APScheduler accepts them on `add_job` since
  3.x.
- Wrapping the call in try/except in `start_scheduler` means a bad
  phase-9 module can't break the 4 core jobs.
- No behavior change for the 4 core jobs.

## References

- Researcher: `handoff/current/phase-23.3.3-{external-research,internal-codebase-audit}.md`
- `backend/slack_bot/scheduler.py:397-428` (function definition + mapping)
- `docs/runbooks/phase9-cron-runbook.md:23` (call expectation)
- APScheduler 3.x User Guide: misfire_grace_time, coalesce semantics
- Google SRE: feature-flag canarying (rejected -- blast radius too low)
