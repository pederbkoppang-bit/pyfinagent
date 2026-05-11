---
step: phase-23.6.2
title: Cosmetic schedule labels in _SLACK_BOT_JOBS + autoresearch description refresh — experiment results
cycle_date: 2026-05-10
verification: 'python3 tests/verify_phase_23_6_2.py'
contract: handoff/current/contract.md
---

# Experiment results — phase-23.6.2

## What was built / changed

### 1. `backend/api/cron_dashboard_api.py:72-95` — `_SLACK_BOT_JOBS` schedule labels

Replaced 11 placeholder schedule strings (`"phase-9.X cron"` /
`"phase-9.X interval"` / `"morning_digest_hour:00 ET"` /
`"evening_digest_hour:00 ET"` / `"watchdog_interval_minutes interval"`)
with APScheduler bracket notation matching what `_trigger_str()` emits
for live `main_apscheduler` rows on the same page. Settings-driven
values (morning_digest, evening_digest, watchdog) carry inline
`# configurable via X` comments per researcher recommendation.

| Job id | New schedule string |
|---|---|
| morning_digest | `cron[hour='8', minute='0']` |
| evening_digest | `cron[hour='17', minute='0']` |
| watchdog_health_check | `interval[0:15:00]` |
| prompt_leak_redteam | `cron[hour='3', minute='15']` |
| daily_price_refresh | `cron[hour='1']` |
| weekly_fred_refresh | `cron[day_of_week='sun', hour='2']` |
| nightly_mda_retrain | `cron[hour='3']` |
| hourly_signal_warmup | `cron[minute='5']` |
| nightly_outcome_rebuild | `cron[hour='4']` |
| weekly_data_integrity | `cron[day_of_week='mon', hour='5']` |
| cost_budget_watcher | `cron[hour='6']` |

All times America/New_York (the timezone applied by `start_scheduler`
in `backend/slack_bot/scheduler.py`).

### 2. `backend/api/cron_dashboard_api.py:113` — autoresearch description

Replaced stale `"Nightly autoresearch memo (FAILING exit 127 since
2026-04-24 -- see phase-23.3.4 audit)"` with current-state
`"Nightly autoresearch memo (exit 1 -- partial .env fix applied;
python entrypoint still failing -- see phase-23.5.19)"`. Reflects
the partial-env-fix progress documented in phase-23.5.19.

### 3. `tests/verify_phase_23_6_2.py` — 6-check verifier (NEW, ~210 LOC)

Imports `_SLACK_BOT_JOBS` + `_LAUNCHD_JOBS` via venv-Python
subprocess (system Python lacks httpx). Checks:

1. No placeholder tokens in any of the 11 entries.
2. Each schedule string exactly matches recommended replacement.
3. All 11 use `cron[...]` or `interval[...]` bracket prefix.
4. autoresearch description: removed "FAILING exit 127", contains
   "exit 1" AND "phase-23.5.19".
5. Live `/api/jobs/all` reflects the edits (hits localhost:8000).
6. All 27 prior phase-23 verifiers (23.5.* + 23.6.0 + 23.6.1)
   exit 0.

## Files changed

| Path | Change |
|---|---|
| `backend/api/cron_dashboard_api.py` | 11 schedule strings + 1 description string rewritten |
| `tests/verify_phase_23_6_2.py` | NEW — 6-check verifier |

No code-flow changes. Pure label/description string substitutions.

## Verification command + verbatim output

Per the contract, the immutable verification is:

```
python3 tests/verify_phase_23_6_2.py
```

Run after backend restart (`launchctl kickstart -k`) and slack-bot
restart so the live API reflects the edits and registry is fresh:

```
=== phase-23.6.2 verifier ===
  [PASS] no placeholder tokens: none of the 11 slack_bot entries contain placeholder tokens
  [PASS] schedules exact match recommended: all 11 schedules match recommended replacements
  [PASS] bracket notation used: all 11 schedules use cron[...] or interval[...] format
  [PASS] autoresearch description updated: autoresearch description updated to current state
  [PASS] live API reflects edits: live API reflects new schedule strings + new description
  [PASS] 27 sibling verifiers green: 27 sibling verifiers all exit 0

PASS (6/6)
EXIT=0
```

All 6 immutable checks PASS. Sibling sweep (27 verifiers) green —
no regression.

## Artifact shape — live `/api/jobs/all` (sample)

```json
{
  "id": "daily_price_refresh",
  "schedule": "cron[hour='1']",
  "description": "Daily refresh of universe price snapshots",
  "source": "slack_bot",
  ...
}
{
  "id": "com.pyfinagent.autoresearch",
  "description": "Nightly autoresearch memo (exit 1 -- partial .env fix applied; python entrypoint still failing -- see phase-23.5.19)",
  ...
}
```

The cron dashboard now renders consistent bracket-notation strings
across all 11 slack_bot static rows, matching the format already
shown for live `main_apscheduler` rows on the same page.

## Anti-patterns avoided

- Did NOT change any actual cron triggers (only LABEL strings).
- Did NOT add `cron-descriptor` / `cRonstrue` deps.
- Did NOT touch the 5 launchd schedule strings (only autoresearch
  description).
- Did NOT self-evaluate — Q/A still owed.

## Out-of-scope items left untouched

- Settings-driven label drift (operator changes
  `morning_digest_hour=9`): documented as low-priority follow-up
  in contract Risk section. Inline comment alerts the next reader.
- StartCalendarInterval next-fire-time computation — reserved for
  phase-23.6.3.

## Status

GENERATE complete. Q/A is mandatory next per harness protocol.
