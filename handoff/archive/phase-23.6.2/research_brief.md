---
step_id: phase-23.6.2
step_name: "Cosmetic fixes: real cron labels in _SLACK_BOT_JOBS + update stale autoresearch description"
tier: simple
gate_passed: true
---

# Research Brief: phase-23.6.2

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://apscheduler.readthedocs.io/en/3.x/userguide.html | 2026-05-10 | doc | WebFetch | APScheduler CronTrigger and IntervalTrigger configuration; guide documents print_jobs() for listings but does not state str() format -- confirmed empirically (see internal section) |
| https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html | 2026-05-10 | doc | WebFetch | CronTrigger constructor params and from_crontab(); bracket notation not documented but empirically confirmed as `cron[hour='1']` etc. |
| https://bradymholt.github.io/cRonstrue/ | 2026-05-10 | blog/tool | WebFetch | `*/5 * * * *` -> "Every 5 minutes"; `0 23 ? * MON-FRI` -> "At 11:00 PM, Monday through Friday"; output pattern: "At HH:MM AM/PM [, day range]" |
| https://github.com/bradymholt/cRonstrue | 2026-05-10 | code | WebFetch | cRonstrue v3.14.0 (March 2026). Default: 12-hour time ("At 11:00 PM, ..."); use24HourTimeFormat option available. Verbose mode adds "every day". |
| https://betterstack.com/community/guides/scaling-python/apscheduler-scheduled-tasks/ | 2026-05-10 | blog | WebFetch | APScheduler CronTrigger configured with day_of_week, hour, minute params; IntervalTrigger with minutes= param. Confirms API shape but not str() format. |
| https://docs.dagster.io/guides/automate/schedules | 2026-05-10 | doc | WebFetch | Dagster uses `name="daily_refresh"` + `cron_schedule="0 0 * * *"` in ScheduleDefinition; UI displays schedule name as the primary label alongside cron expression. No standardized human-readable label convention enforced. |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://github.com/Salamek/cron-descriptor | code | WebFetch returned no output examples; library noted for reference |
| https://pypi.org/project/cron-descriptor/ | doc | Fetched; no example output strings documented on PyPI page |
| https://github.com/cuu508/cronsim | code | Search snippet only; cronsim v2.4+ has explain() method |
| https://linuxconfig.org/how-to-fix-bash-127-error-return-code | blog | Search snippet; exit 127 = command not found (canonical POSIX) |
| https://apscheduler.readthedocs.io/en/stable/modules/triggers/interval.html | doc | Search snippet; IntervalTrigger documented as `interval[H:MM:SS]` format |
| https://github.com/agronholm/apscheduler | code | Search snippet; APScheduler source confirms bracket format |
| https://pypi.org/project/croniter/ | doc | Search snippet; croniter iterates datetime objects, not a formatting library |
| https://hightouch.com/blog/airflow-alternatives-a-look-at-prefect-and-dagster | blog | Search snippet; no label convention specifics |

## Recency scan (2024-2026)

Searched for:
1. "cron expression human readable format dashboard labels 2026"
2. "APScheduler trigger string representation CronTrigger IntervalTrigger format 2025 2026"
3. "cron expression human readable string library Python croniter cronsim 2025"

Result: cRonstrue v3.14.0 released March 2026 (added option to trim leading zeros in hours description). cronsim v2.4+ added explain() method. No new 2024-2026 work supersedes the APScheduler bracket-notation format -- it is stable since APScheduler 3.x and unchanged in 3.11.2 (current). The bracket notation `cron[hour='1']` / `interval[0:05:00]` is the APScheduler-native format for `str(trigger)`.

## Key findings

1. **APScheduler bracket notation IS the canonical format** -- empirically confirmed via `.venv` run: `str(CronTrigger(hour=1))` returns `cron[hour='1']`; `str(IntervalTrigger(minutes=15))` returns `interval[0:15:00]`. This matches what `_job_to_dict()` at `cron_dashboard_api.py:181` calls via `_trigger_str(getattr(job, "trigger", None))` -- i.e., the `main_apscheduler` block already emits this format. (Source: empirical, APScheduler 3.x docs)

2. **All 11 job triggers confirmed** -- See empirical results table below. The three configurable jobs (morning_digest, evening_digest, watchdog) use settings defaults (hour=8, hour=17, minutes=15); the 4 core fixed jobs and 7 phase-9 jobs have hard-coded triggers.

3. **autoresearch exit code transitioned 127 -> 1** -- Phase-23.3.4 audit (2026-04-24) observed exit 127 caused by malformed `KEY= value` lines in `backend/.env` (bash `set -a; . .env` parsed as command invocation). Phase-23.5.19 (2026-05-10) confirmed live `launchctl print` shows `last exit code = 1`, `runs=4`. This indicates partial .env remediation (lines 24/25 fixed; line 56 or python entrypoint still fails). (Source: `handoff/archive/phase-23.5.19/research_brief.md:50`, `experiment_results.md:53-63`)

4. **"FAILING exit 127" description is double-stale** -- exit code AND mechanism are both outdated. The job still fails (status="failed" in /api/jobs/all) but with exit 1 from the python entrypoint, not exit 127 from bash. (Source: `handoff/archive/phase-23.5.19/experiment_results.md:30-36`)

5. **cRonstrue format is verbose natural language** -- produces strings like "At 1:00 AM" (daily) or "At 2:00 AM, on Sunday". This is NOT what `_job_to_dict()` emits and would be inconsistent with the main_apscheduler source. (Source: cRonstrue GitHub WebFetch)

## Internal code inventory

| File | Lines inspected | Role | Status |
|------|----------------|------|--------|
| `backend/api/cron_dashboard_api.py` | 55-199 | _SLACK_BOT_JOBS tuple (lines 72-95), _LAUNCHD_JOBS (101-114), _job_to_dict (170-198), _trigger_str (162-167) | Read in full for relevant range |
| `backend/slack_bot/scheduler.py` | 127-227, 505-610 | 4 core job registration + register_phase9_jobs mapping | Read in full for relevant ranges |
| `backend/config/settings.py` | grep hits | morning_digest_hour=8, evening_digest_hour=17, watchdog_interval_minutes=15 defaults | Confirmed via grep |
| `handoff/archive/phase-23.5.19/research_brief.md` | all | Prior researcher findings on autoresearch exit code transition | Read; authoritative |
| `handoff/archive/phase-23.5.19/experiment_results.md` | 1-80 | Empirical launchctl output; status="failed", exit code=1, runs=4 | Read |

## Empirical APScheduler str() results (from venv run)

All 11 jobs:
```
morning_digest:          cron[hour='8', minute='0']      (settings default hour=8)
evening_digest:          cron[hour='17', minute='0']     (settings default hour=17)
watchdog_health_check:   interval[0:15:00]               (settings default 15 min)
prompt_leak_redteam:     cron[hour='3', minute='15']
daily_price_refresh:     cron[hour='1']
weekly_fred_refresh:     cron[day_of_week='sun', hour='2']
nightly_mda_retrain:     cron[hour='3']
hourly_signal_warmup:    cron[minute='5']
nightly_outcome_rebuild: cron[hour='4']
weekly_data_integrity:   cron[day_of_week='mon', hour='5']
cost_budget_watcher:     cron[hour='6']
```

Note: morning_digest/evening_digest/watchdog are configurable; bracket notation reflects defaults. The static manifest in `_SLACK_BOT_JOBS` should encode the defaults explicitly (they cannot call `settings` at module import time without creating a settings dependency in the manifest).

## _SLACK_BOT_JOBS current state (verbatim, lines 72-95)

```python
_SLACK_BOT_JOBS: tuple[dict[str, str], ...] = (
    {"id": "morning_digest",        "schedule": "cron daily morning_digest_hour:00 ET",
     "description": "Slack morning digest (top movers + holdings recap)"},
    {"id": "evening_digest",        "schedule": "cron daily evening_digest_hour:00 ET",
     "description": "Slack evening digest (P&L + closed trades)"},
    {"id": "watchdog_health_check", "schedule": "interval watchdog_interval_minutes",
     "description": "Slack-bot self-watchdog (alerts on backend unreachability)"},
    {"id": "prompt_leak_redteam",   "schedule": "cron daily 03:15 ET",
     "description": "Nightly red-team prompt-leak audit"},
    {"id": "daily_price_refresh",      "schedule": "phase-9.2 cron",
     "description": "Daily refresh of universe price snapshots"},
    {"id": "weekly_fred_refresh",      "schedule": "phase-9.3 cron",
     "description": "Weekly refresh of FRED macro series"},
    {"id": "nightly_mda_retrain",      "schedule": "phase-9.4 cron",
     "description": "Nightly MDA feature-importance retrain"},
    {"id": "hourly_signal_warmup",     "schedule": "phase-9.5 interval",
     "description": "Hourly cache warmup for enrichment signals"},
    {"id": "nightly_outcome_rebuild",  "schedule": "phase-9.6 cron",
     "description": "Nightly outcome-tracking refresh"},
    {"id": "weekly_data_integrity",    "schedule": "phase-9.7 cron",
     "description": "Weekly BQ data-integrity audit"},
    {"id": "cost_budget_watcher",      "schedule": "phase-9.8 interval",
     "description": "Cost-budget watcher + soft-cap alerts"},
)
```

## _LAUNCHD_JOBS autoresearch row (line 112-113, verbatim)

```python
{"id": "com.pyfinagent.autoresearch",     "schedule": "launchd cron 02:00 daily",
 "description": "Nightly autoresearch memo (FAILING exit 127 since 2026-04-24 -- see phase-23.3.4 audit)"},
```

## What _job_to_dict() emits (comparison shape, lines 170-186)

`_job_to_dict()` for main_apscheduler jobs calls `_trigger_str(job.trigger)` which is `str(trigger)`. This yields the bracket notation: `cron[hour='1']`, `interval[0:15:00]` etc. The static manifests (slack_bot, launchd) use the `"schedule"` key directly -- whatever string is in the manifest appears verbatim in the operator dashboard. Consistency with the main_apscheduler format means using the APScheduler bracket notation.

## Consensus vs debate

Consensus: APScheduler bracket notation (`cron[...]` / `interval[...]`) is the native format used by the existing `_job_to_dict()` pipeline and is the lowest-friction consistent label for the static manifests.

Debate: cRonstrue-style natural language ("At 1:00 AM daily") is more human-friendly but introduces a second format that diverges from what the live `main_apscheduler` block emits. Since both sources appear on the same `/cron` dashboard page, consistency is the stronger argument.

## Pitfalls (from literature and code)

- **Settings-at-module-import-time**: `morning_digest_hour`, `evening_digest_hour`, and `watchdog_interval_minutes` are runtime settings. The static manifest cannot call `get_settings()` at module level without adding a settings dependency. The bracket notation with defaults must be hardcoded (or use a note indicating the default), not dynamically constructed.
- **Timezone annotation**: The actual jobs run in America/New_York. The bracket notation as emitted by APScheduler omits timezone in the string when it matches the system timezone. Safe to omit from static labels; the description field carries enough context.

## Application to pyfinagent

The static manifest `_SLACK_BOT_JOBS` (cron_dashboard_api.py:72-95) should replace placeholder strings with the APScheduler bracket notation that matches what the live scheduler emits. This makes the operator dashboard internally consistent: static manifest jobs look the same as live `main_apscheduler` jobs.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total (14 collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (cron_dashboard_api.py, scheduler.py, config/settings.py, phase-23.5.19 archive)
- [x] Contradictions / consensus noted (bracket notation vs natural language; exit 127 vs exit 1)
- [x] All claims cited per-claim

---

## THREE ANSWERS

### Answer 1: Schedule-label format recommendation

**Use APScheduler bracket notation** (`cron[...]` / `interval[...]`).

Rationale: `_job_to_dict()` at `cron_dashboard_api.py:181` calls `_trigger_str(job.trigger)` which is `str(trigger)` -- the live `main_apscheduler` block already emits this format. Matching it in the static manifests makes both source=main_apscheduler rows and source=slack_bot rows look identical on the dashboard. No new dependency required; no string-computation logic added.

cRonstrue-style ("At 1:00 AM daily") is ruled out: it would create a format split between live APScheduler rows and static-manifest rows on the same dashboard page.

Raw cron (`0 1 * * *`) is valid but APScheduler bracket notation is more readable AND consistent with what the live system already shows.

### Answer 2: Per-row replacement strings

```python
{
    "morning_digest":        "cron[hour='8', minute='0']",     # default; configurable via morning_digest_hour
    "evening_digest":        "cron[hour='17', minute='0']",    # default; configurable via evening_digest_hour
    "watchdog_health_check": "interval[0:15:00]",              # default; configurable via watchdog_interval_minutes
    "prompt_leak_redteam":   "cron[hour='3', minute='15']",
    "daily_price_refresh":   "cron[hour='1']",
    "weekly_fred_refresh":   "cron[day_of_week='sun', hour='2']",
    "nightly_mda_retrain":   "cron[hour='3']",
    "hourly_signal_warmup":  "cron[minute='5']",
    "nightly_outcome_rebuild": "cron[hour='4']",
    "weekly_data_integrity": "cron[day_of_week='mon', hour='5']",
    "cost_budget_watcher":   "cron[hour='6']",
}
```

All times are America/New_York (ET). The three configurable jobs encode the settings defaults; a comment noting this is recommended.

### Answer 3: autoresearch description update

Replace:
```
"Nightly autoresearch memo (FAILING exit 127 since 2026-04-24 -- see phase-23.3.4 audit)"
```

With:
```
"Nightly autoresearch memo (exit 1 -- partial .env fix applied; python entrypoint still failing -- see phase-23.5.19)"
```

Rationale: exit 127 was the original bash `set -a; . .env` abort caused by malformed `KEY= value` lines. As of phase-23.5.19 (2026-05-10), launchctl reports `last exit code = 1, runs=4` -- the bash stage now passes but the python entrypoint or a subsequent command fails. "Partial .env fix applied" accurately reflects that lines 24/25 were remediated but another error remains. The phase-23.5.19 reference gives the operator the correct audit trail.

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 8,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "gate_passed": true
}
```
