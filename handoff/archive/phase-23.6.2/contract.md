---
step: phase-23.6.2
title: Cosmetic fixes — real cron labels in _SLACK_BOT_JOBS + update stale autoresearch description
cycle_date: 2026-05-10
harness_required: true
verification: 'python3 tests/verify_phase_23_6_2.py'
research_brief: handoff/current/phase-23.6.2-research-brief.md
---

# Contract — phase-23.6.2

## Hypothesis

Replacing the placeholder schedule labels in `_SLACK_BOT_JOBS`
(`cron_dashboard_api.py:62-85`) with APScheduler bracket notation
matching the format `_trigger_str()` already emits for live jobs
makes the `/cron` dashboard show consistent, informative schedules
across all 19 rows. Updating the autoresearch description string
in `_LAUNCHD_JOBS:103` from the stale "FAILING exit 127" to the
current "exit 1" state keeps operator messaging in sync with
reality.

Per researcher: APScheduler bracket notation (`cron[hour='1']`,
`interval[0:15:00]`) is what `_trigger_str()` at
`cron_dashboard_api.py:162-167` calls `str(trigger)` to produce
for `main_apscheduler` rows. Matching format in the static manifest
makes both live and static rows look identical on the same page.

## Research-gate summary

`researcher` agent `aa95ff717af6d530f` (re-spawn after
`af942a3c133df1dcd` stopped mid-task) ran tier=simple and returned
`gate_passed: true`:
- 6 external sources read in full (APScheduler 3.x userguide +
  CronTrigger module, cRonstrue demo + GitHub, BetterStack
  APScheduler, Dagster schedules).
- 8 snippet-only + 6 read-in-full = 14 URLs (≥10 floor).
- Recency scan 2024-2026.
- Three-query discipline.
- 5 internal files inspected, including phase-23.5.19 archive for
  the autoresearch exit-code transition.

Brief: `handoff/current/phase-23.6.2-research-brief.md`.

**Researcher's three answers:**

1. **Format:** APScheduler bracket notation (matches live
   `_trigger_str()` output; consistent with main_apscheduler block
   on the same page; ruled out cRonstrue natural language and raw
   cron because they'd create a format split).

2. **Per-row replacement strings** (empirically derived from
   `str(CronTrigger(...))` in the project venv, with
   settings defaults `morning_digest_hour=8`,
   `evening_digest_hour=17`, `watchdog_interval_minutes=15`):

   ```python
   {
       "morning_digest":          "cron[hour='8', minute='0']",
       "evening_digest":          "cron[hour='17', minute='0']",
       "watchdog_health_check":   "interval[0:15:00]",
       "prompt_leak_redteam":     "cron[hour='3', minute='15']",
       "daily_price_refresh":     "cron[hour='1']",
       "weekly_fred_refresh":     "cron[day_of_week='sun', hour='2']",
       "nightly_mda_retrain":     "cron[hour='3']",
       "hourly_signal_warmup":    "cron[minute='5']",
       "nightly_outcome_rebuild": "cron[hour='4']",
       "weekly_data_integrity":   "cron[day_of_week='mon', hour='5']",
       "cost_budget_watcher":     "cron[hour='6']",
   }
   ```

   All times America/New_York (ET). The 3 configurable jobs encode
   settings defaults; a Python comment notes "configurable via
   setting X" alongside them.

3. **autoresearch description** — replace
   `"Nightly autoresearch memo (FAILING exit 127 since 2026-04-24
   -- see phase-23.3.4 audit)"` with
   `"Nightly autoresearch memo (exit 1 -- partial .env fix
   applied; python entrypoint still failing -- see phase-23.5.19)"`.

## Immutable success criteria (verbatim — DO NOT EDIT)

```
python3 tests/verify_phase_23_6_2.py
```

The verifier exits 0 only when:

1. All 11 `_SLACK_BOT_JOBS` entries' `schedule` field matches the
   bracket-notation format (`cron[...]` or `interval[...]`).
2. None of the 11 entries' `schedule` field still contains the
   placeholder pattern (`phase-9.\d+ cron`, `phase-9.\d+ interval`,
   or `<setting>:00 ET`).
3. Each of the 11 schedule strings exactly matches the
   researcher's recommended replacement (per the table above).
4. `_LAUNCHD_JOBS` autoresearch description no longer contains
   `"FAILING exit 127"` AND contains the new wording (`"exit 1"`
   AND `"phase-23.5.19"`).
5. Live `/api/jobs/all` shows the new schedule strings + new
   description (backend reload picked up the edits).
6. All 27 prior phase-23 verifiers exit 0 (no regression).

## Plan steps

1. (DONE — RESEARCH) `gate_passed: true`.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:**
   a. Edit `backend/api/cron_dashboard_api.py:62-85` —
      replace each `_SLACK_BOT_JOBS` entry's `"schedule"` value
      with the researcher's recommended string. Add inline comments
      noting "configurable via X" for the 3 settings-driven values.
   b. Edit `backend/api/cron_dashboard_api.py:113` —
      replace the stale autoresearch description.
   c. Add `tests/verify_phase_23_6_2.py` — 6-check verifier.
   d. Restart backend (`launchctl kickstart -k`) so the cron
      dashboard endpoint picks up the new strings.
   e. Slack-bot restart (so `_seed_next_run_registry` re-seeds the
      registry against the freshly-restarted backend).
   f. Run sibling verifier sweep — all 27 prior must stay green.
4. **EVALUATE phase:** spawn fresh `qa` agent.
5. **LOG phase:** append `harness_log.md` AFTER Q/A. Flip status.

## Anti-patterns guarded

1. **Changing the actual cron triggers** — only the LABEL strings
   change; the underlying `register_phase9_jobs()` mapping +
   `start_scheduler` `add_job` calls are unchanged.
2. **Adding `cron-descriptor` or `cRonstrue` as a dep** — out of
   scope; bracket notation is what the live trigger emits.
3. **Touching the 5 launchd labels other than autoresearch
   description** — only the autoresearch DESCRIPTION line changes.
4. **Self-evaluation by Main** — Q/A is mandatory.

## Out of scope

- Changing actual cron expressions (purely cosmetic).
- Migrating to a different schedule-label format library.
- Updating the `_LAUNCHD_JOBS` schedule strings (already informative).
- Refactoring `_trigger_str()` or `_static_to_dict()`.

## Backwards compatibility

- Pure label/description string change.
- No code changes to `_static_to_dict`, `_trigger_str`, or any
  handler.
- Existing tests in `tests/api/test_cron_dashboard.py` assert
  shape (`source`, `schedule` key presence) but NOT specific
  schedule strings — confirmed in 23.5.2.5 + 23.5.13.2 reviews.
- The 23.5.x verifiers don't grep schedule strings — only `status`
  and `next_run`.

## Risk

- **Frontend rendering** — the dashboard renders `schedule`
  verbatim. New strings are short enough (< 40 chars) to fit
  without wrap. If the frontend has any string-based filter on the
  old labels (e.g. "show only phase-9 jobs"), it would break.
  Mitigation: research-confirmed no such filter exists.
- **Settings-driven values drift** — if the operator changes
  `morning_digest_hour` from 8 to e.g. 9, the label hardcodes 8.
  Mitigation: inline comment + low-priority drift; the operator
  can update the label OR the dashboard can derive at request-
  time in a future enhancement.

## References

- Research brief: `handoff/current/phase-23.6.2-research-brief.md`.
- Files to edit: `backend/api/cron_dashboard_api.py:62-85, 113`.
- Phase-23.5.19 archive (autoresearch exit-code transition):
  `handoff/archive/phase-23.5.19/`.
- APScheduler trigger str format:
  https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html
