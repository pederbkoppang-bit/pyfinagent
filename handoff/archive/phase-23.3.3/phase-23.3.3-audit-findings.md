# Phase-23.3.3 audit findings — Slack-bot phase-9 jobs

**Cycle date:** 2026-05-07
**Scope:** the 7 phase-9 slack-bot jobs (daily_price_refresh,
weekly_fred_refresh, nightly_mda_retrain, hourly_signal_warmup,
nightly_outcome_rebuild, weekly_data_integrity, cost_budget_watcher).

## Verdict: PASS WITH FIX (substantive)

All 7 modules exist on disk in `backend/slack_bot/jobs/`. The
`register_phase9_jobs(scheduler, replace_existing=True)` function is
defined at `backend/slack_bot/scheduler.py:397-428` and correctly
maps each id to its module + cron trigger. **But the function had
zero callsites in the codebase** -- the 7 jobs have been dormant
since the file was added. Confirmed via `grep -rn
"register_phase9_jobs" backend/ scripts/` (researcher a40a5015c28ebd163).

The runbook `docs/runbooks/phase9-cron-runbook.md:23` explicitly says:
"Called once at Slack bot process startup AFTER `start_scheduler` has
set up the morning/evening digest + watchdog jobs." Dormancy is an
oversight, not intentional deferral.

## Per-job inventory (researcher's audit)

| Job | Module | Default state | Activation cost |
|---|---|---|---|
| daily_price_refresh | jobs/daily_price_refresh.py | Stub fetch (hardcoded dict); no real yfinance | $0 |
| weekly_fred_refresh | jobs/weekly_fred_refresh.py | Stub fetch; needs FRED_API_KEY for real wiring | $0 |
| nightly_mda_retrain | jobs/nightly_mda_retrain.py | Stub train_fn; PromotionGate rejects stub | $0 |
| hourly_signal_warmup | jobs/hourly_signal_warmup.py | In-memory cache warm via settings.watchlist | $0 |
| nightly_outcome_rebuild | jobs/nightly_outcome_rebuild.py | Stub fetch (returns []) | $0 |
| weekly_data_integrity | jobs/weekly_data_integrity.py | Real BQ row-count drift; alert_fn not wired (alerts silently dropped) | <$0.01 BQ |
| cost_budget_watcher | jobs/cost_budget_watcher.py | Real BQ INFORMATION_SCHEMA + BudgetEnforcer | <$0.01 BQ |

Total activation cost: <$0.02 in BQ. No yfinance / FRED / LLM API calls in stub state. Safe to bulk-activate.

## What was changed

```diff
 # backend/slack_bot/scheduler.py:start_scheduler -- after _scheduler.start():
+    try:
+        registered = register_phase9_jobs(_scheduler)
+        logger.info("phase-9 jobs registered: %s", registered)
+    except Exception as exc:
+        logger.warning("register_phase9_jobs fail-open at startup: %r", exc)

 # backend/slack_bot/scheduler.py:register_phase9_jobs mapping --
 # added misfire_grace_time + coalesce to every job kwargs:
-        "daily_price_refresh": ("...", "cron", {"hour": 1}),
+        "daily_price_refresh": ("...", "cron",
+            {"hour": 1, "misfire_grace_time": 3600, "coalesce": True}),
 ... (all 7 jobs updated; tier-correct grace times: 3600 daily, 7200 weekly, 600 hourly)
```

## Why option (a) over feature flag

Researcher recommended option (a) bulk-activation. Rejected
alternatives:
- **(b) feature flag**: blast radius in stub state is near-zero. A
  `phase9_jobs_enabled: bool = False` setting risks latent
  configuration drift -- the flag could be forgotten at False and
  jobs remain dark permanently. The runbook documents NO such flag.
- **(c) subset-first**: the risk differential between "cheap" and
  "expensive" jobs only exists when real fetch/write impls are
  injected. With stubs, activating all 7 has the same blast radius
  as activating 2.

## Operator-restart caveat (load-bearing)

Same as phase-23.3.2: the slack-bot daemon (PID 16385, running since
2026-04-08) won't pick up the new registrations until restarted:

```bash
pkill -f "slack_bot.app"
nohup python -m backend.slack_bot.app > handoff/logs/slack_bot.log 2>&1 &
```

After restart, all 11 jobs (4 core + 7 phase-9) are registered and
the heartbeat-push wired in phase-23.3.2 will surface their fires
on `/api/jobs/status` AND `/cron`'s Jobs tab.

## Sibling concerns deferred

1. **Real fetch/write injection** for daily_price_refresh +
   weekly_fred_refresh -- separate per-job phases.
2. **`alert_fn` injection** for weekly_data_integrity (drift computed
   but alerts silently dropped) -- P2 follow-up.
3. **Settings injection** for `cost_budget_watcher`'s soft-cap
   threshold (currently uses default) -- P2.
4. **Slack-bot stdout/stderr capture to a log file** so the new
   `phase-9 jobs registered: [...]` log line is visible to the
   /cron Logs tab -- bundled with phase-23.3.5.

## Verification

- `python tests/verify_phase_23_3_3.py` -> 4/4 OK.
- `pytest tests/services/test_phase9_registration.py -q` -> 4 passed.
- AST-parse on modified file: clean.

## Q/A

Per same-session pragmatism: deterministic verifier is the canonical
gate. Operator-restart is the behavioral activation step.
