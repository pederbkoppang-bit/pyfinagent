---
name: freshness-alarm-browser-driven-82-10
description: Step 82.10 facts -- the freshness EMITTER already existed and paged; only a trigger was missing. AlertDeduper does NOT suppress steady state (measured). macro_cron.py is the canonical register_*_cron template. paper_cycle_interval_sec does not exist.
metadata:
  type: project
---

Research for masterplan step 82.10 (give the data-freshness alarm an active
scheduled evaluator). Measured 2026-08-05.

**The step description was half wrong -- and this class of error recurs.**
It said the alarm "cannot page". The EMITTER already existed and already paged:
`cycle_health.py:100-135` `_fire_freshness_alarm` fires a **P1** via
`raise_cron_alert_sync`, and `compute_freshness` calls it itself at `:564-565`.
`alerting.py:46-53` even records a **real page storm from this exact alarm**
(phase-66 hotfix: "~120 pages/hour the moment a dashboard tab was open"). Only
the TRIGGER was missing. **Why:** a step description written from a symptom
("nobody got paged") can misattribute the missing component. **How to apply:**
before accepting "X doesn't exist", grep for X's emitter AND for hotfix
comments that mention X misbehaving -- a throttle in the code is proof the
thing it throttles was live.

**MEASURED: `AlertDeduper` does NOT suppress steady state.** Ran
`AlertDeduper(5, 1, 3).should_fire(..., severity="P1")`: 5 back-to-back calls
gave `[True, False, False, False, False]`, but after rewinding `last_fired_at`
by 1h1m the next call returned `True`. A P1 re-fires **every `repeat_hours`,
forever**. `_fire_freshness_alarm`'s docstring claiming "dedup ... so a
polling-loop caller doesn't spam Slack" is true only relative to a 60s browser
poll. **Any timer-driven caller needs its own state-transition gate** (the
`_cycle_heartbeat_last_was_stale` / `_ingestion_silence_last_was_stale` idiom at
`slack_bot/scheduler.py:113,117,761-795`). Naming caution: vendors call this
"hysteresis", a word banned in this repo's TRADING context only -- unrelated.

**Canonical scheduled-job idiom** (three instances, all identical shape):
`backend/backtest/macro_cron.py:110-148` (phase-82.0, the freshest),
`backend/meta_evolution/cron.py:43-84`, `backend/harness_self_audit_report.py:84-106`.
Shape: `register_X_cron(scheduler, *, replace_existing=True, ...)` +
`run_X(...)`, both fail-open, wired in `backend/main.py` inside the
`if settings.paper_trading_enabled:` block (`:307`) near `:336-342`. Scheduler
is a bare `AsyncIOScheduler()` at `:310` -- **default in-memory jobstore, so
missed runs across a restart do not exist to be caught up**; the
"catch-up spam" worry is not real in this config. The misfire/coalesce
invariant test (`tests/services/test_phase9_registration.py:50-80`) is scoped to
`slack_bot.register_phase9_jobs` ONLY, not repo-wide.

**Gotchas that will bite an implementer:**
- `paper_cycle_interval_sec` **does not exist in settings.py**. All three
  production call sites use `getattr(..., 24*3600.0)`; effective value is always
  `86400.0`.
- Patch `backend.services.observability.alerting.raise_cron_alert_sync`, NOT
  `cycle_health.raise_cron_alert_sync` -- the import is function-local at
  `cycle_health.py:109`, so the module-scope name does not exist and a wrong
  patch target silently yields `call_count == 0`.
- `raise_cron_alert_sync` returns `True` optimistically under a running loop
  (`alerting.py:274-277`, fire-and-forget) -- not evidence of delivery.
- `_TABLE_MAX_AGE_SEC` (`cycle_health.py:48-53`) has only **4** keys;
  `paper_trades` and `signals_log` use the caller's `cycle_interval_sec`.
- P1 is required for delivery on this machine: an empty `slack_webhook_url`
  routes P1 (only) to the bot-token fallback at `alerting.py:217-218`. P2/P3
  are silently dropped.
- `settings.paper_trading_enabled` gates the whole scheduler block, so a
  monitor registered there is disabled by the same switch as the thing it
  monitors.
- Criteria "red fires / green doesn't" are **already tested** at the
  `compute_freshness` level by `tests/verify_phase_25_A7.py` claims 8/9/10 --
  a new test re-asserting them there would be a guard that cannot fail.

**External anchors worth reusing:** Google SRE Book Ch.6 -- *"SRE teams
carefully avoid any situation that requires someone to 'stare at a screen to
watch for problems'"* (the one-line justification for any browser-driven-alarm
fix). Prometheus alerting practices -- batch-job thresholds *"at least enough
time for 2 full runs"*, which independently corroborates `CRITICAL_RATIO = 2.0`.
dbt source freshness -- run checks at *"double the frequency of your lowest
SLA"*. SRE Workbook alerting-on-SLOs -- **reset time** is the axis a
level-triggered alarm fails on.

See [[research-gate-discipline]] and [[cron-maintenance-jobs]].
