# Phase-9 Cron Runbook

**Version:** 1.0 -- 2026-04-20
**Owner:** Peder
**Scope:** 7 scheduled jobs that run inside the Slack bot process under APScheduler.

---

## 1. Job inventory

| Job ID | Cadence | Module |
|--------|---------|--------|
| daily_price_refresh | daily 01:00 UTC | backend/slack_bot/jobs/daily_price_refresh.py |
| weekly_fred_refresh | Sun 02:00 UTC | backend/slack_bot/jobs/weekly_fred_refresh.py |
| nightly_mda_retrain | daily 03:00 UTC | backend/slack_bot/jobs/nightly_mda_retrain.py |
| hourly_signal_warmup | HH:05 UTC | backend/slack_bot/jobs/hourly_signal_warmup.py |
| nightly_outcome_rebuild | daily 04:00 UTC | backend/slack_bot/jobs/nightly_outcome_rebuild.py |
| weekly_data_integrity | Mon 05:00 UTC | backend/slack_bot/jobs/weekly_data_integrity.py |
| cost_budget_watcher | daily 06:00 UTC | backend/slack_bot/jobs/cost_budget_watcher.py |

## 2. How jobs are wired

`backend/slack_bot/scheduler.py::register_phase9_jobs(scheduler)` appends all 7 jobs to an existing APScheduler instance with `replace_existing=True`. Called once at Slack bot process startup AFTER `start_scheduler` has set up the morning/evening digest + watchdog jobs.

Idempotency is enforced per-job via `backend/slack_bot/job_runtime.py::IdempotencyKey` (daily / weekly / hourly). Double-registration (e.g. hot reload) is safe.

## 3. Manual invocation

Run any job on demand:

```bash
python -c "from backend.slack_bot.jobs.daily_price_refresh import run; print(run())"
```

Idempotency store is in-memory by default; production wires to BQ.

## 4. Observability

Each run emits `{job, status, duration_s}` events through the `heartbeat` context manager's `sink` callable. Default sink is `logger.info`; wire to a BQ `job_heartbeat` table in a later phase.

## 5. Failure modes + runbook

| Failure | Symptom | Runbook |
|---------|---------|---------|
| yfinance / fredapi rate-limit | daily_price_refresh or weekly_fred_refresh logs exception, `status=failed` | Idempotency key NOT marked; retries next hour/day automatically. |
| BQ credentials expired | outcome_rebuild + data_integrity write 0 rows; heartbeat status still ok | Owner re-auths BQ; next run catches up. |
| MDA retrain produces rejected model | nightly_mda_retrain logs `promoted=False` reason=...; baseline unchanged | Expected behavior. Investigate rejection reason in BQ `ts_forecast_shadow_log`. |
| cost_budget_watcher trips | Slack alert; cron_budget cap exceeded | Owner reviews `.claude/cron_budget.yaml` + inspects provider costs. |
| data_integrity detects drift | Slack alert with table name + delta_pct | Owner checks upstream data source for outage or schema change. |
| scheduler.py import fail | Slack bot boot crash | Revert last scheduler.py change; `launchctl kickstart -k gui/$UID/com.pyfinagent.frontend` + backend. |

## 6. Test suite

`tests/slack_bot/test_scheduler_phase9.py` covers: all 7 jobs register, replace_existing flag honored, ID list stable, runbook exists. Individual job modules have their own pytest files.

## 7. Re-enabling after outage

Idempotency keys are in-memory by default -- an outage restart resets them. On restart, expect each daily job to run once even if it already ran earlier that day (harmless under idempotency because write operations are themselves idempotent by date/week on the BQ side via the ingesters' dedup keys).

## 8. Changing schedule

Edit `register_phase9_jobs` mapping in `backend/slack_bot/scheduler.py`. Restart Slack bot via launchd.

## 9. References

- `backend/slack_bot/scheduler.py::register_phase9_jobs`
- `backend/slack_bot/jobs/*.py`
- `backend/slack_bot/job_runtime.py`
- `tests/slack_bot/test_scheduler_phase9.py`
- Harness log cycles 9.1 through 9.10 (2026-04-20 02:22 through 02:55 UTC)
