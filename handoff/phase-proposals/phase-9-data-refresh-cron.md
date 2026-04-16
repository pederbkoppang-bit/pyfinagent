# Phase 9 -- Data Refresh & Retraining Cron

## Goal

Close the biggest gap from the pyfinagent codebase audit: there is no scheduled
refresh of BigQuery source tables, no walk-forward retraining of the MDA
weights, and no automated drift / cost guardrails. Today `backend/slack_bot/
scheduler.py` runs only three APScheduler jobs (morning digest, evening digest,
watchdog). Backtests and live signals quietly consume whatever snapshot the
tables had at the last manual refresh, which means:

- `financial_reports.historical_prices` and `historical_fundamentals` go stale
  between manual pulls (yfinance / SEC).
- `historical_macro` only updates when a human runs a FRED migration.
- `quant_model.py` weights are frozen; no concept-drift response.
- The first request for a watchlist ticker pays the full signal compute cost.
- Outcome-tracker memory ages out because nobody re-aligns signal -> outcome
  mappings.
- Nobody watches BQ bytes scanned or LLM $ / day; cost regressions are found
  after the fact.

Phase 9 adds a production-grade scheduler layer reusing the existing
APScheduler instance in `backend/slack_bot/scheduler.py` (no new framework, no
new process) plus idempotent refresh jobs that write to staging tables and
atomically swap, with exponential backoff and Slack alerting on failure.

## Success criteria

A reviewer can verify Phase 9 is complete when all of the following hold:

1. `backend/slack_bot/scheduler.py` registers seven new jobs alongside the
   existing three, each with a unique `id`, an explicit `misfire_grace_time`,
   `coalesce=True`, `max_instances=1` (mutual exclusion), and a heartbeat file
   path.
2. Each refresh job writes to `<dataset>.<table>__staging`, validates row count
   + null ratio, then executes an atomic swap via BigQuery `CREATE OR REPLACE
   TABLE ... AS SELECT * FROM ...__staging` inside a single `execute_sql` call.
3. The walk-forward retraining job produces a new `optimizer_best.json`
   candidate under `backend/backtest/experiments/results/phase9_walkforward/`
   and only promotes it to `optimizer_best.json` when out-of-sample Sharpe
   beats the incumbent by at least +0.10 on a holdout window.
4. Signal cache warmup pre-populates Redis / in-process LRU for every ticker in
   the current watchlist hourly (aligned to `:05` so it runs after the top-of-
   hour BQ ETL the upstream providers use).
5. Data-integrity scan emits a Slack message (reusing `send_trading_escalation`
   with severity `P2`) whenever any monitored table has staleness > threshold,
   null ratio > threshold, or BQ bytes billed > 3x the trailing-7-day median.
6. Cost-budget watcher queries
   `region-US.INFORMATION_SCHEMA.JOBS_BY_PROJECT` plus `llm_usage_log` and
   posts a `P1` Slack alert if daily spend exceeds the configured budget.
7. All seven jobs survive a crash mid-run: re-running the job is a no-op if
   the staging table is incomplete (detected via a `status` column), and the
   atomic swap is either fully applied or fully absent.
8. `python -c "from backend.slack_bot.scheduler import start_scheduler"`
   succeeds; `python -m pytest tests/slack_bot/test_scheduler_phase9.py`
   passes; Slack bot startup logs list all ten jobs.

## Step-by-step plan

### Step 9.1 -- Heartbeat + idempotency primitives

- Add `backend/slack_bot/job_runtime.py` with:
  - `heartbeat(job_id: str) -> None` writing `handoff/heartbeats/<job_id>.json`
    with `{last_run_iso, status, duration_s, rows_written}`.
  - `with_idempotent_swap(dataset, table, build_sql)` context manager that
    creates `<table>__staging`, runs `build_sql`, validates, then issues the
    atomic `CREATE OR REPLACE TABLE` swap.
  - `with_backoff(fn, max_attempts=5, base=2.0)` -- exponential backoff with
    jitter for provider calls. On final failure, call
    `send_trading_escalation(..., severity="P1")`.

### Step 9.2 -- Daily price + fundamentals refresh

- Job `refresh_prices_daily`, cron `minute=15 hour=12 timezone=UTC` (07:15
  America/New_York, pre-market but after overnight provider ETL).
- Source: `yfinance` for last 5 trading days per watchlist ticker; SEC EDGAR
  for 10-Q / 10-K deltas. Write into `financial_reports.historical_prices` and
  `historical_fundamentals` via the staging + swap pattern.
- Cost budget: ~300 yfinance calls/day, ~15 SEC EDGAR calls/day, ~2 GB BQ
  bytes processed.

### Step 9.3 -- Weekly macro refresh

- Job `refresh_macro_weekly`, cron `day_of_week=sun hour=11 minute=0 tz=UTC`.
- Source: FRED series list (`DGS10`, `DGS2`, `T10Y2Y`, `UNRATE`, `CPIAUCSL`,
  `DFF`, `VIXCLS`, `M2SL`). Write into
  `pyfinagent_data.historical_macro` via staging + swap.
- Cost budget: ~40 FRED calls/week, ~50 MB BQ bytes processed.

### Step 9.4 -- Nightly walk-forward MDA retraining

- Job `retrain_mda_nightly`, cron `hour=8 minute=30 tz=UTC` (03:30 NY, after
  all daily data is in BQ).
- Rolling 252-trading-day training window, 21-day out-of-sample holdout.
- Invokes `backend/backtest/quant_optimizer.py` with `--walk-forward` flag and
  writes candidate `optimizer_best.json` under the results dir. Promotion
  gated on `oos_sharpe - incumbent_sharpe >= 0.10`; promotion emits a Slack
  `P2` notice.

### Step 9.5 -- Hourly signal cache warmup

- Job `warmup_signal_cache`, cron `minute=5 hour='9-16' tz=America/New_York`
  (market hours + shoulder).
- For every ticker in `watchlist` table: call the internal signal endpoint
  (`http://backend:8000/api/signals/<ticker>`) and cache the response in the
  existing in-process cache (`cache.preload_macro` pattern reuses the
  `cache.py` module).
- Exclusion: skip tickers whose last signal is less than 30 minutes old.

### Step 9.6 -- Nightly outcome-tracker rebuild

- Job `rebuild_outcome_tracker_nightly`, cron `hour=9 minute=0 tz=UTC`.
- Recomputes signal -> realized-outcome mappings from the prior trading day
  and upserts into `pyfinagent_data.outcome_tracker`. Agent memories reload
  on next BM25 refresh in the orchestrator.

### Step 9.7 -- Weekly data-integrity scan

- Job `integrity_scan_weekly`, cron `day_of_week=mon hour=10 minute=0 tz=UTC`.
- Checks per monitored table: max(update_ts), row count delta vs 7-day median,
  per-column null ratio, schema drift vs stored snapshot in
  `pyfinagent_data.schema_snapshots`. Emits Slack `P2` with a summary block
  and a diff payload attachment.

### Step 9.8 -- Cost-budget watcher

- Job `cost_budget_watcher`, cron `hour='*/6' minute=10 tz=UTC` (every 6h).
- Query BQ `INFORMATION_SCHEMA.JOBS_BY_PROJECT` for trailing-24h bytes billed,
  sum `llm_usage_log` dollars, sum yfinance + FRED + API-Ninjas request
  counts. If any exceeds the threshold in `backend/config/settings.py`
  (`cost_budget_bq_tb_day`, `cost_budget_llm_usd_day`, `cost_budget_api_req_day`),
  post `P1` escalation and set a global kill-switch flag
  `pyfinagent_data.runtime_flags.cost_circuit_breaker=true`.

### Step 9.9 -- Wire into existing scheduler

- Extend `start_scheduler` in `backend/slack_bot/scheduler.py` with the seven
  new `add_job(...)` calls. Keep the existing three untouched.
- Log the full job list at INFO on startup (one line per job: id, trigger,
  next_run_time).

### Step 9.10 -- Tests + docs

- `tests/slack_bot/test_scheduler_phase9.py` mocks APScheduler and asserts
  each job is registered with the correct trigger, `coalesce`, `max_instances`,
  and `misfire_grace_time`.
- `tests/slack_bot/test_job_runtime.py` unit-tests backoff, heartbeat file
  writes, and staging-swap semantics against a fake BQ client.
- Update `CLAUDE.md` (Harness Protocol section) with a pointer to the new
  job table; do **not** modify `.claude/masterplan.json` (handled by the
  masterplan aggregator step).

## Research findings

### APScheduler production patterns

- APScheduler's `AsyncIOScheduler` is already used by
  `backend/slack_bot/scheduler.py`. The docs call out three settings every
  production job should set: `coalesce=True` (collapse missed runs into a
  single run), `max_instances=1` (mutual exclusion), and `misfire_grace_time`
  (seconds after the scheduled run past which a missed run is dropped).
  Sources: https://apscheduler.readthedocs.io/en/3.x/userguide.html ,
  https://apscheduler.readthedocs.io/en/3.x/modules/schedulers/base.html ,
  https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html .
- `AsyncIOScheduler` + `httpx.AsyncClient` avoids thread-pool exhaustion
  under load; the existing codebase already does this. Source:
  https://www.python-httpx.org/async/ .
- Timezone handling: APScheduler cron triggers take `timezone=` explicitly;
  passing naive datetimes is a common footgun. Source:
  https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html .
- Flask-APScheduler production guide covers heartbeat + job listener
  patterns we reuse:
  https://viveksb007.github.io/2018/03/apscheduler-jobs-getting-processed-multiple-times .

### Orchestration patterns from Airflow / Dagster / Prefect

- Airflow's idempotency guidance ("an operator must be safe to re-run with
  identical inputs") is the conceptual source for the staging + atomic swap
  pattern. Sources:
  https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html ,
  https://airflow.apache.org/docs/apache-airflow/stable/concepts/tasks.html .
- Dagster's "software-defined assets" model frames each refresh as producing
  a named asset with known upstreams; we adopt the mental model even though
  we stay inside APScheduler. Source:
  https://docs.dagster.io/concepts/assets/software-defined-assets .
- Prefect 2.x documents the exponential-backoff retry policy we copy:
  https://docs.prefect.io/latest/concepts/tasks/#retries .
- Netflix Meson (now Maestro) posts on model retraining cadence -- nightly
  training with promotion gates is the canonical pattern:
  https://netflixtechblog.com/meson-workflow-orchestration-for-netflix-recommendations-fc932b4b3c66 ,
  https://netflixtechblog.com/maestro-netflixs-workflow-orchestrator-ee13a06f9c78 .

### Walk-forward retraining and concept drift

- Google's TFX production ML whitepaper ("TFX: A TensorFlow-based
  Production-Scale Machine Learning Platform", KDD 2017) defines the
  train / validate / push gating pattern we copy for the MDA promotion step:
  https://dl.acm.org/doi/10.1145/3097983.3098021 ,
  https://www.tensorflow.org/tfx/guide .
- Chip Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022),
  Ch. 9 covers shadow deployment + promotion gates and is the source for the
  "+0.10 Sharpe over incumbent" threshold pattern:
  https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/ .
- Emmanuel Ameisen, *Building Machine Learning Powered Applications*
  (O'Reilly, 2020), Ch. 11 on monitoring covers the drift signals we pick
  up in the integrity scan:
  https://www.oreilly.com/library/view/building-machine-learning/9781492045106/ .
- Bayer et al., "On the detection of concept drift in nonstationary time
  series", surveys ADWIN + Page-Hinkley; we use population-stability-index
  as the lightest drift proxy because it is computable with a single BQ
  query: https://arxiv.org/abs/2004.05785 ,
  https://arxiv.org/abs/1010.4784 (Bifet & Gavalda, ADWIN).
- Gama et al., "A survey on concept drift adaptation" (ACM Computing
  Surveys 2014) is the canonical reference:
  https://dl.acm.org/doi/10.1145/2523813 .

### BigQuery best practices

- BigQuery atomic swap via `CREATE OR REPLACE TABLE ... AS SELECT` is the
  officially recommended idempotent publish pattern:
  https://cloud.google.com/bigquery/docs/tables#replace_table_data .
- `INFORMATION_SCHEMA.JOBS_BY_PROJECT` is the authoritative source for
  trailing bytes billed: https://cloud.google.com/bigquery/docs/information-schema-jobs .
- Partitioned + clustered tables reduce bytes scanned by 10-100x in the
  integrity scan; the `historical_*` tables are already partitioned on
  `date`: https://cloud.google.com/bigquery/docs/best-practices-performance-overview .

### Cost + provider rate limits

- yfinance has no official rate limit but community consensus is ~2000
  requests / hour per IP before throttling:
  https://github.com/ranaroussi/yfinance/issues/1729 .
- FRED free tier: 120 requests / minute:
  https://fred.stlouisfed.org/docs/api/fred/ .
- SEC EDGAR fair-access policy: 10 requests / second, must set a
  descriptive User-Agent:
  https://www.sec.gov/os/accessing-edgar-data .

## Proposed masterplan.json snippet

```json
{
  "id": "phase-9",
  "name": "Data Refresh & Retraining Cron",
  "status": "proposed",
  "depends_on": ["phase-5.5", "phase-6"],
  "steps": [
    {
      "id": "9.1",
      "name": "Heartbeat + idempotency primitives (backend/slack_bot/job_runtime.py)"
    },
    {
      "id": "9.2",
      "name": "Daily price + fundamentals refresh job"
    },
    {
      "id": "9.3",
      "name": "Weekly FRED macro refresh job"
    },
    {
      "id": "9.4",
      "name": "Nightly walk-forward MDA retraining with promotion gate"
    },
    {
      "id": "9.5",
      "name": "Hourly signal cache warmup for watchlist"
    },
    {
      "id": "9.6",
      "name": "Nightly outcome-tracker rebuild"
    },
    {
      "id": "9.7",
      "name": "Weekly data-integrity scan"
    },
    {
      "id": "9.8",
      "name": "Cost-budget watcher with circuit breaker"
    },
    {
      "id": "9.9",
      "name": "Wire seven jobs into backend/slack_bot/scheduler.py"
    },
    {
      "id": "9.10",
      "name": "Tests + docs (tests/slack_bot/test_scheduler_phase9.py)"
    }
  ]
}
```

### APScheduler job definitions table

| job_id | trigger | cadence | timezone | mutual exclusion | heartbeat path |
|---|---|---|---|---|---|
| refresh_prices_daily | cron | minute=15 hour=12 | UTC | max_instances=1, coalesce=True, misfire_grace_time=900 | handoff/heartbeats/refresh_prices_daily.json |
| refresh_macro_weekly | cron | day_of_week=sun hour=11 minute=0 | UTC | max_instances=1, coalesce=True, misfire_grace_time=3600 | handoff/heartbeats/refresh_macro_weekly.json |
| retrain_mda_nightly | cron | hour=8 minute=30 | UTC | max_instances=1, coalesce=False, misfire_grace_time=1800 | handoff/heartbeats/retrain_mda_nightly.json |
| warmup_signal_cache | cron | minute=5 hour=9-16 | America/New_York | max_instances=1, coalesce=True, misfire_grace_time=300 | handoff/heartbeats/warmup_signal_cache.json |
| rebuild_outcome_tracker_nightly | cron | hour=9 minute=0 | UTC | max_instances=1, coalesce=True, misfire_grace_time=1800 | handoff/heartbeats/rebuild_outcome_tracker_nightly.json |
| integrity_scan_weekly | cron | day_of_week=mon hour=10 minute=0 | UTC | max_instances=1, coalesce=True, misfire_grace_time=3600 | handoff/heartbeats/integrity_scan_weekly.json |
| cost_budget_watcher | cron | hour=*/6 minute=10 | UTC | max_instances=1, coalesce=True, misfire_grace_time=600 | handoff/heartbeats/cost_budget_watcher.json |

### Cost budget (expected daily)

| Resource | Expected | Notes |
|---|---|---|
| yfinance calls | ~300 / day | 30 tickers x 5 trading days x retries |
| FRED calls | ~40 / week | weekly macro pull only |
| SEC EDGAR calls | ~15 / day | delta pulls on 10-Q/10-K |
| BQ bytes billed | ~4 GB / day | mostly integrity scan + INFORMATION_SCHEMA |
| LLM $ | $0.00 | no LLM calls in phase 9 refresh layer |

### Backoff policy

- Exponential backoff with jitter: delay_s = base ** attempt + uniform(0, 1)
- base=2.0, max_attempts=5 (per Prefect default, see citation above)
- On final failure: `send_trading_escalation(severity="P1", title=job_id,
  details={"error": str(exc), "attempts": 5})`.

### Verification commands

```bash
# Static syntax + import check
source .venv/bin/activate
python -c "import ast; ast.parse(open('backend/slack_bot/scheduler.py').read())"
python -c "import ast; ast.parse(open('backend/slack_bot/job_runtime.py').read())"
python -c "from backend.slack_bot.scheduler import start_scheduler; print('import ok')"

# Unit tests
python -m pytest tests/slack_bot/test_scheduler_phase9.py -v
python -m pytest tests/slack_bot/test_job_runtime.py -v

# Dry-run the scheduler and list registered jobs
python -c "
import asyncio
from slack_bolt.async_app import AsyncApp
from backend.slack_bot.scheduler import start_scheduler, _scheduler
app = AsyncApp(token='xoxb-test', signing_secret='test')
start_scheduler(app)
for j in _scheduler.get_jobs():
    print(j.id, j.trigger, j.next_run_time)
"

# BQ staging + swap smoke test (read-only inspection)
bq query --use_legacy_sql=false --dry_run \
  'SELECT COUNT(*) FROM `sunny-might-477607-p8.financial_reports.historical_prices__staging`'

# Heartbeat presence
ls handoff/heartbeats/ && cat handoff/heartbeats/refresh_prices_daily.json

# Cost-budget watcher sanity query
bq query --use_legacy_sql=false --dry_run \
  'SELECT SUM(total_bytes_billed) FROM `region-US`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
   WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)'
```

## References

1. https://apscheduler.readthedocs.io/en/3.x/userguide.html
2. https://apscheduler.readthedocs.io/en/3.x/modules/schedulers/base.html
3. https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html
4. https://viveksb007.github.io/2018/03/apscheduler-jobs-getting-processed-multiple-times
5. https://www.python-httpx.org/async/
6. https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html
7. https://airflow.apache.org/docs/apache-airflow/stable/concepts/tasks.html
8. https://docs.dagster.io/concepts/assets/software-defined-assets
9. https://docs.prefect.io/latest/concepts/tasks/#retries
10. https://netflixtechblog.com/meson-workflow-orchestration-for-netflix-recommendations-fc932b4b3c66
11. https://netflixtechblog.com/maestro-netflixs-workflow-orchestrator-ee13a06f9c78
12. https://dl.acm.org/doi/10.1145/3097983.3098021
13. https://www.tensorflow.org/tfx/guide
14. https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/
15. https://www.oreilly.com/library/view/building-machine-learning/9781492045106/
16. https://arxiv.org/abs/2004.05785
17. https://arxiv.org/abs/1010.4784
18. https://dl.acm.org/doi/10.1145/2523813
19. https://cloud.google.com/bigquery/docs/tables#replace_table_data
20. https://cloud.google.com/bigquery/docs/information-schema-jobs
21. https://cloud.google.com/bigquery/docs/best-practices-performance-overview
22. https://github.com/ranaroussi/yfinance/issues/1729
23. https://fred.stlouisfed.org/docs/api/fred/
24. https://www.sec.gov/os/accessing-edgar-data
