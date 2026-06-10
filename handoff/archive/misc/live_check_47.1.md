# Live Check — phase-47.1: historical_prices freshness restored

Captured 2026-05-28 ~23:22 CEST (21:22 UTC) on the running local system.

## 1. Freshness endpoint (curl) — band off-red
```
$ curl -s http://localhost:8000/api/paper-trading/freshness | jq '.sources.historical_prices'
{
  "last_tick_age_sec": 207.0,
  "interval_sec": 93600.0,
  "ratio": 0.0022115384615384614,
  "band": "green"
}
```
Was: `band=red`, age ~52 days. Now: **green**.

## 2. BQ MAX(ingested_at) — financial_reports.historical_prices (us-central1)
```
BEFORE: max_date=2025-12-30  max_ingested=2026-04-06 19:11:30  age_hours=1250  rows=1780404
AFTER:  max_date=2026-05-28  max_ingested=2026-05-28 21:15:29  age_hours=0     rows=1831731  tickers=507
```
+51,327 rows; 5-month market-data gap (2025-12-31 -> 2026-05-28) filled; ingest recency ~0h.

## 3. Scheduler — daily_price_refresh future next_run + catch-up fired
```
$ curl -s http://localhost:8000/api/jobs/all | (filter daily_price_refresh)
{"id":"daily_price_refresh","source":"slack_bot","schedule":"cron[hour='1']",
 "next_run":"2026-05-29T01:00:00+00:00","last_run":null,"status":"scheduled"}

backend_slack.log (post-restart PID 42151):
  apscheduler: Running job "run_production (trigger: date[2026-05-28 21:20:28 UTC])"
  job_runtime: {'job':'daily_price_refresh','status':'ok','duration_s':60.83,'skipped':False}
  apscheduler: Removed job daily_price_refresh_catchup
```
Confirms: explicit UTC tz registration (next_run at 01:00Z), AND the catch-up-on-start path
executed `run_production` successfully on restart (then auto-removed the one-off).

## 4. Immutable verification command — exit 0
```
$ source .venv/bin/activate && python -c "import ast; ast.parse(...); ast.parse(...)" \
  && grep -qE 'ingest_prices|DataIngestionService' backend/slack_bot/jobs/daily_price_refresh.py \
  && curl -s .../freshness | python3 -c "...band not in (red,unknown)..."
freshness OK band=green
EXIT_CODE=0
```
