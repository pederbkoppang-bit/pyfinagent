# Experiment Results — phase-47.1: Restore historical_prices freshness

**Cycle:** 1 of the production-ready+money push. **Window:** 2026-05-28T23:00-23:25 CEST.
**Step:** 47.1 | **Result:** ready for Q/A.

## What was built / changed

### Code (3 files)
1. `backend/slack_bot/jobs/daily_price_refresh.py`
   - NEW module-level `run_production(*, day=None, lookback_days=7, store=None)` — acquires the
     daily idempotency heartbeat, then calls
     `DataIngestionService(bigquery.Client(...), get_settings()).ingest_prices(get_sp500_tickers(), start, end)`
     over a rolling window. Full OHLCV -> `financial_reports.historical_prices` (with `ingested_at`),
     idempotent dedup on `(ticker,date)`. Fail-open. Module-level => picklable (jobstore-ready later).
   - `__all__` now exports `run_production`. `run()` left untouched (test/back-compat).
2. `backend/slack_bot/scheduler.py`
   - `register_phase9_jobs`: `daily_price_refresh` now resolves to `run_production` (not bare `run`);
     removed from `prod_fns_per_job` so NO close-only fetch/write closures are injected.
   - All 7 phase-9 cron entries now pin `timezone=ZoneInfo("UTC")` (previously inherited the
     ambiguous system default — a missed-fire contributor). `daily_price_refresh` misfire grace
     raised 3600 -> 21600s.
   - `start_scheduler`: NEW catch-up-on-start — schedules a one-off `date`-trigger `run_production`
     ~20s after boot so a Mac-asleep/down 01:00-UTC tick self-heals. `timedelta` added to imports.
3. `backend/slack_bot/jobs/_production_fns.py`
   - Deprecation banner on `make_price_*` factories (they wrote the WRONG table
     `pyfinagent_data.price_snapshots`, close-only, 5 tickers). No longer wired for daily_price_refresh.

### One-time backfill (live action)
`run_production(lookback_days=160)` executed once via the venv:
`BACKFILL_RESULT {'written': 51327, 'tickers': 503, 'window': ['2025-12-19','2026-05-29'], 'path':'ingest_prices'}`
Closed the real 5-month market-data gap (`max_date` 2025-12-30 -> 2026-05-28).

## Verbatim verification output

Immutable command (masterplan phase-47.1, run with `.venv` active per CLAUDE.md):
```
$ source .venv/bin/activate && python -c "import ast; ast.parse(...daily_price_refresh.py...); ast.parse(...scheduler.py...)" \
  && grep -qE 'ingest_prices|DataIngestionService' backend/slack_bot/jobs/daily_price_refresh.py \
  && curl -s .../api/paper-trading/freshness | python3 -c "...band assert..."
freshness OK band=green
EXIT_CODE=0
```

Live scheduler evidence after slack-bot restart (PID 1580 -> 42151):
```
/api/jobs/all -> daily_price_refresh: {"id":"daily_price_refresh","source":"slack_bot",
  "schedule":"cron[hour='1']","next_run":"2026-05-29T01:00:00+00:00","status":"scheduled"}
backend_slack.log:
  apscheduler: Running job "run_production (trigger: date[...21:20:28 UTC])"
  job_runtime: {'job':'daily_price_refresh','status':'ok','duration_s':60.83,...}   <- catch-up fired & succeeded
  apscheduler: Removed job daily_price_refresh_catchup
```

BQ post-state (`financial_reports.historical_prices`, us-central1):
```
max_date=2026-05-28  max_ingested=2026-05-28 21:15:29  age_h=0  rows=1831731 (was 1780404)  tickers=507
```

Freshness endpoint entry: `historical_prices -> {"last_tick_age_sec":207,"interval_sec":93600,"ratio":0.0022,"band":"green"}` (was band=red, ~52d).

## Success-criteria mapping (masterplan phase-47.1)
1. daily_price_refresh writes full OHLCV to financial_reports.historical_prices via ingest_prices — **MET** (run_production -> ingest_prices; catch-up ran ok; 51327 rows; price_snapshots/5-ticker stub abandoned)
2. freshness band off-red — **MET** (band=green)
3. BQ MAX(ingested_at) within 2 days — **MET** (age_h=0)
4. phase-9 crons explicit timezone + daily future next_run + catch-up-on-start — **MET** (UTC tz on all 7; next_run 2026-05-29T01:00Z; catch-up fired ok then removed)
5. ast.parse passes + live_check captures evidence — **MET** (ast OK; live_check_47.1.md written)

## DISCLOSURE TO Q/A (anti-rubber-stamp transparency)
- The masterplan `verification.command` originally assumed `sources` was a **list of dicts**
  (`{x.get('name'):x for x in d.get('sources',[])}`). The real `/api/paper-trading/freshness`
  shape is `sources` = **dict keyed by table name**. The original command therefore threw
  `AttributeError` and would have produced a FALSE FAIL. I corrected the command to
  `d.get('sources',{}).get('historical_prices',{}).get('band',...)`. **The success_criteria list
  was NOT touched** — only the broken mechanical check. This is a pre-Q/A error correction
  (prevents a false negative), not goalpost-moving. Please independently re-curl the endpoint.
- The command uses `python` (resolves inside `.venv`); run with `source .venv/bin/activate` per
  CLAUDE.md (without venv it returns 127 command-not-found — environment, not a logic defect).

## Scope honesty
- This step makes BQ `historical_prices` fresh + durable. It does NOT by itself produce a trade:
  the live screener reads yfinance directly (`autonomous_loop.py:324,372`), so this is the
  prerequisite for the backtest re-baseline + freshness alarms. The trade unblocker is **47.2**
  (empty `new_candidates` set).
- Persistent SQLAlchemy jobstore deliberately NOT added (cannot pickle the `functools.partial`
  prod-fn closures — research brief). Catch-up-on-start + UTC tz + raised misfire grace are the
  restart-survival path this cycle; a module-level `run_production` keeps the jobstore option open.

## Files
backend/slack_bot/jobs/daily_price_refresh.py, backend/slack_bot/scheduler.py,
backend/slack_bot/jobs/_production_fns.py, .claude/masterplan.json (phase-47 added + command fix),
handoff/current/{contract.md, research_brief_phase_44_1_price_freshness.md, live_check_47.1.md}.

## Cycle-2 fix (post-Q/A CONDITIONAL — agent ad63febb6bb3b3c81)
Q/A cycle-1 verdict was **CONDITIONAL**: all 5 immutable criteria independently MET, harness
compliance 5/5, verification.command edits ruled acceptable (not goalpost-moving), scope honest —
but the misfire_grace_time 3600 -> 21600 change left `test_register_phase9_grace_times_per_tier`
RED without updating it (a red guard = disabled mutation detector). Legitimate catch.

Fix applied (cycle-2):
1. `tests/services/test_phase9_registration.py` — `test_register_phase9_grace_times_per_tier`:
   split `daily_price_refresh` into its own assertion expecting **21600** (phase-47.1 intentional);
   other daily jobs stay 3600. Also de-fragilized the PRE-EXISTING red guard
   `test_start_scheduler_source_calls_register_phase9_jobs` (matched `register_phase9_jobs(_scheduler`
   prefix instead of the stale full-literal that broke when phase-23.6 added app=/loop= kwargs).
2. **Additional consumers found (operator "grep all consumers" mandate)** —
   `tests/slack_bot/test_phase9_production_wiring.py` also encoded the OLD wiring (asserted
   daily_price_refresh is a `functools.partial` with fetch_fn/write_fn). Updated 2 tests
   (`test_register_without_app_returns_bare_run`, `test_register_with_app_uses_functools_partial`)
   + the module docstring to assert daily_price_refresh resolves to `run_production` (not wrapped);
   kept the bare-run/partial guard on `weekly_fred_refresh` as the probe job. The Q/A's targeted
   run had not surfaced these two; the full-consumer sweep did.

Verbatim re-run (5 affected files):
```
$ python -m pytest tests/slack_bot/test_phase9_production_wiring.py \
    tests/services/test_phase9_registration.py tests/slack_bot/test_daily_price_refresh.py \
    tests/slack_bot/test_scheduler_phase9.py tests/api/test_observability.py -q
30 passed, 1 warning in 7.13s
```
No production code changed in cycle-2 (test-guard updates only); the immutable verification command
+ all 5 success_criteria remain MET. Ready for fresh Q/A.
