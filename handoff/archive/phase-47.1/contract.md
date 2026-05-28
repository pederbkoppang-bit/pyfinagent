# Contract — phase-47.1: Restore historical_prices freshness

**Cycle:** 1 of the production-ready+money push (see `active_goal.md`).
**Step id:** 47.1 | **Phase:** phase-47 | **Status:** in-progress | **Harness:** required | **Tier:** moderate

## Research-gate summary (PASSED)
Researcher `a34468862e0c792ab`, tier=moderate, `gate_passed: true`. 8 sources read in full
(floor 5), 19 URLs (floor 10), recency scan performed, 10 internal files inspected. Brief:
`handoff/current/research_brief_phase_44_1_price_freshness.md`.

Three independent root causes for the 52.1-day `historical_prices` staleness (fixing one alone re-rots):
- **RC1 — wrong destination.** `backend/slack_bot/jobs/_production_fns.py:111` writes
  `pyfinagent_data.price_snapshots` (schema `ticker,date,close,recorded_at` — no `ingested_at`).
  Freshness (`cycle_health.py:489`) + backtest (`data_ingestion.py`, `cache.py`) read
  `financial_reports.historical_prices` keyed on `MAX(ingested_at)`. `price_snapshots` has **zero
  other consumers** (researcher confirmed) -> safe to abandon.
- **RC2 — wrong universe + path.** `daily_price_refresh.py:35` is a 5-ticker close-only stub
  (`["AAPL","MSFT","NVDA","SPY","QQQ"]`). The audited bulk path is
  `DataIngestionService.ingest_prices()` — full OHLCV, idempotent dedup on `(ticker,date)`.
- **RC3 — fire lost to restart.** `scheduler.py:196` `AsyncIOScheduler()` = in-memory jobstore;
  phase-9 cron jobs (`scheduler.py:796-810`) carry NO explicit timezone (unlike MAIN jobs at
  :199-243). The daily `hour=1` fire is dropped whenever the slack-bot is down/asleep at that tick
  (`misfire_grace=3600` too small); `hourly_signal_warmup` re-fires within the hour so it shows "ok".

**Plan-altering finding:** the LIVE screener reads **yfinance directly** (`autonomous_loop.py:324,372`),
NOT `historical_prices` BQ. So this step is the prerequisite for the backtest re-baseline + freshness
alarms — it is NOT the direct no-trades unblocker (that is 47.2, the empty candidate set).

**Researcher gotcha:** a `SQLAlchemyJobStore` cannot pickle `functools.partial(run, **prod_fns)`
closures (`scheduler.py:823`). Therefore this cycle does NOT add a persistent jobstore; it uses a
**module-level production fn + catch-up-on-start** (picklable, jobstore-ready later) to fix RC3.

## Hypothesis
Rewiring `daily_price_refresh` to call `DataIngestionService.ingest_prices(get_sp500_tickers(), ...)`
(full OHLCV -> `historical_prices`), pinning the phase-9 cron timezone to UTC, and adding a
catch-up-on-start that fires the ingest when `historical_prices` is stale, plus a one-time gap
backfill, makes the freshness band go off-red and keeps it fresh across restarts — without the
pickle/jobstore risk.

## Immutable success criteria (verbatim from masterplan.json phase-47.1)
1. daily_price_refresh production path writes full OHLCV to financial_reports.historical_prices via ingest_prices (NOT pyfinagent_data.price_snapshots, NOT the 5-ticker stub)
2. historical_prices freshness band is off-red (not red, not unknown) on /api/paper-trading/freshness after backfill
3. BQ MAX(ingested_at) on financial_reports.historical_prices within 2 days of now
4. phase-9 cron jobs registered with explicit timezone; daily_price_refresh has a future next_run_time + a catch-up-on-start path when stale
5. ast.parse passes on all modified backend files; live_check_47.1.md captures freshness curl + BQ MAX(ingested_at)

Immutable verification command: see masterplan phase-47.1 `verification.command`.

## Plan steps
1. **`daily_price_refresh.py`** — add module-level `run_production(*, day=None, lookback_days=7)` that
   acquires the idempotency heartbeat (same pattern as `run`) and calls
   `DataIngestionService(bq_client, settings).ingest_prices(get_sp500_tickers(), start, end)` over a
   rolling recent window. Keep `run()` unchanged for tests/back-compat.
2. **`scheduler.py` register_phase9_jobs** — resolve `daily_price_refresh` to `run_production`
   (module-level, picklable); drop it from `prod_fns_per_job` (no closure injection); add
   `"timezone": "UTC"` to every phase-9 cron entry; raise daily `misfire_grace_time`.
3. **`scheduler.py` start_scheduler** — after registration, add a catch-up: if
   `historical_prices` is stale (BQ `MAX(ingested_at)` age > ~20h, fail-open), schedule a one-off
   `date`-trigger `run_production` ~10s out so downtime self-heals.
4. **`_production_fns.py`** — leave `make_price_*` factories but stop wiring them for
   daily_price_refresh; add a deprecation note pointing to `run_production`.
5. **One-time backfill** — run `ingest_prices(get_sp500_tickers(), start=<last_date+1>, end=<tomorrow>)`
   once to fill the ~52-day gap -> band green immediately.
6. **Verify** — ast.parse; freshness curl off-red; BQ `MAX(ingested_at)` recent; write
   `live_check_47.1.md`. Then spawn fresh Q/A.

## Blast radius
`backend/slack_bot/jobs/daily_price_refresh.py`, `backend/slack_bot/scheduler.py`,
`backend/slack_bot/jobs/_production_fns.py`, one BQ write to `financial_reports.historical_prices`
(idempotent, append-only; no DROP/DELETE). Slack-bot restart required to load the new registration.

## References
- `handoff/current/research_brief_phase_44_1_price_freshness.md` (gate)
- `handoff/current/roadmap_master.md` (workstreams 1-2)
- `backend/backtest/data_ingestion.py:93` ingest_prices; `backend/tools/screener.py:29` get_sp500_tickers
- `backend/services/cycle_health.py:489` freshness reader; `.claude/rules/backend-slack-bot.md`
