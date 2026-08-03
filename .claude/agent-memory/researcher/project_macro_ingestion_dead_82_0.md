---
name: project-macro-ingestion-dead-82-0
description: historical_macro was NEVER scheduled (not a regression); backtest_end_date="2025-12-31" is the FRED observation_end so a bare re-ingest inserts zero rows; the P1 freshness alarm is browser-driven; MACRO_MAX_AGE_DAYS=35 takes a GLOBAL max so daily series mask a dead GDP
metadata:
  type: project
---

Measured 2026-07-31 (step 82.0 research gate). `financial_reports.historical_macro`
dead since 2026-03-25 (128d stale ingest, MAX(date)=2025-12-31).

**Root cause = absence, not breakage.** `ingest_macro` has ONE caller
(`run_full_ingestion`), whose three non-test call sites are a cold-start-only
guard (`backtest_engine.py:1303`, gated `if prices_count == 0`), a manual API
task (`api/backtest.py:259`), and a one-shot migration. Zero `add_job`, zero
launchd plist, zero crontab entry. `git log -S "ingest_macro"` shows **no commit
ever removed a caller** — don't go hunting for the regression, there isn't one.

**The trap that makes a "fix" look like it worked:** `settings.py:244`
`backtest_end_date="2025-12-31"` flows through `api/backtest.py:262` ->
`run_full_ingestion` -> `ingest_macro` -> the FRED `&observation_end=` param
(`data_ingestion.py:313`). That constant IS the observed MAX(date). Re-running
the UI ingest or the migration script returns 200 and inserts **zero rows**.

**Monitoring existed and was correctly red — and paged nobody.**
`cycle_health.py:565` -> `_fire_freshness_alarm` raises a P1, but
`compute_freshness` is only reachable from HTTP handlers
(`observability_api.py:36/:55`, `paper_trading.py:25`) that only the frontend
calls. No cron polls them. The alarm fires only while a human has the dashboard
open. Generalize: **an alarm with no clock is not monitoring.**

**Two threshold traps.** (a) `cache.py:232-251` takes `max_date` across ALL
series, so daily DGS10/T10Y2Y mask a dead GDP/CPI — the guard cannot detect the
failure it was written for. (b) 35 days is unsatisfiable per-series: FRED dates
monthly series to month-START and quarterly to quarter-START, so a healthy GDP
newest row reaches ~211d and a healthy CPIAUCSL ~72d. (c) `MAX(ingested_at)`
only advances when rows are INSERTED, so append+dedupe means a healthy no-op run
looks identical to a dead job — needs a separate run-receipt heartbeat.

**Blast radius is backtest-only.** Live regime uses `backend/tools/fred_data.py`
direct-to-FRED, NOT this table. Two refuted priors worth not re-deriving:
`sortino.py:108` queries `pyfinagent_data.historical_macro` (wrong dataset — it
lives in `financial_reports`) for series `DGS3MO`/`DTB3` (not in `FRED_SERIES`),
so its tier-1 MAR lookup has ALWAYS been dead, independent of staleness.
`data_server.py:185` serves `cached_macro(today)` stamped `as_of: today` over
7-month-old rows.

**Re-ingest is safe for history:** dedupe on `(series_id,date)` with no
UPDATE/MERGE path means a revision can never overwrite an existing row. But the
table is a vintage mosaic with no `realtime_start`, and there is already a large
publication-lag look-ahead (GDP row dated 2026-04-01 is visible from that date
but wasn't published until 2026-07-30, ~120d).

FRED key verified working 2026-07-31; all 7 series current to 2026-07-30.

**Why:** the operator's standing "historical_macro un-freeze token" was spent on
this; a one-shot re-ingest that goes stale again is an explicit failure of the
step.

**How to apply:** any macro-feed work must sever the end-date from
`backtest_end_date`, schedule the job, and split the data-cadence SLA (per
series) from the pipeline heartbeat (per run). Full brief:
`handoff/current/research_brief_82.0.md`. Related: [[project_metric_source_paths]],
[[project_observability_ops_residuals_60_4]].
