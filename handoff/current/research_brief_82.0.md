# Research Brief -- Step 82.0: `financial_reports.historical_macro` dead since 2026-03-25

Tier: **moderate** (caller-specified). Audit-class: **false**. Date: 2026-07-31.
Write-first: file created before any source read; grown incrementally.

---

## 1. ROOT CAUSE (stated plainly)

**There has never been a scheduled caller of `ingest_macro`. The feed did not
break in March 2026 -- it was never on a cadence. 2026-03-25 is the timestamp of
the last *manual* invocation.** A second, independent defect means a bare
re-ingest today would insert **zero** rows.

### 1a. No scheduler -- evidence chain (verified by direct read)

1. `ingest_macro` is defined at `backend/backtest/data_ingestion.py:297` and has
   **exactly one caller in the repo**: `run_full_ingestion` at
   `backend/backtest/data_ingestion.py:373`. (`grep -rn "ingest_macro"` across
   `*.py|*.sh|*.json|*.plist|*.md|*.js|*.ts` returns only the definition, that
   call site, and prose in `handoff/archive/**`.)
2. `run_full_ingestion` has **three non-test call sites, none scheduled**:
   - `backend/backtest/backtest_engine.py:1303` -- guarded by `if prices_count == 0:`
     (`:1300`), a **cold-start-only** auto-ingest, wrapped in a "non-fatal"
     `except` (`:1307`). `historical_prices` is not empty, so this branch is dead.
   - `backend/api/backtest.py:259` -- `_run_ingestion_async`, a **manual**
     operator-triggered background task behind the `/ingest` API.
   - `scripts/migrations/extend_historical_data.py:101` -- a **one-shot migration**.
3. No APScheduler registration. Every `add_job(` in `backend/` (`main.py:367`
   queue processor, `api/paper_trading.py:1382` `paper_trading_daily`,
   `autoresearch/cron.py:29`, `meta_evolution/cron.py:62`,
   `services/mcp_health_cron.py:200`, `slack_bot/scheduler.py:227/239/251/263/330`)
   was enumerated -- **none is an ingestion job**. `cron_control.CONTROLLABLE`
   (`backend/services/cron_control.py:36-39`) holds only `paper_trading_daily`
   and `ticket_queue_process_batch`.
4. No launchd plist. `~/Library/LaunchAgents/com.pyfinagent.*` = backend,
   frontend, slack-bot, autoresearch, ablation, watchdogs, proxy, logrotate.
   Grep for `ingest|macro` across them: **no match**. `crontab -l` = one line
   (`slack_mention_checker.sh`).
5. **Not a regression.** `git log -S "ingest_macro"` returns no commit that
   removed a caller: `3e9c3034` (introduction), `238bc024`/`5291f772`/`ffdb8816`
   (phase-50.1/50.2 FX work that only *cites* the httpx idiom in prose), and
   `ccdf2e28` (phase-66.2, which recorded "macro staleness ruled backtest-only").
   Nothing was disabled or deleted -- the scheduled caller never existed.

### 1b. The date cap -- why a bare re-ingest is a no-op

`backend/config/settings.py:244` pins `backtest_end_date: str = Field("2025-12-31", ...)`.
`backend/api/backtest.py:262` passes `end_date=req.end_date or settings.backtest_end_date`
into `run_full_ingestion`, which forwards it to `ingest_macro` (`:373`), where it
becomes the FRED `&observation_end=` parameter (`data_ingestion.py:313`).
`scripts/migrations/extend_historical_data.py:104` hardcodes the same literal.

**`MAX(date) = 2025-12-31` in the live table is exactly this constant.** The
212-day date lag is not the ingestion failing; it is the ingestion obeying its
argument. Re-running today via the UI or the migration script requests
`observation_end=2025-12-31` and inserts nothing -- an apparently successful
ingest that leaves the table just as dead. **Any repair that does not sever the
macro end-date from `backtest_end_date` is a no-op.**

### 1c. Why nobody was paged

A freshness monitor **does** exist and **is** correctly configured:
`backend/services/cycle_health.py:51` sets `"historical_macro": 3_024_000.0`
(35 d); `:507` measures `MAX(ingested_at)` age; `:565` calls
`_fire_freshness_alarm(sources)`; `:100-125` raises a **P1**
`freshness_critical_historical_macro`. At 128 days the ratio is 3.66 vs
`CRITICAL_RATIO = 2.0` (`:42`) -> red -> should page.

It never fires on a schedule because **`compute_freshness` is only reachable
from HTTP handlers**: `backend/api/observability_api.py:36` and `:55`, and
`backend/api/paper_trading.py:25`. The only callers of those routes are the
frontend (`frontend/src/lib/api.ts:608`, `:612`; `components/LiveBadge.tsx`).
**No cron or script polls them** (grep over `*.sh`/`*.py` for the route paths
returns only frontend + tests). The alarm is browser-driven: it fires only while
a human has the dashboard open, and `alerting.py`'s `repeat_hours` window
(default 1 h) then suppresses repeats. Monitoring existed; it was decoupled from
any clock.

---

## 2. FRED API key + upstream availability -- VERIFIED WORKING

Probed live 2026-07-31 with the key from `backend/.env` (value not printed;
`len=32`, the FRED standard). All 7 series returned HTTP 200 with current data:

| series | freq | latest obs (2026-07-31) | age of newest row |
|---|---|---|---|
| `DGS10` | daily (B) | 2026-07-30 | 1 d |
| `T10Y2Y` | daily (B) | 2026-07-30 | 1 d |
| `FEDFUNDS` | monthly | 2026-06-01 | 60 d |
| `CPIAUCSL` | monthly | 2026-06-01 | 60 d |
| `UNRATE` | monthly | 2026-06-01 | 60 d |
| `UMCSENT` | monthly | 2026-06-01 | 60 d |
| `GDP` | quarterly | 2026-04-01 | 121 d |

**The repair is unblocked**: no key problem, no upstream gap, ~7 months of data
waiting. Note FRED dates monthly series to the **first day of the month** and
quarterly to the **first day of the quarter** -- this is the single most
important fact for the SLA design below.

---

## 3. Is `MACRO_MAX_AGE_DAYS = 35` right? No -- it is wrong in both directions

`backend/backtest/cache.py:26` sets a flat 35 days; `:232-251` compares
`date.today()` to **`max_date` taken across ALL series** (the loop at `:234-240`
keeps a single global maximum).

Two structural failures:

- **Too lax / blind.** Because the max is global, the daily series mask the rest.
  Once `DGS10`/`T10Y2Y` are flowing, `max_date` is ~1 day old and the gate passes
  **even if `GDP`, `CPIAUCSL`, `UNRATE`, `UMCSENT` and `FEDFUNDS` all stopped
  updating**. The guard cannot detect the failure it was written for.
- **Too strict / structurally unsatisfiable per series.** A healthy `GDP` newest
  row is 121 days old *today* and reaches ~211 days just before the next advance
  estimate. A healthy `CPIAUCSL` row reaches ~72 days just before the next
  release. Applied per series, 35 days would be permanently red for 5 of 7.

`MACRO_MAX_AGE_DAYS` is one number doing two incompatible jobs. Split it.

### Recommended per-series freshness SLA

Derived from FRED's dating convention + measured release cadence (§2, §7
sources). The threshold is the **worst-case age of the newest row immediately
before the next scheduled release**, plus buffer.

| series | cadence | dating | worst-case age before next release | **SLA (max_date age)** |
|---|---|---|---|---|
| `DGS10` | daily (B) | actual day | 3 d (long weekend + holiday) | **5 d** |
| `T10Y2Y` | daily (B) | actual day | 3 d | **5 d** |
| `FEDFUNDS` | monthly | month-start | ~63 d (H.15, ~1st business day) | **70 d** |
| `UMCSENT` | monthly | month-start | ~62 d (final, ~last Friday) | **70 d** |
| `UNRATE` | monthly | month-start | ~67 d (Employment Situation, ~1st Friday) | **75 d** |
| `CPIAUCSL` | monthly | month-start | ~72 d (CPI, ~12-14 d after month end) | **80 d** |
| `GDP` | quarterly | quarter-start | ~211 d (BEA advance, ~30 d after quarter end) | **225 d** |

**This is a data-cadence SLA, not a pipeline SLA.** They must be separate checks
(§5).

---

## 4. Point-in-time: does re-ingesting revised values threaten the backtests?

**Direct answer: the re-ingest itself does NOT corrupt existing history -- but
the table already carries a large look-ahead bias that the re-ingest will extend
into 2026. Fix the read side in the same step, or the fresh data makes the
existing bias worse by covering more of the sample.**

Three findings:

**(a) Existing rows are safe.** `ingest_macro` dedupes on `(series_id, date)`
(`data_ingestion.py:305`, `:328-329`) and only ever `insert_rows_json`s new keys
(`:341`). There is no UPDATE/MERGE path. A row written in 2024 **cannot** be
overwritten by a 2026 revision. Backtest history for dates <= 2025-12-31 is
byte-preserved. The `optimizer_best.json` baseline is not silently invalidated.

**(b) The table is a vintage mosaic, and nothing records which vintage.** The
request at `:310-315` passes no `realtime_start`/`realtime_end`; both **default
to today's date** (`fredr` reference, read in full), i.e. `output_type=1`
observations for the real-time period `[today, today]` = **latest vintage**.
Every row therefore carries the vintage as of whenever it happened to be first
ingested, and the schema has no `realtime_start` column to say which. Rows from
the 2024 backfill, the 2026-03-25 run, and a 2026-07 run would be three
different vintages sitting in one column.

**(c) The bigger, already-live bug is publication lag, not revision.**
`cache.py:380 cached_macro(cutoff_date)` selects the newest row with
`date <= cutoff` (fast path `:391-395`, BQ fallback `:402-411`). Because FRED
dates monthly series to month-start, a backtest at cutoff `2026-06-15` is handed
the `CPIAUCSL` row dated `2026-06-01` -- **a value not published until
~2026-07-14**. `GDP` is worst: the row dated `2026-04-01` is visible to any
cutoff >= 2026-04-01 but was not published until 2026-07-30, a **~120-day
look-ahead**. This affects the 6 macro features at
`backend/backtest/historical_data.py:270-275` (`fed_funds_rate`, `cpi_yoy`,
`unemployment_rate`, `yield_curve_spread`, `consumer_sentiment`, `treasury_10y`;
`GDP` is ingested but unused as a feature). The two daily series are unaffected.

Literature is unambiguous that this class of leakage inflates results:
StarQube reports factor-timing strategies show **"15-25% higher Sharpe ratios in
backtests using final revised figures compared to initial releases"**; Kuang
(2026) measures revisions at **"8.2 percent of later-outcome mean squared
error"** for real-activity targets vs **"3.6 percent"** for inflation -- i.e.
revision bias is materially worse for the GDP/UNRATE family than for CPI;
MACROCAST (Carriero, Pettenuzzo & Shekhar, 2026) names exactly this failure as
**"revision bias, as training on fully revised data diverges from the
preliminary, vintage-specific releases available to real-time forecasters"** and
rules it out by fine-tuning only on **"vintage-specific ALFRED data"**.

**Call:** proceed with the re-ingest -- it is safe for existing rows -- but land
the point-in-time fix **before or with** the backfill, because rows written
without a `realtime_start` can never be retro-attributed to a vintage.

---

## 5. Recommended durable repair

Scoped to the ingestion + monitoring surface, as instructed.

**R1 -- Sever the date cap (blocking; without this nothing else matters).**
`ingest_macro` must default `observation_end` to `date.today()`, not inherit
`settings.backtest_end_date` (`settings.py:244`). `backtest_end_date` is a
*backtest window* setting; using it as an *ingestion horizon* is the conflation
that produced the 212-day date lag. Same idiom already applied at
`backend/agents/mcp_servers/data_server.py:181-182` (phase-75.3 replaced a
hardcoded end-of-2025 literal with `date.today().isoformat()`).

**R2 -- Schedule it.** Register a daily APScheduler job on the `"main"`
scheduler, mirroring `paper_trading_daily` (`backend/api/paper_trading.py:1382`):
`trigger="cron"`, `day_of_week="mon-fri"`, `replace_existing=True`,
`misfire_grace_time=3600`, `coalesce=True` (the `:1394-1400` comment documents
why the 1 s default grace silently skipped a run on 2026-05-25 -- inherit that
lesson, do not re-learn it). Add the job id to `cron_control.CONTROLLABLE`
(`backend/services/cron_control.py:36`) so it appears in the cron dashboard and
can be paused/triggered like every other job. A weekday-daily cadence is correct
even though 5 of 7 series are monthly: the two daily series need it, and a
no-op run is free (§R3 makes it cheap).

**R3 -- Idempotency hardening.** `_get_existing_macro` (`data_ingestion.py:288`)
**fails OPEN**: `except Exception: return set()` (`:294-295`). On a transient BQ
error the dedupe set is empty and every observation is re-inserted, duplicating
the whole table. This is precisely the failure that phase-75.9 fixed for prices
-- `_get_existing_price_dates` now re-raises (`:101-103`) with the rationale
spelled out at `:81-88`. Mirror it. Also bound the query: `:290` is an
unbounded `SELECT DISTINCT series_id, date` over the full table; add
`WHERE series_id IN UNNEST(@series)` and a 30 s timeout (CLAUDE.md BQ rule --
`_get_existing_price_dates:99` has `timeout=30`, `_get_existing_macro:292` has
none).
Note on approach: the append-only + `(series_id,date)` dedupe pattern is
**correct as-is** for this workload and should be kept rather than converted to
MERGE. BigQuery's own guidance confirms the alternative is costly -- CDC/upsert
requires the Storage Write API default stream, protobuf encoding, declared
primary keys, and a `max_staleness` tuning exercise, and DML against
streaming-buffer rows fails outright. For 7 series x ~1 row/day, append + dedupe
is the right tool.

**R4 -- Point-in-time safety (do this with the backfill, not after).**
Preferred: add a `realtime_start` DATE column to `historical_macro` and populate
it from FRED (either `output_type=4`, initial-release-only, or explicit
realtime params), then have `cached_macro` filter `realtime_start <= cutoff`
alongside `date <= cutoff`. Fallback if the schema change is out of scope: a
per-series publication-lag offset applied inside `cached_macro` so a monthly row
is invisible until `date + lag`. Be explicit in the contract that the fallback
fixes **publication lag only, not revision bias** -- it is a partial measure.
The backfill order matters: rows written today without `realtime_start` cannot
later be attributed to a vintage.

**R5 -- Two-layer monitoring (the current single layer cannot work).**
The literature is consistent that a data-row check and a job-liveness check are
different instruments. Conduktor: distinguish stalled pipelines from
legitimately-quiet ones using **"heartbeat records...synthetic events injected
at regular intervals with known timestamps."** OneUptime: **"Standard alerting
rules fire when a metric crosses a threshold. But what happens when the metric
simply stops arriving?... it does not return zero, it returns empty."**

- **Layer 1 -- job heartbeat / dead-man's switch.** Write a run receipt on every
  invocation (success or failure) and alert if none in 48 h. This is required
  because `MAX(ingested_at)` **only advances when rows are inserted** -- the
  dedupe means a healthy run with nothing new to fetch leaves `ingested_at`
  frozen. `cycle_health.py:507` therefore conflates "job didn't run" with "job
  ran, nothing new", and would false-alarm on a monthly-only table. OneUptime's
  named anti-pattern applies: **"an untested dead man's switch is worse than
  none at all because it gives false confidence"** -- the step should include a
  fault-injection proof that the alert actually fires.
- **Layer 2 -- per-series data freshness.** `MAX(date)` **grouped by
  `series_id`** against the §3 SLA table. Replaces the global-max logic at
  `cache.py:232-251`.
- **Fix the polling gap.** Whatever the thresholds, `_fire_freshness_alarm` must
  be reachable from a scheduled job, not only from an HTTP handler a browser
  happens to call (§1c). Simplest: have the new ingest job call
  `compute_freshness` at the end of each run.

---

## 6. Blast radius -- what else is degraded (confirm/refute as asked)

| Consumer | Verdict | Evidence |
|---|---|---|
| Backtest feature vector | **CONFIRMED degraded** | `historical_data.py:270-275` builds 6 of the ~49 features from `cache.cached_macro` (`:48`). With `_macro_full` empty they resolve to the newest row <= cutoff, i.e. 2025-12-31 values for every 2026 cutoff. |
| Backtest hang / 40-min stall | ~~**CONFIRMED**~~ **WITHDRAWN 2026-08-03 -- see annotation below** | ~~`preload_macro` returns 0 at `cache.py:251` -> `_macro_full` empty -> `cached_macro` skips the fast path at `:387` and issues a **per-cutoff-date BQ query** at `:399-417`. Matches the CLAUDE.md "backtests hang after ~40min" note.~~ |
| `backend/services/cycle_health.py:507` | **CONFIRMED** red, but never polled | See §1c. |
| `backend/metrics/sortino.py:101-121` | **REFUTED as a staleness victim -- it is broken for two other reasons** | `:108` queries `` `{project}.pyfinagent_data.historical_macro` `` but the table lives in **`financial_reports`** (`settings.py:59 bq_dataset_reports="financial_reports"`; `migrate_backtest_data.py:27 DATASET = "financial_reports"`). And it asks for series `'DGS3MO', 'DTB3'`, neither of which is in `FRED_SERIES` (`data_ingestion.py:22` -- the table has `DGS10`). Tier 1 is dead twice over; it has *always* fallen through to tier 2/3 via the fail-open at `:120-121`. Fixing the feed will not fix this. |
| `backend/agents/mcp_servers/data_server.py:184-185` | **CONFIRMED degraded, silently** | Calls `cached_macro(date.today())` and labels the response `"as_of": cutoff` = today, while the newest underlying row is 2025-12-31. The MCP `macro://` resource reports **7-month-old values stamped with today's date** to Layer-2 agents. |
| **LIVE analysis pipeline** | **REFUTED -- not degraded** | `backend/services/macro_regime.py:23` imports `get_macro_indicators` from `backend/tools/fred_data.py`, which hits FRED directly (`fred_data.py:13 FRED_BASE`, `:16 SERIES`, `:35 _fetch_series`). `macro_regime.py:358` + `:532` state explicitly that the net-liquidity path "writes NO BQ table (historical_macro untouched)". |

> ### ANNOTATION 2026-08-03 -- one row above is WITHDRAWN
>
> This brief is a **dated record of what was believed on 2026-07-31**. It is
> annotated here rather than rewritten, so the reasoning trail stays auditable.
>
> **The "Backtest hang / 40-min stall -- CONFIRMED" row is FALSE and is
> withdrawn.** `preload_macro` did **not** return 0 and no hang was occurring.
> `historical_macro.date` is a **STRING** column, and the phase-25.D7 staleness
> gate tested `isinstance(rd, datetime.date)` -- false for every production row
> -- so the refusal branch never executed. Measured on the pre-step table:
> `preload_macro` returned **4412** (it CACHED the stale data); post-fix it
> returns **4729**.
>
> The real defect was worse than a hang: backtests were **silently trained on
> 212-day-old macro features**, and had been for as long as that guard existed.
> Discovered by the step-82.0 cycle-1 Q/A FAIL; full detail in
> `handoff/current/experiment_results.md` and
> `handoff/current/evaluator_critique_82.0.md`.
>
> **Every other row in this table was independently re-verified and stands**,
> including the THREE that pending steps depend on: `sortino.py`
> REFUTED-as-staleness-victim (82.8), `data_server.py` CONFIRMED degraded
> (82.9), and `cycle_health.py` CONFIRMED red but never polled (82.10).
> Note the LIVE-analysis-pipeline row immediately above also stands: this is
> NOT a live-money defect. Executors may rely on every row in this table
> EXCEPT the withdrawn one.

**Scope answer:** a 128-day-dead feed is degrading **backtests and the MCP macro
resource**, not live trading signal quality. The live regime path is independent.
This bounds the urgency: it is a research-integrity defect, not a live-money one
-- consistent with what phase-66.2 (`ccdf2e28`) concluded ("macro staleness ruled
backtest-only"), which this brief re-verifies rather than inherits.

### Adjacent defects found (out of scope for 82.0 -- queue as their own steps)

1. `sortino.py:108` wrong dataset + wrong series (row 4 above). Silent fail-open.
2. `data_server.py:185` stale-data-labelled-`as_of`-today (row 6 above).
3. `_get_existing_macro` fail-open dedupe (§R3) -- in scope for 82.0 only because
   the repair must not ship a duplication hazard; the identical pattern may exist
   in `_get_existing_fundamentals`, unverified here.

---

## 7. Sources

### Read in full (>=5 required; counts toward the gate) -- 8

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|---|
| 1 | https://arxiv.org/html/2607.05882v2 | 2026-07-31 | paper (preprint) | WebFetch, arXiv native HTML | Kuang, *Revision Risk in Real-Time Macroeconomic Forecasting*: revisions are **"8.2 percent of later-outcome mean squared error"** (real activity) vs **"3.6 percent"** (inflation); decomposition `Rw(f) = Rv(f) + E[Δt(v,w)²] + 2E[et(v)Δt(v,w)]`. Revision bias is worse for GDP/UNRATE than CPI. |
| 2 | https://arxiv.org/abs/2606.28670 | 2026-07-31 | paper (preprint) | WebFetch (abs; `/html/` 404) | Carriero, Pettenuzzo & Shekhar, *MACROCAST* (27 Jun 2026): names **"revision bias, as training on fully revised data diverges from the preliminary, vintage-specific releases available to real-time forecasters"**; eliminates it via **"vintage-specific ALFRED data"**. |
| 3 | https://github.com/mortada/fredapi/blob/master/README.md | 2026-07-31 | official lib docs | WebFetch | `get_series()` **is equivalent to `get_series_latest_release()`**; ALFRED accessors `get_series_first_release` / `get_series_as_of_date` / `get_series_all_releases`; each observation carries `date` / `realtime_start` / `realtime_end`. |
| 4 | https://sboysel.github.io/fredr/reference/fredr.html | 2026-07-31 | official lib docs | WebFetch | Parameter reference for `fred/series/observations`: **`realtime_start` "Defaults to today's date"**, `realtime_end` likewise; `observation_end` defaults `9999-12-31`; `limit` max `100000`; `output_type` 1=real-time period, 2=all vintages, 3=new/revised only, **4=initial release only**. This is the evidence that our un-parameterised call returns latest vintage. |
| 5 | https://oneuptime.com/blog/post/2026-02-06-heartbeat-dead-man-switch-opentelemetry-pipeline/view | 2026-07-31 | eng. blog | WebFetch | **"Standard alerting rules fire when a metric crosses a threshold. But what happens when the metric simply stops arriving?... it does not return zero, it returns empty."** Anti-pattern: **"an untested dead man's switch is worse than none at all because it gives false confidence."** |
| 6 | https://www.conduktor.io/glossary/data-freshness-monitoring-sla-management | 2026-07-31 | vendor eng. doc | WebFetch | **"SLAs for data freshness should be derived from business requirements, not arbitrary technical targets."** Distinguishes event-time lag from ingestion-time lag; stalled-vs-quiet pipelines separated via **"heartbeat records...synthetic events injected at regular intervals."** |
| 7 | https://docs.cloud.google.com/bigquery/docs/change-data-capture | 2026-07-31 | official docs | curl + tag-strip (JS-rendered; WebFetch inadequate) | CDC upserts require Storage Write API **default stream**, **protobuf**, **declared primary keys** (<=16 cols); `max_staleness` tuning formula given (p95 apply duration x2 + 7 min buffer). Justifies keeping append+dedupe over MERGE for 7 series. |
| 8 | https://starqube.com/point-in-time-data/ | 2026-07-31 | industry practitioner | WebFetch | Factor-timing strategies show **"15-25% higher Sharpe ratios in backtests using final revised figures compared to initial releases"**; prescribes **"temporal tables that track effective dates and version histories"** and a dual-query architecture (latest vs as-of). |

### Identified but snippet-only (does NOT count toward gate) -- 14

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://fred.stlouisfed.org/docs/api/fred/series_observations.html | official docs | **Unfetchable.** WebFetch -> HTTP 403; `curl` -> `(92) HTTP/2 stream 1 was not closed cleanly: INTERNAL_ERROR` on 3 attempts (varied UA, `--http1.1`, `--compressed`). Same parameter set read in full via source #4 (fredr mirrors the API 1:1). |
| https://fred.stlouisfed.org/releases/calendar | official docs | Release-calendar UI; cadence obtained by direct API probe (§2) instead. |
| https://github.com/leaderanalytics/Vyntix.Fred.FredClient/issues/4 | community | GDP Q1-2014 three-release worked example; corroborates #3/#4, adds nothing. |
| https://macrosynergy.com/research/macroeconomic-data-and-systematic-trading-strategies/ | industry | HTTP 403. |
| https://macrosynergy.com/research/how-macro-quantamental-trading-signals-will-transform-asset-management/ | industry | Not fetched; JPMaQS "35 million vintages" claim is snippet-level. |
| https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/... | official (RTDSM) | HTTP 403. RTDSM is cited second-hand via source #1's data section. |
| https://www.bloomberg.com/company/press/bloomberg-introduces-point-in-time-economic-data-... | vendor press | Commercial PIT product; not applicable (local-only, free-tier project). |
| https://www.prnewswire.com/news-releases/bloomberg-introduces-point-in-time-economic-data-...html | vendor press | Duplicate of above. |
| https://analystprep.com/study-notes/cfa-level-2/problems-in-backtesting/ | education | Textbook-tier restatement of look-ahead bias. |
| https://www.getdbt.com/blog/data-slas-best-practices | vendor eng. | HTTP 404 (link rot). |
| https://streamkap.com/resources-and-guides/data-freshness-monitoring | vendor eng. | Overlaps #6. |
| https://cloud.google.com/blog/products/bigquery/life-of-a-bigquery-streaming-insert | official blog | Streaming-buffer internals; #7 covers what we need. |
| https://www.systemsarchitect.io/services/google-bigquery/strategies/pt/leverage-streaming-inserts-with-deduplication-stra | community | `insertId` dedup is best-effort/undocumented-window -- noted, not relied on. |
| https://atlan.com/know/ai-agent/data-for-ai/data-observability-for-ai-pipelines/ | vendor | 2026 recency hit; see recency scan. |

**URLs collected: 22 unique** (8 read in full + 14 snippet-only).

### Search-query composition (3-variant discipline)

- **Year-less canonical:** "FRED API series observations realtime_start realtime_end ALFRED vintage revisions documentation"; "data freshness monitoring dead man's switch heartbeat scheduled ingestion pipeline staleness SLA"; "macroeconomic data revisions look-ahead bias backtest real-time vintage data asset pricing"; "BigQuery MERGE idempotent upsert time series ingestion streaming buffer best practices"
- **Current-year (2026):** "data pipeline observability 2026 freshness anomaly detection silent failure macro data ingestion"; "FRED release calendar publication lag CPI UNRATE GDP monthly quarterly schedule 2026"
- **Last-2-year (2025):** "point-in-time macroeconomic data 2025 systematic strategy backtest vintage revision bias quant"

### Recency scan (2024-2026) -- PERFORMED

**Result: 3 new findings that complement (do not supersede) the canonical prior art.**

1. **Revision bias is now quantified, not just asserted.** Kuang (Jul 2026,
   source #1) puts numbers on it -- 8.2% vs 3.6% of MSE -- and the asymmetry
   maps directly onto our series mix: our `GDP`/`UNRATE` features sit in the
   high-revision bucket, `CPIAUCSL` in the low one. Prior art (Croushore &
   Stark RTDSM) established that revisions matter; the 2026 work sizes it.
2. **Vintage-consistency is now treated as a first-class model-design
   constraint.** MACROCAST (Jun 2026, source #2) is the first TSFM to rule out
   both temporal contamination and revision bias by construction. Directionally
   this raises the bar: "we used latest-vintage macro" is no longer a defensible
   default in 2026 literature.
3. **Point-in-time macro became a purchasable commodity in 2025-2026**
   (Bloomberg PIT Economic Releases, 3,000+ indicators back to 1997; JPMaQS).
   Not adoptable here -- pyfinagent is local-only, free-tier -- but it confirms
   ALFRED-based DIY vintage capture (source #3's `get_series_first_release` /
   `output_type=4`) is the correct free substitute rather than an exotic choice.

No 2024-2026 source contradicts the older canonical guidance; the older sources
remain valid and the new ones sharpen the magnitude estimates.

---

## 8. Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL (**8**)
- [x] 10+ unique URLs total (**22**)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full papers / pages read, not abstracts (arXiv native HTML for #1;
      #2 abstract-page only -- `/html/2606.28670v1` returned 404, disclosed;
      #7 via curl + tag-strip per the JS-rendered-docs rule)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered ingestion, cache, features, monitoring,
      alerting, schedulers, launchd, crontab, git history
- [x] Contradictions noted (sortino "degraded" hypothesis **refuted**; live
      pipeline "degraded" hypothesis **refuted**; both were plausible priors)
- [x] Claims cited per-claim, not in a footer
- [ ] `_get_existing_fundamentals` not audited for the same fail-open pattern
      (flagged in §6, not verified)

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 14,
  "urls_collected": 22,
  "recency_scan_performed": true,
  "internal_files_inspected": 19,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Root cause: ingest_macro was NEVER scheduled -- its only caller is run_full_ingestion, whose three non-test call sites are a cold-start-only guard (backtest_engine.py:1303), a manual API task (api/backtest.py:259), and a one-shot migration. No add_job, no launchd plist, no crontab entry; git log -S shows no caller was ever removed. 2026-03-25 is the last manual run. A SECOND defect makes a bare re-ingest a no-op: settings.py:244 backtest_end_date='2025-12-31' flows into the FRED observation_end param, which is exactly the observed MAX(date). The P1 freshness alarm exists and is correctly red, but compute_freshness is only reachable from HTTP handlers the frontend calls -- no cron polls it, so it is browser-driven. FRED key verified working; all 7 series current to 2026-07-30. MACRO_MAX_AGE_DAYS=35 is wrong both ways: it takes a GLOBAL max across series so daily DGS10 masks a dead GDP, and 35d is unsatisfiable per-series (healthy GDP reaches 211d). Per-series SLA table supplied. Re-ingest is SAFE for existing rows (dedupe never overwrites), but the table is a vintage mosaic with no realtime_start, and a large publication-lag look-ahead already exists (GDP ~120d). Live trading is NOT degraded -- macro_regime uses tools/fred_data.py direct. sortino.py:108 queries the wrong dataset AND wrong series: broken independently.",
  "brief_path": "handoff/current/research_brief_82.0.md",
  "gate_passed": true
}
```
