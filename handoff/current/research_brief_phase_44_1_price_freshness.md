# Research Brief — phase-44.1: Restore historical_prices freshness

**Tier:** moderate
**Date:** 2026-05-28
**Objective:** Validated, best-practice fix for getting
`financial_reports.historical_prices` fresh again (52.1d stale, 48x SLA)
so the autonomous paper-trading loop can decide on real data.

Status: IN PROGRESS — writing incrementally per WRITE-FIRST directive.

---

## Internal code-audit (Explore role)

### Finding 1 — Wrong-destination bug CONFIRMED (table + schema + column mismatch)

| Aspect | Daily job writes | Freshness + backtest read |
|---|---|---|
| Table | `pyfinagent_data.price_snapshots` (`_production_fns.py:111`) | `financial_reports.historical_prices` (`cycle_health.py:489`; `data_ingestion.py:79,95`; `cache.py:102,293`) |
| Schema | `(ticker, date, close, recorded_at)` (`_production_fns.py:86-89`) | `(ticker, date, market, currency, open, high, low, close, volume, ingested_at)` (`migrate_backtest_data.py:30-39`) |
| Freshness key column | `recorded_at` | **`ingested_at`** — does NOT exist in snapshot schema |

The two tables are different datasets, different schemas, and the
freshness probe keys on a column (`ingested_at`) the snapshot table
doesn't even have. Even if `cycle_health` pointed at `price_snapshots`
it would still read NULL age. The daily job has NEVER fed the table the
freshness strip and backtest read from. **Root cause #1 confirmed.**

### Finding 2 — `historical_prices` consumers (who depends on what)

| Consumer | File:line | Reads/Writes | Notes |
|---|---|---|---|
| Freshness strip | `cycle_health.py:489` | reads `historical_prices.ingested_at` | UI `/api/paper-trading/freshness` + `/api/observability/freshness` |
| Backtest ingest | `data_ingestion.py:79,95,365` | writes `historical_prices` (full OHLCV) | the CORRECT ingest path; schema-complete |
| Backtest cache | `cache.py:102,293` | reads `historical_prices` | `preload_prices` bulk query for backtests |
| Backtest engine | `backtest_engine.py:283,1189,1198,1200` | auto-ingest trigger if `historical_prices` empty | only fires when row count == 0, not on staleness |

**`price_snapshots` consumers: NONE found** outside the writer. Grep for
`price_snapshots` returns only `_production_fns.py`. => Redirecting the
write away from `price_snapshots` breaks nothing (no reader depends on it
staying populated). This clears **Option A (redirect write)** of the
downstream-breakage risk.

### Finding 3 — Screener uses LIVE yfinance, NOT historical_prices BQ

`autonomous_loop.py:324` calls `screen_universe(period="6mo", ...)` from
`backend/tools/screener.py`; candidate enrichment at `:372,378` uses
`yf.Ticker(x).info` live. The screener path does **not** read
`historical_prices`. **Implication for impact:** stale `historical_prices`
does NOT directly starve the daily screen/decide/trade loop — that path
pulls live yfinance. Stale BQ prices DO starve: (a) the **backtest /
optimizer** engine (Sharpe/DSR evidence the harness optimizes against),
(b) the **freshness strip** that gates operator confidence and fires P1
Slack alarms (`cycle_health.py:548`). So phase-44.1's value is restoring
the evidence base + clearing the red freshness band, not unblocking the
live screener per se. (Flag for the contract: confirm whether any
decision gate hard-blocks on freshness band == red before trading.)

### Finding 4 — "never_run" root cause: in-memory jobstore + restart timing

- `scheduler.py:196` creates `AsyncIOScheduler()` with **NO `jobstores=`
  argument** => default `MemoryJobStore`. All `next_run_time` + fire
  history is lost on every process restart.
- The heartbeat listener IS wired (`scheduler.py:248-251`:
  `add_listener(_aps_to_heartbeat, EVENT_JOB_EXECUTED|ERROR|MISSED)`),
  pushing to `http://127.0.0.1:8000/api/jobs/heartbeat`
  (`job_status_api.py`). So an "ok" status proves a real fire post-restart.
- **The asymmetry is the smoking gun.** `hourly_signal_warmup` (cron
  `minute=5`, fires every hour) shows "ok" => it HAS fired since the last
  restart. `daily_price_refresh` (cron `hour=1`) shows "never_run" =>
  it has NOT fired since the last restart, AND the in-memory store has no
  memory of any earlier fire. `misfire_grace_time=3600` (`scheduler.py:798`)
  only forgives a 1-hour miss; a restart after 02:00 means the 01:00 tick
  is permanently skipped that day, and tomorrow's tick only lands if the
  process survives past 01:00. `register_phase9_jobs` is reached
  (registration is fail-open per-job, `scheduler.py:817,828`, but the
  outer try at `:277-281` succeeds — `hourly_signal_warmup` firing proves
  the registration loop ran).
- **No slack-bot launchd plist exists.** `~/Library/LaunchAgents/` has
  `com.pyfinagent.{backend,frontend,backend-watchdog,mas-harness,ablation,
  autoresearch,claude-code-proxy}` but NO slack-bot entry. The bot is
  started manually / ad-hoc and is NOT auto-respawned by launchd. Any
  manual restart wipes the in-memory schedule. **Root cause #2 confirmed:
  the daily fire is lost to restart because nothing persists next_run.**

### Finding 5 — Daily job universe is a 5-ticker STUB

`daily_price_refresh.py:35`: `universe = tickers or ["AAPL","MSFT","NVDA",
"SPY","QQQ"]`. Even with the destination fixed, the daily job would only
refresh 5 tickers — not the S&P-500-sized universe the backtest needs.
The CORRECT bulk path is `data_ingestion.py::ingest_prices(tickers,
start, end)` which pulls full OHLCV in `_YF_BATCH` chunks with
`group_by="ticker", auto_adjust=True` and writes the complete schema
incl. `ingested_at` (`:152`). **A fix must also widen the universe** (or
delegate the daily refresh to the ingestion path with the real universe
from `screener.get_sp500_tickers()`).

### Finding 6 — historical_prices schema / partition (for backfill)

`migrate_backtest_data.py:30-39`: flat table, **no TimePartitioning, no
clustering**. `date` is STRING (`YYYY-MM-DD`), `ingested_at` is TIMESTAMP.
Inserts via `insert_rows_json` (streaming buffer). De-dup is application-
level: `data_ingestion._get_existing_price_dates()` (`:77-91`) reads
existing `(ticker, date)` pairs and skips them before insert => the
ingest path is already idempotent on `(ticker, date)`. ~378K rows for
S&P-500 x 3yr per the module docstring. **Backfill of ~52 days x ~500
tickers ≈ 26K rows** — well within streaming-insert tolerance.

## External research

### Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://apscheduler.readthedocs.io/en/3.x/userguide.html | 2026-05-28 | official doc | WebFetch full | MemoryJobStore "simply keeps jobs in memory… use when you always recreate jobs at start"; persistent stores survive restarts; misfire only recoverable with a persistent store; `coalesce` default **False**; timezone via constructor `timezone=` arg; "Job stores must never be shared between schedulers." |
| 2 | https://medium.com/@trading.dude/why-yfinance-keeps-getting-blocked-and-what-to-use-instead-92d84bb2cc01 | 2026-05-28 | industry blog | WebFetch full | yfinance scrapes undocumented Yahoo endpoints, no credential => IP-flagged; Yahoo tightened limits ~early 2024 so even `yf.Ticker('AAPL').info` errors; "only real alternative for reliable, high-volume data is a paid API"; treat yfinance as prototype-grade for automated daily jobs. |
| 3 | https://docs.cloud.google.com/bigquery/docs/managing-partitioned-tables | 2026-05-28 | official doc | WebFetch full | "If a qualifying DELETE covers all rows in a partition, BigQuery removes the entire partition… without scanning bytes or consuming slots"; partition decorator copy supports force-overwrite; require-partition-filter recommended. |
| 4 | https://oneuptime.com/blog/post/2026-02-17-how-to-use-merge-statements-in-bigquery-for-upsert-operations/view | 2026-05-28 | industry blog (2026) | WebFetch full | Canonical partition-aware MERGE template; dedup source via `ROW_NUMBER() OVER (PARTITION BY key ORDER BY ts DESC)`; put partition col in ON clause for pruning; MERGE is idempotent on a reliable business key. |
| 5 | https://datalakehousehub.com/blog/2026-02-de-best-practices-04-idempotent-pipelines/ | 2026-05-28 | industry blog (2026) | WebFetch full | "Idempotency is not about preventing retries; it's about making retries safe." For daily OHLCV, **partition-overwrite by date is simplest + safest**; insert-only+dedup-on-read is "not recommended" (every consumer sees dirty data before cleanup); business key `symbol+date+timeframe`. |
| 6 | https://apscheduler.readthedocs.io/en/3.x/faq.html | 2026-05-28 | official doc | WebFetch full | "Run the scheduler in a dedicated process"; "Sharing a persistent job store among two or more processes will lead to incorrect scheduler behavior… APScheduler does not currently have any interprocess synchronization"; jobs silently don't run if the process exits before they fire or threads disabled. |
| 7 | https://big-data-demystified.ninja/2020/03/26/bigquery-error-update-or-delete-statement-over-table-would-affect-rows-in-the-streaming-buffer-which-is-not-supported/ | 2026-05-28 | community/industry | WebFetch full | Exact error: "UPDATE or DELETE statement over table would affect rows in the streaming buffer, which is not supported"; streamed rows linger ~up to 90 min; workaround = restrict DML to rows older than ~180 min, OR use Storage Write API (DML-on-recent supported since Nov 2023), OR avoid DML entirely. |
| 8 | https://montecarlo.ai/blog-data-freshness-explained/ | 2026-05-28 | industry blog | WebFetch full | Freshness != "job ran"; verify rows actually arrived via timestamp comparison, not orchestration logs; check freshness MORE often than the SLA window; proactive automated alerts over complaint-driven detection. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://github.com/ranaroussi/yfinance/issues/2411 | primary (issue) | Page returned only metadata (closed-not-planned); no comment body extractable |
| https://github.com/ranaroussi/yfinance/issues/2518 | primary (issue) | Snippet sufficient: 429 persists in 0.2.61 even after prior fixes |
| https://github.com/ranaroussi/yfinance/issues/2125 | primary (issue) | Snippet: maintainer suggests try/except + backoff in loops |
| https://oneuptime.com/blog/post/2026-02-17-how-to-set-up-incremental-data-loading-patterns-for-bigquery-using-scheduled-queries-and-partitions/view | industry (2026) | Covered by sources 4+5 |
| https://github.com/apache/airflow/issues/59408 | primary (issue) | Snippet confirms: "DML jobs fail on streaming tables – missing native mechanism to wait for streaming buffer flush" — corroborates source 7 |
| https://github.com/agronholm/apscheduler/issues/465 | primary (issue) | APScheduler 4.0 progress tracker (recency scan) |
| https://apscheduler.readthedocs.io/en/master/versionhistory.html | official doc | APScheduler 4.0 version history (recency scan) |
| https://www.getdbt.com/blog/data-slas-best-practices | industry | 404 at fetch time; replaced by Monte Carlo (source 8) |
| https://betterstack.com/community/guides/scaling-python/apscheduler-scheduled-tasks/ | industry | Corroborates persistent-jobstore-for-production guidance |
| https://docs.cloud.google.com/bigquery/docs/change-data-capture | official doc | CDC ingestion; over-engineered for this use case |
| https://tacnode.io/post/what-is-stale-data | industry | Staleness causes/detection; covered by source 8 |

URLs collected total: 19 (8 read in full + 11 snippet-only).

### Recency scan (2024-2026)

Searched the 2024-2026 window for: APScheduler 4.0 persistence changes; BigQuery
idempotent MERGE/partition-overwrite (2026 oneuptime + datalakehousehub posts);
yfinance 429 breakages (2025 issues #2411/#2518); BigQuery Storage Write API
DML-on-recent (Nov 2023+). **Findings that complement/supersede canonical
sources:**

1. **APScheduler 4.0 (alpha)** is a ground-up redesign: persistent data stores
   become shareable across processes/nodes, workers pull jobs from the store, and
   the Shelve store is dropped in favor of **SQLAlchemyJobStore on SQLite**.
   BUT it is still alpha (`4.0.0a1`), old 3.x job data is NOT compatible, and
   schema migration is required. **=> For phase-44.1, stay on APScheduler 3.x +
   SQLAlchemyJobStore(SQLite).** The 4.0 multi-process capability is irrelevant
   here (single-process daemon). (Source: apscheduler versionhistory, GitHub #465.)
2. **BigQuery Storage Write API now permits DML on recently-streamed rows**
   (since Nov 2023). This is a forward option but NOT needed: the existing ingest
   path already does app-level `(ticker, date)` dedup, so we avoid DML-on-streaming
   entirely. (Source 7 comment + Airflow #59408.)
3. **yfinance 2025-2026 breakages are ongoing** (issues #2411 Apr-2025 on 0.2.54,
   #2518 still-429 on 0.2.61). No fix has made it reliable for high-volume daily
   pulls — reinforces "design the job to fail-open + retry, don't assume success."
4. **2026 idempotency consensus** (oneuptime + datalakehousehub, both Feb-2026)
   converges on **partition-overwrite as the simplest safe daily-load pattern** and
   explicitly deprecates insert-only+dedup-on-read.

No finding contradicts the recommended approach below; the recency scan
strengthens it (partition-overwrite preference is current best practice, and the
in-memory-jobstore fragility is unchanged across APScheduler 3.x).

### Key findings (per-claim cited)

1. **In-memory jobstore loses all schedule state on restart** — "[MemoryJobStore]
   simply keeps the jobs in memory… Use this when you always recreate your jobs at
   the start of your application" (Source 1, APScheduler userguide). The project
   neither persists nor reliably recreates fires across restarts => daily fire is
   lost. Misfire recovery only works "when a job is scheduled in a persistent job
   store and the scheduler is shut down and restarted" (Source 1).
2. **A persistent jobstore is the documented fix for restart-survival**, and for a
   single-process daemon SQLAlchemyJobStore(SQLite) is sufficient — the multi-
   process warning ("Sharing a persistent job store among two or more processes
   will lead to incorrect scheduler behavior", Source 6) does NOT apply because the
   slack-bot scheduler is one process.
3. **MERGE/DELETE/UPDATE fail on streaming-buffer rows for ~up to 90 min** —
   "UPDATE or DELETE statement over table would affect rows in the streaming
   buffer, which is not supported" (Source 7). Since `historical_prices` is fed by
   `insert_rows_json` (streaming), a MERGE-based dedup would intermittently fail.
   **=> Prefer the existing app-level `(ticker, date)` dedup (already implemented
   in `data_ingestion._get_existing_price_dates`) over a BQ MERGE.**
4. **Partition-overwrite by date is the 2026 best-practice idempotent daily-load
   pattern** (Sources 4+5), but `historical_prices` is currently NOT partitioned
   (`migrate_backtest_data.py` — flat table). The cheapest correct fix that needs
   no schema migration is the existing insert-with-app-dedup path; partitioning is
   a nice-to-have follow-up, not a blocker for 44.1.
5. **yfinance is prototype-grade for automated daily batch** (Source 2) — must be
   wrapped in fail-open + exponential backoff; `yf.download()` bulk is less likely
   to trip the limiter than many `yf.Ticker().info` calls (Source 2 + search). The
   existing `data_ingestion.ingest_prices` already uses bulk `yf.download` in
   batches with per-batch try/except (`:103-111`) — good. Add backoff + smaller
   batch if 429s appear.
6. **Freshness must verify data ARRIVED, not just that the job ran** (Source 8 +
   dbt). The current freshness strip already keys on `MAX(ingested_at)` — correct
   in principle, but it was reading a table the writer never populated (the
   wrong-destination bug). Fixing the write closes that gap. Add a check that runs
   more often than the 26h SLA (Source 8: "check freshness MORE often than the SLA
   window").

### Consensus vs debate (external)

- **Consensus:** persistent jobstore for any job that must survive restart;
  idempotent loads via partition-overwrite or business-key MERGE; never trust
  "job ran" as a freshness proxy; yfinance unreliable at scale.
- **Debate / nuance:** MERGE vs partition-overwrite vs insert+dedup. The 2026
  sources lean partition-overwrite; but the streaming-buffer restriction (Source 7)
  makes any DML risky on a streaming-fed table. The project's existing app-level
  dedup sidesteps the debate entirely and is the lowest-risk choice for 44.1.

### Pitfalls (from literature)

- Adding a SQLAlchemyJobStore but pointing it at a path shared by >1 process =>
  duplicate/missed fires (Source 6). Keep it single-process, single SQLite file.
- Running a one-time MERGE backfill against the streaming-buffered table => "rows
  in streaming buffer" error (Source 7). Use the app-dedup insert path, or wait
  >90 min after the last stream, or use a load job.
- Assuming the daily job "works" because the process is up — orchestration success
  != data arrival (Source 8). Verify via the freshness endpoint after the fix.
- `coalesce` defaults to False (Source 1); on a persistent store a long outage
  could queue multiple missed daily fires. Set `coalesce=True` (already set in
  `register_phase9_jobs`, scheduler.py:798) so a recovered daemon fires once.

## Recommended fix approach (phase-44.1)

The staleness has **three independent root causes**; a durable fix must address
all three or the table re-rots:
(RC1) wrong destination — daily job writes `price_snapshots`, not `historical_prices`;
(RC2) wrong universe + wrong path — daily job is a 5-ticker stub via a partial-OHLCV
schema, while the real bulk path is `data_ingestion.ingest_prices`;
(RC3) the fire is lost to restart — in-memory jobstore + no slack-bot launchd
keepalive means `daily_price_refresh` (hour=1) has no memory of prior fires.

### (a) Destination + path fix — choose Option A' (delegate to ingestion path)

**Recommended: rewire `daily_price_refresh` to call the existing, correct
`backend/backtest/data_ingestion.py::ingest_prices()` against the real universe**
(`screener.get_sp500_tickers()`), rather than the 5-ticker `_production_fns`
stub that writes the wrong table/schema.

Why this over the literal "redirect the write to historical_prices" (Option A):
- `data_ingestion.ingest_prices` already writes the COMPLETE schema incl.
  `ingested_at` (the column freshness keys on), uses bulk `yf.download` with
  per-batch try/except, and is **already idempotent** on `(ticker, date)` via
  `_get_existing_price_dates` (avoids the streaming-buffer MERGE trap, Source 7).
- `price_snapshots` has NO other consumer (Finding 2) => abandoning it breaks
  nothing. (If operator prefers to keep a thin daily-close snapshot for some
  future use, leave the stub writing `price_snapshots` AND add the ingestion
  call; but the freshness/ backtest fix only needs the ingestion call.)
- Lowest net-new code; reuses the audited path the backtest already trusts.

Implementation sketch (daily job): fetch `tickers = get_sp500_tickers()`;
`start = today - 5d` (small rolling window so each daily run only pulls recent
bars; app-dedup skips already-present `(ticker,date)`); `end = today + 1d`;
`DataIngestion(...).ingest_prices(tickers, start, end)`. Keep the whole thing
fail-open (log + return counts) consistent with the existing job contract.

Do NOT use a BigQuery MERGE for dedup on this table: it is streaming-fed
(`insert_rows_json`) and DML over streaming-buffer rows fails for up to ~90 min
(Source 7). The app-level `(ticker,date)` skip is the correct idempotency
mechanism here (Source 5 "make retries safe").

### (b) Scheduler-firing fix — persistent jobstore + explicit UTC + keepalive

1. **Add a persistent jobstore** to the slack-bot `AsyncIOScheduler`
   (scheduler.py:196): `jobstores={"default": SQLAlchemyJobStore(url="sqlite:///"
   + <repo>/handoff/state/slackbot_jobs.sqlite)}`. This is THE documented fix for
   surviving restarts (Sources 1+6). Single-process => the multi-process hazard
   (Source 6) does not apply. Keep the SQLite file out of any path another process
   writes.
2. **Set an explicit timezone.** The phase-9 jobs use bare `hour=1` with no tz
   (scheduler.py:797) — they inherit the scheduler default (machine local). Pin
   `timezone=ZoneInfo("UTC")` (or America/New_York to match the digests) on the
   scheduler and/or per-job so the fire time is deterministic across DST and Mac
   tz. (Source 1: timezone via constructor arg.)
3. **Keep `coalesce=True` (already set) and raise misfire grace** for the daily
   job from 3600s toward e.g. 6-12h so a daemon that comes up mid-morning still
   fires the missed 01:00 tick once (Source 1: misfire only recoverable with a
   persistent store — so this only takes effect AFTER step 1).
4. **Add a slack-bot launchd keepalive** (no plist exists today — Finding 4):
   `com.pyfinagent.slack-bot` with `KeepAlive` + `RunAtLoad`, mirroring
   `com.pyfinagent.backend.plist`. This stops the process from staying down for
   days (the deepest cause of "never_run"). Combined with the persistent jobstore,
   a restart now replays the missed daily fire instead of forgetting it.
   (Local-only deployment doctrine: plist lives in `~/Library/LaunchAgents/`, not
   the repo — note it in the handoff.)

### (c) One-time 52-day backfill command

Use the SAME audited ingestion path (idempotent, safe to re-run — Source 5):

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate
python -c "
from datetime import date, timedelta
from backend.backtest.data_ingestion import DataIngestion
from backend.tools.screener import get_sp500_tickers
end = (date.today() + timedelta(days=1)).isoformat()
start = (date.today() - timedelta(days=60)).isoformat()  # 60d covers the 52d gap + margin
tickers = get_sp500_tickers()
n = DataIngestion().ingest_prices(tickers, start, end)
print('backfill inserted rows:', n)
"
```

(Confirm the exact `DataIngestion` constructor signature against
`backend/backtest/data_ingestion.py` before running — Q/A should verify it takes
no required args or pass the BQ client/project as that file expects.) Because
dedup is on `(ticker, date)`, re-running is safe and will insert only missing
bars. ~52d x ~500 tickers ≈ 26K rows — well within streaming tolerance; no
buffer/DML issue because it is insert-only.

### (d) Exact verification

1. **Freshness endpoint band off-red + age < 2 days:**
   ```bash
   curl -s http://127.0.0.1:8000/api/paper-trading/freshness \
     | python -m json.tool
   ```
   Assert `sources.historical_prices.band` in {"green","yellow"} (NOT "red") and
   `sources.historical_prices.last_tick_age_sec` < 172800 (2 days). (Canonical
   route per backend-api rule; `/api/observability/freshness` is the alias.)
2. **Row freshness directly in BQ** (data actually arrived, not just job ran —
   Source 8):
   ```sql
   SELECT MAX(ingested_at) AS latest_ingest,
          DATE_DIFF(CURRENT_DATE(), MAX(DATE(date)), DAY) AS bar_age_days,
          COUNT(*) AS rows_last_5d
   FROM `sunny-might-477607-p8.financial_reports.historical_prices`
   WHERE ingested_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
   ```
   Expect `latest_ingest` within the last day and a non-zero `rows_last_5d`.
3. **Job fires after a restart (proves RC3 fixed):** restart the slack bot,
   confirm `daily_price_refresh` has a non-null `next_run_time` immediately
   (persistent store), and after the next scheduled tick `GET /api/jobs/all`
   (or `/api/jobs/status`) shows `daily_price_refresh` status != "never_run".
   This is the live_check evidence shape for the masterplan step.

### Open items for the contract (validate, don't assume)

- Confirm `DataIngestion` constructor + `get_sp500_tickers` import path in the
  live tree before wiring (full-codebase-audit doctrine — grep all consumers).
- Decide whether any trade/decision gate hard-blocks on freshness band==red
  (Finding 3 says the live screener uses yfinance, so likely NOT — but the
  contract should state this explicitly so "fresh prices" is scoped to
  backtest-evidence + operator-confidence, not a literal trade unblock).
- Confirm whether the slack-bot scheduler module-level `run` functions are
  importable at module scope (required for SQLAlchemyJobStore serialization —
  Source: APScheduler "functions must be module-level imports when used with
  persistent jobstores"). The phase-9 jobs ARE module-level (`backend.slack_bot.
  jobs.<x>.run`), so this should hold; but `functools.partial(run, **prod_fns)`
  with closures from `_production_fns` factories is NOT directly serializable —
  **this is a real gotcha**: a SQLAlchemyJobStore cannot pickle the partial-wrapped
  closures. Mitigation: either (i) keep phase-9 jobs in the in-memory store and add
  the persistent store only for jobs with picklable top-level callables, or (ii)
  refactor `_production_fns` injection so the daily job resolves its fetch/write
  fns at call time from module-level functions (no closure capture). Flag this for
  the GENERATE phase — it materially shapes the implementation.

## Hard-blocker checklist + JSON envelope

Hard blockers — `gate_passed` is true only if all checked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (8 read in full)
- [x] 10+ unique URLs total incl. snippet-only (19 collected)
- [x] Recency scan (last 2 years) performed + reported (APScheduler 4.0, 2026 BQ
      idempotency posts, 2025 yfinance issues, Storage Write API Nov-2023)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (8 files: _production_fns,
      cycle_health, data_ingestion, cache, backtest_engine, autonomous_loop,
      scheduler, cron_dashboard_api/job_status_api + migrate_backtest_data schema)
- [x] Contradictions / consensus noted (MERGE vs partition-overwrite vs app-dedup)
- [x] All claims cited per-claim with URL or file:line

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 11,
  "urls_collected": 19,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "gate_passed": true
}
```

### Sources

1. https://apscheduler.readthedocs.io/en/3.x/userguide.html
2. https://medium.com/@trading.dude/why-yfinance-keeps-getting-blocked-and-what-to-use-instead-92d84bb2cc01
3. https://docs.cloud.google.com/bigquery/docs/managing-partitioned-tables
4. https://oneuptime.com/blog/post/2026-02-17-how-to-use-merge-statements-in-bigquery-for-upsert-operations/view
5. https://datalakehousehub.com/blog/2026-02-de-best-practices-04-idempotent-pipelines/
6. https://apscheduler.readthedocs.io/en/3.x/faq.html
7. https://big-data-demystified.ninja/2020/03/26/bigquery-error-update-or-delete-statement-over-table-would-affect-rows-in-the-streaming-buffer-which-is-not-supported/
8. https://montecarlo.ai/blog-data-freshness-explained/
9. https://apscheduler.readthedocs.io/en/master/versionhistory.html (recency scan)
10. https://github.com/agronholm/apscheduler/issues/465 (recency scan)
11. https://github.com/ranaroussi/yfinance/issues/2411 (yfinance 429, 2025)
12. https://github.com/ranaroussi/yfinance/issues/2518 (yfinance 429 persists, 0.2.61)
13. https://github.com/apache/airflow/issues/59408 (streaming-buffer DML corroboration)
