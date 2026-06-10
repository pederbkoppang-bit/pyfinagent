# live_check 53.3 — Data-stack elevation (BQ cost/perf + freshness/lineage)

**Date:** 2026-06-10. Measure-first ($0 dry-run); correctness-preserving; NO
DROP/DELETE/schema mutation. 30s fallback timeout preserved.

## Hot-query audit + partition/cluster gap (criterion 1)

The 3 hot historical tables (`financial_reports.historical_{prices,fundamentals,macro}`,
us-central1) are **NOT partitioned and NOT clustered** (researcher dry-run: `preload_prices`
scans the identical 112,351,601 bytes with vs without its `date` filter → a WHERE/date
filter cannot prune). The 90-99% lever (partition-by-date + cluster-by-ticker) requires
table recreation = **OPERATOR-GATED** (documented below). The autonomously-landable,
results-preserving lever is **column projection** (kill `SELECT *`).

## Optimization landed + BEFORE/AFTER bytes (criterion 2) — $0 dry-run, MY measurement

Two `SELECT *` reads of `historical_fundamentals` → explicit 12-column projection
(`backend/backtest/cache.py:153` preload + `:351` fallback). The 12 columns are the ONLY
ones any consumer reads (`historical_data.py` `.get()`s 10 + the `ticker` grouping key +
the `report_date` ORDER BY); the 4 dropped columns (`filing_date, ingested_at, market,
currency`) have **0** call sites anywhere in the backend (re-grep-proven this cycle).

```
$0 dry-run (QueryJobConfig dry_run=True, use_query_cache=False), historical_fundamentals:
  OLD  SELECT *        : 655,079 bytes
  NEW  12-col project  : 515,937 bytes
  DELTA                : -139,142 bytes  (-21.2%)   [no bytes billed]
```

**Honesty note:** the researcher's brief cited −41% using a 10-column set — but 2 of those
dropped columns ARE consumed, so a 10-col projection would CHANGE results. The
results-preserving projection is all **12** consumed columns → the honest, measured win is
**−21.2%** (I kept the 2 columns the over-prune would have dropped). Cost delta scales with
the byte reduction on every preload (500-ticker universe) + every fallback miss.

## Freshness / lineage check (criterion 2)

`GET /api/paper-trading/freshness` (recorded 2026-06-10): overall=red, but the
signal/price hot tables are **GREEN** — `historical_prices` green, `historical_fundamentals`
green, `signals_log` green, `paper_trades` green, `paper_portfolio_snapshots` green,
heartbeat green; only **`historical_macro` = red** (stale macro). Mechanism:
`cycle_health.compute_freshness` (MAX-timestamp-vs-now, `cycle_health.py:426/473`) — the
canonical dbt/Monte Carlo pattern.

**Lineage discrepancy (DOCUMENTED, NOT auto-fixed — operator follow-up):** `sortino.py:108`
reads `pyfinagent_data.historical_macro` (DGS3MO/DTB3 for the MAR), but the WRITER
(`data_ingestion.py`) + the freshness check + the cache `_table()` use
`financial_reports.historical_macro`. So Sortino reads a DIFFERENT dataset's macro table
than the one that is written/refreshed — which is consistent with the `historical_macro`
red band. Repointing Sortino to `financial_reports` would CHANGE its MAR input (a result
change) → operator-gated, NOT this cycle.

## DO-NO-HARM (criterion 3)

- Projection-only change: no WHERE / ORDER BY / LIMIT / timeout touched → byte-identical
  RESULTS. The 30s fallback timeout (`cache.py:351` query, `LIMIT 5`) is unchanged.
- Dropped columns proven unused (0 call sites; consumers use `.get(k, default)`).
  4 cache/fundamentals tests pass (no regression).
- NO DROP / NO unqualified DELETE / NO schema mutation / NO repartition. $0 (dry-run only).

## Operator-gated recommendations (documented, NOT landed)

1. **Partition + cluster the 3 hot historical tables** (`historical_{prices,fundamentals,
   macro}`) by their date column + cluster by `ticker` — the 90-99% bytes-scanned lever.
   Needs a re-runnable idempotent `scripts/migrations/*.py` (table recreation = schema
   mutation = operator approval).
2. **Fix the Sortino macro lineage** (repoint `sortino.py:108` to `financial_reports.
   historical_macro` OR backfill `pyfinagent_data.historical_macro`) — changes the MAR
   input, so operator-gated.
3. **Refresh `historical_macro`** (the red freshness band) — the macro daily refresh / FRED
   pull (operator/cron).

## OPERATOR TO CONFIRM

The −21.2% bytes reduction is a $0 dry-run estimate; confirm the cost delta on the next
real preload via `INFORMATION_SCHEMA.JOBS.total_bytes_billed`. The partition/cluster
migration (#1) is the big win and needs your approval (schema mutation).
