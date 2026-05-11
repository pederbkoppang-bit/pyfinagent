---
step: phase-23.2.1
title: Verify autonomous loop ran daily for 7+ days — research brief
tier: simple
date: 2026-05-07
researcher: researcher agent (merged external + Explore)
---

## Research: BigQuery daily snapshot verification + autonomous loop audit

### Queries run (three-variant discipline)

1. Current-year frontier: `BigQuery MERGE upsert idempotency daily snapshot tables 2026`
2. Last-2-year window: `BigQuery snapshot table daily pipeline failure modes gaps duplicate rows timezone 2025`
3. Year-less canonical: `post-deployment validation daily cron jobs scheduled job ran every day monitoring`
4. Recency / freshness: `data freshness SLO no-gap assertions time-partitioned tables BigQuery 2025 2026`

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://oneuptime.com/blog/post/2026-02-17-how-to-use-merge-statements-in-bigquery-for-upsert-operations/view | 2026-05-07 | Blog (2026) | WebFetch | "MERGE is inherently idempotent: running it with the same source data twice results in the same target state." |
| https://oneuptime.com/blog/post/2026-02-17-how-to-implement-idempotent-data-pipelines-in-gcp-to-handle-retry-safe-processing/view | 2026-05-07 | Blog (2026) | WebFetch | "A pipeline is idempotent when repeated executions with the same input produce the same output." Partition-level overwrite + MERGE are the two primary BigQuery patterns. |
| https://dev.to/cronmonitor/how-to-monitor-cron-jobs-in-2026-a-complete-guide-28g9 | 2026-05-07 | Blog (2026) | WebFetch | "Dead man's switch pattern": alert when expected success signal does NOT arrive. "Cron jobs fail silently by design." Exit-code capture + start-AND-completion signals are the recommended instrumentation. |
| https://medium.com/@ankitraj.srivastava15/fixing-duplicate-rows-in-bigquery-while-merging-daily-snapshot-tables-4c075a66d086 | 2026-05-07 | Blog/Medium (Mar 2026) | WebFetch | Dedup via `ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY ingestion_ts DESC) WHERE rn=1` before MERGE. Confirms MERGE alone is insufficient when source already has duplicates. |
| https://docs.cloud.google.com/bigquery/docs/table-snapshots-intro | 2026-05-07 | Official Google Cloud docs | WebFetch | BQ native snapshots: read-only, 7-day time-travel window, no built-in freshness SLO or gap-detection. Distinctly different from user-maintained daily snapshot tables. |
| https://www.metaplane.dev/blog/stay-fresh-four-ways-to-track-update-times-for-bigquery-tables-and-views | 2026-05-07 | Industry blog (Metaplane) | WebFetch | Four freshness tracking methods: MAX() on timestamp column; `__TABLES__` metadata; INFORMATION_SCHEMA; MAX(_PARTITIONTIME) for partitioned tables. None natively detect gaps — gap detection requires date-series joins in user space. |
| https://betterstack.com/community/comparisons/cronjob-monitoring-tools/ | 2026-05-07 | Industry blog | WebFetch | Missed-run detection: heartbeat-based "dead man's switch." Performance anomaly detection ("taking longer than usual") is a secondary signal. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://cronitor.io/guides/cron-jobs | Guide/2026 | Snippet summary sufficient; core concepts covered by dev.to source |
| https://hevodata.com/learn/bigquery-upsert/ | Blog | Covered by oneuptime MERGE source |
| https://docs.cloud.google.com/bigquery/docs/partitioned-tables | Official docs | Freshness tracking covered by Metaplane source |
| https://docs.cloud.google.com/bigquery/docs/scheduling-queries | Official docs | Not directly applicable (BQ scheduled queries, not APScheduler) |
| https://www.owox.com/blog/articles/bigquery-partitioned-tables | Blog/2025 | Covered by Metaplane source |
| https://medium.com/google-cloud/creating-bigquery-table-snapshots-dynamically-c3d14ccd368a | Medium | Native snapshots, not custom daily tables |
| https://hevodata.com/learn/what-is-bigquery-snapshot/ | Blog | Covered by official docs source |
| https://earezki.com/ai-news/2026-04-25-i-built-a-cron-job-monitoring-api-in-a-weekend/ | Community | Lower authority; core concepts covered |
| https://uptimerobot.com/knowledge-hub/cron-monitoring/cron-job-guide/ | Blog | Duplicate coverage |
| https://domain-monitor.io/blog/how-to-monitor-cron-jobs/ | Blog | Duplicate coverage |

---

### Recency scan (2024-2026)

Searched for 2025-2026 literature on BigQuery MERGE idempotency, daily snapshot pipeline failure modes, cron job monitoring, and data freshness SLOs.

Findings from the 2024-2026 window:
- **Mar 2026 (Medium)**: Duplicate-row deduplication pattern for BigQuery daily snapshot tables — directly applicable to pyfinagent's phase-23.1.18 problem class (MERGE is insufficient when source already has duplicates; ROW_NUMBER pre-dedup is required).
- **Feb 2026 (OneUptime blog series)**: BigQuery MERGE idempotency + idempotent GCP pipeline patterns — confirms the MERGE-on-natural-key pattern is current best practice; BigQuery removed DML concurrency limits in 2026.
- **Feb 2026 (Dev.to/CronMonitor)**: Dead-man's-switch pattern for cron monitoring — the most current authoritative guide for post-deployment verification of daily scheduled jobs.
- **No new arXiv or peer-reviewed papers** on data-freshness SLOs or no-gap assertions for time-partitioned tables were found in the 2024-2026 window. The freshness-as-a-test-case concept remains in practitioner blogs, not academic literature.

No findings in the 2024-2026 window that supersede the canonical sources above; all complement.

---

### Key findings

**External:**

1. MERGE on natural key is idempotent by construction — "running it with the same source data twice results in the same target state." (Source: OneUptime MERGE blog 2026, https://oneuptime.com/blog/post/2026-02-17-how-to-use-merge-statements-in-bigquery-for-upsert-operations/view)

2. BQ native snapshots provide no freshness SLO and no gap-detection. For custom daily tables, gap detection requires user-space date-series queries (generate a date spine, left-join against actual rows, look for NULLs). (Source: Google Cloud docs https://docs.cloud.google.com/bigquery/docs/table-snapshots-intro + Metaplane blog)

3. The dead-man's-switch pattern is the practitioner standard for detecting missed cron runs: alert when a success signal does NOT arrive within a grace period, rather than polling for failures. The BQ verification query (`GROUP BY DATE(snapshot_date) ... expect ~9 rows, no gaps`) is a post-hoc version of this pattern. (Source: Dev.to/CronMonitor 2026)

4. Common silent-failure modes for daily snapshot pipelines: (a) partial-day rows from a crash mid-cycle, (b) duplicate rows from retry-on-error without idempotency, (c) timezone bugs if the scheduler fires in local time but the date key is computed in UTC, (d) streaming-buffer conflicts causing insert drops that are not surfaced as exceptions. (Sources: OneUptime idempotency blog 2026; Medium/Ankit Raj Mar 2026)

5. For freshness verification, `MAX(timestamp_col)` on the table is more reliable than `__TABLES__` metadata (which only reflects schema changes). For a STRING-keyed snapshot table, the equivalent is `MAX(snapshot_date)` or a date-series gap query. (Source: Metaplane 2026)

---

### Internal code inventory

| File | Lines inspected | Role | Status |
|------|-----------------|------|--------|
| `backend/db/bigquery_client.py` | 673-713 | `save_paper_snapshot` — canonical write path to `paper_portfolio_snapshots` | MERGE on `snapshot_date`; idempotent since phase-23.1.18 |
| `backend/db/bigquery_client.py` | 715-724 | `get_paper_snapshots` — reader; ORDER BY `snapshot_date` DESC | Active reader |
| `scripts/migrations/migrate_paper_trading.py` | ~70 | Schema definition for `paper_portfolio_snapshots` | `snapshot_date` is **STRING** ("REQUIRED") |
| `backend/services/paper_trader.py` | 427-455 | `save_daily_snapshot` wrapper — computes snap dict, calls `bq.save_paper_snapshot` | `snapshot_date` set via `datetime.now(timezone.utc).strftime("%Y-%m-%d")` |
| `backend/services/paper_trader.py` | 488-524 | `adjust_cash_and_mtm` — second `save_daily_snapshot` call site | Triggered by manual cash adjustments |
| `backend/services/autonomous_loop.py` | 443-452 | Step 8 in `run_daily_cycle` — calls `trader.save_daily_snapshot` on the success path | Only triggered if prior steps do not raise |
| `backend/services/autonomous_loop.py` | 53-62 | `run_daily_cycle` docstring confirms step 7 = "Save snapshot" | Canonical cycle steps |
| `backend/api/paper_trading.py` | 35, 901-908, 911-917 | APScheduler `paper_trading_daily` job — `cron` trigger at `settings.paper_trading_hour` | Confirmed scheduler job ID and cron type |
| `handoff/cycle_history.jsonl` | last 16 entries | Ground truth for which UTC days the loop ran | See table below |
| `handoff/archive/phase-23.1.18/contract.md` | all | Phase that introduced MERGE dedup for `paper_portfolio_snapshots` | MERGE on `snapshot_date` confirmed; `save_paper_snapshot` is idempotent since this phase |

---

### cycle_history.jsonl — completed cycle dates (last 16 entries)

| UTC Date | Cycle ID | Status | Notes |
|----------|----------|--------|-------|
| 2026-04-20 | 5e436733 | completed | 18:00 UTC |
| 2026-04-21 | dc0172e6 | completed | 18:00 UTC |
| 2026-04-24 | 3d4a06bf | completed | 18:00 UTC |
| 2026-04-24 | 14120cd3 | completed | 17:50 UTC (second run same day) |
| 2026-04-26 | 51d05189 | completed | 16:35 UTC |
| 2026-04-26 | f3109df7 | **error** | 21:08 UTC |
| 2026-04-26 | 0e8c4a20 | **error** | 21:11 UTC |
| 2026-04-26 | a54a21fc | completed | 21:16 UTC, 5 trades |
| 2026-04-26 | d609d781 | completed | 23:43 UTC |
| 2026-04-27 | 83122f72 | completed | 14:53 UTC |
| 2026-04-27 | c567e422 | completed | 15:15 UTC |
| 2026-04-27 | 6007f728 | completed | 18:00 UTC |
| 2026-04-28 | 7a6905f7 | completed | 18:00 UTC |
| 2026-04-29 | 0cc2cd67 | completed | 18:00 UTC |
| 2026-05-05 | 435644e4 | **running** (stale) | 18:00 UTC — status never flipped to completed |
| 2026-05-06 | 6bdc99a6 | completed | 18:00 UTC |

**Gaps visible**: 2026-04-22, 2026-04-23, 2026-04-25 have no cycle entries. 2026-04-30 through 2026-05-04 have no cycle entries. Today (2026-05-07) has no entry yet. These are weekday gaps and the verification query covers a 9-day window ending today — QA must confirm via BQ whether snapshots actually exist for each calendar day (weekend gaps are expected for market-aware loops) or whether the 7+ day assertion can be satisfied by the available dates.

**Important**: `cycle_history.jsonl` records loop execution, NOT BQ snapshot writes directly. A completed cycle writes a snapshot at Step 8 of `run_daily_cycle` (autonomous_loop.py:443-452). Error-status cycles may not have reached Step 8 and may not have written a snapshot.

---

### Snapshot_date column type — confirmed

`snapshot_date` is **STRING** (schema: `bigquery.SchemaField("snapshot_date", "STRING", mode="REQUIRED")`), stored as `"%Y-%m-%d"` formatted UTC date (`scripts/migrations/migrate_paper_trading.py:70`; `paper_trader.py:443`).

The verification command uses `PARSE_DATE('%Y-%m-%d', snapshot_date)` to convert to DATE for the range filter — this is correct given the STRING type. The query will work as written.

**MERGE key**: The MERGE in `save_paper_snapshot` (`bigquery_client.py:695-712`) merges on `T.snapshot_date = S.snapshot_date`. Since `snapshot_date` is a UTC date string and the cron fires daily at `paper_trading_hour` in ET, there is a timezone-boundary risk: if `paper_trading_hour` is set to, say, 18 in ET and the server runs in UTC, `datetime.now(timezone.utc).strftime("%Y-%m-%d")` would produce the correct UTC date, but if the scheduler fires in local time (ET) and UTC is behind or ahead, the date string could be off by one. Current code uses `timezone.utc` explicitly, so this is safe.

---

### Write path summary

1. **APScheduler** (`paper_trading.py:_add_scheduler_job`) fires `run_daily_cycle` at `paper_trading_hour:00` on a `cron` trigger — job ID `paper_trading_daily`.
2. `run_daily_cycle` (`autonomous_loop.py:53+`) executes 10 steps. Step 8 (line 448) calls `trader.save_daily_snapshot(...)`.
3. `save_daily_snapshot` (`paper_trader.py:427-455`) builds the snap dict with `snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")` and calls `self.bq.save_paper_snapshot(snap)`.
4. `save_paper_snapshot` (`bigquery_client.py:673-713`) runs a MERGE on `snapshot_date` — idempotent since phase-23.1.18.
5. Additional call site: `adjust_cash_and_mtm` (`paper_trader.py:524`) also calls `save_daily_snapshot` for manual cash corrections — also idempotent by MERGE.

There is **no plain INSERT** path remaining for `paper_portfolio_snapshots`. All writes go through `save_paper_snapshot` which uses MERGE.

---

### Idempotency status

**Confirmed idempotent.** Phase-23.1.18 replaced a plain INSERT with MERGE on the natural key (`snapshot_date`). Multiple writes in the same UTC day will overwrite the existing row, not produce duplicates. The dedup cleanup script (`scripts/cleanup_phase_23_1_18.py`) handled the pre-existing duplicates. No residual plain-INSERT path exists.

---

### Pitfalls for QA to watch

1. **Stale "running" entry**: `cycle_id 435644e4` (2026-05-05) has status `running` in `cycle_history.jsonl`. If Step 8 executed before the crash/hang, a snapshot row for `2026-05-05` may exist in BQ. If Step 8 did not execute, no snapshot exists for that date. The BQ verification query will reveal which.

2. **Weekend/holiday gaps**: 2026-04-22 (Wed) and 2026-04-23 (Thu) have no cycle entries — these are market days. 2026-04-30 through 2026-05-04 also have no entries (some are weekdays). These gaps may or may not be acceptable per the step definition. The verification query covers `DATE_SUB(CURRENT_DATE(), INTERVAL 9 DAY)` = 2026-04-28. Based on `cycle_history.jsonl`, 2026-04-28 (Tue) and 2026-04-29 (Wed) have completed cycles; 2026-04-30 through 2026-05-04 do not — only 2 rows would exist in that 9-day window without 2026-05-05 and 2026-05-06. The "expect ~9 rows" assertion may fail on the current data.

3. **Error cycles**: Two error-status cycles on 2026-04-26 (f3109df7, 0e8c4a20) — whether they produced partial snapshots is unknown without BQ inspection.

4. **`snapshot_date` uniqueness**: The MERGE guarantees at most one row per date, but the verification query COUNTs per date, so each row should count as 1. If the verification returns < 9 distinct dates, it is a real gap, not a MERGE artifact.

---

### Consensus vs debate (external)

**Consensus**: MERGE on natural key is the correct idempotency mechanism for daily snapshot tables in BigQuery. All 2026 sources agree. No debate on this point.

**Debate**: Gap detection method. Some sources recommend MAX(timestamp) + staleness threshold; others recommend date-spine left-join for explicit gap enumeration. The verification query uses a GROUP BY COUNT approach, which is equivalent to the date-spine approach but requires the caller to manually inspect the result for gaps. The query is fit for purpose.

---

### Application to pyfinagent

| Finding | Maps to | File:line |
|---------|---------|-----------|
| MERGE idempotency confirmed | `save_paper_snapshot` uses MERGE since phase-23.1.18 | `bigquery_client.py:695-712` |
| `snapshot_date` is STRING not DATE | Verification query needs `PARSE_DATE` — already present in immutable verification | `scripts/migrations/migrate_paper_trading.py:70` |
| Write is on Step 8 success path only | Cycles that error before Step 8 produce no snapshot | `autonomous_loop.py:448-452` |
| `cycle_history.jsonl` gap 2026-04-30 to 2026-05-04 | Only 4-5 snapshot rows likely exist in the 9-day window | `handoff/cycle_history.jsonl` last 16 entries |
| UTC-safe date computation | `timezone.utc` used in `save_daily_snapshot` — no timezone bug | `paper_trader.py:443` |

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 fetched in full)
- [x] 10+ unique URLs total (17 URLs collected: 7 read in full + 10 snippet-only)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (bigquery_client.py, paper_trader.py, autonomous_loop.py, paper_trading.py API, migrate_paper_trading.py, cycle_history.jsonl, phase-23.1.18 contract)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim (not just listed in a footer)

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "gate_passed": true
}
```
