# Research Brief: phase-25.A7 -- Per-table Freshness Endpoint (5 Tables)

Tier: **moderate** (stated by caller).

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://www.metaplane.dev/blog/stay-fresh-four-ways-to-track-update-times-for-bigquery-tables-and-views | 2026-05-13 | authoritative blog (Metaplane) | WebFetch full | Four methods ranked: (1) MAX(time_col), (2) `__TABLES__`.last_modified_time, (3) INFORMATION_SCHEMA.last_change_time (schema-only), (4) MAX(_PARTITIONTIME). Method 1 is most accurate; method 2 is the fallback when no timestamp column exists. |
| https://www.metaplane.dev/blog/data-freshness-definition-examples | 2026-05-13 | authoritative blog (Metaplane) | WebFetch full | Freshness categories: real-time (<1s), near-real-time (<1min), hourly, daily (<24h), weekly/monthly. Financial reporting cadences align with daily (prices), quarterly (fundamentals), monthly (macro). |
| https://discuss.google.dev/t/information-schema-tables-monitoring-last-modified-time/125698 | 2026-05-13 | official forum (Google Developer) | WebFetch full | Confirmed SQL: `SELECT table_id, TIMESTAMP_MILLIS(last_modified_time) AS last_modified_time FROM project.dataset.__TABLES__`. `__TABLES__` is dataset-scoped; `INFORMATION_SCHEMA.last_change_time` is schema-changes only, not data changes. |
| https://eponkratova.medium.com/stale-data-detection-with-dbt-and-bigquery-dataset-metadata-662196cf9370 | 2026-05-13 | practitioner blog (Medium) | WebFetch full | Exact dbt pattern for `__TABLES__` freshness fallback: `SAFE_CAST(DATE_DIFF(current_date, date(last_modified_time), DAY) as integer)`. Per-table thresholds configured separately in sources.yml. Daily = pass if refreshed within 1 day. |
| https://docs.datahub.com/docs/managed-datahub/observe/freshness-assertions | 2026-05-13 | official docs (DataHub) | WebFetch full | Freshness assertion config: evaluation_schedule + change_window + change_source. Supported sources: Audit Log, Information Schema, Last Modified Column, High Watermark Column. Fail-state raises incident. No static green/amber/red; pass/fail binary at the assertion level. |
| https://www.conduktor.io/glossary/data-freshness-monitoring-sla-management | 2026-05-13 | practitioner blog (Conduktor) | WebFetch full | Tiered SLA urgency: real-time fraud=sub-second, operational=minutes, analytics=hourly/daily, strategic=weekly/monthly. Freshness check frequency should be <=50% of SLA window (check every 30min for a 1h SLA). |
| https://docs.slack.dev/block-kit/ | 2026-05-13 | official docs (Slack) | WebFetch full | Block Kit blocks: section (mrkdwn text), context (supplementary info), divider. Alert shape: `attachments[].blocks[section].text.mrkdwn`. Existing `backend/tools/slack.py::send_notification` already uses this exact shape. |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.elementary-data.com/post/data-freshness-best-practices-and-key-metrics-to-measure-success | practitioner blog | Content covered by Metaplane reads; no per-table thresholds extracted |
| https://uptrace.dev/blog/sla-slo-monitoring-requirements | practitioner blog | Generic SLO patterns; no per-table cadence specifics |
| https://www.montecarlodata.com/blog-data-freshness-explained/ | practitioner blog (Monte Carlo) | Per-table cadence described qualitatively: prices=daily, fundamentals=quarterly, macro=monthly -- confirms our derivation |
| https://www.siffletdata.com/blog/data-freshness | practitioner blog | Confirmed "threshold-based alerting" pattern; no specific thresholds |
| https://api.slack.com/incoming-webhooks | official Slack docs | Covered by Block Kit read |
| https://www.synq.io/blog/data-observability-guide | practitioner guide | General observability; no per-table specifics |
| https://icedq.com/data-observability | practitioner guide | General; no per-table specifics |
| https://medium.com/metaplane/stay-fresh-four-ways-to-track-update-times-for-bigquery-tables-and-views-5f0b09e8a04e | blog (Mirror of Metaplane) | Duplicate of fetched Metaplane URL |
| https://discuss.google.dev/t/information-schema-tables-monitoring-last-modified-time/125698/2 | forum reply | Included in thread fetch |
| https://issuetracker.google.com/issues/139273728 | Google issue tracker | last_modified_time feature request; confirms it is not in INFORMATION_SCHEMA.TABLES |

---

## Recency scan (2024-2026)

Search queries run:
1. `BigQuery INFORMATION_SCHEMA TABLE_STORAGE last_modified_time freshness monitoring 2026` (current-year frontier)
2. `per-table SLA freshness monitoring data quality expected cadence daily weekly monthly patterns 2025` (last-2-year window)
3. `data freshness SLA BigQuery __TABLES__ monitoring fallback no timestamp column` (year-less canonical)

**Result:** No new 2024-2026 literature supersedes the established patterns. Key 2026 finding: Slack Block Kit released new block types April 2026 (changelog entry), but the `attachments + section + mrkdwn` shape used in `backend/tools/slack.py` remains valid and is not deprecated. The `__TABLES__.last_modified_time` fallback pattern has been stable since BigQuery GA; Google Developer forums (2024-2026) confirm no replacement is available for data-level freshness without a timestamp column.

---

## Key findings

1. **Timestamp column mapping for all 5 tables**: All three historical tables (`historical_prices`, `historical_fundamentals`, `historical_macro`) are ingested by `DataIngestionService` which appends an `ingested_at` timestamp column on every row (data_ingestion.py:96, 188, 280). The `signals_log` table uses `recorded_at` (bigquery_client.py:398). `paper_portfolio_snapshots` uses `snapshot_date` (already covered by existing `compute_freshness`). All five tables have a queryable timestamp column -- no `__TABLES__` fallback required. (Source: internal code audit)

2. **CRITICAL dataset discovery -- all tables are in `financial_reports`**: `_pt_table()` resolves to `settings.gcp_project_id + "." + settings.bq_dataset_reports + "." + name` where `bq_dataset_reports = "financial_reports"` (settings.py:40). `DataIngestionService._table()` uses the same `settings.bq_dataset_reports` (data_ingestion.py:34-37). The historical tables (`historical_prices`, `historical_fundamentals`, `historical_macro`) live in `financial_reports`, NOT `pyfinagent_hdw`. The `signals_log` table is also in `financial_reports` (bigquery_client.py:398). `_bq_max_event_age` calls `bq._pt_table(table_logical)` which uses the same `financial_reports` dataset. Therefore: **all 5 new tables can be queried with the existing `_bq_max_event_age(bq, table_logical, time_col)` helper using `_pt_table()` resolution -- no new dataset resolver needed.** (Source: settings.py:40, data_ingestion.py:34-37, bigquery_client.py:486-487)

3. **Per-table `expected_max_age_sec` derivation**: Industry consensus (Metaplane, Conduktor, Monte Carlo) aligns with business cadence -- the SLA window should be set at the expected update interval, with warn at 1.5x and critical at 2x (already encoded in `WARN_RATIO` and `CRITICAL_RATIO` in cycle_health.py:40-41). Per-table intervals:
   - `historical_prices`: yfinance ingest runs nightly; US market data is T+1 day; SLA = 26h (93,600s) -- a small buffer over 24h to allow for ingestion lag
   - `historical_fundamentals`: quarterly reports; SLA = 95 days (8,208,000s)
   - `historical_macro`: FRED updates monthly; SLA = 35 days (3,024,000s) -- buffer over 30 days
   - `signals_log`: written every autonomous cycle; SLA = same as `cycle_interval_sec` (already the pattern for paper_trades)
   - `paper_portfolio_snapshots`: daily snapshot; SLA = 26h (93,600s) -- same as prices
   (Source: Metaplane cadence table + data_ingestion.py FRED_SERIES + autonomous cycle cadence)

4. **`_band` is already per-table capable**: `_band(age_sec, interval_sec)` is generic -- it takes an `interval_sec` argument. The current `compute_freshness` passes the single `cycle_interval_sec` for all sources. The fix is: pass the per-table expected interval instead of the shared cycle interval. No changes to `_band` itself needed. (Source: cycle_health.py:57-65)

5. **Existing `AlertDeduper` handles idempotency**: `backend/services/observability/alerting.py::raise_cron_alert_sync` is already wired into the system with a 5-minute window dedup (default `window_minutes=5`, `repeat_hours=1`, `consecutive_threshold=3`). The `should_fire(source, error_type)` key is `(source, error_type)` -- per-table alarms should use `source="cycle_health"` and `error_type=f"freshness_critical_{table_name}"` so they dedup per table independently. Critical-severity bypasses the `consecutive_threshold` filter (fires immediately) but still respects `repeat_hours=1`. (Source: alerting.py:51-90, 67-77)

6. **`compute_freshness` is the sole callee, not the caller**: Both the canonical route (`paper_trading.py::get_freshness`) and the alias (`observability_api.py::get_observability_freshness`) import and call `compute_freshness(bq, cycle_interval_sec)` directly. No `slack_fn` injection point exists today. The Slack alarm must be wired inside `compute_freshness` itself (or a thin wrapper called from `compute_freshness`), using `raise_cron_alert_sync`. (Source: observability_api.py:33-41, cycle_health.py:201)

7. **`__TABLES__` fallback is available but not needed**: The `__TABLES__.last_modified_time` SQL pattern (Google Developer forums + Metaplane) could be used if any table lacked an `ingested_at` column. Since all 5 tables have explicit timestamp columns, Method 1 (MAX(time_col)) is preferred -- it measures the most recent data row, not the most recent table write (which may include schema-only changes). (Source: Metaplane four-methods read)

8. **Worst-of-N vs per-table**: Industry pattern (Conduktor, Monte Carlo) is to expose per-table bands and let consumers aggregate. The Monte Carlo note that "a single upstream freshness breach could trigger downstream problems" argues for surfacing a derived `overall_band` = worst of all table bands. This maps to a simple `_worst_band(bands: list[str])` helper. (Source: Monte Carlo snippet, Conduktor read)

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/services/cycle_health.py` | 249 | `compute_freshness`, `_bq_max_event_age`, `_band` | Active; target for extension |
| `backend/services/observability/alerting.py` | 234 | `raise_cron_alert` / `raise_cron_alert_sync` + `AlertDeduper` | Active; will be called from `compute_freshness` |
| `backend/tools/slack.py` | 39 | `send_notification(webhook_url, message, metadata, alert_type)` | Active; already used by alerting.py |
| `backend/api/observability_api.py` | 80 | Alias `/api/observability/freshness` -> `compute_freshness` | Active; no changes needed |
| `backend/db/bigquery_client.py` | 940 | `_pt_table(name)` resolves to `financial_reports.{name}` | Active; line 486-487 |
| `backend/config/settings.py` | - | `bq_dataset_reports = "financial_reports"` | Active; line 40 |
| `backend/backtest/data_ingestion.py` | 360 | Historical table ingestion; `ingested_at` column written at lines 96, 188, 280 | Active; confirms timestamp column names |
| `backend/backtest/cache.py` | 48 | `_table(name)` also uses `_dataset` (same dataset) | Active; confirms dataset |
| `tests/verify_phase_25_A7.py` | - | Verification script (does not exist yet; must be created) | New |

---

## Consensus vs debate (external)

**Consensus**: Use MAX(timestamp_col) as the primary freshness signal (Method 1). Expose per-table bands with per-table expected intervals rather than a single shared interval. Alerting should be dedup-protected to avoid polling-loop flood. Overall health = worst-of-N.

**Debate**: Static thresholds vs anomaly-detection (DataHub, Monte Carlo). For pyfinagent the data cadences are deterministic (nightly price ingest, quarterly fundamentals, monthly macro, per-cycle signals) -- static `expected_max_age_sec` constants are appropriate; anomaly detection adds no value for predictable batch pipelines.

---

## Pitfalls (from literature)

1. **Single shared interval breaks multi-cadence tables**: Passing `cycle_interval_sec` (per-cycle, e.g., 24h) to `_band` for `historical_fundamentals` (quarterly) always returns green because age/interval is always <<1.0. Must use per-table intervals. (Source: cycle_health.py:57-65 + Metaplane cadence table)
2. **`_pt_table()` is dataset-scoped to `financial_reports`**: All five target tables resolve there. If the masterplan prompt says "pyfinagent_hdw", that is incorrect for the current dataset layout. The implementation must use `_pt_table()` as-is.
3. **Slack alarm flood in polling loop**: `compute_freshness` is called on every `GET /api/observability/freshness` request. Without dedup, a critical band would fire a Slack message on every poll. The `AlertDeduper.repeat_hours=1` window prevents this -- each `(source, error_type)` key fires at most once per hour unless severity is P0/critical (which fires every call per alerting.py:67-77). Use P1 for freshness critical band (not P0) to get repeat-hour protection.
4. **`snapshot_date` is a bare DATE string**: Existing `_bq_max_event_age` already handles this via `SAFE.TIMESTAMP(MAX(snapshot_date))` (cycle_health.py:168-177). Same wrapping applies if any historical table stores `ingested_at` as a DATE instead of TIMESTAMP -- but `data_ingestion.py` stores `datetime.now(timezone.utc).isoformat()` which is a full RFC3339 TIMESTAMP, so `SAFE.TIMESTAMP()` will coerce correctly.
5. **`historical_macro` date column**: The macro table has a `date` column (date_str = obs["date"] at data_ingestion.py:303). The `ingested_at` column is the right freshness signal (not `date`, which is the observation date and can be months old even in a fresh row).

---

## Application to pyfinagent (file:line anchors)

| Finding | File:line | Action |
|---------|-----------|--------|
| `_bq_max_event_age` signature (reuse) | cycle_health.py:161-198 | Call for 3 new historical tables + signals_log |
| `_band(age_sec, interval_sec)` signature (reuse) | cycle_health.py:57-65 | Pass per-table `expected_max_age_sec` instead of shared `cycle_interval_sec` |
| `compute_freshness` insertion point | cycle_health.py:201-248 | Extend `sources` dict with 4 new tables; add `_worst_band` + Slack alarm call |
| `raise_cron_alert_sync` import | alerting.py:185-219 | Import and call with `source="cycle_health"`, `error_type=f"freshness_critical_{table}"` |
| `_pt_table` dataset = `financial_reports` | bigquery_client.py:486-487 | All 5 tables resolve via `_pt_table()` -- no new resolver |
| `ingested_at` column -- historical_prices | data_ingestion.py:96 | Use as `time_col` for `_bq_max_event_age` |
| `ingested_at` column -- historical_fundamentals | data_ingestion.py:188 | Use as `time_col` |
| `ingested_at` column -- historical_macro | data_ingestion.py:280 | Use as `time_col` (NOT `date` -- that is observation date) |
| `recorded_at` column -- signals_log | bigquery_client.py:398 | Use as `time_col` for signals_log; table name is "signals_log" |
| `snapshot_date` column -- paper_portfolio_snapshots | cycle_health.py:215 | Already queried; keep existing entry |

---

## Per-table `expected_max_age_sec` recommendations

| Table | Logical name | Time column | Update cadence | expected_max_age_sec | Rationale |
|-------|-------------|-------------|---------------|---------------------|-----------|
| historical_prices | `historical_prices` | `ingested_at` | Nightly (US market T+1) | `93_600` (26h) | 24h + 2h buffer for ingestion lag; market closed weekends -- age may reach 72h; consider using `cycle_interval_sec` if it is > 24h |
| historical_fundamentals | `historical_fundamentals` | `ingested_at` | Quarterly (90 days) | `8_208_000` (95 days) | 90-day quarter + 5-day filing lag buffer |
| historical_macro | `historical_macro` | `ingested_at` | Monthly (FRED release) | `3_024_000` (35 days) | 30-day month + 5-day release lag buffer |
| signals_log | `signals_log` | `recorded_at` | Per autonomous cycle | `cycle_interval_sec` | Same cadence as paper_trades; reuse caller-provided interval |
| paper_portfolio_snapshots | `paper_portfolio_snapshots` | `snapshot_date` | Daily snapshot | `93_600` (26h) | Already in compute_freshness; use same interval as historical_prices |
| paper_trades | `paper_trades` | `created_at` | Per cycle | `cycle_interval_sec` | Already in compute_freshness; unchanged |

**Note on weekend gap**: `historical_prices` is sourced from yfinance; no trading on weekends means no new rows from Friday close to Sunday. The raw age can reach 72h+ without an actual staleness issue. The recommended approach: use `93_600` (26h) as the nominal interval but accept that `_band` will show amber/red over weekends. This is acceptable for an observability dashboard -- operators learn to expect it. Alternatively, set `expected_max_age_sec=259_200` (3 days) to eliminate weekend noise, trading off sensitivity.

---

## Files to modify

| File | Change type | What |
|------|-------------|------|
| `backend/services/cycle_health.py` | MODIFY `compute_freshness` | Add 4 new table queries; per-table intervals; `_worst_band` helper; `raise_cron_alert_sync` call on critical band |
| `tests/verify_phase_25_A7.py` | CREATE | Verification script per masterplan immutable verification command |

No changes required to: `observability_api.py` (picks up new shape automatically), `bigquery_client.py`, `alerting.py`, `tools/slack.py`.

---

## Verbatim Python signatures for new helpers

```python
# In cycle_health.py -- insert after WARN_RATIO / CRITICAL_RATIO constants

# Per-table expected maximum age in seconds.
# Historical tables have wildly different cadences than the per-cycle paper tables.
_TABLE_MAX_AGE_SEC: dict[str, float] = {
    "historical_prices":       93_600,     # 26h -- nightly ingest, T+1 US market
    "historical_fundamentals": 8_208_000,  # 95 days -- quarterly + filing lag
    "historical_macro":        3_024_000,  # 35 days -- monthly FRED + release lag
    # signals_log and paper_* use caller-provided cycle_interval_sec
}

def _worst_band(bands: list[str]) -> str:
    """Return worst band across a list of band strings.
    Priority order: red > amber > green > unknown.
    """
    order = {"red": 3, "amber": 2, "green": 1, "unknown": 0}
    if not bands:
        return "unknown"
    return max(bands, key=lambda b: order.get(b, 0))
```

```python
# New table queries to splice into compute_freshness() after the existing
# trade_age / snap_age lines (cycle_health.py:214-215):

hist_prices_age  = _bq_max_event_age(bq, "historical_prices",       "ingested_at")
hist_fund_age    = _bq_max_event_age(bq, "historical_fundamentals",  "ingested_at")
hist_macro_age   = _bq_max_event_age(bq, "historical_macro",         "ingested_at")
signals_age      = _bq_max_event_age(bq, "signals_log",              "recorded_at")
```

```python
# Per-table `sources` dict entries (replaces existing sources dict build):

sources = {
    "paper_trades": {
        "last_tick_age_sec": trade_age,
        "interval_sec": cycle_interval_sec,
        "ratio": (trade_age / cycle_interval_sec) if (trade_age and cycle_interval_sec) else None,
        "band": _band(trade_age, cycle_interval_sec),
    },
    "paper_portfolio_snapshots": {
        "last_tick_age_sec": snap_age,
        "interval_sec": _TABLE_MAX_AGE_SEC.get("historical_prices", cycle_interval_sec),  # 26h
        "ratio": (snap_age / 93_600.0) if snap_age is not None else None,
        "band": _band(snap_age, 93_600.0),
    },
    "historical_prices": {
        "last_tick_age_sec": hist_prices_age,
        "interval_sec": _TABLE_MAX_AGE_SEC["historical_prices"],
        "ratio": (hist_prices_age / _TABLE_MAX_AGE_SEC["historical_prices"])
                 if hist_prices_age is not None else None,
        "band": _band(hist_prices_age, _TABLE_MAX_AGE_SEC["historical_prices"]),
    },
    "historical_fundamentals": {
        "last_tick_age_sec": hist_fund_age,
        "interval_sec": _TABLE_MAX_AGE_SEC["historical_fundamentals"],
        "ratio": (hist_fund_age / _TABLE_MAX_AGE_SEC["historical_fundamentals"])
                 if hist_fund_age is not None else None,
        "band": _band(hist_fund_age, _TABLE_MAX_AGE_SEC["historical_fundamentals"]),
    },
    "historical_macro": {
        "last_tick_age_sec": hist_macro_age,
        "interval_sec": _TABLE_MAX_AGE_SEC["historical_macro"],
        "ratio": (hist_macro_age / _TABLE_MAX_AGE_SEC["historical_macro"])
                 if hist_macro_age is not None else None,
        "band": _band(hist_macro_age, _TABLE_MAX_AGE_SEC["historical_macro"]),
    },
    "signals_log": {
        "last_tick_age_sec": signals_age,
        "interval_sec": cycle_interval_sec,
        "ratio": (signals_age / cycle_interval_sec) if (signals_age and cycle_interval_sec) else None,
        "band": _band(signals_age, cycle_interval_sec),
    },
}
```

```python
# Slack alarm wiring -- append inside compute_freshness() after sources dict is built,
# before the return statement:

all_bands = [v["band"] for v in sources.values()]
overall_band = _worst_band(all_bands)

# Fire Slack alarm on any critical-band table (dedup via AlertDeduper)
if overall_band == "red":
    _fire_freshness_alarm(sources)

# Return updated payload
return {
    "sources": sources,
    "overall_band": overall_band,
    "heartbeat": { ... },  # unchanged
    "bq_ingest_lag_sec": bq_ingest_lag,
    "thresholds": {
        "warn_ratio": WARN_RATIO,
        "critical_ratio": CRITICAL_RATIO,
        "cycle_interval_sec": cycle_interval_sec,
    },
    "computed_at": _now_iso(),
}
```

```python
# Slack alarm helper -- add as module-level function in cycle_health.py:

def _fire_freshness_alarm(sources: dict) -> None:
    """Fire a Slack alert for any table in red band. Dedup via AlertDeduper."""
    try:
        from backend.services.observability.alerting import raise_cron_alert_sync
        for table_name, info in sources.items():
            if info.get("band") == "red":
                raise_cron_alert_sync(
                    source="cycle_health",
                    error_type=f"freshness_critical_{table_name}",
                    severity="P1",
                    title=f"Data freshness critical: {table_name}",
                    details={
                        "table": table_name,
                        "last_tick_age_sec": str(info.get("last_tick_age_sec")),
                        "interval_sec": str(info.get("interval_sec")),
                        "ratio": str(round(info["ratio"], 2)) if info.get("ratio") else "N/A",
                    },
                )
    except Exception as exc:
        logger.warning("_fire_freshness_alarm failed: %r", exc)
```

---

## Slack alarm shape

The existing `send_notification(webhook_url, message, metadata, alert_type)` in `backend/tools/slack.py:12` produces:

```json
{
  "attachments": [{
    "color": "#ffc107",
    "blocks": [{
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "*[P1] Data freshness critical: historical_prices*\n*table:* historical_prices\n*last_tick_age_sec:* 186200\n*interval_sec:* 93600\n*ratio:* 1.99\n*source:* cycle_health\n*severity:* P1\n*error_type:* freshness_critical_historical_prices"
      }
    }]
  }]
}
```

`alert_type="warning"` -> color `#ffc107` (amber). `raise_cron_alert` uses `alert_type="error"` when severity is in `_CRITICAL_SEVERITIES` or P1 (alerting.py:166) -- so P1 produces `alert_type="error"` -> color `#dc3545` (red). Correct for a freshness breach.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 read in full)
- [x] 10+ unique URLs total (17 collected including snippet-only)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (9 files inspected)
- [x] Contradictions / consensus noted (static vs anomaly-detection debate; weekend gap tradeoff)
- [x] All claims cited per-claim (not just listed in a footer)

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
