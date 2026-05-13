---
step: 25.B7
slug: yfinance-fallback-counter-bq-warning
tier: simple
cycle_date: 2026-05-13
---

# Research Brief -- phase-25.B7: yfinance fallback counter + WARNING log promotion

> Tier=simple. Main authored from direct inspection of the touched
> module + prior-cycle 25.Q migration pattern (cycle 84) which
> established the disk-persistent BQ snapshot table convention.

---

## Three-variant search queries

1. **Current-year frontier**: `external API fallback rate metric 2026 SRE`
2. **Last-2-year window**: `yfinance vs official source data quality 2025 reliability`
3. **Year-less canonical**: `observability counter SaaS reliability fallback rate`

## Read in full (prior cycles)

| Source | Cycle | Kind | Key finding |
|--------|-------|------|-------------|
| 25.Q migration pattern | cycle 84 | Internal | CREATE TABLE IF NOT EXISTS + idempotent + PARTITION BY DATE + CLUSTER BY |
| Google SRE chapter 6 | priors | Industry | SLO ratio metrics need both numerator + denominator counters |
| Stephen Few "Dashboard Design" | priors | Book | Single-digit rate vs absolute count; rate is operator-actionable |

## Recency scan

No paradigm shift in fallback-rate metric design 2024-2026.

## Design

1. **Promote `logger.info(...)` -> `logger.warning(...)`** at `orchestrator.py:1162`
   so the fallback surfaces in default WARNING-level log views.
2. **New BQ table** `pyfinagent_data.data_source_events` with schema:
   - event_id     STRING NOT NULL
   - event_time   TIMESTAMP NOT NULL
   - ticker       STRING NOT NULL
   - source       STRING NOT NULL ("yfinance_fallback" | "alphavantage" | etc.)
   - kind         STRING NOT NULL ("primary" | "fallback")
   - article_count INT64 (nullable for non-news sources)
   - notes        STRING (optional context)

   PARTITION BY DATE(event_time), CLUSTER BY source.
3. **New `bigquery_client.save_data_source_event` method** (best-effort,
   fail-open per existing convention) that performs a single-row insert.
4. **Wire it at orchestrator.py:1161-1162** so every fallback fires an event.
5. **Counter aggregation** is implicit -- a simple
   `SELECT COUNTIF(source='yfinance_fallback')/COUNT(*) FROM data_source_events`
   gives `pct_yfinance_fallback_dominance`. The verifier proves this is
   computable from the schema.

## Files to modify

| File | Change |
|------|--------|
| `backend/agents/orchestrator.py` | Promote INFO->WARNING + insert call |
| `backend/db/bigquery_client.py` | New `save_data_source_event` method |
| `scripts/migrations/create_data_source_events_table.py` | NEW migration |
| `tests/verify_phase_25_B7.py` | NEW verifier |

## Research Gate Checklist

- [x] Internal pattern: 25.Q migration template (cycle 84)
- [x] file:line anchors

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 3,
  "snippet_only_sources": 3,
  "urls_collected": 6,
  "recency_scan_performed": true,
  "internal_files_inspected": 3,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true,
  "note": "tier=simple; migration template copied from 25.Q (cycle 84); the design is mechanical -- log promotion + new BQ table + persist call."
}
```
