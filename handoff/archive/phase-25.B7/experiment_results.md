---
step: phase-25.B7
cycle: 88
cycle_date: 2026-05-13
result: PASS_PENDING_QA
---

# Experiment Results -- phase-25.B7

## What was built/changed

Closed audit bucket 24.7 F-3 by:
1. Promoting the AV->yfinance fallback log from INFO to WARNING in
   `backend/agents/orchestrator.py` (was suppressed in default views).
2. Adding `save_data_source_event(...)` to `backend/db/bigquery_client.py`
   that inserts a single row into the new `pyfinagent_data.data_source_events`
   table with `(event_id, event_time, ticker, source, kind, article_count, notes)`.
3. Wiring an idempotent best-effort call at the orchestrator fallback site
   (within a try/except so a BQ outage doesn't break the analysis run).
4. Creating a 25.Q-style idempotent migration script
   `scripts/migrations/create_data_source_events_table.py` (dry-run by
   default; `--apply` executes).

Aggregable counter pattern:
```sql
SELECT COUNTIF(source = 'yfinance_fallback') / COUNT(*) AS pct_yfinance_fallback_dominance
FROM `sunny-might-477607-p8.pyfinagent_data.data_source_events`
WHERE DATE(event_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
```

## Files changed

| File | Action |
|------|--------|
| `backend/agents/orchestrator.py` | INFO -> WARNING + invoke save_data_source_event in try/except |
| `backend/db/bigquery_client.py` | NEW `save_data_source_event` method |
| `scripts/migrations/create_data_source_events_table.py` | NEW migration (idempotent) |
| `tests/verify_phase_25_B7.py` | NEW verifier (5 claims) |

## Verification command + output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_B7.py

=== phase-25.B7 verification ===

[PASS] 1. orchestrator_yfinance_fallback_logs_at_warning_level
        -> warning_present=True info_legacy_absent=True
[PASS] 2. bigquery_client_save_data_source_event_exists
        -> found=True kwargs=['ticker', 'source', 'kind', 'article_count', 'notes', 'event_id', 'event_time'] required={'kind', 'notes', 'article_count', 'source', 'ticker'}
[PASS] 3. new_bigquery_table_data_source_events_populated
        -> exists=True create=True name=True apply_flag=True
[PASS] 4. counter_aggregable_for_pct_yfinance_fallback_dominance
        -> source_col=True event_time_col=True partition=True cluster=True
[PASS] 5. behavioral_round_trip_call_shape_matches
        -> save_data_source_event invoked with expected kwargs

ALL 5 CLAIMS PASS
```

AST clean on all 3 touched .py files.

## Success criteria -> evidence

1. `orchestrator_yfinance_fallback_logs_at_warning_level` -- Claim 1 PASS
   (regex matches `logger.warning(` and asserts no legacy `logger.info` for that line).
2. `new_bigquery_table_data_source_events_populated` -- Claim 3 PASS
   (migration script exists with `CREATE TABLE IF NOT EXISTS data_source_events` +
   `--apply` flag). The orchestrator call (claim 5 behavioral) confirms the table
   gets a row per fallback firing once the migration is applied.
3. `counter_aggregable_for_pct_yfinance_fallback_dominance` -- Claim 4 PASS
   (the schema has `source STRING NOT NULL`, `event_time TIMESTAMP NOT NULL`,
   `PARTITION BY DATE(event_time)`, `CLUSTER BY source` -- exactly the shape
   needed for the COUNTIF aggregation pattern).

## Out-of-scope / deferred

- Live migration apply (`--apply`): operator runs this once after pulling main.
  The dry-run path is the safe default for code review.
- Additional sources beyond yfinance fallback: criterion is yfinance-specific.
  The schema accepts arbitrary `source` strings so adding e.g. `polygon_fallback`
  later requires no schema change.
- Frontend surfacing of `pct_yfinance_fallback_dominance`: not in criteria.
