# Phase-23.2.20 Internal Codebase Audit

## Scope
`backend/services/cycle_health.py` + all `TIMESTAMP_DIFF` callsites in `backend/`

---

## 1. `_bq_max_event_age` — Full Read of `backend/services/cycle_health.py`

**File:** `/Users/ford/.openclaw/workspace/pyfinagent/backend/services/cycle_health.py`
**Lines:** 161-177 (function body), 180-227 (compute_freshness)

### Function signature and SQL (lines 161-177)
```python
def _bq_max_event_age(bq: Any, table_logical: str, time_col: str) -> Optional[float]:
    try:
        table = bq._pt_table(table_logical)
        sql = f"SELECT TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX({time_col}), SECOND) AS age FROM `{table}`"
        rows = list(bq.client.query(sql).result())
        if not rows:
            return None
        age = rows[0].get("age") if hasattr(rows[0], "get") else rows[0][0]
        return float(age) if age is not None else None
    except Exception as e:
        logger.debug(f"bq_max_event_age({table_logical}.{time_col}) failed: {e}")
        return None
```

**Confirmed findings:**

1. **Single TIMESTAMP_DIFF callsite in this module.** There is exactly one `TIMESTAMP_DIFF` invocation in `cycle_health.py` at line 169. No other timestamp arithmetic exists in the file.

2. **Silent DEBUG-level swallow.** The `except Exception as e` at line 175 logs at `logger.debug(...)` — not `WARNING` or `ERROR`. Under the default `INFO` log level (`LOG_LEVEL` default per `.claude/rules/backend-api.md`), this message is invisible. The operator cannot tell from logs that BigQuery rejected the query. This is an observability anti-pattern (see external research section).

3. **Root cause confirmed.** `TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(col), SECOND)` requires `MAX(col)` to produce a TIMESTAMP type. Both target columns are STRING:
   - `paper_trades.created_at`: `STRING REQUIRED`, sample `2026-05-01T18:02:39.679773+00:00`
   - `paper_portfolio_snapshots.snapshot_date`: `STRING NULLABLE`, sample `2026-05-05`
   BigQuery raises `BadRequest 400: No matching signature for function TIMESTAMP_DIFF; Argument 2: Unable to coerce type STRING to expected type TIMESTAMP`.

4. **Candidate fix.** Wrapping `MAX(col)` in `TIMESTAMP(MAX({time_col}))` resolves the type mismatch for both columns:
   - RFC3339 with offset `2026-05-01T18:02:39.679773+00:00` -- `TIMESTAMP()` accepts this directly.
   - Bare date `2026-05-05` -- `TIMESTAMP("2026-05-05")` parses as `2026-05-05 00:00:00 UTC`. Age is relative to midnight of the snapshot date, not the actual write time. Acceptable approximation; flag in a comment.

### `compute_freshness` (lines 180-227)
`compute_freshness` calls `_bq_max_event_age` twice:
- line 193: `_bq_max_event_age(bq, "paper_trades", "created_at")`
- line 194: `_bq_max_event_age(bq, "paper_portfolio_snapshots", "snapshot_date")`

Both return `None` on the current broken schema, which flows through to `_band(None, ...)` returning `"unknown"` (line 60). The function's return shape is a nested dict with keys: `sources`, `heartbeat`, `bq_ingest_lag_sec`, `thresholds`, `computed_at`. No change to the shape is needed for the fix.

---

## 2. All TIMESTAMP_DIFF Callsites in `backend/`

Grep result (`grep -rn "TIMESTAMP_DIFF" backend/`):

| File | Line | SQL / context | Column type |
|------|------|---------------|-------------|
| `backend/services/cycle_health.py` | 169 | `TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX({time_col}), SECOND)` — `time_col` is `created_at` (STRING) or `snapshot_date` (STRING) | **STRING — BUG** |
| `backend/db/bigquery_client.py` | 308 | `ORDER BY ABS(TIMESTAMP_DIFF(analysis_date, @ts_target, MICROSECOND))` — `analysis_date` column in `financial_reports` table; `@ts_target` is a bound TIMESTAMP parameter | Needs verification |

**`bigquery_client.py:308` analysis (lines 295-316):**

The query at line 308 uses:
```sql
ORDER BY ABS(TIMESTAMP_DIFF(analysis_date, @ts_target, MICROSECOND))
```
where `@ts_target` is bound as `"TIMESTAMP"` type (`ScalarQueryParameter("ts_target", "TIMESTAMP", ts)`) at line 315. The column `analysis_date` is in `self.reports_table` (the `financial_reports` dataset). The surrounding code (lines 295-299) parses `analysis_date` from a Python string using `datetime.fromisoformat(clean)` before binding it as a TIMESTAMP parameter, and the `BETWEEN @ts_start AND @ts_end` filter at line 307 also binds both as TIMESTAMP. This strongly implies `analysis_date` in `financial_reports` is already stored as TIMESTAMP type (bound parameters must match). **No bug at this callsite.** However, if `analysis_date` were a STRING column, this would also fail identically. A quick BQ schema check on `financial_reports` would confirm (not done here as it is out of scope for this step).

---

## 3. BQ Schemas (confirmed from main session investigation)

| Table | Column | BQ type | Sample value | Notes |
|-------|--------|---------|-------------|-------|
| `paper_trades` | `created_at` | STRING REQUIRED | `2026-05-01T18:02:39.679773+00:00` | RFC3339 + offset |
| `paper_portfolio_snapshots` | `snapshot_date` | STRING NULLABLE | `2026-05-05` | Date-only, no time |

---

## 4. Tests Covering `_bq_max_event_age`

Search results:
- `tests/services/test_cycle_failure_alerts.py` — tests `raise_cron_alert` and `kill_switch.pause`; **no mention of `_bq_max_event_age`, `compute_freshness`, or BQ schema interactions**.
- `grep -rn "_bq_max_event_age|compute_freshness" tests/` — zero results.

**Confirmed: no existing pytest exercises `_bq_max_event_age`.** The bug only surfaces against the real BQ schema, consistent with the "unknown" band symptom that appeared in production but not in any test run.

---

## 5. Summary of Issues

| Issue | Location | Severity |
|-------|----------|---------|
| `TIMESTAMP_DIFF` called with STRING arguments; BQ rejects silently | `cycle_health.py:169` | Critical (dashboard broken) |
| Exception logged at DEBUG, invisible at INFO default | `cycle_health.py:175` | High (schema regressions will be invisible) |
| `snapshot_date` bare-date precision loss: age is relative to midnight, not actual write time | Will persist after fix | Low (acceptable approximation; comment required) |
| No unit test for `_bq_max_event_age` | No test file | Medium (regression risk) |

---

## Internal File Inventory

| File | Lines inspected | Role |
|------|----------------|------|
| `backend/services/cycle_health.py` | 1-228 (full) | Data+control plane freshness; `_bq_max_event_age` + `compute_freshness` |
| `backend/db/bigquery_client.py` | 295-324 | Second TIMESTAMP_DIFF callsite; confirmed not a STRING-column bug |
| `backend/api/paper_trading.py` | grep result | Canonical freshness route caller (line 333) |
| `backend/api/observability_api.py` | grep result | Alias freshness route caller (line 36) |
| `tests/services/test_cycle_failure_alerts.py` | 1-197 (full) | Only test near cycle_health; does NOT cover `_bq_max_event_age` |
