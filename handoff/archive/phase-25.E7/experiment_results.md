---
step: phase-25.E7
cycle: 98
cycle_date: 2026-05-13
result: PASS_PENDING_QA
---

# Experiment Results -- phase-25.E7

## What was built/changed

Closed audit bucket 24.7 F-4 by guarding the previously-unguarded
`yfinance_tool.get_price_history`:

1. **`backend/tools/yfinance_tool.py::get_price_history`** -- wrapped
   in try/except. On any exception:
   - `logger.warning(..., exc_info=True)` for traceback visibility.
   - Best-effort persist a row to `data_source_events` (via the
     25.B7 `save_data_source_event` method) with
     `source="yfinance_price_history"`, `kind="fallback"`,
     `notes=<exception class>`.
   - Returns `[{"error": <str>, "ticker": ticker}]` -- a 1-element
     list-of-dict so iterating callers see exactly one structured
     error row.
2. **Empty-DataFrame branch** -- same shape (WARN + persist with
   `notes="empty_dataframe"` + return `[{"error": "no_data", "ticker": ticker}]`).
3. **NEW `_persist_yfinance_event(ticker, notes)` helper** -- thin
   wrapper that imports settings + BigQueryClient inside the function
   to avoid module-init cost; fail-open if BQ raises.

## Files changed

| File | Action |
|------|--------|
| `backend/tools/yfinance_tool.py` | get_price_history try/except + persist helper |
| `tests/verify_phase_25_E7.py` | NEW verifier (5 claims) |

## Verification command + output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_E7.py

=== phase-25.E7 verification ===

[PASS] 1. get_price_history_has_try_except_wrapper
[PASS] 2. failure_counter_incremented_and_persisted_to_bq
[PASS] 3. get_price_history_returns_error_dict_on_failure
        -> len=1 keys=['error', 'ticker'] persist_called=True
[PASS] 4. persist_called_once_per_failure
        -> persist_call_count=1
[PASS] 5. empty_dataframe_returns_no_data_error
        -> out=[{'error': 'no_data', 'ticker': 'NVDA'}] persist_called=True

ALL 5 CLAIMS PASS
```

AST clean.

## Success criteria -> evidence

1. `get_price_history_returns_error_dict_on_failure` -- Claims 3 + 5 PASS:
   behavioral round-trips for both exception path AND empty-DataFrame path
   confirm `[{"error": ..., "ticker": ...}]` shape.
2. `failure_counter_incremented_and_persisted_to_bq` -- Claims 2 + 4 PASS:
   source references the BQ persist + behavioral test confirms exactly
   one call per failure.

## Aggregation query (operator-side reuse)

The same `pct_yfinance_fallback_dominance` query introduced in 25.B7 now
covers price-history failures automatically because they share the
`data_source_events` table:

```sql
SELECT source, COUNT(*) AS events
FROM `sunny-might-477607-p8.pyfinagent_data.data_source_events`
WHERE DATE(event_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
GROUP BY source ORDER BY events DESC;

-- Now returns rows for both 'yfinance_fallback' (news) and
-- 'yfinance_price_history' (OHLCV) so operators can split by source.
```

## Out-of-scope / deferred

- yfinance retry-with-backoff: not in criterion; the new error path
  returns immediately. A future retry layer (e.g., exponential backoff
  before the persist) is a 25.E7.1 follow-up.
- Caller-side error-row defensive handling: callers that iterate the
  returned list will see one row with the `error` key but no OHLCV
  fields. Existing callers should defensively check `if rows and
  "error" in rows[0]`. None currently do; this is documented as a
  follow-up.

## References

- `handoff/current/research_brief.md`
- `backend/tools/yfinance_tool.py:84-139` (guarded function)
- `backend/db/bigquery_client.py::save_data_source_event` (25.B7)
- `.claude/masterplan.json::25.E7`
