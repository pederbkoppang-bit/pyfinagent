---
step: phase-23.2.20
cycle_date: 2026-05-05
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_2_20.py'
---

# Experiment Results — phase-23.2.20

## Hypothesis recap

User screenshot showed two gray "unknown" circles on the Cycle segment
(paper_trades, paper_snapshots). Live `/api/paper-trading/freshness`
returned `last_tick_age_sec=null, band=unknown` for both. Forensic:
`backend/services/cycle_health.py:_bq_max_event_age` ran
`SELECT TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(col), SECOND)` against
columns that are STRING in BigQuery (`paper_trades.created_at` and
`paper_portfolio_snapshots.snapshot_date`). BigQuery rejected with
`Unable to coerce type STRING to expected type TIMESTAMP`. The function
swallowed the exception at `logger.debug` (silent at INFO default) and
returned None, so the band rendered "unknown" indefinitely.

## What was changed

### Fix A — SAFE.TIMESTAMP wrapper
`backend/services/cycle_health.py:161-188`:
- SQL changed to
  `SELECT TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), SAFE.TIMESTAMP(MAX({col})), SECOND) AS age FROM {table}`.
- `SAFE.TIMESTAMP()` accepts both RFC3339 strings (`paper_trades.created_at`
  = `2026-05-01T18:02:39.679773+00:00`) and bare dates
  (`paper_portfolio_snapshots.snapshot_date` = `2026-05-05`); the latter
  parses to midnight UTC (acceptable approximation for daily snapshots).
- `SAFE.*` returns NULL on malformed input rather than raising, preserving
  the fail-open contract for monitoring queries.

### Fix B — Silent-failure visibility
Same function's except clause: bumped from `logger.debug(...)` to
`logger.warning(...)`. Default backend log level is INFO, so any future
schema regression will surface in normal logs. Inline comment explains
why.

### Tests
- `tests/services/test_freshness_query_shape.py` — 5 tests, all pass:
  - `test_sql_uses_safe_timestamp_wrapper` — asserts SQL contains `SAFE.TIMESTAMP(MAX(`
  - `test_returns_age_on_successful_query`
  - `test_returns_none_on_empty_result`
  - `test_returns_none_when_age_is_null` — the SAFE.TIMESTAMP-NULL path
  - `test_failed_query_logs_at_warning_not_debug` — uses `caplog` to assert WARNING-level log
- `tests/verify_phase_23_2_20.py` — 2-check immutable verifier (regex-asserts SAFE.TIMESTAMP and `logger.warning` in `_bq_max_event_age`; asserts test names exist).

## Files modified / added

```
backend/services/cycle_health.py                       -- SAFE.TIMESTAMP wrapper + warning-level log
tests/services/test_freshness_query_shape.py           -- NEW, 5 regression tests
tests/verify_phase_23_2_20.py                          -- NEW, 2-check verifier
handoff/current/contract.md                            -- updated for phase-23.2.20
handoff/current/phase-23.2.20-external-research.md     -- researcher output
handoff/current/phase-23.2.20-internal-codebase-audit.md -- researcher output
```

## Verification (verbatim output)

```
$ source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_2_20.py
OK backend/services/cycle_health.py
OK tests/services/test_freshness_query_shape.py

phase-23.2.20 verification: ALL PASS (2/2)

$ PYTHONPATH=. pytest tests/services/test_freshness_query_shape.py -q
.....                                                                    [100%]
5 passed in 0.01s

$ PYTHONPATH=. pytest tests/services/test_cycle_failure_alerts.py \
                     tests/services/test_kill_switch_no_deadlock.py \
                     tests/services/test_sod_daily_roll.py \
                     tests/services/test_snapshot_upsert.py \
                     tests/db/test_tickets_db_no_fd_leak.py \
                     tests/api/test_pause_resume_timeout.py -q
26 passed, 1 warning in 14.35s

$ PYTHONPATH=. python -c "from backend.config.settings import get_settings; from backend.db.bigquery_client import BigQueryClient; from backend.services.cycle_health import _bq_max_event_age; bq = BigQueryClient(get_settings()); print('paper_trades age:', _bq_max_event_age(bq, 'paper_trades', 'created_at')); print('paper_portfolio_snapshots age:', _bq_max_event_age(bq, 'paper_portfolio_snapshots', 'snapshot_date'))"
paper_trades age: 350302.0
paper_portfolio_snapshots age: 69663.0
```

Live `compute_freshness` post-fix:
```json
{
  "sources": {
    "paper_trades": {"last_tick_age_sec": 350305.0, "ratio": 4.054, "band": "red"},
    "paper_snapshots": {"last_tick_age_sec": 69665.0, "ratio": 0.806, "band": "green"}
  },
  ...
}
```

The `paper_trades` red band correctly flags an underlying real problem
(no trades since 05-01 due to the cycle-hang issue addressed in
phase-23.2.18). Pre-fix this stale-trade signal was masked by
"unknown".

## Research-gate evidence

Researcher (ab8e01334b8517a2a) returned `gate_passed: true`:
- 6 sources read in full via WebFetch (Medium SAFE BigQuery, OWOX 2025
  timestamp guide, Secoda type casting, Reintech BQ error handling,
  Index.dev silent-failures, TDS BQ optimization).
- 16 URLs collected; 10 in snippet-only.
- Recency scan 2024-2026 — no breaking changes; dbt-fusion#599 is an
  independent reproduction of the same TIMESTAMP_DIFF type mismatch.
- 5 internal files inspected; confirmed `bigquery_client.py:308`'s
  TIMESTAMP_DIFF is NOT the same bug (uses bound TIMESTAMP param).
- Key external finding: SAFE.TIMESTAMP() returns NULL on malformed
  input rather than failing the query — preferred for monitoring
  queries that must not fail-loudly on a single bad row.
- Key internal finding: silent `logger.debug` at the swallowed-exception
  site was the observability failure; bug had been latent since the
  columns became STRING.

## Backwards compatibility

- `SAFE.TIMESTAMP()` returns NULL on malformed input rather than
  raising; `_bq_max_event_age` still returns None on failure,
  preserving the fail-open contract.
- `logger.warning` bump is purely additive: more log output, no
  behavioral change for callers or tests.
- No schema changes, no API shape changes, no UI changes.

## Honest disclosures

- **The "red" band on paper_trades is real, not a false positive.** The
  fix unmasks an underlying staleness (last paper_trade row is from
  2026-05-01, ~4 days old) caused by the cycle-hang issue addressed
  in phase-23.2.18. The red signal is correct and operator-actionable.
- **`SAFE.TIMESTAMP("YYYY-MM-DD")` parses to midnight UTC.** For
  `snapshot_date='2026-05-05'` the reported age is "since
  2026-05-05T00:00:00 UTC", not "since the actual snapshot write
  time". Acceptable approximation for daily snapshots; flagged in the
  inline comment.
- **Live backend was not restarted as part of this phase.** The fix is
  in code that the live process holds via the python module — uvicorn
  --reload picks it up on save. Operator should restart explicitly to
  guarantee freshness endpoint immediately reflects the fix; otherwise
  next module reload (or backend restart for any reason) suffices.
- **No migration of the STRING columns to TIMESTAMP/DATE.** Surgical
  fix only. A future phase could add a `created_at_ts: TIMESTAMP`
  column with a backfill, but that's a separate effort.
- **`bigquery_client.py:308` was checked and is NOT affected.** Its
  `TIMESTAMP_DIFF` operates on a column bound via `ScalarQueryParameter
  (..., 'TIMESTAMP', ts)` — both sides are TIMESTAMP. No fix needed.
