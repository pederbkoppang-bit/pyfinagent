---
step: phase-23.2.20
title: Cycle freshness BQ TIMESTAMP_DIFF type-coercion fix + silent-failure visibility
cycle_date: 2026-05-05
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_2_20.py'
research_brief: handoff/current/phase-23.2.20-external-research.md (also see phase-23.2.20-internal-codebase-audit.md)
---

# Contract — phase-23.2.20

## Hypothesis

User screenshot: Cycle segment of OpsStatusBar shows three circles —
heartbeat green, paper_trades gray (unknown), paper_snapshots gray
(unknown). Live `/api/paper-trading/freshness` returns
`{paper_trades:{band:"unknown",last_tick_age_sec:null,...},
paper_snapshots:{band:"unknown",last_tick_age_sec:null,...}}`.

**Root cause** at `backend/services/cycle_health.py:169`:
```sql
SELECT TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX({col}), SECOND) AS age FROM `{table}`
```
Both columns are STRING in BigQuery:
- `paper_trades.created_at`: STRING REQUIRED, sample `2026-05-01T18:02:39.679773+00:00`
- `paper_portfolio_snapshots.snapshot_date`: STRING NULLABLE, sample `2026-05-05`

BigQuery rejects with `BadRequest 400: Argument 2: Unable to coerce type
STRING to expected type TIMESTAMP`. `_bq_max_event_age` swallows the
exception at line 175 and returns None; the band shows "unknown".

**Compounding bug**: line 175 logs at `logger.debug()`. Default backend log
level is INFO (per `.claude/rules/backend-api.md`), so this BQ rejection
has been silently invisible for as long as the columns have been STRING.
A schema regression of any kind in the future will fail the same way.

Direct python+BigQuery test confirmed `SAFE.TIMESTAMP(MAX(col))` works
for both columns: paper_trades age=349672s (~4 days, matches yesterday's
audit "stale 05-01"); paper_snapshots age=69035s (~19h, consistent with
00:00 UTC snapshot today).

## Research-gate summary

Researcher (ab8e01334b8517a2a) returned `gate_passed: true`:
- 6 sources read in full via WebFetch (Medium SAFE.* / OWOX 2025 BQ
  timestamp guide / Secoda type casting / Reintech BQ error handling /
  Index.dev silent-failures / TDS BQ optimization)
- 16 URLs collected; 10 in snippet-only
- Recency scan 2024-2026 — no new BQ functions or deprecations
  affecting TIMESTAMP() / TIMESTAMP_DIFF in the window
- 5 internal files inspected; confirmed `bigquery_client.py:308`'s
  TIMESTAMP_DIFF is NOT the same bug (uses bound TIMESTAMP param)
- Key finding: SAFE.TIMESTAMP() returns NULL on malformed input rather
  than killing the query — preferred for monitoring queries that must
  not fail-loudly on a single bad row

## Immutable success criteria (verbatim — DO NOT EDIT)

1. `_bq_max_event_age` SQL wraps `MAX({col})` in `SAFE.TIMESTAMP(...)`.
2. The except clause at the swallowed-exception site logs at WARNING
   level (not DEBUG) so future schema regressions surface in normal
   backend logs.
3. Live `/api/paper-trading/freshness` returns non-null
   `last_tick_age_sec` and a band of `green`, `amber`, or `red` (not
   `unknown`) for both `paper_trades` and `paper_snapshots`.
4. Regression test `tests/services/test_freshness_query_shape.py`:
   - asserts the SQL string built by `_bq_max_event_age` includes
     `SAFE.TIMESTAMP(`
   - asserts the function returns the parsed age when the BQ client
     returns a row with `age=N`
   - asserts the function logs at WARNING (not DEBUG) when the BQ
     client raises
5. `python tests/verify_phase_23_2_20.py` exits 0.
6. `python -c "import ast; ast.parse(open(P).read())"` passes for the
   modified .py file.
7. Live BQ probe via the python client:
   `_bq_max_event_age(bq, "paper_trades", "created_at")` returns a
   positive int and `_bq_max_event_age(bq, "paper_portfolio_snapshots",
   "snapshot_date")` returns a positive int.

## Plan steps

1. `backend/services/cycle_health.py::_bq_max_event_age`:
   - Change SQL to
     `SELECT TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), SAFE.TIMESTAMP(MAX({time_col})), SECOND) AS age FROM \`{table}\``.
   - Bump the except-clause logger from `logger.debug` to `logger.warning`.
   - Add a one-line comment explaining the SAFE.TIMESTAMP wrapper and
     the bare-date approximation (snapshot_date='2026-05-05' parses
     to midnight UTC).
2. Regression test `tests/services/test_freshness_query_shape.py`:
   - Build a fake `bq` with `_pt_table` and `client.query` mocked.
   - Test 1: confirm SQL contains `SAFE.TIMESTAMP(`.
   - Test 2: confirm function returns the age value when the mock
     returns a row.
   - Test 3: monkey-patch the mock to raise; assert the warning was
     logged via `caplog`.
3. Verifier `tests/verify_phase_23_2_20.py`:
   - `ast.parse` modified file
   - assert `SAFE.TIMESTAMP(` in `cycle_health.py`
   - assert `logger.warning` (not `logger.debug`) in `_bq_max_event_age`'s
     except clause
   - assert test file exists with the 3 expected test names
4. Live verification step (operator-visible):
   restart backend after deploy, hit `/api/paper-trading/freshness`,
   assert `paper_trades.band != "unknown"` AND `paper_snapshots.band != "unknown"`.
5. Append `handoff/harness_log.md` AFTER Q/A PASS, BEFORE any masterplan flip.

## Out of scope

- Migration of the STRING columns to TIMESTAMP / DATE: separate effort
  with a backward-compat shim. The SAFE.TIMESTAMP wrapper is the
  surgical fix.
- Refactor of `_bq_max_event_age` to fail-loud (re-raise instead of
  swallow): out of scope; the warning-level log gives equivalent
  visibility without changing the freshness API's fail-open contract.
- Frontend tooltip enrichment: previously considered for this phase
  but redirected to BQ-only fix per user clarification ("we have
  unknown for two circles"). Tooltip improvements remain a phase-2
  candidate.

## Backwards compatibility

- SAFE.TIMESTAMP() returns NULL on malformed input (rather than
  failing the query); the function still returns None on failure,
  preserving the existing fail-open contract.
- Logger bump from DEBUG to WARNING is purely additive: more log
  output, no behavioral change for callers.

## References

- Researcher: `handoff/current/phase-23.2.20-external-research.md`,
  `handoff/current/phase-23.2.20-internal-codebase-audit.md`
- `backend/services/cycle_health.py:161-178`
- `backend/db/bigquery_client.py:295-324` (NOT the same bug)
- BQ docs: TIMESTAMP() / SAFE.TIMESTAMP() (OWOX 2025)
- dbt-fusion#599 — independent reproduction of the type mismatch
