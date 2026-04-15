# Research Gate -- Phase 4.2.4 BQ Durable Persistence Scaffold for `signals_log`

**Step:** Phase 4.2.4 BQ durable persistence scaffold. Adds the
write-path from `SignalsServer._append_signal_history` to a durable
`signals_log` BigQuery table, plus an idempotent migration script to
create the table.

**Scope (this cycle):** publish-event write path only. Outcome-update
path (track_signal_accuracy -> BQ DML) and warm-load read path are
deferred to a follow-up cycle once the streaming-buffer trade-offs
are resolved (see Category 3 below).

**Path taken:** Primary researcher-subagent path skipped per
`.claude/context/known-blockers.md` and the 2026-04-14-2026 +
2026-04-14-2245 session logs -- both `researcher` and
`general-purpose` subagents are flaky due to `Stream idle timeout -
partial response received` on web-heavy briefs. Fell back to
in-session WebSearch: 7 queries across 7 topic categories,
70+ URLs returned, all 7 categories covered with margin.

## Categories

### 1. BigQuery `insert_rows_json` vs DML INSERT vs Storage Write API

Rows written via the legacy `tabledata.insertAll` streaming API
(which is what `client.insert_rows_json` calls under the hood) sit
in a streaming buffer for up to 30-90 minutes before they become
queryable for DML. During that window, UPDATE / DELETE / MERGE /
TRUNCATE on those rows fails with the "rows in streaming buffer"
error. The Storage Write API (gRPC) does not have this restriction
and is also cheaper (no 1KB-per-row minimum, ~50% cheaper, lower
latency), but the Python client is bare-metal and requires protobuf
schemas + protoc.

**Lock-in:** use `insert_rows_json` for this cycle. It matches the
`save_outcome` and `save_report` precedent in `BigQueryClient` and
keeps zero new dependencies. The Storage Write API migration is a
separate Phase 4.2.4-follow-up. The streaming buffer DML restriction
is the reason we are NOT doing the outcome-update path this cycle.

### 2. BigQuery time-series partitioning + clustering best practice

For append-only time-series tables, partition on a DATE/TIMESTAMP
column so partition pruning kicks in on common WHERE filters.
Cluster on the dimension(s) most often used in WHERE clauses --
typically a categorical key like `ticker`. Daily partitions are
appropriate when the per-day row volume is non-trivial; for very
sparse data with a wide date range, monthly or yearly partitioning
plus clustering on the partitioning column performs better.

**Lock-in:** partition `signals_log` by ingestion `created_at`
TIMESTAMP at DAY granularity. Cluster by `ticker`. signal volume
will be small (<1000/day expected) so DAY granularity may be
oversized, but it matches the existing `paper_trades` /
`paper_portfolio_snapshots` convention (those tables don't actually
declare partitioning either, so we set the precedent here).

### 3. BigQuery DML UPDATE on streaming-buffer rows -- the 90-min trap

UPDATE / DELETE on rows that are still in the streaming buffer
fails with a hard error ("UPDATE or DELETE statement over table
would affect rows in the streaming buffer, which is not supported").
Buffer flush is unpredictable: typical 30 minutes, can extend to
90+ minutes during quiet periods. Mitigations:

1. Wait and retry with exponential backoff -- not viable for
   foreground request paths.
2. Switch to Storage Write API -- a separate cycle (see Category 1).
3. Use append-only event-log design: write a NEW row at outcome
   time, query computes latest state via window function -- changes
   the read path.
4. Defer the update path entirely until a follow-up cycle.

**Lock-in:** option (4) for this cycle. The in-memory
`_signals_by_id` dict is the canonical source of truth during a
process lifetime; BQ is the durable read-after-restart path. This
cycle ships only the publish-event write; outcome updates stay
in-memory until a future cycle moves to Storage Write API or
event-log design.

### 4. Quant trading signal log schema design

Production signal-log schemas in trading systems follow append-only
event-store patterns: NEW row per event (publish, fill, amend,
exit), immutable, queryable forensically. The schema captures
what / when / by whom / before+after state. Cryptographic chaining
or signed batches add tamper-evidence for regulatory contexts;
out of scope for this cycle but the append-only foundation is the
prerequisite.

**Lock-in:** `signals_log` schema mirrors the in-memory record
shape created by `_append_signal_history` -- 17 fields total,
covering signal identity (signal_id, ticker, signal_type, date),
prediction state (confidence, factors_json, entry_price, timestamp),
and outcome placeholders (outcome, scored, hit, exit_price,
exit_date, forward_return_pct, holding_days). For this cycle the
outcome fields are written as their initial published values
(outcome="pending", scored=False, all six outcome metrics NULL).
A subsequent cycle will append a SECOND audit row with the
post-scoring state, never UPDATE the original.

### 5. Idempotent BigQuery CREATE TABLE migration in Python

The `google-cloud-bigquery` client's idiomatic idempotent migration
pattern is `try: client.get_table(ref) except NotFound: client.create_table(table)`.
The existing `scripts/migrations/migrate_paper_trading.py` uses
the broader `except Exception:` form, which the new migration will
match for consistency. All four existing migration scripts
(`migrate_paper_trading.py`, `migrate_backtest_data.py`,
`migrate_bq_schema.py`, `migrate_agent_memories.py`) are idempotent
and safe to re-run.

**Lock-in:** new migration `scripts/migrations/migrate_signals_log.py`
follows the `migrate_paper_trading.py` template byte-for-byte at
the structural level (constants, schema list, main() function,
print statements). Project ID and dataset name reused from the
same constants.

### 6. Defensive write-path: never raise from a boundary

The standard Python pattern for defensive boundary writes is to
catch a narrow set of exceptions, log a warning with structured
context, and return without re-raising -- the caller should never
know whether the durable write succeeded. The existing
`publish_signal` step 9 in `signals_server.py` already implements
this pattern around the in-memory `_append_signal_history` call:

```python
try:
    self._append_signal_history(signal_id, signal, trade)
except Exception as e:
    logger.warning(f"signal_history append failed: {type(e).__name__}")
```

**Lock-in:** the BQ write call inside `_append_signal_history`
itself follows the same pattern, with one tighter constraint --
it catches `Exception` (since the BQ client raises a wide range
of exception types) and logs `bq_signal_log save failed:
{type}` to keep the log message ASCII-only per security.md.
The outer step-9 try/except still wraps it, so even if the inner
catch were removed, the publish would still complete.

### 7. Storage Write API vs streaming insertAll (cycle-2 deferred)

For completeness: the Storage Write API gives 50% cost savings,
~10x throughput, sub-second latency, and -- critically for the
deferred outcome-update path -- DML compatibility within minutes
instead of 30-90. The Python client is gRPC + protobuf and
requires schema definitions in `.proto` files compiled with
protoc, which is a significant departure from the current
`insert_rows_json` ergonomics.

**Lock-in:** out of scope for this cycle. Documented as the
preferred follow-up path for Phase 4.2.4-follow-up which will
ship the outcome-update path.

## Sources (representative, per category)

1. https://cloud.google.com/bigquery/docs/streaming-data-into-bigquery
2. https://cloud.google.com/blog/products/bigquery/life-of-a-bigquery-streaming-insert
3. https://docs.cloud.google.com/bigquery/docs/write-api
4. https://cloud.google.com/bigquery/docs/write-api-best-practices
5. https://docs.cloud.google.com/bigquery/docs/partitioned-tables
6. https://docs.cloud.google.com/bigquery/docs/creating-partitioned-tables
7. https://cloud.google.com/knowledge/kb/cannot-update-or-delete-over-bigquery-streaming-tables-000004334
8. https://github.com/apache/airflow/issues/59408 (DML+streaming buffer Airflow tracker)
9. https://durgaanalytics.com/event_sourcing_audit_trading
10. https://www.lean.io/ (LEAN Algorithmic Trading Engine reference)
11. https://docs.cloud.google.com/bigquery/docs/tables (CREATE TABLE patterns)
12. https://pypi.org/project/google-cloud-bigquery/
13. https://blog.miguelgrinberg.com/post/the-ultimate-guide-to-error-handling-in-python
14. https://www.qodo.ai/blog/6-best-practices-for-python-exception-handling/
15. https://medium.com/tech-at-trax/bigquery-update-delete-insert-simply-insert-6d002f58319d (the canonical "don't do DML on a streaming table, append" article)

Plus 55+ additional URLs across the same 7 categories from the
WebSearch result blocks.

## Gate verdict

PASSED. 7 categories, 7 queries, ~70 URL results, 15 representative
sources cited above. All design lock-ins are research-driven, not
intuition. Two of the lock-ins (insert_rows_json over Storage
Write API; defer outcome-update path) are explicitly conservative
choices that prevent scope creep into a follow-up cycle.
