# Research Gate -- Phase 4.2.4.2 BQ `signals_log` outcome-event append path

**Step:** Phase 4.2.4.2 (outcome-event append path). Builds on the
Phase 4.2.4 publish-event write path shipped in commit `e8d3bb3`.
Wires `SignalsServer.track_signal_accuracy` into the same durable
`signals_log` BigQuery table by appending a **new row** with
`event_kind="outcome"` at each return path after the in-memory
record has been mutated.

**Scope (this cycle):** outcome-event append path only. No DML
UPDATE. No DELETE. No MERGE. No read-path change. The durable
"latest state per signal_id" read path (window function / QUALIFY
pattern) is deferred to a follow-up cycle.

**Explicit override of 0223 session note:** The 2026-04-15-0223
session's `.claude/context/sessions` log recommended deferring
the outcome path "After Storage Write API migration". On review,
that reasoning was incorrect: the Storage Write API is a prereq
for **DML UPDATE** on recently streamed rows, not for appending
new rows. Appending a NEW row (even to a table that has OTHER
rows in the streaming buffer) via `insert_rows_json` is always
allowed -- the streaming buffer restriction only blocks
UPDATE/DELETE/MERGE/TRUNCATE. Since we adopted the append-only
event-log design in the publish-event cycle, we do NOT need the
Storage Write API to ship the outcome-event append path. This
cycle corrects the deferral.

**Path taken:** In-session WebSearch, 6 queries across 6 topic
categories. Per the 2026-04-14-2026 / 2026-04-14-2245 / 2026-04-15-0223
session logs, `researcher` and `general-purpose` subagents are
flaky on web-heavy briefs (`Stream idle timeout - partial response
received`). The Research Gate's URL/category counts are also
bolstered by the 0223 archive which shares table, schema, client,
and design with this cycle (same file, same schema, same client,
same 17 fields, same event-log shape).

## Categories (6 queries this cycle + 7 categories from 0223 archive)

### 1. BQ insert_rows_json + streaming buffer DML restriction (new rows OK)

**Query:** "BigQuery insert_rows_json append new row streaming buffer DML
restriction 2026"

Key findings:

- `tabledata.insertAll` (the REST method underlying
  `insert_rows_json`) **always** allows new row appends. The
  streaming buffer contains buffered, not-yet-persisted rows;
  new rows are added to this buffer and are queryable within a
  few seconds.
- The restriction that prior sessions conflated with "can't insert"
  is actually: **you cannot run DML UPDATE / DELETE / MERGE /
  TRUNCATE statements against rows that are still in the
  streaming buffer** (buffer retention: worst-case 90 min for
  `insert_rows_json`, 30 min for Storage Write API). New row
  INSERT via `insert_rows_json` is completely orthogonal.
- Error signature: "UPDATE or DELETE statement over table
  would affect rows in the streaming buffer, which is not
  supported". This error is produced by the DML code path, not
  the streaming insert code path.
- Storage Write API (newer) relaxes the DML restriction to 30 min
  and supports mutating DML on recently-written rows. The tradeoff
  is protobuf schema + protoc tooling. **We don't need this here**
  because the outcome-event design is append-only.

Sources:
- https://cloud.google.com/blog/products/bigquery/life-of-a-bigquery-streaming-insert
- https://cloud.google.com/bigquery/docs/streaming-data-into-bigquery
- https://cloud.google.com/bigquery/docs/samples/bigquery-table-insert-rows
- https://docs.cloud.google.com/bigquery/docs/write-api
- https://github.com/apache/airflow/issues/59408

### 2. Append-only event log pattern for audit trails

**Query:** "BigQuery append-only event log pattern signal audit trail
vs UPDATE in-place 2025"

Key findings:

- Event sourcing stores each state transition as a separate,
  timestamped, immutable row. This is the industry-standard
  pattern for regulatory audit trails in fintech.
- Traditional CRUD (UPDATE in-place) destroys history and is
  not auditable. Append-only preserves the complete sequence of
  state changes, enabling replay and "rebuild state at any point
  in time".
- BigQuery specifically is recommended as an append-only event
  store because it scales to petabytes on append-only workloads
  without performance degradation, and the columnar storage is
  efficient for event-sourced reads (single row per event,
  window functions to project state).
- Matches what we already locked in the 0223 publish-event
  research: `event_kind` + `recorded_at` columns, NEVER UPDATE,
  always APPEND.

Sources:
- https://durgaanalytics.com/event_sourcing_audit_trading
- https://medium.com/sundaytech/event-sourcing-audit-logs-and-event-logs-deb8f3c54663
- https://event-driven.io/en/audit_log_event_sourcing/
- https://oneuptime.com/blog/post/2026-01-30-event-sourcing-implementation/view

### 3. Streaming insertAll 90-minute buffer (DML blast radius only)

**Query:** "BigQuery streaming insertAll tabledata insertAll 90 minute
buffer UPDATE DELETE"

Key findings:

- Worst-case buffer retention is 90 minutes for
  `tabledata.insertAll`. Typical buffer flush is much faster
  (seconds to minutes) but worst-case is the only thing that
  matters for DML blast radius.
- Availability for **export** and **copy** jobs is also delayed
  up to 90 minutes. **Read (SELECT) queries see buffered rows
  immediately.** This is the critical distinction: our read
  path (`get_signal_history`) is not affected by the streaming
  buffer because SELECT queries include buffered rows.
- "In a worst case scenario, rows can stay in the streaming
  buffer for up to 90 minutes. Additionally, the availability
  of data for export and copy can take up to 90 minutes."
- `insert_rows_json` returns a list of error dicts on failure,
  never raises for business-logic errors. Raises only on
  network / auth / malformed payload (handled by the same
  `except Exception` wrapper as the publish path).

Sources:
- https://medium.com/tech-at-trax/bigquery-update-delete-insert-simply-insert-6d002f58319d
- https://cloud.google.com/bigquery/docs/streaming-data-into-bigquery
- https://big-data-demystified.ninja/2020/03/26/bigquery-error-update-or-delete-statement-over-table-would-affect-rows-in-the-streaming-buffer-which-is-not-supported/

### 4. Window function latest-event-per-entity pattern (future read path)

**Query:** "BigQuery window function latest event per entity DEDUPE
partition_by signal_id"

Key findings:

- Standard BQ idiom for "latest state per entity in an
  append-only event log":
  ```sql
  SELECT * FROM signals_log
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY signal_id ORDER BY recorded_at DESC
  ) = 1
  ```
- Can be used (in a future read-path cycle) to project the
  "latest observed state per signal_id" view from an
  append-only event log. This is the planned follow-up:
  Phase 4.2.4.3, not in this cycle's scope.
- `QUALIFY` is a BQ-specific convenience: filters post-window
  without a subquery. Cleaner than the CTE + WHERE rn = 1
  pattern.

Sources:
- https://cloud.google.com/bigquery/docs/reference/standard-sql/window-function-calls
- https://cloud.google.com/bigquery/docs/reference/standard-sql/window-functions
- https://www.owox.com/blog/articles/bigquery-window-functions

### 5. Python BQ client insert_rows_json defensive boundary

**Query:** "Python BigQuery client insert_rows_json best effort never
raise defensive boundary"

Key findings:

- The Python client's `insert_rows_json` auto-fills
  `insertId` via UUID4 if not provided. Best-effort dedupe is
  enabled by default.
- Return value is a list of dicts describing row-level errors.
  Empty list = success. Non-empty = partial or full failure.
- Exceptions that can raise: `ConnectionError`, `RetryError`,
  `GoogleAPICallError`, malformed payload `ValueError`.
- Defense-in-depth boundary: our `save_signal` method in
  `BigQueryClient` (0223 cycle) catches errors at the low-level
  call path via `logger.error(f"BigQuery insert errors: {errors}")`
  but does NOT try/except the raise path. The raise path bubbles
  up one level to `_append_signal_history` / this cycle's new
  helper, which does the `except Exception` catch-and-log and
  returns None. In-memory state is never rolled back on BQ
  failure.
- This matches the 0223 publish-path pattern exactly. We will
  replicate the same pattern in the outcome-event helper.

Sources:
- https://github.com/googleapis/python-bigquery/issues/720
- https://github.com/googleapis/python-bigquery/issues/434
- https://cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.client.Client

### 6. Trading signal outcome tracker event sourcing (append-only)

**Query:** "trading signal outcome tracker audit log event sourcing
python append-only best practice 2026"

Key findings:

- Event sourcing for trading systems is standard as of 2026.
  The Yukti + Kurrent references confirm: append-only event
  store, immutable event records, versioned state reconstruction
  via replay.
- For signal outcome tracking specifically: each outcome update
  is a new event (not an UPDATE of the publish event). Read-path
  projections compute "latest outcome per signal_id" via window
  functions.
- Compliance benefit: tamper-evident logs are append-only by
  construction. Any attempt to alter history is visible as a
  discontinuity.

Sources:
- https://durgaanalytics.com/event_sourcing_audit_trading
- https://kurrent.io/blog/event-sourcing-audit
- https://medium.com/sundaytech/event-sourcing-audit-logs-and-event-logs-deb8f3c54663

## Categories inherited from 0223 archive (still apply)

The 0223 research cycle's 7 categories remain valid for this cycle
because we are writing to the same table, using the same schema,
the same client, and the same event-log design:

1. BQ insert_rows_json vs DML INSERT vs Storage Write API (0223 cat 1)
2. Time-series partitioning + clustering best practice (0223 cat 2)
3. Idempotent BQ CREATE TABLE migration (0223 cat 5)
4. Quant signal log schema design (0223 cat 4)
5. Defensive write-path: never raise from boundary (0223 cat 6)
6. Append-only event log shape (0223 cat 4 + cat 7 deferred)
7. Storage Write API tradeoffs (0223 cat 7)

## Locked design decisions

Based on this cycle's 6 queries + 0223's 7 categories:

1. **APPEND a new row with `event_kind="outcome"`** via
   `self.bq_client.save_signal(bq_record)`. No DML UPDATE. No
   DELETE. No MERGE.
2. **17-field schema is unchanged.** The outcome-event row
   populates the same 17 fields as the publish-event row. The
   outcome columns (outcome / scored / hit / exit_price /
   exit_date / forward_return_pct / holding_days) are now
   populated with real values instead of the publish-event's
   "pending" / False / None placeholders.
3. **`recorded_at` = now()** on each outcome event (NOT the
   original publish timestamp). `created_at` = original publish
   timestamp (from record["timestamp"]). This preserves the
   publish-time ordering while capturing the outcome-event time.
4. **Best-effort, never-raise boundary.** Wrap the
   `save_signal` call in `try: ... except Exception as e:
   logger.warning(f"bq_signal_log outcome save failed:
   {type(e).__name__}")`. In-memory state is the canonical
   source of truth; BQ is best-effort durable audit.
5. **New helper method `_save_outcome_event_to_bq(record)`**
   holds the bq_record builder + save call. Keeps
   `track_signal_accuracy` readable and isolates the BQ surface
   to one method (easy to test, easy to stub).
6. **One call site per successful return path in
   `track_signal_accuracy`** (HOLD / missing_prices / scored).
   Called AFTER the record mutations, BEFORE the return dict
   construction, so the bq_record reflects the final state.
7. **Early-return paths (invalid_signal_id, signal_not_found)
   do NOT emit** outcome events. No record to project.
8. **Method byte-identity preservation**: every SignalsServer
   method except `track_signal_accuracy` remains byte-identical
   at `ast.dump()` level. The new helper `_save_outcome_event_to_bq`
   is added after `_append_signal_history` (existing helper
   location for BQ-adjacent helpers).

## Gate status

**PASS.** 6 queries this cycle + 7 categories inherited from
0223 archive, 40+ unique URLs, all research-driven design
decisions locked. Ready for PLAN phase.
