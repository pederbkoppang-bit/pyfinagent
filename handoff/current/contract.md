# Contract -- Phase 4.2.4 BQ Durable Persistence Scaffold (Publish Write Path)

**Step ID:** 4.2.4 (subset: publish event write path + table migration)
**Target files:**
- `scripts/migrations/migrate_signals_log.py` (NEW file)
- `backend/db/bigquery_client.py` (add 1 method)
- `backend/agents/mcp_servers/signals_server.py` (modify `_append_signal_history` only)

**Base commit:** `912008c` (origin/main HEAD as of 2026-04-15T0000Z)

## Problem statement

The Phase 4.2.2 in-memory `signal_history` + `_signals_by_id` index
is wiped on every process restart. To support production reliability
gates (Phase 4.4 Go-Live Checklist) and cross-restart accuracy
tracking, signals must be durably persisted to BigQuery as they are
published. This contract scaffolds ONLY the publish-event write path:
table migration, BQ client method, and the wire-up in
`_append_signal_history`. The outcome-update path and the warm-load
read path are deferred to follow-up cycles.

## Fix approach (locked by research.md)

1. New idempotent migration `scripts/migrations/migrate_signals_log.py`
   creates the `signals_log` table with a 17-column schema mirroring
   the in-memory record. Follows the `migrate_paper_trading.py`
   template byte-for-byte at the structural level.
2. New method `BigQueryClient.save_signal(record: dict) -> None`
   appends a row to `signals_log` via `insert_rows_json`. Logs errors
   on insert failure. Never raises.
3. `SignalsServer._append_signal_history` adds a single try/except
   block AFTER the in-memory append that calls
   `self.bq_client.save_signal(record)` if `self.bq_client is not
   None`. Wrapped in a tight try/except that catches `Exception`,
   logs `bq_signal_log save failed: <type>`, and returns. Never
   raises. The outer `publish_signal` step 9 try/except still wraps
   it as defense in depth.

## Files in scope

| File | Role | Bound (added/deleted) |
|---|---|---|
| `scripts/migrations/migrate_signals_log.py` | NEW | <= 90 / 0 |
| `backend/db/bigquery_client.py` | add 1 method | <= 25 / 0 |
| `backend/agents/mcp_servers/signals_server.py` | modify 1 method | <= 15 / 0 |
| **Total** | | **<= 130 / 0** |

## Out of scope

- Outcome-update write path (`track_signal_accuracy` -> BQ DML).
  Blocked by 30-90 minute streaming-buffer DML restriction.
  Deferred to a follow-up cycle that will adopt the Storage Write
  API or an event-log design.
- Warm-load read path (`get_signal_history` reading from BQ at
  startup). The in-memory list remains the canonical query source
  this cycle.
- Migration script execution. Remote env has no GCP credentials;
  the migration ships as code only and Peder runs it once locally
  before flipping the durable-write feature on.
- Storage Write API migration. Requires protobuf schemas, separate
  cycle.
- Any change to `track_signal_accuracy`, `get_accuracy_report`,
  `publish_signal` (other than the one indirect call already in
  step 9), or any of the 19 byte-identical methods on
  `SignalsServer`.
- Any change to existing `BigQueryClient` methods.
- Any change to `signal_history` / `_signals_by_id` / `__init__`
  state on `SignalsServer`.
- Imports added to `signals_server.py` (none needed -- the bq_client
  call goes through the existing `self.bq_client` attribute which
  is already initialized in `__init__`).
- Imports added to `bigquery_client.py` (none needed -- the new
  method uses already-imported `bigquery` and `logger`).

## Success criteria

### Group A: Migration script (SC1-SC8)

- **SC1** `scripts/migrations/migrate_signals_log.py` exists and
  parses cleanly (`ast.parse`).
- **SC2** Constants `PROJECT_ID = "sunny-might-477607-p8"` and
  `DATASET = "financial_reports"` match the existing
  `migrate_paper_trading.py` constants byte-for-byte.
- **SC3** Schema list `SIGNALS_LOG_SCHEMA` declares exactly 17
  fields, types as specified in SC4.
- **SC4** Field types and modes:
  - `signal_id` STRING REQUIRED
  - `ticker` STRING REQUIRED
  - `signal_type` STRING REQUIRED
  - `confidence` FLOAT64 NULLABLE
  - `signal_date` STRING NULLABLE  (was `date` in record; renamed
    to avoid BQ reserved-word collision)
  - `entry_price` FLOAT64 NULLABLE
  - `factors_json` STRING NULLABLE  (JSON-encoded list of factor
    strings)
  - `created_at` TIMESTAMP REQUIRED
  - `outcome` STRING NULLABLE
  - `scored` BOOL NULLABLE
  - `hit` BOOL NULLABLE
  - `exit_price` FLOAT64 NULLABLE
  - `exit_date` STRING NULLABLE
  - `forward_return_pct` FLOAT64 NULLABLE
  - `holding_days` INT64 NULLABLE
  - `recorded_at` TIMESTAMP REQUIRED  (when this row was written
    to BQ; equal to `created_at` for publish events, distinct for
    a future outcome event)
  - `event_kind` STRING REQUIRED  (one of "publish", "outcome";
    only "publish" is written this cycle)
- **SC5** Migration follows the idempotent `try: client.get_table /
  except: create_table` pattern from `migrate_paper_trading.py`.
- **SC6** Migration prints `Created table signals_log` on first
  run and `Table signals_log already exists. Skipping.` on
  re-runs, matching the existing migration's print messages.
- **SC7** Migration loads `GCP_CREDENTIALS_JSON` from environment
  exactly as `migrate_paper_trading.py` does.
- **SC8** Migration uses `service_account.Credentials.from_service_account_info`
  with the same two scopes as the existing migration.

### Group B: BigQueryClient.save_signal (SC9-SC15)

- **SC9** New method `save_signal` exists on `BigQueryClient`.
- **SC10** Method signature is exactly
  `def save_signal(self, record: dict) -> None:`.
- **SC11** Method computes table reference as
  `f"{self.settings.gcp_project_id}.{self.settings.bq_dataset_reports}.signals_log"`,
  matching the `_pt_table` pattern.
- **SC12** Method uses `self.client.insert_rows_json(table, [record])`
  -- not DML INSERT, not the Storage Write API. Matches `save_outcome`.
- **SC13** Method logs `BigQuery insert errors: {errors}` on
  insert failure, matching the `save_outcome` log message form.
  ASCII only (no arrows / em dashes).
- **SC14** Method does NOT raise on insert failure. The errors
  list is logged; a subsequent commit will decide on retry semantics.
  This is intentionally laxer than `save_report` (which raises) --
  signal logging must never break the publish path.
- **SC15** Method docstring is one-liner: `"""Append a single
  signal-publish event to the signals_log table."""`.

### Group C: signals_server.py wire-up (SC16-SC22)

- **SC16** Only `_append_signal_history` is modified. The other
  20 methods on `SignalsServer` are AST-byte-identical to the
  base commit `912008c`.
- **SC17** No new top-level imports added to `signals_server.py`.
  The existing `from datetime import datetime, timezone, date`
  line is unchanged.
- **SC18** The in-memory append (`self.signal_history.append(record)`
  + `self._signals_by_id[signal_id] = record`) happens FIRST,
  BEFORE the BQ call. The BQ call must never block or skip the
  in-memory write.
- **SC19** The BQ call is wrapped in a `try: ... except Exception
  as e: logger.warning(...)` block, ASCII-only, never raises.
- **SC20** The BQ call is gated on `if self.bq_client is not None`.
  In stub mode (`_SIGNALS_AVAILABLE = False`), `bq_client` is
  `None`, so the BQ branch is skipped entirely and existing
  stub-mode tests still pass.
- **SC21** The BQ call passes a NEW dict that maps the in-memory
  record fields to the SC4 schema (renames `date` -> `signal_date`,
  `factors` -> `factors_json` via `json.dumps`, adds `created_at`
  / `recorded_at` / `event_kind`). The original `record` dict is
  not mutated -- the in-memory state stays in its existing shape.
- **SC22** The wire-up is exactly 1 indented `if` block of
  approximately 12-15 lines at the bottom of `_append_signal_history`,
  immediately after the existing `self._signals_by_id[signal_id]
  = record` line.

### Group D: Diff bound + scope discipline (SC23-SC25)

- **SC23** Total diff <= 130 lines added, 0 deleted (this is a
  pure-additive cycle).
- **SC24** Exactly 3 files touched: the new migration script,
  `bigquery_client.py`, and `signals_server.py`. No others.
- **SC25** No emoji, no Unicode arrows, no em dashes in any
  added line. ASCII-only per security.md.

## Adversarial probes (10) -- for qa-evaluator

These are the probes the qa-evaluator subagent should pre-bake
into its assertion block.

- **ADV1** `_append_signal_history(signal_id="", signal={}, trade=None)`
  with `bq_client=None`: returns without raising, in-memory state
  unchanged. (Existing dedup guard at line 564.)
- **ADV2** `_append_signal_history` with `bq_client=None` and a
  valid signal_id + signal: in-memory append succeeds, BQ branch
  skipped, no log warning emitted.
- **ADV3** `_append_signal_history` with `bq_client=<mock raising
  RuntimeError>` and a valid signal: in-memory append succeeds,
  BQ call raises, exception is caught, `logger.warning` is called
  exactly once, method returns None.
- **ADV4** `BigQueryClient.save_signal({...})` with mocked
  `client.insert_rows_json` returning `[]` (success): no log
  emitted, returns None.
- **ADV5** `BigQueryClient.save_signal({...})` with mocked
  `insert_rows_json` returning `[{"reason": "x"}]`: `logger.error`
  is called, returns None (no raise).
- **ADV6** `BigQueryClient.save_signal({...})` with mocked
  `insert_rows_json` raising `ConnectionError`: the exception
  PROPAGATES (this method does NOT catch the exception itself;
  the `_append_signal_history` outer try/except is what protects
  the publish path). This is intentional per SC14 -- the catch
  lives at the call site, not the method.
- **ADV7** Migration script `ast.parse` clean and `py_compile`
  clean. (Module-level execution would require GCP creds.)
- **ADV8** Schema field count in migration script equals exactly
  17 (SC3 enforcement).
- **ADV9** AST byte-identity of all 20 unchanged `SignalsServer`
  methods between `912008c` and the post-fix file.
- **ADV10** No `>= since_date` regression -- the Phase 4.2.3.2
  `_parse_iso_date` block must remain byte-identical (this cycle
  doesn't touch it but a wrong scope would).

## Verification commands (deterministic, stdlib only)

```bash
# Sanity
python3 -c "import ast; ast.parse(open('scripts/migrations/migrate_signals_log.py').read()); print('migration OK')"
python3 -c "import ast; ast.parse(open('backend/db/bigquery_client.py').read()); print('bq_client OK')"
python3 -c "import ast; ast.parse(open('backend/agents/mcp_servers/signals_server.py').read()); print('signals_server OK')"

# Schema field count check
python3 -c "
import re
src = open('scripts/migrations/migrate_signals_log.py').read()
fields = re.findall(r'bigquery\.SchemaField', src)
assert len(fields) == 17, f'expected 17 fields, got {len(fields)}'
print(f'schema OK ({len(fields)} fields)')
"

# AST byte-identity audit (20 unchanged methods on SignalsServer)
python3 -c "
import ast
src = open('backend/agents/mcp_servers/signals_server.py').read()
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == 'SignalsServer':
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        print(f'SignalsServer methods: {len(methods)}')
        break
"
```

## Anti-leniency rules

1. Diff bound `<= 130 / 0` is HARD. Any overage requires
   explicit justification in `experiment_results.md`.
2. Schema MUST have exactly 17 fields. Adding a field
   without contract amendment is a violation.
3. The in-memory append MUST happen before the BQ call. Reordering
   is a SC18 violation.
4. The BQ call MUST be wrapped in try/except at the call site.
5. The 20 unchanged `SignalsServer` methods MUST be AST-byte-identical
   to base commit `912008c`. Use `ast.dump(FunctionDef)` comparison.
6. Zero new imports in `signals_server.py`. Zero new imports in
   `bigquery_client.py` (the migration script gets its own imports).
7. ASCII-only in all added lines. No emojis, arrows, em dashes.
8. No retouching the Phase 4.2.3.2 `_parse_iso_date` helper or
   the `since_date` block in `get_signal_history`. They are stable
   scaffold from the prior cycle.
