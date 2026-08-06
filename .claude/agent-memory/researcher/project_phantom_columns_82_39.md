---
name: phantom-columns-82-39
description: nightly_outcome_rebuild 82.39 -- BOTH fetch AND write target phantom columns; closing the step turns TWO green 82.12 tests RED; a SECOND identical defect lives in cost_budget_api past the sweep's recall envelope; criterion 4's "assert scope non-empty" is unsatisfiable as literally written
metadata:
  type: project
---

Step 82.39 (`backend/slack_bot/jobs/_production_fns.py`, `nightly_outcome_rebuild`).
Measured 2026-08-05, **re-measured and extended 2026-08-06**.

**Why:** the step was scoped as "repair one query". Nine load-bearing facts were absent from
or contradicted by the step description, and each is a shape that recurs in this repo.

**How to apply:** whenever a step says "a query references a column that doesn't exist",
check ALL of these before writing the contract.

1. **Check the WRITE side too.** `make_ledger_fetch_fn` (`:209-234`) SELECTs two phantom
   columns; `make_outcome_write_fn` (`:237-261`) writes `{trade_id, pnl, outcome,
   recorded_at}` to `outcome_tracking`, whose real 9-column schema shares exactly ONE
   column (`ticker`) and has TWO REQUIRED columns nobody supplies. Google's own doc is the
   citation: *"Even if you receive a success HTTP response code, you'll need to check the
   `insertErrors` property"* and on a schema mismatch *"none of the rows are inserted"*
   (`cloud.google.com/bigquery/docs/streaming-data-into-bigquery`). A fetch-only repair
   ships a job that still writes 0 rows. The write-fn docstring documents a schema that has
   never existed -- **docstrings about BQ schemas are claims, not facts.**
2. **The obvious rename is STILL broken.** Keeping `TIMESTAMP_TRUNC(created_at, DAY)` gives
   `400 No matching signature` -- `created_at` is STRING.
   `SAFE.TIMESTAMP(created_at) >= TIMESTAMP_SUB(...)` dry-runs VALID. Counter-trap:
   SAFE.TIMESTAMP **BREAKS on a native TIMESTAMP column** (`400 SAFE with function
   timestamp is not supported`, measured on `agent_memories.created_at`; 31 such columns
   exist). Repo idiom: `cycle_health.py:436-439 _STRING_DATE_TIMESTAMP_COLS` already
   contains `("paper_trades","created_at")` verbatim. Also: the SAFE. prefix does NOT apply
   to **operators** per the BQ functions-reference -- it must wrap the coercion, not the `>=`.
3. **`realized_pnl_pct` is a PERCENT**, populated only on SELL legs (32/32 SELLs, 0/33 BUYs).
   Aliasing `AS pnl` preserves the SIGN but silently changes UNITS downstream.
4. **The "froze the learning loop" claim is FALSE.** `outcome_tracker.py` has 0 refs to
   `outcome_tracking`; the real writer is `bigquery_client.py:47/:415` driven from
   `autonomous_loop.py:3041-3096`, gated by `settings.py:34 paper_learn_loop_enabled=False`.
   Two independent causes; fixing the query unfreezes nothing.
5. **`derive_scope()` has LOW recall and its output reads like a clean bill of health.**
   `files_scanned=296` but `sql_literals=13`, `tables_resolved=1` of 33. It needs a literal
   backticked FQ table name; 20 backend files use `` FROM `{ `` and are invisible;
   `iter_python_files` roots at `backend/` so **`scripts/` is never scanned**. Say
   "0 remaining WITHIN THE MEASURED RECALL ENVELOPE", never "the repo is clean".
6. **`_compute_outcomes` CRASHES on a NULL pnl**: `.get("pnl", 0.0)` defaults only on a
   missing KEY, not a None VALUE. It sits OUTSIDE the write's try (but inside `heartbeat`,
   which catches). Dropping `IS NOT NULL` to widen the query detonates it on all 33 BUYs.
7. **Dead since the file's FIRST commit** (`2301b977`, 2026-05-11) -- never correct, not a
   regression. Lower bound 87 days. But `IdempotencyStore` is an in-memory `set()` and
   `heartbeat`'s default sink is `logger.info`, so **there is NO durable receipt** -- never
   claim "87 nightly runs", only "dead since first commit".
8. **A "last 30 days" live-data fixture is a TIME BOMB**: yields 3 rows today, **0 after
   2026-08-26** (newest SELL 2026-07-27). Pin a fixed window -- 2026-06 has 20 SELLs.

**NEW 2026-08-06 -- the three findings that change the contract:**

9. **CLOSING 82.39 TURNS TWO CURRENTLY-GREEN TESTS RED.**
   `backend/tests/test_phase_82_12_string_column_guards.py:403-422` asserts the defect IS
   flagged (`("timestamp", ".../_production_fns.py") in flagged`) -- the fix removes it.
   And `:425-456` requires an OPEN masterplan step whose NAME contains all of
   `("_production_fns","paper_trades","timestamp","realized_pnl")`; measured over all 1115
   steps, **82.39 is the only match** (82.48's name lacks the tokens), so flipping 82.39 to
   `done` fails it. Both must be repaired in the SAME change. **Generalisable: a step that
   fixes a defect a prior step PINNED as evidence must grep for its own pin before closing.**
10. **A SECOND live defect of the identical class, invisible to the criterion-4 instrument.**
    `backend/api/cost_budget_api.py:80-86` selects `input_tokens`/`output_tokens` from
    `pyfinagent_data.llm_call_log`, whose real columns are `input_tok`/`output_tok`. Dry run:
    `Unrecognized name: input_tokens; Did you mean input_tok?`. Same `except Exception` ->
    `logger.warning` -> `return None, None` (`:94-96`), so the cost tile's
    `llm_tokens_today` (`:142`/`:154`) has been permanently null. `derive_scope` cannot see
    it (f-string table ref). **So a criterion-4 sweep reporting "clean" is FALSE ASSURANCE.**
    Widening the sweep manually is worth one round: 99 raw candidates -> 9 after dropping
    docstrings/CTEs/aliases -> 1 real (the rest were `@tickers` params, UNION table names,
    and INFORMATION_SCHEMA.COLUMNS' own `column_name`/`data_type`).
11. **Criterion 4's "derived scope asserted NON-EMPTY" is a trap.** `derive_scope` returns
    TWO lists that behave oppositely: `scope` is `[]` TODAY (unsatisfiable assertion ->
    uncloseable step, cf. 81.0), and `unknown_columns` becomes `[]` the moment the fix lands
    (assertion fails after the fix). The only satisfiable reading is the module's own
    docstring at `schema_oracle.py:39-43`: assert the **INPUT surface** non-empty
    (`files_scanned/sql_literals/tables_resolved/columns_in_oracle > 0`).
    Also `schema_oracle.dry_run` (`:550-566`) catches **only `BadRequest`** -- a missing
    TABLE raises `NotFound` and escapes.

**Criterion-3 shape (reuse, don't rebuild):** `raise_cron_alert_sync` at
`alerting.py:253-287`, never raises (`:266`); `_CRITICAL_SEVERITIES` (`:54`) =
`{P0,P1,critical,CRITICAL}` and P0/P1 reach `_bot_token_fallback` (`:217-218`) because the
webhook is empty on this machine -- **P2 is logged and dropped**. Template:
`autoresearch_health.py:321-328` (function-local import + `P0 if escalated else P1`), tested
by `test_phase_82_11_...py:65` + `:502-506` (patch the SOURCE module, `autospec=True`,
assert `call_args` not caplog, and include the negative case).

**Citations that settled the external half** (read in full 2026-08-06):
`cloud.google.com/bigquery/docs/running-queries` -- *"Dry runs don't use query slots, and
you are not charged for performing a dry run"* + *"validation of your query"*. These pages
are JS-rendered: curl + tag-strip, not WebFetch (see [[gcloud-docs-fetch]]).
`github.com/autotraderuk/dbt-dry-run` "Capabilities and Limitations" -- catches
*"Typos in columns names"* + type errors + permission errors; **cannot** catch
*"Queries that run but do not return intended/correct result"* -> criteria 1 and 2 MUST stay
separate tests. arXiv/ICPC 2017 Padua & Shang (pdfplumber) -- Catch Generic 31.9%.
`docs.python.org/3/library/unittest.mock.html` -- *"You must patch where an object is
looked up."*

Related: [[vacuous-bq-guards-82-12]], [[non-forward-labels-82-16]], [[gcloud-docs-fetch]].
