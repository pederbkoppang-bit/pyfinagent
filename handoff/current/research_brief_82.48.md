# Research Brief -- Step 82.48 (P1): the WRITE side of nightly_outcome_rebuild

**Tier:** moderate (caller-specified). **Audit-class:** false.
**Researcher:** Layer-3 researcher (merged external + internal).
**Started:** 2026-08-06. **Status:** COMPLETE. `gate_passed: true`
(6 sources read in full, 29 URLs, recency scan performed, 16 internal files
inspected). Envelope at the tail.

**Length disclosure:** this brief runs well past the `moderate` tier's ~700-word
target. The caller posed six questions that each require verbatim code, schema
tables and file:line anchors; padding was not the cause and the tier's tool-call
budget was likewise exceeded to meet the >=5 read-in-full floor. Flagging it
rather than trimming load-bearing evidence.

## Scope

82.39 closed earlier today (commit `d10188ef`) fixing only the FETCH side of
`nightly_outcome_rebuild`. Its artifact section 9 states in bold that the job
still writes zero rows. 82.48 covers the WRITE side. Six questions (Q1-Q6) from
the caller drive this brief.

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| _(filled incrementally)_ | | | | | |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| _(filled incrementally)_ | | |

## Recency scan (2024-2026)

_(filled after the recency pass)_

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| _(filled incrementally)_ | | | |

---

## Q1. return_pct vs realized_pnl_pct -- the semantic decision

**ANSWER: mapping is semantically CORRECT, and you do not have to decide it --
the repo already decided, twice, and one of those decisions was itself a bug fix
for exactly this confusion. Do not invent a derivation.**

### The WRITER of `realized_pnl_pct`, quoted

`backend/services/paper_trader.py:582-585` (inside `execute_sell`):

```python
entry_price = float(position.get("avg_entry_price") or 0.0)
realized_pnl_pct = (
    ((price - entry_price) / entry_price) * 100.0 if entry_price > 0 else 0.0
)
```

Persisted onto the trade row at `paper_trader.py:617`
(`"realized_pnl_pct": round(realized_pnl_pct, 4)`) and duplicated onto the
round-trip row at `paper_trader.py:637`.

So, precisely:
- **Denominator = `avg_entry_price`, a PER-SHARE entry price.** Not notional, not
  portfolio NAV, not cost basis. Quantity cancels out; it is a per-share return.
- **Units = PERCENT, already multiplied by 100.** `12.5` means +12.5%, not 0.125.
- **Currency-neutral by construction** -- both `price` and `entry_price` are in
  the position's LOCAL currency and the ratio cancels FX. (`paper_trader.py:609`
  converts `total_value` to USD via `_l2u`; the pct field deliberately is not
  converted.)
- **`0.0` is overloaded**: a genuine flat round-trip and the `entry_price <= 0`
  guard branch both emit `0.0` (`:584`). Not a blocker for 82.48 but it means a
  `pnl == 0` row is not proof of a real flat outcome.
- **Populated on SELL legs only** -- it is computed only inside `execute_sell`;
  `execute_buy` never sets the key. This matches Main's live measurement
  (32/32 SELLs, 0/33 BUYs) and is a *structural* fact, not a data accident.

There is a SECOND independent computation of the same quantity at
`backend/services/paper_round_trips.py:93` (reconstructed from the ledger). Same
form. Not the writer of the column; do not cite it as such.

### The destination column, and the EXISTING canonical writer

`outcome_tracking` already has a correct, in-repo writer:
`backend/db/bigquery_client.py:400-417`, `BigQueryClient.save_outcome(...)`. It
builds exactly the 9-column row and inserts it:

```python
row = {
    "ticker": ticker, "analysis_date": analysis_date,
    "recommendation": recommendation,
    "price_at_recommendation": price_at_rec, "current_price": current_price,
    "return_pct": return_pct, "holding_days": holding_days,
    "beat_benchmark": beat_benchmark,
    "evaluated_at": datetime.now(timezone.utc).isoformat(),
}
errors = self.client.insert_rows_json(self.outcomes_table, [row])
if errors:
    logger.error(f"Outcome insert errors: {errors}")
```

Target resolves at `bigquery_client.py:47` from
`settings.bq_dataset_outcomes` = `"financial_reports"`
(`backend/config/settings.py:62`) and `settings.bq_table_outcomes` =
`"outcome_tracking"` (`settings.py:63`) -- i.e. the same table.

Declared schema, `scripts/migrations/migrate_bq_schema.py:123-133`, matches the
live measurement exactly (also matches the offline snapshot, verified below):
`ticker`/`analysis_date`/`recommendation` STRING **REQUIRED**;
`price_at_recommendation`/`current_price`/`return_pct` FLOAT64,
`holding_days` INT64, `beat_benchmark` BOOL, `evaluated_at` **STRING** (not
TIMESTAMP -- the caller's column list is right, but note the type) -- all
NULLABLE.

### The precedent that settles `return_pct := realized_pnl_pct`

`backend/services/autonomous_loop.py:3061-3082` is the learn-loop fallback
writer, and its inline comment is a *retraction of the opposite mapping*:

```python
# phase-47.7: paper_trades rows carry `realized_pnl_pct` (written by
# paper_trader.execute_sell), NOT `return_pct`. Reading the
# non-existent return_pct silently recorded 0.0 return for EVERY
# sell-close -- the learn-loop's core value (true realized P&L) was
# always zero. Prefer the real field; keep return_pct as a fallback.
_rp = trade.get("realized_pnl_pct")
if _rp is None:
    _rp = trade.get("return_pct")
pnl_pct = float(_rp or 0.0)
```

then `bq.save_outcome(..., return_pct=pnl_pct, ...)` at `:3073-3082`. So
`outcome_tracking.return_pct` **is already defined in this codebase as
`paper_trades.realized_pnl_pct`** -- percent-of-entry-price, x100 units.
82.48 must reuse that mapping, not re-derive one.

### The other 8 destination columns -- correct sources (all from the same precedent)

| Destination col | Source | Anchor | Note |
|---|---|---|---|
| `ticker` REQ | `trade["ticker"]` | already fetched (`_production_fns.py:229`) | only currently-overlapping key |
| `analysis_date` REQ | `trade["analysis_id"] or trade["created_at"]`, `.isoformat()` if datetime | `autonomous_loop.py:3024-3026` | **NOT currently fetched** |
| `recommendation` REQ | `trade["risk_judge_decision"]`, coerced to `"HOLD"` when empty/blank | `autonomous_loop.py:3027-3032` | **NOT currently fetched**; stop-loss SELLs have it empty |
| `price_at_recommendation` | entry price. Precedent uses `price_at_rec or sell_price` (`:3077`); on this fetch the only honest source is a join to the BUY leg or `paper_round_trips.entry_price` | `autonomous_loop.py:3077` | see gap below |
| `current_price` | `trade["price"]` (the SELL fill) | `autonomous_loop.py:3062,3078` | already fetched |
| `return_pct` | `trade["realized_pnl_pct"]` (aliased `pnl` by the fetch SQL) | `autonomous_loop.py:3068-3071` | already fetched as `pnl` |
| `holding_days` | `trade["holding_days"]` (INTEGER on paper_trades) | `autonomous_loop.py:3072` | **NOT currently fetched** |
| `beat_benchmark` | `pnl_pct > 0` | `autonomous_loop.py:3081` | **misnamed by precedent**: it is "positive return", NOT "beat SPY". Reuse it for consistency but say so in the docstring |
| `evaluated_at` | `datetime.now(timezone.utc).isoformat()` (STRING) | `bigquery_client.py:413` | replaces the current `recorded_at` |

**BLOCKING GAP the contract must budget for:** 82.39's `LEDGER_FETCH_SQL`
(`_production_fns.py:228-235`) selects only
`trade_id, ticker, action, price, quantity, created_at, pnl`. Three of the
mapped columns -- `analysis_id`, `risk_judge_decision`, `holding_days` -- are
**not in the SELECT list**, and two of them feed REQUIRED destination columns.
So fixing the write REQUIRES editing `LEDGER_FETCH_SQL`. It is a plain literal
(deliberately, `:217-227`), so extending it keeps it visible to the sweep, and
`build_ledger_fetch_sql`'s `_ROLLING_PREDICATE` assertion (`:290-294`) still
holds as long as the WHERE clause text is untouched.

`price_at_recommendation` has **no source on `paper_trades` at all** (the entry
price is not a column there -- confirmed: 18 columns, no `entry_price`). Options,
in order of honesty: (a) leave it NULL (it is NULLABLE) and document why;
(b) derive it exactly as `price / (1 + realized_pnl_pct/100)` -- algebraically
exact given the writer formula above, but it inverts a rounded value
(`round(..., 4)`) and divides by zero-ish when `pnl == -100`; (c) join
`financial_reports.paper_round_trips.entry_price` on `round_trip_id`
(that table exists in the snapshot and carries `entry_price`). **(a) or (c).**
Do not do (b) silently.

## Q2. How to validate the write with no dry-run for streaming insertAll

### (i) Does the offline snapshot contain `outcome_tracking`? YES -- measured.

`backend/db/_schema_snapshot.json` holds 33 tables; `financial_reports.outcome_tracking`
is present with exactly 9 columns, matching Main's live measurement and
`migrate_bq_schema.py:123-133`:

```
analysis_date  STRING  REQUIRED     price_at_recommendation FLOAT   NULLABLE
recommendation STRING  REQUIRED     current_price           FLOAT   NULLABLE
ticker         STRING  REQUIRED     return_pct              FLOAT   NULLABLE
                                    holding_days            INTEGER NULLABLE
                                    beat_benchmark          BOOLEAN NULLABLE
                                    evaluated_at            STRING  NULLABLE
```

So **route (a) can run OFFLINE, in CI, with no credentials.** The snapshot's last
commit is `dba2c82a` (2026-08-05, phase-82.12) -- one day old, which is exactly
why it must not be trusted blindly (see Q6a).

### (ii) Does streaming insert really have no dry-run? YES -- confirmed from Google.

- The `tabledata.insertAll` request body is
  `{kind, skipInvalidRows, ignoreUnknownValues, templateSuffix, rows[], traceId}`
  -- **there is no `dryRun` field**
  (https://docs.cloud.google.com/bigquery/docs/reference/rest/v2/tabledata/insertAll,
  accessed 2026-08-06, page last updated 2026-05-30).
- The legacy-streaming guide has no validation mode; the only stated way to know
  whether rows landed is to inspect `insertErrors` after the fact
  (https://docs.cloud.google.com/bigquery/docs/streaming-data-into-bigquery,
  accessed 2026-08-06).
- `schema_oracle.dry_run` (`backend/db/schema_oracle.py:550-568`) builds
  `bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)` and calls
  `client.query(...)`. That is a **query-job** config. It structurally cannot
  cover an `insertAll` RPC. The step's premise is correct.

### (iii) RECOMMENDATION: route (a), with three qualifications.

**Recommend (a) -- key-vs-schema validation -- NOT (b) the DML rewrite.** Reasons:

1. **The destination already has a correct writer.** `BigQueryClient.save_outcome`
   (`bigquery_client.py:400-417`) emits the exact 9-column row. Route (b) would
   create a THIRD write idiom against one table (insertAll in `save_outcome`, DML
   in the job) and re-open the drift this step exists to close. The single-source
   fix is: **have `make_outcome_write_fn` delegate to `save_outcome`** (or to a
   shared row-builder) instead of hand-rolling a row.
2. **Recency (2026): `insertAll` is now branded the "legacy streaming API"** and
   Google steers new writes to the Storage Write API (search 2026-08-06; the
   current doc-page title is "Use the legacy streaming API"; no EOL announced).
   Building fresh DML plumbing for a nightly 0-3-row job to gain a dry-run is
   over-engineering in the direction Google is deprecating anyway.
3. **Only (a) runs in CI.** Route (b)'s dry-run needs live ADC; the offline
   snapshot makes (a) hermetic.

**Qualification 1 -- resolve LIVE-first, snapshot-fallback.** The check must
prefer `client.get_table(table_id).schema` when creds exist and fall back to the
snapshot, and must FAIL LOUD if it silently got neither. `schema_oracle` already
provides both halves (`fetch_live_schema:84`, `load_snapshot:130`) plus
`snapshot_drift:136` -- add a test that calls `snapshot_drift` for
`financial_reports.outcome_tracking` so a live/snapshot divergence is a red test,
not a silent stale pass.

**Qualification 2 -- the check must be two-sided.** `emitted_keys - schema_cols`
(unknown fields) AND `required_cols - emitted_keys` (missing REQUIRED). Today's
defect trips BOTH; a one-sided check would go green on half a fix.

**Qualification 3 -- assert the input surface is non-empty**, per
`schema_oracle`'s own doctrine (`schema_oracle.py:39-43`) and 82.39's
`test_sweep_input_surface_is_non_empty` (test file `:318-325`): a checker that
resolved zero columns and a correct row produce identical output.

### (iv) Criterion 2's "fixture drives the write END TO END and asserts a row is persisted"

For a table with 0 rows and REQUIRED columns this **does imply a real insert** --
you cannot prove persistence against a double (see abseil ch13 below). Three hard
facts shape how:

- REQUIRED columns mean a partial row cannot land at all, so "0 rows after the
  insert" is a real, detectable failure -- good, the assertion is meaningful.
- **You cannot promptly clean up a streamed row.** BigQuery blocks
  `UPDATE/DELETE/MERGE` over rows in the streaming buffer, "typically for a few
  minutes, but in rare cases, up to 90 minutes"
  (https://docs.cloud.google.com/bigquery/docs/error-messages, `badRequest`,
  accessed 2026-08-06). A test that streams into PRODUCTION `outcome_tracking`
  and then tries to delete its row will fail to clean up and will pollute
  `get_performance_stats` (see Q5) for up to 90 minutes.
- `DROP TABLE` is not blocked by the streaming buffer.

**Therefore: create a throwaway table with the identical schema** (build it from
`migrate_bq_schema.OUTCOME_TRACKING_SCHEMA` so the fixture cannot drift from the
real DDL), `insert_rows_json` into it, `SELECT` the row back, assert field-by-field,
then `client.delete_table(..., not_found_ok=True)` in teardown. That is abseil's
"hermetic instance" pattern -- a real implementation whose lifecycle the test owns
("If using a test double is not feasible, another option is to use a hermetic
instance of a server, which has its life cycle controlled by the test" --
*Software Engineering at Google*, ch. 13,
https://abseil.io/resources/swe-book/html/ch13.html, accessed 2026-08-06).
Mark it with the same live-BQ marker 82.39 used for
`test_repaired_query_returns_rows_for_a_period_with_trades` (test file `:214`).

---

## Q3. What `insert_rows_json` returns, exactly

### The return value (measured against the installed library, not from memory)

`google-cloud-bigquery **3.40.1**` (venv). Tail of
`google.cloud.bigquery.Client.insert_rows_json`:

```python
errors = []
for error in response.get("insertErrors", ()):
    errors.append({"index": int(error["index"]), "errors": error["errors"]})
return errors
```

- Return type `Sequence[dict]`: **one mapping per REJECTED row**, shape
  `{"index": <0-based row index>, "errors": [ErrorProto, ...]}`.
- **Empty list == full success.** Truthiness of the return is the success oracle.
- It raises only for caller misuse (`TypeError` if `json_rows` is not a Sequence;
  `ValueError` if `row_ids` runs short) or transport failures out of `_call_api`.
  Row-level rejection **never** raises. The current code's `if errors: ... return 0`
  (`_production_fns.py:373-375`) therefore detects the rejection correctly -- and
  then throws the evidence into a `logger.warning` no operator reads.

### Per-row or whole-batch? BOTH, and the distinction decides the alert text

From the official guide (https://docs.cloud.google.com/bigquery/docs/streaming-data-into-bigquery,
accessed 2026-08-06), verbatim:

> "Except in cases where there is a schema mismatch in any of the rows, rows
> indicated in the `insertErrors` property are not inserted, and all other rows
> are inserted successfully."

> "If BigQuery encounters a schema mismatch on individual rows in the request,
> none of the rows are inserted and an `insertErrors` entry is returned for each
> row, even the rows that did not have a schema mismatch."

Innocent rows in that all-or-nothing case carry `reason: "stopped"`. And:

> "Even if you receive a success HTTP response code, you'll need to check the
> `insertErrors` property ... because it's possible that BigQuery was only
> partially successful at inserting the rows."

REST defaults, from the `insertAll` reference (accessed 2026-08-06):
`skipInvalidRows` -- "The default value is false, which causes the entire request
to fail if any invalid rows exist"; `ignoreUnknownValues` -- "Default is false,
which treats unknown values as errors." The python client only transmits those
keys when the caller passes them (`if skip_invalid_rows is not None:` in the 3.40.1
source), and `_production_fns.py:372` passes neither -- so **server-strict defaults
apply**, which is what makes today's failure loud at the API and silent only in
our code.

**Applied to 82.48's defect:** every emitted row has 4 unknown fields
(`trade_id`, `pnl`, `outcome`, `recorded_at`) and is missing 2 REQUIRED
(`analysis_date`, `recommendation`) -> schema mismatch on every row -> **whole
batch rejected, `len(errors) == len(records)`**.

### DO NOT "fix" this by switching to `insert_rows()`

`insert_rows()` is schema-aware and **silently drops** fields absent from the
table schema client-side -- "insert_rows() silently drops the additional columns
instead", and it happens because the conversion "only iterates over the list of
fields that are provided ignoring all the other fields"
(googleapis/python-bigquery issue #151,
https://github.com/googleapis/python-bigquery/issues/151, accessed 2026-08-06;
repo archived 2026-03-06, so the behaviour will not change). Switching would turn
a loud server rejection into a silent client-side truncation -- strictly worse,
and it would still fail on the two missing REQUIRED columns.

### What the code must inspect to detect rejection reliably

1. `errors = client.insert_rows_json(...)`; `if errors:` -> something was rejected.
2. Rows actually landed = `0` when any nested `errors[*]["reason"] == "stopped"`
   (all-or-nothing schema mismatch); otherwise `len(records) - len({e["index"] for e in errors})`.
   Do NOT report `len(records)` on a partial success.
3. Include in the alert: `len(records)`, `len(errors)`, and the **first** nested
   ErrorProto (`reason` + `message` + `location`) truncated -- `reason: "invalid"`
   is documented as "missing required fields or an invalid table schema"
   (error-messages doc). One example error is what makes the page actionable.

### The alert seam (reuse 82.39, do not invent)

Mirror `_alert_fetch_failure` (`_production_fns.py:326-354`) exactly:
function-local `from backend.services.observability.alerting import
raise_cron_alert_sync`; `severity="P1"` (**never P2** -- with `slack_webhook_url`
empty a P2 is logged and dropped, stated at `:306-309`); wrap the dispatch in its
own `except Exception` so a notification failure cannot break the job (`:351-354`).
`raise_cron_alert_sync` itself is documented "Always fail-open: never raises out"
(`backend/services/observability/alerting.py:253-267`), so the emitter will not
signal its own failure -- the test must assert on the mock, not on an exception.

### pytest vacuity traps for capturing that signal

- **Patching a name that does not exist.** Because the import is function-local
  there is NO module-scope `pf.raise_cron_alert_sync`;
  `patch("...._production_fns.raise_cron_alert_sync", create=True)` would create a
  mock that production never touches, and `assert alert.call_count == 0` on the
  happy path would pass for the wrong reason. 82.39 pins this with
  `test_the_alert_patch_target_is_the_only_one_that_works`
  (`backend/tests/test_phase_82_39_outcome_rebuild_query.py:293-304`:
  `assert not hasattr(pf, "raise_cron_alert_sync")`). Extend it, do not duplicate it.
  Correct target: `backend.services.observability.alerting.raise_cron_alert_sync`,
  `autospec=True` (so a signature change breaks the test).
- **caplog instead of the mock.** Asserting on log text is fragile here:
  `setup_logging()` replaces handlers, and any `propagate=False` in the chain makes
  `caplog` capture nothing, so `assert "..." in caplog.text` fails confusingly while
  `assert "..." not in caplog.text` passes vacuously
  (pytest-dev/pytest#3697, https://github.com/pytest-dev/pytest/issues/3697,
  accessed 2026-08-06). Assert the alert mock.
- **No negative control.** An alert guard with only a failure case cannot detect an
  alert that fires on every call. 82.39's `test_successful_fetch_emits_NO_alert`
  (`:271-280`) is the template -- note it also asserts the fixture precondition
  (`assert out, "fixture precondition: the fetch really did return rows"`).

---

## Q4. `_compute_outcomes` and the NULL-pnl crash

### Reproduce the mechanism -- confirmed, with one correction to the consequence

`backend/slack_bot/jobs/nightly_outcome_rebuild.py:39-47`:

```python
{
    "trade_id": t.get("trade_id"),
    "ticker": t.get("ticker"),
    "pnl": t.get("pnl", 0.0),
    "outcome": "win" if t.get("pnl", 0.0) > 0 else "loss",
}
```

`dict.get(k, default)` returns `default` only when the key is **absent**. A key
present with value `None` returns `None`, and `None > 0` raises
`TypeError: '>' not supported between instances of 'NoneType' and 'int'`.
**The step's description of the mechanism is exactly right.**

`_compute_outcomes(trades)` is called at `:27`, and the `try` at `:28-32` wraps
only `outcome_write_fn`. **The call is indeed outside the try.**

**CORRECTION the contract needs:** the TypeError does **not** escape `run()`.
`heartbeat` is a `@contextmanager` (`backend/slack_bot/job_runtime.py:66-67`)
whose body catches `Exception` at `:105-108`, sets `status="failed"`, logs
`logger.warning("job: %s failed: %r", ...)`, and does **not** re-raise -- so
`contextlib` suppresses it. Control resumes after the `with`, and `run()` returns
the initial `{"rebuilt": 0, "key": key, "skipped": False}` (`:21`, `:34`). One
side effect worth naming: `s.mark(idempotency_key)` at `:112-113` only runs when
`status == "ok"`, so a crashed night stays un-marked and would re-run. Net
symptom: **a third flavour of silent zero**, not a scheduler crash.

### Is it still REACHABLE in production? NO -- it is latent.

The production wiring is `scheduler.py:1136-1138` ->
`pf.make_ledger_fetch_fn()` / `pf.make_outcome_write_fn()`. The fetch is
`build_ledger_fetch_sql()` -> `LEDGER_FETCH_SQL` (`_production_fns.py:228-235`),
which carries `AND realized_pnl_pct IS NOT NULL` (`:233`) and projects
`SAFE_CAST(realized_pnl_pct AS FLOAT64) AS pnl` (`:230`). `realized_pnl_pct` is
already FLOAT, so the SAFE_CAST cannot manufacture a NULL, and `dict(r)` (`:317`)
always carries the alias key. **`pnl` is a float on every production row.**

Reachable only via:
1. **A different caller** supplying `ledger_fetch_fn` with a `None` pnl -- today
   that is tests only; `_default_fetch` returns `[]` (`:50-51`) and its comment
   is stale ("production reads pyfinagent_pms.paper_trades" -- wrong dataset;
   production reads `financial_reports`, `_production_fns.py:215`).
2. **A future edit dropping the `IS NOT NULL` predicate.** Note 82.39's suite does
   NOT assert that predicate's presence: `test_repaired_query_returns_rows_...`
   asserts `r["pnl"] is not None` on *returned rows* (test `:229`) -- a data
   assertion over a live window, not a SQL-invariant. So today the only thing
   standing between production and this TypeError is an unasserted SQL clause.

**Adjacent, not-a-crash defect worth one line in the fix:** a FLOAT `NaN` passes
`IS NOT NULL`, and `nan > 0` is `False`, so a NaN P&L is silently graded `"loss"`.
`NaN` is also not JSON-serializable by the default encoder, so it would 400 the
insert. Guard the boundary once for both: `None` -> skip or NULL `return_pct`
(the column is NULLABLE), `NaN` -> same. Do **not** use `t.get("pnl", 0.0) or 0.0`
-- that conflates "unknown" with "flat", which is the same class of silent-zero
error phase-47.7 already fixed once (`autonomous_loop.py:3063-3067`).

---

## Q5. Is `outcome_tracking` the RIGHT destination? -- YES. It has FIVE consumers.

**The 82.39 brief's statement is literally true and materially misleading.**
`backend/services/outcome_tracker.py` contains no string `outcome_tracking` --
but it reads the table at `outcome_tracker.py:201` via
`self.bq.get_performance_stats()`. A string grep for the table name cannot see a
consumer that goes through the client method. Derive structurally instead: the
table is addressable ONLY through `BigQueryClient.outcomes_table`
(`bigquery_client.py:47`, built from `settings.bq_dataset_outcomes` = 
`"financial_reports"` at `settings.py:62` and `settings.bq_table_outcomes` =
`"outcome_tracking"` at `settings.py:63`) or a fully-qualified literal. Both
enumerated:

| Site | Direction | Anchor |
|---|---|---|
| `BigQueryClient.save_outcome` | WRITE (correct) | `bigquery_client.py:400-417` |
| `make_outcome_write_fn._write` | WRITE (**broken -- this step**) | `_production_fns.py:364-379` |
| `BigQueryClient.get_performance_stats` | READ (`COUNT`, `COUNTIF(return_pct>0)`, `AVG(return_pct)`, `COUNTIF(beat_benchmark)`) | `bigquery_client.py:481-499` |
| `SkillOptimizer._analyze_agent_performance` | READ (direct SQL: `SELECT ticker, analysis_date, return_pct, beat_benchmark`) | `skill_optimizer.py:211-215` |

...and `get_performance_stats` has four live callers:
`backend/services/outcome_tracker.py:201`, `backend/agents/meta_coordinator.py:258`,
`backend/agents/skill_optimizer.py:331`, `backend/services/perf_metrics.py:635`.

**Conclusion: fixing this write does NOT produce an orphan table.** Two of the
consumers are degrading *silently today* because the table is empty:
`get_performance_stats` returns the zero-literal `{"total_recommendations": 0,
"win_rate": 0, "avg_return": 0}` (`bigquery_client.py:499`), and
`skill_optimizer` "degrades to neutral scores" -- its own comment at
`skill_optimizer.py:220-227` says the empty-outcomes path scores "every agent
against zero outcomes" at `accuracy=0.5/sample_size=0`. So the write fix has a
real downstream consumer that is currently making decisions on a zero-row table.

**Caveat to state in the contract:** the other producer,
`autonomous_loop._learn_from_closed_trades`, is gated by
`settings.paper_learn_loop_enabled` = **False** (`backend/config/settings.py:34`;
operator token still owed per `handoff/away_ops/pending_tokens.json:207`). So
after 82.48, `nightly_outcome_rebuild` becomes the **only** live producer of
`outcome_tracking`. That raises the stakes on getting `return_pct` semantics right
(Q1) -- and it means a duplicate-row policy matters: `save_outcome` APPENDS, it is
not an upsert (`autonomous_loop.py:3057-3060`), and the nightly job re-reads a
rolling 30-day window every night, so **without a dedup key the same SELL will be
written ~30 times**. That is a NEW defect the fix would introduce; the contract
must handle it (idempotency is per-day via `IdempotencyKey.daily`, which prevents
double-runs on one day, NOT re-writes across days).

---

## Q6. Guard-design -- what would make each criterion's guard VACUOUS here

**(a) Key-vs-schema check that passes because it read a stale snapshot.**
The snapshot is a file in the repo (`_schema_snapshot.json`, last written
2026-08-05). If someone adds a column to the live table, or the write starts
emitting a key that a *future* live schema lacks, the offline check still goes
green. Worse: `load_snapshot` returning `{}` for a missing table would make
`emitted - schema_cols` = everything OR `{} - emitted` = nothing depending on
which side you write first -- an empty oracle makes every consumer look valid
(`schema_oracle.py:41-43`). *Countermeasures:* assert the resolved column set is
non-empty AND contains the 3 REQUIRED names before comparing; add a
`snapshot_drift`-based test (`schema_oracle.py:136`) so live/snapshot divergence
is red; prefer the live `client.get_table(...).schema` when creds exist.
*Mutation to run:* delete `outcome_tracking` from a copy of the snapshot and
confirm the guard goes RED, not green.

**(b) End-to-end fixture that "persists" into a mock.**
A `MagicMock()` client whose `insert_rows_json` returns `[]` will make any
assertion of the form `assert n == len(records)` pass while the production row
shape is still wrong -- the mock has no schema, so it cannot reject anything.
This is precisely abseil's warning: "With stubbing, there is no way to ensure the
function being stubbed behaves like the real implementation," and Google's own
retrospective, "we suffered greatly given that they required constant effort to
maintain while rarely finding bugs" (*SWE at Google* ch. 13). Note the shape of
the trap here: the CURRENT code already returns `len(records)` on a mock -- i.e.
a mock-based "it persisted N rows" test **passes today, against the broken
writer**. *Countermeasure:* the persistence assertion must be a real insert +
`SELECT` read-back (throwaway table, Q2 iv); reserve mocks for the *rejection*
path, where you are asserting our reaction to a documented API response, not the
API itself. *Mutation to run:* revert `_write` to today's row shape and confirm
the E2E test goes RED.

**(c) Alert guard that fires on every call.**
Assert both directions -- rejection alerts exactly once AND a clean write alerts
zero times -- with the fixture precondition asserted in both
(82.39 `:259-268` and `:271-280`). Additional vacuity risks specific to this
step: (i) `assert alert.called` is satisfied by an alert raised somewhere else in
the same test; pin `kw["error_type"]` and `kw["source"]`; (ii) `severity` must be
asserted `== "P1"` -- a P2 regression is invisible in behaviour on this machine
because P2 is dropped when `slack_webhook_url` is empty; (iii) patching the wrong
(non-existent) module-scope name, see Q3.
*Mutation to run:* change `severity` to `"P2"` and confirm RED; move the alert
call outside the `if errors:` branch and confirm the zero-alert test goes RED.

**(d) NULL-pnl test whose fixture cannot represent a NULL.**
Two concrete ways it goes vacuous here. First, `{"pnl": 0.0}` instead of
`{"pnl": None}` -- the row grades `"loss"`, no exception, test green, defect
untouched. Second, **omitting the key entirely** (`{}`): `t.get("pnl", 0.0)`
then legitimately returns `0.0` and no TypeError occurs -- the fixture exercises
the branch the code already handles and proves nothing about the branch it does
not. The fixture must be an explicit `{"trade_id": ..., "ticker": ...,
"pnl": None}`. Third, and subtler: if the test calls `run()` rather than
`_compute_outcomes` directly, the `heartbeat` swallow (Q4) means
`pytest.raises(TypeError)` **will not fire** -- `run()` returns
`{"rebuilt": 0}` instead. A `pytest.raises` around `run()` would be red for the
wrong reason today and green for the wrong reason after any fix.
*Mutation to run:* revert the coercion and confirm the test goes RED with the
`None` fixture; also assert the fixture itself (`assert row["pnl"] is None`)
so a later "cleanup" that swaps `None` for `0.0` breaks loudly.

**Cross-cutting:** criterion coverage must be derived from `git diff`, not
asserted. The two REQUIRED columns mean the write fix necessarily edits
`LEDGER_FETCH_SQL` (Q1); a guard suite scoped only to `make_outcome_write_fn`
would leave the fetch change ungated while reporting full coverage.

---

## Internal code inventory

| File:lines | Role | Status |
|---|---|---|
| `backend/slack_bot/jobs/_production_fns.py:357-381` | `make_outcome_write_fn` -- the broken writer; docstring documents a schema that never existed (`:360-361`) | **DEFECT (this step)** |
| `backend/slack_bot/jobs/_production_fns.py:228-235` | `LEDGER_FETCH_SQL`, plain literal, 7 projected columns | Correct but **must be extended** (3 more columns needed) |
| `backend/slack_bot/jobs/_production_fns.py:245-295` | `build_ledger_fetch_sql` seam + `_ROLLING_PREDICATE` no-op assertion | Reuse as-is |
| `backend/slack_bot/jobs/_production_fns.py:326-354` | `_alert_fetch_failure` -- P1, function-local import, fail-open squared | Reuse shape for the write alert |
| `backend/slack_bot/jobs/nightly_outcome_rebuild.py:26-34` | `run()`; `_compute_outcomes` at `:27` is OUTSIDE the `try` at `:28-32` | Confirmed |
| `backend/slack_bot/jobs/nightly_outcome_rebuild.py:37-47` | `_compute_outcomes`; `.get("pnl", 0.0) > 0` NULL trap | Latent (Q4) |
| `backend/slack_bot/jobs/nightly_outcome_rebuild.py:50-51` | `_default_fetch` comment cites the WRONG dataset (`pyfinagent_pms`) | Stale comment |
| `backend/slack_bot/job_runtime.py:66-114` | `@contextmanager heartbeat`; `except Exception` at `:105-108` does not re-raise | Explains the silent zero |
| `backend/services/paper_trader.py:582-585,617,637` | THE writer of `realized_pnl_pct` = `((price-entry)/entry)*100` | Authoritative for Q1 |
| `backend/services/autonomous_loop.py:3024-3032,3061-3082` | learn-loop fallback: the `return_pct := realized_pnl_pct` precedent + `analysis_date`/`recommendation` sources | Reuse mapping |
| `backend/db/bigquery_client.py:47,400-417,481-499` | `outcomes_table`; `save_outcome` (correct writer, also swallows `errors` at `:416-417`); `get_performance_stats` (reader) | `:416-417` is the SAME defect class, second instance |
| `backend/agents/skill_optimizer.py:211-227,331` | second reader; documents "degrades to neutral scores" on empty outcomes | Live consumer |
| `backend/db/schema_oracle.py:84,116,130,136,550-568` | live/snapshot schema + `snapshot_drift` + `dry_run` (QUERY-job only) | Reuse for Q2 |
| `backend/db/_schema_snapshot.json` | contains `financial_reports.outcome_tracking` (9 cols, 3 REQUIRED) | Offline oracle available |
| `scripts/migrations/migrate_bq_schema.py:122-144` | `OUTCOME_TRACKING_SCHEMA` + idempotent creator | Source of truth for the throwaway-table fixture |
| `backend/tests/test_phase_82_39_outcome_rebuild_query.py:247-325` | alert-guard + non-empty-surface idioms | Extend, don't duplicate |
| `backend/services/observability/alerting.py:253-267` | `raise_cron_alert_sync`, "Always fail-open: never raises out" | Assert the mock, not an exception |
| `backend/config/settings.py:34,62,63` | `paper_learn_loop_enabled=False`; outcomes dataset/table | Makes this job the sole live producer |

---

## Consensus vs debate (external)

Consensus: check `insertErrors` after every streaming insert; a 200 does not mean
the rows landed (Google guide + REST reference + the Java/PHP samples, which both
branch on `hasErrors()`/`isSuccessful()`). Consensus: prefer real implementations
over doubles for persistence assertions (abseil ch13; the practitioner blogs in
the snippet table agree).

Debate: whether to keep using `tabledata.insertAll` at all. Google's 2026 posture
steers new work to the Storage Write API (lower cost, exactly-once); practitioner
posts push the same way. Counterweight for pyfinagent: `insertAll` is not
deprecated, the whole repo already uses it, and this job writes 0-3 rows a night.
Migrating is a separate, larger decision -- **not** 82.48's scope, but worth a
one-line note in the contract so the choice is recorded rather than defaulted.

## Pitfalls (from literature)

1. A 200 + non-empty `insertErrors` is the normal failure mode -- never infer
   success from the HTTP code (Google guide).
2. Schema mismatch is **all-or-nothing**; do not report partial success counts
   without inspecting `reason` (Google guide).
3. `insert_rows()` silently drops unknown fields client-side -- moving "up" to the
   schema-aware API makes the failure quieter (python-bigquery #151).
4. Over-mocking produces suites that "rarely find bugs"; state testing over
   interaction testing (abseil ch13).
5. A fake/double must be contract-tested against the real API or it is "active
   misinformation" (abseil ch13 fidelity section).
6. `caplog` silently captures nothing when `propagate=False` or handlers are
   replaced (pytest #3697) -- log-based assertions are a vacuity vector.

## Application to pyfinagent

The fix is small and almost entirely a **reuse** exercise: build the 9-column row
from the `autonomous_loop.py:3024-3082` mapping, delegate the insert to (or share
a builder with) `bigquery_client.py:400-417`, extend `LEDGER_FETCH_SQL`
(`_production_fns.py:228-235`) by `analysis_id, risk_judge_decision, holding_days`,
add `_alert_write_failure` modelled on `_production_fns.py:326-354` at P1, and
coerce `pnl` at `nightly_outcome_rebuild.py:43-44`. Validation is a two-sided
key-vs-schema check resolved live-first / snapshot-fallback
(`schema_oracle.py:84/130/136`) plus one real-insert read-back against a
throwaway table built from `migrate_bq_schema.py:123-133`. Two things the step
description does not mention and the contract must own: the **duplicate-row**
consequence of a rolling 30-day re-read against an append-only writer, and the
identical swallowed-`errors` defect at `bigquery_client.py:416-417`.

---

## Recency scan (2024-2026)

Searched 2026-08-06 for 2024-2026 material on BigQuery streaming-insert error
semantics and on mock-vs-real persistence testing. **Two findings that change how
the contract should be worded, none that overturn the core answers:**

1. **`tabledata.insertAll` is now presented by Google as the "legacy streaming
   API"** (current doc-page title, surfaced 2026-08-06), with the Storage Write
   API recommended for new projects (lower price, exactly-once). No EOL date has
   been announced and the method remains fully supported. Effect on this step:
   reinforces the recommendation NOT to build new DML/insert plumbing (Q2 route b)
   and to reuse the existing `save_outcome` insertAll path; also worth a recorded
   note that a Storage-Write-API migration is a separate future decision.
2. **googleapis/python-bigquery was archived 2026-03-06.** The `insert_rows()`
   silent-drop behaviour documented in issue #151 will not be changed upstream, so
   the "never switch to `insert_rows`" guidance is permanent, not provisional.
   (Installed version here is 3.40.1.)

No 2024-2026 source contradicts the per-row/whole-batch semantics quoted in Q3, and
no newer guidance supersedes abseil ch13 on test doubles.

### Search-query variants run (three-variant discipline)

- Year-less canonical: `BigQuery insert_rows_json return value errors schema mismatch python client`;
  `mock-based tests give false confidence "test doubles" verify real behavior integration test anti-pattern`
- Current-year frontier (2026): `BigQuery streaming insert deprecated Storage Write API migration 2026 legacy tabledata.insertAll`
- Last-2-year window (2025): `pytest caplog propagate false negative logger not captured assertion vacuous 2025`

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://docs.cloud.google.com/bigquery/docs/streaming-data-into-bigquery | 2026-08-06 | Official doc (Google) | WebFetch | "If BigQuery encounters a schema mismatch on individual rows in the request, none of the rows are inserted and an `insertErrors` entry is returned for each row, even the rows that did not have a schema mismatch"; innocent rows get `reason: stopped`; must check `insertErrors` even on a 200; no dry-run/validation mode exists |
| 2 | https://docs.cloud.google.com/bigquery/docs/reference/rest/v2/tabledata/insertAll | 2026-08-06 | Official REST reference (page updated 2026-05-30) | curl + tag-strip (JS-rendered; per `feedback_gcloud_docs_fetch`) | Request body has **no `dryRun`**; `skipInvalidRows` default false "causes the entire request to fail if any invalid rows exist"; `ignoreUnknownValues` default false "treats unknown values as errors"; response = `{kind, insertErrors[{index, errors[ErrorProto]}]}` |
| 3 | https://github.com/googleapis/python-bigquery/issues/151 | 2026-08-06 | Vendor issue tracker (archived 2026-03-06) | WebFetch | "insert_rows() silently drops the additional columns instead"; client-side conversion "only iterates over the list of fields that are provided ignoring all the other fields" -- opposite of the API. Do NOT migrate to `insert_rows` |
| 4 | https://abseil.io/resources/swe-book/html/ch13.html | 2026-08-06 | Book chapter, *Software Engineering at Google* | WebFetch | "A real implementation should be preferred over a test double"; "With stubbing, there is no way to ensure the function being stubbed behaves like the real implementation"; state testing > interaction testing; hermetic instance "which has its life cycle controlled by the test" |
| 5 | https://docs.cloud.google.com/bigquery/docs/error-messages | 2026-08-06 | Official doc (Google) | curl + tag-strip (first ~9K chars of stripped text -- the error-code table is long; the cited rows were within the fetched text) | `invalid` (400) = "any type of invalid input other than an invalid query, such as missing required fields or an invalid table schema"; `badRequest` = DML "over table ... would affect rows in the streaming buffer, which is not supported ... typically for a few minutes, but in rare cases, up to 90 minutes" |
| 6 | https://docs.cloud.google.com/bigquery/docs/samples/bigquery-table-insert-rows-explicit-none-insert-ids | 2026-08-06 | Official code sample | curl + tag-strip | Canonical consumer pattern in two languages: branch on `response.hasErrors()` / `$insertResponse->isSuccessful()` and iterate `getInsertErrors()` / `failedRows()` printing `reason` + `message` -- i.e. inspecting the return value is the documented contract, not an optional nicety |
**Fetch-method note:** rows 2, 5, 6 used `curl` + HTML tag-strip because
`cloud.google.com` reference pages are JS-rendered and return nav-only text to
`WebFetch` (established in `feedback_gcloud_docs_fetch`); this counts as
read-in-full. **pytest-dev/pytest#3697 is deliberately NOT in this table** --
it was read as a search-result synthesis, not a full page fetch, so per
`.claude/rules/research-gate.md` ("search snippets do NOT count") it sits in the
snippet-only table below. The Q3/Q6 `caplog` claim rests on it plus the official
pytest logging doc; treat that one claim as the least-hardened in this brief.

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://cloud.google.com/bigquery/docs/write-api-streaming | Official doc | Storage Write API -- out of scope for this step's fix |
| https://docs.cloud.google.com/bigquery/docs/write-api-grpc | Official doc | same |
| https://github.com/googleapis/python-bigquery/issues/434 | Vendor issue | ConnectionError in insert_rows_json -- transport, not schema |
| https://github.com/googleapis/python-bigquery/issues/1278 | Vendor issue | passing a Table object -- not our failure mode |
| https://github.com/googleapis/python-bigquery/issues/1396 | Vendor issue | intermittent NotFound on project -- unrelated |
| https://github.com/googleapis/google-cloud-go/issues/6033 | Vendor issue | Go client, many-tables streaming -- unrelated |
| https://oneuptime.com/blog/post/2026-02-17-how-to-troubleshoot-bigquery-streaming-insert-rows-not-appearing-in-table-queries/view | Blog (community tier) | superseded by the official guide (source #1) |
| https://oneuptime.com/blog/post/2026-02-17-how-to-fix-bigquery-schema-mismatch-errors-when-loading-data-from-cloud-storage/view | Blog | load jobs, not streaming |
| https://oneuptime.com/blog/post/2026-02-17-how-to-stream-data-into-bigquery-using-the-storage-write-api/view | Blog | Storage Write API -- out of scope |
| https://medium.com/google-cloud/the-hidden-powerhouse-demystifying-the-bigquery-storage-write-api-cdbc04b78806 | Blog (Jun 2026) | recency-scan evidence only |
| https://medium.com/@bravnic/bigquery-storage-write-api-at-scale-7affcc2d7a93 | Blog | scale concerns irrelevant at 0-3 rows/night |
| https://www.codurance.com/publications/tdd-anti-patterns-chapter-2 | Practitioner | abseil ch13 is the stronger source for the same point |
| https://blog.pragmatists.com/test-doubles-fakes-mocks-and-stubs-1a7491dfa3da | Practitioner | taxonomy already covered by abseil ch13 |
| https://www.amazingcto.com/mocking-is-an-antipattern-how-to-test-without-mocking/ | Opinion blog | polemic; abseil is the citable version |
| https://fusionauth.io/blog/to-mock-or-not-mock-auth | Vendor blog | auth-specific |
| https://www.drizz.dev/post/unit-test-mocks-vs-real-objects-when-to-fake-it-and-when-not-to | Blog | duplicate of abseil's guidance |
| https://dev.to/patoliyainfotech/advanced-mocking-strategies-mastering-test-doubles-behavior-verification-5282 | Community | lowest tier |
| https://en.wikipedia.org/wiki/MockServer | Encyclopedia | tool page, not guidance |
| https://github.com/pytest-dev/pytest/issues/3697 | Upstream issue (pytest) | READ AS SEARCH SYNTHESIS ONLY -- `propagate=False` / replaced root handlers means `caplog` captures nothing, so log-text assertions are a vacuity vector. Basis for the Q3/Q6 caplog warning; not fetched in full, so it does not count toward the gate |
| https://docs.pytest.org/en/stable/how-to/logging.html | Official doc | corroborates #3697; `caplog.set_level` / `at_level` |
| https://github.com/pytest-dev/pytest/issues/7335 | Upstream issue | duplicate of #3697's mechanism |
| https://qaskills.sh/blog/pytest-caplog-assert-specific-log-level | Blog | community tier |
| https://pypi.org/project/pytest-structlog/0.5 | Package page | not used in this repo |

**Total unique URLs: 29** (6 read in full + 23 snippet-only).

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL (6: 4 official Google docs,
      1 vendor issue tracker, 1 book chapter -- all tier 1-2 of the hierarchy)
- [x] 10+ unique URLs total (29)
- [x] Recency scan (2024-2026) performed + reported (2 findings)
- [x] Full pages read (not abstracts) for the read-in-full set -- fetch method
      disclosed per row, incl. the one truncation
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module the caller scoped, plus the
      consumer set and the second swallowed-`errors` instance
- [x] Contradictions noted (82.39's "zero references" claim corrected; the step's
      "TypeError escapes" implication corrected)
- [x] Claims cited per-claim with URL + access date / file:line

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 23,
  "urls_collected": 29,
  "recency_scan_performed": true,
  "internal_files_inspected": 16,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_82.48.md",
  "gate_passed": true
}
```
