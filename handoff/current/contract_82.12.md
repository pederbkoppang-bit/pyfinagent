# Contract -- masterplan step 82.12

**Step id:** 82.12 (phase-82, priority P1, harness_required: true)
**Date:** 2026-08-05 | **Cycle:** 1

---

## 1. Research gate summary

**Brief:** `handoff/current/research_brief_82.12.md` (696 lines)
**Envelope:** `gate_passed: true` -- tier `complex`, **audit-class**:
`external_sources_read_in_full: 7`, `snippet_only_sources: 32`, `urls_collected: 39`,
`recency_scan_performed: true`, `internal_files_inspected: 20`,
`coverage: {audit_class: true, rounds: 6, dry_rounds: 3, K_required: 2, dry: true}`.

**Rail note:** the researcher's structured return was DROPPED by the Workflow rail
(`agent({schema}): subagent completed without calling StructuredOutput`). **Write-first
saved the work** -- the complete brief including the gate envelope was already on disk,
so no re-spawn was needed. This is the third dropped return this session and the second
time write-first has been the only reason the work survived.

### The gate INVERTED the step. Three findings, each re-measured by Main.

**(a) The literal sweep the step describes yields ZERO defects.** The researcher measured
534 non-test `isinstance` calls, 8 of them date-typed, 5 live -- and **all 5 are
CORRECT**. The 82.0 fix closed the only vacuous instance. `_STRING_DATE_TIMESTAMP_COLS`
is 6/6 correct today. Both numeric-named STRING columns are false positives.

This is the dangerous shape of step: *"go find more of the same"* when there is no more
of the same. It invites either an honest empty report that reads as a wasted cycle, or
padded false positives. **Neither is acceptable.** The brief's re-framing is adopted:

> Build the **schema oracle** the codebase has never had, prove the current surface is
> clean against it, and wire it as a standing check so the next instance cannot land.

"Zero remaining vacuous guards" then becomes the SUCCESS condition, proved by
construction, rather than an embarrassment.

**(b) THE SWEEP FOUND A LIVE P1 -- of the same root cause, a different symptom.**
`backend/slack_bot/jobs/_production_fns.py` (`_fetch`, the `nightly_outcome_rebuild`
source) selects `timestamp` and `realized_pnl` from `paper_trades`. **Those columns do
not exist.** MAIN VERIFIED AGAINST THE LIVE SCHEMA:

```
paper_trades columns: 18
  'timestamp'          present=False
  'realized_pnl'       present=False
  'created_at'         present=True  type=STRING
  'realized_pnl_pct'   present=True  type=FLOAT
```

The query 400s, and the `except Exception` at the bottom of `_fetch` swallows it and
returns `[]`. So `nightly_outcome_rebuild` has been running on **zero trades**, silently.
Same root cause as 82.0 (an unverified assumption about a BQ column, swallowed
fail-open), different symptom (wrong NAME, not wrong TYPE).

**(c) `dry_run` is the right instrument, and it is free.** MAIN VERIFIED all four
predicted cases live at $0:

```
FAIL | broken _production_fns query     | 400 Unrecognized name
OK   | fixed equivalent                 | bytes_billed=4936   (dry run: $0)
FAIL | STRING date >= CURRENT_DATE()    | 400 No matching signature for >=
OK   | STRING date >= '2026-01-01'      | bytes_billed=56808  (dry run: $0)
```

`QueryJobConfig(dry_run=True)` makes **BigQuery itself** parse and type-check against the
live schema. Case 3 confirms STRING has an empty coerce-to set; case 4 confirms the
lexical-ISO-date pattern is correct by construction. A dry-run check would have caught
(b) on the day it shipped.

---

## 2. Hypothesis

The date/`isinstance` form of this defect class is already extinct, so a "find more"
sweep cannot succeed on its own terms. What is missing is the **oracle**: nothing in the
codebase can currently answer "does this column exist, and what type is it?" Building
that oracle, proving the current surface clean against it, and wiring it as a standing
check both discharges the step's criteria honestly and closes the door the class came
through -- including the NAME-assumption variant that the type-only framing would miss.

---

## 3. Immutable success criteria (copied VERBATIM from `.claude/masterplan.json`)

1. `the sweep enumerates BQ columns declared STRING that are consumed as dates/numbers, derived from live table schemas rather than from a hand-written list, and the derived scope is asserted non-empty`
2. `every guard applied to a BQ-read value in that scope is classified vacuous / correct / needs-coercion, with file:line for each`
3. `every hit classified vacuous is either fixed or has its own queued follow-up step`
4. `a test feeds each fixed guard the PRODUCTION column type and asserts the guard fires on bad input, so a fixture that cannot represent the production failure cannot pass`

**Verification command (immutable):**

```
source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_12_string_column_guards.py -q
```

**Reading of criterion 4, stated up front so it cannot be quietly satisfied by an empty
set.** "Each fixed guard" could be read as vacuous if zero guards need fixing this
cycle. That reading is rejected. The guard fixed FOR THIS DEFECT CLASS is
`cache.preload_macro`'s staleness gate (repaired in 82.0, the canonical instance), and it
is in scope for criterion 4 here: the test must feed it the **production column type**
(`str`, per the oracle -- not a hardcoded literal) and assert it fires on bad input.

---

## 4. Plan

### 4.1 The oracle -- `backend/db/schema_oracle.py` (NEW)

Two-sided derivation. **A regex over column NAMES is itself a hand-written list** (of
name tokens) and would satisfy criterion 1 in letter while violating it in spirit; the
brief measured that it under-covers (`calendar_events.window`,
`analysis_results.overall_reliability`, `strategy_decisions.decay_attribution`).

- **Schema side:** `list_tables()` + `get_table().schema` over the project's datasets ->
  `{table: {column: (field_type, mode)}}`. Cached to a checked-in JSON snapshot so the
  test runs without ADC in CI; a separate live-refresh path diffs snapshot-vs-live, and
  **that diff IS the schema-drift detector**.
- **Consumer side:** every read site -- SQL literals naming a known table, and Python
  guards over a BQ-read value. No name filter.
- **Join:** flag only where a consumer applies date/number semantics to a column the
  oracle says is STRING. The name heuristic, if used at all, is a **reporting aid, never
  the gate**.
- **Assert non-empty at every stage** -- `files_scanned > 0`, `tables_resolved > 0`,
  `columns_in_oracle > 0`, and the criterion-1 scope itself. The researcher hit this for
  real: its first scanner reported "0 unknown identifiers", which looked like a clean
  bill of health but was a relative-path bug plus a non-greedy regex capturing `sunny`
  from the fully-qualified table name. **A checker that scans nothing reports exactly
  what a clean codebase reports.**

### 4.2 The standing check -- SQL validation via `dry_run`

Extract each SQL literal, substitute placeholder params, `dry_run` it, assert no
`BadRequest`. $0. Marked so it can be skipped without ADC, but the snapshot-based
identifier check must still run offline so the guard is not silently disabled in CI.

### 4.3 Classification, with the traps encoded

Every hit classified **VACUOUS** / **CORRECT** / **NEEDS-COERCION**, with file:line.
Traps that must not produce a cry-wolf result:

1. **Lexical ISO-date comparison is CORRECT by construction** and must not be flagged.
   Boundary to encode: ISO-date-STRING vs ISO-date-STRING = OK; STRING-holding-a-number
   vs anything = NOT OK (`'9' > '10'`).
2. **`datetime` is a subclass of `date`**, so `isinstance(ts, date)` is True for a
   TIMESTAMP. The correct plain-date test is
   `isinstance(d, date) and not isinstance(d, datetime)`.
3. **INERT is not VACUOUS.** Branches that never execute in production but are
   *fallthrough optimisations* rather than safety decisions are reported at lower
   severity. Only a dead branch carrying a **safety decision** is VACUOUS.
4. **`_STRING_DATE_TIMESTAMP_COLS` is 6/6 correct today** -- a latent-drift hazard, not a
   live defect. Converting it to an oracle lookup is a refactor, not a bug fix.
5. The drifted test tag at `test_phase_23_2_11...py:38` is benign (both branches return
   byte-identical results) -- report as drift, not breakage.
6. `stop_advanced_at_R` is a TIMESTAMP string, not an R-multiple -- name-based scoping
   mis-reads it.

### 4.4 Criterion-4 tests -- three parts per fixed guard

Pattern copied from `backend/tests/test_phase_82_0_macro_ingestion.py`, which already
builds rows with `"date": d.isoformat()` and asserts the fixture's own type:

1. **Precondition:** the fixture emits the type **the ORACLE declares** -- read from the
   oracle, not hardcoded `"STRING"`, so schema drift breaks the test.
2. **Positive:** the guard fires on bad input (stale / unparseable / None / `""`).
3. **Negative:** the guard does not fire on good input.

Then **mutate**: delete the guard body and confirm the suite goes red, naming the tests
actually killed via `pytest -rf`.

### 4.5 Queued, NOT absorbed

- **The `_production_fns` P1 (finding b)** -- its own step. It is a live data defect with
  a real consequence (a nightly job on zero rows) and fixing it inline would both widen
  this step's surface and bury it.
- `preload_prices` / `preload_fundamentals` have **no staleness gate at all** (missing
  guard, not vacuous guard) -- its own step.
- `pyrightconfig.json` pins `.venv312` / Python 3.12 while the project runs 3.14, and
  sets no `typeCheckingMode` (measured: 0 diagnostics at default) -- its own step.

### 4.6 Out of scope

No change to any query's semantics, no schema migration, no change to
`_STRING_DATE_TIMESTAMP_COLS`'s current behaviour, no live-position or credential touch.

---

## 5. Files expected to change

| File | Change |
|------|--------|
| `backend/db/schema_oracle.py` | NEW -- live+snapshot schema oracle, non-empty assertions |
| `backend/db/_schema_snapshot.json` | NEW -- checked-in snapshot so the guard runs without ADC |
| `backend/tests/test_phase_82_12_string_column_guards.py` | NEW -- the immutable verification target |
| `.claude/masterplan.json` | new queued steps (4.5); 82.12 status flip LAST |

---

## 6. References

- `handoff/current/research_brief_82.12.md` (7 sources read in full; audit-class coverage dry at K=2 after 6 rounds)
- BigQuery data-types + conversion-rules docs; `google-cloud-bigquery` RowIterator type mapping (STRING->str, DATE->date, TIMESTAMP->datetime, FLOAT->float, measured live)
- BigQuery `dry_run` semantics (not billed)
- Dead-code / unsatisfiable-predicate analysis; mutation-testing guidance for guard clauses
- Internal: `backend/backtest/cache.py` (the 82.0 fix), `backend/services/cycle_health.py` (`_STRING_DATE_TIMESTAMP_COLS`), `backend/tests/test_phase_82_0_macro_ingestion.py` (fixture-precondition pattern)
