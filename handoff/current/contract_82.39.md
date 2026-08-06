# Contract -- phase-82.39

**Step:** 82.39 (P1) -- `nightly_outcome_rebuild`'s BQ fetch selects columns that
do not exist, so the job has been running on zero trades, silently.
**Date:** 2026-08-06. **Cycle:** 1.
**Research gate:** PASSED -- `handoff/current/research_brief_82.39.md`,
`gate_passed: true`, **audit_class: true** with `dry: true` after 10 rounds /
2 dry rounds, 8 external sources read in full, 20 URLs, 18 internal files.

---

## 1. Research-gate summary

Every number below was re-measured by Main against live BigQuery, not taken on
trust.

1. **Both of the step's claims reproduce exactly.**
   `financial_reports.paper_trades` = 65 rows / 18 columns; `timestamp`
   **absent**, `realized_pnl` **absent**; the real columns are `created_at`
   (STRING, REQUIRED) and `realized_pnl_pct` (FLOAT, NULLABLE).
   `financial_reports.outcome_tracking` = **0 rows** / 9 columns.
2. **Dry runs are free and are a real validator.** Google's documentation
   states verbatim: *"Dry runs don't use query slots, and you are not charged"*.
   Proven live to catch this exact defect (`Unrecognized name: timestamp at
   [5:27]`). They do **not** validate results, so criteria 1 and 2 cannot be
   merged -- criterion 1 proves the query is legal, criterion 2 proves it
   selects data.
3. **The repaired predicate is
   `SAFE.TIMESTAMP(created_at) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)`.**
   Repo precedent: `cycle_health.py` already lists `('paper_trades','created_at')`
   in `_STRING_DATE_TIMESTAMP_COLS`. **Trap:** `SAFE.TIMESTAMP` on a *native*
   TIMESTAMP column returns 400 "SAFE with function timestamp is not
   supported" -- 31 such columns exist -- so the idiom is per-column, never
   portable.
4. **What the repaired query would actually return, measured:** BUY 33 rows /
   0 with pnl; SELL 32 / 32 with pnl. The step's 32-32 and 0-33 confirmed.
   SELLs by month: 2026-05 = 8, 2026-06 = 20, 2026-07 = 4.
   **CRITERION-2 TIME BOMB:** a rolling 30-day window returns 3 rows today and
   **0 after 2026-08-26**. A fixture pinned to the rolling window would pass now
   and silently rot. Pin `2026-06-01..2026-07-01` (20 rows) instead.
5. **Dead since the file's first commit** `2301b977` (2026-05-11) -- never
   correct, not a regression. Lower bound 87 days. **But there is NO durable
   receipt** (`IdempotencyStore` is an in-memory `set()`, `heartbeat` sinks to
   `logger.info`), so "87 nightly runs" is NOT claimable and this contract does
   not claim it.
6. **The step's own consequence claim is FALSE and is corrected here.** The step
   says outcome tracking "feeds agent memories (BM25) and the learning loop, so
   a long-dead rebuild may mean the reflection corpus has been frozen".
   Measured: `outcome_tracker.py` has **0 references** to `outcome_tracking`;
   the real writer is `autonomous_loop.py` and it is gated by
   `settings.py` `paper_learn_loop_enabled = False`. The corpus is not frozen by
   this defect. The defect is still real -- the job produces nothing -- but the
   blast radius is smaller than the step asserts, and saying so is the point.
7. **A SECOND LIVE DEFECT of the identical class, found in audit round 8 and
   confirmed by Main:** `backend/api/cost_budget_api.py` selects
   `input_tokens` / `output_tokens` from `pyfinagent_data.llm_call_log`, whose
   real columns are `input_tok` / `output_tok` (measured: 5519 rows, 15 columns,
   `input_tokens` present=False). Same fail-open swallow, so `llm_tokens_today`
   has been permanently null. **`derive_scope` CANNOT SEE IT** -- see finding 8.
8. **The criterion-4 instrument has narrow recall, and this must be stated
   loudly rather than discovered later.** Measured `derive_scope(ORACLE)`:
   `files_scanned=296`, `sql_literals=13`, `tables_resolved=1` **of 33 tables in
   the oracle**, `unknown_columns=2` (both real, zero false positives),
   `scope=0`. Twenty backend files build table refs by f-string and are
   invisible to it; `scripts/` is never scanned. So a criterion-4 "clean" report
   is **false assurance**, and finding 7 is the proof: a live defect of exactly
   the class the sweep exists to find sits outside its envelope.

### The criterion-4 wording problem, and why the step is still closeable

Criterion 4 says *"its derived scope asserted non-empty"*. Measured today,
`derive_scope(ORACLE)["scope"]` is **`[]`** -- so a literal assertion would be
red before the fix, which is the uncloseable-by-construction trap that killed
step 81.0.

It is closeable here, and **precisely because of this step's fix**. `scope` is
"(table, column) where the oracle declares STRING and a consumer applies
date/number semantics". It is empty today because the only resolved table's
query applies `TIMESTAMP_TRUNC` to a column that **does not exist**. The
repaired query applies `SAFE.TIMESTAMP(...)` to `created_at`, which **is** a
STRING column in the oracle. Verified directly:

```
>>> so._semantics_for(repaired_sql, "created_at")
'date'
```

So the fix moves `created_at` into `scope`, and the assertion becomes
satisfiable *as a consequence of the repair* rather than in spite of it. No
criterion is amended.

### Blast radius -- closing this step turns TWO currently-green tests RED

Measured, both in `backend/tests/test_phase_82_12_string_column_guards.py`:

- `test_query_selecting_nonexistent_columns_is_detected` asserts
  `("timestamp", "backend/slack_bot/jobs/_production_fns.py") in flagged`.
  Repairing the query removes the flag.
- `test_the_nonexistent_column_defect_is_queued_as_its_own_step` requires an
  **OPEN** step whose name contains all of
  `("_production_fns", "paper_trades", "timestamp", "realized_pnl")`. Measured
  over all **1115** steps: 82.39 is the **only** match. 82.48 carries only 2 of
  the 4 tokens (`_production_fns`, `realized_pnl`), so it does not substitute.

Both must be rewritten *preserving intent*, disclosed -- not deleted, and not
left to break.

---

## 2. Hypothesis

The fetch's SELECT and WHERE both reference phantom columns; BigQuery 400s; the
broad `except Exception` at the bottom of `_fetch` logs a warning and returns
`[]`. So `_compute_outcomes` receives an empty list, the write writes nothing,
and the job reports a successful no-op. Repairing the query against the live
schema makes it select real rows; routing the failure branch through the
canonical alert seam makes any future failure audible instead of silent.

Falsifiable predictions:
- The current query dry-runs to a 400 naming `timestamp`; the repaired query
  dry-runs valid.
- The repaired query over `2026-06-01..2026-07-01` returns 20 rows.
- A fetch whose BQ call raises emits exactly one `P1` alert; a successful fetch
  emits none.

## 3. Immutable success criteria (verbatim from `.claude/masterplan.json`)

1. "the nightly_outcome_rebuild query is validated against the live schema by a
   BigQuery dry run (which is not billed) and reported valid, asserted by a test
   that fails on the current query"
2. "a fixture proves the repaired query returns rows for a period where
   paper_trades demonstrably has trades, so the fix is not merely syntactically
   valid but actually selects data"
3. "a run in which the BQ fetch fails emits an operator-visible signal rather
   than only a warning log and an empty list, asserted by a test capturing the
   emitted signal"
4. "the unknown-column sweep over the repo is re-run, its derived scope asserted
   non-empty, and every remaining member is fixed or has its own queued
   follow-up step"

**Verification command (immutable):**
`source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_39_outcome_rebuild_query.py -q`

### Seam-and-mutant map

| # | Seam | Guard drives | Mutant that must kill it |
|---|---|---|---|
| 1 | the SQL the production closure actually issues | a live dry run of the *production* SQL string, obtained from the production builder -- not a copy pasted into the test | revert the SQL to `timestamp` / `realized_pnl`; the dry run must 400 |
| 2 | the same builder, over a fixed window | a live query over `2026-06-01..2026-07-01` asserting 20 rows | make the builder emit `WHERE FALSE`; drop the `realized_pnl_pct` projection |
| 3 | the `except` branch of `_fetch` | the production closure with the BQ client forced to raise; capture the alert payload | delete the alert call; downgrade severity to `P2` (dropped while the webhook is empty) |
| 4 | `derive_scope` over the real repo | the live sweep, asserting `scope` non-empty AND every `unknown_columns` member fixed-or-queued | re-introduce a phantom column; remove the queued step for the `cost_budget_api` defect |

## 4. Plan

**D1 -- extract a real seam, then repair it.** The SQL currently lives inline
inside `_fetch`, so no test can reach it without copying it -- and a copied
string is the classic guard that proves nothing about production. Extract
`build_ledger_fetch_sql(project, start, end)` and have `_fetch` call it. Repair:
`created_at` + `realized_pnl_pct AS pnl`, predicate
`SAFE.TIMESTAMP(created_at) >= ...`, keep `realized_pnl_pct IS NOT NULL`.

**D2 -- make the swallow audible.** Keep the broad `except` (a nightly job must
not crash the scheduler) but emit `raise_cron_alert_sync` at **P1** from the
failure branch, reusing the seam phase-82.11 established. Never `P2` -- with
`slack_webhook_url` empty a P2 is logged and dropped.

**D3 -- the criterion-4 sweep, honestly reported.** Re-run `derive_scope`,
assert `scope` non-empty and every `unknown_columns` member fixed or queued,
**and** record the instrument's measured recall limits (1 of 33 tables
resolved; f-string table refs invisible; `scripts/` unscanned) so nobody reads
a clean sweep as a clean repo. Queue the `cost_budget_api` defect as its own
step, and queue a step to widen the sweep's recall.

**D4 -- the two 82.12 tests, rewritten preserving intent.**
- The detection test's intent is *"the checker can see this defect class"*.
  After the repair the live defect is gone, so the guard becomes: the live scan
  no longer flags these identifiers **and** the checker still detects a
  synthetic instance of the same defect (recall preserved, driven on a fixture).
- The queued-step test's intent is *"the defect is queued, not absorbed"*.
  Rewrite to the disjunction 82.39's own criterion 4 uses: the defect must be
  **either** named by an open step **or** demonstrably fixed. Those two states
  are disjoint, not a subset relation, so this is not the `A or B` escape hatch.

## 5. Non-scope

- **The WRITE half is NOT fixed here.** `make_outcome_write_fn` emits 5 keys of
  which only `ticker` exists on `outcome_tracking`, and both REQUIRED columns
  are unsupplied, so the job will STILL write 0 rows after this step. That is
  **82.48**, and this contract exists partly to make sure 82.39 is not closed
  believing the job works. Per Google's streaming documentation a schema
  mismatch means **none** of the rows insert, and `insert_rows_json` **returns**
  errors rather than raising.
- `_compute_outcomes`' NULL-pnl crash is also 82.48. This step keeps the
  `realized_pnl_pct IS NOT NULL` predicate, which happens to keep NULLs out of
  it, but that is a side effect and not a fix.
- No live positions touched; paper trading untouched.

## 6. References

- `handoff/current/research_brief_82.39.md` (audit-class gate, `dry: true`)
- Google Cloud, *Estimate and control costs* / dry-run billing guarantee
- Google Cloud, streaming insert schema-mismatch semantics
- Internal: `backend/slack_bot/jobs/_production_fns.py`,
  `backend/slack_bot/jobs/nightly_outcome_rebuild.py`,
  `backend/db/schema_oracle.py:216-234,453-525,550`,
  `backend/services/cycle_health.py` `_STRING_DATE_TIMESTAMP_COLS`,
  `backend/services/observability/alerting.py:253-287`,
  `backend/services/autoresearch_health.py` (the 82.11 template),
  `backend/tests/test_phase_82_12_string_column_guards.py:403-456`
