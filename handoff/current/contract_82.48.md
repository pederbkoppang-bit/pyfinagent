# Contract -- phase-82.48

**Step:** 82.48 (P1) -- the WRITE side of `nightly_outcome_rebuild` is broken.
**Date:** 2026-08-06. **Cycle:** 1.
**Research gate:** PASSED -- `handoff/current/research_brief_82.48.md`,
`gate_passed: true`, 6 sources read in full, 29 URLs, recency scan, 16 internal
files.

---

## 1. A CORRECTION I OWE FIRST, because I published the error today

82.39's brief claimed `outcome_tracking` has **zero** consumers, and I repeated
that in 82.39's artifact and harness-log entry as a *correction to the step*.
**It is wrong.** Re-derived here:

```
bigquery_client.py:481   def get_performance_stats(...)   <- reads outcome_tracking
callers:
  backend/services/outcome_tracker.py:201
  backend/agents/meta_coordinator.py:258
  backend/agents/skill_optimizer.py:331
  backend/services/perf_metrics.py:635
```

`outcome_tracker.py` has no *literal* mention of the table -- it reads it through
`self.bq.get_performance_stats()`. A name grep cannot see that, and a name grep
is what produced the claim. **Same class as every other failure this week: a set
derived by matching a string instead of by following the code.** And
`skill_optimizer.py` documents that it currently degrades to neutral scores
*because the table is empty* -- so the broken write has a real, live consequence
that my "correction" talked the reader out of.

Corrected in 82.39's record as part of this step.

## 2. What is actually broken, measured

- `outcome_tracking`: 0 rows, 9 columns, 3 REQUIRED (`ticker`,
  `analysis_date`, `recommendation`).
- `make_outcome_write_fn` emits `{trade_id, ticker, pnl, outcome, recorded_at}`.
  Only `ticker` overlaps; both REQUIRED columns are unsupplied. Its docstring
  documents a schema that never existed.
- `insert_rows_json` **returns** `[{index, errors}]` per rejected row and never
  raises. Measured against installed `google-cloud-bigquery 3.40.1`. On a schema
  mismatch Google rejects the **entire batch**, tagging innocent rows `stopped`.
  The code swallows the return.

## 3. Decisions, and none of them is a guess

1. **`return_pct := realized_pnl_pct` is already settled in-repo, not a judgment
   call I get to make.** `paper_trader.py:583-585` defines it as
   `((price - entry_price) / entry_price) * 100` -- percent of per-share average
   entry price, already x100, SELL legs only. `autonomous_loop.py:3063-3082`
   already maps it to `return_pct`, and its phase-47.7 comment is an explicit
   retraction of the opposite mapping. The step forbade inventing a notional
   derivation; none is invented.
2. **Delegate the row shape to the existing correct writer.**
   `bigquery_client.save_outcome` (:400-417) already builds the right 9 columns.
   Reuse it rather than author a second shape -- but its own error handling
   swallows the return, which §2 shows is the defect, so this step supplies a
   write that INSPECTS the return.
3. **The fetch must be extended.** `analysis_date` and `recommendation` are
   REQUIRED and are not in 82.39's `LEDGER_FETCH_SQL`; their sources are
   `created_at` / `analysis_id` and `risk_judge_decision`, plus `holding_days`.
   So this step necessarily edits the query 82.39 shipped.
   `price_at_recommendation` has **no source on `paper_trades`** -- left NULL
   rather than back-derived, and that is stated rather than silently done.
4. **A defect the fix would INTRODUCE, so it is handled here, not queued:** the
   fetch reads a rolling 30-day window nightly and the writer APPENDS. The daily
   idempotency key prevents a double-run on one day, not re-writes across 30
   days -- the same SELL would land ~30 times. The write therefore skips rows
   already present, by reading back the existing `(ticker, analysis_date)` pairs.
5. **The NULL-pnl crash is real but NOT reachable in production.**
   `.get("pnl", 0.0)` returns `None` for a present-but-`None` key, and the call
   is outside the `try` -- but `LEDGER_FETCH_SQL` keeps NULLs out, and
   `heartbeat` is a `@contextmanager` whose `except` does not re-raise, so
   `contextlib` suppresses it anyway. Fixed regardless (criterion 4 requires
   it), and the unreachability is stated so nobody reads the fix as a live-bug
   repair. Adjacent and worth knowing: a FLOAT `NaN` passes `IS NOT NULL`, and
   `nan > 0` is False, so it would be silently graded `"loss"`.

## 4. Immutable success criteria (verbatim)

1. "the keys the write function emits are validated against the destination
   table's real schema, asserted by a test that FAILS against the current
   emitted shape and names the missing REQUIRED columns"
2. "a fixture drives the write path end to end and asserts a row is actually
   persisted, so a repair cannot pass on shape agreement alone"
3. "a write that is rejected by BigQuery emits an operator-visible signal rather
   than being swallowed, asserted by a test capturing the emitted signal -- note
   insert_rows_json RETURNS errors rather than raising"
4. "_compute_outcomes is shown to handle a NULL pnl without raising, asserted by
   a fixture containing a row whose pnl value is None"

**Verification command (immutable):**
`source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_48_outcome_write_schema.py -q`

**Criterion 2 means a REAL insert, and it cannot be production.** Streaming
insertAll has no dry-run (no `dryRun` field in the REST body;
`schema_oracle.dry_run` is a query-job config and structurally cannot cover it),
and DML against the streaming buffer is blocked for up to 90 minutes -- so a
streamed test row could not be cleaned up and would pollute
`get_performance_stats` for the four consumers in §1. The fixture therefore
creates a **throwaway table** from the real schema, inserts, reads back, and
drops it in teardown.

## 5. Guard traps this step must avoid (named in advance)

- A key-vs-schema check that passes because it reads a **stale snapshot**: the
  resolved column set is asserted non-empty and to contain the 3 REQUIRED names.
- An end-to-end fixture that "persists" into a **mock**: a `MagicMock` returning
  `[]` makes "N rows persisted" pass TODAY against the broken writer.
  Persistence is a real read-back.
- An alert guard that **fires on every call**: negative control plus
  `severity == "P1"` pinned (a P2 is dropped while `slack_webhook_url` is empty),
  patched at `backend.services.observability.alerting.raise_cron_alert_sync`
  because `_production_fns` has no module-scope name.
- A NULL-pnl fixture that uses `0.0` or **omits the key**: `.get("pnl", 0.0)`
  legitimately returns `0.0` when absent, so such a fixture proves nothing. And
  `pytest.raises` around `run()` will never fire, because `heartbeat` suppresses.

## 6. Non-scope

`bigquery_client.py:416-417` swallows `insert_rows_json`'s return on the very
writer this step delegates to -- a second instance of the same class. Queued, not
fixed here. No change to `paper_learn_loop_enabled`. No live positions.

## 7. References

- `handoff/current/research_brief_82.48.md`
- Google Cloud: streaming insertAll error semantics; streaming-buffer DML limits
- `googleapis/python-bigquery#151` (why NOT `insert_rows()`)
- Internal: `backend/slack_bot/jobs/_production_fns.py`,
  `backend/slack_bot/jobs/nightly_outcome_rebuild.py`,
  `backend/db/bigquery_client.py:400-417,481-489`,
  `backend/services/paper_trader.py:583-585`,
  `backend/services/autonomous_loop.py:3063-3082`,
  `backend/db/_schema_snapshot.json`,
  `backend/tests/test_phase_82_39_outcome_rebuild_query.py`
