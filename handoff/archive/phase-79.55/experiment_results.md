# Experiment Results -- phase-82.48 (cycles 1-2)

**Step:** 82.48 (P1). **Date:** 2026-08-06.
**Contract:** `handoff/current/contract_82.48.md`.
**Research brief:** `handoff/current/research_brief_82.48.md` (`gate_passed: true`).

---

## 1. A CORRECTION I OWE, because I published the error earlier today

82.39's brief claimed `outcome_tracking` has **zero** consumers, and I repeated
it in 82.39's artifact and harness-log entry **as a correction to the step**.
It is wrong. Re-derived:

```
bigquery_client.py:481  get_performance_stats()  <- reads outcome_tracking
callers: outcome_tracker.py:201, meta_coordinator.py:258,
         skill_optimizer.py:331, perf_metrics.py:635
```

`outcome_tracker.py` reads the table through `self.bq.get_performance_stats()`
and never names it -- so **a name grep cannot see it, and a name grep is what
produced the claim.** Same class as every other failure this week: a set derived
by matching a string instead of by following the code. And `skill_optimizer`
documents that it degrades to neutral scores *because the table is empty* -- a
live consequence my "correction" talked the reader out of.

## 2. What was broken, and what shipped

| | |
|---|---|
| Destination | `outcome_tracking`: 0 rows, 9 columns, 3 REQUIRED |
| Emitted (pre-fix) | `{trade_id, ticker, pnl, outcome, recorded_at}` -- only `ticker` overlaps; both REQUIRED columns unsupplied |
| Why silent | `insert_rows_json` RETURNS per-row errors and never raises; a schema mismatch rejects the ENTIRE batch. The return was swallowed. |

Shipped: `build_outcome_row` emitting the real 9 columns; `_alert_write_rejected`
dispatching **P1** (never P2 -- dropped while `slack_webhook_url` is empty);
`_drop_already_written`; the fetch extended with `analysis_id`,
`risk_judge_decision`, `holding_days`; `_compute_outcomes` made NULL-safe.

## 3. Decisions, none of them guessed

- **`return_pct := realized_pnl_pct` was already settled in-repo.**
  `paper_trader.py:583-585` defines it as `((price-entry_price)/entry_price)*100`
  -- percent of per-share entry, already x100, SELL legs only -- and
  `autonomous_loop.py:3063-3082` already maps it to `return_pct`, its phase-47.7
  comment being an explicit retraction of the opposite mapping. Nothing invented.
- **`price_at_recommendation` is left NULL.** `paper_trades` carries no entry
  price on the SELL leg; back-deriving one would be fabricating a number.
- **A defect the fix would have INTRODUCED, handled here rather than queued:**
  the fetch reads a rolling 30-day window nightly and the writer APPENDS. The
  daily idempotency key stops a second run *today*, not a re-write of the same
  30 days *tomorrow* -- one SELL would land ~30 times. `_drop_already_written`
  reads back existing `(ticker, analysis_date)` pairs, fail-open.
- **The NULL-pnl crash is real but was NOT reachable in production** -- the fetch
  predicate keeps NULLs out and `heartbeat` suppresses the raise anyway. Fixed
  because criterion 4 requires it; the unreachability is stated so nobody reads
  it as a live-bug repair. Adjacent, NOT fixed: a FLOAT `NaN` passes
  `IS NOT NULL` and `nan > 0` is False, so a NaN is silently graded `"loss"`.

## 4. A defect my own guard caught mid-build

My first fallback chain ended at `or ""`. **BigQuery's REQUIRED mode rejects
NULL but ACCEPTS an empty string**, so that would have cheerfully inserted
outcomes with no identity -- polluting the very table those four consumers read.
`test_required_columns_...` failed on it. Changed to SKIP the trade and log,
rather than fabricate an identity.

## 5. Verbatim verification output (CYCLE 1 -- superseded by section 9)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_48_outcome_write_schema.py -q
...............                                                          [100%]
15 passed in 6.60s

$ python -m pytest backend/tests/ -q -k "82_39 or 82_48 or 82_12 or outcome or nightly or production_fns or slack or job"
195 passed, 1 skipped, 2568 deselected, 1 warning in 26.45s

$ echo "$FILES" | xargs uvx ruff check --select F821,F401,F811
All checks passed!   (exit 0, over a git-derived asserted-non-empty scope)
```

**Criterion 2 is satisfied by a REAL BigQuery round trip**, not a mock: the
guard creates a throwaway table from the live schema, inserts the repaired row,
`SELECT`s it back, and drops the table in teardown. It cannot use production --
rows streamed into `outcome_tracking` cannot be deleted for up to 90 minutes and
would pollute `get_performance_stats` for its four consumers.
`test_a_mock_cannot_substitute_for_the_round_trip` pins why: a `MagicMock`
returning `[]` accepts the BROKEN shape too.

Derived sizes, regenerated last:

```
$ git diff --numstat -- backend/slack_bot/jobs/_production_fns.py backend/slack_bot/jobs/nightly_outcome_rebuild.py
164	7	backend/slack_bot/jobs/_production_fns.py
52	8	backend/slack_bot/jobs/nightly_outcome_rebuild.py

$ wc -l backend/tests/test_phase_82_48_outcome_write_schema.py
     415 backend/tests/test_phase_82_48_outcome_write_schema.py
$ python3 -c "ast walk for test_ functions"
16
```

## 6. Mutation matrix

| # | Mutant | Result |
|---|---|---|
| M1 | revert to the pre-fix emitted shape | KILLED |
| M2 | drop `analysis_date` from the emitted row | KILLED |
| M3 | delete the rejection alert | KILLED |
| M4 | downgrade the WRITE alert to P2 | KILLED |
| M5 | make the WRITE alert fire unconditionally | KILLED |
| M6 | disable the dedup read-back | KILLED |
| M7 | restore the None-unsafe pnl expression | KILLED |
| M8 | fabricate an empty analysis anchor instead of skipping | KILLED |
| M9 | drop the extra columns from the fetch SQL | KILLED |

**9 of 9 killed.** Licenses "these 9 died", not "no survivor exists".

### The three survivors, and which one was real

**M9 was a genuine gap.** Dropping `analysis_id, risk_judge_decision,
holding_days` from the fetch left every guard green, because they all built
trade dicts BY HAND -- nothing tied the fetch's projection to what the write
needs. The write would then skip every row: a silent return to zero written
outcomes, the exact defect this step fixes. Closed by
`test_the_fetch_supplies_every_field_the_write_REQUIRES`, driven from the SQL
BUILDER rather than a source scan.

**M4 and M5 were MIS-TARGETED, not guard gaps.** `severity="P1"` now appears
twice in the file (82.39's fetch alert and this step's write alert) and
`if errors:` four times; both mutants hit the earlier, unrelated site. A mutant
that changes something the guard never reads proves nothing -- retargeted to
unique anchors and re-run rather than counted as coverage.

## 7. Files changed

| File | Change |
|---|---|
| `backend/slack_bot/jobs/_production_fns.py` | `+164 / -7` -- real 9-column row, P1 rejection alert, dedup, fetch extended |
| `backend/slack_bot/jobs/nightly_outcome_rebuild.py` | `+52 / -8` -- NULL-safe, skip-not-fabricate, carries the destination fields |
| `backend/tests/test_phase_82_48_outcome_write_schema.py` | NEW, 415 lines, 16 tests |

## 8. Queued / non-scope

`bigquery_client.py:416-417` swallows `insert_rows_json`'s return on
`save_outcome` -- a SECOND instance of the same class, on the writer this step
delegates its row shape to. Queued, not fixed here. The NaN-graded-as-loss
adjacency is queued with it. No change to `paper_learn_loop_enabled`; no live
positions touched.

---

## 9. Cycle-2 corrections (Q/A CONDITIONAL -> fixed)

Four findings, all real, all cheap. Verbatim verdict in
`evaluator_critique_82.48.md`.

**F1 -- my "offline fallback" was structurally dead, and the docstring said
otherwise.** `_schema_snapshot.json` nests tables under a `tables` key; I looked
the dotted name up at TOP level, so the fallback returned `{}` unconditionally
and the immutable verification command was credential-dependent while the file
claimed "the default test path is offline and free". It failed LOUD (the
non-empty assert fires), so nothing passed vacuously -- but the CLAIM did not
reproduce. Fixed; now resolves 9 columns offline.

**F2 -- guards stopped one seam short, again.** The live round trip drove
`_compute_outcomes -> build_outcome_row -> insert`, but never
`make_outcome_write_fn()._write`, which is what production calls. All three
tests that DID drive `_write` patched `_bq_client` with a MagicMock that accepts
any shape. Added `test_the_PRODUCTION_closure_persists_into_bigquery`, which
drives the real closure into a throwaway table.

**And that new guard immediately caught a real production defect.**
`_drop_already_written` queried a HARDCODED table name while the insert used
`OUTCOME_TABLE` -- so the two could diverge, and the mock-based dedup test had
been passing for a reason that does not hold against a real client. The dedup
now redirects with the target asserted present, so the rewrite cannot be a
silent no-op. In production both names were the same table, so this was latent
-- but only a real round trip could show it.

**F3 -- an unbounded query with its own bound discarded.** `keys` was computed
and never used (`ruff F841`), while the dedup did a full-table scan every night
against CLAUDE.md's "always bound queries". Now parameterised with
`IN UNNEST(@tickers)` / `IN UNNEST(@analysis_dates)` -- bound AND
injection-safe. The remaining F841 in that file (`col`) is pre-existing, as the
Q/A independently confirmed from the diff hunks.

**F4 -- a loop that asserted nothing on one of its two iterations.** The Q/A
PROBED it rather than reasoning: `analysis_id` was never in the fixture, so
removing it changed nothing. Rewritten as explicit cases, including a positive
control that `analysis_id` is the PREFERRED anchor when present.

Post-fix: `16 passed`, mandated ruff gate exit 0, regression `196 passed`.
