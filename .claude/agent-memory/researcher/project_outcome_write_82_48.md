---
name: outcome-write-82-48
description: 82.48 nightly_outcome_rebuild WRITE side -- the return_pct mapping is already decided in-repo (phase-47.7), outcome_tracking has 5 live consumers not zero, the NULL-pnl crash is NOT reachable in prod, and fixing the write introduces a ~30x duplicate-row defect
metadata:
  type: project
---

Step 82.48 (P1), researched 2026-08-06. Five things that would have been guessed
wrong without measuring.

**1. The Q1 "semantic decision" was already made -- twice -- and one was a
retraction of the opposite mapping.** `realized_pnl_pct` is written at
`backend/services/paper_trader.py:583-585` as `((price - entry_price)/entry_price)
* 100.0` -- percent of the PER-SHARE `avg_entry_price`, already x100, FX-cancelling,
SELL-legs-only by construction (`execute_buy` never sets the key). And
`autonomous_loop.py:3063-3071` already maps it to `outcome_tracking.return_pct`,
with an inline comment recording that reading the non-existent `return_pct` off a
trade row "silently recorded 0.0 return for EVERY sell-close". A correct 9-column
writer ALREADY EXISTS at `bigquery_client.py:400-417` (`save_outcome`).

**Why:** the step said "the step forbids guessing" -- the answer was not a
judgment call, it was a lookup. Re-deriving would have re-opened a closed bug.
**How to apply:** on any "what should column X receive" question in this repo,
grep for an existing writer of the destination table before reasoning about
semantics.

**2. `outcome_tracking` is NOT an orphan -- 82.39's brief said outcome_tracker.py
has ZERO references, which is literally true and materially misleading.**
`outcome_tracker.py:201` reads the table through `self.bq.get_performance_stats()`.
A string grep for the table name cannot see a consumer that goes through a client
method. Structural derivation (the table is addressable only via
`BigQueryClient.outcomes_table` at `bigquery_client.py:47` or an FQ literal) gives
2 writers + 2 readers + 4 callers of the reader (`outcome_tracker.py:201`,
`meta_coordinator.py:258`, `skill_optimizer.py:331`, `perf_metrics.py:635`).
`skill_optimizer.py:220-227` documents that it currently "degrades to neutral
scores" because the table is empty.
**Why:** a grep-derived consumer set under-counts every indirection.
**How to apply:** derive consumer sets from the ACCESSOR (the attribute/method that
builds the table ref), not from the table name. Same class as
[[feedback_measure_dont_assert_claims]].

**3. Fixing the write introduces a NEW duplicate-row defect the step doesn't
mention.** The fetch re-reads a ROLLING 30-day window every night
(`_production_fns.py:232`) and `save_outcome` APPENDS -- it is not an upsert
(stated at `autonomous_loop.py:3057-3060`). `IdempotencyKey.daily` prevents a
double-run on ONE day, not re-writes across 30 days. So the same SELL lands ~30
times unless a dedup key is added.

**4. The NULL-pnl TypeError is real but NOT reachable in production, and it does
not crash anything.** `.get("pnl", 0.0)` returning `None` on a present-but-None key
is correct, and `_compute_outcomes` really is called outside the try
(`nightly_outcome_rebuild.py:27` vs the try at `:28-32`). BUT `heartbeat` is a
`@contextmanager` whose `except Exception` at `job_runtime.py:105-108` does NOT
re-raise, so contextlib SUPPRESSES it -- `run()` returns `{"rebuilt": 0}`, a third
flavour of silent zero. And 82.39's `realized_pnl_pct IS NOT NULL` predicate keeps
NULLs out of the production path entirely; the only protection is that unasserted
SQL clause (82.39's suite asserts `r["pnl"] is not None` on returned ROWS, which is
a data assertion, not a SQL invariant). Adjacent: a FLOAT NaN passes IS NOT NULL,
`nan > 0` is False, so NaN silently grades "loss".

**5. Streaming insertAll genuinely has no dry-run, but the OFFLINE snapshot works.**
`backend/db/_schema_snapshot.json` DOES contain `financial_reports.outcome_tracking`
(9 cols, 3 REQUIRED), so a key-vs-schema check runs in CI with no creds.
`schema_oracle.dry_run:550` is a QUERY-job config and structurally cannot cover
insertAll. Google confirms per-row vs whole-batch: a schema mismatch rejects the
ENTIRE batch with one `insertErrors` entry per row (innocent rows get
`reason: stopped`). Do NOT "fix" by switching to `insert_rows()` -- it silently
drops unknown fields client-side (python-bigquery #151, repo archived 2026-03-06).
A streamed test row also CANNOT be cleaned up promptly (DML over the streaming
buffer is blocked up to 90 min), so an end-to-end persistence fixture must use a
throwaway table built from `migrate_bq_schema.OUTCOME_TRACKING_SCHEMA`.

**Second instance, same defect class:** `bigquery_client.py:416-417` also swallows
the `insert_rows_json` return into a `logger.error`. Worth queueing separately
per [[feedback_queue_discovered_defects_in_masterplan]] semantics.
