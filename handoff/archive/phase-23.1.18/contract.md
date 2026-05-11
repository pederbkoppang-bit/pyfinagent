---
step: phase-23.1.18
title: paper_portfolio_snapshots dedup + MERGE upsert + deterministic red-line query
cycle_date: 2026-04-29
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_18.py'
research_brief: handoff/current/phase-23.1.18-external-research.md (also see phase-23.1.18-internal-codebase-audit.md)
---

# Contract — phase-23.1.18

## Hypothesis

User reported the Red Line Monitor on the home page ends at ~$14,000
while live NAV is $15,647.74. BQ inspection confirmed
`paper_portfolio_snapshots` has DUPLICATE rows per snapshot_date
(04-29: 2 rows, 04-27: 3 rows, 04-26: 3 rows). The
`/api/sovereign/red-line` endpoint uses `ANY_VALUE(total_nav)`
(`sovereign_api.py:133`) — non-deterministic, empirically picks
the stale row.

Three compounding causes:
1. `save_paper_snapshot` (`bigquery_client.py:675`) does plain
   `INSERT INTO` — every call appends a new row.
2. `autonomous_loop` calls `save_daily_snapshot` at two sites
   (lines 309 + 424) which can write twice in the same UTC day.
3. Schema has no `created_at` column → no discriminator for
   "latest" row.

If we (A) convert `save_paper_snapshot` to `MERGE ... ON
snapshot_date` (idempotent natural-key write — same pattern as
phase-23.1.15's `save_paper_position`), (B) one-shot dedupe the
existing duplicates (keep MAX(total_nav) per date — heuristic
since post-repair always exceeds stale), and (C) defensively
update the red-line query to use `MAX(total_nav)` instead of
`ANY_VALUE`, then the chart's terminal value matches the live
NAV and future double-writes are structurally impossible.

## Research-gate summary

- External brief: `handoff/current/phase-23.1.18-external-research.md`
  — 6 sources read in full (oneuptime MERGE upsert + dedup
  streaming + non-deterministic MERGE 2026; Google Cloud
  dedup tutorials; BQ default-values docs; Hevo upsert
  primer). 16 URLs collected. Recency scan 2024-2026.
  `gate_passed: true`.
- Internal audit: `handoff/current/phase-23.1.18-internal-codebase-audit.md`
  — 7 files inspected. Confirms two upstream call sites
  (autonomous_loop:309,424) plus the phase-23.1.17 repair script
  added today's duplicate row.

Researcher recommends **A + B + C**. Defers Fix D (add
`created_at` column) — once Fix A is in place, no new
duplicates are created so the column is moot for this use case.

## Plan steps

1. **Fix A — `save_paper_snapshot` MERGE upsert**
   (`backend/db/bigquery_client.py`). Replace plain INSERT with
   `MERGE ... ON T.snapshot_date = S.snapshot_date WHEN MATCHED
   UPDATE ... WHEN NOT MATCHED INSERT ...`. snapshot_date is the
   natural key. Idempotent re-write within a day overwrites
   prior rows for that date. Same pattern as phase-23.1.15's
   `save_paper_position` MERGE.

2. **Fix B — Cleanup script**
   (`scripts/cleanup_phase_23_1_18.py`). Two-mode (dry-run
   default, --apply with --yes for headless). Strategy:
   `CREATE OR REPLACE TABLE ... AS SELECT ... ROW_NUMBER() OVER
   (PARTITION BY snapshot_date ORDER BY total_nav DESC) AS rn
   ... WHERE rn = 1`. Rationale: in our data, the post-repair
   row always has the higher total_nav because mark_to_market
   ran. This dedupes 04-29 (2→1), 04-27 (3→1), 04-26 (3→1),
   etc. Idempotent — re-runs are no-ops once unique.

3. **Fix C — Defensive MAX in red-line query**
   (`backend/api/sovereign_api.py:133`). Replace `ANY_VALUE` with
   `MAX(total_nav)`. Defense-in-depth: even if MERGE is bypassed
   somehow, the chart picks the most-favorable row consistently.

4. **Tests** (`tests/services/test_snapshot_upsert.py`): three
   new tests:
   - `save_paper_snapshot` issues a MERGE statement (mock
     bq.client.query, assert MERGE INTO + ON T.snapshot_date =
     S.snapshot_date in the SQL).
   - `save_paper_snapshot` rejects rows missing `snapshot_date`.
   - red-line endpoint uses MAX(total_nav) — grep
     sovereign_api.py source for `MAX(total_nav)` in the
     _fetch_snapshots SQL.

5. **Immutable verification**
   (`tests/verify_phase_23_1_18.py`): asserts the MERGE block
   exists in save_paper_snapshot, the MAX(total_nav) is in the
   red-line query, the cleanup script exists with both modes,
   the new tests pass.

## Immutable verification command

```bash
source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_18.py
```

Must exit 0 with one ok-line.

## Acceptance criteria

- `pytest tests/services/test_snapshot_upsert.py -q` passes.
- `python tests/verify_phase_23_1_18.py` exits 0.
- After running cleanup `--apply`: BQ verify query
  `SELECT snapshot_date, COUNT(*) FROM paper_portfolio_snapshots
  GROUP BY snapshot_date HAVING COUNT(*) > 1` returns ZERO
  rows.
- Red Line Monitor on home page ends at $15,647.74 (matches
  paper-trading live NAV).

## Backwards compatibility

- MERGE behaves identically to INSERT for new (no-conflict)
  rows — autonomous_loop callers see no API change.
- Cleanup script dry-run is default; --apply opt-in.
- MAX(total_nav) in the red-line query is a one-token change;
  no shape difference in the response.
- No frontend changes required — chart automatically reflects
  the corrected backend data.

## References

- `handoff/current/phase-23.1.18-external-research.md`
- `handoff/current/phase-23.1.18-internal-codebase-audit.md`
- `backend/db/bigquery_client.py:669-685` (save_paper_snapshot)
- `backend/api/sovereign_api.py:122-147` (_fetch_snapshots)
- `backend/services/paper_trader.py:421-450` (save_daily_snapshot wrapper)
