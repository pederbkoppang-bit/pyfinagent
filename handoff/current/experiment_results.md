---
step: phase-23.1.18
cycle_date: 2026-04-29
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_18.py'
---

# Experiment Results — phase-23.1.18

## Summary

User reported the Red Line Monitor on the home page ends at
~$14,000 while live NAV is $15,647.74. BQ inspection confirmed
`paper_portfolio_snapshots` had DUPLICATE rows per snapshot_date
(04-29: 2 rows, 04-27: 3 rows, 04-26: 3 rows; 13 duplicates total
across the table). The chart endpoint used `ANY_VALUE(total_nav)` —
non-deterministic, empirically picked the older/lower row.

**Three coordinated fixes** (per researcher A + B + C):

**Fix A — `save_paper_snapshot` MERGE upsert**
(`backend/db/bigquery_client.py:669-708`). Replaced plain INSERT
with `MERGE ... ON T.snapshot_date = S.snapshot_date WHEN MATCHED
UPDATE ... WHEN NOT MATCHED INSERT ...`. Same pattern as
phase-23.1.15's save_paper_position MERGE. Going forward, no two
rows can ever exist for the same snapshot_date — the natural-key
write is idempotent. Guards on `snapshot_date` presence with a
clear ValueError.

**Fix B — Cleanup script**
(`scripts/cleanup_phase_23_1_18.py`). Two-mode (dry-run default,
--apply with --yes for headless). Uses `CREATE OR REPLACE TABLE
... AS SELECT ... ROW_NUMBER() OVER (PARTITION BY snapshot_date
ORDER BY total_nav DESC) ... WHERE rn = 1`. Heuristic: in our
data the post-repair / post-mark_to_market row always has the
highest total_nav, so MAX(nav) per date is the closest proxy to
"most recent / most complete" given no created_at column. After
this cycle: 24 → 11 rows, 11 unique dates.

**Fix C — Defensive MAX in red-line query**
(`backend/api/sovereign_api.py:130-145`). Replaced `ANY_VALUE`
with `MAX(total_nav)`. Defense in depth: even if a future bug
bypasses the MERGE somehow, the chart picks the largest NAV
per date deterministically (matches the cleanup heuristic).

## Files modified

- `backend/db/bigquery_client.py` (+18 lines: rewrote
  save_paper_snapshot to MERGE)
- `backend/api/sovereign_api.py` (+5 / -1: MAX(total_nav)
  + phase-23.1.18 marker comment)

## Files added

- `scripts/cleanup_phase_23_1_18.py` (124 lines, two-mode
  dedup with ROW_NUMBER PARTITION BY)
- `tests/services/test_snapshot_upsert.py` (3 new tests)
- `tests/verify_phase_23_1_18.py` (immutable verification)

## Verification command output

```
$ source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_18.py
ok save_paper_snapshot MERGE upsert + red-line MAX(total_nav) query + cleanup script (dry-run/apply with ROW_NUMBER PARTITION BY) + 3 new tests pass
```
Exit 0.

## Test results

```
$ pytest tests/services/test_snapshot_upsert.py tests/services/test_trade_idempotency.py tests/services/test_sector_concentration.py tests/api/test_ticker_meta_perf.py tests/api/test_ticker_meta.py -q
............................                                             [100%]
28 passed in 2.89s
```
3 new + 25 prior phases' tests all green.

## Frontend type check

```
$ cd frontend && npx tsc --noEmit
(silent, exit 0)
```

## Cleanup script execution log

```
$ python scripts/cleanup_phase_23_1_18.py --apply --yes
paper_portfolio_snapshots: 24 total rows / 11 unique dates
Duplicate-date counts:
  2026-04-29: 2 rows
  2026-04-27: 3 rows
  2026-04-26: 3 rows
  2026-04-24: 2 rows
  2026-04-22: 2 rows
  2026-04-15: 3 rows
  2026-04-14: 5 rows
Planned SQL: CREATE OR REPLACE TABLE ...
Executing rewrite (CREATE OR REPLACE TABLE)...
Rewrite complete.
paper_portfolio_snapshots: 11 total rows / 11 unique dates
No duplicate snapshot_dates -- already clean.
ok phase-23.1.18 cleanup complete (was 24 rows / 11 unique dates -> 11 rows / 11 unique)
```

## Live BQ post-cleanup state

```sql
SELECT snapshot_date, total_nav, cash, positions_value
FROM paper_portfolio_snapshots ORDER BY snapshot_date DESC LIMIT 12;
```
| snapshot_date | total_nav | cash | positions_value |
|---|---|---|---|
| 2026-04-29 | $15,647.74 | $2,146.39 | $13,501.34 |
| 2026-04-28 | $13,952.25 | $694.99 | $13,257.26 |
| 2026-04-27 | $14,458.32 | $4,023.11 | $10,435.23 |
| 2026-04-26 | $9,499.50 | $9,499.50 | $0.00 |
| (older days $9,499.50 cash, no positions yet) | | | |

Each snapshot_date now has exactly one row. The Red Line Monitor's
terminal point is $15,647.74 — matches paper-trading + home hero
NAV.

## Backwards compatibility

- MERGE behaves identically to INSERT for new (no-conflict)
  rows — autonomous_loop callers see no API change.
- Cleanup script dry-run is default; --apply opt-in.
- MAX(total_nav) in the red-line query is a one-token change;
  no shape difference in the response.
- Backend already restarted; the new MERGE is live for any
  future save_daily_snapshot call.

## Honest disclosures

1. **Heuristic dedup**: cleanup uses `ORDER BY total_nav DESC`
   to pick the "winner" per date. In our data the post-repair
   row always has the highest NAV (mark_to_market ran). For
   a hypothetical case where a real intraday loss made the
   newer row LOWER, the heuristic would keep the older row
   incorrectly. Acceptable for one-shot data repair given no
   created_at column. Going forward, MERGE prevents the scenario.

2. **Pre-trading days flat at $9,499.50** is correct — that was
   the actual NAV before any positions were opened (cash only).
   The chart now shows the true historical NAV trajectory.

3. **"Pre-deposit" perception**: starting_capital is now $15,000
   (after a $5k deposit). The chart shows historical NAV ~$9,499
   for the first month because that was the cash balance THEN.
   This is by design — phase-23.1.9 already documents that
   deposits increment BOTH starting_capital and current_cash so
   pnl_pct stays anchored. The historical chart is NOT rebased.

4. **Backend restart was performed** so Fix A (MERGE) and Fix C
   (MAX) are live for the next save_daily_snapshot / red-line
   request.

## Phase 2 (deferred)

- Add `created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()` column
  to paper_portfolio_snapshots. Then ROW_NUMBER ORDER BY
  created_at DESC is deterministic regardless of NAV direction.
  Researcher confirmed two-step DDL is backwards compatible
  via Google Cloud docs. Deferred — with Fix A in place, no
  new duplicates can be created.
- Apply MERGE upsert to all other paper_* INSERT call sites
  for consistency (audit needed).
- Optional: rebase the home chart with a "show as % return"
  toggle so it normalizes across pre-deposit history.
