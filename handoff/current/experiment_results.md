---
step: phase-23.1.15
cycle_date: 2026-04-29
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_15.py'
---

# Experiment Results — phase-23.1.15

## Summary

User flagged "Western Digital Corporation appears twice" in trades view
while position count says 14. BQ forensics confirmed two integrity
violations causing ~$1,451 phantom cash debit:

**Bug A — WDC duplicate trade.** TWO distinct trade_ids 5 minutes
apart on 2026-04-26 (`56072f0c…` 21:12:28 + `e5447bd9…` 21:17:41),
each debiting $949.95+fee. paper_positions had only ONE WDC row
(cost_basis $949.95, entry_date 21:17:41). Net cash leak: $950.90.

**Bug B — XOM orphan trade.** 1 trade ($500, reason
`test_paper_trade`) from 2026-03-28 with no matching position.
Net cash leak: $500.50.

**Root cause** (per cycle_history.jsonl): cycle `0e8c4a20`
(21:11:35→21:12:31) errored AFTER booking the WDC trade and
debiting cash but before its position write was visible. Cycle
`a54a21fc` 4 min later read `get_positions()`, saw no WDC
(BigQuery snapshot-isolation across separate jobs — confirmed in
the external research brief), dropped through `execute_buy`'s
else-branch, created a fresh position row and debited cash again.
No idempotency defense.

## Fix surfaces

**Fix A — Idempotency guard in `paper_trader.execute_buy`** (after
the existing-position lookup). When `existing is None`, query
paper_trades for the same ticker + action='BUY' in the last 30
minutes within 1% qty tolerance. If a match is found, log a
warning and return None. Defends against crash-and-retry
double-buys regardless of whether the underlying race is BQ
snapshot lag, transient query error, or future schedule overlap.

**Fix B — MERGE upsert in
`bigquery_client.save_paper_position`**. Plain `INSERT INTO` was
replaced with `MERGE … ON T.ticker = S.ticker WHEN MATCHED THEN
UPDATE … WHEN NOT MATCHED THEN INSERT …`. ticker is now the
natural-key idempotency boundary at the BQ layer. Two writes for
the same ticker no longer produce two rows; the second silently
upserts. Backwards compatible: every existing caller's behavior
is unchanged for the new-row case.

**Fix E — Cleanup script** (`scripts/cleanup_phase_23_1_15.py`).
Two-mode: `--dry-run` (default) prints SQL + diff; `--apply` (with
optional `--yes` for headless) executes the DELETE+UPDATE chain.
DELETEs target the two specific trade_ids by hash; UPDATE refunds
$1,451.40 to current_cash. Idempotent: re-runs after success
no-op on the DELETEs and skip the UPDATE.

## Files modified

- `backend/services/paper_trader.py` (+ idempotency-guard block in
  `execute_buy`, +1 import: `timedelta`)
- `backend/db/bigquery_client.py` (rewrote `save_paper_position` to
  MERGE; new helper `get_paper_trades_for_ticker_since`)

## Files added

- `scripts/cleanup_phase_23_1_15.py` (213 lines, two-mode cleanup)
- `tests/services/test_trade_idempotency.py` (4 tests)
- `tests/verify_phase_23_1_15.py` (immutable verification)

## Verification command output

```
$ source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_15.py
ok execute_buy idempotency-guard + paper_positions MERGE upsert + get_paper_trades_for_ticker_since helper + cleanup script (dry-run/apply) + 4 new tests pass
```
Exit 0.

## Test results

```
$ pytest tests/services/test_trade_idempotency.py tests/services/test_sector_concentration.py -q
............                                                             [100%]
12 passed in 0.81s
```
4 new idempotency tests + 8 phase-23.1.13/14 sector tests all green.

## Cleanup script execution log

The cleanup ran in two stages due to a column-type mismatch
discovered live (paper_portfolio.updated_at is STRING, not
TIMESTAMP):

1. **First `--apply`** (20:17): Steps 1+2 DELETEs succeeded
   (WDC duplicate row gone, XOM test row gone). Step 3 UPDATE
   failed with `BadRequest: Value of type TIMESTAMP cannot be
   assigned to updated_at, which has type STRING`.
2. **Fix shipped**: cleanup script now uses
   `FORMAT_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S+00:00',
   CURRENT_TIMESTAMP())` to write the ISO string the column expects.
3. **Second `--apply`** (20:18): Both DELETEs returned 0 rows
   (idempotent — already gone). The script's `(1 if X_deleted else 0)`
   gate then short-circuited Step 3 — protective design but it
   meant the missed refund from stage 1 wasn't applied.
4. **One-shot recovery UPDATE** (20:18): ran the +$1,451.40
   credit directly via the Python BQ client. Result:
   `current_cash` $694.99 → $2,146.39.

## Post-state reconciliation (BQ verify)

```sql
WITH trades AS (
  SELECT ticker, SUM(total_value) AS sum_value, COUNT(*) AS n_trades
  FROM paper_trades GROUP BY ticker
),
positions AS (
  SELECT ticker, cost_basis FROM paper_positions
)
SELECT
  COUNT(*) AS join_rows,
  SUM(CASE WHEN n_trades > 1 THEN 1 ELSE 0 END) AS dup_trade_tickers,
  SUM(CASE WHEN positions.cost_basis IS NULL THEN 1 ELSE 0 END) AS orphan_trades,
  ROUND(SUM(COALESCE(sum_value, 0)) - SUM(COALESCE(cost_basis, 0)), 2) AS leak_dollars
FROM trades FULL OUTER JOIN positions USING (ticker)
```

Result: **14 join_rows, 0 dup_trade_tickers, 0 orphan_trades,
$0.00 leak**. Books reconcile.

## Backwards compatibility

- Idempotency guard fires only on the `existing is None` path —
  positions that already exist still flow through the additive
  update branch unchanged.
- MERGE is functionally identical to plain INSERT when no matching
  row exists; behavior change only on the duplicate-write path.
- 30-minute window covers the observed 5-minute cycle re-entry
  with headroom; daily-cycle apps don't typically buy the same
  ticker twice within 30 minutes intentionally.
- Cleanup script is dry-run by default; `--apply` requires
  explicit opt-in.

## Honest disclosures

1. **Mid-cleanup column-type bug surfaced live.** First `--apply`
   ran the two DELETEs successfully, but Step 3 UPDATE failed
   on a `STRING` column receiving a TIMESTAMP. The script was
   fixed and re-run, then a one-shot recovery UPDATE was issued
   to apply the missed refund. Final state is correct (BQ
   reconciliation shows zero leak), but the path was bumpy. The
   script as shipped is now correct end-to-end if re-run from
   scratch.

2. **Idempotency guard window is 30 minutes**. A ticker bought,
   sold, and re-bought within 30 min would currently be blocked
   by the guard. This is unlikely under daily cycles but could
   bite a future intraday strategy. If that becomes a concern,
   tighten the qty-tolerance check or add a `bypass_idempotency`
   kwarg.

3. **MERGE upsert overwrites the entry_date when called for an
   existing ticker.** This was already the behavior of the
   delete+insert pattern at line 144-161 (which we kept). Future
   refactor could collapse that block to a single MERGE — left
   for Phase 2.

4. **No real-money risk** — paper trading only. The phantom debit
   was virtual cash.

## Phase 2 (deferred)

- Collapse the manual delete+insert pattern in execute_buy /
  mark_to_market / execute_sell partial-exit to single MERGE
  calls (cleaner, fewer write hops).
- Add a deterministic `client_order_id` field on paper_trades
  set by `decide_trades` (e.g.,
  `f"{cycle_id}-{ticker}-{action}"`) — Alpaca-style idempotency.
- Run a one-time data-integrity audit job nightly that reports
  trade-vs-position reconciliation drift to Slack.
