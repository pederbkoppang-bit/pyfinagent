---
step: phase-23.1.15
title: Trade idempotency + paper_positions MERGE upsert + WDC/XOM cash leak cleanup
cycle_date: 2026-04-29
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_15.py'
research_brief: handoff/current/phase-23.1.15-external-research.md (also see phase-23.1.15-internal-codebase-audit.md)
---

# Contract — phase-23.1.15

## Hypothesis

User flagged "Western Digital Corporation appears twice" in trades view
while position count says 14. BQ forensics confirmed:

- **WDC**: TWO distinct trade_ids 5 minutes apart on 2026-04-26
  (`56072f0c...` 21:12:28 + `e5447bd9...` 21:17:41), each debiting
  $949.95+fee from cash. paper_positions has only ONE WDC row
  (cost_basis $949.95, entry_date 21:17:41 — proving cycle 2 hit
  the else-branch). Net cash leak: **$950.90**.
- **XOM**: 1 trade ($500, reason=`test_paper_trade`) with NO matching
  paper_positions row. Net cash leak: **$500.00**.

Total phantom outflow: ~$1,450.90.

**Root cause** (per cycle_history.jsonl forensics in researcher
brief): cycle `0e8c4a20` (21:11:35→21:12:31) errored AFTER booking
the WDC trade and debiting cash but BEFORE the position write
landed visibly. Cycle `a54a21fc` 4 min later saw no WDC in
`get_positions()`, dropped through `execute_buy`'s else-branch,
created a fresh position row, debited cash again. No idempotency
defense.

If we (1) gate `execute_buy` on a 30-min recent-trade lookback
when no `existing` position is found, (2) make `save_paper_position`
a MERGE on ticker (so future writes are idempotent by natural key),
and (3) clean up the two known leaked rows + restore $1,450.90
to cash, then the bug class is closed and the books reconcile.

## Research-gate summary

- External brief: `handoff/current/phase-23.1.15-external-research.md`
  — 8 sources read in full (BigQuery transactions, reliability,
  DML, MERGE upsert tutorial, deduplication patterns, MERGE
  non-deterministic-match 2026, MERGE-vs-dedup tutorial, Hevo BQ
  transaction). 18 URLs collected. Recency scan 2024-2026
  performed. `gate_passed: true`.
- Internal audit: `handoff/current/phase-23.1.15-internal-codebase-audit.md`
  — 6 files inspected with file:line anchors and concrete patch
  sketches.

Key findings:
- BigQuery DML INSERT is strongly consistent **within a job** but
  has snapshot-isolation across **separate jobs** — a new query
  job's snapshot is set at job-start, so a recently-INSERTed row
  may not be visible to a subsequent SELECT in a fresh job. The
  WDC bug fits this exactly.
- `save_paper_position` does plain `INSERT INTO`, no MERGE. Two
  inserts for the same ticker produces two rows; `get_paper_position`
  uses LIMIT 1 with no ORDER BY — effectively non-deterministic.
- Researcher recommends Fix A (idempotency guard) + Fix B (MERGE
  upsert) + Fix E (cleanup script). Skips C (deterministic
  position_id; BQ doesn't enforce uniqueness) and D
  (verify-after-write; redundant once B lands).

## Plan steps

1. **Fix A — Idempotency guard in `execute_buy`** (paper_trader.py
   after line 95): when `existing is None`, query paper_trades for
   any BUY of the same ticker in the last 30 minutes within 1% qty
   tolerance. If found, log and return None. Requires new BQ
   helper `get_paper_trades_for_ticker_since(ticker, since_iso,
   action)`.

2. **Fix B — MERGE upsert in `save_paper_position`**
   (bigquery_client.py:549-567): replace the plain INSERT with
   `MERGE ... ON T.ticker = S.ticker WHEN MATCHED UPDATE ...
   WHEN NOT MATCHED INSERT ...`. Backwards-compat: every existing
   caller's behavior unchanged for the new-row case; the
   already-exists case now upserts cleanly instead of silently
   producing a duplicate row.

3. **Fix E — Cleanup script** (`scripts/cleanup_phase_23_1_15.py`):
   two-mode (`--dry-run` default, `--apply` to mutate).
   - DELETE WDC trade `e5447bd9-9cb0-437b-b2a2-c851703b77b1`
     (the 21:17:41 one — keep the 21:12:28 canonical row).
   - DELETE the XOM `test_paper_trade` row.
   - Compute net refund = $949.95 + $0.95 + $500 + (XOM fee from
     row, if any). UPDATE paper_portfolio SET current_cash =
     current_cash + refund.
   - Verify post-state: 14 trades remaining, current_cash
     incremented, no orphan trade-without-position.

4. **Tests** (`tests/services/test_trade_idempotency.py`): three
   new unit tests:
   - Idempotency guard skips a duplicate BUY within 30 min window.
   - Idempotency guard does NOT skip a BUY of a different qty
     (>1% delta).
   - `save_paper_position` MERGE writes the right SQL shape (mock
     bq.client.query, assert MERGE INTO + ON T.ticker = S.ticker
     are in the query string).

5. **Immutable verification** (`tests/verify_phase_23_1_15.py`):
   asserts the idempotency guard exists in execute_buy, the MERGE
   replaces INSERT in save_paper_position, the new BQ helper
   exists, the cleanup script exists with dry-run + apply modes,
   and the 3 new tests pass.

## Immutable verification command

```bash
source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_15.py
```

Must exit 0 with a single ok-line.

## Acceptance criteria

- `pytest tests/services/test_trade_idempotency.py
  tests/services/test_sector_concentration.py -q` passes.
- `python tests/verify_phase_23_1_15.py` exits 0.
- `cd frontend && npx tsc --noEmit` exit 0.
- `python scripts/cleanup_phase_23_1_15.py --dry-run` prints the
  EXACT SQL it would run + diff of what would change. Exit 0.
- After `--apply`: BQ verify queries show 14 paper_trades rows
  (was 16), current_cash incremented by ~$1,450.90, no orphan
  trades, no missing positions for any non-test trade.

## Backwards compatibility

- Idempotency guard only fires on `existing is None` path; existing
  positions still take the additive update path unchanged.
- 30-min window covers the observed 5-min cycle re-entry window
  with headroom; no false positives expected for daily cycles
  with sell+rebuy in the same day (rare in this app).
- MERGE is functionally identical to INSERT when no row exists;
  only behavior change is on the duplicate-write path.
- Cleanup script dry-run is default; --apply is opt-in.

## References

- `handoff/current/phase-23.1.15-external-research.md`
- `handoff/current/phase-23.1.15-internal-codebase-audit.md`
- `backend/services/paper_trader.py:69-187` (execute_buy)
- `backend/db/bigquery_client.py:549-567` (save_paper_position)
- `backend/services/autonomous_loop.py:71-85` (_running flag),
  `:211-213` (held_tickers filter)
