# Sprint Contract — phase-25.2 — Backfill missing stops

**Cycle:** phase-25 cycle 7
**Date:** 2026-05-12
**Step ID:** 25.2
**Priority:** P0
**Depends on:** 25.1 (DONE — Step 5.6 stop enforcement wired)

## Research-gate
Reuses phase-24.1 cycle 2 researcher gate (6 sources). Fix per audit F-5.

## Hypothesis
Adding `PaperTrader.backfill_missing_stops()` + one-shot maintenance script will close the 6-position stop-less gap (TER, ON, INTC, DELL, GLW, CIEN). After backfill, the next autonomous cycle's Step 5.6 (25.1) sells any position below its newly-set stop — TER especially (already -12.30%).

## Success criteria (verbatim)
1. all_open_positions_have_stop_loss_price_not_null_in_bq
2. ter_position_closed_or_sell_trade_with_reason_stop_loss_backfill_exists
3. backfill_uses_paper_default_stop_loss_pct_against_avg_entry_price

## Plan
1. Add `PaperTrader.backfill_missing_stops(default_pct: float | None = None) -> dict`
2. One-shot script `scripts/maintenance/backfill_stops.py` with interactive confirm + `--yes` for CI
3. Verifier `tests/verify_phase_25_2.py` (10 claims incl. behavioral round-trip with MagicMock)
4. Q/A
5. harness_log Cycle 63
6. Flip 25.2

## References
- `docs/audits/phase-24-2026-05-12/24.1-execution-trading-findings.md` F-5
- `backend/services/paper_trader.py:414` (existing check_stop_losses kept)
- `backend/services/paper_trader.py:424-498` (new backfill_missing_stops)
- `scripts/maintenance/backfill_stops.py` (new — operator-run with confirmation)
