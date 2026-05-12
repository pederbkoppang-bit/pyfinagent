# Sprint Contract — phase-25.6 — No-stop-on-entry hard block

**Cycle:** phase-25 cycle 8
**Date:** 2026-05-12
**Step ID:** 25.6
**Priority:** P0
**Depends on:** 25.1 (DONE)

## Research-gate
Reuses phase-24.1 cycle 2 researcher gate. Fix per F-4 (fallback only for NEW buys; defense-in-depth needed).

## Hypothesis
Adding a None-check at the top of `execute_buy()` that synthesizes a default 8% stop will guarantee every new position has stop_loss_price NOT NULL in BQ.

## Success criteria (verbatim)
1. execute_buy_with_none_stop_loss_synthesizes_default_8pct
2. warning_log_emitted_when_default_stop_applied
3. no_new_positions_with_stop_loss_price_null_post_25_6

## Plan
1. Add 10-line block at top of `execute_buy()` (`backend/services/paper_trader.py:82`): if `stop_loss_price is None and price > 0`, synthesize via `round(price * (1.0 - default_pct/100.0), 4)` + log.warning
2. Verifier `tests/verify_phase_25_6.py` (8 claims)
3. Q/A
4. Cycle 64 log
5. Flip 25.6

## References
- `docs/audits/phase-24-2026-05-12/24.1-execution-trading-findings.md` F-4
- `backend/services/paper_trader.py:82-101` (execute_buy entry block)
- `backend/config/settings.py:184` (paper_default_stop_loss_pct default 8.0)
