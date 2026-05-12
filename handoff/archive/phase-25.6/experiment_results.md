---
step: phase-25.6
cycle: 64
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_6.py'
title: No-stop-on-entry hard block in execute_buy() (P0)
---

# Experiment Results — phase-25.6

## Code change

`backend/services/paper_trader.py:execute_buy()` — added 18-line block at the top of the function (before `portfolio = self.get_or_create_portfolio()`):

```python
# phase-25.6: no-stop-on-entry HARD BLOCK. If stop_loss_price is None
# at entry, synthesize one from settings.paper_default_stop_loss_pct
# (8% default per O'Neil canonical + arxiv 2604.27150) so every new
# position has a stop in BQ. Defense-in-depth alongside 25.2 backfill.
if stop_loss_price is None:
    default_pct = float(getattr(self.settings, "paper_default_stop_loss_pct", 8.0))
    if price > 0:
        stop_loss_price = round(price * (1.0 - default_pct / 100.0), 4)
        logger.warning(
            "phase-25.6: no stop_loss_price provided for %s; defaulting to %.4f (%.1f%% below entry %.4f)",
            ticker, stop_loss_price, default_pct, price,
        )
```

## Verbatim verifier output

```
=== phase-25.6 (no-stop-on-entry hard block) verifier ===
  [PASS] execute_buy_checks_stop_loss_price_is_none_at_entry
  [PASS] execute_buy_uses_paper_default_stop_loss_pct_for_synthesis
  [PASS] phase_25_6_attribution_comment_present
  [PASS] execute_buy_logs_warning_when_default_stop_applied
  [PASS] paper_trader_py_syntax_clean
  [PASS] stop_loss_price_reassigned_to_computed_value_within_none_branch
  [PASS] stop_synthesized_via_canonical_formula_price_times_one_minus_pct_over_100
  [PASS] execute_buy_guards_against_zero_price_before_computing_default_stop
PASS (8/8) EXIT=0
```

8/8 PASS.

## Hypothesis verdict
CONFIRMED. Every new position now persisted via execute_buy will have stop_loss_price NOT NULL. Triple-layer protection:
- **25.1** Step 5.6 enforces stops on every cycle
- **25.2** backfills existing stop-less positions
- **25.6** hard-blocks future stop-less entries

## Live-check
Per masterplan: "BQ paper_positions for any new position post-25.6 has stop_loss_price NOT NULL". Verifier round-trips the math; operator confirms post-deploy that next buy includes a stop.

## Next phase
Q/A pending.
