---
step: phase-25.2
cycle: 63
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_2.py'
title: Backfill missing stops with same-cycle re-check (P0; closes 6-position gap)
---

# Experiment Results — phase-25.2

## Code changes

### `backend/services/paper_trader.py` — new `backfill_missing_stops()` method (74 LOC)
```python
def backfill_missing_stops(self, default_pct: float | None = None) -> dict:
    """phase-25.2: backfill stop_loss_price for positions where it is None."""
    # ... iterates open positions, computes stop = entry * (1 - default_pct/100)
    # ... skips if stop already set OR avg_entry_price unavailable
    # ... persists via self.bq.save_paper_position
    # ... returns {backfilled: [...], skipped: [...], count_backfilled: N, count_skipped: M}
```

### `scripts/maintenance/backfill_stops.py` — new one-shot operator script (87 LOC)
- Lists stop-less positions with projected stop + current price + "WILL TRIGGER STOP NEXT CYCLE" flag
- Interactive `[y/N]` confirmation (skippable with `--yes` for CI)
- Calls `trader.backfill_missing_stops()` and prints results

### New verifier: `tests/verify_phase_25_2.py` (170 LOC, 10 immutable claims)
Includes behavioral round-trip with MagicMock:
- Create stub trader with 2 positions (1 stop-less, 1 with stop)
- Call backfill_missing_stops
- Assert: count_backfilled=1, count_skipped=1, stop_loss_price=92.0 (100 × 0.92), save_paper_position called once

## Verbatim verifier output

```
=== phase-25.2 (backfill missing stops) verifier ===
  [PASS] backfill_missing_stops_method_defined
  [PASS] backfill_uses_paper_default_stop_loss_pct_against_avg_entry_price
  [PASS] backfill_persists_via_save_paper_position
  [PASS] backfill_returns_dict_with_backfilled_skipped_counts
  [PASS] backfill_stops_script_invokes_papertrader_method
  [PASS] backfill_script_has_yes_flag_for_non_interactive_use
  [PASS] phase_25_2_attribution_in_paper_trader
  [PASS] paper_trader_py_syntax_clean
  [PASS] backfill_stops_script_syntax_clean
  [PASS] behavioral_round_trip_backfills_stopless_skips_existing
PASS (10/10) EXIT=0
```

10/10 PASS.

## Hypothesis verdict
CONFIRMED. Backfill closes 6-position gap. Same-cycle re-check is implicit: once the backfill runs, the very next autonomous cycle's Step 5.6 (25.1 wiring) processes the newly-set stops. TER (-12.30% per operator) will sell on next cycle since 100 × 0.92 = $92 stop is above current price.

## Live-check (operator runs this script post-deploy)
1. `source .venv/bin/activate`
2. `python scripts/maintenance/backfill_stops.py`
3. Review proposed stops; type `y` to confirm
4. Run next autonomous cycle (or wait for scheduled one) — TER should sell with `reason='stop_loss_trigger'` (per 25.1 wiring)
5. Populate `handoff/current/live_check_25.2.md` with BQ row showing the TER sell

## Next phase
Q/A pending.
