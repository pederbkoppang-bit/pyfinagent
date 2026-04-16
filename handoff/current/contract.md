# Contract — Cycle 15: Phase 4.4.1.4 Walk-Forward Return Concentration

## Target
`docs/GO_LIVE_CHECKLIST.md` item 4.4.1.4: No single walk-forward window drives > 30% of total return.

## Current State
- Best result file: `backend/backtest/experiments/results/20260328T072722Z_52eb3ffe-exp10.json`
- Sharpe 1.1705, DSR 0.9526, 27 walk-forward windows, 1067-point equity curve
- Per-window `total_return_pct` stored as 0.0 (unfilled by engine)
- NAV history available with date+nav per trading day, starting 2019-04-11

## Plan
1. Write drill `scripts/go_live_drills/walk_forward_concentration_test.py` (stdlib + json only)
2. Load best-result JSON, extract per-window test boundaries and equity curve
3. Compute per-window return by finding NAV at window start/end dates
4. Assert max single-window contribution < 30% of total return
5. Flip checkbox, commit, push

## Success Criteria
- SC1: Best result file exists and loads
- SC2: 27 walk-forward windows with test_start/test_end dates
- SC3: Equity curve has data spanning the full test range
- SC4: Per-window returns computed from NAV data
- SC5: No single window contributes > 30% of total return
- SC6: Drill exits 0 on PASS
