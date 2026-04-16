# Experiment Results -- Phase 4.4.1.2 DSR >= 0.95 on OOS Data

**Cycle:** 16
**Date:** 2026-04-16

## Drill Output

```
DRILL PASS: 13/13 checks passed
DSR = 0.9526 >= 0.95 on 27-window walk-forward OOS data
Sharpe = 1.1705, num_trials = 11, embargo = 5d
```

## Key Findings

- Best result: `20260328T072722Z_52eb3ffe-exp10.json` (Sharpe 1.1705)
- DSR = 0.9526, above 0.95 threshold with 0.0026 margin
- Walk-forward: 27 windows, 12mo train / 3mo test, expanding window, 5-day embargo
- OOS verified: all windows have train_end < test_start, no overlap
- DSR deflation meaningful: 11 trials used in computation
- Cross-check: optimizer_best.json and result JSON agree exactly on DSR and Sharpe

## Files Changed

- NEW: `scripts/go_live_drills/dsr_oos_test.py` (stdlib-only, 13 checks)
- MODIFIED: `docs/GO_LIVE_CHECKLIST.md` (item 4.4.1.2 flipped + evidence)
