# Contract -- Phase 4.4.1.2 DSR >= 0.95 on Out-of-Sample Data

**Cycle:** 16
**Date:** 2026-04-16
**Item:** 4.4.1.2

## Target
Verify that the Deflated Sharpe Ratio (DSR) of the best backtest result clears the 0.95 gate, and that the result is computed on out-of-sample (OOS) data via walk-forward methodology.

## Success Criteria
1. optimizer_best.json exists with DSR field
2. Best result JSON has DSR >= 0.95
3. dsr_significant flag is True
4. Cross-check: optimizer_best.json and result JSON agree on DSR and Sharpe
5. num_trials > 1 (DSR deflation requires multiple trials)
6. Walk-forward structure present (per_window data, n_windows > 0)
7. No train/test overlap in any window
8. Embargo days > 0 (information leakage prevention)
9. Train/test window configuration present

## Approach
Write a stdlib-only drill that reads the persisted backtest artifacts and verifies all criteria programmatically. No backend deps needed -- the evidence is in the JSON files.
