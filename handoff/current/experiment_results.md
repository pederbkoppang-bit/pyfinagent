# Experiment Results — Cycle 15: Phase 4.4.1.4

## Drill Output
- Script: `scripts/go_live_drills/walk_forward_concentration_test.py`
- Exit code: 0 (PASS)
- 12/12 checks passed, 0 failed

## Key Findings
- Best result: Sharpe 1.1705, DSR 0.9526, 27 walk-forward windows
- Source: `20260328T072722Z_52eb3ffe-exp10.json`
- Total return: 98.56% ($100,472.93) over 2019-04-11 to 2025-08-04
- **Max single-window contribution: 14.0%** (W24: 2025-05-05 to 2025-08-04)
- **Threshold: 30% — PASS with 16pp margin**

## Return Distribution
- 13 positive windows, 4 negative, 10 flat (no trades)
- Top-3 windows contribute 38.0% of total return
- Returns distributed across 2019-2025 with no single-period dominance
- Flat windows are expected: ML filter rejecting all candidates in those periods
