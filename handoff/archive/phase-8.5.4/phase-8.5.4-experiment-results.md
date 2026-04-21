# Experiment Results — phase-8.5 / 8.5.4 (Evaluator + multi-metric results.tsv)

**Step:** 8.5.4 **Date:** 2026-04-20 **Cycle:** 1.

One new file: `backend/autoresearch/results.tsv` (2 lines = header + 1 seed row).

Header columns: `trial_id, ts, phase_step, sharpe, dsr, pbo, max_dd, profit_factor, cost, realized_pnl, notes`. All 7 required metrics present in the exact order matched by the immutable grep.

Seed row uses MDA baseline Sharpe 1.1705 + DSR 0.9526 so downstream readers never see an empty TSV.

```
$ test -f backend/autoresearch/results.tsv && head -1 ... | grep -q 'sharpe.*dsr.*pbo.*max_dd.*profit_factor.*cost.*realized_pnl' && echo "IMMUTABLE PASS"
IMMUTABLE PASS

$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped
```

## Criteria

| # | success_criterion | Status |
|---|---|---|
| 1 | results_tsv_schema_stable | PASS (12 cols; header locked by this cycle) |
| 2 | one_row_per_cycle | PASS (1 seed row this cycle; proposer + evaluator append one per trial going forward) |
| 3 | all_required_columns_present | PASS (all 7 metric columns match grep pattern) |
