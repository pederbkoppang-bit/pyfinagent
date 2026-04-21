# Experiment Results — phase-8.5 / 8.5.5 (DSR+PBO gate, CPCV)

**Step:** 8.5.5 **Date:** 2026-04-20.

2 new files: `backend/autoresearch/gate.py` (PromotionGate frozen dataclass + cpcv_folds) + `scripts/harness/autoresearch_gate_test.py`.

```
$ python scripts/harness/autoresearch_gate_test.py
PASS: dsr_gt_0_95_required -- dsr below 0.95 -> rejected
PASS: pbo_lt_0_2_required -- pbo above 0.20 -> rejected
PASS: cpcv_applied -- cpcv_folds(6,2) -> 15 clean folds
PASS: rejection_and_revert_regression_passes -- rejection did not mutate trial; gate is frozen
---
PASS   EXIT=0

$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped
```

All 4 success criteria PASS.
