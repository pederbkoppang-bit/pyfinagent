# Live Check — phase-47.4: Sharpe/maxDD metric integrity

Captured 2026-05-29 on the running local system, AFTER backend reload (loads the chronological-sort fix).

## 1. Go-live gate (/api/paper-trading/gate) — maxDD + Sharpe corrected
```
realized_max_dd_pct:      5.3112      (was 60.08%  -- phantom "growth read backwards as a crash")
live_sharpe:              5.42        (was -5.72   -- sign flip from reversed NAV series)
max_dd_within_tolerance:  True        (was False   -- wrongly-red go-live boolean now correct, 5.31 <= 20)
promote_eligible:         False       (correct -- still needs 100 round-trips + PSR + DSR + sr_gap; now 1/5 not 0/5)
```

## 2. Cockpit (/api/paper-trading/portfolio) — user-facing Sharpe corrected
```
sharpe_ratio: 5.42        (was the screenshotted -5.72)
```
This is the exact number from the operator's cockpit screenshot, now sign-correct.

## 3. Deterministic guards (immutable command) — exit 0
```
$ python -m pytest tests/services/test_phase_47_4_metric_order_invariance.py -q
..                                                                       [100%]
2 passed in 1.29s
$ python -c "import ast; ast.parse(perf_metrics.py); ast.parse(paper_go_live_gate.py)"
ast OK
EXIT_CODE=0
```

## 4. Mutation-resistance proof (test catches the bug)
```
post-fix Sharpe chron==desc: True  | value: 22.91   (order-invariant, positive)
post-fix maxDD  chron==desc: True  | value: 0.495   (order-invariant, small)
OLD (unsorted) Sharpe chron vs desc: 22.91 vs -23.35  -> DIFFER  (the +/- sign-flip, same shape as -5.72)
```
The guard FAILS on the pre-fix path (opposite signs / inflated DD) and PASSES once each helper sorts
chronologically. Single shared root cause fixed in two one-line sorts:
- `perf_metrics.py::compute_sharpe_from_snapshots` (fixes cockpit + /performance + all 4 callers)
- `paper_go_live_gate.py::_snapshot_max_dd_pct`

NOTE: `/performance` endpoint returned `sharpe: None` for the naive top-level parse; the authoritative
corrected values are on `/gate` (live_sharpe 5.42) and `/portfolio` (sharpe_ratio 5.42), both fed by
the fixed `compute_sharpe_from_snapshots`. Small-sample caveat (n_obs=27): a point Sharpe is not yet
statistically trustworthy (Lopez de Prado MinTRL) -- this step corrects the MATH (order), not the
sample-size trustworthiness; a sample-size gate on the cockpit Sharpe is a separate follow-up.
