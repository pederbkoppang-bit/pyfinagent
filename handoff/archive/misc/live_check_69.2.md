# Live Check — Step 69.2 (P0 gate correctness, OFFLINE)

Offline backtest/analytics step (zero live-money surface). Evidence per the masterplan `live_check`
field: DSR-vs-reference run, purge/embargo assertion, boundary-snap + fracdiff test runs, go-live-boolean
tests, and the git diff proving DSR>=0.95 / PBO<=0.5 thresholds byte-untouched.

## Test suite — `backend/tests/test_gate_correctness_69.py` (18 passed)

```
test_dsr_reference_value_matches_bailey                 PASSED   # C1: corrected DSR 0.9004880 (Bailey ref)
test_dsr_bug_path_inflated_without_correction           PASSED   # C1: bug path 0.9999999
test_dsr_default_is_byte_identical_to_ppy1              PASSED   # C1 do-no-harm: default == ppy=1
test_dsr_correction_dramatically_lowers_inflated_value  PASSED   # C1: ~sqrt(252) inflation removed
test_purge_label_reaching_into_test_is_purged           PASSED   # C2: label overlaps test -> purge
test_purge_label_ending_before_test_is_kept             PASSED   # C2: no overlap -> keep
test_purge_uses_1_5_holding_days_horizon                PASSED   # C2: true 1.5*holding_days horizon
test_price_asof_snaps_weekend_to_prior_trading_day      PASSED   # C3: boundary snap
test_price_asof_none_when_no_data                       PASSED   # C3: no-data path
test_predict_uses_train_median_not_zero_for_missing     PASSED   # C4: train-median imputation (was fillna(0))
test_predict_nonstationary_placed_on_train_scale        PASSED   # C4: 546.72 raw -> 0.02 train-scale
test_predict_no_train_medians_falls_back_to_zero        PASSED   # C4: degenerate fallback
test_load_backtest_max_dd_returns_none_when_absent      PASSED   # C5: None -> 20% cap fallback
test_dd_tolerance_falls_back_to_20pct_when_no_backtest_dd PASSED # C5: DD tolerance
test_sustained_psr_insufficient_history                 PASSED   # C5: short history -> not sustained
test_sustained_psr_strong_uptrend_sustains              PASSED   # C5: steady uptrend -> sustained
test_sustained_psr_flat_noisy_not_sustained             PASSED   # C5: flat/noisy -> not sustained
test_immutable_thresholds_unchanged                     PASSED   # do-no-harm: 0.95/0.95/20.0/100

18 passed in 1.48s
```

Lint gate (qa.md §1a), after the cycle-2 F401 fix:
```
$ uvx ruff check --select F821,F401,F811 backend/backtest/analytics.py backend/backtest/backtest_engine.py backend/services/paper_go_live_gate.py backend/tests/test_gate_correctness_69.py
All checks passed!        # exit 0
```

Independent DSR (scipy): corrected (ppy=250) DSR = **0.9004880** (z=1.284; Bailey reference); N=46 → 0.9505;
Normal crosses 0.95 near N=88; bug path (annualized SR + daily T, ppy=1) = **0.9999999** (z=5.29).

## Do-no-harm — immutable thresholds byte-untouched

`git diff backend/services/paper_go_live_gate.py` shows NO change to the threshold constant DEFINITIONS
(only the comparison LOGIC changed). Current values on disk:
```
TRADES_THRESHOLD = 100
PSR_THRESHOLD = 0.95
DSR_THRESHOLD = 0.95
MAX_DD_ABS_TOLERANCE = 20.0
```
`quant_optimizer.py:123` `dsr_threshold: float = 0.95` byte-untouched; DSR>=0.95 / PBO<=0.5 promotion gates
unchanged (the fix corrects the STATISTIC, not the threshold). `compute_deflated_sharpe` default
`periods_per_year=1` is byte-identical to the pre-fix behavior; only `generate_report` opts in (ppy=252).
1028 tests still collect (no import breakage); the sole external `_build_training_data` consumer
(`run_ablation.py:253`) calls positionally so the new kwargs default to None = pre-fix behavior.

## Q/A verdict (fresh, cycle-2, workflow structured-output)

`{"ok": true, "verdict": "PASS", "violated_criteria": []}` — ruff gate re-run bare → exit 0; import removal
git-grep-confirmed; C4 accept-on-intent condition satisfied (followons_69.2.md filed); all 5 criteria green;
FAIL→PASS flip grounded in the real code change (not sycophancy). Full ruling in `evaluator_critique.md`
(Cycle 1 FAIL + Cycle 2 remediation + Cycle 2 PASS).

## Deferred / follow-on
- Incumbent re-validation under the corrected DSR/purge gates: DEFERRED behind the historical_macro
  un-freeze token (code + fixtures only this step).
- C4 true fix (per-ticker time-series FFD in build_feature_vector): `audit_phase69/followons_69.2.md`
  FO-69.2-A (future live-adjacent step; 69.4 hand-off seed).
