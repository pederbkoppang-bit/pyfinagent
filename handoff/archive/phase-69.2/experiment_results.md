# Experiment Results — Step 69.2 (P0 gate correctness, OFFLINE)

- **Phase / step**: phase-69 → 69.2
- **Date**: 2026-07-11
- **Type**: OFFLINE backtest/analytics + promotion-gate corrections (zero live-money surface)

## What was changed (5 fixes; 199 insertions across 3 files)

`git diff --stat`:
```
 backend/backtest/analytics.py          |  30 ++++++++-
 backend/backtest/backtest_engine.py    | 119 +++++++++++++++++++++++++++------
 backend/services/paper_go_live_gate.py |  76 +++++++++++++++++++--
```

1. **DSR unit correction** — `analytics.py:292-335` (`compute_deflated_sharpe`) + caller `:654-661`.
   Added `periods_per_year: int = 1` (default = no-op, byte-identical for existing callers); when set,
   de-annualizes `observed_sr` (/sqrt(ppy)) and `variance_of_srs` (/ppy). `generate_report` now passes
   `periods_per_year=252`. Optimizer (`quant_optimizer` dsr>=0.95 reject) + `strategy_backtest_adapter`
   read DSR via `generate_report` → auto-fixed. `dsr_52wh_verdict.py` (ablation script) left byte-identical.
2. **Purge + embargo** — `backtest_engine._label_overlaps_test` (new) + `_build_training_data` (now takes
   `test_start/test_end`, purges samples whose `[sample_date, sample_date+1.5*holding_days]` overlaps the
   test window; exit_dates use the true `1.5*holding_days`). `walk_forward.py:61` embargo retained as the
   post-test gap.
3. **Boundary snap** — `backtest_engine._price_asof` (new) at the entry (`:488`) and liquidation (`:514`)
   lookups: exact date, else widen `[d-7, d]` and take the last available close.
4. **Fracdiff-at-predict / NaN-fill** — `backtest_engine`: persist `_train_feature_medians`; new
   `_build_predict_features` staticmethod applies the SAME imputation as train (train medians, was
   `fillna(0)`) and places non-stationary features on the train (fracdiff'd) scale via the train median.
5. **Go-live booleans** — `paper_go_live_gate._sustained_psr_ge` (min expanding-window PSR over the last
   30 days ≥ 0.95) + `_load_backtest_max_dd` (None → documented 20% cap fallback); the two under-spec
   booleans now use the sustainment + backtest-DD+5pp tolerance.

## Verification command output (verbatim)

```
$ python -m pytest backend/tests/test_gate_correctness_69.py -q -x --timeout=180
..................                                                       [100%]
18 passed in 1.36s
```

Independent DSR check (scipy): corrected (ppy=250) DSR = **0.9004880** (Bailey reference); N=46 → 0.9505;
bug path (ppy=1) = **0.9999999**; default (no ppy) == ppy=1 (byte-identical).

## Do-no-harm evidence

- Immutable threshold CONSTANTS byte-untouched: `git diff` shows NO change to `DSR_THRESHOLD = 0.95`,
  `PSR_THRESHOLD = 0.95`, `MAX_DD_ABS_TOLERANCE = 20.0`, `dsr_threshold: float = 0.95` — only the
  comparison LOGIC changed. `test_immutable_thresholds_unchanged` asserts the values.
- `compute_deflated_sharpe` default `periods_per_year=1` is byte-identical to the pre-fix behavior
  (`test_dsr_default_is_byte_identical_to_ppy1`); only `generate_report` opts in.
- No existing test references `compute_deflated_sharpe` / `compute_gate` / `_build_training_data` /
  `generate_report` (grep empty). Full suite still collects: **1028 tests collected** (no import breakage).
- Zero live-money surface: all changes are in the offline backtest/analytics + the promotion GATE
  computation (tightening a gate to its documented spec can only make promotion STRICTER, never move money).

## Honest scope note — criterion 4 (fracdiff-at-predict)

The immutable criterion asks the fracdiff transform + NaN-fill be "applied identically at predict as
train." My fix makes the **NaN-fill policy identical** (train medians, not `fillna(0)`) and puts predict
non-stationary features on the **train (fracdiff'd) scale** via the train median — eliminating the
register's concrete skew (train=median+fracdiff'd vs predict=0-fill+raw-levels; "model fed raw
price/market-cap levels it never trained on"). It does **not** apply the windowed fracdiff convolution at
predict, because (a) a single cross-sectional predict row has no series window to convolve, and (b) the
series lives in `build_feature_vector`, which is **live-adjacent** (`data_server.py`) and thus out-of-bounds
for a zero-live-surface offline step. A full per-ticker time-series fracdiff (relocating fracdiff to the
feature builder) is flagged as a larger follow-on. Q/A should judge whether the scale+fill consistency
satisfies the criterion's intent given this architectural constraint, or whether it is CONDITIONAL.

## Deferred (explicit)

Incumbent re-validation of the live strategy under the corrected DSR/purge gates requires the operator to
lift the historical_macro freeze / authorize optimizer runs — out of 69.2 scope. This step ships code +
fixtures only; the corrected statistic is proven against the Bailey reference on fixtures.

## Process note

Ordering slip disclosed in `contract.md`: research → code → contract (contract formalized concurrently
with GENERATE; the plan was in `research_brief_69.2.md` §Application, written during the gate before any
code). Full five-file set complete before the status flip.

## Cycle-2 remediation (after Q/A FAIL on the ruff gate)

Cycle-1 Q/A (workflow structured-output) returned FAIL on exactly one deterministic gate: ruff F401 —
unused `import numpy as np` at `backend/tests/test_gate_correctness_69.py:13` ("absent the F401 this
evaluation is a PASS"; all 5 immutable criteria green, C4 accept-on-intent, thresholds byte-untouched).
Fixed per the cycle-2 flow:

1. Removed the unused `import numpy as np` (no `np.` references remained).
2. Re-verified (verbatim):
   - `uvx ruff check --select F821,F401,F811 backend/backtest/analytics.py backend/backtest/backtest_engine.py backend/services/paper_go_live_gate.py backend/tests/test_gate_correctness_69.py` → **All checks passed!** (exit 0).
   - `python -m pytest backend/tests/test_gate_correctness_69.py -q` → **18 passed in 1.48s**.
3. Filed the C4 required follow-on: `handoff/current/audit_phase69/followons_69.2.md` (FO-69.2-A — true
   per-ticker time-series FFD in the feature builder, future live-adjacent step) + documented the
   median-neutralization LIMITATION in the `_build_predict_features` docstring (a 69.4 hand-off seed).

Changed evidence (F401 removed, ruff clean, follow-on filed) → a fresh Q/A evaluates the updated state.
