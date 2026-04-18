# Contract -- Cycle 81 / phase-4.8 step 4.8.4

Step: 4.8.4 Drift monitor (PSI + 20-day rolling IC)

## Hypothesis

Two canonical post-deployment model-drift signals:
- **PSI** (Population Stability Index, Siddiqi 2006 credit-scoring
  lit): detects feature-distribution shift between baseline window
  and current window. Standard thresholds: <0.10 no shift, 0.10-0.25
  minor, >0.25 significant. We freeze at >0.25.
- **20-day rolling IC** (Information Coefficient, Spearman rank
  correlation between model predictions and forward returns). IC
  <= 0 sustained = the model has lost predictive edge. We freeze
  when rolling-20-day IC stays <= 0 for the last 5 days.

Ship the library + runner + audit.

## Scope

1. **NEW** `backend/services/drift_monitor.py`
   - `compute_psi(baseline, current, bins=10)` -> float
   - `compute_ic(predictions, forward_returns)` -> float (Spearman)
   - `rolling_ic(pred_series, ret_series, window=20)` ->
     list[float]
   - `run()` -> dict `{models: [{name, psi, ic_20d,
     ic_20d_trend, frozen, freeze_reasons}]}`
   - Seeds 3 models with deterministic baseline/current feature
     samples + a synthetic prediction/return series when live data
     is absent. Records `data_source: "seeded"|"live"`.
   - Constants: `PSI_FREEZE_THRESHOLD=0.25`,
     `IC_FREEZE_SUSTAINED_DAYS=5`, `IC_FREEZE_MAX=0.0`.

2. **NEW** `scripts/audit/drift_monitor_audit.py`
   Three teeth tests:
   (a) baseline run: no drift, no freeze.
   (b) PSI trip: feature-shift fixture -> psi > 0.25 -> frozen=true,
       "psi_exceeded" in freeze_reasons.
   (c) IC trip: negative-IC fixture for sustained window -> frozen=
       true, "ic_sustained_negative" in freeze_reasons.

## Immutable success criteria

1. psi_weekly_logged -- run()['models'][i]['psi'] is float.
2. ic_20d_rolling_logged -- run()['models'][i]['ic_20d'] is float.
3. auto_freeze_fires_at_thresholds -- audit (b) and (c) both flip
   frozen=true with the correct reason string.

## Verification (immutable)

    python -c "from backend.services.drift_monitor import run; r=run(); assert 'models' in r and all('psi' in m and 'ic_20d' in m for m in r['models'])"

Plus: `python scripts/audit/drift_monitor_audit.py --check` -> PASS.

## Anti-rubber-stamp

qa must verify:
- PSI formula is the canonical Siddiqi form:
  PSI = sum_i (a_i - e_i) * ln(a_i / e_i) with zero-bin handling.
- Spearman rank correlation used for IC (not Pearson) -- standard
  convention in quant (Barra, Qian/Hua/Sorensen 2007).
- Freeze thresholds are COMPARED (not constant true/false) and
  freeze_reasons list is populated dynamically.

## References

- Siddiqi 2006 "Credit Risk Scorecards" on PSI
- Qian/Hua/Sorensen 2007 "Quantitative Equity Portfolio Management"
  on Information Coefficient conventions
- AFML ch.10 (backtest statistics, concept drift)
