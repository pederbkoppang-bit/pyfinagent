# Live Check — phase-48.4: Live rotation bake-off SMOKE (REAL evidence)

This step's whole point is LIVE validation, so this live_check is REAL (not n/a). Verbatim, 2026-05-29.

## Real run (4 real walk-forward backtests, 2022-01-01..2024-06-30, 6 windows each; $0 LLM; audit-only)
Command: `PYTHONPATH=$PWD python scripts/run_rotation_smoke.py` (exit 0). Per-seed metrics CAPTURED from the real adapter (the REAL generate_report + compute_pbo ran on REAL BacktestResults):
```
triple_barrier   : dsr=1.0   pbo=0.4887334887334887  sharpe=1.8232784959596307  n_variants=2  n_windows=6
quality_momentum : dsr=0.0   sharpe=0.0              (pbo OMITTED -- undersized matrix -> producer SKIPPED)
```
Selector verdict (verbatim):
```
{selected_id: "triple_barrier", switched: false, reason: "no_candidate_passed_gate",
 ranked: [], incumbent_id: "triple_barrier", num_trials: 2, delta_dsr: null}
```
Persisted audit row (`backend/backtest/experiments/rotation_log.jsonl`, last line):
```
{"selected_id":"triple_barrier","incumbent_id":"triple_barrier","switched":false,
 "reason":"no_candidate_passed_gate","delta_dsr":null,"ranked":[],"num_trials":2,
 "allocation_pct":0.0,"status":"bakeoff_verdict","num_param_variants":2,"window":"2022-01-01..2024-06-30"}
```

## What this PROVES (the machinery works live)
- The full chain ran end-to-end on REAL backtests: make_rotation_engine (full kwargs) -> 4 real walk-forward backtests -> nav_history -> generate_report DSR + per-strategy (T x K) compute_pbo -> producer -> selector -> persisted row.
- triple_barrier produced REAL finite metrics: dsr=1.0, **pbo=0.489 (a genuine non-degenerate value from a real T>=32 matrix)**, sharpe=1.82, n_windows=6.
- The gate worked CORRECTLY: triple_barrier passed DSR (1.0>=0.95) but its pbo 0.489 > 0.20 -> gate-vetoed; qm degenerate -> skipped (no false-good 0.0); so `no_candidate_passed_gate` -> retain incumbent. Zero deploy side-effects (allocation_pct=0).

## The live bug it CAUGHT (fixed mid-cycle)
First run = both seeds degenerate (sharpe=0). Diagnostic (`scripts/diag_rotation_backtest.py`): a direct triple_barrier backtest was healthy (Sharpe 1.75, 160 trades, 228 nav rows). Root cause: optimizer_best's `target_annual_vol=0` ("vol-targeting disabled") was mapped to the trader's `target_vol=0` which ZEROES position size (no trades). Fixed: 0/missing -> default 0.15. Post-fix triple_barrier traded (Sharpe 1.82). The $0 mock tests missed this (they tested the mapping arithmetic, not the trader's target_vol=0 no-trade semantics) -- live validation is exactly why.

## FLAGGED follow-ups (NOT this cycle)
- `quality_momentum` produces NO trades on 2022-2024 (degenerate) -> qm-strategy investigation needed before qm is a useful rotation seed.
- Seed redundancy: post-fix tb_baseline + tb_risk_managed both vol-target at 0.15, differing only by tp_pct -> reseed.
- K=2 PBO is coarse (0.489); the real bake-off should use K~8-16. The full 4-seed x K~8 bake-off, the deployment params->settings.paper_* bridge, and the weekly cron remain DEFERRED.
