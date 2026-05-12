---
step: phase-25.A6
cycle: 75
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_A6.py'
title: Explicit live-vs-backtest Sharpe reconciliation (P1)
audit_basis: phase-24.6 F-3 (paper_go_live_gate.py:87-94 used NAV-divergence as proxy for Sharpe gap)
---

# Experiment Results -- phase-25.A6

## Code changes

### `backend/services/perf_metrics.py`
- New imports: `json`, `datetime`, `timezone`, `Path`, `Any`.
- New module-level constants: `_OPTIMIZER_BEST_PATH`, `SR_GAP_THRESHOLD = 0.30`.
- New private helpers: `_load_optimizer_best_sharpe()`, `_shadow_curve_sharpe(bq, min_points, risk_free_rate)`, `_reconciliation_divergence_pct(bq)`.
- New public `compute_sharpe_gap(bq, *, backtest_sharpe_source="optimizer_best", risk_free_rate=0.04, min_snapshots=6) -> dict`:
  - 3-tier fallback chain: `optimizer_best` -> `shadow_curve` -> `proxy_fallback` -> `no_data`.
  - Returns dict: `live_sharpe`, `backtest_sharpe`, `gap_abs`, `gap_rel`, `threshold`, `gap_within_threshold`, `source`, `note`, `proxy_fallback`, `computed_at`.
  - Industry-benchmark threshold 30% per Jacquier et al. arxiv 2501.03938 (Jan 2025: 30-50% IS-to-OOS decay range; 30% is the stricter lower bound). Cited in the docstring.

### `backend/services/paper_go_live_gate.py`
- Added `from backend.services.perf_metrics import compute_sharpe_gap` import.
- Replaced lines 87-94 (`sr_gap_proxy = latest_divergence_pct / 100.0`) with a call to `compute_sharpe_gap(bq)`.
- `booleans["sr_gap_le_30pct"]` now derives from `sharpe_gap.gap_within_threshold` (None -> False, gate stays red on no-data).
- `details` dict augmented with 6 new diagnostic fields: `live_sharpe`, `backtest_sharpe`, `sharpe_gap_rel`, `sharpe_gap_source`, `sharpe_gap_proxy_fallback`, `sharpe_gap_note`.
- The legacy `latest_reconciliation_divergence_pct` field preserved as a sibling signal.

### `tests/verify_phase_25_A6.py` (new file)
- 11 immutable claims with 6 behavioral round-trips:
  - Claims 1-4: structural (signature, threshold, no legacy proxy, details fields).
  - Claim 5: **Behavioral primary-source** -- temp `optimizer_best.json` with `sharpe=1.0` + climbing snapshots with noise (variance > 0 -> finite Sharpe) -> `source="optimizer_best"`, `backtest_sharpe=1.0`, non-None gap fields.
  - Claim 6: **Behavioral threshold-failure** -- steep climb (live~5.57, backtest=0.5) -> `gap_within_threshold=False`.
  - Claim 7: **Behavioral no-data** -- empty snapshots + empty shadow + None divergence -> all None.
  - Claim 8: **Behavioral fallback-2 (shadow curve)** -- missing `optimizer_best.json` + noisy shadow curve >=6 points -> `source="shadow_curve"`.
  - Claim 9: **Behavioral fallback-3 (proxy)** -- missing primary + empty shadow + divergence=15% -> `source="proxy_fallback"`, `proxy_fallback=True`, `gap_rel=0.15`.
  - Claim 10: **Behavioral compute_gate integration** -- with all mocks in place, the gate's `details` dict exposes `live_sharpe`, `backtest_sharpe`, `sharpe_gap_source`; `booleans["sr_gap_le_30pct"]` is set.
  - Claim 11: industry-benchmark attribution in docstring.

## Verbatim verifier output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_A6.py
PASS: new_function_compute_live_realized_sharpe_vs_backtest_exists
PASS: threshold_at_30pct_per_industry_benchmark
PASS: paper_go_live_gate_uses_explicit_sharpe_not_nav_proxy
PASS: compute_gate_details_includes_sharpe_diagnostics
PASS: behavioral_primary_source_optimizer_best_json
PASS: behavioral_threshold_failure_gap_above_30pct
PASS: behavioral_no_data_gate_stays_red
PASS: behavioral_fallback_shadow_curve_used_when_optimizer_best_absent
PASS: behavioral_fallback_proxy_when_both_primary_and_shadow_unavailable
PASS: compute_gate_uses_new_helper_and_exposes_details
PASS: industry_benchmark_attribution_in_docstring

11/11 claims PASS, 0 FAIL
```

## Backend gates

- `python -c "import ast; ast.parse(open('backend/services/perf_metrics.py').read())"` -- OK
- `python -c "import ast; ast.parse(open('backend/services/paper_go_live_gate.py').read())"` -- OK
- 6 behavioral round-trips exercise the actual function (primary / fail / no-data / shadow / proxy / gate integration).

## Hypothesis verdict

CONFIRMED. Three immutable success criteria mapped:
- Criterion 1 (`new_function_compute_live_realized_sharpe_vs_backtest_exists`) -- claim 1 signature + claims 5-9 behavioral covering all four fallback tiers.
- Criterion 2 (`paper_go_live_gate_uses_explicit_sharpe_not_nav_proxy`) -- claim 3 (legacy proxy line gone, new helper called) + claim 10 (behavioral integration).
- Criterion 3 (`threshold_at_30pct_per_industry_benchmark`) -- claim 2 (SR_GAP_THRESHOLD = 0.30 in both files) + claim 11 (industry-benchmark docstring attribution).

## Live-check

Per masterplan: "Reconciliation report shows explicit live_sharpe - backtest_sharpe gap on next cycle".

Live evidence pending in `handoff/current/live_check_25.A6.md`. After next paper-trading cycle, calling `GET /api/paper-trading/gate` (or whatever surfaces compute_gate) should show `details.live_sharpe`, `details.backtest_sharpe`, `details.sharpe_gap_rel`, `details.sharpe_gap_source` populated (vs the previous output which only had `latest_reconciliation_divergence_pct`).

## Non-regressions

- SR_GAP_THRESHOLD value unchanged (0.30).
- 5-boolean gate contract unchanged.
- Legacy `latest_reconciliation_divergence_pct` preserved in `details` as a sibling signal.
- Proxy-fallback path preserves existing behavior when explicit Sharpe is unavailable.
- No new BQ schema; no frontend changes.

## Next phase

Q/A pending.
