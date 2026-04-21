# Experiment Results — phase-10.5 (Sortino with configurable MAR)

**Step:** 10.5 **Date:** 2026-04-20

## What was done

1. Fresh researcher (moderate): 7 in full, 17 URLs, recency scan, gate_passed=true. Brief at `handoff/current/phase-10.5-research-brief.md`. Corrected candidate design on two points:
   - Existing `compute_sortino` uses `std(ddof=1)` on negatives, NOT canonical LPM_2 — must ship new module separately
   - `pyfinagent_data.historical_macro` lacks DGS3MO/DTB3 today — default fetcher must try BQ first but fall back to `backend/backtest/analytics.get_risk_free_rate` and then 0.045
2. Contract authored at `handoff/current/phase-10.5-contract.md`.
3. Created `backend/metrics/__init__.py` (module docstring explaining why separate from `perf_metrics.py`).
4. Created `backend/metrics/sortino.py` (126 lines):
   - Public `sortino(returns, *, mar=None, periods_per_year=252, mar_fetch_fn=None) -> float`
   - **Canonical LPM_2 formula:** `dd = sqrt(mean(clip(mar - returns, 0, None) ** 2))` over ALL T periods
   - `mar` accepts scalar, 1-D array, or None; None triggers `mar_fetch_fn or _default_mar_fetcher`; fetcher returns annualized rate, caller divides by `periods_per_year`
   - `_default_mar_fetcher`: 3-tier fallback (BQ `historical_macro` for DGS3MO/DTB3 → `analytics.get_risk_free_rate` → hardcoded 0.045)
   - **Zero-downside sentinel:** `float('nan')` (per Empyrical + Gale Finance 2026)
   - **Insufficient samples (< 2):** `float('nan')`
   - Fail-open on any BQ/fetcher exception
   - ASCII-only logger messages
5. Created `backend/metrics/tests/test_sortino.py` (11 pytest cases):
   - `test_formula_matches_sortino_price_1994` — hand-computed worked example, Sortino ~= 13.1823
   - `test_downside_deviation_only_below_mar` — all-positive returns → NaN; mixed with one negative → finite
   - `test_default_mar_pulls_from_pyfinagent_data_macro` — monkeypatches `google.cloud.bigquery.Client`, asserts SQL references `historical_macro` + `DGS3MO`/`DTB3`
   - `test_configurable_mar_per_candidate_scalar` — different scalar MARs → different Sortinos
   - `test_configurable_mar_per_candidate_array` — per-period MAR array accepted
   - `test_mar_array_shape_mismatch_raises` — ValueError on mismatched length
   - `test_all_returns_above_mar_returns_nan` — zero-downside sentinel
   - `test_annualization_daily_vs_monthly` — `sqrt(252/12) = sqrt(21)` ratio
   - `test_mar_fetch_fn_injectable` — custom fetcher called when `mar=None`
   - `test_mar_fetch_fn_fail_open_to_default` — raising fetcher → fallback to 0.045
   - `test_fewer_than_two_samples_returns_nan` — edge guard

## Verification (verbatim)

```
$ python -c "import ast; [ast.parse(open(f).read()) for f in ['backend/metrics/__init__.py','backend/metrics/sortino.py','backend/metrics/tests/__init__.py','backend/metrics/tests/test_sortino.py']]; print('AST OK')"
AST OK

$ python -m pytest backend/metrics/tests/test_sortino.py -q
...........                                                              [100%]
11 passed in 0.98s
(exit 0)

$ pytest tests/autoresearch/ tests/slack_bot/ backend/metrics/ -q
........................................................................ [ 94%]
....                                                                     [100%]
76 passed in 1.48s
```

## Success criteria (masterplan, immutable)

| # | Criterion | Status |
|---|---|---|
| 1 | `formula_matches_sortino_price_1994` | PASS — hand-computed LPM_2 example matches to 1e-3 |
| 2 | `downside_deviation_only_below_mar` | PASS — clip pattern; all-above-MAR returns NaN |
| 3 | `default_mar_pulls_from_pyfinagent_data_macro` | PASS — SQL contains `historical_macro` + `DGS3MO OR DTB3`; stub BQ client invoked once |
| 4 | `configurable_mar_per_candidate` | PASS — scalar + array MAR both supported; distinct values → distinct Sortinos |

## Backend-services rule compliance

`.claude/rules/backend-services.md` says "Never compute Sharpe, drawdown, or alpha outside `perf_metrics.py`." Sortino is not listed; masterplan mandates `backend/metrics/sortino.py`. Decision: new module is the canonical Sortino source; `perf_metrics.compute_sortino` (divergent formula, `std(ddof=1)` on negatives) remains untouched for back-compat with `paper_metrics_v2.py:111`. Documented in `backend/metrics/__init__.py` docstring.

## Carry-forwards (out of scope)

- Add `DGS3MO` to `weekly_fred_refresh._DEFAULT_SERIES` so BQ `historical_macro` actually has data (fetcher currently falls through to local DTB3 CSV)
- Deprecate/unify `perf_metrics.compute_sortino` — callers would need to migrate
- Phase-10.6 will call `sortino(returns, periods_per_year=12)` for the monthly Champion/Challenger gate
