# Sprint Contract — phase-10.5 (Sortino with configurable MAR)

**Step id:** 10.5 **Date:** 2026-04-20 **Tier:** moderate **Harness-required:** true

## Why

phase-10.1 `sprint_calendar.yaml` monthly_anchor references `backend/metrics/sortino.py` with "MAR default 0.045". phase-10.6 (Monthly Champion/Challenger) needs a canonical Sortino to evaluate strategies against. This step ships the canonical Sortino module with configurable MAR.

## Research-gate summary

Fresh researcher (moderate): `handoff/current/phase-10.5-research-brief.md` — 7 sources in full, 17 URLs, three-variant queries, recency scan, gate_passed=true.

Critical findings from the brief:
1. **Existing `compute_sortino` at `backend/services/perf_metrics.py:297-311` uses `std(ddof=1)` on negative-only values** — diverges from canonical Sortino & Price (1994) LPM_2. Keep-as-is (it's called by `paper_metrics_v2.py:111`); NEW module ships the canonical LPM_2 form.
2. **`pyfinagent_data.historical_macro` lacks DGS3MO/DTB3** — only DGS10 + FEDFUNDS. `_default_mar_fetcher` must try BQ first, fall back to `backend/backtest/analytics.get_risk_free_rate()` (reads local DTB3 CSV cache), fall back to hardcoded 0.045.
3. **Sentinel for zero-downside (all returns ≥ MAR):** return `float('nan')` per Empyrical + Gale Finance 2026 (not `+inf`, not `0.0`).
4. **Annualization:** `sqrt(periods_per_year)`; default 252 (daily), accept 12 (monthly) for phase-10.6.
5. **Rule reconciliation:** `backend-services.md` says "never compute Sharpe, drawdown, alpha outside `perf_metrics.py`" — Sortino is not in that list, and masterplan explicitly dictates `backend/metrics/sortino.py`. We'll ship in `backend/metrics/` and leave `perf_metrics.compute_sortino` untouched for back-compat.

## Immutable success criteria (masterplan-verbatim)

Test command: `python -m pytest backend/metrics/tests/test_sortino.py -q`

Criteria:
1. **`formula_matches_sortino_price_1994`** — LPM_2 formula: `DD = sqrt(mean(min(0, R_t - MAR)^2))` over ALL T periods (not just below-MAR count); Sortino = `(mean(excess) / DD) * sqrt(periods_per_year)`. Verified against a worked example where the manual calculation is known.
2. **`downside_deviation_only_below_mar`** — clipping: above-MAR returns contribute zero to DD. Test: `returns = [+0.05, +0.05, -0.02]` with `mar=0`; only `-0.02` contributes.
3. **`default_mar_pulls_from_pyfinagent_data_macro`** — `mar=None` triggers `mar_fetch_fn()` call; default fetcher attempts `pyfinagent_data.historical_macro` first (mock the BQ client in the test).
4. **`configurable_mar_per_candidate`** — `sortino(returns, mar=0.03)` ≠ `sortino(returns, mar=0.08)` when returns have some but not all above-MAR values; scalar and array MAR both supported.

## Plan

1. Create `backend/metrics/__init__.py` (empty).
2. Create `backend/metrics/sortino.py`:
   - Public `sortino(returns, *, mar=None, periods_per_year=252, mar_fetch_fn=None) -> float`
   - LPM_2 formula: `dd = sqrt(np.mean(np.clip(mar - returns, 0, None) ** 2))`
   - `mar` accepts scalar float, 1-D array (per-period), or `None` (trigger fetch)
   - `mar_fetch_fn` injectable; default `_default_mar_fetcher`
   - `_default_mar_fetcher()` tries BQ `pyfinagent_data.historical_macro` for a DGS3MO-like series → falls back to `get_risk_free_rate()` → falls back to `0.045`; fail-open; annualized rate returned; caller divides by `periods_per_year` to get per-period MAR
   - Zero-downside sentinel: `float('nan')`
   - Insufficient-samples (len < 2 after excess computation): `float('nan')`
   - ASCII-only logs
3. Create `backend/metrics/tests/__init__.py` (empty).
4. Create `backend/metrics/tests/test_sortino.py` with ≥6 pytest cases:
   - `test_formula_matches_sortino_price_1994` — known worked example
   - `test_downside_deviation_only_below_mar` — positive excesses contribute 0
   - `test_default_mar_pulls_from_pyfinagent_data_macro` — monkeypatches BQ client, asserts it was called
   - `test_configurable_mar_per_candidate_scalar` — different scalar MAR → different Sortino
   - `test_configurable_mar_per_candidate_array` — per-period MAR array accepted
   - `test_all_returns_above_mar_returns_nan` — sentinel
   - `test_annualization_daily_vs_monthly` — `sqrt(252)` vs `sqrt(12)`
   - `test_mar_fetch_fn_injectable` — custom fetcher called when `mar=None`
5. Run verification:
   - `python -c "import ast; ast.parse(open('backend/metrics/sortino.py').read())"`
   - `python -m pytest backend/metrics/tests/test_sortino.py -q` (the immutable criterion)
   - `pytest tests/autoresearch/ tests/slack_bot/ backend/metrics/ -q` (no regression)
6. Spawn fresh Q/A. If CONDITIONAL/FAIL: fix → updated handoff → fresh Q/A.
7. Log, flip masterplan, close task #65.

## References

- `handoff/current/phase-10.5-research-brief.md` (7 in full, 17 URLs, gate_passed=true)
- `backend/services/perf_metrics.py:297-311` (existing divergent implementation, keep untouched)
- `backend/backtest/analytics.py:89-122` (`get_risk_free_rate` — DTB3 CSV cache)
- `backend/slack_bot/jobs/weekly_fred_refresh.py` (FRED refresh; DGS3MO carry-forward for phase-10.6)
- `.claude/rules/backend-services.md` ("single metric source" rule; Sortino not in scope)
- Sortino & Price (1994), *Journal of Investing*; LPM_2 canonical formula

## Carry-forwards (out of scope)

- Add `DGS3MO` to `weekly_fred_refresh._DEFAULT_SERIES` — would populate `historical_macro` with 3M T-Bill; deferred as a small housekeeping ticket
- Deprecate or unify `perf_metrics.compute_sortino` with the new canonical — would require updating `paper_metrics_v2.py`; deferred to avoid scope creep
- Phase-10.6 will call `sortino(returns, periods_per_year=12)` for monthly Champion/Challenger gate
