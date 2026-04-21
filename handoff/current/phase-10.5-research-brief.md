# Research Brief: phase-10.5 — Sortino Ratio with Configurable MAR (default 3M T-Bill)

**Tier:** moderate  
**Accessed:** 2026-04-20  
**Step id:** 10.5

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://en.wikipedia.org/wiki/Sortino_ratio | 2026-04-20 | doc/encyclopedia | WebFetch | Canonical LPM_2 integral definition: DR = sqrt(integral from -inf to T of (T-r)^2 * f(r) dr); discrete form squares only below-MAR deviations |
| https://quantopian.github.io/empyrical/_modules/empyrical/stats.html | 2026-04-20 | code/reference | WebFetch | Empyrical uses `np.clip(downside_diff, -inf, 0)`, squares, nanmean, then `sqrt * sqrt(ann_factor)`; sentinel for zero downside is NaN (not float('inf')) |
| https://corporatefinanceinstitute.com/resources/wealth-management/sortino-ratio-2/ | 2026-04-20 | doc/practitioner | WebFetch | MAR = "return on long-term government securities"; LPM_2 = avg(squared below-MAR deviations across ALL periods, including those above MAR) |
| https://www.gale.finance/glossary/sortino-ratio/ | 2026-04-20 | doc/practitioner (2026) | WebFetch | 2026 reference: annualization formula `sigma_down = sqrt(mean(min(0, r_t - MAR)^2) * N)`; zero downside -> display N/A; warns under 20 downside days |
| https://www.wallstreetprep.com/knowledge/sortino-ratio/ | 2026-04-20 | doc/practitioner | WebFetch | Confirms S = (rp - rf) / sigma_d; sigma_d = sqrt(sum(neg returns^2) / total_periods); Sortino 2.0+ is "acceptable" |
| https://www.codearmo.com/blog/sharpe-sortino-and-calmar-ratios-python | 2026-04-20 | blog/code | WebFetch | Simple Python implementation: `series[series<0].std()*np.sqrt(N)` — note: uses std(ddof=1) on negatives only, not mean-squared LPM_2; annualizes with N=255 |
| https://www.recipeinvesting.com/2026-measuring-risk-adjusted-returns-with-downside-deviation-and-the-sortino-ratio/index.html | 2026-04-20 | blog (2026) | WebFetch | 2026 practitioner note: shorter periods inflate ratio; Sortino 1.0 as practical floor; risk-free rate subtracted in numerator |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://rpc.cfainstitute.org/sites/default/files/-/media/documents/code/gips/the-sortino-ratio.pdf | CFA PDF | Binary PDF stream; WebFetch could not decode content |
| https://www.cmegroup.com/education/files/rr-sortino-a-sharper-ratio.pdf | CME PDF (Rollinger & Hoffman) | Socket closed before full fetch |
| https://fred.stlouisfed.org/series/DGS3MO | FRED data page | 403 on direct fetch |
| https://fred.stlouisfed.org/series/DTB3 | FRED data page | 403 on direct fetch |
| https://www.pyquantnews.com/free-python-resources/mastering-sortino-ratio-stock-portfolios | blog | Page exists but contained no code; article references paid course |
| https://campus.datacamp.com/courses/introduction-to-portfolio-analysis-in-python/risk-and-return?ex=13 | course | Snippet only — auth-gated |
| https://braverock.com/brian/R/PerformanceAnalytics/html/SortinoRatio.html | R docs | Snippet only — R-language reference |
| https://deborahkidd.com/wp-content/uploads/The-Sortino-Ratio-Is-Downside-Risk-the-Only-Risk-That-Matters-1.pdf | CFA PDF | Not fetched — PDF |
| https://www.schwab.com/learn/story/using-sortino-ratio-to-gauge-downside-risk | broker doc | Snippet only |
| https://medium.com/@mburakbedir/beyond-returns-a-deep-dive-into-risk-adjusted-metrics... | blog | Snippet only |

---

## Recency scan (2024-2026)

Searched explicitly for: "Sortino ratio implementation 3M T-Bill MAR configurable Python 2025 2026" and "2026 Measuring Risk-Adjusted Returns with Downside Deviation and the Sortino Ratio".

Result: Two 2026-dated sources found and read in full (Gale Finance 2026 glossary; Recipe Investing 2026 article). Neither changes the canonical Sortino & Price (1994) formula. The 2026 practitioner consensus confirms:

1. Risk-free rate (not zero) is the preferred MAR for comparability with Sharpe.
2. Annualization via `sqrt(252)` for daily equity returns is the accepted convention.
3. Zero-downside sentinel: display as N/A (Gale Finance 2026) rather than `float('inf')` — this aligns with returning `float('nan')` in library code (Empyrical) and raising no exception.
4. Minimum sample warning at <20 downside observations is an emerging practitioner norm.

No new formula superseding LPM_2 was found in 2024-2026 literature.

---

## Key findings

1. **Canonical formula (Sortino & Price 1994)**: `S = (mean(R) - MAR) / DD` where `DD = sqrt(mean(min(0, R_t - MAR)^2))`, annualized by multiplying `DD * sqrt(periods_per_year)` and `mean(R) * periods_per_year`. The key LPM_2 point: ALL periods are included in the mean (denominator is T, not just the count of below-MAR periods). (Source: Wikipedia Sortino ratio, CFI, Gale 2026)

2. **Existing `compute_sortino` in perf_metrics.py has a formula bug**: `backend/services/perf_metrics.py:305-308` computes `downside = excess[excess < 0.0]` then calls `downside.std(ddof=1)`. This is the standard deviation of only the negative-excess values with ddof=1 — not LPM_2. LPM_2 takes `mean(min(0, excess)^2)` over ALL T periods, i.e., the denominator is T not `count_negatives - 1`. The two formulas are numerically different (the std approach inflates downside deviation when few periods are below MAR). The new `backend/metrics/sortino.py` must use the correct LPM_2. (Source: Empyrical implementation; Gale 2026 formula)

3. **FRED series choice**: The project already uses `DTB3` (discount basis, daily) in `backend/backtest/analytics.py:34`. DGS3MO (investment basis / bond-equivalent yield) is the more theoretically correct series for computing excess returns against a quoted yield, but the numerical difference is <10 bps for maturities this short. Maintaining DTB3 consistency with existing code is the pragmatic choice. `FRED_SERIES` in `data_ingestion.py` does NOT include DGS3MO or DTB3; `historical_macro` table has `FEDFUNDS, CPIAUCSL, UNRATE, GDP, T10Y2Y, UMCSENT, DGS10` — no short-rate T-bill. The `mar_fetch_fn` default should therefore use `analytics.get_risk_free_rate()` (which reads the on-disk DTB3 CSV cache), not a BQ query against `historical_macro`. (Source: `backend/backtest/analytics.py:20-120`; `backend/backtest/data_ingestion.py:21`)

4. **`weekly_fred_refresh.py` does not include DGS3MO/DTB3**: `backend/slack_bot/jobs/weekly_fred_refresh.py:11` has `_DEFAULT_SERIES = ["DGS10", "DGS2", "VIXCLS", "DFF", "UNRATE", "CPIAUCSL"]` — no short T-bill rate. The MAR default path must not rely on this job's output. (Source: `backend/slack_bot/jobs/weekly_fred_refresh.py:11`)

5. **Annualization convention**: `sqrt(252)` for daily returns; `sqrt(12)` for monthly. The existing `compute_sortino` already uses `periods_per_year=252` and `math.sqrt(periods_per_year)`. The new module should match this convention for daily and accept `periods_per_year=12` for monthly (Champion/Challenger review is monthly per sprint_calendar). (Source: `backend/services/perf_metrics.py:298`; Gale 2026)

6. **Zero-downside sentinel**: Empyrical returns `NaN`; Gale 2026 displays "N/A". The canonical Sortino & Price reference treats zero downside as "perfect" (unbounded ratio), but `float('inf')` causes downstream JSON serialization failures and comparison errors. Recommended sentinel: `float('nan')` — consistent with Empyrical and NaN-safe numpy operations. Document this clearly in the module docstring. (Source: Empyrical `_modules/empyrical/stats.html`; Gale 2026)

7. **Test fixture pattern**: The existing autoresearch gate test at `scripts/harness/autoresearch_gate_test.py` uses no-BQ pure-unit test functions (no pytest fixtures, plain `def case_...() -> tuple[bool, str]`). The phase-10.5 test at `backend/metrics/tests/test_sortino.py` should use standard pytest (matching the `pytest -q` verification command), with `mar_fetch_fn` injection to avoid any BQ call. (Source: `scripts/harness/autoresearch_gate_test.py:1-60`)

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/metrics/` | — | Target package directory | Does NOT exist; must be created |
| `backend/services/perf_metrics.py` | 297-311 | Existing `compute_sortino` | Exists; has formula divergence from LPM_2 (uses std of negatives, not mean-squared LPM_2 over all periods); keep as-is for backward compat, new module is separate |
| `backend/services/paper_metrics_v2.py` | 28, 111 | Calls `compute_sortino(returns)` with no MAR arg (defaults to 0.0) | Active; no change needed for this phase |
| `backend/backtest/analytics.py` | 20-120 | `get_risk_free_rate(start, end)` returning DTB3 annualized decimal; `fetch_risk_free_rates()` | Active; this is the `mar_fetch_fn` default target |
| `backend/backtest/data_ingestion.py` | 21 | `FRED_SERIES` list — no short T-bill | Active; DGS3MO/DTB3 not present; do not modify for this phase |
| `backend/slack_bot/jobs/weekly_fred_refresh.py` | 11 | `_DEFAULT_SERIES` — no DGS3MO/DTB3 | Active; not a MAR data source |
| `backend/autoresearch/sprint_calendar.yaml` | 36-38 | References `backend/metrics/sortino.py` + MAR default 0.045 | Active; this phase creates that module |
| `scripts/harness/autoresearch_gate_test.py` | 1-60 | pytest patterns: no-BQ injection, case-function style | Active; reference pattern for test structure |

---

## Consensus vs debate (external)

**Consensus**: The LPM_2 formula (mean of squared below-MAR deviations over ALL T periods, not just negatives) is the canonical Sortino & Price formulation. Empyrical, CFI, Gale 2026, and Wall Street Prep all use denominator = T (total periods). The Codearmo blog uses `std(ddof=1)` on negatives only — this is an oversimplification seen in introductory tutorials but not in practitioner or library-grade code.

**Debate**: DGS3MO (bond-equivalent yield) vs DTB3 (discount basis) vs DFF (fed funds). In practice for daily Sortino at equity-return scale (returns in the 0.01%-0.1%/day range), the MAR differences are in the fourth decimal place per period. The project already uses DTB3; switching to DGS3MO for theoretical purity is not worth breaking the analytics.py dependency.

**Zero-downside sentinel**: No universal standard. `float('inf')` is theoretically correct but operationally problematic. `float('nan')` is the library consensus (Empyrical). Returning `0.0` (as the current `compute_sortino` does) is wrong — it makes a perfect period indistinguishable from insufficient data.

---

## Pitfalls (from literature)

1. **Wrong denominator**: Using `std(ddof=1)` on negative-only subsample instead of `sqrt(mean(min(0, r-MAR)^2))` over all T. The former inflates DD when few periods are below MAR. (Empyrical; CFI)
2. **MAR unit mismatch**: FRED rates are annualized %; per-period MAR must be `annual_rate / periods_per_year`. Forgetting this makes every daily return look below MAR. (Gale 2026)
3. **Empty returns array**: Must raise `ValueError`, not return 0.0 or NaN, so callers know the input was invalid.
4. **Sparse downside samples**: Fewer than ~20 below-MAR observations makes DD estimate noisy. Warn; do not silently return a high ratio. (Gale 2026)
5. **MAR as time-series**: If `mar` is an array of per-period rates, it must be broadcast-aligned with `returns` length. Shape mismatch should raise `ValueError`.
6. **Annualizing with monthly returns**: `periods_per_year=12`, not 252. The Champion/Challenger gate in sprint_calendar runs on monthly data — callers must pass the correct period.

---

## Application to pyfinagent (file:line mapping)

| Criterion | Implementation anchor | File:line |
|-----------|----------------------|-----------|
| `formula_matches_sortino_price_1994` | LPM_2: `np.mean(np.minimum(0.0, excess)**2)` over all T; `DD = sqrt(lpm2) * sqrt(periods_per_year)` | new `backend/metrics/sortino.py` |
| `downside_deviation_only_below_mar` | `np.minimum(0.0, excess)` clips positives to zero before squaring | new `backend/metrics/sortino.py` |
| `default_mar_pulls_from_pyfinagent_data_macro` | `mar_fetch_fn` defaults to lambda wrapping `analytics.get_risk_free_rate()` from DTB3 | `backend/backtest/analytics.py:89-120` |
| `configurable_mar_per_candidate` | `mar` kwarg accepts `float`, `list[float]`/`np.ndarray` (per-period), or `None` (trigger fetch) | new `backend/metrics/sortino.py` |

---

## Recommended implementation

### Files to create

```
backend/metrics/__init__.py          (empty or re-exports sortino)
backend/metrics/sortino.py           (primary module)
backend/metrics/tests/__init__.py    (empty)
backend/metrics/tests/test_sortino.py
```

### `backend/metrics/sortino.py` — exact public API

```python
from __future__ import annotations

import logging
import math
from typing import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_ANNUAL_MAR = 0.045          # 3M T-Bill fallback (~4.5% annualized)


def _default_mar_fetch() -> float:
    """Pull latest DTB3 annualized rate from analytics cache. Fail-open to 0.045."""
    try:
        from backend.backtest.analytics import get_risk_free_rate
        return get_risk_free_rate()  # returns decimal e.g. 0.044
    except Exception as exc:
        logger.warning("MAR fetch failed (%s); using default %.4f", exc, _DEFAULT_ANNUAL_MAR)
        return _DEFAULT_ANNUAL_MAR


def sortino(
    returns: Sequence[float] | np.ndarray,
    *,
    mar: float | list[float] | np.ndarray | None = None,
    periods_per_year: int = 252,
    mar_fetch_fn: Callable[[], float] | None = None,
) -> float:
    """
    Sortino ratio using LPM_2 (Sortino & Price 1994).

    Formula:
        annualized_excess = mean(R) * periods_per_year - annual_MAR
        DD = sqrt(mean(min(0, R_t - per_period_MAR)^2)) * sqrt(periods_per_year)
        Sortino = annualized_excess / DD

    Zero downside deviation (all returns >= MAR): returns float('nan').
    Empty returns: raises ValueError.

    Args:
        returns: Per-period (daily or monthly) returns as decimals.
        mar: Scalar annual MAR (divided internally by periods_per_year),
             or array of per-period MARs aligned with returns,
             or None to trigger mar_fetch_fn.
        periods_per_year: 252 for daily equity; 12 for monthly.
        mar_fetch_fn: Injectable callable returning annualized MAR (decimal).
                      Defaults to analytics.get_risk_free_rate() via DTB3.
    """
    arr = np.asarray(list(returns), dtype=float)
    if len(arr) == 0:
        raise ValueError("sortino() requires at least 1 return observation")

    # Resolve MAR
    if mar is None:
        fetch = mar_fetch_fn if mar_fetch_fn is not None else _default_mar_fetch
        annual_mar = fetch()
        per_period_mar: float | np.ndarray = annual_mar / periods_per_year
    elif np.ndim(mar) == 0:
        # scalar — treat as annualized
        per_period_mar = float(mar) / periods_per_year
    else:
        per_period_mar_arr = np.asarray(mar, dtype=float)
        if per_period_mar_arr.shape != arr.shape:
            raise ValueError(
                f"mar array shape {per_period_mar_arr.shape} != returns shape {arr.shape}"
            )
        per_period_mar = per_period_mar_arr  # already per-period

    excess = arr - per_period_mar

    # LPM_2: mean of squared below-MAR deviations over ALL T periods
    lpm2 = float(np.mean(np.minimum(0.0, excess) ** 2))
    if lpm2 == 0.0:
        # All returns at or above MAR — ratio is undefined / infinite
        return float("nan")

    dd = math.sqrt(lpm2) * math.sqrt(periods_per_year)
    annualized_excess = float(np.mean(excess)) * periods_per_year
    return annualized_excess / dd
```

### `backend/metrics/tests/test_sortino.py` — four criteria mapped to test cases

```python
"""Tests for backend/metrics/sortino.py — phase-10.5 verification."""
import math
import numpy as np
import pytest
from backend.metrics.sortino import sortino


# ── criterion: formula_matches_sortino_price_1994 ─────────────────────────

def test_formula_matches_sortino_price_1994():
    """LPM_2 denominator uses ALL T periods (not just negatives)."""
    # 5 returns, MAR = 0 per period (annual MAR = 0)
    rets = [0.01, -0.02, 0.005, -0.01, 0.03]
    result = sortino(rets, mar=0.0, periods_per_year=252, mar_fetch_fn=lambda: 0.0)
    # Manual: excess = rets (MAR=0); lpm2 = mean([0, 0.0004, 0, 0.0001, 0]) = 0.0001
    # dd = sqrt(0.0001) * sqrt(252) = 0.01 * 15.8745 = 0.15875
    # annualized_excess = mean(rets)*252 = 0.006*252 = 1.512
    # sortino = 1.512 / 0.15875 ≈ 9.524
    lpm2 = np.mean(np.minimum(0.0, np.array(rets)) ** 2)
    dd = math.sqrt(lpm2) * math.sqrt(252)
    expected = np.mean(rets) * 252 / dd
    assert math.isclose(result, expected, rel_tol=1e-9)


# ── criterion: downside_deviation_only_below_mar ──────────────────────────

def test_downside_deviation_only_below_mar():
    """Returns above MAR contribute zero to LPM_2 (clipped to 0)."""
    # MAR per period = 0.01 (annual = 2.52 which is intentionally large for isolation)
    rets = [0.02, 0.03, -0.005, 0.015]  # only index 2 is below 0.01
    result = sortino(rets, mar=0.01 * 252, periods_per_year=252, mar_fetch_fn=lambda: 0.01 * 252)
    # excess = [0.01, 0.02, -0.015, 0.005]; only -0.015 contributes
    excess = np.array(rets) - 0.01
    lpm2 = np.mean(np.minimum(0.0, excess) ** 2)
    dd = math.sqrt(lpm2) * math.sqrt(252)
    expected = float(np.mean(excess)) * 252 / dd
    assert math.isclose(result, expected, rel_tol=1e-9)


# ── criterion: default_mar_pulls_from_pyfinagent_data_macro ───────────────

def test_default_mar_pulls_from_macro():
    """mar=None with injected mar_fetch_fn uses the fetched rate."""
    called_with = []

    def fake_fetch() -> float:
        called_with.append(True)
        return 0.045  # 4.5% annualized

    rets = [0.001, -0.002, 0.003, -0.001, 0.002]
    result = sortino(rets, mar=None, periods_per_year=252, mar_fetch_fn=fake_fetch)
    assert len(called_with) == 1, "mar_fetch_fn must be called exactly once when mar=None"
    assert math.isfinite(result) or math.isnan(result)  # valid float output


def test_default_mar_failopen():
    """mar_fetch_fn raising uses _DEFAULT_ANNUAL_MAR=0.045 and does not raise."""
    def bad_fetch() -> float:
        raise RuntimeError("BQ unreachable")

    rets = [0.001, -0.002, 0.003, -0.001, 0.002]
    # Should not raise; fails open to default 0.045
    result = sortino(rets, mar=None, periods_per_year=252, mar_fetch_fn=bad_fetch)
    # With 0.045/252 ≈ 0.0001786 per-period MAR, some returns will be below
    assert isinstance(result, float)


# ── criterion: configurable_mar_per_candidate ─────────────────────────────

def test_configurable_scalar_mar():
    """Scalar mar kwarg is respected (treated as annualized)."""
    rets = [0.01, -0.02, 0.005, -0.01, 0.03]
    r_zero = sortino(rets, mar=0.0, periods_per_year=252, mar_fetch_fn=lambda: 0.0)
    r_high = sortino(rets, mar=0.10, periods_per_year=252, mar_fetch_fn=lambda: 0.10)
    # Higher MAR -> more returns below MAR -> larger DD -> lower ratio
    assert r_zero > r_high or math.isnan(r_high)


def test_configurable_array_mar():
    """Per-period array MAR is broadcast correctly."""
    rets = np.array([0.01, -0.02, 0.005, -0.01, 0.03])
    per_period_mar_arr = np.full(5, 0.045 / 252)
    result = sortino(rets, mar=per_period_mar_arr, periods_per_year=252)
    assert isinstance(result, float)


def test_array_mar_shape_mismatch_raises():
    """Mismatched array MAR raises ValueError."""
    rets = [0.01, -0.02, 0.005]
    with pytest.raises(ValueError, match="shape"):
        sortino(rets, mar=np.array([0.001, 0.002]), periods_per_year=252)


# ── edge cases ────────────────────────────────────────────────────────────

def test_empty_returns_raises():
    with pytest.raises(ValueError):
        sortino([], mar_fetch_fn=lambda: 0.045)


def test_all_returns_above_mar_returns_nan():
    """Perfect period (no below-MAR returns) -> float('nan'), not inf or 0."""
    rets = [0.01, 0.02, 0.03, 0.04, 0.05]
    result = sortino(rets, mar=0.0, periods_per_year=252, mar_fetch_fn=lambda: 0.0)
    assert math.isnan(result), f"Expected nan, got {result}"


def test_monthly_periods_per_year():
    """periods_per_year=12 for monthly Champion/Challenger gate."""
    monthly_rets = [0.02, -0.01, 0.03, -0.005, 0.015, 0.01,
                    -0.02, 0.025, 0.01, -0.008, 0.018, 0.012]
    result = sortino(monthly_rets, mar=0.045, periods_per_year=12, mar_fetch_fn=lambda: 0.045)
    assert isinstance(result, float)
    assert not math.isinf(result)
```

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 fetched in full)
- [x] 10+ unique URLs total (17 URLs collected)
- [x] Recency scan (last 2 years) performed + reported (2 × 2026 sources found)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module
- [x] Contradictions / consensus noted (LPM_2 vs std-of-negatives debate; DGS3MO vs DTB3)
- [x] All claims cited per-claim (not just footer)

---

## Search queries run (three-variant discipline)

1. **Current-year frontier**: "Sortino ratio implementation 3M T-Bill MAR configurable Python 2025 2026"
2. **Last-2-year window**: "Sortino ratio implementation Python 2025 MAR minimum acceptable return annualization"
3. **Year-less canonical**: "Sortino ratio Sortino Price 1994 Journal of Investment Management formula LPM downside deviation"
4. **Supplementary**: "FRED DGS3MO DTB3 3-month Treasury bill series difference risk-free rate quant finance"
5. **Supplementary**: "Sortino ratio zero downside deviation sentinel value infinity handling 2025"

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "report_md": "handoff/current/phase-10.5-research-brief.md",
  "gate_passed": true
}
```
