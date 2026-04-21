"""phase-10.5 Sortino canonical implementation tests."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.metrics import sortino as sortino_mod
from backend.metrics.sortino import sortino


def test_formula_matches_sortino_price_1994():
    """LPM_2 formula check against a hand-computed example.

    returns = [+0.10, +0.04, -0.02, -0.05, +0.03]
    mar = 0.00 (per-period)
    excess = [+0.10, +0.04, -0.02, -0.05, +0.03]; mean = +0.02
    downside_excess = [0, 0, 0.02, 0.05, 0]
    dd2 = mean([0, 0, 0.0004, 0.0025, 0]) = 0.00058
    dd  = sqrt(0.00058) ~= 0.02408319...
    sortino_per_period = 0.02 / 0.02408319 = 0.83049...
    annualized(252) = 0.83049 * sqrt(252) = 13.1823...
    """
    returns = [0.10, 0.04, -0.02, -0.05, 0.03]
    s = sortino(returns, mar=0.0, periods_per_year=252)
    assert math.isclose(s, 13.1823, abs_tol=1e-3), f"got {s}"


def test_downside_deviation_only_below_mar():
    """Above-MAR returns must contribute 0 to DD (clip at 0)."""
    # All positive returns; downside deviation must be 0 -> NaN sentinel.
    r_all_pos = [0.05, 0.05, 0.05, 0.05]
    s = sortino(r_all_pos, mar=0.0, periods_per_year=252)
    assert math.isnan(s)

    # Compare: same total return, but one period dips below MAR.
    r_mixed = [0.05, 0.05, 0.05, -0.05]
    s_mixed = sortino(r_mixed, mar=0.0, periods_per_year=252)
    assert not math.isnan(s_mixed)


def test_default_mar_pulls_from_pyfinagent_data_macro(monkeypatch):
    """When mar=None, default fetcher must try BQ historical_macro first."""
    calls = {"bq_client_created": 0, "query_executed": 0}

    class _StubRow(dict):
        def get(self, k, default=None):
            return dict.get(self, k, default)

    class _StubQueryResult:
        def result(self):
            # Return one row: DGS3MO at 4.5%
            return iter([_StubRow(value=4.5)])

    class _StubClient:
        def __init__(self, project=None):
            calls["bq_client_created"] += 1

        def query(self, sql):
            calls["query_executed"] += 1
            assert "historical_macro" in sql, "SQL must reference historical_macro"
            assert "DGS3MO" in sql or "DTB3" in sql, "SQL must reference DGS3MO or DTB3"
            return _StubQueryResult()

    # Patch google.cloud.bigquery.Client
    import google.cloud.bigquery as bq
    monkeypatch.setattr(bq, "Client", _StubClient)

    result = sortino_mod._default_mar_fetcher()
    assert calls["bq_client_created"] == 1
    assert calls["query_executed"] == 1
    # 4.5 > 1.0 so fetcher divides by 100 -> 0.045
    assert math.isclose(result, 0.045, abs_tol=1e-9)


def test_configurable_mar_per_candidate_scalar():
    """Different scalar MAR values must yield different Sortino values."""
    r = [0.10, 0.04, -0.02, -0.05, 0.03]
    s_low = sortino(r, mar=0.01, periods_per_year=252)
    s_high = sortino(r, mar=0.03, periods_per_year=252)
    assert not math.isclose(s_low, s_high), "different MAR must give different Sortino"


def test_configurable_mar_per_candidate_array():
    """Per-period MAR array of same length as returns is accepted."""
    r = [0.10, 0.04, -0.02, -0.05, 0.03]
    mar_series = [0.0, 0.0, 0.01, 0.01, 0.0]
    s = sortino(r, mar=mar_series, periods_per_year=252)
    assert isinstance(s, float)
    assert not math.isnan(s)


def test_mar_array_shape_mismatch_raises():
    r = [0.1, 0.2, 0.3]
    with pytest.raises(ValueError, match="shape"):
        sortino(r, mar=[0.01, 0.01], periods_per_year=252)


def test_all_returns_above_mar_returns_nan():
    """Zero-downside sentinel is NaN, not +inf or 0.0."""
    s = sortino([0.05, 0.06, 0.07], mar=0.0, periods_per_year=252)
    assert math.isnan(s)


def test_annualization_daily_vs_monthly():
    """sqrt(252) vs sqrt(12); scaling factor = sqrt(252/12) = sqrt(21)."""
    r = [0.02, -0.03, 0.01, -0.01, 0.03]
    s_daily = sortino(r, mar=0.0, periods_per_year=252)
    s_monthly = sortino(r, mar=0.0, periods_per_year=12)
    ratio = s_daily / s_monthly
    expected = math.sqrt(252.0 / 12.0)
    assert math.isclose(ratio, expected, rel_tol=1e-9)


def test_mar_fetch_fn_injectable():
    """Custom mar_fetch_fn called when mar=None; output consumed as annual rate."""
    calls = []

    def stub_fetch() -> float:
        calls.append(1)
        return 0.06  # 6% annualized

    r = [0.02, -0.03, 0.01, -0.01, 0.03]
    s = sortino(r, mar=None, mar_fetch_fn=stub_fetch, periods_per_year=252)
    assert calls == [1], "fetcher must be invoked exactly once when mar=None"
    assert isinstance(s, float)


def test_mar_fetch_fn_fail_open_to_default():
    """If fetcher raises, the function falls back to _DEFAULT_ANNUAL_MAR."""
    def boom() -> float:
        raise RuntimeError("fetch failed")

    r = [0.02, -0.03, 0.01, -0.01, 0.03]
    s_boom = sortino(r, mar=None, mar_fetch_fn=boom, periods_per_year=252)
    s_default = sortino(r, mar=None, mar_fetch_fn=lambda: 0.045, periods_per_year=252)
    assert math.isclose(s_boom, s_default, rel_tol=1e-9)


def test_fewer_than_two_samples_returns_nan():
    assert math.isnan(sortino([0.05], mar=0.0))
    assert math.isnan(sortino([], mar=0.0))
