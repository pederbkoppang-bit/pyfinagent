"""phase-69.2 gate-correctness fixtures.

Covers the five offline promotion-gate fixes and asserts the immutable
DSR>=0.95 / PSR>=0.95 / thresholds are byte-untouched:
  1. DSR unit correction (annualized-SR vs per-period-T) pinned to the
     Bailey-Borwein-Lopez de Prado-Zhu reference value (0.9004).
  2. Purge + embargo (label horizon 1.5*holding_days overlap test).
  3. Boundary business-day snap for exact-date price lookups.
  4. Fracdiff-at-predict / NaN-fill consistency (train medians at predict).
  5. Go-live booleans (30-day PSR sustainment + backtest-DD+5pp tolerance).
"""

import pandas as pd

from backend.backtest.analytics import compute_deflated_sharpe
from backend.backtest.backtest_engine import BacktestEngine, _NON_STATIONARY
import backend.backtest.backtest_engine as bt_engine
import backend.services.paper_go_live_gate as gate


# ----------------------------------------------------------------------
# 1. DSR unit correction
# ----------------------------------------------------------------------
def test_dsr_reference_value_matches_bailey():
    # Bailey paper numerical example: annualized SR 2.5, 5y daily (T=1250),
    # N=100 trials, V=0.5, skew=-3, kurt=10, ppy=250 -> DSR ~= 0.90.
    dsr = compute_deflated_sharpe(2.5, 100, 0.5, -3, 10, T=1250, periods_per_year=250)
    assert 0.88 < dsr < 0.92, f"corrected DSR {dsr} should match Bailey ~0.9004"


def test_dsr_bug_path_inflated_without_correction():
    bug = compute_deflated_sharpe(2.5, 100, 0.5, -3, 10, T=1250, periods_per_year=1)
    assert bug > 0.999, "annualized SR + daily T should inflate DSR to ~1.0 (the bug)"


def test_dsr_default_is_byte_identical_to_ppy1():
    # Default periods_per_year must be a no-op (do-no-harm for existing callers).
    default = compute_deflated_sharpe(2.5, 100, 0.5, -3, 10, T=1250)
    ppy1 = compute_deflated_sharpe(2.5, 100, 0.5, -3, 10, T=1250, periods_per_year=1)
    assert default == ppy1


def test_dsr_correction_dramatically_lowers_inflated_value():
    bug = compute_deflated_sharpe(2.5, 100, 0.5, -3, 10, T=1250, periods_per_year=1)
    fix = compute_deflated_sharpe(2.5, 100, 0.5, -3, 10, T=1250, periods_per_year=250)
    assert bug - fix > 0.05, "de-annualization must remove the ~sqrt(252) inflation"


# ----------------------------------------------------------------------
# 2. Purge + embargo (AFML Ch.7)
# ----------------------------------------------------------------------
def test_purge_label_reaching_into_test_is_purged():
    assert BacktestEngine._label_overlaps_test("2024-01-01", 135, "2024-03-01", "2024-06-01") is True


def test_purge_label_ending_before_test_is_kept():
    assert BacktestEngine._label_overlaps_test("2024-01-01", 30, "2024-03-01", "2024-06-01") is False


def test_purge_uses_1_5_holding_days_horizon():
    # holding_days=90 ends before the test window (kept); 1.5*90=135 reaches in (purged).
    keep = BacktestEngine._label_overlaps_test("2024-01-01", 90, "2024-04-01", "2024-06-01")
    purge = BacktestEngine._label_overlaps_test("2024-01-01", 135, "2024-04-01", "2024-06-01")
    assert keep is False and purge is True


# ----------------------------------------------------------------------
# 3. Boundary business-day snap
# ----------------------------------------------------------------------
def test_price_asof_snaps_weekend_to_prior_trading_day(monkeypatch):
    def fake_cached_prices(ticker, start, end):
        if start == end:  # exact-date lookup
            if start == "2024-06-01":  # a Saturday -> empty
                return pd.DataFrame()
            return pd.DataFrame({"close": [100.0]})
        # widened [d-7, d] range -> prior trading closes; last is Friday
        return pd.DataFrame({"close": [98.0, 99.0, 100.0]})

    monkeypatch.setattr(bt_engine.cache, "cached_prices", fake_cached_prices)
    assert BacktestEngine._price_asof("AAA", "2024-05-31") == 100.0          # exact hit
    assert BacktestEngine._price_asof("AAA", "2024-06-01") == 100.0          # weekend -> snap prior


def test_price_asof_none_when_no_data(monkeypatch):
    monkeypatch.setattr(bt_engine.cache, "cached_prices", lambda t, s, e: pd.DataFrame())
    assert BacktestEngine._price_asof("AAA", "2024-06-01") is None


# ----------------------------------------------------------------------
# 4. Fracdiff-at-predict / NaN-fill consistency
# ----------------------------------------------------------------------
def test_predict_uses_train_median_not_zero_for_missing():
    feature_names = ["momentum", "price_at_analysis"]
    train_medians = {"momentum": 0.5, "price_at_analysis": 0.02}
    X = BacktestEngine._build_predict_features({}, feature_names, train_medians)
    assert X.loc[0, "momentum"] == 0.5, "missing feature must impute the TRAIN median, not 0"


def test_predict_nonstationary_placed_on_train_scale():
    assert "price_at_analysis" in _NON_STATIONARY
    feature_names = ["momentum", "price_at_analysis"]
    train_medians = {"momentum": 0.5, "price_at_analysis": 0.02}  # fracdiff'd scale
    fv = {"momentum": 0.7, "price_at_analysis": 546.72}           # raw price level
    X = BacktestEngine._build_predict_features(fv, feature_names, train_medians)
    assert X.loc[0, "price_at_analysis"] == 0.02, "non-stationary must be on train scale, not raw 546.72"
    assert X.loc[0, "momentum"] == 0.7, "stationary feature passes through unchanged"


def test_predict_no_train_medians_falls_back_to_zero():
    # No persisted medians (e.g. degenerate case) -> fillna(0), never crashes.
    X = BacktestEngine._build_predict_features({}, ["a", "b"], None)
    assert X.loc[0, "a"] == 0 and X.loc[0, "b"] == 0


# ----------------------------------------------------------------------
# 5. Go-live booleans
# ----------------------------------------------------------------------
def test_load_backtest_max_dd_returns_none_when_absent():
    # optimizer_best.json currently carries no DD key -> None -> 20% cap fallback.
    assert gate._load_backtest_max_dd() is None


def test_dd_tolerance_falls_back_to_20pct_when_no_backtest_dd():
    bt_dd = gate._load_backtest_max_dd()
    dd_tol = (bt_dd + 5.0) if bt_dd is not None else gate.MAX_DD_ABS_TOLERANCE
    assert dd_tol == 20.0


class _StubBQ:
    def __init__(self, navs):
        self._navs = navs

    def get_paper_snapshots(self, limit=365):
        return [
            {"snapshot_date": f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}", "total_nav": nav}
            for i, nav in enumerate(self._navs)
        ]


def test_sustained_psr_insufficient_history():
    sustained, minp, n = gate._sustained_psr_ge(_StubBQ([100.0, 101.0, 102.0]))
    assert sustained is False and n == 0


def test_sustained_psr_strong_uptrend_sustains():
    rets = [0.004, 0.002, 0.005, 0.003] * 18  # steady low-vol positive drift
    navs = [100.0]
    for r in rets[:70]:
        navs.append(navs[-1] * (1 + r))
    sustained, minp, n = gate._sustained_psr_ge(_StubBQ(navs))
    assert sustained is True, f"steady uptrend should sustain PSR>=0.95 (min={minp}, n={n})"
    assert n >= gate.PSR_SUSTAINED_DAYS


def test_sustained_psr_flat_noisy_not_sustained():
    navs = [100.0]
    for i in range(70):
        navs.append(navs[-1] * (1.01 if i % 2 == 0 else 0.99))  # ~zero mean, high vol
    sustained, minp, n = gate._sustained_psr_ge(_StubBQ(navs))
    assert sustained is False


# ----------------------------------------------------------------------
# Do-no-harm: immutable thresholds byte-untouched
# ----------------------------------------------------------------------
def test_immutable_thresholds_unchanged():
    assert gate.DSR_THRESHOLD == 0.95
    assert gate.PSR_THRESHOLD == 0.95
    assert gate.MAX_DD_ABS_TOLERANCE == 20.0
    assert gate.TRADES_THRESHOLD == 100
