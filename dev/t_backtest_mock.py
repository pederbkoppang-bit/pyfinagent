"""
Mock-test script for the Phase 5A backtest engine changes.

Exercises the full walk-forward pipeline with synthetic data:
- 20 tickers × 2 years of daily OHLCV
- Quarterly fundamentals per ticker
- Monthly FRED macro indicators
- Monkey-patches the cache module to serve synthetic data (no BQ)
- Runs all 5 strategies through the engine
- Validates feature vectors, labels, training, trading, and metrics

Usage:
    .venv312/Scripts/python -m t_backtest_mock
"""

import logging
import math
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger("mock_backtest")

# ── Synthetic Data Generation ────────────────────────────────────

TICKERS = [f"MOCK{i:02d}" for i in range(20)]
START = "2023-01-01"
END = "2025-06-30"
DATE_RANGE = pd.bdate_range(START, END)  # business days only

np.random.seed(42)


def _generate_prices(ticker: str) -> pd.DataFrame:
    """GBM prices with random drift and vol per ticker."""
    n = len(DATE_RANGE)
    seed = hash(ticker) % 2**31
    rng = np.random.RandomState(seed)
    drift = rng.uniform(-0.0002, 0.0005)
    vol = rng.uniform(0.01, 0.03)
    log_returns = drift + vol * rng.randn(n)
    price0 = rng.uniform(20, 400)
    prices = price0 * np.exp(np.cumsum(log_returns))
    volumes = rng.randint(100_000, 10_000_000, size=n)

    df = pd.DataFrame({
        "open": prices * rng.uniform(0.99, 1.01, n),
        "high": prices * rng.uniform(1.00, 1.03, n),
        "low": prices * rng.uniform(0.97, 1.00, n),
        "close": prices,
        "volume": volumes,
    }, index=pd.DatetimeIndex(DATE_RANGE, name="date"))
    return df


def _generate_fundamentals(ticker: str) -> list[dict]:
    """Quarterly financial statements."""
    rng = np.random.RandomState(hash(ticker) % 2**31 + 1)
    quarters = pd.date_range("2022-09-30", END, freq="QE")
    rows = []
    rev = rng.uniform(1e9, 50e9)
    for q in quarters:
        rev *= rng.uniform(0.95, 1.08)
        ni = rev * rng.uniform(0.05, 0.25)
        debt = rev * rng.uniform(0.3, 1.5)
        equity = rev * rng.uniform(0.5, 2.0)
        assets = debt + equity
        ocf = ni * rng.uniform(0.8, 1.4)
        shares = rng.uniform(1e8, 5e9)
        rows.append({
            "ticker": ticker,
            "report_date": q.strftime("%Y-%m-%d"),
            "filing_date": (q + timedelta(days=rng.randint(30, 60))).strftime("%Y-%m-%d"),
            "total_revenue": float(rev),
            "net_income": float(ni),
            "total_debt": float(debt),
            "total_equity": float(equity),
            "total_assets": float(assets),
            "operating_cash_flow": float(ocf),
            "shares_outstanding": float(shares),
            "sector": "Technology" if hash(ticker) % 3 == 0 else "Healthcare" if hash(ticker) % 3 == 1 else "Financials",
            "industry": "Software",
        })
    return rows


FRED_SERIES = {
    "FEDFUNDS": 5.25,
    "CPIAUCSL": 3.2,
    "UNRATE": 3.8,
    "GDP": 2.1,
    "T10Y2Y": 0.5,
    "UMCSENT": 65.0,
    "DGS10": 4.3,
}

# Pre-generate all data
_PRICES_STORE: dict[str, pd.DataFrame] = {t: _generate_prices(t) for t in TICKERS}
_FUNDAMENTALS_STORE: dict[str, list[dict]] = {t: _generate_fundamentals(t) for t in TICKERS}


# ── Monkey-Patching the Cache ────────────────────────────────────

def mock_init_cache(bq_client, project, dataset):
    """No-op: skip BQ initialization."""
    pass


def mock_cached_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Serve from in-memory synthetic data."""
    df = _PRICES_STORE.get(ticker)
    if df is None:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
    return df.loc[mask]


def mock_cached_fundamentals(ticker: str, cutoff_date: str) -> dict:
    """Most recent quarterly fundamental as-of cutoff_date."""
    rows = _FUNDAMENTALS_STORE.get(ticker, [])
    cutoff = pd.Timestamp(cutoff_date)
    candidates = [r for r in rows if pd.Timestamp(r["report_date"]) <= cutoff]
    return candidates[-1] if candidates else {}


def mock_cached_macro(cutoff_date: str) -> dict:
    """Static FRED values."""
    return {
        sid: {"value": val, "date": cutoff_date}
        for sid, val in FRED_SERIES.items()
    }


def patch_cache():
    """Replace the BQ cache functions with in-memory mocks."""
    from backend.backtest import cache as cache_mod
    cache_mod.init_cache = mock_init_cache
    cache_mod.cached_prices = mock_cached_prices
    cache_mod.cached_fundamentals = mock_cached_fundamentals
    cache_mod.cached_macro = mock_cached_macro
    logger.info("Cache module patched with synthetic data")


# ── Monkey-Patching auto-ingest (no BQ available) ───────────────

def patch_auto_ingest():
    """Disable auto-ingest since we have no BQ."""
    from backend.backtest.backtest_engine import BacktestEngine
    BacktestEngine._auto_ingest_if_needed = lambda self, universe_tickers: None
    logger.info("Auto-ingest disabled for mock test")


# ── Tests ────────────────────────────────────────────────────────

def test_feature_vectors():
    """Verify HistoricalDataProvider produces ~49-feature vectors."""
    from backend.backtest.historical_data import HistoricalDataProvider
    provider = HistoricalDataProvider()
    fv = provider.build_feature_vector("MOCK00", "2024-06-15")
    assert fv, "Feature vector is empty"
    assert fv.get("price_at_analysis") is not None, "Missing price_at_analysis"

    # Check new features exist
    new_features = ["volume_ratio_20d", "pb_ratio", "fcf_yield", "dividend_yield", "quality_score"]
    found = [f for f in new_features if f in fv and fv[f] is not None]
    logger.info(f"Feature vector has {len(fv)} keys, new features found: {found}")
    assert len(found) >= 3, f"Expected ≥3 new features, got {len(found)}: {found}"
    logger.info("✓ Feature vector test passed")


def test_label_methods():
    """Verify all strategy label methods produce valid labels."""
    from backend.backtest.backtest_engine import BacktestEngine, STRATEGY_REGISTRY

    engine = BacktestEngine(
        bq_client=MagicMock(), project="test", dataset="test",
        start_date="2024-01-01", end_date="2024-12-31",
        train_window_months=6, test_window_months=2,
    )

    for strategy_name, method_name in STRATEGY_REGISTRY.items():
        method = getattr(engine, method_name)
        labels = []
        for ticker in TICKERS[:5]:
            label = method(ticker, "2024-06-15")
            if label is not None:
                labels.append(label)
        logger.info(f"  Strategy '{strategy_name}' ({method_name}): {len(labels)} labels → {set(labels)}")
        assert len(labels) > 0, f"Strategy {strategy_name} produced no labels"
        assert all(l in (-1, 0, 1) for l in labels), f"Invalid labels for {strategy_name}: {labels}"

    logger.info("✓ Label methods test passed")


def test_candidate_selector():
    """Verify candidate_selector works with scoring_weights."""
    from backend.backtest.candidate_selector import CandidateSelector
    selector = CandidateSelector()

    # Default weights
    results = selector.screen_at_date("2024-06-15", TICKERS, top_n=10)
    assert len(results) > 0, "No candidates returned with default weights"
    logger.info(f"  Default weights: {len(results)} candidates")

    # Custom weights (momentum-heavy)
    results_custom = selector.screen_at_date(
        "2024-06-15", TICKERS, top_n=10,
        scoring_weights={"momentum_weight": 0.7, "rsi_weight": 0.1, "volatility_weight": 0.1, "sma_weight": 0.1},
    )
    assert len(results_custom) > 0, "No candidates with custom weights"
    logger.info(f"  Custom weights: {len(results_custom)} candidates")

    logger.info("✓ Candidate selector test passed")


def test_backtest_engine_all_strategies():
    """Run a short backtest with each strategy and verify results."""
    from backend.backtest.backtest_engine import BacktestEngine, STRATEGY_REGISTRY, BacktestResult

    for strategy_name in STRATEGY_REGISTRY:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing strategy: {strategy_name}")
        logger.info(f"{'='*60}")

        engine = BacktestEngine(
            bq_client=MagicMock(), project="test", dataset="test",
            start_date="2024-01-01", end_date="2025-03-31",
            train_window_months=6, test_window_months=3,
            strategy=strategy_name,
            top_n_candidates=15,
            max_positions=5,
            starting_capital=100_000.0,
        )

        result = engine.run_backtest(TICKERS)

        assert isinstance(result, BacktestResult), f"Expected BacktestResult, got {type(result)}"
        assert result.strategy_params.get("strategy") == strategy_name, "Strategy param mismatch"
        logger.info(f"  Windows: {len(result.windows)}")
        logger.info(f"  Total trades: {result.total_trades}")
        logger.info(f"  Sharpe: {result.aggregate_sharpe:.4f}")
        logger.info(f"  Return: {result.aggregate_return_pct:.2f}%")
        logger.info(f"  Hit rate: {result.aggregate_hit_rate:.2%}")
        logger.info(f"  MDA features: {len(result.feature_importance_mda)}")

        if result.feature_importance_mda:
            top5 = sorted(result.feature_importance_mda.items(), key=lambda kv: kv[1], reverse=True)[:5]
            logger.info(f"  Top-5 MDA: {[f'{k}={v:.4f}' for k, v in top5]}")

    logger.info("\n✓ All strategy backtest tests passed")


def test_quant_optimizer_helpers():
    """Verify the new quant_optimizer helper methods."""
    from backend.backtest.quant_optimizer import QuantStrategyOptimizer
    from backend.backtest.backtest_engine import BacktestEngine, BacktestResult, WindowResult

    engine = BacktestEngine(
        bq_client=MagicMock(), project="test", dataset="test",
        start_date="2024-01-01", end_date="2024-12-31",
    )
    optimizer = QuantStrategyOptimizer(engine)

    # Test _extract_top5_mda
    fake_result = BacktestResult(
        feature_importance_mda={
            "momentum_6m": 0.15, "rsi_14": 0.12, "pe_ratio": 0.10,
            "quality_score": 0.09, "var_95_6m": 0.08, "sma_50_distance": 0.05,
        }
    )
    top5 = optimizer._extract_top5_mda(fake_result)
    assert len(top5) == 5, f"Expected 5, got {len(top5)}"
    assert top5[0] == "momentum_6m", f"Expected momentum_6m first, got {top5[0]}"
    logger.info(f"  _extract_top5_mda: {top5}")

    # Test _detect_feature_drift (no previous → no warning)
    optimizer._prev_top5_mda = []
    optimizer._detect_feature_drift(top5)  # Should not warn

    # Test _detect_feature_drift (with change)
    optimizer._prev_top5_mda = ["momentum_6m", "rsi_14", "pe_ratio", "debt_equity", "var_95_6m"]
    optimizer._detect_feature_drift(top5)  # Should log warning about quality_score replacing debt_equity

    # Test _check_model_staleness (recent)
    engine.model_trained_at = datetime.now(timezone.utc).isoformat()
    optimizer._check_model_staleness()  # Should not warn

    # Test _check_model_staleness (old)
    engine.model_trained_at = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    optimizer._check_model_staleness()  # Should warn

    # Test _propose_random handles categorical
    proposals = [optimizer._propose_random() for _ in range(50)]
    strategy_proposals = [p for p in proposals if p["param"] == "strategy"]
    logger.info(f"  50 random proposals: {len(strategy_proposals)} strategy changes")
    # With ~16 params (15 numeric + 1 categorical), expect ~3 strategy proposals in 50 tries
    # But due to randomness, just check at least 0 (it's possible to get 0)
    assert all(p["value"] in ["triple_barrier", "quality_momentum", "mean_reversion", "factor_model", "meta_label"]
               for p in strategy_proposals), "Invalid strategy value in proposal"

    logger.info("✓ Quant optimizer helper tests passed")


def test_model_staleness_tracking():
    """Verify model_trained_at is set after training."""
    from backend.backtest.backtest_engine import BacktestEngine

    engine = BacktestEngine(
        bq_client=MagicMock(), project="test", dataset="test",
        start_date="2024-01-01", end_date="2025-03-31",
        train_window_months=6, test_window_months=3,
        top_n_candidates=15,
    )

    assert engine.get_model_trained_at() == "", "Should be empty before training"

    result = engine.run_backtest(TICKERS)

    trained_at = engine.get_model_trained_at()
    logger.info(f"  model_trained_at: {trained_at}")
    if result.windows:
        assert trained_at != "", "model_trained_at should be set after successful training"
    logger.info("✓ Model staleness tracking test passed")


# ── Main ─────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Phase 5A Mock Backtest Test Suite")
    logger.info("=" * 60)
    logger.info(f"Tickers: {len(TICKERS)}, Date range: {START} → {END}")
    logger.info(f"Business days: {len(DATE_RANGE)}")
    logger.info("")

    # Patch before imports that trigger cache usage
    patch_cache()
    patch_auto_ingest()

    tests = [
        ("Feature Vectors", test_feature_vectors),
        ("Label Methods", test_label_methods),
        ("Candidate Selector", test_candidate_selector),
        ("Backtest Engine (all strategies)", test_backtest_engine_all_strategies),
        ("Quant Optimizer Helpers", test_quant_optimizer_helpers),
        ("Model Staleness Tracking", test_model_staleness_tracking),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        logger.info(f"\n{'─'*60}")
        logger.info(f"Running: {name}")
        logger.info(f"{'─'*60}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            logger.error(f"✗ FAILED: {name} — {e}", exc_info=True)
            failed += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    logger.info(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
