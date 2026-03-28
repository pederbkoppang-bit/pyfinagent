"""
Backtest engine — main orchestrator for walk-forward ML backtesting.
Trains HistGradientBoosting with Triple Barrier labels, sample weights,
fractional differentiation, and meta-labeling position sizing.
Zero LLM cost.

Model choice: HistGradientBoostingClassifier (sklearn)
  - Inspired by LightGBM (Ke et al., NeurIPS 2017)
  - 5-20x faster than GradientBoostingClassifier on our data sizes (3K-10K samples)
  - Same accuracy (proven by Grinsztajn et al., NeurIPS 2022; McElfresh et al., NeurIPS 2023)
  - Built-in OpenMP parallelism, native missing value handling
  - Histogram binning (255 bins) acts as implicit regularization — beneficial for noisy financial data
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from backend.backtest.historical_data import HistoricalDataProvider
from backend.backtest.candidate_selector import CandidateSelector
from backend.backtest.walk_forward import WalkForwardScheduler, WalkForwardWindow
from backend.backtest.backtest_trader import BacktestTrader
from backend.backtest import cache

logger = logging.getLogger(__name__)

# ── Strategy Registry ────────────────────────────────────────────
# Maps strategy name → label computation method name on BacktestEngine.
# QuantOptimizer rotates among these as a categorical parameter.
STRATEGY_REGISTRY: dict[str, str] = {
    "triple_barrier": "_compute_triple_barrier_label",
    "quality_momentum": "_compute_quality_momentum_label",
    "mean_reversion": "_compute_mean_reversion_label",
    "factor_model": "_compute_factor_label",
    "meta_label": "_compute_triple_barrier_label",  # uses TB labels; meta-labeling applied in _run_window
    "blend": "_compute_blend_label",  # weighted vote across strategies (Dietterich 2000)
}

# Numeric features used for ML training (excludes categorical: ticker, date, sector, industry)
_NUMERIC_FEATURES = [
    "price_at_analysis", "momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m",
    "momentum_12_1",  # Jegadeesh & Titman (1993): 12m minus 1m, avoids short-term reversal
    "rsi_14", "annualized_volatility", "daily_volatility",
    "sma_50_distance", "sma_200_distance",
    "bb_upper_distance", "bb_lower_distance", "bb_pct_b",  # Bollinger Bands for mean reversion
    "var_95_6m", "var_99_6m", "expected_shortfall_6m", "prob_positive_6m",
    "anomaly_count", "amihud_illiquidity", "volume_ratio_20d",
    "pe_ratio", "pb_ratio", "debt_equity", "roe", "profit_margin", "market_cap",
    "total_revenue", "net_income", "total_debt", "total_equity", "total_assets",
    "fcf_yield", "dividend_yield", "quality_score", "revenue_growth_yoy",
    "fed_funds_rate", "cpi_yoy", "unemployment_rate", "yield_curve_spread",
    "consumer_sentiment", "treasury_10y",
]

# Non-stationary features that need fractional differentiation
_NON_STATIONARY = {"price_at_analysis", "market_cap", "total_revenue", "total_debt", "total_equity"}
# Note: daily_volatility is NOT included — it's already a derived statistic (std of returns)

# ── MDA Cache ────────────────────────────────────────────────────
# Persists the latest aggregate MDA weights so that the live quant_model
# tool can read them without re-running a full backtest.
_MDA_CACHE_PATH = Path(__file__).parent / "experiments" / "mda_cache.json"
_latest_mda: dict[str, float] = {}


def get_latest_mda() -> dict[str, float]:
    """Return cached MDA weights. Reads JSON file if module-level cache is empty."""
    global _latest_mda
    if _latest_mda:
        return dict(_latest_mda)
    if _MDA_CACHE_PATH.exists():
        try:
            data = json.loads(_MDA_CACHE_PATH.read_text(encoding="utf-8"))
            _latest_mda = {k: float(v) for k, v in data.items()}
            return dict(_latest_mda)
        except Exception:
            pass
    return {}


def _save_mda_cache(mda: dict[str, float]) -> None:
    """Persist MDA weights to JSON and module-level cache."""
    global _latest_mda
    _latest_mda = dict(mda)
    try:
        _MDA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _MDA_CACHE_PATH.write_text(json.dumps(mda, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save MDA cache: %s", e)


@dataclass
class WindowResult:
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    sharpe_ratio: float
    total_return_pct: float
    alpha_pct: float
    max_drawdown_pct: float
    hit_rate: float
    num_trades: int
    n_candidates: int = 0
    n_train_samples: int = 0
    n_features: int = 0
    feature_importance_mdi: dict[str, float] = field(default_factory=dict)
    feature_importance_mda: dict[str, float] = field(default_factory=dict)
    predictions: list[dict] = field(default_factory=list)


@dataclass
class BacktestResult:
    windows: list[WindowResult] = field(default_factory=list)
    aggregate_sharpe: float = 0.0
    aggregate_return_pct: float = 0.0
    aggregate_alpha_pct: float = 0.0
    aggregate_max_drawdown_pct: float = 0.0
    aggregate_hit_rate: float = 0.0
    total_trades: int = 0
    feature_importance_mdi: dict[str, float] = field(default_factory=dict)
    feature_importance_mda: dict[str, float] = field(default_factory=dict)
    nav_history: list[dict] = field(default_factory=list)
    strategy_params: dict = field(default_factory=dict)
    all_trades: list[dict] = field(default_factory=list)


class BacktestEngine:
    """
    Main walk-forward backtesting orchestrator.
    Trains GradientBoosting on Triple Barrier labels with:
    - Sample weights (average uniqueness for overlapping labels)
    - Fractional differentiation on non-stationary features
    - Meta-labeling probability for position sizing
    - MDI + MDA feature importance
    """

    def __init__(
        self,
        bq_client,
        project: str,
        dataset: str,
        # Walk-forward params
        start_date: str = "2023-01-01",
        end_date: str = "2025-12-31",
        train_window_months: int = 12,
        test_window_months: int = 3,
        embargo_days: int = 5,
        # Triple Barrier params
        holding_days: int = 90,
        tp_pct: float = 10.0,
        sl_pct: float = 10.0,
        # Mean Reversion params
        mr_holding_days: int = 15,
        # Feature params
        frac_diff_d: float = 0.4,
        # Strategy selection
        strategy: str = "triple_barrier",
        # Portfolio params
        starting_capital: float = 100_000.0,
        max_positions: int = 20,
        transaction_cost_pct: float = 0.1,
        target_vol: float = 0.15,
        top_n_candidates: int = 50,
        commission_model: str = "flat_pct",
        commission_per_share: float = 0.005,
        # ML params
        n_estimators: int = 200,
        max_depth: int = 4,
        min_samples_leaf: int = 20,
        learning_rate: float = 0.1,
        # Progress callback
        progress_callback=None,
    ):
        # Defensive unwrap: accept BigQueryClient wrapper or raw bigquery.Client
        if hasattr(bq_client, 'client'):
            bq_client = bq_client.client

        # Initialize cache
        cache.init_cache(bq_client, project, dataset)

        self.data_provider = HistoricalDataProvider()
        self.candidate_selector = CandidateSelector()

        self.scheduler = WalkForwardScheduler(
            start_date=start_date,
            end_date=end_date,
            train_window_months=train_window_months,
            test_window_months=test_window_months,
            embargo_days=embargo_days,
        )

        self.holding_days = holding_days
        self.mr_holding_days = mr_holding_days
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.frac_diff_d = frac_diff_d
        self.top_n_candidates = top_n_candidates
        self.strategy = strategy if strategy in STRATEGY_REGISTRY else "triple_barrier"
        self.model_trained_at: str = ""  # ISO timestamp of last model training

        # Store BQ refs for auto-ingest check
        self._bq_client = bq_client
        self._project = project
        self._dataset = dataset

        self.ml_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "learning_rate": learning_rate,
        }

        self.trader = BacktestTrader(
            starting_capital=starting_capital,
            max_positions=max_positions,
            transaction_cost_pct=transaction_cost_pct,
            target_vol=target_vol,
            commission_model=commission_model,
            commission_per_share=commission_per_share,
        )

        self.progress_callback = progress_callback
        self.stop_check: Callable[[], bool] | None = None
        self._backtest_start_time: float = 0.0
        self._total_windows: int = 0
        self._current_window_id: int = 0
        self._strategy_params = {
            "start_date": start_date, "end_date": end_date,
            "train_window_months": train_window_months,
            "test_window_months": test_window_months,
            "embargo_days": embargo_days, "holding_days": holding_days,
            "mr_holding_days": mr_holding_days,
            "tp_pct": tp_pct, "sl_pct": sl_pct, "frac_diff_d": frac_diff_d,
            "starting_capital": starting_capital, "max_positions": max_positions,
            "top_n_candidates": top_n_candidates, "strategy": self.strategy,
            "target_annual_vol": 0,  # Volatility targeting: 0 = disabled (set >0 to enable, e.g. 0.10 for 10%)
            # Trailing stop: activate after +trigger% above entry, trail at -distance% below HWM
            "trailing_stop_enabled": False,
            "trailing_trigger_pct": 5.0,
            "trailing_distance_pct": 3.0,
            **self.ml_params,
        }

    def get_model_trained_at(self) -> str:
        """Return ISO timestamp of last model training. Empty string if never trained."""
        return self.model_trained_at

    def get_cached_features(self) -> dict | None:
        """Return cached per-window feature data, or None if not available."""
        return getattr(self, "_feature_cache_data", None)

    def set_cached_features(self, cache_data: dict):
        """Pre-load feature cache so _build_training_data is skipped."""
        self._feature_cache_data = cache_data

    def clear_feature_cache(self):
        """Clear any pre-loaded feature cache."""
        self._feature_cache_data = None

    def run_backtest(
        self,
        universe_tickers: list[str] | None = None,
        skip_cache_clear: bool = False,
        best_known_sharpe: float | None = None,
    ) -> BacktestResult:
        """
        Run full walk-forward backtest. Main entry point.
        Auto-checks BQ data availability and triggers ingestion if empty.
        Returns BacktestResult with per-window and aggregate metrics.

        Args:
            skip_cache_clear: If True, skip cache.clear_cache() at end.
                Used by QuantStrategyOptimizer to keep warm cache across iterations.
            best_known_sharpe: If set, enables early stopping. After window 10,
                if interim Sharpe < 85% of this value, the backtest aborts early.
        """
        if universe_tickers is None:
            universe_tickers = self.candidate_selector.get_universe_tickers()

        # Auto-ingest check: if historical_prices is empty, trigger ingestion
        self._auto_ingest_if_needed(universe_tickers)

        # Reset trader for a clean independent run (critical for optimizer —
        # each trial must be independent per Bailey & López de Prado, 2014)
        self.trader.full_reset()

        windows = self.scheduler.generate_windows()
        self._total_windows = len(windows)
        self._backtest_start_time = time.time()
        logger.info(f"Walk-forward: {len(windows)} windows, {len(universe_tickers)} tickers, strategy={self.strategy}")

        # Bulk-preload all price and fundamental data (2 BQ queries instead of ~50,000)
        self._report_progress("preloading", f"Loading data for {len(universe_tickers)} tickers")
        global_start = (self.scheduler.start_date - timedelta(days=756)).isoformat()
        global_end = (self.scheduler.end_date + timedelta(days=int(self.holding_days * 1.5))).isoformat()
        cache.preload_prices(universe_tickers + ["SPY"], global_start, global_end)
        cache.preload_fundamentals(universe_tickers)

        result = BacktestResult(strategy_params=self._strategy_params)
        all_mdi = {}
        all_mda = {}

        for window in windows:
            # Check stop signal between windows
            if self.stop_check and self.stop_check():
                logger.info("BacktestEngine: stopped at window %d/%d", window.window_id, len(windows))
                break

            self._report_progress(
                "screening",
                f"Window {window.window_id}/{len(windows)} — screening candidates",
                window=window.window_id,
            )
            try:
                wr = self._run_window(window, universe_tickers)
                result.windows.append(wr)
                result.total_trades += wr.num_trades

                # Accumulate feature importance
                for feat, imp in wr.feature_importance_mdi.items():
                    all_mdi[feat] = all_mdi.get(feat, 0) + imp
                for feat, imp in wr.feature_importance_mda.items():
                    all_mda[feat] = all_mda.get(feat, 0) + imp

                # Early stopping: after window 10, check interim Sharpe
                if (best_known_sharpe is not None
                        and best_known_sharpe > 0
                        and len(result.windows) == 10):
                    interim_returns = self.trader.get_returns_series()
                    if interim_returns:
                        interim_sharpe = self._sharpe(np.array(interim_returns))
                        threshold = best_known_sharpe * 0.85
                        if interim_sharpe < threshold:
                            logger.info(
                                "Early stopping at window %d/%d: interim Sharpe=%.4f "
                                "< 85%% of best known %.4f (threshold=%.4f)",
                                window.window_id, len(windows),
                                interim_sharpe, best_known_sharpe, threshold,
                            )
                            self._report_progress(
                                "trading",
                                f"Early stop: interim Sharpe {interim_sharpe:.4f} < threshold {threshold:.4f}",
                                window=window.window_id,
                            )
                            break

            except Exception as e:
                logger.error(f"Window {window.window_id} failed: {e}")

        # Aggregate metrics from trader
        returns = self.trader.get_returns_series()
        if returns:
            returns_arr = np.array(returns)
            # Phase 1.5: Dynamic risk-free rate (FRED T-bill data)
            from backend.backtest.analytics import get_risk_free_rate
            dynamic_rf_rate = get_risk_free_rate(self.start_date, self.end_date)
            result.aggregate_sharpe = self._sharpe(returns_arr, risk_free_rate=dynamic_rf_rate)
            result.aggregate_return_pct = float((np.prod(1 + returns_arr) - 1) * 100)
            result.aggregate_max_drawdown_pct = self._max_drawdown(returns_arr)

        # Aggregate feature importance (average across windows)
        n_windows = len(result.windows) or 1
        result.feature_importance_mdi = {k: v / n_windows for k, v in all_mdi.items()}
        result.feature_importance_mda = {k: v / n_windows for k, v in all_mda.items()}

        # Persist MDA cache for live quant_model tool
        if result.feature_importance_mda:
            _save_mda_cache(result.feature_importance_mda)

        # Aggregate hit rate
        all_preds = [p for w in result.windows for p in w.predictions]
        if all_preds:
            correct = sum(1 for p in all_preds if p.get("correct", False))
            result.aggregate_hit_rate = correct / len(all_preds)

        # NAV history from snapshots
        result.nav_history = [
            {"date": s.date, "nav": s.nav, "cash": s.cash}
            for s in self.trader.snapshots
        ]

        # Extract individual trades (capped at 500 for JSON size)
        result.all_trades = [
            {
                "ticker": t.ticker, "action": t.action, "quantity": round(t.quantity, 4),
                "price": round(t.price, 2), "date": t.date, "label": t.label,
                "probability": round(t.probability, 4), "commission": round(t.commission, 2),
            }
            for t in self.trader.trades[:500]
        ]
        logger.info(f"Backtest complete: {len(self.trader.trades)} actual trades recorded, {result.total_trades} signals processed")

        self._report_progress(
            "finalizing",
            f"Aggregating results across {len(result.windows)} windows...",
            window=self._total_windows,
        )
        if not skip_cache_clear:
            cache.clear_cache()
        return result

    def _run_window(self, window: WalkForwardWindow, universe_tickers: list[str]) -> WindowResult:
        """Process a single walk-forward window."""
        wid = window.window_id
        self._current_window_id = wid
        train_end_str = window.train_end.isoformat()
        test_start_str = window.test_start.isoformat()
        test_end_str = window.test_end.isoformat()

        # 1. Screen candidates at train_end
        scoring_weights = {
            k: self._strategy_params[k]
            for k in ("momentum_weight", "rsi_weight", "volatility_weight", "sma_weight")
            if k in self._strategy_params
        }
        candidates = self.candidate_selector.screen_at_date(
            train_end_str, universe_tickers, top_n=self.top_n_candidates,
            scoring_weights=scoring_weights if scoring_weights else None,
        )
        candidate_tickers = [c["ticker"] for c in candidates]

        self._report_progress(
            "screening",
            f"{len(candidate_tickers)} candidates found",
            window=wid, candidates_found=len(candidate_tickers),
        )

        if len(candidate_tickers) < 10:
            logger.warning(f"Window {wid}: only {len(candidate_tickers)} candidates")
            return WindowResult(
                window_id=wid,
                train_start=window.train_start.isoformat(),
                train_end=train_end_str,
                test_start=test_start_str,
                test_end=test_end_str,
                sharpe_ratio=0, total_return_pct=0, alpha_pct=0,
                max_drawdown_pct=0, hit_rate=0, num_trades=0,
            )

        # 2. Build training data (or reuse cached features for ML-only experiments)
        cached = getattr(self, "_feature_cache_data", None)
        if cached and wid in cached:
            train_features, train_labels, sample_weights = cached[wid]
            self._report_progress("building_features", "Using cached features (ML-only change)", window=wid)
            logger.debug("Feature cache HIT for window %d", wid)
        else:
            self._report_progress("building_features", "Building training data", window=wid)
            train_features, train_labels, sample_weights = self._build_training_data(
                candidate_tickers, window.train_start.isoformat(), train_end_str,
            )
            # Store into cache dict if caching is active
            if cached is not None:
                cached[wid] = (train_features, train_labels, sample_weights)
                logger.debug("Feature cache MISS for window %d -- cached", wid)

        if len(train_features) < 20:
            logger.warning(f"Window {wid}: insufficient training data ({len(train_features)} samples)")
            return self._empty_window_result(window)

        # 3. Train ML model
        self._report_progress(
            "training",
            f"Training GradientBoosting on {len(train_features)} samples",
            window=wid, samples_built=len(train_features),
        )
        model, feature_names = self._train_model(train_features, train_labels, sample_weights)

        # 4. MDI feature importance
        # HistGradientBoostingClassifier doesn't expose feature_importances_ (MDI).
        # Per López de Prado AFML Ch. 8, MDA (permutation importance) is more
        # reliable anyway — MDI biases toward high-cardinality features.
        # We approximate MDI via a quick permutation importance on training data.
        if hasattr(model, "feature_importances_"):
            mdi = dict(zip(feature_names, model.feature_importances_))
        else:
            # Use training-set permutation importance as MDI proxy (fast, 2 repeats)
            try:
                _mdi_result = permutation_importance(
                    model, train_features, train_labels,
                    n_repeats=2, random_state=42, n_jobs=1,
                )
                mdi = dict(zip(feature_names, _mdi_result["importances_mean"]))
            except Exception:
                mdi = {f: 0.0 for f in feature_names}

        # 5. MDA feature importance (permutation importance)
        self._report_progress("computing_mda", "Permutation importance", window=wid)
        mda = self._compute_mda(model, train_features, train_labels, feature_names)

        # 5b. Meta-labeling second stage (AFML Ch. 3.6)
        # If strategy is "meta_label", train a secondary model that predicts
        # whether the primary model's predictions are correct. The secondary
        # model's probability output is used for position sizing (fractional Kelly).
        meta_model = None
        if self.strategy == "meta_label" and len(train_features) >= 50:
            self._report_progress("training", "Training meta-label model (AFML Ch. 3.6)", window=wid)
            meta_model = self._train_meta_label_model(
                model, train_features, train_labels, feature_names, sample_weights,
            )

        # 6. Predict on test period candidates
        self._report_progress("predicting", f"Scoring test candidates at {test_start_str}", window=wid)
        test_candidates = self.candidate_selector.screen_at_date(
            test_start_str, universe_tickers, top_n=self.top_n_candidates,
            scoring_weights=scoring_weights if scoring_weights else None,
        )
        test_tickers = [c["ticker"] for c in test_candidates]

        signals, predictions = self._predict_and_trade(
            model, feature_names, test_tickers, test_start_str, test_end_str,
            meta_model=meta_model,
        )

        # 7. Execute trades in the test window
        # Track actual trades executed (not just signal count)
        trades_before = len(self.trader.trades)

        self._report_progress(
            "trading",
            f"Processing {len(signals) if signals else 0} signals",
            window=wid,
        )
        if signals:
            # Get prices at test_start for trading
            prices = {}
            for ticker in test_tickers:
                p = cache.cached_prices(ticker, test_start_str, test_start_str)
                if not p.empty:
                    prices[ticker] = float(p["close"].iloc[-1])

            executed = self.trader.execute_trades(signals, test_start_str, prices)
            logger.info(f"Window {wid}: {len(executed)} trades from {len(signals)} signals (BUY={sum(1 for s in signals if s['label']==1)}, SELL={sum(1 for s in signals if s['label']==-1)}, HOLD={sum(1 for s in signals if s['label']==0)})")

        # 8. Daily mark-to-market through the test window
        # Per Lo (2002): Sharpe annualization requires matching return frequency.
        # We need daily NAV snapshots so √252 annualization is correct and DSR
        # has sufficient T (Bailey & López de Prado, 2014).
        active_tickers = list(set(list(self.trader.positions.keys()) + test_tickers))
        test_days = pd.bdate_range(test_start_str, test_end_str)

        # Trailing stop state: track high-water mark and activation per position
        trailing_enabled = bool(self._strategy_params.get("trailing_stop_enabled", False))
        trailing_trigger_pct = self._strategy_params.get("trailing_trigger_pct", 5.0) / 100.0
        trailing_distance_pct = self._strategy_params.get("trailing_distance_pct", 3.0) / 100.0
        # hwm[ticker] = highest price since entry; activated[ticker] = True when trigger hit
        hwm: dict[str, float] = {}
        trailing_activated: dict[str, bool] = {}

        for day in test_days:
            day_str = day.strftime("%Y-%m-%d")
            day_prices = {}
            for ticker in active_tickers:
                p = cache.cached_prices(ticker, day_str, day_str)
                if not p.empty:
                    day_prices[ticker] = float(p["close"].iloc[-1])

            # Trailing stop check (before MTM so exits are reflected in today's NAV)
            if trailing_enabled and day_prices and self.trader.positions:
                tickers_to_close = []
                for ticker, pos in self.trader.positions.items():
                    price = day_prices.get(ticker)
                    if price is None:
                        continue

                    # Update high-water mark
                    current_hwm = hwm.get(ticker, pos.avg_entry_price)
                    if price > current_hwm:
                        current_hwm = price
                    hwm[ticker] = current_hwm

                    # Check activation: price moved +trigger% above entry
                    if not trailing_activated.get(ticker, False):
                        if price >= pos.avg_entry_price * (1 + trailing_trigger_pct):
                            trailing_activated[ticker] = True
                        continue  # Not yet activated, skip stop check

                    # Trailing stop: close if price drops -distance% below HWM
                    trailing_stop_price = current_hwm * (1 - trailing_distance_pct)
                    if price <= trailing_stop_price:
                        tickers_to_close.append(ticker)

                # Execute trailing stop exits
                for ticker in tickers_to_close:
                    pos = self.trader.positions.get(ticker)
                    if pos is None:
                        continue
                    price = day_prices[ticker]
                    proceeds = pos.quantity * price
                    cost = self.trader._compute_commission(pos.quantity, price)
                    self.trader.cash += proceeds - cost
                    self.trader.total_commission += cost
                    from backend.backtest.backtest_trader import Trade
                    self.trader.trades.append(Trade(
                        ticker=ticker, action="SELL", quantity=pos.quantity,
                        price=price, date=day_str, label=-1,
                        probability=0, commission=cost,
                    ))
                    del self.trader.positions[ticker]
                    hwm.pop(ticker, None)
                    trailing_activated.pop(ticker, None)

            if day_prices and self.trader.positions:
                self.trader.mark_to_market(day_str, day_prices)

        # Final mark + close at test end
        end_prices = {}
        for ticker in active_tickers:
            p = cache.cached_prices(ticker, test_end_str, test_end_str)
            if not p.empty:
                end_prices[ticker] = float(p["close"].iloc[-1])

        self.trader.close_all_positions(test_end_str, end_prices)

        # Actual trades = entries + exits in this window
        actual_trades = len(self.trader.trades) - trades_before

        # 9. Compute window metrics
        window_returns = self.trader.get_returns_series()

        wr = WindowResult(
            window_id=window.window_id,
            train_start=window.train_start.isoformat(),
            train_end=train_end_str,
            test_start=test_start_str,
            test_end=test_end_str,
            sharpe_ratio=self._sharpe(np.array(window_returns)) if window_returns else 0,
            total_return_pct=0,  # Computed at aggregate level
            alpha_pct=0,  # Computed at aggregate level
            max_drawdown_pct=self._max_drawdown(np.array(window_returns)) if window_returns else 0,
            hit_rate=sum(1 for p in predictions if p.get("correct")) / len(predictions) if predictions else 0,
            num_trades=actual_trades,
            n_candidates=len(candidate_tickers),
            n_train_samples=len(train_features),
            n_features=len(feature_names),
            feature_importance_mdi=mdi,
            feature_importance_mda=mda,
            predictions=predictions,
        )

        self.trader.reset()
        return wr

    # ── Training data construction ───────────────────────────────

    def _build_training_data(
        self,
        tickers: list[str],
        train_start: str,
        train_end: str,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Build feature matrix + Triple Barrier labels + sample weights
        for the training window.
        """
        features_list = []
        labels_list = []
        entry_dates = []
        exit_dates = []

        # Sample at biweekly intervals within the training window
        # (~26 samples/ticker/window vs ~12 with monthly — doubles training set)
        current = pd.Timestamp(train_start) + pd.DateOffset(weeks=2)
        end = pd.Timestamp(train_end)

        sample_dates = []
        while current <= end:
            sample_dates.append(current.strftime("%Y-%m-%d"))
            current += pd.DateOffset(weeks=2)

        total_iterations = len(sample_dates) * len(tickers)
        iteration_count = 0

        for sample_date in sample_dates:
            for ticker in tickers:
                iteration_count += 1
                try:
                    fv = self.data_provider.build_feature_vector(ticker, sample_date)
                    if not fv or fv.get("price_at_analysis") is None:
                        continue

                    label = self._compute_label(ticker, sample_date)
                    if label is None:
                        continue

                    features_list.append(fv)
                    labels_list.append(label)

                    # Track entry/exit dates for sample weight computation
                    entry_dates.append(pd.Timestamp(sample_date))
                    exit_dates.append(
                        pd.Timestamp(sample_date) + timedelta(days=self.holding_days)
                    )

                    # Throttled progress: emit every 200 samples
                    if len(features_list) % 200 == 0:
                        self._report_progress(
                            "building_features",
                            f"{len(features_list)} samples built ({iteration_count}/{total_iterations} iterations)",
                            window=self._current_window_id,
                            samples_built=len(features_list),
                            samples_total=total_iterations,
                        )
                except Exception:
                    continue

        if not features_list:
            return pd.DataFrame(), np.array([]), np.array([])

        # Convert to DataFrame and extract numeric features
        df = pd.DataFrame(features_list)
        feature_cols = [c for c in _NUMERIC_FEATURES if c in df.columns]
        X = df[feature_cols].copy()

        # Apply fractional differentiation to non-stationary features
        for col in feature_cols:
            if col in _NON_STATIONARY and col in X.columns:
                series = X[col].dropna()
                if len(series) > 10:
                    diffed = HistoricalDataProvider.fractional_diff(series, d=self.frac_diff_d)
                    X.loc[diffed.index, col] = diffed

        # Fill NaN with median (robust to outliers)
        X = X.fillna(X.median())
        X = X.fillna(0)  # Remaining NaN → 0

        labels = np.array(labels_list)

        # Compute sample weights via average uniqueness
        weights = self._compute_sample_weights(entry_dates, exit_dates)

        return X, labels, weights

    def _compute_triple_barrier_label(self, ticker: str, entry_date: str) -> int | None:
        """
        Triple Barrier Method (López de Prado Ch. 3):
        +1 if price hits TP barrier first, -1 if SL barrier first, 0 if time expires.

        Supports two barrier modes:
        1. Fixed percentage (default): tp_pct/sl_pct as configured
        2. Volatility-adjusted (AFML recommended): barriers = daily_vol × multiplier
           Activated when strategy_params contains 'vol_barrier_multiplier' > 0.
           This adapts barriers to each stock's risk profile — a 40% vol stock
           gets wider barriers than a 15% vol stock.

        Transaction cost adjustment (Almgren & Chriss 2000): barriers are shifted
        inward by the estimated round-trip cost to avoid labeling trades as winners
        when actual profit after costs would be negative.
        """
        end_date = (pd.Timestamp(entry_date) + timedelta(days=int(self.holding_days * 1.5))).strftime("%Y-%m-%d")
        prices = cache.cached_prices(ticker, entry_date, end_date)

        if prices.empty:
            return None

        entry_price = float(prices["close"].iloc[0])

        # Determine barrier width
        vol_multiplier = self._strategy_params.get("vol_barrier_multiplier", 0)

        if vol_multiplier and vol_multiplier > 0:
            # Volatility-adjusted barriers (AFML Ch. 3 recommended approach)
            # Compute daily vol from recent prices (20-day lookback)
            lookback_start = (pd.Timestamp(entry_date) - timedelta(days=40)).strftime("%Y-%m-%d")
            lookback_prices = cache.cached_prices(ticker, lookback_start, entry_date)
            if lookback_prices.empty or len(lookback_prices) < 10:
                # Fallback to fixed barriers if insufficient lookback data
                tp_pct_effective = self.tp_pct
                sl_pct_effective = self.sl_pct
            else:
                daily_returns = lookback_prices["close"].pct_change().dropna()
                daily_vol = float(daily_returns.std())
                # Barrier = daily_vol × multiplier (as percentage)
                # Typical multiplier range: 1.0-5.0
                tp_pct_effective = daily_vol * vol_multiplier * 100
                sl_pct_effective = daily_vol * vol_multiplier * 100
        else:
            # Fixed percentage barriers (original behavior)
            tp_pct_effective = self.tp_pct
            sl_pct_effective = self.sl_pct

        # Adjust barriers for round-trip transaction costs
        round_trip_cost_pct = 2 * self.trader.transaction_cost_pct / 100
        tp_price = entry_price * (1 + tp_pct_effective / 100 + round_trip_cost_pct)
        sl_price = entry_price * (1 - sl_pct_effective / 100 + round_trip_cost_pct)

        # Walk forward through prices
        trading_days = 0
        for idx in range(1, len(prices)):
            trading_days += 1
            price = float(prices["close"].iloc[idx])

            if price >= tp_price:
                return 1  # Hit take-profit (net of costs)
            if price <= sl_price:
                return -1  # Hit stop-loss
            if trading_days >= self.holding_days:
                return 0  # Time expired

        return 0  # Not enough data, treat as hold

    def _compute_sample_weights(
        self, entry_dates: list, exit_dates: list
    ) -> np.ndarray:
        """
        Sample weights via average uniqueness (López de Prado Ch. 4).
        Prevents GradientBoosting from overfitting to regime-dominant
        overlapping labels.
        """
        n = len(entry_dates)
        if n == 0:
            return np.array([])

        weights = np.ones(n)
        for i in range(n):
            overlap_count = 0
            for j in range(n):
                if i == j:
                    continue
                # Check if labels overlap in time
                if entry_dates[j] < exit_dates[i] and exit_dates[j] > entry_dates[i]:
                    overlap_count += 1

            # Average uniqueness = 1 / (1 + overlap_count)
            weights[i] = 1.0 / (1.0 + overlap_count)

        # Normalize so weights sum to n
        if weights.sum() > 0:
            weights = weights * n / weights.sum()

        return weights

    # ── Model training ───────────────────────────────────────────

    def _train_model(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weights: np.ndarray,
    ) -> tuple[HistGradientBoostingClassifier, list[str]]:
        """Train HistGradientBoosting classifier with sample weights.

        Uses histogram-based gradient boosting (inspired by LightGBM, Ke et al. 2017).
        Key advantages over traditional GradientBoostingClassifier:
        - 5-20x faster training via 255-bin histogram splitting
        - Built-in OpenMP multi-core parallelism
        - Native missing value handling (eliminates fillna hacks)
        - Implicit regularization from binning — reduces overfitting on noisy financial data
        """
        model = HistGradientBoostingClassifier(
            max_iter=self.ml_params["n_estimators"],  # n_estimators → max_iter
            max_depth=self.ml_params["max_depth"],
            min_samples_leaf=self.ml_params["min_samples_leaf"],
            learning_rate=self.ml_params["learning_rate"],
            random_state=42,
            early_stopping=False,  # Disable to match previous deterministic behavior
            max_bins=255,  # Maximum precision (default)
        )
        model.fit(X, y, sample_weight=sample_weights)
        from datetime import datetime as _dt, timezone as _tz
        self.model_trained_at = _dt.now(_tz.utc).isoformat()
        return model, list(X.columns)

    def _compute_mda(
        self,
        model: HistGradientBoostingClassifier,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: list[str],
        n_repeats: int = 5,
    ) -> dict[str, float]:
        """
        MDA (permutation importance) — primary feature ranking.
        López de Prado Ch. 8: MDI biases toward high-cardinality features;
        MDA is the authoritative ranking.
        """
        try:
            result = permutation_importance(
                model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=1,
            )
            return dict(zip(feature_names, result["importances_mean"]))
        except Exception as e:
            logger.warning(f"MDA computation failed: {e}")
            return {}

    # ── Prediction + trade signal generation ─────────────────────

    def _predict_and_trade(
        self,
        model: HistGradientBoostingClassifier,
        feature_names: list[str],
        test_tickers: list[str],
        test_start: str,
        test_end: str,
        meta_model: HistGradientBoostingClassifier | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """
        Generate predictions on test period candidates.
        Returns (signals for trader, predictions for analytics).

        If meta_model is provided (meta-labeling, AFML Ch. 3.6):
        - Primary model provides direction (label)
        - Meta-model provides probability that primary is correct
        - This probability is used for position sizing (fractional Kelly)
        - Predictions with meta_probability < 0.5 are filtered to HOLD (label=0)
        """
        signals = []
        predictions = []

        for ticker in test_tickers:
            try:
                fv = self.data_provider.build_feature_vector(ticker, test_start)
                if not fv or fv.get("price_at_analysis") is None:
                    continue

                # Build feature row
                row = {f: fv.get(f, 0) for f in feature_names}
                X_test = pd.DataFrame([row])[feature_names].fillna(0)

                pred_label = int(model.predict(X_test)[0])
                pred_proba = model.predict_proba(X_test)[0]

                # Get probability for the predicted class
                classes = list(model.classes_)
                if pred_label in classes:
                    probability = float(pred_proba[classes.index(pred_label)])
                else:
                    probability = 0.5

                # Meta-labeling: override probability with meta-model's confidence
                # that the primary prediction is correct (AFML Ch. 3.6)
                if meta_model is not None:
                    meta_features = self._build_meta_features(
                        X_test, pred_label, probability, feature_names,
                    )
                    meta_proba = meta_model.predict_proba(meta_features)[0]
                    # Meta-model predicts P(primary is correct)
                    # Class 1 = "primary was correct"
                    meta_classes = list(meta_model.classes_)
                    if 1 in meta_classes:
                        meta_probability = float(meta_proba[meta_classes.index(1)])
                    else:
                        meta_probability = 0.5

                    # Filter: if meta-model says primary is likely wrong, don't trade
                    if meta_probability < 0.5:
                        pred_label = 0  # Override to HOLD
                    probability = meta_probability  # Use meta confidence for sizing

                volatility = fv.get("annualized_volatility", 0.3) or 0.3
                amihud = fv.get("amihud_illiquidity", 0.0) or 0.0

                # Volatility targeting: scale position by target_annual_vol / realized_vol
                # Uses rolling 20-day realized vol at entry date for each stock
                vol_target_scale = self._compute_vol_target_scale(ticker, test_start, volatility)

                signals.append({
                    "ticker": ticker,
                    "label": pred_label,
                    "probability": probability,
                    "volatility": volatility,
                    "amihud_illiquidity": amihud,
                    "vol_target_scale": vol_target_scale,
                })

                # Check actual outcome for hit rate
                actual_label = self._compute_label(ticker, test_start)
                predictions.append({
                    "ticker": ticker,
                    "date": test_start,
                    "predicted_label": pred_label,
                    "probability": probability,
                    "actual_label": actual_label,
                    "correct": pred_label == actual_label if actual_label is not None else None,
                })

            except Exception as e:
                logger.debug(f"Prediction failed for {ticker}: {e}")

        return signals, predictions

    # ── Meta-Labeling (AFML Ch. 3.6) ──────────────────────────────

    def _train_meta_label_model(
        self,
        primary_model: HistGradientBoostingClassifier,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        feature_names: list[str],
        sample_weights: np.ndarray,
    ) -> HistGradientBoostingClassifier | None:
        """
        Train a secondary (meta-label) model per López de Prado AFML Ch. 3.6.

        The meta-label model learns WHEN the primary model is correct.
        Input: original features + primary model's prediction + primary confidence.
        Output: binary 0/1 (was primary correct?).

        This enables intelligent bet sizing — the meta-model's probability
        output replaces the primary model's raw probability for position sizing,
        implementing a form of fractional Kelly criterion.

        Uses 3-fold cross-validation on the training set to avoid overfitting
        the meta-labels to the primary model's in-sample predictions.
        """
        from sklearn.model_selection import cross_val_predict

        try:
            # Step 1: Generate primary model predictions via cross-validation
            # This avoids leakage — we can't use the model's own training predictions
            primary_preds = cross_val_predict(
                HistGradientBoostingClassifier(
                    max_iter=self.ml_params["n_estimators"],
                    max_depth=max(2, self.ml_params["max_depth"] - 1),  # Slightly shallower
                    min_samples_leaf=self.ml_params["min_samples_leaf"],
                    learning_rate=self.ml_params["learning_rate"],
                    random_state=42,
                    early_stopping=False,
                ),
                X_train, y_train, cv=3, method="predict",
            )

            # Step 2: Create meta-labels — was the primary prediction correct?
            meta_labels = (primary_preds == y_train).astype(int)

            # Need enough positive and negative meta-labels for training
            if meta_labels.sum() < 10 or (len(meta_labels) - meta_labels.sum()) < 10:
                logger.warning("Meta-labeling: insufficient label diversity, skipping")
                return None

            # Step 3: Build meta-features (original features + primary signal info)
            # Get primary model's predictions and probabilities on training data
            primary_proba = primary_model.predict_proba(X_train)
            primary_labels = primary_model.predict(X_train)

            meta_X = self._build_meta_features_batch(
                X_train, primary_labels, primary_proba, feature_names, primary_model,
            )

            # Step 4: Train meta-model (binary classifier: correct vs incorrect)
            meta_model = HistGradientBoostingClassifier(
                max_iter=max(50, self.ml_params["n_estimators"] // 2),
                max_depth=max(2, self.ml_params["max_depth"] - 1),
                min_samples_leaf=self.ml_params["min_samples_leaf"],
                learning_rate=self.ml_params["learning_rate"],
                random_state=42,
                early_stopping=False,
            )
            meta_model.fit(meta_X, meta_labels, sample_weight=sample_weights)

            accuracy = (meta_model.predict(meta_X) == meta_labels).mean()
            logger.info(
                "Meta-label model trained: %d samples, in-sample accuracy=%.3f, "
                "primary correct rate=%.3f",
                len(meta_labels), accuracy, meta_labels.mean(),
            )
            return meta_model

        except Exception as e:
            logger.warning("Meta-label training failed: %s", e)
            return None

    def _build_meta_features(
        self,
        X_test: pd.DataFrame,
        pred_label: int,
        probability: float,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """Build meta-features for a single test observation."""
        row = X_test.copy()
        row["primary_label"] = pred_label
        row["primary_confidence"] = probability
        row["primary_is_buy"] = int(pred_label == 1)
        row["primary_is_sell"] = int(pred_label == -1)
        return row

    def _build_meta_features_batch(
        self,
        X: pd.DataFrame,
        primary_labels: np.ndarray,
        primary_proba: np.ndarray,
        feature_names: list[str],
        primary_model: HistGradientBoostingClassifier,
    ) -> pd.DataFrame:
        """Build meta-features for the full training set."""
        meta_X = X.copy()
        meta_X["primary_label"] = primary_labels

        # Get max probability across classes for each sample
        meta_X["primary_confidence"] = primary_proba.max(axis=1)

        # Binary indicators for direction
        meta_X["primary_is_buy"] = (primary_labels == 1).astype(int)
        meta_X["primary_is_sell"] = (primary_labels == -1).astype(int)

        return meta_X

    # ── Metric helpers ───────────────────────────────────────────

    @staticmethod
    def _sharpe(returns: np.ndarray, risk_free_rate: float = 0.04) -> float:
        # Lazy import to avoid circular dependency (analytics imports BacktestResult)
        from backend.backtest.analytics import compute_sharpe
        return compute_sharpe(returns, risk_free_rate)

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        cumulative = np.cumprod(1 + returns)
        from backend.backtest.analytics import compute_max_drawdown
        return compute_max_drawdown(cumulative)

    def _empty_window_result(self, window: WalkForwardWindow) -> WindowResult:
        return WindowResult(
            window_id=window.window_id,
            train_start=window.train_start.isoformat(),
            train_end=window.train_end.isoformat(),
            test_start=window.test_start.isoformat(),
            test_end=window.test_end.isoformat(),
            sharpe_ratio=0, total_return_pct=0, alpha_pct=0,
            max_drawdown_pct=0, hit_rate=0, num_trades=0,
        )

    def _report_progress(self, step: str, detail: str = "", **kwargs):
        stats = cache.get_cache_stats()
        data = {
            "window": kwargs.get("window", 0),
            "total_windows": self._total_windows,
            "step": step,
            "step_detail": detail,
            "candidates_found": kwargs.get("candidates_found", 0),
            "samples_built": kwargs.get("samples_built", 0),
            "samples_total": kwargs.get("samples_total", 0),
            "elapsed_seconds": round(time.time() - self._backtest_start_time, 1)
            if self._backtest_start_time
            else 0,
            "cache_hits": stats["hits"],
            "cache_misses": stats["misses"],
        }
        if self.progress_callback:
            self.progress_callback(data)
        logger.info("Backtest: [%s] %s", step, detail)

    # ── Volatility Targeting ────────────────────────────────────

    def _compute_vol_target_scale(self, ticker: str, date: str, fallback_vol: float) -> float:
        """
        Compute volatility targeting scale factor for position sizing.

        Uses rolling 20-day realized volatility of each stock, then scales:
            scale = target_annual_vol / (realized_vol * sqrt(252))

        Capped at 2.0 to prevent extreme leverage. Returns 1.0 if vol targeting
        is disabled (target_annual_vol not set or 0).
        """
        target_annual_vol = self._strategy_params.get("target_annual_vol", 0)
        if not target_annual_vol or target_annual_vol <= 0:
            return 1.0

        # Get rolling 20-day prices for realized vol computation
        lookback_start = (pd.Timestamp(date) - timedelta(days=40)).strftime("%Y-%m-%d")
        prices = cache.cached_prices(ticker, lookback_start, date)

        if prices.empty or len(prices) < 10:
            # Fallback: use the annualized vol from the feature vector
            realized_annual_vol = fallback_vol if fallback_vol > 0 else 0.25
        else:
            daily_returns = prices["close"].pct_change().dropna()
            if len(daily_returns) < 5:
                realized_annual_vol = fallback_vol if fallback_vol > 0 else 0.25
            else:
                daily_vol = float(daily_returns.std())
                realized_annual_vol = daily_vol * np.sqrt(252)

        if realized_annual_vol <= 0:
            return 1.0

        scale = target_annual_vol / realized_annual_vol
        return min(scale, 2.0)  # Cap at 2.0 to prevent extreme leverage

    # ── Strategy Label Dispatcher ────────────────────────────────

    def _compute_label(self, ticker: str, entry_date: str) -> int | None:
        """Dispatch to the active strategy's label method."""
        method_name = STRATEGY_REGISTRY.get(self.strategy, "_compute_triple_barrier_label")
        method = getattr(self, method_name)
        return method(ticker, entry_date)

    # ── Alternative Label Methods ────────────────────────────────

    def _compute_quality_momentum_label(self, ticker: str, entry_date: str) -> int | None:
        """
        Quality Momentum (Asness et al. 2019):
        +1 if 6-month momentum > 0 AND quality_score > median, -1 if both negative, 0 otherwise.
        """
        fv = self.data_provider.build_feature_vector(ticker, entry_date)
        if not fv or fv.get("price_at_analysis") is None:
            return None

        momentum_6m = fv.get("momentum_6m")
        quality_score = fv.get("quality_score", 0) or 0

        if momentum_6m is None:
            return None

        if momentum_6m > 5 and quality_score > 0.3:
            return 1
        if momentum_6m < -5 and quality_score < 0.1:
            return -1
        return 0

    def _compute_mean_reversion_label(self, ticker: str, entry_date: str) -> int | None:
        """
        Mean Reversion (Lo & MacKinlay 1990, Poterba & Summers 1988):
        Two-stage label:
          1. Signal: is the stock oversold or overbought? (SMA distance + RSI)
          2. Validation: does the price actually revert within mr_holding_days?

        Uses mr_holding_days (default 15, range 5-30) for the short reversion
        horizon that the academic literature recommends. Mean reversion works
        at 1-4 week horizons; longer horizons transition to momentum territory
        (Jegadeesh & Titman 1993).

        +1 if oversold AND price reverts up by ≥ half the SMA gap within mr_holding_days
        -1 if overbought AND price reverts down by ≥ half the SMA gap within mr_holding_days
        0 if no signal or reversion doesn't materialize
        """
        fv = self.data_provider.build_feature_vector(ticker, entry_date)
        if not fv or fv.get("price_at_analysis") is None:
            return None

        sma_dist = fv.get("sma_50_distance")
        rsi = fv.get("rsi_14")
        if sma_dist is None or rsi is None:
            return None

        entry_price = fv["price_at_analysis"]

        # Stage 1: Signal detection
        is_oversold = sma_dist < -0.05 and rsi < 35
        is_overbought = sma_dist > 0.10 and rsi > 70

        if not is_oversold and not is_overbought:
            return 0

        # Stage 2: Forward validation — did the price actually revert?
        end_date = (pd.Timestamp(entry_date) + timedelta(days=int(self.mr_holding_days * 2))).strftime("%Y-%m-%d")
        prices = cache.cached_prices(ticker, entry_date, end_date)

        if prices.empty or len(prices) < 3:
            return None

        # Check if price reverts toward SMA within mr_holding_days trading days
        trading_days = 0
        for idx in range(1, len(prices)):
            trading_days += 1
            if trading_days > self.mr_holding_days:
                break
            price = float(prices["close"].iloc[idx])

            if is_oversold:
                # Reversion target: recover at least half the gap to SMA
                # If SMA dist was -8%, target is entry_price × (1 + 0.04)
                reversion_target = entry_price * (1 + abs(sma_dist) / 2)
                if price >= reversion_target:
                    return 1
            elif is_overbought:
                # Reversion target: fall at least half the gap from SMA
                reversion_target = entry_price * (1 - sma_dist / 2)
                if price <= reversion_target:
                    return -1

        return 0  # Signal present but reversion didn't materialize

    def _compute_factor_label(self, ticker: str, entry_date: str) -> int | None:
        """
        Multi-Factor Composite — Fama & French (2015) "A Five-Factor Model",
        augmented with Carhart (1997) momentum and Novy-Marx (2013) profitability.

        Factors and weights:
          1. Value (25%): P/B ratio — Fama-French use book-to-market (HML).
             P/B is the inverse; lower P/B = higher B/M = deeper value.
          2. Momentum (25%): 12-1 month return — Jegadeesh & Titman (1993).
             Uses 12m return minus 1m return to avoid short-term reversal.
          3. Low Volatility (15%): Frazzini & Pedersen (2014) "Betting Against Beta".
          4. Quality/Profitability (25%): Novy-Marx (2013), using the Asness (2019)
             QMJ composite quality_score from build_feature_vector().
          5. Yield (10%): Dividend yield as payout factor proxy.

        Each factor uses a sigmoid-like normalization centered on typical S&P 500
        values rather than hardcoded min/max caps, making it more robust across
        market regimes. The old hardcoded ranges (P/E 5-40, vol 0.10-0.60) broke
        for growth stocks and extreme market conditions.

        +1 if composite > 0.6, -1 if composite < 0.3, 0 otherwise.
        """
        fv = self.data_provider.build_feature_vector(ticker, entry_date)
        if not fv or fv.get("price_at_analysis") is None:
            return None

        # Use P/B (Fama-French HML proxy) instead of P/E for value factor
        pb = fv.get("pb_ratio")
        pe = fv.get("pe_ratio")
        # 12-1 momentum: 12m return minus 1m to avoid short-term reversal
        mom_12m = fv.get("momentum_12m")
        mom_1m = fv.get("momentum_1m")
        vol = fv.get("annualized_volatility")
        quality = fv.get("quality_score")  # Now uses full Asness (2019) QMJ
        div_yield = fv.get("dividend_yield", 0) or 0

        # Need at least momentum and one valuation metric
        if vol is None:
            return None
        if pb is None and pe is None:
            return None
        if mom_12m is None:
            return None

        # 1. Value: lower P/B is better. Sigmoid centered at P/B = 3 (S&P median)
        # P/B < 1.5 → high value score, P/B > 6 → low value score
        if pb is not None and pb > 0:
            value_score = max(0, min(1, 1.0 / (1.0 + np.exp((pb - 3.0) / 1.5))))
        elif pe is not None and pe > 0:
            # Fallback to P/E if P/B unavailable. Centered at P/E = 20
            value_score = max(0, min(1, 1.0 / (1.0 + np.exp((pe - 20.0) / 8.0))))
        else:
            value_score = 0.5  # Neutral if no valuation data

        # 2. Momentum: 12-1 month return (Jegadeesh & Titman 1993)
        mom_12_1 = mom_12m - (mom_1m or 0)
        # Sigmoid centered at 0%: positive momentum → higher score
        mom_score = max(0, min(1, 1.0 / (1.0 + np.exp(-mom_12_1 / 15.0))))

        # 3. Low Volatility: lower is better. Sigmoid centered at 25% (S&P median)
        vol_score = max(0, min(1, 1.0 / (1.0 + np.exp((vol - 0.25) / 0.10))))

        # 4. Quality: use the Asness (2019) QMJ composite directly (already 0-1)
        quality_score = quality if quality is not None else 0.5

        # 5. Yield: dividend yield. Sigmoid centered at 2%
        yield_score = max(0, min(1, 1.0 / (1.0 + np.exp(-(div_yield - 0.02) / 0.015)))) if div_yield > 0 else 0.3

        composite = (
            value_score * 0.25 + mom_score * 0.25 + vol_score * 0.15
            + quality_score * 0.25 + yield_score * 0.10
        )

        if composite > 0.6:
            return 1
        if composite < 0.3:
            return -1
        return 0

    def _compute_blend_label(self, ticker: str, entry_date: str) -> int | None:
        """
        Strategy Blending — weighted vote across base strategies.
        Reference: Dietterich (2000) "Ensemble Methods in Machine Learning"

        Computes labels from triple_barrier, quality_momentum, mean_reversion,
        and factor_model, then takes a weighted average. Weights are tunable
        parameters (tb_weight, qm_weight, mr_weight, fm_weight) that the
        optimizer can search over.

        Final label = sign(weighted_sum) rounded to {-1, 0, +1}.
        """
        # Get blend weights from strategy params (defaults: equal weight)
        tb_w = self._strategy_params.get("tb_weight", 0.4)
        qm_w = self._strategy_params.get("qm_weight", 0.2)
        mr_w = self._strategy_params.get("mr_weight", 0.1)
        fm_w = self._strategy_params.get("fm_weight", 0.3)

        # Compute each strategy's label
        labels = {}
        labels["tb"] = self._compute_triple_barrier_label(ticker, entry_date)
        labels["qm"] = self._compute_quality_momentum_label(ticker, entry_date)
        labels["mr"] = self._compute_mean_reversion_label(ticker, entry_date)
        labels["fm"] = self._compute_factor_label(ticker, entry_date)

        # Weight only strategies that returned a valid label
        weight_map = {"tb": tb_w, "qm": qm_w, "mr": mr_w, "fm": fm_w}
        total_weight = 0.0
        weighted_sum = 0.0

        for key, label in labels.items():
            if label is not None:
                w = weight_map[key]
                weighted_sum += label * w
                total_weight += w

        if total_weight == 0:
            return None

        # Normalize and threshold
        normalized = weighted_sum / total_weight
        if normalized > 0.3:
            return 1
        if normalized < -0.3:
            return -1
        return 0

    # ── Auto-Ingest ──────────────────────────────────────────────

    def _auto_ingest_if_needed(self, universe_tickers: list[str]):
        """Check BQ table row counts; auto-ingest if historical_prices is empty."""
        try:
            from backend.backtest.data_ingestion import DataIngestionService
            from backend.config.settings import get_settings

            settings = get_settings()
            ingestion = DataIngestionService(self._bq_client, settings)
            status = ingestion.get_ingestion_status()

            prices_count = status.get("historical_prices", 0)
            if prices_count == 0:
                logger.warning("Auto-ingest: historical_prices is empty, triggering full ingestion...")
                self._report_progress("Auto-ingesting historical data (first run)...")
                ingestion.run_full_ingestion(universe_tickers[:100])
                logger.info("Auto-ingest complete")
            else:
                logger.info(f"Auto-ingest check: {prices_count} price rows found, skipping ingestion")
        except Exception as e:
            logger.warning(f"Auto-ingest check failed (non-fatal): {e}")
