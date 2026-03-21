"""
Backtest engine — main orchestrator for walk-forward ML backtesting.
Trains GradientBoosting with Triple Barrier labels, sample weights,
fractional differentiation, and meta-labeling position sizing.
Zero LLM cost.
"""

import logging
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance

from backend.backtest.historical_data import HistoricalDataProvider
from backend.backtest.candidate_selector import CandidateSelector
from backend.backtest.walk_forward import WalkForwardScheduler, WalkForwardWindow
from backend.backtest.backtest_trader import BacktestTrader
from backend.backtest import cache

logger = logging.getLogger(__name__)

# Numeric features used for ML training (excludes categorical: ticker, date, sector, industry)
_NUMERIC_FEATURES = [
    "price_at_analysis", "momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m",
    "rsi_14", "annualized_volatility", "sma_50_distance", "sma_200_distance",
    "var_95_6m", "var_99_6m", "expected_shortfall_6m", "prob_positive_6m",
    "anomaly_count", "amihud_illiquidity",
    "pe_ratio", "debt_equity", "roe", "profit_margin", "market_cap",
    "total_revenue", "net_income", "total_debt", "total_equity", "total_assets",
    "fed_funds_rate", "cpi_yoy", "unemployment_rate", "yield_curve_spread",
    "consumer_sentiment", "treasury_10y",
]

# Non-stationary features that need fractional differentiation
_NON_STATIONARY = {"price_at_analysis", "market_cap", "total_revenue", "total_debt", "total_equity"}


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
        # Feature params
        frac_diff_d: float = 0.4,
        # Portfolio params
        starting_capital: float = 100_000.0,
        max_positions: int = 20,
        transaction_cost_pct: float = 0.1,
        target_vol: float = 0.15,
        top_n_candidates: int = 50,
        # ML params
        n_estimators: int = 200,
        max_depth: int = 4,
        min_samples_leaf: int = 20,
        learning_rate: float = 0.1,
        # Progress callback
        progress_callback=None,
    ):
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
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.frac_diff_d = frac_diff_d
        self.top_n_candidates = top_n_candidates

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
        )

        self.progress_callback = progress_callback
        self._strategy_params = {
            "start_date": start_date, "end_date": end_date,
            "train_window_months": train_window_months,
            "test_window_months": test_window_months,
            "embargo_days": embargo_days, "holding_days": holding_days,
            "tp_pct": tp_pct, "sl_pct": sl_pct, "frac_diff_d": frac_diff_d,
            "starting_capital": starting_capital, "max_positions": max_positions,
            "top_n_candidates": top_n_candidates, **self.ml_params,
        }

    def run_backtest(self, universe_tickers: list[str] | None = None) -> BacktestResult:
        """
        Run full walk-forward backtest. Main entry point.
        Returns BacktestResult with per-window and aggregate metrics.
        """
        if universe_tickers is None:
            universe_tickers = self.candidate_selector.get_universe_tickers()

        windows = self.scheduler.generate_windows()
        logger.info(f"Walk-forward: {len(windows)} windows, {len(universe_tickers)} tickers")

        result = BacktestResult(strategy_params=self._strategy_params)
        all_mdi = {}
        all_mda = {}

        for window in windows:
            self._report_progress(f"Window {window.window_id}/{len(windows)}")
            try:
                wr = self._run_window(window, universe_tickers)
                result.windows.append(wr)
                result.total_trades += wr.num_trades

                # Accumulate feature importance
                for feat, imp in wr.feature_importance_mdi.items():
                    all_mdi[feat] = all_mdi.get(feat, 0) + imp
                for feat, imp in wr.feature_importance_mda.items():
                    all_mda[feat] = all_mda.get(feat, 0) + imp

            except Exception as e:
                logger.error(f"Window {window.window_id} failed: {e}")

        # Aggregate metrics from trader
        returns = self.trader.get_returns_series()
        if returns:
            returns_arr = np.array(returns)
            result.aggregate_sharpe = self._sharpe(returns_arr)
            result.aggregate_return_pct = float((np.prod(1 + returns_arr) - 1) * 100)
            result.aggregate_max_drawdown_pct = self._max_drawdown(returns_arr)

        # Aggregate feature importance (average across windows)
        n_windows = len(result.windows) or 1
        result.feature_importance_mdi = {k: v / n_windows for k, v in all_mdi.items()}
        result.feature_importance_mda = {k: v / n_windows for k, v in all_mda.items()}

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

        cache.clear_cache()
        return result

    def _run_window(self, window: WalkForwardWindow, universe_tickers: list[str]) -> WindowResult:
        """Process a single walk-forward window."""
        train_end_str = window.train_end.isoformat()
        test_start_str = window.test_start.isoformat()
        test_end_str = window.test_end.isoformat()

        # 1. Screen candidates at train_end
        candidates = self.candidate_selector.screen_at_date(
            train_end_str, universe_tickers, top_n=self.top_n_candidates,
        )
        candidate_tickers = [c["ticker"] for c in candidates]

        if len(candidate_tickers) < 10:
            logger.warning(f"Window {window.window_id}: only {len(candidate_tickers)} candidates")
            return WindowResult(
                window_id=window.window_id,
                train_start=window.train_start.isoformat(),
                train_end=train_end_str,
                test_start=test_start_str,
                test_end=test_end_str,
                sharpe_ratio=0, total_return_pct=0, alpha_pct=0,
                max_drawdown_pct=0, hit_rate=0, num_trades=0,
            )

        # 2. Build training data
        train_features, train_labels, sample_weights = self._build_training_data(
            candidate_tickers, window.train_start.isoformat(), train_end_str,
        )

        if len(train_features) < 20:
            logger.warning(f"Window {window.window_id}: insufficient training data ({len(train_features)} samples)")
            return self._empty_window_result(window)

        # 3. Train ML model
        model, feature_names = self._train_model(train_features, train_labels, sample_weights)

        # 4. MDI feature importance
        mdi = dict(zip(feature_names, model.feature_importances_))

        # 5. MDA feature importance (permutation importance)
        mda = self._compute_mda(model, train_features, train_labels, feature_names)

        # 6. Predict on test period candidates
        test_candidates = self.candidate_selector.screen_at_date(
            test_start_str, universe_tickers, top_n=self.top_n_candidates,
        )
        test_tickers = [c["ticker"] for c in test_candidates]

        signals, predictions = self._predict_and_trade(
            model, feature_names, test_tickers, test_start_str, test_end_str,
        )

        # 7. Execute trades in the test window
        if signals:
            # Get prices at test_start for trading
            prices = {}
            for ticker in test_tickers:
                p = cache.cached_prices(ticker, test_start_str, test_start_str)
                if not p.empty:
                    prices[ticker] = float(p["close"].iloc[-1])

            self.trader.execute_trades(signals, test_start_str, prices)

        # 8. Mark to market at test end and close positions
        end_prices = {}
        for ticker in list(self.trader.positions.keys()) + test_tickers:
            p = cache.cached_prices(ticker, test_end_str, test_end_str)
            if not p.empty:
                end_prices[ticker] = float(p["close"].iloc[-1])

        self.trader.mark_to_market(test_end_str, end_prices)
        self.trader.close_all_positions(test_end_str, end_prices)

        # 9. Compute window metrics
        window_returns = self.trader.get_returns_series()
        # Use only the returns from this window's snapshots
        window_snapshot_count = len(signals) if signals else 0

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
            num_trades=len(signals) if signals else 0,
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

        # Sample at monthly intervals within the training window
        current = pd.Timestamp(train_start) + pd.DateOffset(months=1)
        end = pd.Timestamp(train_end)

        sample_dates = []
        while current <= end:
            sample_dates.append(current.strftime("%Y-%m-%d"))
            current += pd.DateOffset(months=1)

        for sample_date in sample_dates:
            for ticker in tickers:
                try:
                    fv = self.data_provider.build_feature_vector(ticker, sample_date)
                    if not fv or fv.get("price_at_analysis") is None:
                        continue

                    label = self._compute_triple_barrier_label(ticker, sample_date)
                    if label is None:
                        continue

                    features_list.append(fv)
                    labels_list.append(label)

                    # Track entry/exit dates for sample weight computation
                    entry_dates.append(pd.Timestamp(sample_date))
                    exit_dates.append(
                        pd.Timestamp(sample_date) + timedelta(days=self.holding_days)
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
        """
        end_date = (pd.Timestamp(entry_date) + timedelta(days=int(self.holding_days * 1.5))).strftime("%Y-%m-%d")
        prices = cache.cached_prices(ticker, entry_date, end_date)

        if prices.empty:
            return None

        entry_price = float(prices["close"].iloc[0])
        tp_price = entry_price * (1 + self.tp_pct / 100)
        sl_price = entry_price * (1 - self.sl_pct / 100)

        # Walk forward through prices
        trading_days = 0
        for idx in range(1, len(prices)):
            trading_days += 1
            price = float(prices["close"].iloc[idx])

            if price >= tp_price:
                return 1  # Hit take-profit
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
    ) -> tuple[GradientBoostingClassifier, list[str]]:
        """Train GradientBoosting classifier with sample weights."""
        model = GradientBoostingClassifier(
            n_estimators=self.ml_params["n_estimators"],
            max_depth=self.ml_params["max_depth"],
            min_samples_leaf=self.ml_params["min_samples_leaf"],
            learning_rate=self.ml_params["learning_rate"],
            random_state=42,
        )
        model.fit(X, y, sample_weight=sample_weights)
        return model, list(X.columns)

    def _compute_mda(
        self,
        model: GradientBoostingClassifier,
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
                model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1,
            )
            return dict(zip(feature_names, result["importances_mean"]))
        except Exception as e:
            logger.warning(f"MDA computation failed: {e}")
            return {}

    # ── Prediction + trade signal generation ─────────────────────

    def _predict_and_trade(
        self,
        model: GradientBoostingClassifier,
        feature_names: list[str],
        test_tickers: list[str],
        test_start: str,
        test_end: str,
    ) -> tuple[list[dict], list[dict]]:
        """
        Generate predictions on test period candidates.
        Returns (signals for trader, predictions for analytics).
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

                volatility = fv.get("annualized_volatility", 0.3) or 0.3

                signals.append({
                    "ticker": ticker,
                    "label": pred_label,
                    "probability": probability,
                    "volatility": volatility,
                })

                # Check actual outcome for hit rate
                actual_label = self._compute_triple_barrier_label(ticker, test_start)
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

    # ── Metric helpers ───────────────────────────────────────────

    @staticmethod
    def _sharpe(returns: np.ndarray, risk_free_rate: float = 0.04) -> float:
        if len(returns) < 5:
            return 0.0
        excess = returns - risk_free_rate / 252
        std = excess.std()
        if std == 0:
            return 0.0
        return float((excess.mean() / std) * np.sqrt(252))

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return float(drawdown.min() * 100)

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

    def _report_progress(self, message: str):
        if self.progress_callback:
            self.progress_callback(message)
        logger.info(f"Backtest: {message}")
