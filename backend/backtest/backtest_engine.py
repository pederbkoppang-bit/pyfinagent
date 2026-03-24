"""
Backtest engine — main orchestrator for walk-forward ML backtesting.
Trains GradientBoosting with Triple Barrier labels, sample weights,
fractional differentiation, and meta-labeling position sizing.
Zero LLM cost.
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
from sklearn.ensemble import GradientBoostingClassifier
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
    "meta_label": "_compute_triple_barrier_label",  # same labels, secondary model on top
}

# Numeric features used for ML training (excludes categorical: ticker, date, sector, industry)
_NUMERIC_FEATURES = [
    "price_at_analysis", "momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m",
    "rsi_14", "annualized_volatility", "sma_50_distance", "sma_200_distance",
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
            **self.ml_params,
        }

    def get_model_trained_at(self) -> str:
        """Return ISO timestamp of last model training. Empty string if never trained."""
        return self.model_trained_at

    def run_backtest(
        self,
        universe_tickers: list[str] | None = None,
        skip_cache_clear: bool = False,
    ) -> BacktestResult:
        """
        Run full walk-forward backtest. Main entry point.
        Auto-checks BQ data availability and triggers ingestion if empty.
        Returns BacktestResult with per-window and aggregate metrics.

        Args:
            skip_cache_clear: If True, skip cache.clear_cache() at end.
                Used by QuantStrategyOptimizer to keep warm cache across iterations.
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

        # 2. Build training data
        self._report_progress("building_features", "Building training data", window=wid)
        train_features, train_labels, sample_weights = self._build_training_data(
            candidate_tickers, window.train_start.isoformat(), train_end_str,
        )

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
        mdi = dict(zip(feature_names, model.feature_importances_))

        # 5. MDA feature importance (permutation importance)
        self._report_progress("computing_mda", "Permutation importance", window=wid)
        mda = self._compute_mda(model, train_features, train_labels, feature_names)

        # 6. Predict on test period candidates
        self._report_progress("predicting", f"Scoring test candidates at {test_start_str}", window=wid)
        test_candidates = self.candidate_selector.screen_at_date(
            test_start_str, universe_tickers, top_n=self.top_n_candidates,
            scoring_weights=scoring_weights if scoring_weights else None,
        )
        test_tickers = [c["ticker"] for c in test_candidates]

        signals, predictions = self._predict_and_trade(
            model, feature_names, test_tickers, test_start_str, test_end_str,
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
        for day in test_days:
            day_str = day.strftime("%Y-%m-%d")
            day_prices = {}
            for ticker in active_tickers:
                p = cache.cached_prices(ticker, day_str, day_str)
                if not p.empty:
                    day_prices[ticker] = float(p["close"].iloc[-1])
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
        from datetime import datetime as _dt, timezone as _tz
        self.model_trained_at = _dt.now(_tz.utc).isoformat()
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
        Mean Reversion (Lo & MacKinlay 1990):
        +1 if price significantly below SMA (oversold bounce candidate),
        -1 if significantly above SMA (overbought reversion candidate),
        0 if within normal range.
        """
        fv = self.data_provider.build_feature_vector(ticker, entry_date)
        if not fv or fv.get("price_at_analysis") is None:
            return None

        sma_dist = fv.get("sma_50_distance")
        rsi = fv.get("rsi_14")
        if sma_dist is None or rsi is None:
            return None

        # Oversold: price well below SMA50 + RSI < 30
        if sma_dist < -0.05 and rsi < 35:
            return 1
        # Overbought: price well above SMA50 + RSI > 70
        if sma_dist > 0.10 and rsi > 70:
            return -1
        return 0

    def _compute_factor_label(self, ticker: str, entry_date: str) -> int | None:
        """
        Multi-Factor Composite (Fama-French 5-factor):
        Score based on value (low P/E), momentum, quality (high ROE), low volatility.
        +1 if composite > 0.6, -1 if composite < 0.3, 0 otherwise.
        """
        fv = self.data_provider.build_feature_vector(ticker, entry_date)
        if not fv or fv.get("price_at_analysis") is None:
            return None

        # Normalize factors to 0-1 range
        pe = fv.get("pe_ratio")
        momentum = fv.get("momentum_6m")
        vol = fv.get("annualized_volatility")
        roe = fv.get("roe")
        div_yield = fv.get("dividend_yield", 0) or 0

        if pe is None or momentum is None or vol is None:
            return None

        # Value: lower P/E is better (capped 5-40)
        value_score = max(0, min(1, 1 - (max(5, min(40, pe)) - 5) / 35))
        # Momentum: positive is better (capped -30% to +50%)
        mom_score = max(0, min(1, (max(-30, min(50, momentum)) + 30) / 80))
        # Low vol: lower is better (capped 0.10 to 0.60)
        vol_score = max(0, min(1, 1 - (max(0.10, min(0.60, vol)) - 0.10) / 0.50))
        # Quality: higher ROE is better (capped 0-40%)
        quality_score = max(0, min(1, (roe or 0) / 0.40))
        # Yield
        yield_score = max(0, min(1, div_yield / 0.05))

        composite = (
            value_score * 0.25 + mom_score * 0.25 + vol_score * 0.20
            + quality_score * 0.20 + yield_score * 0.10
        )

        if composite > 0.6:
            return 1
        if composite < 0.3:
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
