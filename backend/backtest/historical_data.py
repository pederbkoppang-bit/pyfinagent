"""
Historical data provider — builds point-in-time feature vectors from BQ data.
All features computed using ONLY data available as-of cutoff_date.
No future leakage allowed.
"""

import logging
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from backend.backtest import cache

logger = logging.getLogger(__name__)

# Features that require fractional differentiation (non-stationary)
_NON_STATIONARY_FEATURES = {"price_at_analysis", "market_cap", "total_revenue", "total_debt", "total_equity"}


class HistoricalDataProvider:
    """Builds point-in-time feature vectors from BQ historical data."""

    def __init__(self):
        pass  # Uses module-level cache layer — no direct BQ access

    # ── Core data accessors ──────────────────────────────────────

    def get_point_in_time_prices(
        self, ticker: str, cutoff_date: str, lookback_days: int = 504
    ) -> pd.DataFrame:
        """Get OHLCV up to cutoff_date from BQ cache."""
        cutoff = pd.Timestamp(cutoff_date)
        start = (cutoff - timedelta(days=int(lookback_days * 1.5))).strftime("%Y-%m-%d")
        df = cache.cached_prices(ticker, start, cutoff_date)
        return df

    def get_point_in_time_fundamentals(self, ticker: str, cutoff_date: str) -> list[dict]:
        """Get up to 5 most recent quarterly fundamentals as-of cutoff.

        Returns list ordered by report_date DESC (index 0 = most recent).
        """
        return cache.cached_fundamentals(ticker, cutoff_date)

    def get_point_in_time_macro(self, cutoff_date: str) -> dict:
        """Get most recent FRED macro values as-of cutoff."""
        return cache.cached_macro(cutoff_date)

    # ── Feature engineering ──────────────────────────────────────

    def build_feature_vector(self, ticker: str, cutoff_date: str) -> dict:
        """
        Build a complete feature vector for a ticker at a point in time.
        Returns ~43 features. Non-stationary features are raw here;
        fractional differentiation is applied at the engine level across
        the full training set for consistency.
        """
        prices = self.get_point_in_time_prices(ticker, cutoff_date)
        fundamentals_list = self.get_point_in_time_fundamentals(ticker, cutoff_date)
        fundamentals = fundamentals_list[0] if fundamentals_list else {}
        macro = self.get_point_in_time_macro(cutoff_date)

        features: dict = {"ticker": ticker, "date": cutoff_date}

        # ── Price-derived features ───────────────────────────────
        if prices.empty or len(prices) < 20:
            return features

        close = prices["close"]
        volume = prices["volume"]
        current_price = float(close.iloc[-1])
        features["price_at_analysis"] = current_price

        # Momentum
        features["momentum_1m"] = self._pct_change(close, 21)
        features["momentum_3m"] = self._pct_change(close, 63)
        features["momentum_6m"] = self._pct_change(close, 126)
        features["momentum_12m"] = self._pct_change(close, 252)

        # RSI (14-day)
        features["rsi_14"] = self._compute_rsi(close, 14)

        # Volatility (annualized)
        daily_returns = close.pct_change().dropna()
        if len(daily_returns) > 5:
            features["annualized_volatility"] = float(daily_returns.std() * math.sqrt(252))
        else:
            features["annualized_volatility"] = None

        # SMA distance
        if len(close) >= 50:
            sma_50 = float(close.tail(50).mean())
            features["sma_50_distance"] = (current_price - sma_50) / sma_50
        if len(close) >= 200:
            sma_200 = float(close.tail(200).mean())
            features["sma_200_distance"] = (current_price - sma_200) / sma_200

        # Volume ratio (current vs 20d avg)
        if volume is not None and len(volume) >= 20:
            avg_vol_20d = float(volume.tail(20).mean())
            if avg_vol_20d > 0:
                features["volume_ratio_20d"] = float(volume.iloc[-1]) / avg_vol_20d

        # ── Monte Carlo VaR (from historical returns) ────────────
        if len(daily_returns) >= 60:
            mc = self._compute_monte_carlo_var(daily_returns, current_price, horizon_days=126)
            features.update(mc)

        # ── Anomaly count (Z-score) ──────────────────────────────
        features["anomaly_count"] = self._compute_anomaly_count(prices)

        # ── Amihud illiquidity (López de Prado Ch. 18) ───────────
        features["amihud_illiquidity"] = self._compute_amihud_illiquidity(prices)

        # ── Fundamentals ─────────────────────────────────────────
        if fundamentals:
            shares = fundamentals.get("shares_outstanding")
            revenue = fundamentals.get("total_revenue")
            net_income = fundamentals.get("net_income")
            total_debt = fundamentals.get("total_debt")
            total_equity = fundamentals.get("total_equity")
            total_assets = fundamentals.get("total_assets")

            features["total_revenue"] = revenue
            features["net_income"] = net_income
            features["total_debt"] = total_debt
            features["total_equity"] = total_equity
            features["total_assets"] = total_assets

            if shares and shares > 0:
                features["market_cap"] = current_price * shares
                if net_income and net_income > 0:
                    eps = net_income * 4 / shares  # Annualize quarterly
                    features["pe_ratio"] = current_price / eps

            if total_equity and total_equity > 0:
                if total_debt is not None:
                    features["debt_equity"] = total_debt / total_equity
                if net_income is not None:
                    features["roe"] = (net_income * 4) / total_equity

            if revenue and revenue > 0:
                features["profit_margin"] = (net_income / revenue) if net_income else None

            # Price-to-Book ratio
            if total_equity and total_equity > 0 and shares and shares > 0:
                book_per_share = total_equity / shares
                if book_per_share > 0:
                    features["pb_ratio"] = current_price / book_per_share

            # FCF yield = (operating_cash_flow - capex) / market_cap
            ocf = fundamentals.get("operating_cash_flow")
            market_cap = features.get("market_cap")
            if ocf is not None and market_cap and market_cap > 0:
                # Approximate capex as 0 when unavailable (conservative — FCF ≈ OCF)
                fcf = ocf * 4  # Annualize quarterly
                features["fcf_yield"] = fcf / market_cap

            # Dividend yield (from fundamentals if available)
            dividends = fundamentals.get("dividends_per_share")
            if dividends and dividends > 0 and current_price > 0:
                features["dividend_yield"] = dividends / current_price

            # Quality score = ROE × profit_margin × (1 − normalized_D/E)
            roe_val = features.get("roe")
            pm_val = features.get("profit_margin")
            de_val = features.get("debt_equity")
            if roe_val is not None and pm_val is not None:
                de_norm = min(1.0, max(0, (de_val or 0)) / 3.0)  # Cap at D/E=3
                features["quality_score"] = max(0, (roe_val * pm_val * (1 - de_norm)))

            # Revenue growth YoY: compare current quarter vs same quarter 4 periods ago
            features["revenue_growth_yoy"] = self._compute_revenue_growth_yoy(
                fundamentals_list, revenue
            )

            features["sector"] = fundamentals.get("sector", "")
            features["industry"] = fundamentals.get("industry", "")

        # ── Macro ────────────────────────────────────────────────
        if macro:
            features["fed_funds_rate"] = macro.get("FEDFUNDS", {}).get("value")
            features["cpi_yoy"] = macro.get("CPIAUCSL", {}).get("value")
            features["unemployment_rate"] = macro.get("UNRATE", {}).get("value")
            features["yield_curve_spread"] = macro.get("T10Y2Y", {}).get("value")
            features["consumer_sentiment"] = macro.get("UMCSENT", {}).get("value")
            features["treasury_10y"] = macro.get("DGS10", {}).get("value")

        return features

    # ── Turbulence Index (FinRL) ─────────────────────────────────

    def compute_turbulence_index(
        self, cutoff_date: str, universe_tickers: list[str], lookback: int = 252
    ) -> float:
        """
        Mahalanobis distance of current cross-asset returns from historical mean.
        High values → systemic market stress → reduce position sizing.
        """
        cutoff = pd.Timestamp(cutoff_date)
        start = (cutoff - timedelta(days=int(lookback * 1.5))).strftime("%Y-%m-%d")

        # Collect daily returns for all tickers
        returns_dict = {}
        for ticker in universe_tickers[:50]:  # Cap for performance
            prices = cache.cached_prices(ticker, start, cutoff_date)
            if prices.empty or len(prices) < lookback // 2:
                continue
            daily_ret = prices["close"].pct_change().dropna()
            returns_dict[ticker] = daily_ret

        if len(returns_dict) < 5:
            return 0.0

        # Align all series to common dates
        returns_df = pd.DataFrame(returns_dict).dropna()
        if len(returns_df) < lookback // 2:
            return 0.0

        # Historical mean and covariance
        hist_returns = returns_df.iloc[:-1]
        current_returns: np.ndarray = returns_df.iloc[-1].values  # type: ignore[assignment]

        mu: np.ndarray = hist_returns.mean().values  # type: ignore[assignment]
        cov: np.ndarray = hist_returns.cov().values  # type: ignore[assignment]

        try:
            cov_inv = np.linalg.pinv(cov)
            diff = current_returns - mu
            turbulence = float(diff @ cov_inv @ diff.T)
            return turbulence
        except Exception:
            return 0.0

    # ── Fractional Differentiation (López de Prado Ch. 5) ────────

    @staticmethod
    def fractional_diff(series: pd.Series, d: float = 0.4, threshold: float = 1e-5) -> pd.Series:
        """
        Fixed-width window fractional differentiation.
        Achieves stationarity while preserving memory.
        Applied to non-stationary features only.
        """
        # Compute weights
        weights = [1.0]
        k = 1
        while True:
            w = -weights[-1] * (d - k + 1) / k
            if abs(w) < threshold:
                break
            weights.append(w)
            k += 1

        weights = np.array(weights[::-1])
        width = len(weights)

        result = pd.Series(index=series.index, dtype=float)
        for i in range(width - 1, len(series)):
            window: np.ndarray = series.iloc[i - width + 1:i + 1].values  # type: ignore[assignment]
            if len(window) == width and not np.any(np.isnan(window)):
                result.iloc[i] = float(np.dot(weights, window))

        return result

    # ── Private helpers ──────────────────────────────────────────

    @staticmethod
    def _compute_revenue_growth_yoy(
        fundamentals_list: list[dict], current_revenue: float | None
    ) -> float | None:
        """Compute YoY revenue growth from quarterly fundamentals history.

        Compares the most recent quarter's revenue to the same quarter
        4 periods ago (Q vs Q-4). Returns growth as a decimal (e.g., 0.15 = 15%).
        """
        if not current_revenue or not fundamentals_list or len(fundamentals_list) < 5:
            return None
        prior_revenue = fundamentals_list[4].get("total_revenue")
        if not prior_revenue or prior_revenue == 0:
            return None
        return (current_revenue - prior_revenue) / abs(prior_revenue)

    @staticmethod
    def _pct_change(series: pd.Series, periods: int) -> float | None:
        if len(series) < periods + 1:
            return None
        return float((series.iloc[-1] / series.iloc[-periods - 1] - 1) * 100)

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> float | None:
        if len(series) < period + 1:
            return None
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        last_gain = gain.iloc[-1]
        last_loss = loss.iloc[-1]
        if last_loss == 0:
            return 100.0
        rs = last_gain / last_loss
        return float(100 - (100 / (1 + rs)))

    @staticmethod
    def _compute_monte_carlo_var(
        daily_returns: pd.Series, current_price: float, horizon_days: int = 126,
        n_sims: int = 1000
    ) -> dict:
        """Run Monte Carlo GBM simulation for VaR estimation."""
        mu = float(daily_returns.mean())
        sigma = float(daily_returns.std())
        rng = np.random.default_rng(42)  # Deterministic for backtesting

        # GBM paths
        z = rng.standard_normal((n_sims, horizon_days))
        daily_drift = mu - 0.5 * sigma**2
        paths = current_price * np.exp(
            np.cumsum(daily_drift + sigma * z, axis=1)
        )
        final_prices = paths[:, -1]
        returns = (final_prices / current_price - 1) * 100

        return {
            "var_95_6m": float(np.percentile(returns, 5)),
            "var_99_6m": float(np.percentile(returns, 1)),
            "expected_shortfall_6m": float(returns[returns <= np.percentile(returns, 5)].mean()),
            "prob_positive_6m": float((returns > 0).mean()),
        }

    @staticmethod
    def _compute_anomaly_count(prices: pd.DataFrame, threshold: float = 2.0) -> int:
        """Count dimensions where Z-score exceeds threshold."""
        count = 0
        close = prices["close"]
        volume = prices["volume"]

        # Price return Z-score
        daily_ret = close.pct_change().dropna()
        if len(daily_ret) > 20:
            z_ret = abs((daily_ret.iloc[-1] - daily_ret.mean()) / daily_ret.std())
            if z_ret > threshold:
                count += 1

        # Volume Z-score
        if volume is not None and len(volume) > 20:
            vol_clean = volume.dropna()
            if len(vol_clean) > 20:
                z_vol = abs((vol_clean.iloc[-1] - vol_clean.mean()) / vol_clean.std())
                if z_vol > threshold:
                    count += 1

        # Price level Z-score (20-day)
        if len(close) > 20:
            recent = close.tail(20)
            z_price = abs((close.iloc[-1] - recent.mean()) / recent.std())
            if z_price > threshold:
                count += 1

        return count

    @staticmethod
    def _compute_amihud_illiquidity(prices: pd.DataFrame, window: int = 60) -> float | None:
        """
        Amihud illiquidity ratio (López de Prado Ch. 18):
        mean(|daily_return| / dollar_volume) over trailing window.
        Higher = more illiquid = higher price impact.
        """
        close = prices["close"]
        volume = prices["volume"]
        if len(close) < window or volume is None:
            return None

        tail = prices.tail(window)
        daily_ret = tail["close"].pct_change().dropna().abs()
        dollar_vol = (tail["close"] * tail["volume"]).iloc[1:]

        # Avoid division by zero
        valid = dollar_vol > 0
        if valid.sum() < window // 2:
            return None

        # Align indices
        common = daily_ret.index.intersection(dollar_vol[valid].index)
        if len(common) < window // 2:
            return None

        illiquidity = (daily_ret.loc[common] / dollar_vol.loc[common]).mean()
        # Scale up for readability (values are tiny)
        return float(illiquidity * 1e6)
