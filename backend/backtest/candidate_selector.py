"""
Candidate selector — adapts the quant screener for historical backtesting.
Screens using BQ-stored price data at a point in time.
"""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from backend.backtest import cache
from backend.backtest.markets import DEFAULT_MARKET

logger = logging.getLogger(__name__)


class CandidateSelector:
    """Screens and ranks stock candidates using historical BQ data."""

    def screen_at_date(
        self,
        cutoff_date: str,
        universe_tickers: list[str],
        top_n: int = 50,
        min_avg_volume: int = 100_000,
        min_price: float = 5.0,
        scoring_weights: dict | None = None,
    ) -> list[dict]:
        """
        Screen and rank candidates using only data available as-of cutoff_date.
        Reuses the composite alpha scoring logic from screener.py.
        scoring_weights: optional dict with momentum_weight, rsi_weight, volatility_weight, sma_weight.
        """
        cutoff = pd.Timestamp(cutoff_date)
        start = (cutoff - timedelta(days=180)).strftime("%Y-%m-%d")

        results = []
        # Only screen tickers that were preloaded — skip BQ fallback per-ticker queries
        # which can block for 30s each across 500+ tickers
        preloaded = cache.get_preloaded_tickers()
        if preloaded:
            scannable = [t for t in universe_tickers if t in preloaded]
            logger.info(f"screen_at_date: scanning {len(scannable)}/{len(universe_tickers)} preloaded tickers at {cutoff_date}")
        else:
            scannable = universe_tickers
            logger.info(f"screen_at_date: scanning {len(scannable)} tickers at {cutoff_date} (no preload)")
        for i, ticker in enumerate(scannable):
            if i % 100 == 0 and i > 0:
                logger.info(f"screen_at_date: progress {i}/{len(scannable)}")
            try:
                prices = cache.cached_prices(ticker, start, cutoff_date)
                if prices.empty or len(prices) < 20:
                    continue

                close = prices["close"]
                volume = prices["volume"]
                current_price = float(close.iloc[-1])
                avg_vol = float(volume.tail(20).mean()) if volume is not None else 0

                if current_price < min_price or avg_vol < min_avg_volume:
                    continue

                # Momentum factors
                momentum_1m = self._pct_change(close, 21)
                momentum_3m = self._pct_change(close, 63)
                momentum_6m = self._pct_change(close, len(close) - 1)

                # RSI (14-day)
                rsi = self._compute_rsi(close, 14)

                # Volatility (annualized)
                daily_ret = close.pct_change().dropna()
                volatility = float(daily_ret.std() * np.sqrt(252)) if len(daily_ret) > 5 else None

                # SMA distance
                sma_50 = float(close.tail(50).mean()) if len(close) >= 50 else current_price
                sma_distance = (current_price - sma_50) / sma_50 * 100

                results.append({
                    "ticker": ticker,
                    "current_price": round(current_price, 2),
                    "avg_volume_20d": int(avg_vol),
                    "momentum_1m": round(momentum_1m, 2) if momentum_1m else None,
                    "momentum_3m": round(momentum_3m, 2) if momentum_3m else None,
                    "momentum_6m": round(momentum_6m, 2) if momentum_6m else None,
                    "rsi_14": round(rsi, 1) if rsi else None,
                    "volatility_ann": round(volatility, 4) if volatility else None,
                    "sma_50_distance_pct": round(sma_distance, 2),
                })
            except Exception as e:
                logger.warning(f"Skipping {ticker} in screen: {e}")

        # Rank by composite alpha score
        ranked = self._rank_candidates(results, **(scoring_weights or {}))
        return ranked[:top_n]

    def get_universe_tickers(self, market: str = DEFAULT_MARKET) -> list[str]:
        """
        Return universe of tickers for a given market.
        
        Currently supports:
        - US: S&P 500 from Wikipedia (known survivorship bias)
        - Other markets: placeholder — returns empty list with warning
        
        Phase 5 will add: NO (OBX), CA (TSX60), DE (DAX), KR (KOSPI50)
        """
        if market != "US":
            logger.warning(
                "Market '%s' not yet supported -- returning empty universe. "
                "Only 'US' is implemented (Phase 2.9 abstraction).", market
            )
            return []
        
        try:
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                header=0,
                storage_options={"User-Agent": "Mozilla/5.0 (pyfinAgent backtest)"},
            )
            df = tables[0]
            return df["Symbol"].str.replace(".", "-", regex=False).tolist()
        except Exception as e:
            logger.warning(f"Failed to fetch S&P 500 list: {e}")
            return _FALLBACK_TICKERS

    # ── Private helpers ──────────────────────────────────────────

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
    def _rank_candidates(
        candidates: list[dict],
        momentum_weight: float = 0.4,
        rsi_weight: float = 0.2,
        volatility_weight: float = 0.2,
        sma_weight: float = 0.2,
    ) -> list[dict]:
        """Rank candidates by composite alpha score (same logic as screener.py)."""
        if not candidates:
            return []

        for c in candidates:
            mom_score = (c.get("momentum_6m") or 0) / 100
            rsi_val = c.get("rsi_14") or 50
            # RSI: penalize extremes, reward mid-range (mean-reversion component)
            rsi_score = 1 - abs(rsi_val - 50) / 50
            vol_val = c.get("volatility_ann") or 0.3
            # Lower vol = better (inverse)
            vol_score = max(0, 1 - vol_val)
            sma_val = c.get("sma_50_distance_pct") or 0
            # Positive SMA distance = above trend = bullish
            sma_score = min(1, max(-1, sma_val / 10))

            c["alpha_score"] = round(
                mom_score * momentum_weight
                + rsi_score * rsi_weight
                + vol_score * volatility_weight
                + sma_score * sma_weight,
                4,
            )

        return sorted(candidates, key=lambda x: x.get("alpha_score", 0), reverse=True)


# Fallback tickers if Wikipedia fetch fails
_FALLBACK_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "JPM", "V", "UNH",
    "JNJ", "WMT", "PG", "MA", "HD", "XOM", "CVX", "LLY", "ABBV", "MRK",
    "COST", "PEP", "KO", "AVGO", "TMO", "MCD", "CSCO", "ACN", "ABT", "DHR",
    "ADBE", "CRM", "NKE", "TXN", "QCOM", "INTC", "AMD", "HON", "UPS", "BA",
    "CAT", "GE", "MMM", "IBM", "GS", "MS", "AXP", "BLK", "C", "WFC",
]
