"""
Stock universe screener — quant-only filter for paper trading candidates.
Uses yfinance batch download. Zero LLM cost.
"""

import logging
from typing import Optional

import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

# S&P 500 tickers — updated periodically. Good starting universe.
# In production, this could be loaded from a file or API.
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Sector ETF mapping for relative strength calculation
SECTOR_ETFS = {
    "Technology": "XLK", "Health Care": "XLV", "Financials": "XLF",
    "Consumer Discretionary": "XLY", "Communication Services": "XLC",
    "Industrials": "XLI", "Consumer Staples": "XLP", "Energy": "XLE",
    "Utilities": "XLU", "Real Estate": "XLRE", "Materials": "XLB",
}


def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 ticker list from Wikipedia."""
    try:
        tables = pd.read_html(SP500_URL, header=0)
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info(f"Loaded {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        logger.warning(f"Failed to fetch S&P 500 list: {e}")
        return _FALLBACK_TICKERS


def screen_universe(
    tickers: Optional[list[str]] = None,
    min_market_cap: float = 1e9,
    min_avg_volume: int = 100_000,
    min_price: float = 5.0,
    period: str = "6mo",
) -> list[dict]:
    """
    Screen a universe of tickers using quant factors.
    Returns raw screening data for each ticker that passes basic filters.

    Cost: $0 (yfinance only, no LLM, no API keys)
    """
    if tickers is None:
        tickers = get_sp500_tickers()

    logger.info(f"Screening {len(tickers)} tickers (period={period})")

    # Batch download price history
    try:
        data = yf.download(tickers, period=period, group_by="ticker",
                           auto_adjust=True, threads=True, progress=False)
    except Exception as e:
        logger.error(f"yfinance batch download failed: {e}")
        return []

    if data is None or data.empty:
        return []

    results = []
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                ticker_data = data
            else:
                ticker_data = data[ticker] if ticker in data.columns.get_level_values(0) else None

            if ticker_data is None or ticker_data.empty:
                continue

            close = ticker_data["Close"].dropna()
            volume = ticker_data["Volume"].dropna()

            if len(close) < 20:
                continue

            current_price = float(close.iloc[-1])
            avg_vol = float(volume.tail(20).mean())

            # Basic filters
            if current_price < min_price or avg_vol < min_avg_volume:
                continue

            # Momentum factors
            momentum_1m = _pct_change(close, 21)
            momentum_3m = _pct_change(close, 63)
            momentum_6m = _pct_change(close, len(close) - 1)

            # RSI (14-day)
            rsi = _compute_rsi(close, 14)

            # Volatility (annualized)
            daily_returns = close.pct_change().dropna()
            volatility = float(daily_returns.std() * (252 ** 0.5)) if len(daily_returns) > 5 else None

            # Mean reversion signal: distance from 50-day SMA
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
            logger.debug(f"Skipping {ticker}: {e}")
            continue

    logger.info(f"Screening complete: {len(results)}/{len(tickers)} passed basic filters")
    return results


def rank_candidates(
    screen_data: list[dict],
    top_n: int = 10,
    strategy: str = "momentum",
) -> list[dict]:
    """
    Rank screened candidates by composite alpha score.

    Strategies:
    - "momentum": Favor strong recent momentum + reasonable RSI (not overbought)
    - "value_momentum": Blend momentum with mean-reversion (SMA distance)

    Returns top_n candidates sorted by composite score (descending).
    """
    if not screen_data:
        return []

    scored = []
    for stock in screen_data:
        mom_1m = stock.get("momentum_1m") or 0
        mom_3m = stock.get("momentum_3m") or 0
        mom_6m = stock.get("momentum_6m") or 0
        rsi = stock.get("rsi_14") or 50
        vol = stock.get("volatility_ann") or 0.3
        sma_dist = stock.get("sma_50_distance_pct") or 0

        if strategy == "momentum":
            # Composite: weight recent momentum more, penalize overbought/oversold
            score = (
                mom_1m * 0.40 +
                mom_3m * 0.35 +
                mom_6m * 0.25
            )
            # RSI penalty: reduce score if extremely overbought (>80) or oversold (<20)
            if rsi > 80:
                score *= 0.7
            elif rsi < 20:
                score *= 0.8
            # Volatility adjustment: slightly penalize very high vol
            if vol > 0.6:
                score *= 0.85
        elif strategy == "value_momentum":
            # Blend: strong 3M momentum but currently pulled back from SMA
            score = mom_3m * 0.5 - abs(sma_dist) * 0.2 + mom_1m * 0.3
        else:
            score = mom_3m

        scored.append({**stock, "composite_score": round(score, 3)})

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    return scored[:top_n]


def _pct_change(series: pd.Series, periods: int) -> Optional[float]:
    """Calculate percentage change over N periods."""
    if len(series) <= periods:
        return None
    old = float(series.iloc[-periods - 1])
    new = float(series.iloc[-1])
    return ((new - old) / old) * 100 if old != 0 else None


def _compute_rsi(series: pd.Series, period: int = 14) -> Optional[float]:
    """Compute RSI indicator."""
    if len(series) < period + 1:
        return None
    delta = series.diff().dropna()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.rolling(window=period).mean().iloc[-1]
    avg_loss = loss.rolling(window=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# Fallback tickers if Wikipedia scrape fails
_FALLBACK_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "UNH", "JNJ", "V", "XOM", "JPM", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "LLY", "PEP", "KO", "AVGO", "COST", "TMO", "MCD", "WMT",
    "CRM", "ACN", "CSCO", "ABT", "DHR", "ADBE", "NKE", "TXN", "NEE",
    "PM", "UNP", "RTX", "LOW", "HON", "ORCL", "BMY", "QCOM", "UPS",
    "INTC", "AMD", "SBUX", "BA",
]
