"""
Quant Model Signal — 12th enrichment data tool.

Bridges backtest ML insights into the live analysis pipeline by scoring
a ticker using the latest MDA (Mean Decrease Accuracy) feature importance
weights from walk-forward backtesting.

Design: Rather than persisting a stale pickled model, this tool uses
MDA feature importance as factor weights to produce a live composite score.
This adapts automatically as QuantOpt discovers new winning params.

Research basis:
- López de Prado Ch. 8: MDA over MDI for feature importance
- Fama-French: multi-factor scoring
- FinRL three-layer: data → agent → analytics feedback loop
"""

import logging

import numpy as np
import yfinance as yf

from backend.backtest.backtest_engine import get_latest_mda

logger = logging.getLogger(__name__)

# Features computable from live yfinance data (no BQ needed).
# These overlap with _NUMERIC_FEATURES in backtest_engine.py.
_LIVE_FEATURES = {
    "momentum_1m", "momentum_3m", "momentum_6m",
    "rsi_14", "annualized_volatility", "sma_50_distance", "sma_200_distance",
    "pe_ratio", "pb_ratio", "debt_equity", "roe", "profit_margin",
    "volume_ratio_20d", "fcf_yield", "dividend_yield",
    "quality_score", "revenue_growth_yoy",
}


def _compute_rsi(prices: "np.ndarray", period: int = 14) -> float:
    """Relative Strength Index from closing prices."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[-period:]))
    avg_loss = float(np.mean(losses[-period:]))
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _build_live_features(ticker: str) -> dict[str, float]:
    """Build a feature dict from live yfinance data for a single ticker."""
    features: dict[str, float] = {}
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        if hist.empty or len(hist) < 30:
            return features

        closes: np.ndarray = hist["Close"].to_numpy(dtype=float)
        volumes: np.ndarray = hist["Volume"].to_numpy(dtype=float)
        current = float(closes[-1])

        # Momentum returns
        for label, days in [("momentum_1m", 21), ("momentum_3m", 63), ("momentum_6m", 126)]:
            if len(closes) > days:
                features[label] = float((current / closes[-days] - 1))
            else:
                features[label] = 0.0

        # Volatility (annualized)
        daily_rets = np.diff(np.log(closes))
        features["annualized_volatility"] = float(np.std(daily_rets) * np.sqrt(252))

        # RSI
        features["rsi_14"] = _compute_rsi(closes, 14)

        # SMA distances
        sma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else current
        sma200 = float(np.mean(closes[-200:])) if len(closes) >= 200 else current
        features["sma_50_distance"] = (current - sma50) / sma50 if sma50 else 0.0
        features["sma_200_distance"] = (current - sma200) / sma200 if sma200 else 0.0

        # Volume ratio
        if len(volumes) >= 21 and np.mean(volumes[-20:]) > 0:
            features["volume_ratio_20d"] = float(volumes[-1] / np.mean(volumes[-20:]))
        else:
            features["volume_ratio_20d"] = 1.0

        # Fundamental ratios from yfinance info
        features["pe_ratio"] = float(info.get("trailingPE") or 0)
        features["pb_ratio"] = float(info.get("priceToBook") or 0)
        features["debt_equity"] = float(info.get("debtToEquity", 0) or 0) / 100.0
        features["roe"] = float(info.get("returnOnEquity") or 0)
        features["profit_margin"] = float(info.get("profitMargins") or 0)

        # FCF yield
        market_cap = float(info.get("marketCap") or 0)
        ocf = float(info.get("operatingCashflow") or 0)
        features["fcf_yield"] = (ocf * 4) / market_cap if market_cap > 0 else 0.0

        # Dividend yield
        features["dividend_yield"] = float(info.get("dividendYield") or 0)

        # Quality score (ROE × margin × (1 − normalized D/E))
        de_norm = min(features["debt_equity"], 1.0)
        features["quality_score"] = features["roe"] * features["profit_margin"] * (1 - de_norm)

        # Revenue growth YoY
        features["revenue_growth_yoy"] = float(info.get("revenueGrowth") or 0)

    except Exception as e:
        logger.warning("Failed to build live features for %s: %s", ticker, e)
    return features


def _score_ticker(features: dict[str, float], mda: dict[str, float]) -> float:
    """Compute a weighted factor score using MDA importance as weights.

    Higher score → more favorable (bullish factors weighted by importance).
    Returns a Z-score-style value centered around 0.
    """
    if not features or not mda:
        return 0.0

    # Directional sign: for some features, higher = better; others, higher = worse.
    # Positive direction: momentum, roe, profit_margin, quality_score, fcf_yield,
    #   dividend_yield, revenue_growth_yoy, sma_50_distance, sma_200_distance
    # Negative direction (lower = better): pe_ratio, pb_ratio, debt_equity,
    #   annualized_volatility
    # Neutral: rsi_14 (centered at 50), volume_ratio_20d (centered at 1)
    _DIRECTION = {
        "momentum_1m": 1, "momentum_3m": 1, "momentum_6m": 1,
        "rsi_14": 0,  # 50 is neutral
        "annualized_volatility": -1,
        "sma_50_distance": 1, "sma_200_distance": 1,
        "pe_ratio": -1, "pb_ratio": -1, "debt_equity": -1,
        "roe": 1, "profit_margin": 1,
        "volume_ratio_20d": 0,
        "fcf_yield": 1, "dividend_yield": 1,
        "quality_score": 1, "revenue_growth_yoy": 1,
    }

    weighted_sum = 0.0
    total_weight = 0.0
    contributing_factors: list[dict] = []

    for feat, val in features.items():
        weight = abs(mda.get(feat, 0.0))
        if weight < 1e-6:
            continue
        direction = _DIRECTION.get(feat, 0)
        if direction == 0:
            continue  # skip non-directional features

        contribution = direction * val * weight
        weighted_sum += contribution
        total_weight += weight
        contributing_factors.append({
            "feature": feat, "value": round(val, 4),
            "mda_weight": round(weight, 4), "contribution": round(contribution, 4),
        })

    score = weighted_sum / total_weight if total_weight > 0 else 0.0
    return score


def _classify_signal(score: float) -> str:
    """Map composite score to a signal bucket."""
    if score > 0.08:
        return "STRONG_BULLISH"
    if score > 0.03:
        return "BULLISH"
    if score < -0.08:
        return "STRONG_BEARISH"
    if score < -0.03:
        return "BEARISH"
    return "NEUTRAL"


def get_quant_model_signal(ticker: str) -> dict:
    """
    12th enrichment signal: Score a ticker using MDA-weighted factors
    from the latest walk-forward backtest.

    Returns standard enrichment format:
    {ticker, signal, summary, score, top_factors, mda_source, data}
    """
    try:
        # Load MDA weights
        mda = get_latest_mda()
        mda_source = "backtest" if mda else "equal_weight"

        # Fallback: equal weights over all live features
        if not mda:
            mda = {f: 1.0 / len(_LIVE_FEATURES) for f in _LIVE_FEATURES}

        # Build live features
        features = _build_live_features(ticker)
        if not features:
            return {
                "ticker": ticker,
                "signal": "ERROR",
                "summary": "Could not compute live features from yfinance",
                "score": 0.0,
                "top_factors": [],
                "mda_source": mda_source,
                "data": {},
            }

        # Score
        score = _score_ticker(features, mda)
        signal = _classify_signal(score)

        # Top contributing factors (by abs contribution)
        contributing = []
        for feat, val in features.items():
            w = abs(mda.get(feat, 0.0))
            d = {
                "momentum_1m": 1, "momentum_3m": 1, "momentum_6m": 1,
                "annualized_volatility": -1, "sma_50_distance": 1, "sma_200_distance": 1,
                "pe_ratio": -1, "pb_ratio": -1, "debt_equity": -1,
                "roe": 1, "profit_margin": 1, "fcf_yield": 1, "dividend_yield": 1,
                "quality_score": 1, "revenue_growth_yoy": 1,
            }.get(feat, 0)
            if d != 0 and w > 1e-6:
                contributing.append({
                    "feature": feat, "value": round(val, 4),
                    "weight": round(w, 4), "contribution": round(d * val * w, 4),
                })
        contributing.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        top_factors = contributing[:5]

        # Summary
        top_str = ", ".join(f"{f['feature']}({f['contribution']:+.3f})" for f in top_factors[:3])
        summary = (
            f"Quant model score: {score:.4f} -> {signal}. "
            f"MDA source: {mda_source}. Top factors: {top_str}"
        )

        return {
            "ticker": ticker,
            "signal": signal,
            "summary": summary,
            "score": round(score, 4),
            "top_factors": top_factors,
            "mda_source": mda_source,
            "data": {
                "features": {k: round(v, 4) for k, v in features.items()},
                "mda_weights_used": len(mda),
                "features_computed": len(features),
            },
        }
    except Exception as e:
        logger.error("Quant model signal failed for %s: %s", ticker, e)
        return {
            "ticker": ticker,
            "signal": "ERROR",
            "summary": f"Quant model error: {e}",
            "score": 0.0,
            "top_factors": [],
            "mda_source": "error",
            "data": {},
        }
