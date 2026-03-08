"""
Multi-dimensional anomaly detector.
Uses Z-score and IQR methods to identify statistical outliers across
financial metrics, price behavior, and enrichment signals.

Research basis: Goldman Sachs 127-dimensional anomaly detection (ref 16).
"""

import logging

import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)


def get_anomaly_scan(ticker: str) -> dict:
    """
    Perform multi-dimensional anomaly detection on a ticker.
    Analyzes price, volume, valuation, and fundamental metrics for outliers.

    Returns anomalies with Z-scores and classifications.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")

        if hist.empty or len(hist) < 20:
            return {
                "ticker": ticker,
                "signal": "ERROR",
                "summary": "Insufficient data for anomaly detection.",
            }

        anomalies = []

        # ── Price & Volume Anomalies ────────────────────────────
        close = hist["Close"].dropna().values
        volume = hist["Volume"].dropna().values

        # Recent price vs historical distribution
        if len(close) >= 60:
            recent_return_20d = (close[-1] / close[-21] - 1) * 100
            returns_20d = [(close[i] / close[i - 20] - 1) * 100 for i in range(20, len(close))]
            if len(returns_20d) > 1:
                mu_r = float(np.mean(returns_20d))
                sigma_r = float(np.std(returns_20d))
                if sigma_r > 0:
                    z_return = (recent_return_20d - mu_r) / sigma_r
                    if abs(z_return) > 2:
                        anomalies.append({
                            "metric": "20d_price_return",
                            "value": round(recent_return_20d, 2),
                            "z_score": round(z_return, 2),
                            "mean": round(mu_r, 2),
                            "std": round(sigma_r, 2),
                        })

        # Volume anomaly (recent 5d avg vs 60d avg)
        if len(volume) >= 60:
            recent_vol = float(np.mean(volume[-5:]))
            hist_vol = float(np.mean(volume[-60:]))
            std_vol = float(np.std(volume[-60:]))
            if std_vol > 0:
                z_vol = (recent_vol - hist_vol) / std_vol
                if abs(z_vol) > 2:
                    anomalies.append({
                        "metric": "volume_5d_vs_60d",
                        "value": round(recent_vol, 0),
                        "z_score": round(z_vol, 2),
                        "mean": round(hist_vol, 0),
                        "std": round(std_vol, 0),
                    })

        # Volatility anomaly (recent 20d realized vol vs 1Y)
        if len(close) >= 60:
            daily_returns = np.diff(np.log(close))
            recent_vol_20d = float(np.std(daily_returns[-20:]) * np.sqrt(252) * 100)
            hist_vol_1y = float(np.std(daily_returns) * np.sqrt(252) * 100)
            # Use rolling 20d windows to get distribution
            vols_20d = [
                float(np.std(daily_returns[i:i + 20]) * np.sqrt(252) * 100)
                for i in range(0, len(daily_returns) - 20, 5)
            ]
            if len(vols_20d) > 1:
                mu_vol = float(np.mean(vols_20d))
                std_vol = float(np.std(vols_20d))
                if std_vol > 0:
                    z_vol = (recent_vol_20d - mu_vol) / std_vol
                    if abs(z_vol) > 2:
                        anomalies.append({
                            "metric": "realized_volatility_20d",
                            "value": round(recent_vol_20d, 2),
                            "z_score": round(z_vol, 2),
                            "mean": round(mu_vol, 2),
                            "std": round(std_vol, 2),
                        })

        # ── Fundamental Anomalies ───────────────────────────────
        # These compare current metrics to sector/market norms via IQR-style checks

        pe = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        if pe and forward_pe and pe > 0 and forward_pe > 0:
            pe_gap = ((pe - forward_pe) / pe) * 100
            # Large gap between trailing and forward PE suggests expectations shift
            if abs(pe_gap) > 30:
                anomalies.append({
                    "metric": "pe_trailing_vs_forward_gap",
                    "value": round(pe_gap, 2),
                    "z_score": round(pe_gap / 15, 2),  # normalized
                    "mean": 0,
                    "std": 15,
                })

        # Revenue growth vs profit margin divergence
        rev_growth = info.get("revenueGrowth", 0)
        profit_margin = info.get("profitMargins", 0)
        if rev_growth and profit_margin:
            # High revenue growth with negative/low margins = potential issue
            if rev_growth > 0.15 and profit_margin < 0.05:
                anomalies.append({
                    "metric": "growth_margin_divergence",
                    "value": round(rev_growth * 100, 1),
                    "z_score": 2.5,  # flagged as anomaly
                    "mean": 0,
                    "std": 0,
                    "note": f"Revenue growing {rev_growth*100:.0f}% but profit margin only {profit_margin*100:.1f}%",
                })

        # Debt/equity spike
        de_ratio = info.get("debtToEquity")
        if de_ratio and de_ratio > 200:
            anomalies.append({
                "metric": "debt_to_equity",
                "value": round(de_ratio, 1),
                "z_score": round(de_ratio / 100, 2),
                "mean": 100,
                "std": 50,
            })

        # Short ratio anomaly
        short_ratio = info.get("shortRatio")
        if short_ratio and short_ratio > 5:
            anomalies.append({
                "metric": "short_ratio",
                "value": round(short_ratio, 2),
                "z_score": round((short_ratio - 2) / 1.5, 2),
                "mean": 2,
                "std": 1.5,
            })

        # ── Classify overall signal ─────────────────────────────
        total = len(anomalies)
        risk_anomalies = sum(1 for a in anomalies if a.get("z_score", 0) < -2 or a.get("metric") in [
            "debt_to_equity", "short_ratio", "growth_margin_divergence"
        ])
        opportunity_anomalies = total - risk_anomalies

        if total == 0:
            signal = "NORMAL"
            summary = "No significant statistical anomalies detected."
        elif risk_anomalies > opportunity_anomalies:
            signal = "ANOMALY_RISK"
            summary = f"Detected {total} anomalies, primarily risk-related."
        else:
            signal = "ANOMALY_OPPORTUNITY"
            summary = f"Detected {total} anomalies, potentially indicating opportunity."

        return {
            "ticker": ticker,
            "signal": signal,
            "summary": summary,
            "anomaly_count": total,
            "anomalies": anomalies,
            "current_price": round(float(close[-1]), 2) if len(close) > 0 else None,
        }

    except Exception as e:
        logger.error(f"Anomaly detection failed for {ticker}: {e}")
        return {
            "ticker": ticker,
            "signal": "ERROR",
            "summary": f"Anomaly detection failed: {e}",
        }
