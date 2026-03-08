"""
Options flow analysis tool — uses yfinance options chain data.
Detects unusual volume, put/call ratios, and volatility skew.
"""

import logging
from datetime import datetime

import yfinance as yf

logger = logging.getLogger(__name__)


def get_options_flow(ticker: str) -> dict:
    """
    Analyze options chain for unusual activity signals.
    Uses the nearest expiration date for the most liquid options.
    """
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return {"ticker": ticker, "signal": "NO_DATA", "summary": "No options data available."}

        # Use the two nearest expirations for liquidity
        chains = []
        for exp in expirations[:2]:
            chain = stock.option_chain(exp)
            chains.append({"expiration": exp, "calls": chain.calls, "puts": chain.puts})

        total_call_volume = 0
        total_put_volume = 0
        total_call_oi = 0
        total_put_oi = 0
        unusual_calls = []
        unusual_puts = []

        for c in chains:
            calls_df = c["calls"]
            puts_df = c["puts"]
            exp = c["expiration"]

            if not calls_df.empty:
                total_call_volume += int(calls_df["volume"].fillna(0).sum())
                total_call_oi += int(calls_df["openInterest"].fillna(0).sum())

                # Detect unusual call volume (volume > 3x open interest)
                for _, row in calls_df.iterrows():
                    vol = row.get("volume", 0) or 0
                    oi = row.get("openInterest", 0) or 0
                    if oi > 100 and vol > 3 * oi:
                        unusual_calls.append({
                            "expiration": exp,
                            "strike": float(row["strike"]),
                            "volume": int(vol),
                            "openInterest": int(oi),
                            "ratio": round(vol / max(oi, 1), 1),
                        })

            if not puts_df.empty:
                total_put_volume += int(puts_df["volume"].fillna(0).sum())
                total_put_oi += int(puts_df["openInterest"].fillna(0).sum())

                for _, row in puts_df.iterrows():
                    vol = row.get("volume", 0) or 0
                    oi = row.get("openInterest", 0) or 0
                    if oi > 100 and vol > 3 * oi:
                        unusual_puts.append({
                            "expiration": exp,
                            "strike": float(row["strike"]),
                            "volume": int(vol),
                            "openInterest": int(oi),
                            "ratio": round(vol / max(oi, 1), 1),
                        })

        pc_ratio = total_put_volume / max(total_call_volume, 1)
        pc_oi_ratio = total_put_oi / max(total_call_oi, 1)

        # Signal determination
        signal = "NEUTRAL"
        if pc_ratio < 0.5 and len(unusual_calls) >= 2:
            signal = "STRONG_BULLISH"
        elif pc_ratio < 0.7:
            signal = "BULLISH"
        elif pc_ratio > 1.5 and len(unusual_puts) >= 2:
            signal = "STRONG_BEARISH"
        elif pc_ratio > 1.2:
            signal = "BEARISH"

        return {
            "ticker": ticker,
            "expirations_analyzed": [c["expiration"] for c in chains],
            "total_call_volume": total_call_volume,
            "total_put_volume": total_put_volume,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "put_call_ratio": round(pc_ratio, 3),
            "put_call_oi_ratio": round(pc_oi_ratio, 3),
            "unusual_calls": unusual_calls[:5],
            "unusual_puts": unusual_puts[:5],
            "signal": signal,
            "summary": (
                f"P/C Volume Ratio: {pc_ratio:.2f}, P/C OI Ratio: {pc_oi_ratio:.2f}. "
                f"Call Vol: {total_call_volume:,}, Put Vol: {total_put_volume:,}. "
                f"Unusual calls: {len(unusual_calls)}, Unusual puts: {len(unusual_puts)}. "
                f"Signal: {signal}."
            ),
        }

    except Exception as e:
        logger.error("Failed to fetch options flow for %s: %s", ticker, e)
        return {"ticker": ticker, "signal": "ERROR", "summary": f"Error: {e}"}
