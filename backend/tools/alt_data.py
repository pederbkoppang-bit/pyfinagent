"""
Alternative data tool — Google Trends via pytrends.
Tracks search interest as a leading indicator for company momentum.
"""

import logging

logger = logging.getLogger(__name__)


def get_google_trends(ticker: str, company_name: str) -> dict:
    """
    Fetch Google Trends interest data for a company.
    Uses pytrends library (no API key required).
    """
    try:
        from pytrends.request import TrendReq

        pytrends = TrendReq(hl="en-US", tz=360)

        # Search for both ticker and company name
        keywords = [ticker, company_name] if company_name != ticker else [ticker]
        pytrends.build_payload(keywords[:5], cat=0, timeframe="today 12-m", geo="US")
        interest = pytrends.interest_over_time()

        if interest.empty:
            return {
                "ticker": ticker,
                "signal": "NO_DATA",
                "summary": "No Google Trends data available.",
            }

        # Get the ticker column
        col = ticker if ticker in interest.columns else interest.columns[0]
        values = interest[col].tolist()
        dates = [d.strftime("%Y-%m-%d") for d in interest.index]

        # Compute momentum
        recent = values[-4:] if len(values) >= 4 else values
        older = values[-12:-4] if len(values) >= 12 else values[:max(len(values) // 2, 1)]
        recent_avg = sum(recent) / len(recent) if recent else 0
        older_avg = sum(older) / len(older) if older else recent_avg

        momentum = ((recent_avg - older_avg) / max(older_avg, 1)) * 100
        current_interest = values[-1] if values else 0
        peak_interest = max(values) if values else 0

        signal = "NEUTRAL"
        if momentum > 30 and current_interest > 60:
            signal = "RISING_STRONG"
        elif momentum > 15:
            signal = "RISING"
        elif momentum < -30:
            signal = "DECLINING_STRONG"
        elif momentum < -15:
            signal = "DECLINING"

        # Sparse data for chart (weekly points)
        trend_data = [
            {"date": dates[i], "interest": values[i]}
            for i in range(0, len(dates), max(1, len(dates) // 52))
        ]

        return {
            "ticker": ticker,
            "company": company_name,
            "current_interest": current_interest,
            "peak_interest": peak_interest,
            "momentum_pct": round(momentum, 1),
            "trend_data": trend_data,
            "signal": signal,
            "summary": (
                f"Google Trends interest: {current_interest}/100 (peak: {peak_interest}). "
                f"Momentum: {momentum:+.1f}%. Signal: {signal}."
            ),
        }

    except ImportError:
        logger.warning("pytrends not installed -- skipping Google Trends analysis")
        return {
            "ticker": ticker,
            "signal": "UNAVAILABLE",
            "summary": "pytrends library not installed. Run: pip install pytrends",
        }
    except Exception as e:
        logger.error("Failed to fetch Google Trends for %s: %s", ticker, e)
        return {"ticker": ticker, "signal": "ERROR", "summary": f"Error: {e}"}
