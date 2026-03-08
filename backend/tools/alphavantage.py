"""
Alpha Vantage market intelligence tool.
Migrated from pyfinagent-app/tools/alphavantage.py — no Streamlit dependency.
"""

import logging
from collections import Counter
import httpx

logger = logging.getLogger(__name__)


async def get_market_intel(ticker: str, api_key: str) -> dict:
    """
    Fetches News & Sentiment from Alpha Vantage with dynamic competitor discovery.
    """
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit=50&apikey={api_key}"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

        feed = data.get("feed", [])
        if not feed:
            return {"error": "No news found or API limit reached.", "feed": []}

        # Dynamic competitor discovery
        mentioned_tickers = []
        for article in feed:
            for t_sent in article.get("ticker_sentiment", []):
                sym = t_sent.get("ticker")
                if sym and sym != ticker and sym not in ["FOREX", "CRYPTO"]:
                    mentioned_tickers.append(sym)

        likely_rivals = [t[0] for t in Counter(mentioned_tickers).most_common(5)]

        news_summary = []
        for article in feed[:15]:
            news_summary.append({
                "title": article.get("title"),
                "published": article.get("time_published"),
                "source": article.get("source"),
                "sentiment_score": article.get("overall_sentiment_score"),
                "sentiment_label": article.get("overall_sentiment_label"),
                "summary": article.get("summary"),
            })

        return {
            "target": ticker,
            "derived_competitors": likely_rivals,
            "sentiment_summary": news_summary,
        }

    except Exception as e:
        logger.error(f"Failed to fetch market intel from Alpha Vantage: {e}")
        return {"error": str(e)}
