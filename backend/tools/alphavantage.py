"""
Alpha Vantage market intelligence tool.
Migrated from pyfinagent-app/tools/alphavantage.py — no Streamlit dependency.
"""

import logging
from collections import Counter
import httpx

logger = logging.getLogger(__name__)


def _yfinance_fallback(ticker: str) -> list[dict]:
    """Fetch news from yfinance as fallback when AV is unavailable."""
    try:
        import yfinance as yf
        yf_news = yf.Ticker(ticker).news or []
        articles = []
        for item in yf_news:
            content = item.get("content", item)
            title = content.get("title", "")
            summary = content.get("summary", "") or title
            provider = content.get("provider", {})
            src = provider.get("displayName", "unknown") if isinstance(provider, dict) else "unknown"
            articles.append({
                "title": title,
                "published": content.get("pubDate", ""),
                "source": src,
                "sentiment_score": None,
                "sentiment_label": None,
                "summary": summary,
            })
        return articles
    except Exception as e:
        logger.warning("yfinance news fallback failed: %s", e)
        return []


async def get_market_intel(ticker: str, api_key: str) -> dict:
    """
    Fetches News & Sentiment from Alpha Vantage with dynamic competitor discovery.
    Falls back to yfinance news if AV is rate-limited or returns no data.
    """
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit=50&apikey={api_key}"
    rate_limited = False
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

        # Detect rate limit responses
        if "Information" in data or "Note" in data:
            msg = data.get("Information", data.get("Note", ""))
            logger.warning("Alpha Vantage rate limit for %s: %s", ticker, msg)
            rate_limited = True
            feed = []
        else:
            feed = data.get("feed", [])

    except Exception as e:
        logger.error("Failed to fetch market intel from Alpha Vantage: %s", e)
        rate_limited = True
        feed = []

    # Fallback to yfinance news
    source = "alphavantage"
    if not feed:
        yf_articles = _yfinance_fallback(ticker)
        if yf_articles:
            logger.info("AV unavailable for %s — using %d yfinance articles", ticker, len(yf_articles))
            return {
                "target": ticker,
                "derived_competitors": [],
                "sentiment_summary": yf_articles,
                "source": "yfinance",
                "rate_limited": rate_limited,
            }
        error_msg = "Alpha Vantage rate limit reached (25 req/day free tier). No yfinance articles available." if rate_limited else "No news found for this ticker."
        return {"error": error_msg, "feed": [], "rate_limited": rate_limited}

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
        "source": "alphavantage",
        "rate_limited": False,
    }
