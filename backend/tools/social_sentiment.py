"""
Social sentiment tool — Alpha Vantage social sentiment endpoint.
Reddit integration available as optional extension via PRAW.
"""

import logging

import httpx

logger = logging.getLogger(__name__)


async def get_social_sentiment(ticker: str, api_key: str) -> dict:
    """
    Fetch social-media sentiment from Alpha Vantage.
    Covers Reddit, Twitter/X, StockTwits, and financial blogs.
    """
    url = (
        f"https://www.alphavantage.co/query?"
        f"function=NEWS_SENTIMENT&tickers={ticker}&topics=technology,finance"
        f"&sort=RELEVANCE&limit=50&apikey={api_key}"
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        feed = data.get("feed", [])
        if not feed:
            return {
                "ticker": ticker,
                "signal": "NO_DATA",
                "summary": "No social sentiment data available.",
            }

        # Categorize by source type
        source_sentiments: dict[str, list[float]] = {}
        all_scores = []
        mention_count = 0

        for article in feed:
            source = article.get("source", "unknown")
            score = article.get("overall_sentiment_score")
            if score is not None:
                score = float(score)
                all_scores.append(score)
                source_sentiments.setdefault(source, []).append(score)

            # Count ticker-specific mentions
            for ts in article.get("ticker_sentiment", []):
                if ts.get("ticker") == ticker:
                    mention_count += 1

        avg_sentiment = sum(all_scores) / len(all_scores) if all_scores else 0

        # Compute sentiment velocity (are recent articles more positive?)
        recent = all_scores[:10]
        older = all_scores[10:] if len(all_scores) > 10 else []
        recent_avg = sum(recent) / len(recent) if recent else 0
        older_avg = sum(older) / len(older) if older else recent_avg
        velocity = recent_avg - older_avg

        # Source breakdown
        source_breakdown = {}
        for src, scores in sorted(source_sentiments.items(), key=lambda x: -len(x[1]))[:10]:
            source_breakdown[src] = {
                "count": len(scores),
                "avg_sentiment": round(sum(scores) / len(scores), 3),
            }

        # Signal
        signal = "NEUTRAL"
        if avg_sentiment > 0.25 and velocity > 0.05:
            signal = "STRONG_BULLISH"
        elif avg_sentiment > 0.15:
            signal = "BULLISH"
        elif avg_sentiment < -0.25 and velocity < -0.05:
            signal = "STRONG_BEARISH"
        elif avg_sentiment < -0.15:
            signal = "BEARISH"

        return {
            "ticker": ticker,
            "total_articles": len(feed),
            "mention_count": mention_count,
            "avg_sentiment": round(avg_sentiment, 4),
            "recent_sentiment": round(recent_avg, 4),
            "sentiment_velocity": round(velocity, 4),
            "source_breakdown": source_breakdown,
            "signal": signal,
            "summary": (
                f"Avg sentiment: {avg_sentiment:.3f} across {len(feed)} articles. "
                f"Velocity: {velocity:+.3f} (recent vs older). "
                f"Mentions: {mention_count}. Signal: {signal}."
            ),
        }

    except Exception as e:
        logger.error("Failed to fetch social sentiment for %s: %s", ticker, e)
        return {"ticker": ticker, "signal": "ERROR", "summary": f"Error: {e}"}
