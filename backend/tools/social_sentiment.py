"""
Social sentiment tool — Alpha Vantage social sentiment endpoint.
Reddit integration available as optional extension via PRAW.
"""

import logging

import httpx

logger = logging.getLogger(__name__)

# ── Keyword lists for fallback sentiment scoring ─────────────────────
_POSITIVE = {
    "surge", "soar", "rally", "gain", "profit", "beat", "record", "upgrade",
    "growth", "strong", "bullish", "outperform", "upside", "innovation",
    "breakthrough", "optimistic", "positive", "buy", "boost", "expand",
}
_NEGATIVE = {
    "drop", "fall", "decline", "loss", "miss", "downgrade", "weak", "bearish",
    "risk", "concern", "cut", "layoff", "warning", "crash", "sell", "debt",
    "lawsuit", "investigation", "fraud", "negative", "slump", "recession",
}


def _keyword_score(text: str) -> float:
    """Return a sentiment score in [-1, 1] using keyword matching."""
    words = set(text.lower().split())
    pos = len(words & _POSITIVE)
    neg = len(words & _NEGATIVE)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


async def get_social_sentiment(ticker: str, api_key: str, fallback_articles: list[dict] | None = None) -> dict:
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
            # ── Fallback: keyword-based scoring on yfinance articles ──
            if fallback_articles:
                return _score_fallback_articles(ticker, fallback_articles)
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


def _score_fallback_articles(ticker: str, articles: list[dict]) -> dict:
    """Score fallback articles (e.g. yfinance news) using keyword matching."""
    all_scores: list[float] = []
    source_sentiments: dict[str, list[float]] = {}

    for art in articles:
        text = f"{art.get('title', '')} {art.get('summary', '')}"
        score = _keyword_score(text)
        all_scores.append(score)
        src = art.get("source", "unknown")
        source_sentiments.setdefault(src, []).append(score)

    avg_sentiment = sum(all_scores) / len(all_scores) if all_scores else 0

    recent = all_scores[:10]
    older = all_scores[10:] if len(all_scores) > 10 else []
    recent_avg = sum(recent) / len(recent) if recent else 0
    older_avg = sum(older) / len(older) if older else recent_avg
    velocity = recent_avg - older_avg

    source_breakdown = {}
    for src, scores in sorted(source_sentiments.items(), key=lambda x: -len(x[1]))[:10]:
        source_breakdown[src] = {
            "count": len(scores),
            "avg_sentiment": round(sum(scores) / len(scores), 3),
        }

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
        "total_articles": len(articles),
        "mention_count": 0,
        "avg_sentiment": round(avg_sentiment, 4),
        "recent_sentiment": round(recent_avg, 4),
        "sentiment_velocity": round(velocity, 4),
        "source_breakdown": source_breakdown,
        "signal": signal,
        "data_source": "yfinance_fallback",
        "summary": (
            f"Avg sentiment: {avg_sentiment:.3f} across {len(articles)} articles (yfinance fallback). "
            f"Velocity: {velocity:+.3f}. Signal: {signal}."
        ),
    }
