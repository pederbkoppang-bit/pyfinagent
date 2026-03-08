"""
Signals API — on-demand enrichment data for individual tickers.
Provides insider trading, options flow, social sentiment, patents,
earnings tone, macro indicators, alt data, and sector analysis.
"""

import asyncio
import logging

from fastapi import APIRouter, Depends

from backend.config.settings import Settings, get_settings
from backend.tools import (
    alt_data,
    anomaly_detector,
    earnings_tone,
    fred_data,
    monte_carlo,
    nlp_sentiment,
    options_flow,
    patent_tracker,
    sec_insider,
    sector_analysis,
    social_sentiment,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/signals", tags=["signals"])


def _yf_news_to_articles(yf_news: list[dict]) -> list[dict]:
    """Normalize yfinance .news list into the AV-compatible article format."""
    articles = []
    for item in yf_news or []:
        content = item.get("content", item)  # nested under 'content'
        title = content.get("title", "")
        summary = content.get("summary", "") or title
        provider = content.get("provider", {})
        source = provider.get("displayName", "unknown") if isinstance(provider, dict) else "unknown"
        canon = content.get("canonicalUrl", {})
        url = canon.get("url", "") if isinstance(canon, dict) else ""
        articles.append({
            "title": title,
            "summary": summary,
            "source": source,
            "url": url,
            "overall_sentiment_score": None,
        })
    return articles


@router.get("/{ticker}")
async def get_all_signals(ticker: str, settings: Settings = Depends(get_settings)):
    """Fetch all enrichment signals for a ticker in parallel."""
    ticker = ticker.upper()

    async def _safe(coro_or_func, label, *args):
        try:
            if asyncio.iscoroutinefunction(coro_or_func):
                return await coro_or_func(*args)
            else:
                return await asyncio.to_thread(coro_or_func, *args)
        except Exception as e:
            logger.warning("Signal %s failed for %s: %s", label, ticker, e)
            return {"signal": "ERROR", "summary": str(e)}

    import yfinance as yf
    from backend.tools import alphavantage

    info = await asyncio.to_thread(lambda: yf.Ticker(ticker).info)
    company_name = info.get("longName", ticker)

    # Fetch AV articles first so NLP sentiment has data to analyze
    av_data = await _safe(alphavantage.get_market_intel, "av_articles", ticker, settings.alphavantage_api_key)
    articles = av_data.get("sentiment_summary", []) if isinstance(av_data, dict) else []

    # Fallback: use yfinance news when AV returns nothing
    fallback_articles: list[dict] = []
    if not articles:
        yf_news = await asyncio.to_thread(lambda: yf.Ticker(ticker).news)
        fallback_articles = _yf_news_to_articles(yf_news)
        articles = fallback_articles  # feed into NLP
        logger.info("AV empty for %s — using %d yfinance articles as fallback", ticker, len(fallback_articles))

    insider, options, social, patent, earnings, fred, alt, sector, nlp, anomalies, mc = await asyncio.gather(
        _safe(sec_insider.get_insider_trades, "insider", ticker),
        _safe(options_flow.get_options_flow, "options", ticker),
        _safe(social_sentiment.get_social_sentiment, "social", ticker, settings.alphavantage_api_key, fallback_articles),
        _safe(patent_tracker.get_patent_data, "patent", company_name, ticker, 3, settings.patentsview_api_key),
        _safe(earnings_tone.get_earnings_tone, "earnings", ticker, settings.api_ninjas_key, 4, settings.gcs_bucket_name),
        _safe(fred_data.get_macro_indicators, "fred", settings.fred_api_key),
        _safe(alt_data.get_google_trends, "alt_data", ticker, company_name),
        _safe(sector_analysis.get_sector_analysis, "sector", ticker),
        _safe(nlp_sentiment.get_nlp_sentiment, "nlp_sentiment", ticker, articles, settings.gcp_project_id, settings.gcp_location),
        _safe(anomaly_detector.get_anomaly_scan, "anomalies", ticker),
        _safe(monte_carlo.get_monte_carlo_simulation, "monte_carlo", ticker),
    )

    return {
        "ticker": ticker,
        "company_name": company_name,
        "insider": insider,
        "options": options,
        "social_sentiment": social,
        "patent": patent,
        "earnings_tone": earnings,
        "fred_macro": fred,
        "alt_data": alt,
        "sector": sector,
        "nlp_sentiment": nlp,
        "anomalies": anomalies,
        "monte_carlo": mc,
    }


@router.get("/{ticker}/insider")
async def get_insider(ticker: str):
    return await sec_insider.get_insider_trades(ticker.upper())


@router.get("/{ticker}/options")
async def get_options(ticker: str):
    return await asyncio.to_thread(options_flow.get_options_flow, ticker.upper())


@router.get("/{ticker}/sentiment")
async def get_sentiment(ticker: str, settings: Settings = Depends(get_settings)):
    import yfinance as yf
    result = await social_sentiment.get_social_sentiment(ticker.upper(), settings.alphavantage_api_key)
    if result.get("signal") == "NO_DATA":
        yf_news = await asyncio.to_thread(lambda: yf.Ticker(ticker.upper()).news)
        fallback = _yf_news_to_articles(yf_news)
        if fallback:
            result = await social_sentiment.get_social_sentiment(ticker.upper(), settings.alphavantage_api_key, fallback)
    return result


@router.get("/{ticker}/patents")
async def get_patents(ticker: str, settings: Settings = Depends(get_settings)):
    import yfinance as yf
    info = await asyncio.to_thread(lambda: yf.Ticker(ticker.upper()).info)
    company_name = info.get("longName", ticker.upper())
    return await patent_tracker.get_patent_data(company_name, ticker.upper(), api_key=settings.patentsview_api_key)


@router.get("/{ticker}/earnings-tone")
async def get_earnings(ticker: str, settings: Settings = Depends(get_settings)):
    return await earnings_tone.get_earnings_tone(ticker.upper(), settings.api_ninjas_key, bucket_name=settings.gcs_bucket_name)


@router.get("/macro/indicators")
async def get_macro(settings: Settings = Depends(get_settings)):
    return await fred_data.get_macro_indicators(settings.fred_api_key)


@router.get("/{ticker}/alt-data")
async def get_alt(ticker: str):
    import yfinance as yf
    info = await asyncio.to_thread(lambda: yf.Ticker(ticker.upper()).info)
    company_name = info.get("longName", ticker.upper())
    return await asyncio.to_thread(alt_data.get_google_trends, ticker.upper(), company_name)


@router.get("/{ticker}/sector")
async def get_sector(ticker: str):
    return await asyncio.to_thread(sector_analysis.get_sector_analysis, ticker.upper())


@router.get("/{ticker}/nlp-sentiment")
async def get_nlp(ticker: str, settings: Settings = Depends(get_settings)):
    """Transformer-based NLP sentiment via Vertex AI embeddings."""
    import yfinance as yf
    from backend.tools import alphavantage
    av_data = await alphavantage.get_market_intel(ticker.upper(), settings.alphavantage_api_key)
    articles = av_data.get("sentiment_summary", [])
    if not articles:
        yf_news = await asyncio.to_thread(lambda: yf.Ticker(ticker.upper()).news)
        articles = _yf_news_to_articles(yf_news)
    return await nlp_sentiment.get_nlp_sentiment(
        ticker.upper(), articles, settings.gcp_project_id, settings.gcp_location
    )


@router.get("/{ticker}/anomalies")
async def get_anomalies(ticker: str):
    """Multi-dimensional anomaly detection scan."""
    return await asyncio.to_thread(anomaly_detector.get_anomaly_scan, ticker.upper())


@router.get("/{ticker}/monte-carlo")
async def get_monte_carlo(ticker: str):
    """Monte Carlo VaR simulation."""
    return await asyncio.to_thread(monte_carlo.get_monte_carlo_simulation, ticker.upper())
