"""
Social media velocity screener overlay — phase-28.15.

Lifts the velocity computation already present in `backend/tools/social_sentiment.py`
(line 95: `velocity = recent_avg - older_avg`) into the candidate screener tier.
Uses the existing Alpha Vantage NEWS_SENTIMENT endpoint that bundles Reddit,
Twitter/X, StockTwits, and financial blogs in a single API call.

Why Alpha Vantage and not StockTwits/ApeWisdom directly (per Researcher):
- StockTwits developer portal returns 403 / suspended as of 2026
- ApeWisdom is Reddit-only without an SLA
- Alpha Vantage bundles multiple sources in one call (cross-source convergence)

Per supplement Gap 2 + DNUT July 2025: 500% StockTwits mention spike preceded
90% pre-market surge — velocity spikes ARE alpha-positive when sufficiently
high (>=0.10) AND backed by adequate mention count (>=3, noise guard).

Cost: free tier Alpha Vantage = 5 req/min. Bounded to top 2*paper_screen_top_n
(~20 candidates), throttled at 0.5s per ticker.

Graceful degradation: AV rate limit / no API key / empty results → empty dict →
`apply_social_velocity_to_score` is identity.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

_CONCURRENCY = 2  # AV free tier is strict — keep concurrency low
_PER_TICKER_SLEEP_S = 0.5


class SocialVelocitySignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(description="Uppercase equity ticker.")
    velocity: float = Field(description="recent_avg sentiment - older_avg (typically [-1, 1] range).")
    avg_sentiment: float = Field(description="Average sentiment over all articles in the AV feed.")
    mention_count: int = Field(description="Count of ticker-specific mentions in AV feed.")
    source_count: int = Field(description="Distinct source types in the AV feed (cross-source convergence proxy).")
    boost_multiplier: float = Field(description="Final multiplier on composite_score. 1.0 = no boost.")
    boost_tier: str = Field(description="strong | moderate | none.")


def _classify_boost(
    velocity: float, mention_count: int,
    min_threshold: float, min_mentions: int, strong_threshold: float,
    strong_boost: float, moderate_boost: float,
) -> tuple[float, str]:
    if mention_count < min_mentions:
        return 1.0, "none"
    if velocity >= strong_threshold:
        return 1.0 + strong_boost, "strong"
    if velocity >= min_threshold:
        return 1.0 + moderate_boost, "moderate"
    return 1.0, "none"


async def _fetch_one_velocity(
    ticker: str,
    av_key: str,
    min_threshold: float, min_mentions: int, strong_threshold: float,
    strong_boost: float, moderate_boost: float,
    sem: asyncio.Semaphore,
) -> Optional[SocialVelocitySignal]:
    async with sem:
        try:
            from backend.tools.social_sentiment import get_social_sentiment
            result = await get_social_sentiment(ticker, av_key, fallback_articles=None)
        except Exception as e:
            logger.debug("social_velocity_screen: %s fetch failed: %s", ticker, e)
            return None
        await asyncio.sleep(_PER_TICKER_SLEEP_S)
        if not isinstance(result, dict) or result.get("signal") in (None, "NO_DATA", "ERROR"):
            return None
        # social_sentiment.py exposes the field as `sentiment_velocity` (line 122)
        velocity_val = result.get("sentiment_velocity")
        if velocity_val is None:
            velocity_val = result.get("velocity")  # back-compat fallback
        if velocity_val is None:
            return None
        try:
            velocity_val = float(velocity_val)
        except (TypeError, ValueError):
            return None
        mention_count = int(result.get("mention_count") or 0)
        avg_sentiment = float(result.get("avg_sentiment") or 0)
        source_breakdown = result.get("source_breakdown") or {}
        source_count = len(source_breakdown) if isinstance(source_breakdown, dict) else 0

        boost, tier = _classify_boost(
            velocity_val, mention_count, min_threshold, min_mentions,
            strong_threshold, strong_boost, moderate_boost,
        )
        if tier == "none":
            return None
        return SocialVelocitySignal(
            ticker=ticker.upper(),
            velocity=round(velocity_val, 4),
            avg_sentiment=round(avg_sentiment, 4),
            mention_count=mention_count,
            source_count=source_count,
            boost_multiplier=round(boost, 4),
            boost_tier=tier,
        )


async def fetch_social_velocity_signals(
    tickers: list[str],
    min_threshold: float = 0.10,
    min_mentions: int = 3,
    strong_threshold: float = 0.20,
    strong_boost: float = 0.06,
    moderate_boost: float = 0.03,
) -> dict[str, SocialVelocitySignal]:
    """Per-ticker social velocity classification via Alpha Vantage NEWS_SENTIMENT.

    Returns one entry per ticker with qualifying velocity (>= min_threshold) AND
    sufficient mentions (>= min_mentions). Empty dict if no AV key, all calls fail,
    or no tickers qualify.
    """
    if not tickers:
        return {}
    settings = get_settings()
    av_key = getattr(settings, "alphavantage_api_key", "") or ""
    if not av_key:
        logger.info("social_velocity_screen: no alphavantage_api_key set; signal disabled")
        return {}
    if hasattr(av_key, "get_secret_value"):
        try:
            av_key = av_key.get_secret_value()
        except Exception:
            pass
    sem = asyncio.Semaphore(_CONCURRENCY)
    results = await asyncio.gather(
        *(
            _fetch_one_velocity(
                t, av_key, min_threshold, min_mentions, strong_threshold,
                strong_boost, moderate_boost, sem,
            )
            for t in tickers
        ),
        return_exceptions=False,
    )
    out: dict[str, SocialVelocitySignal] = {}
    for sig in results:
        if sig is not None:
            out[sig.ticker] = sig
    logger.info(
        "social_velocity_screen: %d/%d tickers flagged (strong>=%.2f +%g; moderate>=%.2f +%g; min mentions=%d)",
        len(out), len(tickers), strong_threshold, strong_boost, min_threshold, moderate_boost, min_mentions,
    )
    return out


def apply_social_velocity_to_score(
    base_score: float,
    ticker: Optional[str],
    signals: Optional[dict[str, SocialVelocitySignal]],
) -> float:
    """Multiply score by signals[ticker].boost_multiplier. Identity if missing."""
    if not signals or not ticker:
        return base_score
    sig = signals.get(ticker.upper())
    if sig is None:
        return base_score
    from backend.services.overlay_math import sign_safe_mult  # phase-69.3 sign-safe (default-OFF byte-identical)
    return sign_safe_mult(base_score, sig.boost_multiplier)
