"""
Analyst EPS revision-breadth overlay — phase-28.1.

Pulls `yf.Ticker(t).upgrades_downgrades` per top-N candidate ticker, filters to
`Action in (up, down)` within a 100-day lookback (Mill Street canonical), and
computes a revision-breadth score per ticker:

    breadth = (n_up - n_down) / (n_up + n_down)

When `|breadth| > threshold` (default 0.10), the screener's composite score is
multiplied by `(1 + breadth * weight)` (default weight 0.15). A full +1.0
breadth yields a +15% boost; a full -1.0 breadth yields a -15% penalty.

Source: Mill Street Research 19-year backtest (Sharpe ~1.60 combined with price
momentum; t=2.93). Confirmed by arXiv 2502.20489 (sell-side analyst reports
generate 68bps/month alpha) and arXiv 2410.20597 (analyst-network alpha).

Cost: $0 LLM. Per-ticker yfinance HTTP, throttled at 0.3s/call with a
Semaphore(4) concurrency cap. Top-N (~10-30 tickers) is fast (~5-10s).

Graceful degradation: returns empty dict on any error; `apply_revisions_to_score`
is identity when no signal is provided, preserving the cycle.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

_DEFAULT_LOOKBACK_DAYS = 100
_DEFAULT_MIN_ANALYSTS = 3
_DEFAULT_THRESHOLD = 0.10
_DEFAULT_WEIGHT = 0.15
_CONCURRENCY = 4
_THROTTLE_S = 0.3


class RevisionSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(description="Uppercase equity ticker.")
    breadth: float = Field(
        description="(n_up - n_down) / (n_up + n_down). Range [-1.0, 1.0]. Positive = net upgrades.",
    )
    n_up: int = Field(description="Count of analyst upgrade actions in the lookback window.")
    n_down: int = Field(description="Count of analyst downgrade actions in the lookback window.")
    n_total: int = Field(description="n_up + n_down (used for min-analysts guard).")
    lookback_days: int = Field(description="Lookback window applied (typically 100).")


def _compute_breadth(df, lookback_days: int) -> Optional[tuple[float, int, int]]:
    """Compute (breadth, n_up, n_down) from a yfinance upgrades_downgrades DataFrame.

    yfinance returns a `GradeDate`-indexed DataFrame with naive datetime64[s].
    `Action` column values: {up, down, main, init, reit}. Only `up`/`down` count
    toward revision breadth -- `main` (reiteration) and `init`/`reit` are signal-free.

    Returns None if no actionable rows in the window or the DataFrame shape is
    unexpected.
    """
    try:
        if df is None or len(df) == 0:
            return None
        if "Action" not in df.columns:
            return None
        # yfinance index is tz-naive datetime64[s]; use a tz-naive cutoff to avoid
        # the "Invalid comparison between dtype=datetime64[s] and datetime" TypeError.
        cutoff = datetime.now() - timedelta(days=lookback_days)
        try:
            mask = df.index >= cutoff
        except TypeError:
            # Fallback: re-convert the index to tz-naive then compare.
            naive_idx = df.index.tz_convert(None) if getattr(df.index, "tz", None) else df.index
            mask = naive_idx >= cutoff
        recent = df[mask]
        if len(recent) == 0:
            return None
        actions = recent["Action"].astype(str).str.lower()
        n_up = int((actions == "up").sum())
        n_down = int((actions == "down").sum())
        total = n_up + n_down
        if total == 0:
            return None
        breadth = (n_up - n_down) / total
        return breadth, n_up, n_down
    except Exception as e:
        logger.debug("revision-breadth compute failed: %s", e)
        return None


async def _fetch_one(ticker: str, lookback_days: int, min_analysts: int,
                     sem: asyncio.Semaphore) -> Optional[RevisionSignal]:
    async with sem:
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed; analyst_revisions skipped")
            return None
        try:
            t_obj = yf.Ticker(ticker)
            df = await asyncio.to_thread(lambda: t_obj.upgrades_downgrades)
        except Exception as e:
            logger.debug("yfinance upgrades_downgrades failed for %s: %s", ticker, e)
            return None
        await asyncio.sleep(_THROTTLE_S)
        result = _compute_breadth(df, lookback_days)
        if result is None:
            return None
        breadth, n_up, n_down = result
        total = n_up + n_down
        if total < min_analysts:
            return None
        return RevisionSignal(
            ticker=ticker.upper(),
            breadth=round(breadth, 4),
            n_up=n_up,
            n_down=n_down,
            n_total=total,
            lookback_days=lookback_days,
        )


async def fetch_revision_signals(
    tickers: list[str],
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    min_analysts: int = _DEFAULT_MIN_ANALYSTS,
) -> dict[str, RevisionSignal]:
    """Fetch analyst revision-breadth signals for the given tickers.

    Args:
        tickers: list of ticker symbols (typically the top-N candidates).
        lookback_days: window over which to count up/down actions (default 100).
        min_analysts: minimum (n_up + n_down) for the signal to be returned.

    Returns:
        dict[ticker, RevisionSignal] -- one entry per ticker that had >= min_analysts
        actions in the window. Empty dict if all tickers fail or no signals qualify.
    """
    if not tickers:
        return {}
    sem = asyncio.Semaphore(_CONCURRENCY)
    results = await asyncio.gather(
        *(_fetch_one(t, lookback_days, min_analysts, sem) for t in tickers),
        return_exceptions=False,
    )
    out: dict[str, RevisionSignal] = {}
    for sig in results:
        if sig is not None:
            out[sig.ticker] = sig
    logger.info(
        "analyst_revisions: %d/%d tickers produced signals (lookback=%dd, min_analysts=%d)",
        len(out), len(tickers), lookback_days, min_analysts,
    )
    return out


def apply_revisions_to_score(
    base_score: float,
    ticker: Optional[str],
    revision_signals: Optional[dict[str, RevisionSignal]],
    threshold: float = _DEFAULT_THRESHOLD,
    weight: float = _DEFAULT_WEIGHT,
) -> float:
    """Apply revision-breadth multiplier to a candidate's composite score.

    No-op when no signal exists for the ticker or |breadth| <= threshold (deadband).
    Otherwise: score *= (1 + breadth * weight).

    Examples:
        breadth = +0.50, weight = 0.15 -> score *= 1.075 (+7.5% boost)
        breadth = -0.30, weight = 0.15 -> score *= 0.955 (-4.5% penalty)
        breadth = +0.08, weight = 0.15 -> score unchanged (below threshold)
    """
    if not revision_signals or not ticker:
        return base_score
    sig = revision_signals.get(ticker.upper())
    if sig is None:
        return base_score
    if abs(sig.breadth) <= threshold:
        return base_score
    from backend.services.overlay_math import sign_safe_mult  # phase-69.3 sign-safe (default-OFF byte-identical)
    return sign_safe_mult(base_score, 1.0 + sig.breadth * weight)
