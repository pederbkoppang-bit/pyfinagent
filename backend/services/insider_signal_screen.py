"""
Opportunistic insider-buying screener overlay — phase-28.10.

Wraps `backend.tools.sec_insider.get_insider_trades` with the Cohen-Malloy-Pomorski
(2012) opportunistic/routine classifier:

    ROUTINE       = insider traded in the SAME calendar month in EACH of the
                    3 prior consecutive years.
    OPPORTUNISTIC = all other trades by insiders with at least 3 years of history.
    UNKNOWN       = insider has fewer than 3 years of history (cold-start guard).

Per CMP: OPPORTUNISTIC trades earn 82bps/month abnormal return (value-weighted);
ROUTINE earn ~0.

Aggregates per-ticker opportunistic-BUY dollar value over the last
`insider_signal_window_days` (default 30). Tickers with material aggregate
($500K+ moderate / $2M+ strong) get a small score multiplier.

Cost: SEC EDGAR is free + rate-limited. Per-ticker `get_insider_trades` fetch with
48-month lookback. Bounded to top-N candidates (typically ~20), not full universe.

Graceful degradation: empty dict on any error; `apply_insider_signal_to_score`
is identity when no signal exists for ticker.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

_CONCURRENCY = 3  # SEC EDGAR rate-limit safety
_PER_TICKER_SLEEP_S = 0.5


class InsiderSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(description="Uppercase equity ticker.")
    n_opportunistic_buys: int = Field(description="Count of opportunistic-BUY transactions in window.")
    aggregate_usd: float = Field(description="Sum of opportunistic-BUY dollar values in window.")
    n_unique_insiders: int = Field(description="Count of distinct insiders contributing.")
    boost_multiplier: float = Field(description="Final multiplier to apply (1.0 = no boost).")
    boost_tier: str = Field(description="strong | moderate | none.")


def _is_routine(insider_history: list[dict], target_month: int, target_year: int) -> bool:
    """CMP rule: routine = insider traded in same calendar month in each of 3 prior years.

    `insider_history` is the full list of {date, type, value, ...} for this insider.
    A trade in (target_year, target_month) is ROUTINE if there exists at least one
    trade in (target_year - k, target_month) for k = 1, 2, 3.
    """
    months_traded: set[tuple[int, int]] = set()
    for t in insider_history:
        try:
            d = datetime.strptime(t.get("date", ""), "%Y-%m-%d").date()
            months_traded.add((d.year, d.month))
        except (ValueError, TypeError):
            continue
    for offset in (1, 2, 3):
        if (target_year - offset, target_month) not in months_traded:
            return False
    return True


def _has_min_history(insider_history: list[dict], min_years: int = 3) -> bool:
    """An insider needs at least `min_years` of trade history for CMP classification."""
    if not insider_history:
        return False
    dates: list[datetime] = []
    for t in insider_history:
        try:
            dates.append(datetime.strptime(t.get("date", ""), "%Y-%m-%d"))
        except (ValueError, TypeError):
            continue
    if not dates:
        return False
    span = (max(dates) - min(dates)).days
    return span >= 365 * min_years


def _classify_trades(trades: list[dict]) -> list[dict]:
    """Annotate each trade with cmp_class in {'routine','opportunistic','unknown'}."""
    by_insider: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        key = (t.get("insider_name") or "").strip().lower() or "<anon>"
        by_insider[key].append(t)

    annotated: list[dict] = []
    for insider, hist in by_insider.items():
        has_enough_history = _has_min_history(hist, min_years=3)
        for t in hist:
            t = dict(t)
            try:
                d = datetime.strptime(t.get("date", ""), "%Y-%m-%d").date()
                if not has_enough_history:
                    t["cmp_class"] = "unknown"
                elif _is_routine(hist, d.month, d.year):
                    t["cmp_class"] = "routine"
                else:
                    t["cmp_class"] = "opportunistic"
            except (ValueError, TypeError):
                t["cmp_class"] = "unknown"
            annotated.append(t)
    return annotated


def _classify_boost(aggregate_usd: float, min_usd: float, strong_usd: float,
                     strong_boost: float, moderate_boost: float) -> tuple[float, str]:
    if aggregate_usd >= strong_usd:
        return 1.0 + strong_boost, "strong"
    if aggregate_usd >= min_usd:
        return 1.0 + moderate_boost, "moderate"
    return 1.0, "none"


async def _fetch_one(
    ticker: str,
    lookback_months: int,
    window_days: int,
    min_usd: float,
    strong_usd: float,
    strong_boost: float,
    moderate_boost: float,
    sem: asyncio.Semaphore,
) -> Optional[InsiderSignal]:
    async with sem:
        try:
            from backend.tools.sec_insider import get_insider_trades
        except ImportError:
            return None
        try:
            result = await get_insider_trades(ticker, months=lookback_months)
        except Exception as e:
            logger.debug("insider_signal_screen: %s get_insider_trades failed: %s", ticker, e)
            return None
        trades = result.get("trades", []) if isinstance(result, dict) else []
        await asyncio.sleep(_PER_TICKER_SLEEP_S)
        if not trades:
            return None

        annotated = _classify_trades(trades)

        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).strftime("%Y-%m-%d")
        recent_opportunistic_buys = [
            t for t in annotated
            if t.get("cmp_class") == "opportunistic"
            and (t.get("type") or "").upper() == "BUY"
            and (t.get("date") or "") >= cutoff
        ]
        if not recent_opportunistic_buys:
            return None

        aggregate = float(sum((t.get("value") or 0) for t in recent_opportunistic_buys))
        if aggregate < min_usd:
            return None

        insiders = {(t.get("insider_name") or "").lower() for t in recent_opportunistic_buys}
        boost, tier = _classify_boost(aggregate, min_usd, strong_usd, strong_boost, moderate_boost)
        return InsiderSignal(
            ticker=ticker.upper(),
            n_opportunistic_buys=len(recent_opportunistic_buys),
            aggregate_usd=round(aggregate, 2),
            n_unique_insiders=len(insiders),
            boost_multiplier=round(boost, 4),
            boost_tier=tier,
        )


async def fetch_insider_signals(
    tickers: list[str],
    lookback_months: int = 48,
    window_days: int = 30,
    min_usd: float = 500_000.0,
    strong_usd: float = 2_000_000.0,
    strong_boost: float = 0.07,
    moderate_boost: float = 0.04,
) -> dict[str, InsiderSignal]:
    """Per-ticker CMP-classified opportunistic insider-buy detection.

    Returns one entry per ticker with qualifying (>=min_usd aggregate) opportunistic
    BUY activity in the lookback window. Empty dict if no qualifying signals or all
    fetches fail.
    """
    if not tickers:
        return {}
    sem = asyncio.Semaphore(_CONCURRENCY)
    results = await asyncio.gather(
        *(
            _fetch_one(t, lookback_months, window_days, min_usd, strong_usd, strong_boost, moderate_boost, sem)
            for t in tickers
        ),
        return_exceptions=False,
    )
    out: dict[str, InsiderSignal] = {}
    for sig in results:
        if sig is not None:
            out[sig.ticker] = sig
    logger.info(
        "insider_signal_screen: %d/%d tickers flagged (strong>=$%.0f +%g; moderate>=$%.0f +%g)",
        len(out), len(tickers), strong_usd, strong_boost, min_usd, moderate_boost,
    )
    return out


def apply_insider_signal_to_score(
    base_score: float,
    ticker: Optional[str],
    signals: Optional[dict[str, InsiderSignal]],
) -> float:
    """Multiply score by signals[ticker].boost_multiplier. Identity if missing."""
    if not signals or not ticker:
        return base_score
    sig = signals.get(ticker.upper())
    if sig is None:
        return base_score
    return base_score * sig.boost_multiplier
