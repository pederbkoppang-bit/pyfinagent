"""
Options-flow OI-surge screener overlay — phase-28.9.

Detects near-expiry OUT-OF-THE-MONEY CALL OPTIONS with abnormal volume relative
to both (a) the chain's rolling avg volume per strike and (b) open interest.
Per Wayne State / Journal of Portfolio Management research: this specific
configuration (near-expiry + OTM + call + volume surge) is predictive of forward
stock returns; generic large options trades are NOT.

Cost: $0 LLM. Per-ticker `yf.Ticker.option_chain` HTTP call — bounded to top-N
candidates (10-30 typically), throttled with `Semaphore(4)` + 0.3s sleep.

Graceful degradation: returns empty dict on any error;
`apply_options_surge_to_score` is identity when no signal provided.

Signal magnitudes (defaults; settings-driven):
    OTM strike > spot * 1.01
    DTE in [2, 45]
    volume > max(5x avg-per-strike, 3x OI)
    boost: +6% (>=2 surge strikes), +3% (exactly 1)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

_CONCURRENCY = 4
_THROTTLE_S = 0.3


class OptionsSurgeSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(description="Uppercase equity ticker.")
    n_surges: int = Field(description="Count of strike rows matching the surge predicate within DTE window.")
    max_vol_oi_ratio: float = Field(description="Highest volume/OI ratio observed among the surge strikes.")
    max_vol_avg_ratio: float = Field(description="Highest volume / chain-avg-volume ratio observed.")
    boost_multiplier: float = Field(description="Final multiplier to apply to composite_score. 1.0 = no surge.")
    surge_strikes: list[float] = Field(default_factory=list, description="Strikes that triggered the surge.")


def _classify_boost(n_surges: int, strong_boost: float, moderate_boost: float) -> float:
    if n_surges >= 2:
        return 1.0 + strong_boost
    if n_surges == 1:
        return 1.0 + moderate_boost
    return 1.0


async def _fetch_one(
    ticker: str,
    otm_threshold: float,
    dte_min: int,
    dte_max: int,
    vol_avg_mult: float,
    vol_oi_mult: float,
    strong_boost: float,
    moderate_boost: float,
    sem: asyncio.Semaphore,
) -> Optional[OptionsSurgeSignal]:
    async with sem:
        try:
            import yfinance as yf
        except ImportError:
            return None
        try:
            stock = yf.Ticker(ticker)
            spot = await asyncio.to_thread(lambda: float(stock.fast_info.last_price))
            if spot <= 0:
                return None
            expirations = await asyncio.to_thread(lambda: list(stock.options or []))
        except Exception as e:
            logger.debug("options_flow_screen: %s fetch failed: %s", ticker, e)
            return None
        if not expirations:
            return None

        today = datetime.now(timezone.utc).date()
        surges: list[tuple[float, float, float]] = []
        for exp_str in expirations[:6]:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            dte = (exp_date - today).days
            if dte < dte_min or dte > dte_max:
                continue
            try:
                chain = await asyncio.to_thread(lambda e=exp_str: stock.option_chain(e))
                calls_df = chain.calls
            except Exception as e:
                logger.debug("options_flow_screen: %s exp %s chain failed: %s", ticker, exp_str, e)
                continue
            if calls_df is None or len(calls_df) == 0:
                continue
            volumes = calls_df["volume"].fillna(0)
            avg_vol = float(volumes.mean()) if len(volumes) else 0.0
            if avg_vol <= 0:
                continue
            for _, row in calls_df.iterrows():
                strike = float(row.get("strike") or 0)
                vol = float(row.get("volume") or 0)
                oi = float(row.get("openInterest") or 0)
                if strike <= 0 or vol <= 0:
                    continue
                if strike < spot * otm_threshold:
                    continue
                vol_avg_ratio = vol / avg_vol if avg_vol > 0 else 0
                vol_oi_ratio = vol / oi if oi > 0 else float("inf") if vol > 50 else 0
                if vol_avg_ratio >= vol_avg_mult and vol_oi_ratio >= vol_oi_mult:
                    surges.append((strike, vol_avg_ratio, vol_oi_ratio))
        await asyncio.sleep(_THROTTLE_S)

        if not surges:
            return None
        n = len(surges)
        max_vo = max((min(s[2], 1e6) for s in surges), default=0.0)
        max_va = max((s[1] for s in surges), default=0.0)
        return OptionsSurgeSignal(
            ticker=ticker.upper(),
            n_surges=n,
            max_vol_oi_ratio=round(max_vo, 2),
            max_vol_avg_ratio=round(max_va, 2),
            boost_multiplier=round(_classify_boost(n, strong_boost, moderate_boost), 4),
            surge_strikes=[round(s[0], 2) for s in surges[:10]],
        )


async def fetch_oi_surge_signals(
    tickers: list[str],
    otm_threshold: float = 1.01,
    dte_min: int = 2,
    dte_max: int = 45,
    vol_avg_mult: float = 5.0,
    vol_oi_mult: float = 3.0,
    strong_boost: float = 0.06,
    moderate_boost: float = 0.03,
) -> dict[str, OptionsSurgeSignal]:
    """Per-ticker options-surge detection. Returns one entry per ticker with
    a qualifying surge. Empty dict if none qualify or all fetches fail."""
    if not tickers:
        return {}
    sem = asyncio.Semaphore(_CONCURRENCY)
    results = await asyncio.gather(
        *(
            _fetch_one(
                t, otm_threshold, dte_min, dte_max,
                vol_avg_mult, vol_oi_mult, strong_boost, moderate_boost, sem,
            )
            for t in tickers
        ),
        return_exceptions=False,
    )
    out: dict[str, OptionsSurgeSignal] = {}
    for sig in results:
        if sig is not None:
            out[sig.ticker] = sig
    logger.info(
        "options_flow_screen: %d/%d tickers flagged (strong>=2 surges +%g; moderate=1 +%g)",
        len(out), len(tickers), strong_boost, moderate_boost,
    )
    return out


def apply_options_surge_to_score(
    base_score: float,
    ticker: Optional[str],
    signals: Optional[dict[str, OptionsSurgeSignal]],
) -> float:
    """Multiply composite_score by signals[ticker].boost_multiplier. Identity if missing."""
    if not signals or not ticker:
        return base_score
    sig = signals.get(ticker.upper())
    if sig is None:
        return base_score
    from backend.services.overlay_math import sign_safe_mult  # phase-69.3 sign-safe (default-OFF byte-identical)
    return sign_safe_mult(base_score, sig.boost_multiplier)
