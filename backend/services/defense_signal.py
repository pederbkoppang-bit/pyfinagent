"""
Defense/war-stocks reference-case signal — phase-28.14.

When BOTH (a) Caldara-Iacoviello GPR-Acts is above its quantile threshold (reuses
the phase-28.3 `_fetch_gpr_acts` cache) AND (b) XAR (SPDR S&P Aerospace & Defense
ETF) 5-day momentum > 0 (institutional money confirming the GPR signal via the
defense ETF), boost individual defense-ticker composite scores in the screener.

Per supplement Gap 1:
- Emerald SEF 2023: 30 European defense firms; +1.00% (-1,-1) anticipatory; +11.65% CAAR (0,3) after Ukraine invasion
- PMC11700249: 81.4% of 75 defense firms reacted to Ukraine invasion
- Researcher: BAE.L + RHM.DE most GPR-sensitive; XAR preferred over ITA (ITA 19% GE/commercial-aviation noise)

Why AND-gate (not OR):
- GPR alone fires on any geopolitical event — including ones that DON'T move defense stocks
- XAR positive momentum confirms institutional flow is actually pricing the GPR signal into defense
- Both together = high-confidence convergence

Default OFF. Cycle-level signal (one fetch per cycle, not per-ticker).

Graceful degradation: any fetch failure → triggered=False → identity.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class DefenseSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    triggered: bool = Field(description="True if GPR above threshold AND XAR 5d momentum > min.")
    gpr_above_threshold: bool = Field(description="Phase-28.3 GPR check outcome.")
    gpr_current: float = Field(description="Latest GPRA value (or 0 if fetch failed).")
    gpr_threshold: float = Field(description="GPRA quantile threshold.")
    xar_5d_momentum: float = Field(description="XAR cumulative return over the window (decimal).")
    xar_threshold: float = Field(description="Configured XAR min momentum.")
    pledge_keyword_hit: bool = Field(description="True if any defense-pledge keyword appeared in news (optional).")
    boost_multiplier: float = Field(description="Multiplier to apply to defense-list ticker composite (1.0 = no boost).")
    defense_tickers: list[str] = Field(description="The defense-ticker list this signal targets.")


async def _fetch_xar_momentum(window_days: int = 5) -> Optional[float]:
    """Compute XAR cumulative return over `window_days`. Returns None on failure."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        period_days = max(window_days + 5, 30)
        period = f"{period_days}d"
        df = await asyncio.to_thread(
            lambda: yf.download("XAR", period=period, interval="1d",
                                auto_adjust=True, progress=False)
        )
    except Exception as e:
        logger.debug("defense_signal: XAR yfinance fetch failed: %s", e)
        return None
    if df is None or len(df) == 0:
        return None
    try:
        close = df["Close"].dropna()
        if hasattr(close, "squeeze"):
            close = close.squeeze()
        if len(close) < window_days + 1:
            return None
        old = float(close.iloc[-window_days - 1])
        new = float(close.iloc[-1])
        if old <= 0:
            return None
        return (new - old) / old
    except Exception as e:
        logger.debug("defense_signal: XAR compute failed: %s", e)
        return None


async def fetch_defense_trigger(
    defense_tickers_csv: str = "LMT,NOC,RTX,GD,LHX,BA,LDOS,HII,KTOS,BAE.L,RHM.DE,SAAB-B.ST",
    xar_window_days: int = 5,
    xar_min_momentum: float = 0.0,
    boost: float = 0.05,
    gpr_quantile: float = 0.90,
    gpr_cache_hours: int = 24,
    pledge_keywords_csv: str = "",
    pledge_hit_provider=None,
) -> DefenseSignal:
    """Cycle-level defense trigger fetch.

    Returns a DefenseSignal with `triggered` reflecting the AND-gate over GPR + XAR.
    Reuses phase-28.3 `_fetch_gpr_acts` (cached). `pledge_hit_provider` is an
    optional callable(keywords_list) -> bool for news-keyword scanning; when None,
    pledge_keyword_hit is False (not gating).
    """
    tickers = [t.strip().upper() for t in defense_tickers_csv.split(",") if t.strip()]

    try:
        from backend.services.macro_regime import _fetch_gpr_acts
        gpr_info = await _fetch_gpr_acts(cache_hours=gpr_cache_hours, quantile=gpr_quantile)
    except Exception as e:
        logger.debug("defense_signal: GPR fetch failed: %s", e)
        gpr_info = None
    if not gpr_info:
        gpr_above = False
        gpr_current = 0.0
        gpr_threshold = 0.0
    else:
        gpr_above = bool(gpr_info.get("above_threshold", False))
        gpr_current = float(gpr_info.get("current", 0.0))
        gpr_threshold = float(gpr_info.get("threshold", 0.0))

    xar_mom = await _fetch_xar_momentum(xar_window_days)
    if xar_mom is None:
        xar_above = False
        xar_mom_val = 0.0
    else:
        xar_mom_val = xar_mom
        xar_above = xar_mom > xar_min_momentum

    pledge_hit = False
    if pledge_keywords_csv and pledge_hit_provider:
        try:
            keywords = [k.strip() for k in pledge_keywords_csv.split(",") if k.strip()]
            pledge_hit = bool(pledge_hit_provider(keywords))
        except Exception:
            pledge_hit = False

    triggered = gpr_above and xar_above

    sig = DefenseSignal(
        triggered=triggered,
        gpr_above_threshold=gpr_above,
        gpr_current=round(gpr_current, 4),
        gpr_threshold=round(gpr_threshold, 4),
        xar_5d_momentum=round(xar_mom_val, 6),
        xar_threshold=float(xar_min_momentum),
        pledge_keyword_hit=pledge_hit,
        boost_multiplier=round(1.0 + boost, 4) if triggered else 1.0,
        defense_tickers=tickers,
    )
    logger.info(
        "defense_signal: triggered=%s (GPR above=%s current=%.2f thr=%.2f; XAR %dd mom=%.3f%% above=%s); boost=%.3f",
        triggered, gpr_above, gpr_current, gpr_threshold,
        xar_window_days, xar_mom_val * 100, xar_above, sig.boost_multiplier,
    )
    return sig


def apply_defense_boost_to_score(
    base_score: float,
    ticker: Optional[str],
    signal: Optional[DefenseSignal],
) -> float:
    """Multiply score by signal.boost_multiplier when ticker is in defense list AND triggered."""
    if not signal or not signal.triggered or not ticker:
        return base_score
    if ticker.upper() in {t.upper() for t in signal.defense_tickers}:
        from backend.services.overlay_math import sign_safe_mult  # phase-69.3 sign-safe (default-OFF byte-identical)
        return sign_safe_mult(base_score, signal.boost_multiplier)
    return base_score
