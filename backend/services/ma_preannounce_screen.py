"""
M&A pre-announcement detection — phase-28.16 (FINAL phase-28 item).

Three-leg aggregator over the public footprint of informed M&A activity:

  Leg 1 — OTM options surge (Augustin-Brenner-Subrahmanyam)
          near-expiry OTM call volume spike + IV term-structure inversion.
          Implemented by `options_flow_screen.py` (phase-28.9).

  Leg 2 — Insider opportunistic buying (Cohen-Malloy-Pomorski; Duong-Pi-Sapp 2025)
          Form 4 BUY clusters by non-routine insiders.
          Implemented by `insider_signal_screen.py` (phase-28.10).

  Leg 3 — Schedule 13D / 13G filings (NEW)
          Activist beneficial-ownership > 5% within 10-day deadline.
          STUBBED for Phase-2: SEC EDGAR full-text-search API returned 403 to
          direct WebFetch in this environment; requires authenticated/programmatic
          client (etf-scraper-style session). `_fetch_13d_filings_for` returns
          [] until the EDGAR client is wired.

Aggregation:
  legs_triggered_count == 0 -> identity (no signal)
  legs_triggered_count == 1 -> moderate boost (+5%)
  legs_triggered_count >= 2 -> strong boost   (+10%, high-confidence convergence)

LEGALITY BOUNDARY (per Augustin et al.):
  The picker observes ONLY PUBLIC DATA — options chain, Form 4, 13D filings.
  It does NOT infer or act on material non-public information.
  Augustin documents what informed traders DO; we observe the PUBLIC FOOTPRINT.

Default OFF. Zero additional network cost — reuses signals from 28.9 + 28.10
(autonomous_loop already fetches them when their respective flags are on).
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class MAPreannounceSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(description="Uppercase ticker.")
    legs_triggered: list[str] = Field(description="List of leg names that fired: options, insider, 13d.")
    legs_count: int = Field(description="Count of legs that triggered (0-3).")
    boost_multiplier: float = Field(description="Final multiplier on composite_score (1.0 = no boost).")
    boost_tier: str = Field(description="strong | moderate | none.")


async def _fetch_13d_filings_for(ticker: str) -> list[dict]:
    """STUB: SEC EDGAR Schedule 13D fetch for this ticker.

    Returns recent 13D/13G/13D-A filings naming the ticker. Currently returns
    [] because the SEC EDGAR full-text-search API (efts.sec.gov/LATEST/search-index)
    returns HTTP 403 to direct WebFetch in this environment — requires an
    authenticated / programmatic client (e.g. browser-style session + cookies).

    Future implementation path (documented in supplement Gap 3 + phase-28.16 brief):
        - Use `httpx.AsyncClient` with browser User-Agent + cookies persisted
        - Or add `sec-edgar` PyPI dep that handles session
        - Endpoint: https://efts.sec.gov/LATEST/search-index?q={CIK}&forms=SC+13D,SC+13G,SC+13D/A
        - Returns JSON with filings list + filed-on dates

    Until then, Leg 3 is documented but not gating: HIGH-confidence (2+ legs)
    can be satisfied by Legs 1+2 alone (options + insider — the two strongest
    signals per the academic literature).
    """
    # TODO phase-28.16-followup-13d-edgar: wire authenticated SEC EDGAR client
    return []


def _classify_boost(legs_count: int, strong_boost: float, moderate_boost: float) -> tuple[float, str]:
    if legs_count >= 2:
        return 1.0 + strong_boost, "strong"
    if legs_count == 1:
        return 1.0 + moderate_boost, "moderate"
    return 1.0, "none"


def compute_ma_preannounce_signals(
    tickers: list[str],
    options_surge_signals: Optional[dict] = None,
    insider_signals: Optional[dict] = None,
    schedule_13d_signals: Optional[dict] = None,
    strong_boost: float = 0.10,
    moderate_boost: float = 0.05,
) -> dict[str, MAPreannounceSignal]:
    """Pure aggregator over the three legs. Caller supplies the per-leg lookups.

    Args:
        tickers: list of candidate tickers to evaluate
        options_surge_signals: dict[ticker, OptionsSurgeSignal] from phase-28.9
        insider_signals: dict[ticker, InsiderSignal] from phase-28.10
        schedule_13d_signals: dict[ticker, dict] from Leg 3 (currently {})
        strong_boost: multiplier when 2+ legs fire (default 0.10)
        moderate_boost: multiplier when 1 leg fires (default 0.05)

    Returns:
        dict[ticker, MAPreannounceSignal] for tickers with >=1 leg firing.
    """
    if not tickers:
        return {}
    options_surge_signals = options_surge_signals or {}
    insider_signals = insider_signals or {}
    schedule_13d_signals = schedule_13d_signals or {}

    out: dict[str, MAPreannounceSignal] = {}
    for t in tickers:
        tu = t.upper()
        legs = []
        if tu in options_surge_signals:
            legs.append("options")
        if tu in insider_signals:
            legs.append("insider")
        if tu in schedule_13d_signals:
            legs.append("13d")
        if not legs:
            continue
        boost, tier = _classify_boost(len(legs), strong_boost, moderate_boost)
        out[tu] = MAPreannounceSignal(
            ticker=tu,
            legs_triggered=legs,
            legs_count=len(legs),
            boost_multiplier=round(boost, 4),
            boost_tier=tier,
        )
    logger.info(
        "ma_preannounce_screen: %d tickers with M&A pre-announcement signal (strong>=2 legs +%g; moderate=1 leg +%g)",
        len(out), strong_boost, moderate_boost,
    )
    return out


def apply_ma_preannounce_to_score(
    base_score: float,
    ticker: Optional[str],
    signals: Optional[dict[str, MAPreannounceSignal]],
) -> float:
    """Multiply score by signals[ticker].boost_multiplier. Identity if missing."""
    if not signals or not ticker:
        return base_score
    sig = signals.get(ticker.upper())
    if sig is None:
        return base_score
    from backend.services.overlay_math import sign_safe_mult  # phase-69.3 sign-safe (default-OFF byte-identical)
    return sign_safe_mult(base_score, sig.boost_multiplier)
