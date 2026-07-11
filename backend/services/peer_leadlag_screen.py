"""
Peer-correlation laggard catch-up signal — phase-28.17.

Implements the intra-sector lead-lag effect (Hou 2007 + DeltaLag arXiv 2511.00390):
within the same sector, big leaders price information first; small laggards with
low analyst coverage catch up over 1-3 months.

Researcher (supplement Gap 4):
- Group by SECTOR (not sub-industry) — 11 GICS sectors > sparse sub-industry groups
  on a ~500-stock universe. Both satisfy the spec intent (peer comparison).
- Leader: momentum_1m > +10%
- Laggard: momentum_1m < +2%
- Qualify boost: analyst_count < 5 (Hou 2007: lag strongest with low coverage)
  AND market_cap >= $2B (DeltaLag liquidity gate)
- Boost: 1.08 (8%, conservative vs DeltaLag ~10 bpts/day gross alpha)

Cost: pure function over screen_data + caller-provided lookup. Lookup is filled
by the caller (autonomous_loop) via ~20 yfinance .info calls per cycle when the
flag is on — bounded by top 2×paper_screen_top_n.

Graceful degradation: missing analyst/market_cap → laggard NOT qualifying →
identity. Missing sector → ticker excluded from grouping.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class PeerLagSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(description="Uppercase laggard ticker.")
    sector: str = Field(description="GICS sector (peer group).")
    own_momentum_1m: float = Field(description="Laggard's 1m momentum (percent).")
    peer_leaders: list[str] = Field(description="Peer tickers with momentum_1m > leader_threshold (up to 5).")
    median_leader_momentum: float = Field(description="Median momentum_1m of leaders in this sector (percent).")
    analyst_count: int = Field(description="Sell-side analyst count for the laggard.")
    market_cap_usd: float = Field(description="Market cap (USD).")
    boost_multiplier: float = Field(description="Final multiplier on composite_score (1.0 = no boost).")


def compute_peer_leadlag_signals(
    screen_data: list[dict],
    analyst_market_cap_lookup: dict[str, dict],
    leader_threshold: float = 10.0,
    laggard_threshold: float = 2.0,
    max_analyst_count: int = 5,
    min_market_cap_usd: float = 2_000_000_000.0,
    boost: float = 0.08,
) -> dict[str, PeerLagSignal]:
    """Pure computation over screen_data + lookup. No I/O.

    Args:
        screen_data: list of {ticker, sector, momentum_1m, ...} dicts (from screen_universe)
        analyst_market_cap_lookup: {ticker: {"analyst_count": int, "market_cap": float}}
        leader_threshold: momentum_1m percent above which a stock is a leader (default 10.0)
        laggard_threshold: momentum_1m percent below which a stock is a laggard (default 2.0)
        max_analyst_count: max sell-side coverage for laggard qualification (default 5)
        min_market_cap_usd: min market cap for laggard qualification (default $2B)
        boost: multiplier added (default 0.08 = +8%)

    Returns:
        dict[ticker, PeerLagSignal] for qualifying laggards. Empty dict if no qualifying setups.
    """
    if not screen_data:
        return {}

    by_sector: dict[str, list[dict]] = defaultdict(list)
    for s in screen_data:
        sector = (s.get("sector") or "").strip()
        if not sector:
            continue
        if s.get("momentum_1m") is None:
            continue
        by_sector[sector].append(s)

    out: dict[str, PeerLagSignal] = {}
    for sector, members in by_sector.items():
        leaders = [m for m in members if (m.get("momentum_1m") or 0) > leader_threshold]
        if len(leaders) < 1:
            continue
        leader_moms = sorted([m.get("momentum_1m") or 0 for m in leaders])
        median_leader = leader_moms[len(leader_moms) // 2] if leader_moms else 0.0
        laggards = [m for m in members if (m.get("momentum_1m") or 0) < laggard_threshold]
        for lag in laggards:
            ticker = (lag.get("ticker") or "").upper()
            if not ticker:
                continue
            info = analyst_market_cap_lookup.get(ticker) or {}
            try:
                analyst_count = int(info.get("analyst_count") or 0)
                market_cap = float(info.get("market_cap") or 0)
            except (ValueError, TypeError):
                continue
            if analyst_count == 0 or market_cap == 0:
                continue  # missing data → don't qualify
            if analyst_count >= max_analyst_count:
                continue
            if market_cap < min_market_cap_usd:
                continue
            out[ticker] = PeerLagSignal(
                ticker=ticker,
                sector=sector,
                own_momentum_1m=float(lag.get("momentum_1m") or 0),
                peer_leaders=[(m.get("ticker") or "").upper() for m in leaders[:5]],
                median_leader_momentum=float(median_leader),
                analyst_count=analyst_count,
                market_cap_usd=market_cap,
                boost_multiplier=round(1.0 + boost, 4),
            )
    logger.info(
        "peer_leadlag_screen: %d laggards qualifying across %d sectors (leader>%.1f%%, laggard<%.1f%%, analysts<%d, mcap>=$%.0fB)",
        len(out), len(by_sector), leader_threshold, laggard_threshold, max_analyst_count,
        min_market_cap_usd / 1e9,
    )
    return out


def apply_peer_leadlag_to_score(
    base_score: float,
    ticker: Optional[str],
    signals: Optional[dict[str, PeerLagSignal]],
) -> float:
    """Multiply score by signals[ticker].boost_multiplier. Identity if missing."""
    if not signals or not ticker:
        return base_score
    sig = signals.get(ticker.upper())
    if sig is None:
        return base_score
    from backend.services.overlay_math import sign_safe_mult  # phase-69.3 sign-safe (default-OFF byte-identical)
    return sign_safe_mult(base_score, sig.boost_multiplier)
