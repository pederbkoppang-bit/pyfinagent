"""
Sector-ETF momentum overlay — phase-28.12.

Ranks the 11 SPDR sector ETFs by trailing 12-month total return and assigns a
small score multiplier to candidates in the top-N winning sectors. Per Quantpedia
sector momentum rotational system: top-3 SPDR sector ETFs by 12m return (monthly
rebalance) yield 13.94% annualized, Sharpe 0.54, +4%/yr vs passive S&P 500.

Cost: $0 LLM. Single yfinance batch fetch for 11 ETFs (one network round-trip).
JSON file cache with 24h TTL matches monthly rebalance cadence.

Graceful degradation: returns empty dict on any error;
`apply_sector_momentum_to_score` is identity when no ranks are provided,
preserving the cycle.

GICS sector names follow the convention in `backend/tools/screener.py::SECTOR_ETFS`
(NOT the slightly-different keys in `backend/tools/sector_analysis.py` — see
phase-28.12 research brief for the discrepancy note).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent / "_cache" / "sector_momentum"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_PATH = _CACHE_DIR / "ranks.json"

# Mirrors backend/tools/screener.py::SECTOR_ETFS (canonical GICS names).
_SECTOR_ETFS: dict[str, str] = {
    "Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}


class RankedSector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sector: str = Field(description="GICS sector name (matches screener.py keys).")
    etf: str = Field(description="SPDR sector ETF ticker (e.g., XLK).")
    momentum: float = Field(description="Trailing N-month total return (decimal, e.g., 0.155 = +15.5%).")
    rank: int = Field(description="1 = highest momentum; 11 = lowest.")
    boost_multiplier: float = Field(description="Multiplier to apply to candidates in this sector. 1.0 for non-top.")


def _cache_fresh(cache_hours: int) -> bool:
    if not _CACHE_PATH.exists():
        return False
    try:
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(
            _CACHE_PATH.stat().st_mtime, tz=timezone.utc
        )
        return age < timedelta(hours=cache_hours)
    except Exception:
        return False


async def fetch_sector_momentum_ranks(
    cache_hours: int = 24,
    lookback_months: int = 12,
    top_n: int = 3,
    boost_top: float = 1.10,
    boost_leader: float = 1.15,
) -> dict[str, RankedSector]:
    """Rank the 11 SPDR sector ETFs by trailing total return.

    Args:
        cache_hours: TTL for the local JSON cache (default 24).
        lookback_months: Lookback window (default 12 = Quantpedia canonical).
        top_n: Number of top sectors that receive the boost (default 3).
        boost_top: Multiplier for non-leader top-N sectors (default 1.10).
        boost_leader: Multiplier for #1 sector (default 1.15).

    Returns:
        dict[GICS_sector_name, RankedSector]. Empty dict on any error.
    """
    if _cache_fresh(cache_hours):
        try:
            raw = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            out = {s: RankedSector.model_validate(v) for s, v in raw.items()}
            logger.info("sector_momentum cache hit: %d sectors", len(out))
            return out
        except Exception as e:
            logger.debug("sector_momentum cache unreadable: %s", e)

    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed; sector_momentum skipped")
        return {}

    tickers = list(_SECTOR_ETFS.values())
    period = f"{max(13, lookback_months + 1)}mo"
    try:
        df = await asyncio.to_thread(
            lambda: yf.download(
                tickers, period=period, interval="1d",
                auto_adjust=True, progress=False, group_by="ticker", threads=True,
            )
        )
    except Exception as e:
        logger.warning("yfinance batch sector-ETF download failed: %s", e)
        return {}

    if df is None or len(df) == 0:
        logger.warning("yfinance returned empty for sector ETFs")
        return {}

    momentum_by_sector: list[tuple[str, str, float]] = []
    lookback_days = lookback_months * 21  # ~trading days
    for sector, etf in _SECTOR_ETFS.items():
        try:
            if etf in df.columns.get_level_values(0):
                etf_close = df[etf]["Close"].dropna()
            else:
                continue
            if hasattr(etf_close, "squeeze"):
                etf_close = etf_close.squeeze()
            if len(etf_close) < lookback_days + 1:
                logger.debug("sector_momentum: %s insufficient history (%d closes)", etf, len(etf_close))
                continue
            old = float(etf_close.iloc[-lookback_days - 1])
            new = float(etf_close.iloc[-1])
            mom = (new - old) / old if old > 0 else 0.0
            momentum_by_sector.append((sector, etf, mom))
        except Exception as e:
            logger.debug("sector_momentum: %s compute failed: %s", etf, e)
            continue

    if not momentum_by_sector:
        return {}

    momentum_by_sector.sort(key=lambda t: t[2], reverse=True)

    out: dict[str, RankedSector] = {}
    for idx, (sector, etf, mom) in enumerate(momentum_by_sector):
        rank = idx + 1
        if rank == 1:
            mult = boost_leader
        elif rank <= top_n:
            mult = boost_top
        else:
            mult = 1.0
        out[sector] = RankedSector(
            sector=sector,
            etf=etf,
            momentum=round(mom, 6),
            rank=rank,
            boost_multiplier=mult,
        )

    try:
        payload = {s: json.loads(r.model_dump_json()) for s, r in out.items()}
        _CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as e:
        logger.debug("sector_momentum cache write failed: %s", e)

    top3 = [(r.sector, r.etf, r.momentum) for r in out.values() if r.rank <= top_n]
    logger.info(
        "sector_momentum: top-%d sectors %s",
        top_n, [(s, f"{m * 100:+.1f}%") for s, _, m in top3],
    )
    return out


def apply_sector_momentum_to_score(
    base_score: float,
    sector: Optional[str],
    ranks: Optional[dict[str, RankedSector]],
) -> float:
    """Apply sector-momentum boost to a candidate's composite score.

    No-op when no ranks dict, no sector, or sector not in ranks. Otherwise
    multiplies by the ranked entry's `boost_multiplier` (which is 1.0 for
    non-top sectors -> identity).
    """
    if not ranks or not sector:
        return base_score
    entry = ranks.get(sector)
    if entry is None:
        return base_score
    from backend.services.overlay_math import sign_safe_mult  # phase-69.3 sign-safe (default-OFF byte-identical)
    return sign_safe_mult(base_score, entry.boost_multiplier)
