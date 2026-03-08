"""
Portfolio API — CRUD for positions + performance tracking.
Uses an in-memory store for simplicity (no external DB dependency).
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.tools import yfinance_tool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

# ── In-memory store ──────────────────────────────────────────────

_positions: dict[str, dict] = {}


# ── Models ───────────────────────────────────────────────────────

class PositionCreate(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    quantity: float = Field(..., gt=0)
    avg_entry_price: float = Field(..., gt=0)
    recommendation: Optional[str] = None
    recommendation_score: Optional[float] = None


class PositionResponse(BaseModel):
    id: str
    ticker: str
    quantity: float
    avg_entry_price: float
    cost_basis: float
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    recommendation: Optional[str] = None
    recommendation_score: Optional[float] = None
    added_at: str


# ── Endpoints ────────────────────────────────────────────────────


@router.get("/")
async def list_positions():
    """Return all portfolio positions with live prices."""
    results = []
    for pos_id, pos in _positions.items():
        enriched = _enrich_position(pos_id, pos)
        results.append(enriched)
    return results


@router.post("/", status_code=201)
async def add_position(body: PositionCreate):
    """Add a new position to the portfolio."""
    ticker = body.ticker.upper().strip()
    pos_id = str(uuid.uuid4())[:8]

    position = {
        "ticker": ticker,
        "quantity": body.quantity,
        "avg_entry_price": body.avg_entry_price,
        "cost_basis": round(body.quantity * body.avg_entry_price, 2),
        "recommendation": body.recommendation,
        "recommendation_score": body.recommendation_score,
        "added_at": datetime.now(timezone.utc).isoformat(),
    }
    _positions[pos_id] = position
    logger.info(f"Added position {pos_id}: {ticker} x{body.quantity} @ ${body.avg_entry_price}")
    return _enrich_position(pos_id, position)


@router.delete("/{position_id}")
async def delete_position(position_id: str):
    """Remove a position from the portfolio."""
    if position_id not in _positions:
        raise HTTPException(status_code=404, detail="Position not found")
    removed = _positions.pop(position_id)
    logger.info(f"Removed position {position_id}: {removed['ticker']}")
    return {"message": "Position removed", "id": position_id}


@router.get("/performance")
async def get_portfolio_performance():
    """Calculate portfolio-level performance and recommendation accuracy."""
    if not _positions:
        return {
            "total_cost_basis": 0,
            "total_market_value": 0,
            "total_pnl": 0,
            "total_pnl_pct": 0,
            "positions_count": 0,
            "recommendation_accuracy": None,
            "allocation": [],
        }

    total_cost = 0.0
    total_market = 0.0
    allocation = []
    correct_recs = 0
    total_recs = 0

    for pos_id, pos in _positions.items():
        enriched = _enrich_position(pos_id, pos)
        cost = enriched.get("cost_basis", 0) or 0
        market = enriched.get("market_value", 0) or 0
        total_cost += cost
        total_market += market
        allocation.append({
            "ticker": enriched["ticker"],
            "market_value": market,
            "pnl": enriched.get("unrealized_pnl", 0),
            "pnl_pct": enriched.get("unrealized_pnl_pct", 0),
        })

        # Recommendation accuracy: BUY/STRONG_BUY with positive return = correct
        rec = (enriched.get("recommendation") or "").upper()
        pnl = enriched.get("unrealized_pnl", 0) or 0
        if rec in ("BUY", "STRONG_BUY", "SELL", "STRONG_SELL"):
            total_recs += 1
            is_buy = rec in ("BUY", "STRONG_BUY")
            if (is_buy and pnl > 0) or (not is_buy and pnl < 0):
                correct_recs += 1

    total_pnl = total_market - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

    return {
        "total_cost_basis": round(total_cost, 2),
        "total_market_value": round(total_market, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "positions_count": len(_positions),
        "recommendation_accuracy": round(correct_recs / total_recs * 100, 1) if total_recs > 0 else None,
        "allocation": allocation,
    }


# ── Helpers ──────────────────────────────────────────────────────

def _enrich_position(pos_id: str, pos: dict) -> dict:
    """Add live price data to a position."""
    result = {"id": pos_id, **pos}
    try:
        yf_data = yfinance_tool.get_comprehensive_financials(pos["ticker"])
        current_price = yf_data.get("valuation", {}).get("Current Price")
        if current_price and isinstance(current_price, (int, float)):
            result["current_price"] = round(current_price, 2)
            result["market_value"] = round(pos["quantity"] * current_price, 2)
            result["unrealized_pnl"] = round(result["market_value"] - pos["cost_basis"], 2)
            if pos["cost_basis"] > 0:
                result["unrealized_pnl_pct"] = round(
                    result["unrealized_pnl"] / pos["cost_basis"] * 100, 2
                )
    except Exception as e:
        logger.warning(f"Failed to fetch price for {pos['ticker']}: {e}")
    return result
