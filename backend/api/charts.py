"""
Charts API — price history and financial data for frontend visualization.
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query

from backend.tools import yfinance_tool

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/charts", tags=["charts"])


@router.get("/{ticker}")
async def get_price_history(
    ticker: str,
    period: str = Query("1y", pattern=r"^(1mo|3mo|6mo|1y|2y|5y|max)$"),
):
    """Return OHLCV price history for the given ticker."""
    try:
        rows = await asyncio.to_thread(yfinance_tool.get_price_history, ticker.upper(), period=period)
        if not rows:
            raise HTTPException(status_code=404, detail=f"No price data for {ticker}")
        # Convert timestamps to ISO strings for JSON serialization
        for row in rows:
            if hasattr(row.get("Date"), "isoformat"):
                row["Date"] = row["Date"].isoformat()
        return rows
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error fetching price history for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch price history: {type(exc).__name__}: {exc}",
        )


@router.get("/{ticker}/financials")
async def get_financials(ticker: str):
    """Return yfinance fundamental data (valuation, efficiency, health, institutional)."""
    try:
        data = await asyncio.to_thread(yfinance_tool.get_comprehensive_financials, ticker.upper())
        if "error" in data:
            raise HTTPException(status_code=502, detail=data["error"])
        return data
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error fetching financials for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch financials: {type(exc).__name__}: {exc}",
        )
