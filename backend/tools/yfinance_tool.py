"""
yfinance data tool.
Migrated from pyfinagent-app/tools/yfinance.py — no Streamlit dependency.
"""

import logging
import yfinance as yf

logger = logging.getLogger(__name__)


def get_comprehensive_financials(ticker: str) -> dict:
    """Fetches deep fundamental data for Warren Buffett-style analysis."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # ── Valuation with computed fallbacks ────────────────────────
        trailing_pe = info.get("trailingPE")
        if trailing_pe is None:
            price = info.get("currentPrice")
            eps = info.get("trailingEps")
            if price and eps and eps != 0:
                trailing_pe = round(price / eps, 2)

        peg = info.get("pegRatio")
        if peg is None:
            fwd_pe = info.get("forwardPE")
            eg = info.get("earningsGrowth")  # decimal, e.g. 0.15 = 15%
            if fwd_pe and eg and eg > 0:
                growth_pct = eg * 100
                peg = round(fwd_pe / growth_pct, 2)

        valuation = {
            "Current Price": info.get("currentPrice"),
            "Market Cap": info.get("marketCap"),
            "P/E Ratio": trailing_pe,
            "Forward P/E": info.get("forwardPE"),
            "PEG Ratio": peg,
            "Price/Book": info.get("priceToBook"),
            "Dividend Yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
        }

        efficiency = {
            "Profit Margin": info.get("profitMargins", 0) * 100,
            "Operating Margin": info.get("operatingMargins", 0) * 100,
            "Return on Equity (ROE)": info.get("returnOnEquity", 0) * 100,
            "Revenue Growth": info.get("revenueGrowth", 0) * 100,
        }

        health = {
            "Total Cash": info.get("totalCash"),
            "Total Debt": info.get("totalDebt"),
            "Debt/Equity Ratio": info.get("debtToEquity"),
            "Current Ratio": info.get("currentRatio"),
            "Free Cash Flow": info.get("freeCashflow"),
        }

        institutional = {
            "Inst. Ownership %": info.get("heldPercentInstitutions", 0) * 100,
            "Insider Ownership %": info.get("heldPercentInsiders", 0) * 100,
            "Short Ratio": info.get("shortRatio"),
        }

        return {
            "valuation": valuation,
            "efficiency": efficiency,
            "health": health,
            "institutional": institutional,
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "week_52_high": info.get("fiftyTwoWeekHigh"),
            "week_52_low": info.get("fiftyTwoWeekLow"),
            "revenue": info.get("totalRevenue"),
            "net_income": info.get("netIncomeToCommon"),
        }

    except Exception as e:
        logger.error(f"Failed to fetch yfinance data for {ticker}: {e}")
        return {"error": f"Failed to fetch yfinance data: {str(e)}"}


def get_price_history(ticker: str, period: str = "1y") -> list[dict]:
    """Fetches historical OHLCV data for charts / ML training."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df.reset_index().to_dict(orient="records")
