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
    """Fetches historical OHLCV data for charts / ML training.

    phase-25.E7: wraps the yfinance call in try/except + emits a
    `data_source_events` row on failure so operators can compute
    rate-limit / quota / no-data dominance over any window. Closes
    audit bucket 24.7 F-4 (previously unguarded).

    Returns a single-element error list on failure / empty-DataFrame:
        [{"error": "<reason>", "ticker": ticker}]
    so callers iterating get exactly one structured error row.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df is None or df.empty:
            logger.warning(
                "get_price_history empty result for %s (period=%s)", ticker, period
            )
            _persist_yfinance_event(ticker, "empty_dataframe")
            return [{"error": "no_data", "ticker": ticker}]
        return df.reset_index().to_dict(orient="records")
    except Exception as exc:
        logger.warning(
            "get_price_history failed for %s (period=%s): %r",
            ticker,
            period,
            exc,
            exc_info=True,
        )
        _persist_yfinance_event(ticker, type(exc).__name__)
        return [{"error": str(exc), "ticker": ticker}]


def _persist_yfinance_event(ticker: str, notes: str) -> None:
    """phase-25.E7: best-effort write to `data_source_events` so the
    `pct_yfinance_fallback_dominance` aggregation can include
    price-history failures. Fail-open: any BQ failure is swallowed
    so the caller still gets the structured error response.
    """
    try:
        from backend.config.settings import get_settings
        from backend.db.bigquery_client import BigQueryClient
        bq = BigQueryClient(get_settings())
        bq.save_data_source_event(
            ticker=ticker,
            source="yfinance_price_history",
            kind="fallback",
            article_count=None,
            notes=str(notes)[:200],
        )
    except Exception as bq_exc:
        logger.warning(
            "yfinance_price_history: save_data_source_event fail-open for %s: %r",
            ticker,
            bq_exc,
        )
