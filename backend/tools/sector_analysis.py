"""
Sector analysis tool — relative strength, rotation signals, and peer comparison.
Uses yfinance sector ETFs and peer group data.
"""

import logging

import yfinance as yf

logger = logging.getLogger(__name__)

# SPDR Select Sector ETFs (covers all 11 GICS sectors)
SECTOR_ETFS = {
    "Technology": "XLK",
    "Financial": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
    "Consumer Discretionary": "XLY",
}


def _compute_return(ticker_obj, period: str) -> float | None:
    """Compute the total return for a given period."""
    try:
        hist = ticker_obj.history(period=period)
        if len(hist) < 2:
            return None
        return ((hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1) * 100
    except Exception:
        return None


def get_sector_analysis(ticker: str) -> dict:
    """
    Comprehensive sector analysis: sector performance, relative strength,
    and peer comparison.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")
        company_name = info.get("longName", ticker)

        # Get stock returns
        stock_returns = {}
        for period in ["1mo", "3mo", "6mo", "1y"]:
            ret = _compute_return(stock, period)
            if ret is not None:
                stock_returns[period] = round(ret, 2)

        # Get sector ETF returns
        sector_etf = None
        sector_returns = {}
        for sec_name, etf_ticker in SECTOR_ETFS.items():
            if sec_name.lower() in sector.lower() or sector.lower() in sec_name.lower():
                sector_etf = etf_ticker
                break

        if sector_etf:
            etf = yf.Ticker(sector_etf)
            for period in ["1mo", "3mo", "6mo", "1y"]:
                ret = _compute_return(etf, period)
                if ret is not None:
                    sector_returns[period] = round(ret, 2)

        # SPY benchmark
        spy = yf.Ticker("SPY")
        spy_returns = {}
        for period in ["1mo", "3mo", "6mo", "1y"]:
            ret = _compute_return(spy, period)
            if ret is not None:
                spy_returns[period] = round(ret, 2)

        # All sector ETF performance (for rotation chart)
        sector_performance = {}
        for sec_name, etf_ticker in SECTOR_ETFS.items():
            try:
                etf = yf.Ticker(etf_ticker)
                ret_3m = _compute_return(etf, "3mo")
                if ret_3m is not None:
                    sector_performance[sec_name] = round(ret_3m, 2)
            except Exception:
                pass

        # Relative strength vs sector and market
        rel_vs_sector = {}
        rel_vs_market = {}
        for period in ["1mo", "3mo", "6mo", "1y"]:
            s = stock_returns.get(period)
            sec = sector_returns.get(period)
            mkt = spy_returns.get(period)
            if s is not None and sec is not None:
                rel_vs_sector[period] = round(s - sec, 2)
            if s is not None and mkt is not None:
                rel_vs_market[period] = round(s - mkt, 2)

        # Peer comparison (same industry, top by market cap)
        # Note: yfinance doesn't provide direct peer lookup, so we use
        # the sector info heuristically. Industry peers fetched if available.
        peers = []
        peer_tickers = info.get("recommendedSymbols", [])
        # Not all yfinance versions have this; fall back gracefully
        if not peer_tickers:
            peer_tickers = []

        # Try to get a few peers
        for pt in peer_tickers[:5]:
            sym = pt if isinstance(pt, str) else pt.get("symbol", "")
            if sym and sym != ticker:
                try:
                    p_info = yf.Ticker(sym).info
                    peers.append({
                        "ticker": sym,
                        "name": p_info.get("longName", sym),
                        "market_cap": p_info.get("marketCap"),
                        "pe_ratio": p_info.get("trailingPE"),
                        "revenue_growth": (p_info.get("revenueGrowth", 0) or 0) * 100,
                        "profit_margin": (p_info.get("profitMargins", 0) or 0) * 100,
                        "roe": (p_info.get("returnOnEquity", 0) or 0) * 100,
                    })
                except Exception:
                    pass

        # Signals
        signal = "NEUTRAL"
        sector_tailwind = False
        stock_outperforming = False

        sec_3m = sector_returns.get("3mo", 0)
        spy_3m = spy_returns.get("3mo", 0)
        stock_3m = stock_returns.get("3mo", 0)

        if sec_3m > spy_3m:
            sector_tailwind = True

        if stock_3m > sec_3m:
            stock_outperforming = True

        if sector_tailwind and stock_outperforming:
            signal = "DOUBLE_TAILWIND"
        elif sector_tailwind:
            signal = "SECTOR_TAILWIND"
        elif stock_outperforming:
            signal = "STOCK_OUTPERFORMING"
        elif stock_3m < sec_3m and stock_3m < spy_3m:
            signal = "LAGGING"

        return {
            "ticker": ticker,
            "company_name": company_name,
            "sector": sector,
            "industry": industry,
            "sector_etf": sector_etf,
            "stock_returns": stock_returns,
            "sector_returns": sector_returns,
            "spy_returns": spy_returns,
            "relative_vs_sector": rel_vs_sector,
            "relative_vs_market": rel_vs_market,
            "sector_performance": sector_performance,
            "peers": peers,
            "sector_tailwind": sector_tailwind,
            "stock_outperforming": stock_outperforming,
            "signal": signal,
            "summary": (
                f"{ticker} ({sector}/{industry}). "
                f"3M return: {stock_3m:+.1f}% vs sector {sec_3m:+.1f}% vs S&P {spy_3m:+.1f}%. "
                f"{'Sector tailwind active. ' if sector_tailwind else ''}"
                f"{'Stock outperforming sector. ' if stock_outperforming else ''}"
                f"Signal: {signal}."
            ),
        }

    except Exception as e:
        logger.error("Failed sector analysis for %s: %s", ticker, e)
        return {"ticker": ticker, "signal": "ERROR", "summary": f"Error: {e}"}
