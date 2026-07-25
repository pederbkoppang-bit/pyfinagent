"""
Sector analysis tool — relative strength, rotation signals, and peer comparison.
Uses yfinance sector ETFs and peer group data.
"""

import logging
import math

import yfinance as yf

logger = logging.getLogger(__name__)


def _count_non_finite(obj, _depth: int = 0) -> int:
    """Count NaN/Inf floats anywhere in `obj` (phase-80.27 cycle 2).

    Used to prove the returned payload cannot serialise a bare `NaN` token
    into an LLM prompt. `bool` subclasses `int`, not `float`, so flags are
    not counted. Depth-capped for safety on a pipeline path.
    """
    if _depth > 12:
        return 0
    if isinstance(obj, float):
        return 0 if math.isfinite(obj) else 1
    if isinstance(obj, dict):
        return sum(_count_non_finite(v, _depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_count_non_finite(v, _depth + 1) for v in obj)
    return 0


def _nonfinite_fail_safe_enabled() -> bool:
    """phase-80.27 dark-launch gate. OFF -> byte-identical legacy behaviour.

    The settings import is FUNCTION-LOCAL on purpose: nothing in
    `backend/tools/` imports the settings module today, and adding a
    module-level import here risks a circular-import regression on a
    package that the API, the MCP servers and the Layer-1 orchestrator all
    import. Fails OPEN to the legacy path if settings are unavailable --
    this gate must never be the reason an analysis crashes.
    """
    try:
        from backend.config.settings import get_settings

        return bool(getattr(get_settings(), "tools_nonfinite_fail_safe_enabled", False))
    except Exception:  # pragma: no cover - defensive
        return False

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

        # phase-80.27 -- FAIL-SAFE GUARD, BEFORE the comparison ladder.
        #
        # The `0` defaults above NEVER fire: `_compute_return` stores the key
        # WITH a NaN value (its `if ret is not None` test at :55/:70/:78/:87
        # is a null check being used as a numeric-validity check, and
        # `nan is not None` is True), so `dict.get` returns the NaN, not the
        # default.
        #
        # Every IEEE-754 comparison against NaN is False. So `sec_3m > spy_3m`,
        # `stock_3m > sec_3m` and the `elif` below are ALL False, control falls
        # through to the `signal = "NEUTRAL"` initialiser above, and the tool
        # emits a confident NEUTRAL trading verdict built from nothing. That
        # verdict reaches the LLM debate and the paper-trading loop, while the
        # summary renders literally "3M return: +nan% vs sector +nan%".
        #
        # Initialising to a VALID verdict before a comparison ladder is itself
        # the bug: the default must be the failure state. Guarding here rather
        # than at each comparison keeps one decision point.
        if _nonfinite_fail_safe_enabled() and not all(
            math.isfinite(v) for v in (sec_3m, spy_3m, stock_3m)
        ):
            logger.warning(
                "sector_analysis %s: non-finite inputs "
                "(stock_3m=%r sector_3m=%r spy_3m=%r) -- returning ERROR "
                "instead of a fabricated NEUTRAL (phase-80.27)",
                ticker, stock_3m, sec_3m, spy_3m,
            )
            return {
                "ticker": ticker,
                "signal": "ERROR",
                "summary": (
                    f"Error: sector analysis unavailable for {ticker} -- "
                    "price history returned non-finite values, so no "
                    "sector/market comparison could be computed."
                ),
                "data": {},
            }

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

        payload = {
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

        # phase-80.27 cycle 2 -- PAYLOAD-COMPLETENESS GUARD.
        #
        # The ladder guard above only inspects the three 3mo operands, so a
        # PARTIAL outage -- e.g. a bad bar at the START of the 1mo window,
        # which poisons that horizon alone because _compute_return is
        # Close[-1]/Close[0] -- produced a genuine-looking NEUTRAL whose
        # payload still carried NaN in stock_returns["1mo"]. That dict is
        # serialised by `json.dumps(sector_data, indent=2)` at
        # backend/config/prompts.py:537 and handed to the sector agent, and
        # stdlib json.dumps emits a bare `NaN` token by default. Criterion 3
        # is about the payload the agent RECEIVES, not only the summary
        # string, so the ladder guard alone did not satisfy it.
        #
        # Found by Q/A (cycle 1, finding D1) with a demonstrated leak, not by
        # me. Guarding the assembled payload is the fail-safe reading: any
        # non-finite anywhere means this analysis is not fit to be reasoned
        # over, so it becomes ERROR rather than a partially-fictional verdict.
        if _nonfinite_fail_safe_enabled():
            leaked = _count_non_finite(payload)
            if leaked:
                logger.warning(
                    "sector_analysis %s: %d non-finite value(s) in the assembled "
                    "payload -- returning ERROR so no NaN reaches an LLM prompt "
                    "(phase-80.27)",
                    ticker, leaked,
                )
                return {
                    "ticker": ticker,
                    "signal": "ERROR",
                    "summary": (
                        f"Error: sector analysis unavailable for {ticker} -- "
                        f"{leaked} computed value(s) were non-finite, so the "
                        "analysis is incomplete."
                    ),
                    "data": {},
                }

        return payload

    except Exception as e:
        logger.error("Failed sector analysis for %s: %s", ticker, e)
        return {"ticker": ticker, "signal": "ERROR", "summary": f"Error: {e}"}
