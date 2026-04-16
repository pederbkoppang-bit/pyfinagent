"""
reconciliation.py -- Paper-live vs parallel OOS-backtest NAV reconciliation.

The "reality gap" check: for the same signals paper trading executed on, replay
them against a frictionless out-of-sample backtest using yfinance adjusted
close as the ground-truth fill price. If the paper-live equity curve and the
shadow backtest curve diverge by more than 5% at the latest point, emit an
alert -- it usually means one of:

  - Execution drift (paper fill price != next-day open the backtest assumed)
  - Stale signals (paper acted on day-T data, backtest re-solves on day-T+1)
  - A genuine environment bug (schema/plumbing)

The module is a *read-only* reconciliation layer: no BQ writes, no order
routing. Consumed by GET /api/paper-trading/reconciliation (4.5.3).

Design notes:
  - Signals input == paper_trades (same rows that drove real fills).
  - Shadow backtest = running PnL accumulator with yfinance adjusted-close at
    the trade timestamp, matching quantity, zero txcost, zero slippage.
  - Equity curves aligned by calendar date (paper snapshots have daily
    granularity; shadow curve is sampled at snapshot dates).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

DIVERGENCE_ALERT_THRESHOLD_PCT = 5.0


def _parse_ts(s: Any) -> Optional[datetime]:
    if s is None:
        return None
    if isinstance(s, datetime):
        return s if s.tzinfo else s.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def _to_date(s: Any) -> Optional[str]:
    dt = _parse_ts(s) if s else None
    return dt.date().isoformat() if dt else (str(s)[:10] if s else None)


def _shadow_nav_curve(
    trades: list[dict],
    price_lookup: dict[str, dict[str, float]],
    starting_capital: float,
    dates: list[str],
) -> list[float]:
    """
    Replay trades as a frictionless shadow portfolio. price_lookup maps
    ticker -> {date: adj_close}. Returns NAV at each `dates` entry (same date
    ordering as the paper snapshot curve we'll overlay).
    """
    cash = float(starting_capital)
    positions: dict[str, float] = {}  # ticker -> quantity

    trades_sorted = sorted(
        trades,
        key=lambda t: _parse_ts(t.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc),
    )
    trade_idx = 0
    curve: list[float] = []

    for d in dates:
        # Apply every trade whose date is <= current date.
        while trade_idx < len(trades_sorted):
            t = trades_sorted[trade_idx]
            t_date = _to_date(t.get("created_at"))
            if t_date is None or t_date > d:
                break
            ticker = t.get("ticker")
            qty = float(t.get("quantity") or 0.0)
            # Use paper's fill price as a fallback; prefer yfinance adj_close on trade date.
            px = price_lookup.get(ticker, {}).get(t_date) or float(t.get("price") or 0.0)
            action = (t.get("action") or "").upper()
            if px > 0 and qty > 0:
                if action == "BUY":
                    cost = qty * px
                    if cost <= cash:
                        cash -= cost
                        positions[ticker] = positions.get(ticker, 0.0) + qty
                elif action == "SELL":
                    held = positions.get(ticker, 0.0)
                    sell = min(qty, held)
                    cash += sell * px
                    positions[ticker] = held - sell
                    if positions[ticker] <= 1e-9:
                        positions.pop(ticker, None)
            trade_idx += 1

        # Mark portfolio to market at `d`.
        mv = 0.0
        for ticker, held in positions.items():
            px = price_lookup.get(ticker, {}).get(d)
            if not px:
                # fall back to most recent known price at-or-before d
                hist = price_lookup.get(ticker, {})
                candidates = [k for k in hist.keys() if k <= d]
                if candidates:
                    px = hist[max(candidates)]
            mv += held * (px or 0.0)
        curve.append(round(cash + mv, 2))

    return curve


def _fetch_prices(
    tickers: set[str], start: str, end: str
) -> dict[str, dict[str, float]]:
    """
    Fetch adjusted-close history for each ticker. Returns a nested dict:
    {ticker: {date_iso: adj_close}}. yfinance errors are swallowed; missing
    tickers simply leave an empty sub-dict.
    """
    out: dict[str, dict[str, float]] = {}
    if not tickers:
        return out
    try:
        import yfinance as yf
    except Exception as e:
        logger.warning(f"yfinance import failed for reconciliation: {e}")
        return out
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            if hist.empty:
                continue
            prices: dict[str, float] = {}
            for ts, row in hist.iterrows():
                prices[ts.date().isoformat()] = float(row["Close"])
            out[ticker] = prices
        except Exception as e:
            logger.debug(f"reconciliation: price fetch failed for {ticker}: {e}")
    return out


def compute_reconciliation(
    bq: Any,
    snapshot_limit: int = 365,
    trade_limit: int = 2000,
) -> dict:
    """
    Build the reconciliation series. Reads paper snapshots + trades, fetches
    yfinance closes for every ticker traded, simulates a frictionless shadow
    backtest aligned by date to the paper snapshot curve, and computes
    divergence_pct series + alert flag.
    """
    snapshots = bq.get_paper_snapshots(limit=snapshot_limit) or []
    trades = bq.get_paper_trades(limit=trade_limit) or []
    portfolio = bq.get_paper_portfolio("default") or {}

    if len(snapshots) < 2:
        return {
            "series": [],
            "summary": {
                "n_points": 0,
                "max_divergence_pct": 0.0,
                "latest_divergence_pct": 0.0,
                "alert": False,
                "alert_threshold_pct": DIVERGENCE_ALERT_THRESHOLD_PCT,
            },
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "note": "insufficient_snapshots",
        }

    # Ensure chronological order (oldest -> newest).
    snaps = sorted(snapshots, key=lambda s: str(s.get("snapshot_date")))
    dates = [str(s.get("snapshot_date"))[:10] for s in snaps]
    paper_navs = [float(s.get("total_nav") or 0.0) for s in snaps]

    tickers = {t.get("ticker") for t in trades if t.get("ticker")}
    start = dates[0]
    end = dates[-1]
    prices = _fetch_prices(tickers, start, end)

    starting_capital = float(portfolio.get("starting_capital") or paper_navs[0])
    shadow_navs = _shadow_nav_curve(trades, prices, starting_capital, dates)

    series = []
    max_div = 0.0
    latest_div = 0.0
    for i, d in enumerate(dates):
        paper = paper_navs[i]
        shadow = shadow_navs[i] if i < len(shadow_navs) else 0.0
        if shadow > 0:
            div = abs(paper - shadow) / shadow * 100.0
        else:
            div = 0.0
        max_div = max(max_div, div)
        latest_div = div
        series.append({
            "date": d,
            "paper_nav": round(paper, 2),
            "backtest_nav": round(shadow, 2),
            "divergence_pct": round(div, 4),
        })

    alert = latest_div > DIVERGENCE_ALERT_THRESHOLD_PCT

    return {
        "series": series,
        "summary": {
            "n_points": len(series),
            "max_divergence_pct": round(max_div, 4),
            "latest_divergence_pct": round(latest_div, 4),
            "alert": bool(alert),
            "alert_threshold_pct": DIVERGENCE_ALERT_THRESHOLD_PCT,
        },
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "note": None,
    }
