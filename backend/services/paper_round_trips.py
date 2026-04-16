"""
paper_round_trips -- Round-trip (BUY->SELL) performance metrics.

Pairs historic paper trades per-ticker using FIFO matching, then computes the
aggregate stats used by the Go-Live Gate (4.5.4) and the /round-trips endpoint:

  - win_rate              = wins / closed_trades
  - profit_factor         = sum(wins$) / |sum(losses$)|
  - expectancy_pct        = win_rate * avg_win_pct + (1 - win_rate) * avg_loss_pct
  - median_holding_days
  - avg_mfe_pct / avg_mae_pct / avg_capture_ratio

Definitions follow the industry standard described in Tharp (2007), _Trade Your
Way to Financial Freedom_ -- also adopted by QuantifiedStrategies.com:
  https://www.quantifiedstrategies.com/profit-factor/
  https://www.quantifiedstrategies.com/expectancy-formula/

This module does no math outside its pairing loop -- per backend-services
convention, aggregate statistics are plain arithmetic, not portfolio Sharpe.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _parse_ts(s: Any) -> Optional[datetime]:
    if s is None:
        return None
    if isinstance(s, datetime):
        return s if s.tzinfo else s.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def pair_round_trips(trades: list[dict]) -> list[dict]:
    """
    FIFO-match BUY -> SELL trades per ticker. Returns a list of closed round-trips
    in chronological (exit_date) order. Partial exits are handled by splitting
    buy-side inventory proportionally.

    Each output row contains: ticker, entry_date, exit_date, entry_price,
    exit_price, quantity, realized_pnl_pct, holding_days, mfe_pct, mae_pct,
    capture_ratio (and optional metadata copied through from trade rows).
    """
    # Order chronologically, oldest first.
    by_time = sorted(
        trades,
        key=lambda t: (_parse_ts(t.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc)),
    )
    inventory: dict[str, list[dict]] = {}
    round_trips: list[dict] = []

    for t in by_time:
        ticker = t.get("ticker")
        action = (t.get("action") or "").upper()
        qty = float(t.get("quantity") or 0.0)
        price = float(t.get("price") or 0.0)
        if not ticker or qty <= 0 or price <= 0:
            continue

        if action == "BUY":
            inventory.setdefault(ticker, []).append({
                "qty_remaining": qty,
                "price": price,
                "trade_id": t.get("trade_id"),
                "entry_ts": _parse_ts(t.get("created_at")),
            })
            continue

        if action != "SELL":
            continue

        # Match sell quantity against FIFO buy lots.
        sell_qty = qty
        exit_ts = _parse_ts(t.get("created_at"))
        lots = inventory.get(ticker, [])
        while sell_qty > 1e-9 and lots:
            lot = lots[0]
            matched = min(sell_qty, lot["qty_remaining"])
            entry_price = lot["price"]
            realized_pnl_pct = (
                ((price - entry_price) / entry_price) * 100.0 if entry_price > 0 else 0.0
            )
            entry_ts = lot["entry_ts"]
            holding_days = (
                int((exit_ts - entry_ts).days) if (exit_ts and entry_ts) else 0
            )
            mfe = float(t.get("mfe_pct") or 0.0)
            mae = float(t.get("mae_pct") or 0.0)
            capture = realized_pnl_pct / mfe if mfe > 0 else 0.0

            round_trips.append({
                "ticker": ticker,
                "buy_trade_id": lot.get("trade_id"),
                "sell_trade_id": t.get("trade_id"),
                "entry_date": entry_ts.isoformat() if entry_ts else None,
                "exit_date": exit_ts.isoformat() if exit_ts else None,
                "entry_price": round(entry_price, 4),
                "exit_price": round(price, 4),
                "quantity": round(matched, 6),
                "realized_pnl_pct": round(realized_pnl_pct, 4),
                "realized_pnl_usd": round((price - entry_price) * matched, 2),
                "holding_days": holding_days,
                "mfe_pct": round(mfe, 4),
                "mae_pct": round(mae, 4),
                "capture_ratio": round(capture, 4),
                "exit_reason": t.get("reason", ""),
            })

            lot["qty_remaining"] -= matched
            sell_qty -= matched
            if lot["qty_remaining"] <= 1e-9:
                lots.pop(0)

        # SELLs without a matching BUY (data drift) are intentionally dropped.

    return round_trips


def summarize(round_trips: list[dict]) -> dict:
    """Aggregate win_rate, profit_factor, expectancy, holding-period stats."""
    n = len(round_trips)
    if n == 0:
        return {
            "n_round_trips": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "median_holding_days": 0,
            "avg_mfe_pct": 0.0,
            "avg_mae_pct": 0.0,
            "avg_capture_ratio": 0.0,
        }

    wins = [rt for rt in round_trips if rt["realized_pnl_pct"] > 0]
    losses = [rt for rt in round_trips if rt["realized_pnl_pct"] <= 0]
    win_rate = len(wins) / n
    gross_win = sum(rt["realized_pnl_usd"] for rt in wins)
    gross_loss = abs(sum(rt["realized_pnl_usd"] for rt in losses))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else 0.0
    avg_win = sum(rt["realized_pnl_pct"] for rt in wins) / len(wins) if wins else 0.0
    avg_loss = sum(rt["realized_pnl_pct"] for rt in losses) / len(losses) if losses else 0.0
    expectancy = win_rate * avg_win + (1.0 - win_rate) * avg_loss
    holding = sorted(rt["holding_days"] for rt in round_trips)
    median_h = holding[n // 2] if n % 2 == 1 else (holding[n // 2 - 1] + holding[n // 2]) // 2
    avg_mfe = sum(rt["mfe_pct"] for rt in round_trips) / n
    avg_mae = sum(rt["mae_pct"] for rt in round_trips) / n
    avg_capture = sum(rt["capture_ratio"] for rt in round_trips) / n

    return {
        "n_round_trips": n,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "expectancy_pct": round(expectancy, 4),
        "avg_win_pct": round(avg_win, 4),
        "avg_loss_pct": round(avg_loss, 4),
        "median_holding_days": int(median_h),
        "avg_mfe_pct": round(avg_mfe, 4),
        "avg_mae_pct": round(avg_mae, 4),
        "avg_capture_ratio": round(avg_capture, 4),
    }


def compute_round_trips_response(bq: Any, trade_limit: int = 2000) -> dict:
    """Fetch trades from BQ, pair them, and return the endpoint response dict."""
    trades = bq.get_paper_trades(limit=trade_limit) or []
    rts = pair_round_trips(trades)
    summary = summarize(rts)
    return {
        **summary,
        "round_trips": rts[-50:],  # last 50 closed for the UI table
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
