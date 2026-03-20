"""
Paper Trading Engine — executes virtual trades, tracks positions, computes NAV.

All persistence through BigQueryClient. No real money involved.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf

from backend.config.settings import Settings
from backend.db.bigquery_client import BigQueryClient

logger = logging.getLogger(__name__)


class PaperTrader:
    """Virtual trade execution engine backed by BigQuery."""

    def __init__(self, settings: Settings, bq_client: BigQueryClient):
        self.settings = settings
        self.bq = bq_client

    # ── Portfolio State ──────────────────────────────────────────

    def get_or_create_portfolio(self) -> dict:
        """Load portfolio from BQ, or initialize with starting capital."""
        portfolio = self.bq.get_paper_portfolio("default")
        if portfolio:
            return portfolio
        now = datetime.now(timezone.utc).isoformat()
        row = {
            "portfolio_id": "default",
            "starting_capital": self.settings.paper_starting_capital,
            "current_cash": self.settings.paper_starting_capital,
            "total_nav": self.settings.paper_starting_capital,
            "total_pnl_pct": 0.0,
            "benchmark_return_pct": 0.0,
            "inception_date": now,
            "updated_at": now,
        }
        self.bq.upsert_paper_portfolio(row)
        logger.info(f"Initialized paper portfolio with ${self.settings.paper_starting_capital:,.0f}")
        return row

    def get_positions(self) -> list[dict]:
        return self.bq.get_paper_positions()

    def get_position(self, ticker: str) -> Optional[dict]:
        return self.bq.get_paper_position(ticker)

    # ── Trade Execution ──────────────────────────────────────────

    def execute_buy(
        self,
        ticker: str,
        amount_usd: float,
        price: float,
        reason: str = "new_buy_signal",
        analysis_id: str = "",
        risk_judge_decision: str = "",
        stop_loss_price: Optional[float] = None,
        risk_judge_position_pct: Optional[float] = None,
    ) -> Optional[dict]:
        """Buy shares of a ticker. Returns the trade record or None if can't execute."""
        portfolio = self.get_or_create_portfolio()
        cash = portfolio["current_cash"]

        # Transaction cost
        tx_cost = amount_usd * (self.settings.paper_transaction_cost_pct / 100.0)
        total_cost = amount_usd + tx_cost

        if total_cost > cash:
            logger.warning(f"Insufficient cash for {ticker}: need ${total_cost:.2f}, have ${cash:.2f}")
            return None

        # Check max positions
        positions = self.get_positions()
        existing = next((p for p in positions if p["ticker"] == ticker), None)
        if not existing and len(positions) >= self.settings.paper_max_positions:
            logger.warning(f"Max positions ({self.settings.paper_max_positions}) reached, skipping {ticker}")
            return None

        quantity = amount_usd / price
        now = datetime.now(timezone.utc).isoformat()

        # Record trade
        trade = {
            "trade_id": str(uuid.uuid4()),
            "ticker": ticker,
            "action": "BUY",
            "quantity": round(quantity, 6),
            "price": price,
            "total_value": round(amount_usd, 2),
            "transaction_cost": round(tx_cost, 2),
            "reason": reason,
            "analysis_id": analysis_id,
            "risk_judge_decision": risk_judge_decision,
            "created_at": now,
        }
        self.bq.save_paper_trade(trade)

        # Update or create position
        if existing:
            old_qty = existing["quantity"]
            old_cost = existing["cost_basis"] or (old_qty * existing["avg_entry_price"])
            new_qty = old_qty + quantity
            new_cost = old_cost + amount_usd
            new_avg = new_cost / new_qty
            self.bq.delete_paper_position(ticker)
            pos_row = {
                "position_id": existing["position_id"],
                "ticker": ticker,
                "quantity": round(new_qty, 6),
                "avg_entry_price": round(new_avg, 4),
                "cost_basis": round(new_cost, 2),
                "current_price": price,
                "market_value": round(new_qty * price, 2),
                "unrealized_pnl": round(new_qty * price - new_cost, 2),
                "unrealized_pnl_pct": round(((new_qty * price - new_cost) / new_cost) * 100, 2),
                "entry_date": existing["entry_date"],
                "last_analysis_date": now,
                "recommendation": reason,
                "risk_judge_position_pct": risk_judge_position_pct,
                "stop_loss_price": stop_loss_price,
            }
            self.bq.save_paper_position(pos_row)
        else:
            pos_row = {
                "position_id": str(uuid.uuid4()),
                "ticker": ticker,
                "quantity": round(quantity, 6),
                "avg_entry_price": price,
                "cost_basis": round(amount_usd, 2),
                "current_price": price,
                "market_value": round(amount_usd, 2),
                "unrealized_pnl": 0.0,
                "unrealized_pnl_pct": 0.0,
                "entry_date": now,
                "last_analysis_date": now,
                "recommendation": reason,
                "risk_judge_position_pct": risk_judge_position_pct,
                "stop_loss_price": stop_loss_price,
            }
            self.bq.save_paper_position(pos_row)

        # Update cash
        new_cash = cash - total_cost
        self._update_portfolio_cash(new_cash)

        logger.info(f"BUY {quantity:.4f} x {ticker} @ ${price:.2f} = ${amount_usd:.2f} (fee: ${tx_cost:.2f})")
        return trade

    def execute_sell(
        self,
        ticker: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        reason: str = "signal_flip",
    ) -> Optional[dict]:
        """Sell shares. If quantity is None, sells entire position. Returns trade record."""
        position = self.get_position(ticker)
        if not position:
            logger.warning(f"No position in {ticker} to sell")
            return None

        if price is None:
            price = _get_live_price(ticker) or position.get("current_price", 0)

        sell_qty = quantity or position["quantity"]
        sell_qty = min(sell_qty, position["quantity"])
        sell_value = sell_qty * price
        tx_cost = sell_value * (self.settings.paper_transaction_cost_pct / 100.0)
        net_proceeds = sell_value - tx_cost
        now = datetime.now(timezone.utc).isoformat()

        # Record trade
        trade = {
            "trade_id": str(uuid.uuid4()),
            "ticker": ticker,
            "action": "SELL",
            "quantity": round(sell_qty, 6),
            "price": price,
            "total_value": round(sell_value, 2),
            "transaction_cost": round(tx_cost, 2),
            "reason": reason,
            "analysis_id": "",
            "risk_judge_decision": "",
            "created_at": now,
        }
        self.bq.save_paper_trade(trade)

        remaining = position["quantity"] - sell_qty
        if remaining < 0.0001:
            # Full exit
            self.bq.delete_paper_position(ticker)
        else:
            # Partial - delete and re-insert with reduced quantity
            self.bq.delete_paper_position(ticker)
            pos_row = {
                "position_id": position["position_id"],
                "ticker": ticker,
                "quantity": round(remaining, 6),
                "avg_entry_price": position["avg_entry_price"],
                "cost_basis": round(remaining * position["avg_entry_price"], 2),
                "current_price": price,
                "market_value": round(remaining * price, 2),
                "unrealized_pnl": round(remaining * (price - position["avg_entry_price"]), 2),
                "unrealized_pnl_pct": round(((price - position["avg_entry_price"]) / position["avg_entry_price"]) * 100, 2),
                "entry_date": position["entry_date"],
                "last_analysis_date": position.get("last_analysis_date", ""),
                "recommendation": position.get("recommendation", ""),
                "risk_judge_position_pct": position.get("risk_judge_position_pct"),
                "stop_loss_price": position.get("stop_loss_price"),
            }
            self.bq.save_paper_position(pos_row)

        # Update cash
        portfolio = self.get_or_create_portfolio()
        new_cash = portfolio["current_cash"] + net_proceeds
        self._update_portfolio_cash(new_cash)

        logger.info(f"SELL {sell_qty:.4f} x {ticker} @ ${price:.2f} = ${sell_value:.2f} (fee: ${tx_cost:.2f})")
        return trade

    # ── Mark-to-Market ───────────────────────────────────────────

    def mark_to_market(self) -> dict:
        """Update all positions with live prices. Returns portfolio summary."""
        positions = self.get_positions()
        portfolio = self.get_or_create_portfolio()
        total_positions_value = 0.0

        for pos in positions:
            ticker = pos["ticker"]
            live_price = _get_live_price(ticker)
            if live_price is None:
                live_price = pos.get("current_price", pos["avg_entry_price"])

            market_value = pos["quantity"] * live_price
            cost_basis = pos.get("cost_basis") or (pos["quantity"] * pos["avg_entry_price"])
            pnl = market_value - cost_basis
            pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0

            self.bq.delete_paper_position(ticker)
            pos.update({
                "current_price": live_price,
                "market_value": round(market_value, 2),
                "unrealized_pnl": round(pnl, 2),
                "unrealized_pnl_pct": round(pnl_pct, 2),
            })
            self.bq.save_paper_position(pos)
            total_positions_value += market_value

        nav = portfolio["current_cash"] + total_positions_value
        starting = portfolio["starting_capital"]
        pnl_pct = ((nav - starting) / starting) * 100 if starting > 0 else 0.0
        benchmark_ret = _get_benchmark_return(portfolio.get("inception_date", ""))

        self.bq.upsert_paper_portfolio({
            **portfolio,
            "total_nav": round(nav, 2),
            "total_pnl_pct": round(pnl_pct, 2),
            "benchmark_return_pct": round(benchmark_ret, 2) if benchmark_ret else portfolio.get("benchmark_return_pct", 0.0),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

        return {
            "nav": round(nav, 2),
            "cash": portfolio["current_cash"],
            "positions_value": round(total_positions_value, 2),
            "pnl_pct": round(pnl_pct, 2),
            "benchmark_return_pct": round(benchmark_ret, 2) if benchmark_ret else 0.0,
            "position_count": len(positions),
        }

    # ── Stop Loss Check ──────────────────────────────────────────

    def check_stop_losses(self) -> list[str]:
        """Return tickers where current price is at or below stop loss."""
        positions = self.get_positions()
        triggered = []
        for pos in positions:
            stop = pos.get("stop_loss_price")
            current = pos.get("current_price", 0)
            if stop and current and current <= stop:
                triggered.append(pos["ticker"])
        return triggered

    # ── Snapshot ─────────────────────────────────────────────────

    def save_daily_snapshot(self, trades_today: int = 0, analysis_cost_today: float = 0.0) -> dict:
        """Save daily NAV snapshot. Call after mark_to_market."""
        portfolio = self.get_or_create_portfolio()
        positions = self.get_positions()
        positions_value = sum(p.get("market_value", 0) for p in positions)
        nav = portfolio.get("total_nav", portfolio["current_cash"])
        starting = portfolio["starting_capital"]
        cum_pnl = ((nav - starting) / starting) * 100 if starting > 0 else 0.0
        benchmark = portfolio.get("benchmark_return_pct", 0.0)

        # Get yesterday's snapshot for daily P&L
        snapshots = self.bq.get_paper_snapshots(limit=1)
        prev_nav = snapshots[0].get("total_nav", starting) if snapshots else starting
        daily_pnl = ((nav - prev_nav) / prev_nav) * 100 if prev_nav > 0 else 0.0

        snap = {
            "snapshot_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "total_nav": round(nav, 2),
            "cash": round(portfolio["current_cash"], 2),
            "positions_value": round(positions_value, 2),
            "daily_pnl_pct": round(daily_pnl, 2),
            "cumulative_pnl_pct": round(cum_pnl, 2),
            "benchmark_pnl_pct": round(benchmark, 2),
            "alpha_pct": round(cum_pnl - benchmark, 2),
            "position_count": len(positions),
            "trades_today": trades_today,
            "analysis_cost_today": round(analysis_cost_today, 4),
        }
        self.bq.save_paper_snapshot(snap)
        return snap

    # ── Internal ─────────────────────────────────────────────────

    def _update_portfolio_cash(self, new_cash: float) -> None:
        portfolio = self.get_or_create_portfolio()
        portfolio["current_cash"] = round(new_cash, 2)
        portfolio["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.bq.upsert_paper_portfolio(portfolio)


def _get_live_price(ticker: str) -> Optional[float]:
    """Fetch latest price from yfinance. Returns None on error."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.debug(f"Could not get live price for {ticker}: {e}")
    return None


def _get_benchmark_return(inception_date: str) -> Optional[float]:
    """SPY return since portfolio inception."""
    if not inception_date:
        return None
    try:
        spy = yf.Ticker("SPY")
        start = inception_date[:10]
        hist = spy.history(start=start)
        if len(hist) >= 2:
            first = float(hist["Close"].iloc[0])
            last = float(hist["Close"].iloc[-1])
            return ((last - first) / first) * 100
    except Exception as e:
        logger.debug(f"Could not compute benchmark return: {e}")
    return None
