"""Zero-orders drill (BLOCKER-1 verification).

Proves the decide_trades -> execute_buy path works end-to-end against a
synthetic BUY analysis, so a fix to Claude's HOLD bias is provably
plumbed through to paper_trades writes.

Exits 0 and prints "PASS" on success, exits 1 on failure.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config.settings import Settings
from backend.services.portfolio_manager import decide_trades, TradeOrder
from backend.services.paper_trader import PaperTrader


class StubBQ:
    """In-memory BigQueryClient stub that records method calls."""

    def __init__(self, starting_cash: float = 10000.0) -> None:
        self.saved_trades: list[dict] = []
        self.saved_positions: list[dict] = []
        self.portfolio = {
            "portfolio_id": "default",
            "starting_capital": starting_cash,
            "current_cash": starting_cash,
            "total_nav": starting_cash,
            "total_pnl_pct": 0.0,
            "benchmark_return_pct": 0.0,
            "inception_date": "2026-04-24T00:00:00+00:00",
            "updated_at": "2026-04-24T00:00:00+00:00",
        }

    def get_paper_portfolio(self, pid: str = "default"):
        return dict(self.portfolio)

    def upsert_paper_portfolio(self, row: dict) -> None:
        self.portfolio.update(row)

    def get_paper_positions(self):
        return list(self.saved_positions)

    def get_paper_position(self, ticker: str):
        return next((p for p in self.saved_positions if p["ticker"] == ticker), None)

    def save_paper_position(self, row: dict) -> None:
        self.saved_positions.append(dict(row))

    def update_paper_position(self, ticker: str, updates: dict) -> None:
        for p in self.saved_positions:
            if p["ticker"] == ticker:
                p.update(updates)

    def save_paper_trade(self, row: dict) -> None:
        self.saved_trades.append(dict(row))


def main() -> int:
    settings = Settings()
    bq = StubBQ(starting_cash=10000.0)

    synthetic_buy = {
        "ticker": "AAPL",
        "recommendation": "BUY",
        "final_score": 8,
        "price_at_analysis": 195.0,
        "risk_assessment": {"decision": "APPROVE", "recommended_position_pct": 10.0},
        "analysis_date": "2026-04-24T12:00:00+00:00",
    }
    portfolio_state = {"nav": 10000.0, "cash": 10000.0, "positions_value": 0.0, "position_count": 0}

    orders = decide_trades(
        current_positions=[],
        candidate_analyses=[synthetic_buy],
        holding_analyses=[],
        portfolio_state=portfolio_state,
        settings=settings,
    )

    buy_orders = [o for o in orders if isinstance(o, TradeOrder) and o.action == "BUY"]
    if len(buy_orders) != 1 or buy_orders[0].ticker != "AAPL":
        print(f"FAIL: expected 1 BUY for AAPL, got {len(buy_orders)} orders: "
              f"{[(o.action, o.ticker) for o in orders]}")
        return 1
    print(f"step1: decide_trades emitted BUY for {buy_orders[0].ticker} "
          f"amount=${buy_orders[0].amount_usd:.2f}")

    trader = PaperTrader(settings=settings, bq_client=bq)
    order = buy_orders[0]
    result = trader.execute_buy(
        ticker=order.ticker,
        amount_usd=order.amount_usd or 0,
        price=order.price or 195.0,
        reason=order.reason,
        analysis_id=order.analysis_id,
        risk_judge_decision=order.risk_judge_decision,
        stop_loss_price=order.stop_loss_price,
        risk_judge_position_pct=order.risk_judge_position_pct,
        signals=order.signals,
    )

    if result is None:
        print("FAIL: execute_buy returned None (refused to execute)")
        return 1
    if not bq.saved_trades:
        print("FAIL: execute_buy did not call save_paper_trade")
        return 1
    trade = bq.saved_trades[0]
    if trade.get("action") != "BUY" or trade.get("ticker") != "AAPL":
        print(f"FAIL: unexpected trade row: {trade}")
        return 1
    print(f"step2: paper_trades row written: ticker={trade['ticker']} "
          f"action={trade['action']} qty={trade.get('quantity')} price={trade.get('price')}")

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
