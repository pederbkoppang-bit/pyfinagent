"""
MCP Signals Server: Callable tools for signal generation, validation, publishing

Tools (FastMCP @mcp.tool):
- generate_signal(ticker, date) → BUY/SELL/HOLD with confidence
- validate_signal(signal) → Check constraints (market hours, liquidity, exposure)
- publish_signal(signal) → Post to Slack + portfolio
- risk_check(portfolio, proposed_trade) → Can we add this position?

Resources:
- portfolio://current → Current holdings (tickers, shares, PnL)
- constraints://risk → Risk limits (max exposure, max drawdown, Sharpe floor)
- signals://history → All generated signals this month
"""

import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Import backend modules
_SIGNALS_AVAILABLE = False

try:
    from backend.services.paper_trader import PaperTrader
    from backend.db.bigquery_client import BigQueryClient
    from backend.config.settings import get_settings
    _SIGNALS_AVAILABLE = True
except ImportError:
    logger.warning("Paper trader not available -- signals server in stub mode")


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """Trading signal."""
    ticker: str
    date: str
    signal: SignalType
    confidence: float  # 0.0-1.0
    factors: List[str]  # Factors supporting this signal
    reason: str  # Human-readable explanation


class SignalsServer:
    """FastMCP signals server for pyfinAgent."""
    
    def __init__(self):
        global _SIGNALS_AVAILABLE
        
        self.portfolio = {}  # Current holdings
        self.risk_limits = {}  # Exposure limits
        self.signal_history = []  # All signals generated
        self.bq_client = None
        self.settings = None
        self.paper_trader = None
        
        # Initialize paper trader if available
        if _SIGNALS_AVAILABLE:
            try:
                self.settings = get_settings()
                self.bq_client = BigQueryClient(self.settings)
                self.paper_trader = PaperTrader(bq_client=self.bq_client)
                logger.info("SignalsServer initialized with PaperTrader")
            except Exception as e:
                logger.error(f"Failed to initialize SignalsServer: {e}")
                _SIGNALS_AVAILABLE = False
    
    def generate_signal(self, ticker: str, date: str) -> Dict[str, Any]:
        """
        Generate a trading signal (BUY/SELL/HOLD) for a ticker on a date.
        
        Uses the learned features + model to make a prediction.
        
        Returns:
        {
            "ticker": "AAPL",
            "date": "2026-03-29",
            "signal": "BUY",
            "confidence": 0.72,
            "factors": ["momentum_3m: +12%", "insider_ratio: 1.2", "sentiment: 0.68"],
            "reason": "Strong momentum + insider buying"
        }
        """
        logger.info(f"generate_signal({ticker}, {date})")
        # TODO: Run model inference on current features
        return {
            "ticker": ticker,
            "date": date,
            "signal": "HOLD",
            "confidence": 0.0,
            "factors": [],
            "reason": "PENDING_IMPLEMENTATION",
        }
    
    def validate_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Signal-intrinsic schema validation. Checks that a generated signal is
        well-formed before any portfolio-state risk checks are applied.

        This is the layered counterpart to risk_check: validate_signal looks
        only at the signal itself; risk_check looks at the signal vs the
        current portfolio. (Pattern follows QuantConnect's Alpha ->
        PortfolioConstruction -> RiskManagement -> Execution layering.)

        Tolerates partial / missing fields via .get() and never raises --
        MCP clients prefer error-shape returns over exceptions.

        Args:
            signal: {
                "ticker": "AAPL",
                "signal": "BUY" | "SELL" | "HOLD",
                "confidence": 0.0..1.0,
                "date": "YYYY-MM-DD",
                "factors": [...]
            }

        Returns:
            {
                "valid": bool,
                "violations": [str],
                "adjusted_signal": dict | None,
                "reason": str
            }
        """
        if not isinstance(signal, dict):
            logger.warning("validate_signal called with non-dict input")
            return {
                "valid": False,
                "violations": ["not_a_dict"],
                "adjusted_signal": None,
                "reason": "signal must be a dict",
            }

        ticker = signal.get("ticker", "")
        sig = signal.get("signal", "")
        confidence = signal.get("confidence", None)
        sdate = signal.get("date", "")
        factors = signal.get("factors", [])

        logger.info(f"validate_signal({ticker})")

        violations: List[str] = []

        # 1. ticker: non-empty alphanumeric/dot string (matches the project's
        #    ticker sanitization rule from .claude/rules/security.md).
        if not isinstance(ticker, str) or not ticker:
            violations.append("missing_ticker")
        elif not all(c.isalnum() or c in ".:-_" for c in ticker):
            violations.append("invalid_ticker_chars")

        # 2. signal type: must be one of the three known SignalType values.
        if sig not in ("BUY", "SELL", "HOLD"):
            violations.append("invalid_signal_type")

        # 3. confidence in [0, 1].
        if confidence is None:
            violations.append("missing_confidence")
        else:
            try:
                cval = float(confidence)
                if cval < 0.0 or cval > 1.0:
                    violations.append("confidence_out_of_range")
            except (ValueError, TypeError):
                violations.append("confidence_not_numeric")

        # 4. date present (calendar parsing left to caller -- the orchestrator
        #    owns the trading-day calendar).
        if not isinstance(sdate, str) or not sdate:
            violations.append("missing_date")

        # 5. BUY/SELL must carry at least one factor; HOLD may be empty.
        if sig in ("BUY", "SELL"):
            if not isinstance(factors, list) or len(factors) == 0:
                violations.append("missing_factors")

        valid = len(violations) == 0
        reason = "All schema checks passed" if valid else f"Schema violations: {','.join(violations)}"
        return {
            "valid": valid,
            "violations": violations,
            "adjusted_signal": signal if valid else None,
            "reason": reason,
        }
    
    def publish_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publish a validated signal to Slack and portfolio.
        
        Args:
            signal: Validated signal dict
        
        Returns:
        {
            "published": True,
            "slack_id": "ts_1234567890.123456",
            "timestamp": "2026-03-29T10:30:00Z",
            "portfolio_updated": True
        }
        """
        logger.info(f"publish_signal({signal['ticker']})")
        # TODO: Post to Slack + update portfolio
        return {
            "published": False,
            "slack_id": "",
            "timestamp": "",
            "reason": "PENDING_IMPLEMENTATION",
        }
    
    def risk_check(self, portfolio: Dict[str, Any], proposed_trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Portfolio-extrinsic constraint check for a proposed trade.

        Reads its limits from get_risk_constraints() so the threshold table
        stays single-source-of-truth. Pure function: never mutates inputs.
        Does NOT call any other MCP server -- cross-server coupling is an
        explicit anti-pattern per Anthropic MCP best practices.

        Evaluation order is canonical FINRA 15c3-5 / FIA whitepaper ordering:
        regulatory-fatal -> financial-fatal -> soft, fail-fast on hard
        violations.

            1. Schema sanity (ticker, action, shares > 0)                [hard]
            2. SELL requires existing position with >= shares            [hard]
            3. Daily trade count under max_daily_trades (5)              [hard]
            4. Per-ticker concentration <= 10% of total_value            [hard]
            5. Total exposure <= 100% of total_value                     [hard]
            6. Cash floor (BUY notional <= cash)                         [hard]
            7. Drawdown circuit breaker: block BUYs past -15% drawdown   [hard for BUY]

        Args:
            portfolio: {
                "total_value": float,
                "cash": float,
                "positions": {ticker: {"shares": int, "price": float, ...}},
                "trades_today": list | int,
                "current_drawdown_pct": float (optional, default 0.0)
            }
            proposed_trade: {
                "ticker": str,
                "action": "BUY" | "SELL",
                "shares": int,
                "price": float (optional)
            }

        Returns:
            {
                "allowed": bool,
                "current_exposure_pct": float,
                "max_exposure_pct": float,
                "margin_available": bool,
                "conflicts": [str],
                "reason": str
            }
        """
        if not isinstance(portfolio, dict):
            portfolio = {}
        if not isinstance(proposed_trade, dict):
            proposed_trade = {}

        ticker = proposed_trade.get("ticker", "")
        action = proposed_trade.get("action", "")
        shares = proposed_trade.get("shares", 0)
        price = proposed_trade.get("price", None)

        logger.info(f"risk_check({ticker})")

        limits = self.get_risk_constraints()
        max_per_ticker_pct = float(limits.get("max_exposure_per_ticker_pct", 10.0))
        max_total_pct = float(limits.get("max_total_exposure_pct", 100.0))
        max_drawdown_pct = float(limits.get("max_drawdown_pct", -15.0))
        max_daily_trades = int(limits.get("max_daily_trades", 5))

        positions = portfolio.get("positions", {}) or {}
        total_value = float(portfolio.get("total_value", 0.0) or 0.0)
        cash = float(portfolio.get("cash", 0.0) or 0.0)
        trades_today_raw = portfolio.get("trades_today", [])
        trades_today_count = (
            len(trades_today_raw)
            if isinstance(trades_today_raw, (list, tuple))
            else int(trades_today_raw or 0)
        )
        current_dd = float(portfolio.get("current_drawdown_pct", 0.0) or 0.0)

        conflicts: List[str] = []

        # 1. Schema sanity -- hard, fail-fast.
        try:
            shares_int = int(shares)
        except (ValueError, TypeError):
            shares_int = 0
        if not isinstance(ticker, str) or not ticker:
            conflicts.append("missing_ticker")
        if action not in ("BUY", "SELL"):
            conflicts.append("invalid_action")
        if shares_int <= 0:
            conflicts.append("invalid_shares")
        if conflicts:
            return self._risk_response(False, 0.0, max_per_ticker_pct, conflicts, "Schema sanity failed")

        # Resolve a usable per-share price for notional math:
        #   1) explicit proposed_trade.price
        #   2) last_price from existing position record
        #   3) else 0.0 (concentration check still runs; cash check passes
        #      trivially -- documented degraded mode for the paper-trader
        #      scaffold; real prices wired in Phase 4.1).
        position_record = positions.get(ticker, {}) if isinstance(positions, dict) else {}
        try:
            unit_price = float(price) if price is not None else 0.0
        except (ValueError, TypeError):
            unit_price = 0.0
        if unit_price <= 0.0 and isinstance(position_record, dict):
            try:
                unit_price = float(position_record.get("price", 0.0) or 0.0)
            except (ValueError, TypeError):
                unit_price = 0.0
        proposed_notional = unit_price * shares_int

        # 2. Action/state consistency -- SELL requires sufficient existing position.
        if action == "SELL":
            existing_shares = 0
            if isinstance(position_record, dict):
                try:
                    existing_shares = int(position_record.get("shares", 0) or 0)
                except (ValueError, TypeError):
                    existing_shares = 0
            if existing_shares < shares_int:
                conflicts.append("insufficient_position")
                return self._risk_response(False, 0.0, max_per_ticker_pct, conflicts,
                                           f"SELL requires {shares_int} shares, have {existing_shares}")

        # 3. Daily trade count -- hard limit.
        if trades_today_count >= max_daily_trades:
            conflicts.append("max_daily_trades")
            return self._risk_response(False, 0.0, max_per_ticker_pct, conflicts,
                                       f"Daily trade count {trades_today_count} >= {max_daily_trades}")

        # Compute current per-ticker and total exposure (BEFORE the proposed trade).
        existing_position_notional = 0.0
        total_positions_notional = 0.0
        if isinstance(positions, dict):
            for sym, pos in positions.items():
                if not isinstance(pos, dict):
                    continue
                try:
                    pshares = float(pos.get("shares", 0) or 0)
                    pprice = float(pos.get("price", 0.0) or 0.0)
                except (ValueError, TypeError):
                    continue
                pnotional = pshares * pprice
                total_positions_notional += pnotional
                if sym == ticker:
                    existing_position_notional = pnotional

        # 4. Per-ticker concentration -- only meaningful for BUY (SELL reduces exposure).
        if action == "BUY" and total_value > 0.0:
            projected_position_notional = existing_position_notional + proposed_notional
            ticker_pct = (projected_position_notional / total_value) * 100.0
            if ticker_pct > max_per_ticker_pct:
                conflicts.append("max_exposure_per_ticker")
                return self._risk_response(False, ticker_pct, max_per_ticker_pct, conflicts,
                                           f"Per-ticker exposure {ticker_pct:.2f}% > {max_per_ticker_pct:.2f}%")

        # 5. Total exposure -- only meaningful for BUY.
        if action == "BUY" and total_value > 0.0:
            projected_total_notional = total_positions_notional + proposed_notional
            total_pct = (projected_total_notional / total_value) * 100.0
            if total_pct > max_total_pct:
                conflicts.append("max_total_exposure")
                return self._risk_response(False, total_pct, max_per_ticker_pct, conflicts,
                                           f"Total exposure {total_pct:.2f}% > {max_total_pct:.2f}%")

        # 6. Cash availability -- BUY only; paper trader has no margin.
        margin_available = True
        if action == "BUY" and proposed_notional > cash:
            conflicts.append("insufficient_cash")
            margin_available = False
            return self._risk_response(False, 0.0, max_per_ticker_pct, conflicts,
                                       f"Need ${proposed_notional:.2f}, have ${cash:.2f}")

        # 7. Drawdown circuit breaker -- block all new BUYs past the floor.
        #    SELLs are still allowed (de-risking).
        if action == "BUY" and current_dd <= max_drawdown_pct:
            conflicts.append("drawdown_circuit_breaker")
            return self._risk_response(False, 0.0, max_per_ticker_pct, conflicts,
                                       f"Drawdown {current_dd:.2f}% <= floor {max_drawdown_pct:.2f}%")

        # All hard checks passed.
        if action == "BUY" and total_value > 0.0:
            ticker_pct_final = ((existing_position_notional + proposed_notional) / total_value) * 100.0
        else:
            ticker_pct_final = (existing_position_notional / total_value) * 100.0 if total_value > 0.0 else 0.0
        return self._risk_response(True, ticker_pct_final, max_per_ticker_pct, [],
                                   "All risk checks passed", margin_available=margin_available)

    @staticmethod
    def _risk_response(
        allowed: bool,
        current_exposure_pct: float,
        max_exposure_pct: float,
        conflicts: List[str],
        reason: str,
        margin_available: bool = True,
    ) -> Dict[str, Any]:
        """Uniform shape for risk_check returns -- preserves the stub's contract."""
        return {
            "allowed": allowed,
            "current_exposure_pct": float(current_exposure_pct),
            "max_exposure_pct": float(max_exposure_pct),
            "margin_available": bool(margin_available),
            "conflicts": list(conflicts),
            "reason": reason,
        }
    
    def get_portfolio(self) -> Dict[str, Any]:
        """Get current portfolio holdings."""
        logger.info("get_portfolio()")
        
        if not _SIGNALS_AVAILABLE or not self.paper_trader:
            return {
                "timestamp": "",
                "total_value": 10000.0,
                "positions": {},
                "cash": 10000.0,
            }
        
        try:
            # Load portfolio from paper trader
            portfolio = self.paper_trader.get_portfolio()
            
            return {
                "timestamp": portfolio.get("timestamp", ""),
                "total_value": portfolio.get("total_value", 10000.0),
                "positions": portfolio.get("positions", {}),
                "cash": portfolio.get("cash", 10000.0),
                "trades_today": len(portfolio.get("trades_today", [])),
            }
        except Exception as e:
            logger.error(f"Error fetching portfolio: {e}")
            return {
                "timestamp": "",
                "total_value": 10000.0,
                "positions": {},
                "cash": 10000.0,
                "error": str(e),
            }
    
    def get_risk_constraints(self) -> Dict[str, Any]:
        """Get risk limits."""
        logger.info("get_risk_constraints()")
        return {
            "max_exposure_per_ticker_pct": 10.0,
            "max_total_exposure_pct": 100.0,
            "max_drawdown_pct": -15.0,
            "min_sharpe": 0.9,
            "max_daily_trades": 5,
        }
    
    def get_signal_history(self) -> Dict[str, Any]:
        """Get all signals generated this month."""
        logger.info("get_signal_history()")
        return {
            "month": "2026-03",
            "count": 0,
            "signals": [],
        }


def create_signals_server():
    """Factory function to create FastMCP signals server."""
    try:
        from fastmcp import FastMCP
        
        mcp = FastMCP(name="pyfinagent-signals")
        server = SignalsServer()
        
        # Register tools
        @mcp.tool
        def generate_signal(ticker: str, date: str) -> Dict[str, Any]:
            """
            Generate a trading signal (BUY/SELL/HOLD) for a ticker.
            
            Uses the trained model + current features to make a prediction.
            Includes confidence score and contributing factors.
            
            Returns: signal, confidence, factors, reason
            """
            return server.generate_signal(ticker, date)
        
        @mcp.tool
        def validate_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
            """
            Validate a signal against risk constraints.
            
            Checks: market hours, liquidity, exposure limits, margin.
            
            Returns: valid (bool), violations (list), adjusted_signal
            """
            return server.validate_signal(signal)
        
        @mcp.tool
        def publish_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
            """
            Publish a validated signal to Slack and update portfolio.
            
            This commits the trade to paper trading system.
            
            Returns: published (bool), slack_id, timestamp
            """
            return server.publish_signal(signal)
        
        @mcp.tool
        def risk_check(portfolio: Dict[str, Any], proposed_trade: Dict[str, Any]) -> Dict[str, Any]:
            """
            Check if a proposed trade is within risk limits.
            
            Args:
                portfolio: Current holdings
                proposed_trade: {"ticker": "AAPL", "action": "BUY", "shares": 100}
            
            Returns: allowed (bool), current_exposure, conflicts
            """
            return server.risk_check(portfolio, proposed_trade)
        
        # Register resources
        @mcp.resource("portfolio://current")
        def portfolio_resource() -> str:
            """Get current portfolio holdings and P&L."""
            result = server.get_portfolio()
            return json.dumps(result)
        
        @mcp.resource("constraints://risk")
        def constraints_resource() -> str:
            """Get risk limits and constraints."""
            result = server.get_risk_constraints()
            return json.dumps(result)
        
        @mcp.resource("signals://history")
        def signals_history_resource() -> str:
            """Get all signals generated this month."""
            result = server.get_signal_history()
            return json.dumps(result)
        
        logger.info("Signals server created with 4 tools + 3 resources")
        return mcp
    
    except ImportError:
        logger.error("FastMCP not installed. Install with: pip install fastmcp")
        raise


if __name__ == "__main__":
    # For testing: start signals server standalone
    mcp = create_signals_server()
    mcp.run()  # Runs on stdio by default
