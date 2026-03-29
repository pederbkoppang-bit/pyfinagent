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
try:
    from backend.services.paper_trader import PaperTrader
    from backend.db.bigquery_client import BigQueryClient
    from backend.config.settings import get_settings
    SIGNALS_AVAILABLE = True
except ImportError:
    SIGNALS_AVAILABLE = False
    logger.warning("Paper trader not available — signals server in stub mode")


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
        self.portfolio = {}  # Current holdings
        self.risk_limits = {}  # Exposure limits
        self.signal_history = []  # All signals generated
        self.bq_client = None
        self.settings = None
        self.paper_trader = None
        
        # Initialize paper trader if available
        if SIGNALS_AVAILABLE:
            try:
                self.settings = get_settings()
                self.bq_client = BigQueryClient(self.settings)
                self.paper_trader = PaperTrader(bq_client=self.bq_client)
                logger.info("SignalsServer initialized with PaperTrader")
            except Exception as e:
                logger.error(f"Failed to initialize SignalsServer: {e}")
                SIGNALS_AVAILABLE = False
    
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
        Validate a signal against constraints (market hours, liquidity, exposure).
        
        Args:
            signal: {
                "ticker": "AAPL",
                "signal": "BUY",
                "confidence": 0.72
            }
        
        Returns:
        {
            "valid": True,
            "violations": [],
            "adjusted_signal": {...},  # If any adjustments made
            "reason": "All constraints satisfied"
        }
        """
        logger.info(f"validate_signal({signal['ticker']})")
        # TODO: Check constraints
        return {
            "valid": True,
            "violations": [],
            "adjusted_signal": signal,
            "reason": "PENDING_IMPLEMENTATION",
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
        Check if we can add a proposed trade to the portfolio.
        
        Args:
            portfolio: Current holdings dict
            proposed_trade: {
                "ticker": "AAPL",
                "action": "BUY",
                "shares": 100
            }
        
        Returns:
        {
            "allowed": True,
            "current_exposure_pct": 5.2,
            "max_exposure_pct": 10.0,
            "margin_available": True,
            "conflicts": [],
            "reason": "Trade is within limits"
        }
        """
        logger.info(f"risk_check({proposed_trade['ticker']})")
        # TODO: Check exposure limits, margin, correlations
        return {
            "allowed": True,
            "current_exposure_pct": 0.0,
            "max_exposure_pct": 10.0,
            "margin_available": True,
            "conflicts": [],
            "reason": "PENDING_IMPLEMENTATION",
        }
    
    def get_portfolio(self) -> Dict[str, Any]:
        """Get current portfolio holdings."""
        logger.info("get_portfolio()")
        
        if not SIGNALS_AVAILABLE or not self.paper_trader:
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
