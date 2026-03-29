"""
MCP Data Server: Read-only access to prices, fundamentals, macro, universe, features

Resources (FastMCP @mcp.resource):
- prices://[TICKER] → OHLCV data
- fundamentals://[TICKER] → P/E, P/B, ROE, debt, earnings
- macro://[SERIES] → VIX, yield curve, GDP, inflation
- universe://[MARKET] → List of tradeable tickers
- features://[TICKER] → Computed features (momentum, value, sentiment)
- experiments://list → All historical backtest results
- best_params://current → Current best parameters from optimizer
"""

import json
import logging
from typing import Any, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DataServer:
    """FastMCP data server for pyfinAgent."""
    
    def __init__(self):
        self.bq_client = None  # Will be initialized by FastMCP framework
        self.cache = {}  # Simple in-memory cache for frequently-accessed data
    
    def get_prices(self, ticker: str) -> Dict[str, Any]:
        """
        Get OHLCV data for ticker.
        
        Returns:
        {
            "ticker": "AAPL",
            "prices": [
                {"date": "2023-01-03", "open": 130.0, "high": 130.9, "low": 129.5, "close": 130.0, "volume": 65000000},
                ...
            ]
        }
        """
        # TODO: Implement BQ query
        logger.info(f"get_prices({ticker})")
        return {
            "ticker": ticker,
            "prices": [],  # Placeholder
        }
    
    def get_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """
        Get fundamental metrics for ticker (P/E, P/B, ROE, debt, earnings).
        
        Returns:
        {
            "ticker": "AAPL",
            "metrics": [
                {"date": "2023-12-31", "pe_ratio": 25.3, "pb_ratio": 42.0, "roe": 0.82, "debt_to_equity": 1.2},
                ...
            ]
        }
        """
        # TODO: Implement BQ query
        logger.info(f"get_fundamentals({ticker})")
        return {
            "ticker": ticker,
            "metrics": [],  # Placeholder
        }
    
    def get_macro(self, series: str) -> Dict[str, Any]:
        """
        Get macro time series (VIX, yield curve, GDP, inflation).
        
        Returns:
        {
            "series": "VIX",
            "data": [
                {"date": "2023-01-03", "value": 14.5},
                ...
            ]
        }
        """
        # TODO: Implement FRED/Bloomberg query
        logger.info(f"get_macro({series})")
        return {
            "series": series,
            "data": [],  # Placeholder
        }
    
    def get_universe(self, market: str) -> Dict[str, Any]:
        """
        Get list of tradeable tickers in market.
        
        Returns:
        {
            "market": "US",
            "count": 500,
            "tickers": ["AAPL", "MSFT", "GOOGL", ...],
            "status": "all tickers from current screener universe"
        }
        """
        # TODO: Implement universe list from screener
        logger.info(f"get_universe({market})")
        return {
            "market": market,
            "count": 0,
            "tickers": [],  # Placeholder
        }
    
    def get_features(self, ticker: str) -> Dict[str, Any]:
        """
        Get computed features for ticker (momentum, value, sentiment, etc.).
        
        Returns:
        {
            "ticker": "AAPL",
            "features": [
                {
                    "date": "2023-01-03",
                    "momentum_3m": 0.12,
                    "value_score": 0.45,
                    "sentiment_score": 0.68,
                    "insider_ratio": 1.2,
                    ...
                },
                ...
            ]
        }
        """
        # TODO: Implement feature computation from cache
        logger.info(f"get_features({ticker})")
        return {
            "ticker": ticker,
            "features": [],  # Placeholder
        }
    
    def get_experiment_list(self) -> Dict[str, Any]:
        """
        Get list of all historical backtest experiments (TSV format).
        
        Returns:
        {
            "count": 70,
            "experiments": [
                {
                    "run_id": "opt_001",
                    "timestamp": "2026-03-29 08:00",
                    "params": {...},
                    "sharpe": 1.1705,
                    "dsr": 0.9984,
                    "status": "PASS"
                },
                ...
            ],
            "best": {"run_id": "opt_001", "sharpe": 1.1705}
        }
        """
        # TODO: Load from quant_results.tsv
        logger.info("get_experiment_list()")
        return {
            "count": 0,
            "experiments": [],  # Placeholder
            "best": {},
        }
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get current best parameters from optimizer.
        
        Returns:
        {
            "timestamp": "2026-03-28 18:00",
            "params": {
                "holding_days": 90,
                "tp_pct": 10.0,
                "sl_pct": 10.0,
                "min_samples_leaf": 18,
                "max_depth": 7,
                ...
            },
            "sharpe": 1.1705,
            "dsr": 0.9984,
            "return_pct": 80.2,
            "max_drawdown_pct": -12.0
        }
        """
        # TODO: Load from optimizer_best.json
        logger.info("get_best_params()")
        return {
            "timestamp": "",
            "params": {},
            "sharpe": 0.0,
            "dsr": 0.0,
        }


def create_data_server():
    """Factory function to create FastMCP data server."""
    try:
        from fastmcp import FastMCP
        
        mcp = FastMCP(name="pyfinagent-data")
        server = DataServer()
        
        # Register resources
        @mcp.resource("prices://{ticker}")
        def prices_resource(ticker: str) -> str:
            """Get OHLCV price data for a ticker."""
            result = server.get_prices(ticker)
            return json.dumps(result)
        
        @mcp.resource("fundamentals://{ticker}")
        def fundamentals_resource(ticker: str) -> str:
            """Get fundamental metrics (P/E, P/B, ROE, debt) for a ticker."""
            result = server.get_fundamentals(ticker)
            return json.dumps(result)
        
        @mcp.resource("macro://{series}")
        def macro_resource(series: str) -> str:
            """Get macroeconomic time series (VIX, yield curve, GDP)."""
            result = server.get_macro(series)
            return json.dumps(result)
        
        @mcp.resource("universe://{market}")
        def universe_resource(market: str) -> str:
            """Get list of tradeable tickers in a market (US, NO, CA, EU, KR)."""
            result = server.get_universe(market)
            return json.dumps(result)
        
        @mcp.resource("features://{ticker}")
        def features_resource(ticker: str) -> str:
            """Get computed features (momentum, value, sentiment) for a ticker."""
            result = server.get_features(ticker)
            return json.dumps(result)
        
        @mcp.resource("experiments://list")
        def experiments_resource() -> str:
            """Get list of all historical backtest experiments."""
            result = server.get_experiment_list()
            return json.dumps(result)
        
        @mcp.resource("best_params://current")
        def best_params_resource() -> str:
            """Get current best parameters from the optimizer."""
            result = server.get_best_params()
            return json.dumps(result)
        
        logger.info("Data server created with 7 resources")
        return mcp
    
    except ImportError:
        logger.error("FastMCP not installed. Install with: pip install fastmcp")
        raise


if __name__ == "__main__":
    # For testing: start data server standalone
    mcp = create_data_server()
    mcp.run()  # Runs on stdio by default
