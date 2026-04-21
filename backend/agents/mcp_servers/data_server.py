"""
MCP Data Server: Read-only access to prices, fundamentals, macro, universe, features

Resources (FastMCP @mcp.resource):
- prices://[TICKER] -- OHLCV data
- fundamentals://[TICKER] -- P/E, P/B, ROE, debt, earnings
- macro://[SERIES] -- VIX, yield curve, GDP, inflation
- universe://[MARKET] -- List of tradeable tickers
- features://[TICKER] -- Computed features (momentum, value, sentiment)
- experiments://list -- All historical backtest results
- best-params://current -- Current best parameters from optimizer
"""

import csv
import json
import logging

from backend.utils import json_io  # noqa: E402 -- phase-4.14.5 consolidation
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Import backend modules for data access
_CACHE_AVAILABLE = False
_bq_client = None
_settings = None

try:
    from backend.backtest import cache
    from backend.db.bigquery_client import BigQueryClient
    from backend.config.settings import get_settings
    _CACHE_AVAILABLE = True
except ImportError:
    logger.warning("Cache module not available -- MCP data server running in stub mode")


def _to_float(value: Any) -> float:
    """Best-effort float coercion for TSV row values. Returns 0.0 on failure."""
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


class DataServer:
    """FastMCP data server for pyfinAgent."""
    
    def __init__(self):
        global _bq_client, _settings, _CACHE_AVAILABLE
        
        self.bq_client = None
        self.settings = None
        self._universe_cache: Dict[str, list] = {}

        # Initialize actual data access if available
        if _CACHE_AVAILABLE:
            try:
                self.settings = get_settings()
                self.bq_client = BigQueryClient(self.settings)
                cache.init_cache(self.bq_client.client, self.settings.gcp_project_id, self.settings.bq_dataset_reports)
                logger.info("DataServer initialized with BQ cache")
            except Exception as e:
                logger.error(f"Failed to initialize BQ cache: {e}")
                _CACHE_AVAILABLE = False
    
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
        logger.info(f"get_prices({ticker})")
        
        if not _CACHE_AVAILABLE:
            return {"ticker": ticker, "prices": []}
        
        try:
            # Parse market from ticker (e.g., "NO:EQNR" → market="NO", ticker="EQNR")
            market = "US"
            if ":" in ticker:
                market, ticker = ticker.split(":", 1)
            
            # Query prices from cache (default: 2023-2025, can be customized)
            df = cache.cached_prices(ticker, "2023-01-01", "2025-12-31")
            
            if df is None or df.empty:
                return {"ticker": ticker, "prices": []}
            
            # Convert to JSON-serializable format
            prices = []
            for _, row in df.iterrows():
                prices.append({
                    "date": str(row["date"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]),
                })
            
            return {"ticker": ticker, "prices": prices}
        except Exception as e:
            logger.error(f"Error fetching prices for {ticker}: {e}")
            return {"ticker": ticker, "prices": [], "error": str(e)}
    
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
        logger.info(f"get_fundamentals({ticker})")
        
        if not _CACHE_AVAILABLE:
            return {"ticker": ticker, "metrics": []}
        
        try:
            # Parse market from ticker
            market = "US"
            if ":" in ticker:
                market, ticker = ticker.split(":", 1)
            
            # Query fundamentals from cache
            metrics = cache.cached_fundamentals(ticker, "2025-12-31")
            
            if not metrics:
                return {"ticker": ticker, "metrics": []}
            
            return {"ticker": ticker, "metrics": metrics}
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}")
            return {"ticker": ticker, "metrics": [], "error": str(e)}
    
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
        logger.info(f"get_macro({series})")
        
        if not _CACHE_AVAILABLE:
            return {"series": series, "data": []}
        
        try:
            # Query macro from cache (series_id → FRED code, VIX, etc.)
            macro_data = cache.cached_macro("2025-12-31")
            
            if not macro_data:
                return {"series": series, "data": []}
            
            # Convert to JSON format
            data = []
            for item in macro_data:
                data.append({
                    "date": item.get("date", ""),
                    "value": float(item.get("value", 0)),
                })
            
            return {"series": series, "data": data}
        except Exception as e:
            logger.error(f"Error fetching macro {series}: {e}")
            return {"series": series, "data": [], "error": str(e)}
    
    def get_universe(self, market: str) -> Dict[str, Any]:
        """
        Get list of tradeable tickers in market.

        Delegates to CandidateSelector.get_universe_tickers which returns the
        S&P 500 from Wikipedia for 'US' (falling back to _FALLBACK_TICKERS on
        fetch error). Other markets return an empty list + warning until the
        Phase 5 multi-market work lands.

        Returns:
        {
            "market": "US",
            "count": 500,
            "tickers": ["AAPL", "MSFT", "GOOGL", ...]
        }
        """
        logger.info(f"get_universe({market})")

        if not _CACHE_AVAILABLE:
            return {"market": market, "count": 0, "tickers": [], "error": "cache module unavailable"}

        # Universe changes monthly and the fetch is network-bound; cache per
        # market for the server lifetime.
        if market in self._universe_cache:
            tickers = self._universe_cache[market]
        else:
            try:
                from backend.backtest.candidate_selector import CandidateSelector
                selector = CandidateSelector()
                tickers = selector.get_universe_tickers(market=market)
                self._universe_cache[market] = tickers
            except Exception as e:
                logger.error(f"Error fetching universe for {market}: {e}")
                return {"market": market, "count": 0, "tickers": [], "error": str(e)}

        return {
            "market": market,
            "count": len(tickers),
            "tickers": tickers,
        }
    
    def get_features(self, ticker: str, cutoff_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get computed feature vector for ticker at a point in time.

        Delegates to HistoricalDataProvider.build_feature_vector which returns
        the full ~43-feature vector (momentum, RSI, volatility, Bollinger, 12-1
        momentum, fundamentals, macro, Monte Carlo VaR, anomaly count, Amihud
        illiquidity) used by the backtest engine.

        Args:
            ticker: Ticker symbol (e.g. "AAPL").
            cutoff_date: Point-in-time for features (YYYY-MM-DD). Defaults to today.

        Returns:
        {
            "ticker": "AAPL",
            "cutoff_date": "2026-04-13",
            "features": {"momentum_1m": 0.12, "rsi_14": 55.3, ...}
        }
        """
        if cutoff_date is None:
            cutoff_date = date.today().isoformat()

        logger.info(f"get_features({ticker}, {cutoff_date})")

        if not _CACHE_AVAILABLE:
            return {
                "ticker": ticker,
                "cutoff_date": cutoff_date,
                "features": {},
                "error": "cache module unavailable",
            }

        try:
            from backend.backtest.historical_data import HistoricalDataProvider
            provider = HistoricalDataProvider()
            features = provider.build_feature_vector(ticker, cutoff_date)
            return {
                "ticker": ticker,
                "cutoff_date": cutoff_date,
                "features": features,
            }
        except Exception as e:
            logger.error(f"Error building feature vector for {ticker} @ {cutoff_date}: {e}")
            return {
                "ticker": ticker,
                "cutoff_date": cutoff_date,
                "features": {},
                "error": str(e),
            }
    
    def get_experiment_list(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Get list of all historical backtest experiments from quant_results.tsv.

        Uses stdlib csv (no pandas dep on the MCP path). The "best" block
        reuses get_best_params() so optimizer_best.json stays the single
        source of truth.

        Args:
            last_n: Optional tail slice (return only the last N experiments).

        Returns:
        {
            "count": 451,
            "experiments": [
                {
                    "timestamp": "2026-03-29T08:00:00",
                    "run_id": "opt_001",
                    "param_changed": "holding_days",
                    "metric_before": 0.95,
                    "metric_after": 1.17,
                    "delta": 0.22,
                    "status": "KEPT",
                    "dsr": 0.9988,
                    "top5_mda": "...",
                    "params": {...},
                    "parent_run_id": ""
                },
                ...
            ],
            "best": {...}
        }
        """
        logger.info(f"get_experiment_list(last_n={last_n})")

        tsv_path = (
            Path(__file__).parent.parent.parent
            / "backtest" / "experiments" / "quant_results.tsv"
        )

        experiments: list = []
        if tsv_path.exists():
            try:
                with open(tsv_path, "r", newline="") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        params_raw = row.get("params_json", "") or ""
                        params_parsed: Any = params_raw
                        if params_raw:
                            try:
                                params_parsed = json_io.loads(params_raw)
                            except (ValueError, TypeError):
                                params_parsed = params_raw
                        experiments.append({
                            "timestamp": row.get("timestamp", ""),
                            "run_id": row.get("run_id", ""),
                            "param_changed": row.get("param_changed", ""),
                            "metric_before": _to_float(row.get("metric_before")),
                            "metric_after": _to_float(row.get("metric_after")),
                            "delta": _to_float(row.get("delta")),
                            "status": row.get("status", ""),
                            "dsr": _to_float(row.get("dsr")),
                            "top5_mda": row.get("top5_mda", ""),
                            "params": params_parsed,
                            "parent_run_id": row.get("parent_run_id", ""),
                        })
            except Exception as e:
                logger.error(f"Error reading quant_results.tsv: {e}")

        if last_n is not None and last_n > 0:
            experiments = experiments[-last_n:]

        return {
            "count": len(experiments),
            "experiments": experiments,
            "best": self.get_best_params(),
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
        logger.info("get_best_params()")
        
        try:
            # Load from backend/backtest/experiments/optimizer_best.json
            import json
            best_file = Path(__file__).parent.parent.parent / "backtest" / "experiments" / "optimizer_best.json"
            
            if best_file.exists():
                with open(best_file, "r") as f:
                    data = json.load(f)
                return {
                    "timestamp": data.get("timestamp", ""),
                    "params": data.get("params", {}),
                    "sharpe": data.get("sharpe", 0.0),
                    "dsr": data.get("dsr", 0.0),
                    "return_pct": data.get("return_pct", 0.0),
                    "max_drawdown_pct": data.get("max_drawdown_pct", 0.0),
                }
        except Exception as e:
            logger.error(f"Error loading best params: {e}")
        
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
        
        @mcp.resource("best-params://current")
        def best_params_resource() -> str:
            """Get current best parameters from the optimizer."""
            result = server.get_best_params()
            return json.dumps(result)

        @mcp.tool
        def ping() -> dict:
            """Liveness probe for phase-4.6 smoketest."""
            import time as _t
            return {"ok": True, "server": "pyfinagent-data", "ts": _t.time()}

        logger.info("Data server created with 7 resources + 1 tool (ping)")
        return mcp
    
    except ImportError:
        logger.error("FastMCP not installed. Install with: pip install fastmcp")
        raise


if __name__ == "__main__":
    # For testing: start data server standalone
    mcp = create_data_server()
    mcp.run()  # Runs on stdio by default
