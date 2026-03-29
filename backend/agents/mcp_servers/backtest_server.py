"""
MCP Backtest Server: Callable tools for backtesting, ablation, feature validation

Tools (FastMCP @mcp.tool):
- run_backtest(params) → Run full walk-forward backtest (returns Sharpe, return, etc.)
- run_single_feature_test(feature_code) → Test one feature on holdout period
- run_ablation_study(feature_to_remove) → Impact analysis (what if we remove this?)
- get_feature_importance() → MDI + MDA importance across all windows

Resources (for reference data):
- quant_results://all → TSV of all experiments
- experiments://recent → Last 10 backtest results
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Import backend modules
_BACKTEST_AVAILABLE = False

try:
    from backend.backtest.backtest_engine import BacktestEngine
    from backend.db.bigquery_client import BigQueryClient
    from backend.config.settings import get_settings
    _BACKTEST_AVAILABLE = True
except ImportError:
    logger.warning("Backtest engine not available — backtest server in stub mode")


class BacktestServer:
    """FastMCP backtest server for pyfinAgent."""
    
    def __init__(self):
        global _BACKTEST_AVAILABLE
        
        self.bq_client = None
        self.settings = None
        self.timeout_seconds = 30  # Max time for backtest tool
        
        # Initialize backtest engine if available
        if _BACKTEST_AVAILABLE:
            try:
                self.settings = get_settings()
                self.bq_client = BigQueryClient(self.settings)
                logger.info("BacktestServer initialized with BQ client")
            except Exception as e:
                logger.error(f"Failed to initialize BacktestServer: {e}")
                _BACKTEST_AVAILABLE = False
    
    def run_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run full walk-forward backtest with given parameters.
        
        Args:
            params: {
                "holding_days": 90,
                "tp_pct": 10.0,
                "sl_pct": 10.0,
                "min_samples_leaf": 18,
                "max_depth": 7,
                ...
            }
        
        Returns:
        {
            "status": "PASS",
            "run_id": "backtest_001",
            "sharpe": 1.1705,
            "dsr": 0.9984,
            "return_pct": 80.2,
            "max_drawdown_pct": -12.0,
            "num_trades": 634,
            "by_period": {
                "period_a": {"sharpe": 0.89, "return": 30.2},
                "period_b": {"sharpe": 0.92, "return": 25.1},
                "period_c": {"sharpe": 1.88, "return": 25.0}
            }
        }
        """
        logger.info(f"run_backtest(params={params})")
        
        if not _BACKTEST_AVAILABLE:
            return {
                "status": "ERROR",
                "error": "Backtest engine not available",
            }
        
        try:
            start = time.time()
            
            # Create backtest engine with provided params
            engine = BacktestEngine(
                bq_client=self.bq_client.client,
                project=self.settings.gcp_project_id,
                dataset=self.settings.bq_dataset_reports,
                **params  # Pass params as kwargs
            )
            
            # Run backtest with timeout
            result = engine.run_backtest()
            elapsed = time.time() - start
            
            # Extract key metrics from result
            full_period = result.get("full_period", {})
            
            return {
                "status": "PASS",
                "run_id": result.get("run_id", "backtest_manual"),
                "sharpe": full_period.get("sharpe", 0.0),
                "dsr": full_period.get("dsr", 0.0),
                "return_pct": full_period.get("return_pct", 0.0),
                "max_drawdown_pct": full_period.get("max_drawdown_pct", 0.0),
                "num_trades": full_period.get("num_trades", 0),
                "elapsed_seconds": int(elapsed),
            }
        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "error": str(e),
            }
    
    def run_single_feature_test(self, feature_code: str) -> Dict[str, Any]:
        """
        Test a single new feature on 2-week holdout period (2024-12-16 to 2024-12-31).
        
        Args:
            feature_code: Python code for feature (e.g., "close / close[90] - 1")
        
        Returns:
        {
            "status": "PASS",
            "feature": "momentum_3m",
            "in_sample_sharpe": 1.05,
            "out_of_sample_sharpe": 0.98,
            "dsr": 0.96,
            "correlation_with_existing": 0.42,
            "verdict": "KEEP",
            "reason": "DSR > 0.95, Sharpe improvement +0.03"
        }
        """
        logger.info(f"run_single_feature_test({feature_code})")
        # TODO: Implement feature validation
        return {
            "status": "PENDING_IMPLEMENTATION",
            "verdict": "UNKNOWN",
        }
    
    def run_ablation_study(self, feature_to_remove: str) -> Dict[str, Any]:
        """
        Run ablation study: impact of removing one feature from strategy.
        
        Args:
            feature_to_remove: Name of feature (e.g., "momentum_3m")
        
        Returns:
        {
            "feature_removed": "momentum_3m",
            "sharpe_delta": -0.05,
            "pvalue": 0.001,
            "return_delta_pct": -2.3,
            "affected_periods": ["period_a", "period_c"],
            "conclusion": "Feature is significant (p<0.05), contributes +0.05 Sharpe"
        }
        """
        logger.info(f"run_ablation_study({feature_to_remove})")
        # TODO: Implement ablation (rerun backtest without feature)
        return {
            "status": "PENDING_IMPLEMENTATION",
            "sharpe_delta": 0.0,
        }
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Get feature importance (MDI + MDA) across all 27 walk-forward windows.
        
        Returns:
        {
            "importance_mdi": {
                "momentum_3m": 0.15,
                "value_score": 0.12,
                "insider_ratio": 0.08,
                ...
            },
            "importance_mda": {
                "momentum_3m": 0.18,
                "value_score": 0.10,
                ...
            },
            "conclusion": "Momentum is most important both by MDI and MDA"
        }
        """
        logger.info("get_feature_importance()")
        
        try:
            # Try to load from most recent experiment result
            results_dir = Path(__file__).parent.parent.parent / "backtest" / "experiments" / "results"
            
            if results_dir.exists():
                # Find most recent result file
                results = list(results_dir.glob("*.json"))
                if results:
                    latest = max(results, key=lambda p: p.stat().st_mtime)
                    with open(latest, "r") as f:
                        data = json.load(f)
                    
                    return {
                        "status": "PASS",
                        "importance_mdi": data.get("feature_importance_mdi", {}),
                        "importance_mda": data.get("feature_importance_mda", {}),
                        "source": str(latest),
                    }
        except Exception as e:
            logger.error(f"Error loading feature importance: {e}")
        
        return {
            "status": "NO_DATA",
            "importance_mdi": {},
            "importance_mda": {},
        }
    
    def get_experiment_list(self) -> Dict[str, Any]:
        """Get all experiments from quant_results.tsv."""
        logger.info("get_experiment_list()")
        return {
            "count": 0,
            "experiments": [],
        }
    
    def get_recent_experiments(self, limit: int = 10) -> Dict[str, Any]:
        """Get last N experiments."""
        logger.info(f"get_recent_experiments({limit})")
        return {
            "count": 0,
            "experiments": [],
        }


def create_backtest_server():
    """Factory function to create FastMCP backtest server."""
    try:
        from fastmcp import FastMCP
        
        mcp = FastMCP(name="pyfinagent-backtest")
        server = BacktestServer()
        
        # Register tools
        @mcp.tool
        def run_backtest(params: Dict[str, Any]) -> Dict[str, Any]:
            """
            Run a full walk-forward backtest with the given parameters.
            
            This is a slow operation (~30s for 27 windows).
            Only call when you want to evaluate a new parameter set.
            
            Returns: sharpe, dsr, return_pct, max_drawdown_pct, num_trades, by_period
            """
            return server.run_backtest(params)
        
        @mcp.tool
        def run_single_feature_test(feature_code: str) -> Dict[str, Any]:
            """
            Quickly test a single new feature on a 2-week holdout period.
            
            Useful for rapid feature validation before full backtest.
            
            Args:
                feature_code: Python expression (e.g., "close / close.shift(90) - 1")
            
            Returns: acceptance verdict (KEEP/REJECT), Sharpe, correlation, DSR
            """
            return server.run_single_feature_test(feature_code)
        
        @mcp.tool
        def run_ablation_study(feature_to_remove: str) -> Dict[str, Any]:
            """
            Remove one feature from the strategy and measure impact.
            
            Helps identify which features are actually contributing to returns.
            
            Returns: sharpe_delta, p-value, affected periods
            """
            return server.run_ablation_study(feature_to_remove)
        
        @mcp.tool
        def get_feature_importance() -> Dict[str, Any]:
            """
            Get feature importance across all walk-forward windows (MDI + MDA).
            
            Fast operation (reads from cache).
            
            Returns: importance_mdi, importance_mda (dict of feature → score)
            """
            return server.get_feature_importance()
        
        # Register resources (reference data)
        @mcp.resource("quant_results://all")
        def experiments_resource() -> str:
            """Get all historical backtest results (TSV format)."""
            result = server.get_experiment_list()
            return json.dumps(result)
        
        @mcp.resource("experiments://recent")
        def recent_experiments_resource() -> str:
            """Get last 10 backtest results."""
            result = server.get_recent_experiments(limit=10)
            return json.dumps(result)
        
        logger.info("Backtest server created with 4 tools + 2 resources")
        return mcp
    
    except ImportError:
        logger.error("FastMCP not installed. Install with: pip install fastmcp")
        raise


if __name__ == "__main__":
    # For testing: start backtest server standalone
    mcp = create_backtest_server()
    mcp.run()  # Runs on stdio by default
