"""
MCP Server Architecture for Phase 3.0: LLM Integration

Three FastMCP servers providing Claude with:
1. pyfinagent-data: Resources for prices, fundamentals, macro, features (read-only)
2. pyfinagent-backtest: Tools for backtesting, ablation, feature validation (callable)
3. pyfinagent-signals: Tools for signal generation, validation, publishing (callable)

Usage:
    from backend.agents.mcp_servers import start_all_servers
    start_all_servers()
"""

from .data_server import create_data_server
from .backtest_server import create_backtest_server
from .signals_server import create_signals_server
from .risk_server import create_risk_server

__all__ = [
    "create_data_server",
    "create_backtest_server",
    "create_signals_server",
    "create_risk_server",
]


async def start_all_servers():
    """Start all four MCP servers for autonomous Claude integration."""
    return {
        "data": create_data_server(),
        "backtest": create_backtest_server(),
        "signals": create_signals_server(),
        "risk": create_risk_server(),
    }
