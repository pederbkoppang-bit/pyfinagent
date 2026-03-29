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

__all__ = ["create_data_server", "create_backtest_server", "create_signals_server"]


async def start_all_servers():
    """Start all three MCP servers for autonomous Claude integration."""
    data_server = create_data_server()
    backtest_server = create_backtest_server()
    signals_server = create_signals_server()
    
    # In production, each server runs as a separate process/service
    # For development, they can share a transport (stdio, SSE, HTTP)
    
    return {
        "data": data_server,
        "backtest": backtest_server,
        "signals": signals_server,
    }
