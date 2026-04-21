"""
pyfinagent-data MCP Server (Port 8101)

Read-only queries: experiments, parameters, metrics, portfolio state.
No state mutation. Zero cost beyond BQ.
"""

import logging
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="pyfinagent-data", version="0.1.0")


class Tool(BaseModel):
    name: str
    description: str
    inputSchema: dict


@app.get("/tools")
def list_tools() -> list[Tool]:
    """Return available tools."""
    return [
        Tool(
            name="get_experiments",
            description="List backtests from experiments database",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 20},
                    "status": {"type": "string", "enum": ["kept", "discarded", "all"]},
                    "order_by": {"type": "string", "default": "timestamp"},
                },
            },
        ),
        Tool(
            name="get_best_params",
            description="Return current best parameters",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_sharpe_history",
            description="Return Sharpe progression timeline",
            inputSchema={
                "type": "object",
                "properties": {
                    "window_days": {"type": "integer", "default": 30},
                },
            },
        ),
        Tool(
            name="get_portfolio_state",
            description="Current paper trading portfolio state",
            inputSchema={
                "type": "object",
                "properties": {
                    "market": {"type": "string", "default": "US"},
                },
            },
        ),
    ]


@app.post("/call_tool")
def call_tool(name: str, params: dict):
    """Execute a tool."""
    # TODO: Implement phase 3
    # For now, return stub responses
    if name == "get_experiments":
        return {"status": "ok", "data": []}
    elif name == "get_best_params":
        return {"status": "ok", "data": {"sharpe": 1.1705}}
    elif name == "get_sharpe_history":
        return {"status": "ok", "data": []}
    elif name == "get_portfolio_state":
        return {"status": "ok", "data": {"nav": 10000}}
    else:
        return {"status": "error", "message": f"Unknown tool: {name}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8101)
