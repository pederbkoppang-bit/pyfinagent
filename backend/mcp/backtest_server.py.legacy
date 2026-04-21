"""
pyfinagent-backtest MCP Server (Port 8102)

On-demand backtest execution with cost gating and approval tracking.
Rate-limited: 1 backtest/min. Tracks cost before running.
"""

import logging
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="pyfinagent-backtest", version="0.1.0")


class Tool(BaseModel):
    name: str
    description: str
    inputSchema: dict


@app.get("/tools")
def list_tools() -> list[Tool]:
    """Return available tools."""
    return [
        Tool(
            name="run_backtest",
            description="Execute full walk-forward backtest with given parameters",
            inputSchema={
                "type": "object",
                "properties": {
                    "params": {
                        "type": "object",
                        "description": "Strategy parameters dict",
                    },
                    "start_date": {"type": "string", "default": "2018-01-01"},
                    "end_date": {"type": "string", "default": "2025-12-31"},
                },
                "required": ["params"],
            },
        ),
        Tool(
            name="run_sub_period_test",
            description="Test single sub-period (A/B/C)",
            inputSchema={
                "type": "object",
                "properties": {
                    "params": {"type": "object"},
                    "period": {"type": "string", "enum": ["A", "B", "C"]},
                },
                "required": ["params", "period"],
            },
        ),
        Tool(
            name="estimate_backtest_time",
            description="Estimate runtime and cost before execution",
            inputSchema={
                "type": "object",
                "properties": {
                    "params": {"type": "object"},
                },
                "required": ["params"],
            },
        ),
    ]


@app.post("/call_tool")
def call_tool(name: str, params: dict):
    """Execute a tool."""
    # TODO: Implement phase 3
    # For now, return stub responses
    if name == "run_backtest":
        return {"status": "queued", "run_id": "stub_123", "eta_seconds": 600}
    elif name == "run_sub_period_test":
        return {"status": "queued", "run_id": "stub_456", "eta_seconds": 180}
    elif name == "estimate_backtest_time":
        return {
            "status": "ok",
            "estimated_runtime_seconds": 600,
            "estimated_cost_usd": 0.25,
            "bq_queries": 15,
        }
    else:
        return {"status": "error", "message": f"Unknown tool: {name}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8102)
