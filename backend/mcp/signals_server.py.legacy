"""
pyfinagent-signals MCP Server (Port 8103)

Generate and validate trading signals with LLM gating.
Read-mostly with gated writes. Tracks signal accuracy.
"""

import logging
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="pyfinagent-signals", version="0.1.0")


class Tool(BaseModel):
    name: str
    description: str
    inputSchema: dict


@app.get("/tools")
def list_tools() -> list[Tool]:
    """Return available tools."""
    return [
        Tool(
            name="generate_signals",
            description="ML + screener → BUY/SELL/HOLD signals for top candidates",
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "top_n": {"type": "integer", "default": 10},
                    "market": {"type": "string", "default": "US"},
                },
                "required": ["date"],
            },
        ),
        Tool(
            name="validate_signal",
            description="LLM gate: validate signal before publishing",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "direction": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reasoning": {"type": "string"},
                },
                "required": ["ticker", "direction", "confidence"],
            },
        ),
        Tool(
            name="get_signal_log",
            description="Historical signal accuracy",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "default": 30},
                },
            },
        ),
        Tool(
            name="approve_signal",
            description="Mark signal approved for paper trading",
            inputSchema={
                "type": "object",
                "properties": {
                    "signal_id": {"type": "string"},
                },
                "required": ["signal_id"],
            },
        ),
    ]


@app.post("/call_tool")
def call_tool(name: str, params: dict):
    """Execute a tool."""
    # TODO: Implement phase 3
    # For now, return stub responses
    if name == "generate_signals":
        return {
            "status": "ok",
            "signals": [
                {
                    "ticker": "NVDA",
                    "direction": "BUY",
                    "confidence": 0.78,
                    "reasoning": "Strong momentum",
                }
            ],
        }
    elif name == "validate_signal":
        return {
            "status": "ok",
            "approved": True,
            "feedback": "Signal looks reasonable",
        }
    elif name == "get_signal_log":
        return {
            "status": "ok",
            "accuracy": 0.62,
            "total_signals": 50,
            "profitable": 31,
        }
    elif name == "approve_signal":
        return {"status": "ok", "approved_at": "2026-03-29T10:30:00Z"}
    else:
        return {"status": "error", "message": f"Unknown tool: {name}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8103)
