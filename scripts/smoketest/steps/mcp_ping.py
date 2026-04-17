"""phase-4.6 step 4.6.2: MCP servers respond to ping + list_tools.

Usage:
    python scripts/smoketest/steps/mcp_ping.py --servers data,backtest,signals --timeout 10

Exits 0 on PASS, non-zero on FAIL. Emits JSON to stdout.

Design rationale (from handoff/current/contract.md research gate):
- Uses FastMCP in-memory Client transport: matches the in-process
  deployment model used today (servers are imported into FastAPI,
  not registered as stdio subprocesses in .mcp.json).
- Calls tools/list and then the ping tool on each server. Ping tool
  was added to all 3 servers as part of 4.6.2.
- No subprocess/stdio complexity -> cleaner + faster than stdio for a
  10s boot smoketest.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Repo root on sys.path so `import backend.agents...` resolves when this
# script is invoked via `python scripts/smoketest/steps/mcp_ping.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


FACTORIES = {
    "data": "create_data_server",
    "backtest": "create_backtest_server",
    "signals": "create_signals_server",
}


async def _probe_one(name: str) -> dict:
    """Create server factory, open in-memory client, call tools/list + ping."""
    t0 = time.monotonic()
    try:
        from backend.agents.mcp_servers import (
            create_data_server, create_backtest_server, create_signals_server,
        )
        factory = {
            "data": create_data_server,
            "backtest": create_backtest_server,
            "signals": create_signals_server,
        }[name]
        server = factory()
    except Exception as e:
        return {"server": name, "ok": False, "reason": f"factory_error:{type(e).__name__}:{e}"}

    try:
        # FastMCP 3.x in-memory client: pass the server instance directly.
        from fastmcp import Client
        async with Client(server) as client:
            tools = await client.list_tools()
            tool_names = [t.name for t in tools]
            if not tool_names:
                return {"server": name, "ok": False, "reason": "no_tools_listed", "tool_count": 0}
            if "ping" not in tool_names:
                return {"server": name, "ok": False, "reason": "ping_tool_missing", "tools": tool_names}
            result = await client.call_tool("ping", {})
            data = result.data if hasattr(result, "data") else result
            elapsed = time.monotonic() - t0
            return {
                "server": name,
                "ok": bool(data.get("ok")) if isinstance(data, dict) else False,
                "tool_count": len(tool_names),
                "tools": tool_names,
                "ping_response": data,
                "elapsed_s": elapsed,
            }
    except Exception as e:
        return {"server": name, "ok": False, "reason": f"client_error:{type(e).__name__}:{e}"}


async def _run(servers: list[str], timeout: int) -> dict:
    tasks = [_probe_one(s) for s in servers]
    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
    except asyncio.TimeoutError:
        return {"step": "4.6.2", "verdict": "FAIL", "reason": "overall_timeout", "timeout_s": timeout}

    all_ok = all(r.get("ok") for r in results)
    all_have_tools = all((r.get("tool_count") or 0) >= 1 for r in results)
    verdict = "PASS" if (all_ok and all_have_tools) else "FAIL"
    return {"step": "4.6.2", "verdict": verdict, "per_server": results}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--servers", default="data,backtest,signals")
    ap.add_argument("--timeout", type=int, default=10)
    args = ap.parse_args()
    servers = [s.strip() for s in args.servers.split(",") if s.strip()]
    for s in servers:
        if s not in FACTORIES:
            print(json.dumps({"step": "4.6.2", "verdict": "FAIL", "reason": f"unknown_server:{s}"}))
            return 2

    result = asyncio.run(_run(servers, args.timeout))
    print(json.dumps(result))
    if result["verdict"] == "PASS":
        print("MCP_PING_OK")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
