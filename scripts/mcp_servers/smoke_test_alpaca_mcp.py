"""phase-25.A10: smoke test for the pinned Alpaca MCP server.

Spawns `alpaca-mcp-server==2.0.1` via `uvx` over stdio, performs the
MCP handshake (`initialize` -> `notifications/initialized`), then
issues `tools/list` and asserts the response contains the canonical
V2 read + write tool surface. Authenticates via ALPACA_API_KEY_ID +
ALPACA_API_SECRET_KEY env vars (paper-trade mode forced via
ALPACA_PAPER_TRADE=true).

Exit 0 = pinned MCP attaches and answers `tools/list` end-to-end.
Exit 0 = SKIP when credentials are missing (graceful CI mode).
Exit non-zero = any step failed; stderr captured.

Usage:
    source .venv/bin/activate
    python scripts/mcp_servers/smoke_test_alpaca_mcp.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

PACKAGE_SPEC = "alpaca-mcp-server==2.0.1"
TIMEOUT_S = 30.0

# Canonical V2 read + write tool names we expect to see in tools/list.
# (subset, not exhaustive -- the reconcile script enforces the full deny list)
EXPECTED_READ_TOOLS = {"get_account_info", "get_clock", "get_stock_snapshot"}
EXPECTED_WRITE_TOOLS = {"place_stock_order", "cancel_all_orders", "close_position"}


def _send(proc: subprocess.Popen, payload: dict) -> None:
    """Write a JSON-RPC frame to the server's stdin (newline-framed)."""
    line = json.dumps(payload) + "\n"
    proc.stdin.write(line.encode("utf-8"))
    proc.stdin.flush()


def _recv_until_id(proc: subprocess.Popen, want_id: int, deadline: float) -> dict:
    """Read newline-framed JSON-RPC responses until one matches `id`."""
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.05)
            continue
        try:
            msg = json.loads(line.decode("utf-8"))
        except Exception:
            continue
        if msg.get("id") == want_id:
            return msg
    return {"error": "timeout"}


def main() -> int:
    # phase-25.A10: accept either canonical env var name. The user's .env
    # uses ALPACA_API_KEY_ID + ALPACA_API_SECRET_KEY (longer / explicit names);
    # the alpaca-mcp-server itself expects ALPACA_API_KEY + ALPACA_SECRET_KEY.
    api_key = os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_API_KEY_ID")
    api_secret = (
        os.environ.get("ALPACA_SECRET_KEY")
        or os.environ.get("ALPACA_API_SECRET_KEY")
    )

    if not api_key or not api_secret:
        print(
            "SKIP -- no Alpaca credentials (set ALPACA_API_KEY_ID + "
            "ALPACA_API_SECRET_KEY OR ALPACA_API_KEY + ALPACA_SECRET_KEY; "
            "smoke test gracefully degraded; reconcile script still gates the deny list)"
        )
        return 0

    env = os.environ.copy()
    # Translate to the names the alpaca-mcp-server itself expects.
    env["ALPACA_API_KEY"] = api_key
    env["ALPACA_SECRET_KEY"] = api_secret
    env.setdefault("ALPACA_PAPER_TRADE", "true")

    cmd = ["uvx", "--from", PACKAGE_SPEC, "alpaca-mcp-server"]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    try:
        deadline = time.time() + TIMEOUT_S

        # 1) initialize
        _send(proc, {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "smoke_test_alpaca_mcp", "version": "1.0"},
            },
        })
        init_resp = _recv_until_id(proc, 1, deadline)
        if "result" not in init_resp:
            print(f"FAIL initialize: {init_resp}", file=sys.stderr)
            return 1
        print(f"OK initialize -- protocolVersion={init_resp['result'].get('protocolVersion')}")

        # 2) notifications/initialized (no id)
        _send(proc, {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })

        # 3) tools/list
        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools_resp = _recv_until_id(proc, 2, deadline)
        if "result" not in tools_resp:
            print(f"FAIL tools/list: {tools_resp}", file=sys.stderr)
            return 1

        tool_names = sorted({t.get("name") for t in tools_resp["result"].get("tools", [])})
        print(f"OK tools/list -- {len(tool_names)} tools exposed")

        missing_read = EXPECTED_READ_TOOLS - set(tool_names)
        missing_write = EXPECTED_WRITE_TOOLS - set(tool_names)
        if missing_read:
            print(f"FAIL missing read tools: {sorted(missing_read)}", file=sys.stderr)
            return 1
        if missing_write:
            print(f"FAIL missing write tools: {sorted(missing_write)}", file=sys.stderr)
            return 1
        print(f"OK read+write tool surface confirmed (sampled {len(EXPECTED_READ_TOOLS) + len(EXPECTED_WRITE_TOOLS)} canonical tools present)")
        return 0
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            err = proc.stderr.read().decode("utf-8", errors="replace")
            if err.strip():
                last = err.strip().splitlines()[-3:]
                print("\nserver stderr (tail):\n  " + "\n  ".join(last), file=sys.stderr)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
