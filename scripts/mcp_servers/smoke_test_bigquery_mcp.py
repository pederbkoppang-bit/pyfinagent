"""phase-23.2.21: smoke test for the pinned BigQuery MCP server.

Spawns `mcp-server-bigquery` via `uvx` over stdio, performs the MCP
handshake (`initialize` -> `notifications/initialized`), then issues
`tools/list` and `tools/call list-tables`. Asserts that the response
contains at least one table from the user's BQ project. Authenticates
via the user's Application Default Credentials (no env vars, no
per-session OAuth).

Exit 0 = pinned MCP attaches and answers a real BQ query end-to-end.
Exit non-zero = any step failed; stderr captured.

Usage:
    source .venv/bin/activate
    python scripts/mcp_servers/smoke_test_bigquery_mcp.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = "sunny-might-477607-p8"
LOCATION = "US"
PACKAGE_SPEC = "mcp-server-bigquery==0.3.2"
TIMEOUT_S = 30.0
EXPECTED_PROJECT_PREFIX = PROJECT


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
        # Skip notifications and other ids.
        if msg.get("id") == want_id:
            return msg
    raise TimeoutError(f"no response with id={want_id} within {TIMEOUT_S}s")


def main() -> int:
    cmd = [
        "uvx",
        "--from", PACKAGE_SPEC,
        "mcp-server-bigquery",
        "--project", PROJECT,
        "--location", LOCATION,
    ]
    print(f"spawning: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    try:
        deadline = time.time() + TIMEOUT_S

        _send(proc, {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "smoke-test", "version": "1.0"},
            },
        })
        init_resp = _recv_until_id(proc, 1, deadline)
        if "result" not in init_resp:
            print(f"FAIL initialize: {init_resp}", file=sys.stderr)
            return 1
        server_name = init_resp["result"].get("serverInfo", {}).get("name", "?")
        print(f"OK initialize -- server={server_name}")

        _send(proc, {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })

        _send(proc, {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        })
        tools_resp = _recv_until_id(proc, 2, deadline)
        if "result" not in tools_resp:
            print(f"FAIL tools/list: {tools_resp}", file=sys.stderr)
            return 1
        tool_names = sorted({t.get("name") for t in tools_resp["result"].get("tools", [])})
        print(f"OK tools/list -- {tool_names}")
        for required in ("list-tables", "describe-table", "execute-query"):
            if required not in tool_names:
                print(f"FAIL missing required tool: {required}", file=sys.stderr)
                return 1

        _send(proc, {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "list-tables", "arguments": {}},
        })
        call_resp = _recv_until_id(proc, 3, deadline)
        if "result" not in call_resp:
            print(f"FAIL tools/call list-tables: {call_resp}", file=sys.stderr)
            return 1
        content_blob = json.dumps(call_resp["result"])
        if EXPECTED_PROJECT_PREFIX not in content_blob and "pyfinagent" not in content_blob:
            print(
                f"FAIL list-tables returned no rows mentioning {EXPECTED_PROJECT_PREFIX} "
                f"or pyfinagent. Response: {content_blob[:600]}",
                file=sys.stderr,
            )
            return 1
        print(f"OK tools/call list-tables -- response references project + pyfinagent_*")
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
        # Surface stderr if there was a problem.
        try:
            err = proc.stderr.read().decode("utf-8", errors="replace")
            if err.strip():
                last = err.strip().splitlines()[-3:]
                print("\nserver stderr (tail):\n  " + "\n  ".join(last), file=sys.stderr)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
