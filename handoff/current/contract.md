# Sprint Contract -- phase-4.6 step 4.6.2

Started: 2026-04-17 (Cycle 32)
Step: 4.6.2 - MCP servers respond to ping + list_tools
Status: in-progress

## Research Gate (Passed)

researcher subagent + Explore subagent spawned in parallel.

Key findings:
- All 3 servers use FastMCP; factories at
  `backend/agents/mcp_servers/{data,backtest,signals}_server.py`
  (lines 408, 320, 1721).
- Data Server registers **7 resources but 0 tools** -- immutable
  criterion "list_tools returns at least one tool per server" would
  fail without intervention.
- Backtest Server: 4 tools, Signals Server: 4 tools -- both fine.
- Servers DO have stdio `__main__` blocks but are NOT registered in
  `.mcp.json` today -- they are in-process modules.
- In-process init has graceful BQ degradation (`_CACHE_AVAILABLE=False`
  returns stubs) -- no hard failure on missing creds.
- `fastmcp` pypi package was missing from venv; installed 3.2.4.
- FastMCP has no public tool-listing method -- must go through
  JSON-RPC `tools/list` or the FastMCP Client API.

## Hypothesis

Adding a lightweight `ping()` tool to each of the 3 servers (returns
`{ok: true, server: "<name>", ts: <unix>}`) gives us:
1. A real "respond to JSON-RPC ping" check per criterion 1.
2. At least one tool in every server, satisfying criterion 2 even
   for the data server (which is resource-heavy, tool-empty by design).
This is additive -- no existing tool/resource is modified.

The smoketest `mcp_ping.py` will use FastMCP's in-memory Client
transport (cleaner + faster than stdio subprocess for a 10s boot
smoketest; matches the in-process deployment model used today).

## Success Criteria (immutable)

- all three servers respond to JSON-RPC ping
- list_tools returns at least one tool per server

## Verification Command (immutable)

python scripts/smoketest/steps/mcp_ping.py --servers data,backtest,signals --timeout 10

## Plan

1. Add `ping()` tool to each of 3 servers (3 small edits).
2. Write `scripts/smoketest/steps/mcp_ping.py`:
   - Import create_{data,backtest,signals}_server
   - Create each via FastMCP in-memory client
   - Call `tools/list` -> assert >= 1 tool
   - Call `ping` tool -> assert {ok: true, server: <name>}
   - Emit JSON verdict
3. Run verification (async with 10s timeout).
4. EVALUATE via qa-evaluator + harness-verifier in parallel.
5. LOG + mark done.

## References

- https://gofastmcp.com/servers/tools (FastMCP tool decorator)
- https://gofastmcp.com/clients (FastMCP Client + InMemory transport)
- https://modelcontextprotocol.io/specification/2025-11-25 (tools/list spec)
