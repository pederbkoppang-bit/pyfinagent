# Contract — Cycle 14: Phase 4.4.3.1 MCP Servers Deployed and Authenticated

## Target
`docs/GO_LIVE_CHECKLIST.md` item 4.4.3.1: All three MCP servers (data / backtest / signals) are reachable and respond to health probes.

## Current State
- Three MCP server modules exist: `data_server.py`, `backtest_server.py`, `signals_server.py`
- Each has a `create_*_server()` factory and `if __name__ == "__main__"` standalone entry
- `/api/health` returns `{"status": "ok"}` but has NO MCP server health subfields
- `.mcp.json` only registers Slack, not the three backend MCP servers

## Plan
1. Update `/api/health` in `backend/main.py` to include `mcp_servers` dict
2. Add three server entries to `.mcp.json` with stdio transport
3. Write drill `scripts/go_live_drills/mcp_servers_test.py`
4. Flip checkbox in `docs/GO_LIVE_CHECKLIST.md` with evidence

## Success Criteria
- SC1: Three module files exist at expected paths
- SC2: Three classes (DataServer, BacktestServer, SignalsServer) exist
- SC3: Three factory functions exist
- SC4: `__init__.py` exports all three factories
- SC5: `/api/health` endpoint includes `mcp_servers` health subfields
- SC6: `.mcp.json` has entries for all three servers
- SC7: Each server has `__main__` block for standalone execution
- SC8: Drill exits 0 with all scenarios PASS
