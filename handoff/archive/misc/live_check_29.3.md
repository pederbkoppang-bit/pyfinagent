# Live check — phase-29.3 (4 in-app FastMCP servers in .mcp.json)

**Step ID:** phase-29.3
**Date:** 2026-05-19

## Pre-restart evidence (this cycle)

```
$ jq '.mcpServers | keys' .mcp.json
["alpaca", "bigquery", "paper-search-mcp", "pyfinagent-backtest", "pyfinagent-data", "pyfinagent-risk", "pyfinagent-signals"]

$ python3 -c "import json; json.load(open('.mcp.json'))" && echo "JSON valid"
JSON valid

$ jq -e '[.mcpServers["pyfinagent-backtest","pyfinagent-data","pyfinagent-risk","pyfinagent-signals"] | .env.PYTHONPATH] | all(. == "/Users/ford/.openclaw/workspace/pyfinagent")' .mcp.json
true

# alwaysLoad matrix
$ jq '{backtest: .mcpServers."pyfinagent-backtest".alwaysLoad, data: .mcpServers."pyfinagent-data".alwaysLoad, risk: .mcpServers."pyfinagent-risk".alwaysLoad, signals: .mcpServers."pyfinagent-signals".alwaysLoad}' .mcp.json
{
  "backtest": false,
  "data": true,
  "risk": true,
  "signals": false
}
```

## Pre-flight smoke test results (this cycle)

All 4 servers: FastMCP 3.2.4 banner present on stderr; no Python traceback; clean exit on stdin EOF.

```
PASS risk — FastMCP banner present, no Python traceback
PASS data — FastMCP banner present, no Python traceback
PASS backtest — FastMCP banner present, no Python traceback
PASS signals — FastMCP banner present, no Python traceback
```

This confirms PYTHONPATH resolution works (no `ImportError: No module named 'backend'`) and the venv python at `/Users/ford/.openclaw/workspace/pyfinagent/.venv/bin/python` is invocable.

## Post-restart operator recipe

```
1. /clear or quit + relaunch Claude Code
2. New /mcp panel: confirm 7 servers listed (alpaca, bigquery, paper-search-mcp, pyfinagent-{backtest,data,risk,signals})
3. data + risk should show "connected" (alwaysLoad: true)
4. backtest + signals should show "available" (not connected until first tool call; alwaysLoad: false)
5. ToolSearch query "pyfinagent" should surface a long list (~30+ tools across the 4 in-app servers)
6. Test risk gate chain:
   - Call: pyfinagent-risk.evaluate_candidate(symbol="AAPL", quantity=100, projected_pnl_pct=-15.0)
   - Expected: vetoed=true with reason citing projected_dd > 10% threshold (kill_switch:179, pbo_check:186-198, projected_dd:201-213)
```

## Potential post-restart issues + remedies

| Symptom | Likely cause | Remedy |
|---|---|---|
| `/mcp` panel shows server "failed" | PYTHONPATH not resolving | Confirm Claude Code is running with `cwd = project_root`; if not, add absolute PYTHONPATH (already done) |
| `signals` server takes >5s to "connect" | 1887-line module slow init | Expected first session post-29.3; subsequent sessions warm |
| Session hangs ~5s at startup | `alwaysLoad: true` server failed init | Check BQ ADC: `gcloud auth application-default print-access-token` should succeed |
| `evaluate_candidate` returns vetoed for everything | Kill switch is_paused=True | Check `backend/services/kill_switch.py` state; not a phase-29.3 issue |

## Honest disclosure

This cycle's Q/A subagent (and the other 5 cycles in this overnight session) cannot exercise the new MCPs because Claude Code's MCP inventory snapshot was taken at session start. The 4 new servers will activate for the next (morning) session. The pre-restart evidence above is CONFIG validity + PYTHON IMPORT resolution + FASTMCP STARTUP — not a live JSON-RPC roundtrip. That happens post-restart.
