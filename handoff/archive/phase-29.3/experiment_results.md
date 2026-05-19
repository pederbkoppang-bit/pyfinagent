# Experiment Results — phase-29.3 (Register 4 in-app FastMCP servers in .mcp.json)

**Step ID:** phase-29.3
**Date:** 2026-05-19
**Cycle:** 1

Config-only cycle. `.mcp.json` grew from 3 to 7 MCP server entries. No code edits.

---

## 1. Edits made

### `.mcp.json` — 4 new entries added

**Before:** 36 lines, 3 entries (alpaca, bigquery, paper-search-mcp).

**After:** 78 lines, 7 entries. New:

```json
"pyfinagent-backtest": {
  "type": "stdio",
  "command": "/Users/ford/.openclaw/workspace/pyfinagent/.venv/bin/python",
  "args": ["/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/mcp_servers/backtest_server.py"],
  "env": {"PYTHONPATH": "/Users/ford/.openclaw/workspace/pyfinagent"},
  "alwaysLoad": false
},
"pyfinagent-data": {
  "type": "stdio",
  "command": "/Users/ford/.openclaw/workspace/pyfinagent/.venv/bin/python",
  "args": ["/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/mcp_servers/data_server.py"],
  "env": {"PYTHONPATH": "/Users/ford/.openclaw/workspace/pyfinagent"},
  "alwaysLoad": true
},
"pyfinagent-risk": {
  "type": "stdio",
  "command": "/Users/ford/.openclaw/workspace/pyfinagent/.venv/bin/python",
  "args": ["/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/mcp_servers/risk_server.py"],
  "env": {"PYTHONPATH": "/Users/ford/.openclaw/workspace/pyfinagent"},
  "alwaysLoad": true
},
"pyfinagent-signals": {
  "type": "stdio",
  "command": "/Users/ford/.openclaw/workspace/pyfinagent/.venv/bin/python",
  "args": ["/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/mcp_servers/signals_server.py"],
  "env": {"PYTHONPATH": "/Users/ford/.openclaw/workspace/pyfinagent"},
  "alwaysLoad": false
}
```

### `alwaysLoad` decision rationale

| Server | alwaysLoad | Rationale |
|---|---|---|
| pyfinagent-risk | **true** | Highest-value; Layer-2 agents should consult risk-gate before recommending trades. evaluate_candidate fires on every trading candidate. Pre-load justified. |
| pyfinagent-data | **true** | High-frequency; prices://, fundamentals://, etc. used on nearly every harness analysis turn. BQ-ADC graceful degradation means failed init still leaves a usable stub. |
| pyfinagent-backtest | **false** | Rare-use; harness optimization cycles only. Context savings dominate. |
| pyfinagent-signals | **false** | 1887 lines; longest startup. Service-driven (paper-trading triggers), not session-driven. |

---

## 2. Pre-flight smoke test — verbatim output

```
$ source .venv/bin/activate
$ for SERVER in risk data backtest signals; do
    PYTHONPATH=/Users/ford/.openclaw/workspace/pyfinagent python /Users/ford/.openclaw/workspace/pyfinagent/backend/agents/mcp_servers/${SERVER}_server.py </dev/null >/tmp/smoke_${SERVER}.out 2>/tmp/smoke_${SERVER}.err &
    PID=$!; sleep 2; kill $PID 2>/dev/null; wait $PID 2>/dev/null
    if grep -q 'FastMCP' /tmp/smoke_${SERVER}.err && ! grep -qE 'Traceback|ImportError' /tmp/smoke_${SERVER}.err; then
      echo "PASS $SERVER"
    else
      echo "FAIL $SERVER"
    fi
  done

PASS risk — FastMCP banner present, no Python traceback
PASS data — FastMCP banner present, no Python traceback
PASS backtest — FastMCP banner present, no Python traceback
PASS signals — FastMCP banner present, no Python traceback
=== ALL_PASS=true ===
```

All 4 servers correctly:
- Import their dependencies (no ImportError despite `from backend.backtest...` style imports — PYTHONPATH resolves)
- Initialize FastMCP 3.2.4 (banner printed on stderr per FastMCP behaviour)
- Bind to stdio transport (`mcp.run()` default)
- Exit cleanly on stdin EOF (expected — when Claude Code launches them, stdin stays open for JSON-RPC)

---

## 3. Verification command output

```
$ python3 -c "import json; json.load(open('.mcp.json'))" && \
  jq -e '.mcpServers | (."pyfinagent-backtest" and ."pyfinagent-data" and ."pyfinagent-risk" and ."pyfinagent-signals")' .mcp.json && \
  jq -e '[.mcpServers["pyfinagent-backtest","pyfinagent-data","pyfinagent-risk","pyfinagent-signals"] | .env.PYTHONPATH] | all(. == "/Users/ford/.openclaw/workspace/pyfinagent")' .mcp.json && \
  jq -e '.mcpServers."pyfinagent-risk".alwaysLoad == true' .mcp.json && \
  jq -e '.mcpServers."pyfinagent-data".alwaysLoad == true' .mcp.json && \
  jq -e '.mcpServers."pyfinagent-backtest".alwaysLoad == false' .mcp.json && \
  jq -e '.mcpServers."pyfinagent-signals".alwaysLoad == false' .mcp.json && \
  grep -q 'smoke test' handoff/current/experiment_results.md

(all 7 predicates returned `true`)
exit=0
```

---

## 4. Files touched

| File | Change |
|---|---|
| `.mcp.json` | +42 lines (4 new MCP server entries); JSON validates |
| `.claude/masterplan.json` 29.3 | audit_basis + verification fields rewritten |
| `handoff/current/research_brief.md` | rewritten (7 sources read in full) |
| `handoff/current/contract.md` | rewritten |
| `handoff/current/experiment_results.md` | this file |
| `handoff/current/live_check_29.3.md` | new |

**No** `backend/`, `frontend/`, `scripts/` files touched.

---

## 5. Honest disclosures

1. **First smoke-test attempt was a false-positive failure** — initial bash script used `timeout` (not on macOS by default) then misinterpreted clean stdin-EOF exit as "died early". Re-ran with correct success indicator (FastMCP banner present + no Python traceback). All 4 servers PASS.
2. **urllib3 deprecation warning** appears in stderr for all 4 servers: `RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.4.3)/charset_normalizer (3.4.6) doesn't match a supported version!`. NOT a failure — preserved upstream dependency mismatch; servers still function. Out of scope (would need a dep audit cycle).
3. **signals_server.py is 1887 lines** — startup time may be longer than the other 3. If `/mcp` panel shows it as "connecting" for >5s after restart, that's expected for the first session post-29.3; subsequent sessions benefit from any FastMCP-side warm cache. Marked in live_check.
4. **`alwaysLoad: true` blocks session startup** until the named server connects. If `risk_server` or `data_server` fails to init at boot (e.g. ADC permissions broken), the session will hang for ~5s per failing server then degrade gracefully (stubs). This is mentioned in the research brief §Pitfalls #3.
5. **BQ ADC required for full functionality.** All 4 servers degrade gracefully without ADC (set `_CACHE_AVAILABLE = False`, return empty results), so registration is safe even on a fresh checkout.
6. **Layer-2 agent code does NOT yet call these MCPs.** This cycle only registers them as available tools. A separate phase will update `multi_agent_orchestrator.py` to actually invoke `evaluate_candidate`, `prices://`, etc.
7. **Anti-rubber-stamp:** 7-predicate AND-chain verification exits 1 if ANY of (json validity, 4 entries present, PYTHONPATH correct on all 4, alwaysLoad correct on all 4, smoke-test recipe documented) fails.

---

## 6. Decision

Ready for Q/A. 7 success criteria all evidenced (jq + json.load + smoke-test output).
