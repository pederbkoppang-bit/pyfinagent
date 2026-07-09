# live_check 67.5 -- tripwire dry-run + headless MCP attach, verbatim

Required shape (masterplan 67.5): "(a) tripwire dry-run output under a faked date >=
2026-07-12 showing the warning text verbatim, (b) the headless MCP attach listing,
(c) the fresh Q/A verdict JSON".

## (a) Tripwire dry-run (2026-07-10, TRIPWIRE_FAKE_TODAY=2026-07-13)

```
$ TRIPWIRE_FAKE_TODAY=2026-07-13 CLAUDE_PROJECT_DIR="$PWD" bash .claude/hooks/session-start-fable-tripwire.sh; echo "exit=$?"
{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"TRIPWIRE (phase-67.5): today is 2026-07-13 -- the free Fable 5 window ended 2026-07-12 and 'model: fable' is STILL pinned in: researcher.md qa.md. Masterplan step 67.4 (revert to model: opus, KEEP every artifact improvement) is the TOP P0 for this session, unless the operator has recorded 'FABLE PERMANENT: AUTHORIZE' in handoff/harness_log.md. Every Fable turn may now draw Max usage credits."}}
exit=0
```
Control paths: FAKE_TODAY=2026-07-11 -> silence exit=0; CLAUDE_PROJECT_DIR=/nonexistent
-> silence exit=0 (fail-open); output json.load-parseable (394 context chars).

## (b) MCP approval surface, before -> after (`claude mcp list`, the same
## settings files headless runs read; scripts/away_ops/run_away_session.sh:160-165)

BEFORE (2026-07-10, pre-fix):
```
bigquery: ... - Pending approval (run `claude` to approve)
paper-search-mcp: ... - Pending approval (run `claude` to approve)
pyfinagent-backtest: ... - Pending approval (run `claude` to approve)
pyfinagent-data: ... - Pending approval (run `claude` to approve)
pyfinagent-risk: ... - Pending approval (run `claude` to approve)
pyfinagent-signals: ... - Pending approval (run `claude` to approve)
playwright: ... - Pending approval (run `claude` to approve)
```

AFTER (same day, post settings.local.json enabledMcpjsonServers fix):
```
bigquery: uvx --from mcp-server-bigquery==0.3.2 mcp-server-bigquery --project sunny-might-477607-p8 --location US - Connected
paper-search-mcp: uv run --with paper-search-mcp==0.1.3 python -m paper_search_mcp.server - Connected
pyfinagent-backtest: .venv/bin/python backend/agents/mcp_servers/backtest_server.py - Connected
pyfinagent-data: .venv/bin/python backend/agents/mcp_servers/data_server.py - Connected
pyfinagent-risk: .venv/bin/python backend/agents/mcp_servers/risk_server.py - Connected
pyfinagent-signals: .venv/bin/python backend/agents/mcp_servers/signals_server.py - Connected
playwright: npx -y @playwright/mcp@0.0.76 ... - Connected
```

Posture recorded (settings.local.json, untracked): enabledMcpjsonServers = the 7
names above; disabledMcpjsonServers = ["alpaca"]; stale "slack" entry removed.

## (c) Fresh Q/A verdict JSON

Returned by qa-67-5 2026-07-10: `verdict: PASS, ok: true, violated_criteria: [],
certified_fallback: false`, 18 checks_run -- all dry-runs re-run by the evaluator
(including its OWN boundary test: 2026-07-12 itself fires), `claude mcp list` re-run
live (7/7 Connected), settings.local set-equality vs .mcp.json keys verified
programmatically, tools-frontmatter-unchanged + exactly-2-agents doctrine checks.
Full JSON: evaluator_critique_67.5.md.
