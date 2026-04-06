# Phase 4.0: Move MAS to OpenClaw — PLAN

**Date:** 2026-04-06 22:16 GMT+2  
**Phase:** Phase 4.0 (PLAN phase)  
**Status:** Planning

## Goal

Move the Multi-Agent System (MAS) from a standalone Python orchestrator to OpenClaw-native agents. pyfinAgent backend becomes a pure data/compute service. All agent intelligence lives in OpenClaw.

## Current State (problems)

1. **Three separate processes:** OpenClaw gateway, pyfinAgent backend (uvicorn), Slack bot (socket mode)
2. **Two orchestrators:** OpenClaw agent runtime + Python MultiAgentOrchestrator — duplicate logic
3. **Same Slack bot token** shared between OpenClaw and pyfinAgent — can't run both simultaneously
4. **Event relay hack** — MAS events forwarded via HTTP POST between processes
5. **Direct Anthropic API calls** — MAS bypasses OpenClaw, no unified cost tracking
6. **Slack unresponsive** when OpenClaw main agent is busy — no dedicated agent for pyfinAgent channel

## Target State

```
OpenClaw Gateway (single process)
  ├── Agent: main (Ford) — personal assistant, webchat, iMessage
  ├── Agent: pyfinagent — dedicated Slack channel agent
  │   ├── System prompt: trading-focused orchestrator
  │   ├── Tools: exec (backtest API calls), web_search
  │   ├── Workspace: ~/.openclaw/workspace-pyfinagent
  │   │   ├── SOUL.md — pyfinAgent persona
  │   │   ├── AGENTS.md — agent behavior rules
  │   │   ├── TOOLS.md — backtest API reference
  │   │   └── memory/ — trading context
  │   └── Sessions: isolated from main
  │
  ├── Channels:
  │   ├── Slack C0ANTGNNK8D → bound to pyfinagent agent
  │   ├── iMessage → bound to main agent
  │   └── Webchat → bound to main agent
  │
  └── pyfinAgent Backend (localhost:8000) — data service only
      ├── /api/backtest/* — run backtests, optimizer
      ├── /api/mas/dashboard — system status
      ├── /api/health — health check
      └── Frontend (localhost:3000) — dashboard UI
```

## What Gets Eliminated

| Component | Current | After |
|-----------|---------|-------|
| `backend/slack_bot/` | Separate Python process, Socket Mode | **Deleted** — OpenClaw handles Slack natively |
| `MultiAgentOrchestrator` | Python class, direct Anthropic calls | **Deleted** — OpenClaw sessions_spawn for sub-agents |
| `streaming_integration.py` | Slack word-by-word streaming | **Deleted** — OpenClaw native streaming |
| `mas_events.py` event relay | HTTP POST from Slack bot → backend | **Deleted** — OpenClaw sessions are native |
| `openclaw_client.py` | Gateway HTTP client | **Deleted** — no longer needed |
| `agent_definitions.py` agent configs | Python dataclasses | **Moved** → OpenClaw agent config + SOUL.md |

## What Stays

| Component | Why |
|-----------|-----|
| FastAPI backend on :8000 | Backtest engine, optimizer, BQ queries, portfolio API |
| Frontend on :3000 | Dashboard, charts, UI |
| BigQuery | Data storage |
| `quant_optimizer.py` | Pure computation, no agent logic |
| `backtest_engine.py` | Pure computation |
| MAS Dashboard frontend | Shows OpenClaw sessions instead of custom event bus |

## Migration Steps

### Step 1: Create pyfinagent OpenClaw agent
- Create workspace at `~/.openclaw/workspace-pyfinagent`
- Write SOUL.md with trading-focused persona (from current system prompts)
- Write AGENTS.md with behavior rules
- Write TOOLS.md with backtest API reference
- Register in openclaw.json `agents.list`

### Step 2: Configure Slack binding
- Bind Slack channel C0ANTGNNK8D to `pyfinagent` agent
- Set `requireMention: false` (respond to all messages)
- Ensure main agent no longer receives pyfinAgent Slack messages

### Step 3: Give pyfinagent agent tool access
- Access to `exec` for calling backtest API via curl
- Access to `web_search` for research tasks
- Access to pyfinAgent workspace files (experiment results, PLAN.md)

### Step 4: Kill the separate Slack bot
- Stop `python -m backend.slack_bot.app` process
- Remove from startup scripts
- Verify Slack messages route through OpenClaw

### Step 5: Update MAS Dashboard
- Replace event bus SSE with OpenClaw sessions_list data
- Show pyfinagent agent sessions as "MAS activity"
- Keep backtest/optimizer status endpoints (they're backend, not agent)

### Step 6: Clean up dead code
- Delete `backend/slack_bot/` directory
- Delete `backend/agents/multi_agent_orchestrator.py`
- Delete `backend/agents/openclaw_client.py`
- Delete `backend/agents/mas_events.py` (event bus)
- Keep `backend/agents/agent_definitions.py` only if needed for reference

## Success Criteria

1. **Slack responsive** — pyfinagent agent answers within 5s, never blocked by main agent
2. **Backtest triggerable** — "run a backtest" on Slack → agent calls API → results
3. **No separate processes** — only OpenClaw gateway + uvicorn backend + frontend
4. **Cost tracking unified** — all agent calls visible in OpenClaw
5. **MAS Dashboard updated** — shows OpenClaw sessions instead of event bus
6. **Zero downtime** — old Slack bot killed only after new agent is verified working

## Risk Assessment

- **Low risk:** Steps 1-3 are additive, don't break anything existing
- **Medium risk:** Step 4 (killing Slack bot) — verify thoroughly first
- **Low risk:** Steps 5-6 are cleanup after migration is confirmed working

## Estimated Effort

- Steps 1-3: ~30 minutes (config + workspace files)
- Step 4: ~10 minutes (verify + kill)
- Steps 5-6: ~1-2 hours (dashboard update + code cleanup)
- Total: ~2-3 hours

## Decision: Sub-agents or Single Agent?

**Single pyfinagent agent** (recommended). The MAS had 4 agents (Communication, Ford, Q&A, Research) but the overhead of multi-agent classification + routing + quality gate costs more tokens than it saves. A single well-prompted agent with tool access can:
- Classify internally (no separate Communication Agent call)
- Answer Q&A directly (has access to backtest data)
- Do research (has web_search)
- Self-evaluate (cheaper than separate Quality Gate agent)

If we later need specialized sub-agents, OpenClaw's `sessions_spawn` can create them on-demand — no need to pre-register them.
