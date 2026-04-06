# Phase 4.0: Move MAS to OpenClaw — PLAN (Revised)

**Date:** 2026-04-06 22:20 GMT+2  
**Phase:** Phase 4.0 (PLAN phase — revised after Anthropic article review)  
**Status:** Planning

## Key Insight from Anthropic's Multi-Agent Research System

> "Multi-agent systems work mainly because they help spend enough tokens to solve 
> the problem. Token usage by itself explains 80% of the performance variance."

The orchestrator-worker pattern is correct. We should NOT flatten to a single agent. 
Instead, we move the orchestration FROM Python TO OpenClaw's native `sessions_spawn`.

## Current State → Target State

### Current (Python orchestrator)
```
Slack msg → Python Slack Bot → MultiAgentOrchestrator (Python)
  → anthropic.Anthropic().messages.create() × 4-6 agents
  → Quality Gate → Citation → Response
```

### Target (OpenClaw orchestrator)
```
Slack msg → OpenClaw Gateway → pyfinagent Lead Agent (Opus)
  → sessions_spawn: QA subagent (Sonnet) — parallel
  → sessions_spawn: Research subagent (Sonnet) — parallel
  → Lead synthesizes findings
  → Lead evaluates quality (or spawns quality-gate subagent)
  → Lead adds citations
  → Response to Slack
```

## Architecture (matching Anthropic's diagram)

```
User Query (Slack / iMessage / Webchat)
         │
         ▼
┌─────────────────────────────┐
│  Lead Agent (pyfinagent)    │  Opus — orchestrator
│  think(plan approach)       │
│  save plan → Memory         │
│  retrieve context           │
└──────────┬──────────────────┘
           │ sessions_spawn (parallel)
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌──────────┐
│ QA      │  │ Research  │  Sonnet subagents
│ Subagent│  │ Subagent  │  (each: search → think → complete)
└────┬────┘  └─────┬─────┘
     │             │
     └──────┬──────┘
            ▼
┌─────────────────────────────┐
│  Lead Agent                 │
│  think(synthesize results)  │
│  "More research needed?"    │
│  → if yes: spawn more       │
│  → if no: quality check     │
│  → add citations            │
│  → return to user           │
└─────────────────────────────┘
```

### Key Anthropic Principles Applied

1. **Teach the orchestrator how to delegate** — Lead agent's SOUL.md includes 
   delegation patterns: objective, output format, tool guidance, task boundaries
   
2. **Scale effort to query complexity** — Simple fact → no subagents. 
   Comparison → 2-3 subagents. Deep research → 5+ subagents with divided responsibilities.
   
3. **Interleaved thinking** — Subagents use extended thinking after tool results 
   to evaluate quality, identify gaps, refine queries.
   
4. **Parallel tool calling** — Lead spawns 3-5 subagents in parallel via sessions_spawn.
   Each subagent uses tools in parallel.
   
5. **Start wide, then narrow** — Subagents start with broad queries, evaluate landscape, 
   then drill into specifics.

## OpenClaw Agent Configuration

### Lead Agent: `pyfinagent`
- **Model:** anthropic/claude-opus-4-6
- **Role:** Orchestrator — plans, delegates, synthesizes, evaluates
- **Workspace:** `~/.openclaw/workspace-pyfinagent`
- **Tools:** exec (backtest API), web_search, sessions_spawn (create subagents)
- **Bound to:** Slack channel C0ANTGNNK8D
- **System prompt themes:**
  - You are the Lead Researcher for pyfinAgent trading system
  - Classify query complexity → decide subagent count
  - Decompose into subtasks with clear objectives per subagent
  - Synthesize results, check quality, add citations
  - Access harness data: experiments, plans, evaluator critiques

### Subagents (spawned on-demand via sessions_spawn)
- **Model:** anthropic/claude-sonnet-4-6 (cheaper, still capable)
- **Mode:** `run` (one-shot, terminated after returning results)
- **Each gets:** specific objective, output format, tool access, task boundary
- **Types:**
  - **QA Subagent** — quantitative analysis, harness data, experiment comparison
  - **Research Subagent** — web search, literature, evidence gathering
  - **Quality Gate** — skeptical review of lead's synthesis (separate generation from evaluation)

## Migration Steps

### Step 1: Create pyfinagent Lead Agent workspace
- `~/.openclaw/workspace-pyfinagent/SOUL.md` — orchestrator persona + delegation rules
- `~/.openclaw/workspace-pyfinagent/AGENTS.md` — workspace behavior
- `~/.openclaw/workspace-pyfinagent/TOOLS.md` — backtest API reference + tool descriptions
- Register in openclaw.json with Opus model

### Step 2: Configure Slack binding
- Bind Slack channel C0ANTGNNK8D → pyfinagent agent
- Main agent keeps webchat + iMessage
- pyfinagent agent is independent — never blocked by main

### Step 3: Write subagent prompt templates
- QA subagent system prompt (quantitative focus, harness tools)
- Research subagent system prompt (web search, evidence gathering)
- Quality Gate prompt (skeptical reviewer, never self-evaluate)
- These are passed via sessions_spawn task parameter

### Step 4: Test with parallel queries
- Simple: "What's our Sharpe?" → lead answers directly, no subagents
- Medium: "Compare our strategy to buy-and-hold" → 2 subagents
- Complex: "Research regime detection improvements" → 3+ subagents in parallel

### Step 5: Kill old Slack bot + clean up
- Stop separate Slack bot process
- Delete `backend/slack_bot/`, `multi_agent_orchestrator.py`, event relay
- Keep backend as data service

### Step 6: Update MAS Dashboard
- Show OpenClaw sessions (lead + subagent runs)
- Replace event bus with sessions_list polling

## What Gets Eliminated vs Kept

| Eliminated | Replaced By |
|-----------|-------------|
| Python MultiAgentOrchestrator | OpenClaw lead agent + sessions_spawn |
| Direct Anthropic API calls | OpenClaw native model calls |
| Python Slack bot (Socket Mode) | OpenClaw Slack channel |
| Event bus + HTTP relay | OpenClaw native sessions |
| classify_trivial() + _classify_via_llm() | Lead agent internal classification |
| streaming_integration.py | OpenClaw native Slack streaming |

| Kept | Why |
|------|-----|
| FastAPI backend :8000 | Pure data/compute service |
| backtest_engine.py | Computation, no agent logic |
| quant_optimizer.py | Computation, no agent logic |
| Frontend :3000 | Dashboard UI |
| agent_definitions.py | Reference for system prompts |

## Success Criteria

1. **Slack responsive within 5s** — pyfinagent agent answers, never blocked by main
2. **Subagents spawn in parallel** — complex queries use 2-5 parallel subagents
3. **Quality maintained** — responses match or exceed old MAS quality
4. **Token tracking unified** — all calls visible in OpenClaw
5. **No separate processes** — gateway + backend + frontend only
6. **Separation of generation and evaluation** — lead never evaluates own output

## Estimated Effort
- Steps 1-3: ~1 hour (workspace + config + prompts)
- Step 4: ~30 min (testing)
- Steps 5-6: ~1-2 hours (cleanup + dashboard)
- Total: ~3-4 hours
