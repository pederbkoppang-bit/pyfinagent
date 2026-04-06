# Phase 4.0: Move MAS to OpenClaw — PLAN (v3 — Full Codebase Review)

**Date:** 2026-04-06 22:24 GMT+2  
**Phase:** Phase 4.0 (PLAN phase — after full codebase + research review)  
**Status:** Planning

## Research Reviewed

1. **Anthropic: Multi-Agent Research System** — orchestrator-worker, parallel subagents, interleaved thinking, separation of generation from evaluation
2. **Anthropic: Harness Design for Long-Running Apps** — generator + evaluator, file-based artifact handoff
3. **Our RESEARCH_AGENTIC_COORDINATION_LOOP.md** — 4-session architecture (Coordinator, Q&A, Research, Slack), approved by Peder
4. **Our agent_definitions.py** — 4 agents with system prompts, quality criteria anchors, delegation rules
5. **Our multi_agent_orchestrator.py** — full flow: classify → plan → parallel delegate → synthesize → quality gate → citation
6. **Our harness_state_reader.py** — 7 read-only tools for agents to access harness data
7. **Our harness_memory.py** — episodic + semantic memory for context injection

## Current Architecture (What Exists)

```
┌─────────────────────────────────────────────────────────────┐
│ Python Slack Bot (Socket Mode) — separate process           │
│  └─ StreamingIntegration                                    │
│      └─ MultiAgentOrchestrator                              │
│          ├─ Communication Agent (Sonnet) — classify/route   │
│          ├─ Ford/Main Agent (Opus) — plan, synthesize       │
│          ├─ Q&A Agent (Opus) — quantitative analysis        │
│          ├─ Research Agent (Sonnet) — evidence gathering    │
│          ├─ Quality Gate (Sonnet) — skeptical review        │
│          └─ Citation Agent (Sonnet) — source attribution    │
│                                                             │
│ Agents have 7 harness tools (read_evaluator_critique, etc.) │
│ Agents get quality criteria + memory injection              │
│ Direct anthropic.Anthropic() API calls                      │
│ Interleaved thinking (budget_tokens: 2048)                  │
│ Parallel subagent execution via ThreadPoolExecutor          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ FastAPI Backend (:8000) — separate process                  │
│  ├─ Backtest engine, optimizer, BQ queries                  │
│  ├─ MAS events SSE (event bus — in-memory, per-process)     │
│  └─ Harness data API endpoints                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ OpenClaw Gateway — separate process                         │
│  ├─ Main agent (Ford) — webchat, iMessage                   │
│  ├─ Slack connected (same bot token as Python Slack bot!)    │
│  └─ Can't run both simultaneously                           │
└─────────────────────────────────────────────────────────────┘
```

**Problems:**
1. Same Slack bot token shared — can't run both OpenClaw Slack + Python Slack bot
2. MAS events in-memory, per-process — Slack bot events never reach backend SSE
3. 3 processes to manage
4. Direct Anthropic calls bypass OpenClaw cost tracking

## Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ OpenClaw Gateway (single agent runtime)                     │
│                                                             │
│  Agent: main (Ford)                                         │
│    Bound to: webchat, iMessage                              │
│    Model: Opus                                              │
│    Role: Personal assistant                                 │
│                                                             │
│  Agent: pyfinagent (Lead Researcher)                        │
│    Bound to: Slack channel C0ANTGNNK8D                      │
│    Model: Opus                                              │
│    Workspace: ~/.openclaw/workspace-pyfinagent              │
│    Role: Orchestrator — plans, delegates, synthesizes       │
│    Tools:                                                   │
│      ├─ exec — call backtest API (curl localhost:8000/...)  │
│      ├─ web_search — research                               │
│      ├─ sessions_spawn — create parallel subagents          │
│      └─ Read — harness files (workspace has symlinks)       │
│                                                             │
│    Subagents (spawned on-demand via sessions_spawn):        │
│      ├─ QA subagent (Sonnet) — quantitative analysis        │
│      ├─ Research subagent (Sonnet) — web search, evidence   │
│      ├─ Quality Gate subagent (Sonnet) — skeptical review   │
│      └─ Citation subagent (Sonnet) — source attribution     │
│                                                             │
│  Channels:                                                  │
│    ├─ Slack — routed to pyfinagent agent                    │
│    ├─ iMessage — routed to main agent                       │
│    └─ Webchat — routed to main agent                        │
│                                                             │
│  Native features replacing custom code:                     │
│    ├─ Session tracking (replaces event bus)                  │
│    ├─ Cost tracking (replaces nothing — was missing)        │
│    ├─ Memory (MEMORY.md replaces harness_memory.py)         │
│    ├─ Streaming (native Slack streaming)                    │
│    └─ Cron (master plan execution)                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ FastAPI Backend (:8000) — data/compute service only         │
│  ├─ /api/backtest/* — run backtests, optimizer              │
│  ├─ /api/mas/dashboard — pulls from OpenClaw sessions_list  │
│  ├─ /api/backtest/harness/* — read harness artifacts        │
│  └─ /api/health                                             │
└─────────────────────────────────────────────────────────────┘
```

## Agent Flow (Matching Anthropic's Diagram)

### For Simple Queries (trivial/simple — ~60% of messages)
```
User: "What's our Sharpe?"
  → pyfinagent Lead reads harness data directly
  → Responds immediately (no subagents)
  → Cost: 1 Opus turn (~$0.02)
```

### For Moderate Queries (comparison, analysis)
```
User: "Compare our triple barrier strategy to buy-and-hold"
  → Lead thinks(plan): need QA analysis + data comparison
  → sessions_spawn: QA subagent (Sonnet)
      Task: "Analyze our backtest Sharpe vs buy-and-hold. Read experiment log."
      Tools: exec (curl backtest API), Read (harness files)
  → Lead synthesizes QA findings
  → Cost: 1 Opus + 1 Sonnet (~$0.05)
```

### For Complex Queries (multi-faceted research)
```
User: "Research regime detection improvements for our strategy"
  → Lead thinks(plan): need Research + QA + Quality Gate
  → sessions_spawn (parallel):
      ├─ Research subagent: "Find papers on regime detection for quant trading"
      │   Tools: web_search, web_fetch
      ├─ QA subagent: "Analyze our current regime sensitivity across sub-periods"
      │   Tools: exec (curl backtest API)
  → sessions_yield (wait for both)
  → Lead thinks(synthesize): combine findings
  → Lead thinks: "More research needed?" → if yes, spawn more
  → sessions_spawn: Quality Gate subagent
      Task: "Review this synthesis. Is it actionable? Statistically grounded?"
  → Lead incorporates quality feedback
  → sessions_spawn: Citation subagent
      Task: "Add source citations to this response"
  → Final response to user
  → Cost: 1 Opus + 3-4 Sonnet (~$0.12)
```

### For Harness Triggers (Tier 3)
```
User: "Run a harness cycle" or Lead decides autonomously
  → Lead calls exec: curl -X POST localhost:8000/api/backtest/optimize/start
  → Lead monitors: curl localhost:8000/api/backtest/optimize/status
  → On completion, Lead reads results, spawns QA to analyze
  → Lead writes summary to Slack
```

## Workspace Structure: ~/.openclaw/workspace-pyfinagent

```
workspace-pyfinagent/
├── SOUL.md          — Lead agent persona (from Ford/Main system prompt + delegation rules)
├── AGENTS.md        — Workspace behavior (from RESEARCH_AGENTIC_COORDINATION_LOOP.md)
├── TOOLS.md         — Backtest API reference (every endpoint with curl examples)
├── USER.md          — Peder's context
├── MEMORY.md        — Long-term trading context (migrated from harness_memory)
├── memory/          — Daily logs
├── harness/         — Symlink → pyfinagent/handoff/ (read-only harness artifacts)
├── experiments/     — Symlink → pyfinagent/backend/backtest/experiments/
├── PLAN.md          — Symlink → pyfinagent/PLAN.md
└── RESEARCH.md      — Symlink → pyfinagent/RESEARCH.md
```

Symlinks give the agent direct file access to harness data without HTTP round-trips.
This replaces the 7 custom harness tools with native Read tool access.

## Subagent Prompt Templates

Stored in workspace as reference files. Lead agent reads these and customizes 
the task parameter when calling sessions_spawn.

### QA Subagent Template
```
You are the Analyst for pyfinAgent. Quantitative reasoning specialist.
{quality_criteria}
OBJECTIVE: {objective}
OUTPUT FORMAT: {format}
TOOLS: You can read files and run exec commands to query the backtest API.
TASK BOUNDARY: {boundary} — analyze existing results only, never modify state.
```

### Research Subagent Template  
```
You are the Researcher for pyfinAgent. Evidence-backed insights specialist.
{quality_criteria}
OBJECTIVE: {objective}
Search strategy: Start broad, then narrow. Cite ALL sources.
TOOLS: web_search, web_fetch for papers. Read for local files.
TASK BOUNDARY: {boundary} — find and synthesize information only.
```

### Quality Gate Subagent Template
```
You are the Quality Gate — a skeptical reviewer. 
NEVER praise. ALWAYS find weaknesses.
{quality_criteria}
RESPONSE TO REVIEW: {response}
Score each criterion 1-10. Below 6 on ANY = FAIL.
Output: {verdict: PASS|FAIL, scores: {...}, weaknesses: [...], improvements: [...]}
```

## Migration Steps

### Step 1: Create pyfinagent workspace + agent config
- Create workspace files (SOUL.md, AGENTS.md, TOOLS.md, USER.md)
- Create symlinks to harness/experiments/PLAN/RESEARCH
- Register agent in openclaw.json with model, workspace, agentDir
- Store subagent prompt templates in workspace

### Step 2: Configure Slack binding  
- Add binding: Slack C0ANTGNNK8D → pyfinagent agent
- Remove channel from main agent's scope
- Restart gateway

### Step 3: Test without killing old bot
- Old Slack bot is already killed (same token, can't coexist)
- Send test messages: trivial, moderate, complex
- Verify subagent spawning works
- Verify harness data readable via symlinks

### Step 4: Update MAS Dashboard
- Backend `/api/mas/dashboard` calls OpenClaw sessions_list (via /tools/invoke)  
  filtered by agentId=pyfinagent
- Frontend shows these as MAS activity

### Step 5: Clean up dead code
- Delete: backend/slack_bot/, multi_agent_orchestrator.py, 
  openclaw_client.py, mas_events.py (event bus), streaming_integration.py
- Keep: agent_definitions.py (reference), harness_state_reader.py (backend API still uses it)

## What Changes vs Previous Plan

| Previous Plan | Revised Plan |
|--------------|--------------|
| Single flat agent | Multi-agent: Lead + spawned subagents ✅ |
| Missing Communication Agent | Lead handles classification internally via thinking ✅ |
| No tool specification | TOOLS.md with full API reference + symlinks for harness ✅ |
| No memory migration | MEMORY.md + symlinks to harness artifacts ✅ |
| Vague dashboard update | Sessions_list filtered by agentId ✅ |
| No subagent prompts | Template-based prompts matching current system prompts ✅ |
| No cost analysis | Simple/Moderate/Complex cost breakdown ✅ |

## Success Criteria

1. **Slack responsive** — simple queries answered in <10s, complex in <120s
2. **Subagents spawn in parallel** — complex queries use 2-5 parallel subagents
3. **Harness data accessible** — agent reads experiments, critiques, plans via file access
4. **Quality gate separate** — lead NEVER evaluates own output
5. **No separate processes** — gateway + backend + frontend only
6. **Cost tracking unified** — all calls visible in OpenClaw
7. **Dashboard updated** — shows OpenClaw sessions for pyfinagent

## Estimated Effort
- Step 1 (workspace + config): ~45 min
- Step 2 (Slack binding): ~15 min  
- Step 3 (testing): ~30 min
- Step 4 (dashboard): ~1 hour
- Step 5 (cleanup): ~1 hour
- Total: ~3-4 hours
