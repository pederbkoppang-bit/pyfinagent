---
name: PyFinAgent Project Context
description: Core project info, architecture, and infrastructure for any agent working on this codebase
---

PyFinAgent is an autonomous AI-powered trading signal system. Goal: generate validated signals delivered via Slack for manual trading by May 2026.

**Budget:** Tight (~$230-350/mo, negative cash flow). Every dollar must have ROI. LLM API costs require Peder's explicit approval.

**Stack:** FastAPI (Python 3.14) + Next.js 15 + BigQuery + Gemini + Claude
**BigQuery:** Project `sunny-might-477607-p8`, accessible via MCP (read/write) in harness sessions — see `CLAUDE.md` → "BigQuery Access (MCP)" for tools, datasets, and safety rules.
**Current best:** Sharpe 1.1705, DSR 0.9984
**Architecture:** 4 layers, 38+ agents (Analysis Pipeline, MAS Orchestrator, Harness, Services)

**Key infrastructure:**
- Machine-readable masterplan: `.claude/masterplan.json` (6 phases, 29 steps)
- Three subagents: researcher, qa-evaluator, harness-verifier (in `.claude/agents/`)
- `/masterplan` skill for navigating plan state
- Hooks: changelog on commit, memory sync on masterplan changes, TaskCompleted gate, Stop gate, TeammateIdle
- Agent teams enabled: `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`
- Handoff: `handoff/current/` (active cycle), `handoff/archive/` (historical by phase)

**All work traces back to PLAN.md via .claude/masterplan.json. Every change follows: RESEARCH -> PLAN -> GENERATE -> EVALUATE -> LOG.**
