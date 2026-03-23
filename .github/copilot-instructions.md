# PyFinAgent — Copilot Instructions

> **Single source of truth**: [`AGENTS.md`](../AGENTS.md) at project root. This file is a lean pointer — do not duplicate content here.

## Quick Start

```bash
# Backend
./.venv312/Scripts/python.exe -m uvicorn backend.main:app --reload --port 8000

# Frontend
cd frontend && npm install && npx prisma migrate dev && npm run dev
```

`.venv312` is the canonical Python environment (Python 3.12). Frontend runs on **port 3000**, backend on **port 8000**.

## Key References

| Document | Scope |
|----------|-------|
| [`AGENTS.md`](../AGENTS.md) | Architecture overview, tech stack, 15-step pipeline, conventions, scoring |
| [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) | Full agent registry, 70+ API endpoints, BQ schemas, data tools |
| [`UX-AGENTS.md`](../UX-AGENTS.md) | Frontend component specs, design tokens, Phosphor icons |
| [`trading_agent.md`](../trading_agent.md) | Autonomous trading optimization, 3-loop architecture |
| [`CHANGELOG.md`](../CHANGELOG.md) | Version history (v1.0 → v5.9) |

## Path-Specific Instructions

Detailed conventions are in `.github/instructions/` (8 files with `applyTo:` frontmatter):

- `backend-agents.instructions.md` → `backend/agents/**`
- `backend-api.instructions.md` → `backend/api/**`, `config/**`, `db/**`, `tasks/**`
- `backend-backtest.instructions.md` → `backend/backtest/**`
- `backend-services.instructions.md` → `backend/services/**`
- `backend-tools.instructions.md` → `backend/tools/**`
- `backend-slack-bot.instructions.md` → `backend/slack_bot/**`
- `frontend.instructions.md` → `frontend/**`
- `security.instructions.md` → `backend/**`

## Custom Agents & Prompts

- `.github/agents/backtest-analyst.agent.md` — Read-only analyst for optimizer experiments, TSV logs, feature stability
- `.github/prompts/backtest-debug.prompt.md` — Diagnostic workflow for backtest/optimizer failures

## Mandatory Post-Change Checklist

After ANY code change, update:
1. `CHANGELOG.md` (new version entry)
2. `docs/ARCHITECTURE.md` (affected module descriptions)
3. `.github/instructions/` + `.claude/rules/` (keep pairs in sync)
4. `UX-AGENTS.md` if frontend/UX changed
