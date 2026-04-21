# Experiment Results — Cycle 4.15.14

Step: phase-4.15.14 Agent SDK + Managed Agents

## What was built

`docs/audits/compliance-sdk-managed-agents.md` (13 patterns).

## All live greps = 0

- `claude_agent_sdk` / `claude-agent-sdk` / `from claude_agent` — 0
- `@anthropic-ai/claude-agent-sdk` (frontend) — 0
- `managed_agents` / `/v1/agents` / `/v1/sessions` / `/v1/environments` — 0
- `ant-cli` / `bedrock-mantle` — 0
- No pilot tracker in docs/ or handoff/

## Findings (all confirm prior hybrid strategy)

### Agent SDK (9 patterns)
- Zero adoption — correct per hybrid strategy (Gemini Layer 1
  can't migrate; HMAC tokens stronger; markdown audit load-bearing)
- **Uncaptured Claude Code SDK levers** still unscheduled:
  - `excludeDynamicSections` (10-30% token cost reduction, 0 risk)
  - `CLAUDE_CODE_ENABLE_TELEMETRY=1` (OTel)
  - `enable_file_checkpointing` (replaces `.bak-*` manual backups)
  - `ENABLE_TOOL_SEARCH`
- **Layer 3 Agent SDK migration** (planner/evaluator, ~2-3 days)
  still unscheduled from phase-4.11 cluster E1

### Managed Agents (4 patterns)
- Pilot form not submitted. Phase-4.12 said "CONDITIONAL GO"; no
  evidence of submission.
- 4 blockers unresolved:
  1. In-process FastMCP needs public HTTPS deployment
  2. FastAPI on `:8000` unreachable from container (no VPC peering)
  3. No gcloud/bq/gh in container + no generic secret injection
  4. Permission policy too coarse for HMAC model
- `define_outcome` rubric grader + `callable_agents` parallel threads
  are strongest fits — both behind Research Preview access gate

## Novel findings: none

Pure confirmation of phase-4.11 + phase-4.12 prior deep audits.

## Success criteria

1. every_doc_pattern_status_evidenced — PASS (13 patterns)
2. qa_runs_live_code_checks_not_review — PARTIAL (Q/A next)
3. deviations_cite_doc_page — PASS

## Artifact

- `docs/audits/compliance-sdk-managed-agents.md`
