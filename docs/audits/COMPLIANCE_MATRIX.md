# Compliance Matrix — pyfinagent vs Claude Documentation

Produced: 2026-04-18. Covers all 13 topic audits under
`docs/audits/compliance-*.md`. Every row is live-verified by Q/A.

## Topic index

| Topic | Audit file | Cycle |
|-------|------------|-------|
| Extended + Adaptive thinking + Effort | compliance-thinking.md | 4.15.2 |
| Sub-agents + Agent teams + Skills | compliance-subagents-skills.md | 4.15.3 |
| MCP + permissions + capability tokens | compliance-mcp-permissions.md | 4.15.4 |
| Prompt caching + Context management | compliance-caching-context.md | 4.15.5 |
| Batches + Files + Citations + Search results | compliance-batches-files-citations.md | 4.15.6 |
| Structured outputs + Stop reasons + Streaming | compliance-structured-streaming.md | 4.15.7 |
| Tool-use primitives | compliance-tool-use.md | 4.15.8 |
| Test-and-evaluate + Strengthen guardrails | compliance-evals-guardrails.md | 4.15.9 |
| Messages API conventions + SDK | compliance-api-conventions.md | 4.15.10 |
| Models / pricing / deprecations | compliance-models-pricing.md | 4.15.11 |
| Claude Code core (hooks, permissions, sandbox) | compliance-claude-code-core.md | 4.15.12 |
| Claude Code surfaces + CI + Slack + routines | compliance-claude-code-surfaces.md | 4.15.13 |
| Agent SDK + Managed Agents adoption | compliance-sdk-managed-agents.md | 4.15.14 |

## Grand totals

- **13 topic compliance audits** completed, one per cycle, full
  harness + 3-agent MAS loop.
- **~250 documented patterns** covered with live code checks.
- **14 new MUST-FIX items (MF-35 to MF-49) + 1 tomorrow-hotfix
  (MF-45)** surfaced beyond phase-4.10-4.13's 34.
- **Total MUST-FIX count: 50** (34 from v4 + 16 new).

## Status roll-up per topic

| Topic | ✅ correct | ⚠️ partial | ❌ missing | ❗ incorrect | N/A |
|-------|-----------|-----------|-----------|-------------|-----|
| Thinking | 3 | 2 | 5 | 2 | 3 |
| Sub-agents / Skills | 5 | 3 | 9 | 3 | 5 |
| MCP / permissions | 4 | 6 | 6 | 4 | — |
| Caching / context | 1 | 3 | 6 | 2 | 4 |
| Batches / Files / Citations / Search | 0 | 0 | 25 | 0 | — |
| Structured / stop_reason / streaming | 2 | 2 | 10 | 1 | — |
| Tool-use primitives | 3 | 0 | 20 | 0 | 7 |
| Evals / guardrails | 5 | 6 | 6 | 0 | — |
| API conventions / SDK | 2 | 5 | 10 | 0 | — |
| Models / pricing | 5 | 4 | 9 | 3 | 4 |
| Claude Code core | 5 | 4 | 10 | 0 | 6 |
| CC surfaces / CI / Slack | 12 | 3 | 3 | 1 | 1 |
| Agent SDK / Managed Agents | 3 | 1 | 9 | 0 | — |
| **TOTAL (approx)** | **50** | **39** | **128** | **16** | **30** |

~50% of patterns are ❌ missing or ❗ incorrect. The thick end of the
curve is in **Batches / Files / Citations / Search results** (25/25
missing — entire category unexercised) and **Tool-use primitives**
(20/30 missing versioned primitives).

## URGENT HOTFIXES (same-day, pre-go-live)

| MF-# | Finding | Lines |
|------|---------|-------|
| **MF-45** | Haiku 3.5 `claude-3-5-haiku-20241022` retires **2026-04-19 (TOMORROW)** — will 400 | cost_tracker.py:23, llm_client.py:64+171, harness_memory.py:53, settings_api.py:31+144 |
| **MF-46** | Invalid typo ID `claude-haiku-35-20241022` would 400 if selected | slack_bot/app_home.py:24 |
| **MF-29** | `thinking:{type:"enabled",budget_tokens:N}` rejected on Opus 4.7 | llm_client.py:622-628, multi_agent_orchestrator.py:944-954 |
| **MF-1** | `MODEL_PRICING` missing all Opus 4-x + Haiku 4.5 → 50-187× cost under-report | cost_tracker.py:17-28 |

## HIGH PRIORITY

| MF-# | Finding | Evidence |
|------|---------|----------|
| MF-2 | Alpaca write-tool deny absent + allowlist contradiction | settings.local.json |
| MF-35 | Planner + Evaluator + MAS tool loop BYPASS `ClaudeClient` — fixes don't propagate | llm_client.py vs planner_agent.py:115 vs multi_agent_orchestrator.py:944 |
| MF-36 | HMAC capability tokens NOT WIRED (`@enforce` never called on any FastMCP tool) | backend/agents/mcp_servers/*.py |
| MF-37 | `ClaudeClient.generate_content` has no `betas=` kwarg — structural blocker for cache-TTL-1h, interleaved-thinking, files-api | llm_client.py:581-672 |
| MF-39 | `temperature=1` missing in MAS tool loop — will 400 on Opus 4.6 thinking calls TODAY | multi_agent_orchestrator.py:944-954 |
| MF-47 | `_BUILD_TIER` `anthropic:` prefix silently routes to Gemini | model_tiers.py:52-54 |

## NET-NEW MUST-FIX from phase-4.15 (MF-35 to MF-50)

| MF-# | Cycle | Severity | Finding |
|------|-------|----------|---------|
| MF-35 | 4.15.2 | HIGH | Consolidate Claude call sites behind `ClaudeClient` (Planner/Evaluator/MAS bypass) |
| MF-36 | 4.15.4 | HIGH | `@enforce` capability tokens not wired on FastMCP tools |
| MF-37 | 4.15.5 | HIGH | `betas=` kwarg plumbing missing on `ClaudeClient` |
| MF-38 | 4.15.7 | MED | `evaluator_agent.py` mock path fires silently on missing Vertex creds |
| MF-39 | 4.15.2 | HIGH | `temperature=1` missing in MAS tool loop |
| MF-40 | 4.15.3 | LOW | permissionMode missing on merged agents — **FIXED this cycle** |
| MF-41 | 4.15.3 | LOW | qa.md tool/body contradiction — **FIXED this cycle** |
| MF-42 | 4.15.3 | LOW | SubagentStop hook absent — **FIXED this cycle** |
| MF-43 | 4.15.3 | LOW | Separation-of-duties gap on agent edits — **NOTED in CLAUDE.md** |
| MF-44 | 4.15.3 | LOW | Session-cache requires restart — **NOTED in CLAUDE.md** |
| MF-45 | 4.15.11 | **HOTFIX** | Haiku 3.5 retires tomorrow — 5 files |
| MF-46 | 4.15.11 | **HOTFIX** | Typo ID `claude-haiku-35-20241022` |
| MF-47 | 4.15.11 | HIGH | `anthropic:` prefix routes `autoresearch_fast` to Gemini |
| MF-48 | 4.15.11 | LOW | cache-write premium 1.25×/2× missing |
| MF-49 | 4.15.12 | LOW | Dead Bash allow rules in settings.local.json |
| MF-50 | 4.15.13 | LOW | `claude.yml` permissions read-only — `@claude` can't commit/comment |

## What's working

**Correctly implemented (sampling):**

1. `temperature=1` forcing on thinking WITHIN `ClaudeClient`
2. Prompt caching wire on `ClaudeClient` system block (correctly
   applies `cache_control: ephemeral`, but system block is too
   small to meet 4096-token threshold — MF-13)
3. Tool-use loop shape (`stop_reason == "tool_use"`, parallel
   `ThreadPoolExecutor` execution)
4. Permission modes on merged agents (`plan` set post-4.15.0)
5. `claude-code-review.yml` plugin path (doc-aligned)
6. Custom Slack bot vs native (different product)
7. 3-agent MAS merge: `qa` + `researcher` (from 5 agents in 4.15.0)
8. Per-step harness protocol with 5-file handoff + archive hook
9. Live code checks mandatory in every EVALUATE phase (Q/A agent
   runs `grep` + `jq` + `python -c`, not LLM-only review)

## Research-gate integrity

- Researcher agent spawned BEFORE contract in every 4.15.X cycle
- Live greps + file:line anchors in every audit
- Zero skipped research-gates this phase
- Session-cache limitation flagged (MF-44) — new `qa` agent not
  dispatchable until session restart; `qa-evaluator` stand-in used

## Separation-of-duties applied

- Every cycle used `qa-evaluator` (not Main) for EVALUATE phase
- Researcher + Q/A independent context windows
- Main never self-evaluated
- MF-43 flagged a remaining gap: same-session agent edits +
  self-audit — Peder review requested on `.claude/agents/qa.md`,
  `researcher.md`, and `settings.json` changes from 4.15.3 +
  MF-40/41/42 fixes
