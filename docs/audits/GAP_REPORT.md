# Gap Report — Docs vs Implementation (phase-4.15 consolidation)

Produced: 2026-04-18.
Last updated: 2026-04-18 (phase-4.14 T1 close; phase-4.14 0/1/2 done).

## Executive summary

13 topic audits across ~250 documented patterns, every finding
live-verified by Q/A via `grep`, `jq`, `python -c`, not code
review. The phase-4.14 MUST-FIX list is 50 items (51 with MF-51
added this cycle).

**8 items fixed** (MF-40, 41, 42, 43, 44 prior cycle; MF-29, MF-1,
MF-2 closed 2026-04-18 via phase-4.14.0/1/2 harness batch with Q/A
PASS). **43 items remaining** across Tiers 1-5.
**HOTFIX status**: MF-45 / MF-46 (retired Haiku 3.5 / typo) — live
grep confirms zero matches in `backend/`, likely closed as side-effect
of MF-1's MODEL_PRICING rewrite; a dedicated close cycle should
formalize their status before the 2026-04-19 retirement deadline.

## Priority tiers (updated from v4)

### TIER 0 — SAME-DAY HOTFIX (2 items, new this phase)

| MF-# | Finding | Time-to-break |
|------|---------|---------------|
| **MF-45** | `claude-3-5-haiku-20241022` in 5 live files retires 2026-04-19 | ~12 hours |
| **MF-46** | `app_home.py:24` typo `claude-haiku-35-20241022` 400s if selected | immediate |

### TIER 1 — HIGHEST PRIORITY (1 item remaining, 3 closed 2026-04-18)

| MF-# | Finding |
|------|---------|
| MF-39 | `temperature=1` missing in MAS tool loop — 400s on Opus 4.6 thinking calls |

Closed this cycle (see FIXED section below): MF-29, MF-1, MF-2.

### TIER 2 — GO-LIVE BLOCKERS (structural)

| MF-# | Finding |
|------|---------|
| MF-35 | Consolidate Claude call sites behind `ClaudeClient` (Planner/Evaluator/MAS bypass) |
| MF-36 | `@enforce` capability tokens not wired on FastMCP tools (security story undermined) |
| MF-37 | `betas=` kwarg plumbing missing on `ClaudeClient` (blocks 3 separate wins) |
| MF-47 | `anthropic:` prefix in `_BUILD_TIER` silently routes to Gemini |
| MF-28 | Add `output_config.effort` pass-through — **FIXED-IN-ClaudeClient 2026-04-18 via phase-4.14.3**; MAS tool loop + planner_agent + autonomous_loop direct-client paths still bypass ClaudeClient (5 callsites); full closure blocked on MF-35 consolidation |
| MF-26 + MF-27 | Full stop_reason dispatch + max_tokens retry |
| MF-21 | Migrate Claude JSON to `output_config.format` structured outputs |
| MF-3 + MF-4 | Permission mode bypass → acceptEdits/auto + sandboxing |
| MF-5 + MF-6 | Opus 4.7 sampling-param guard + retired-model purge |

### TIER 3 — CORRECTNESS (8 items; established in v4)

Items 7 (stale snapshot), 9 (retry-after), 10 (exception classes),
11 (request_id), 12 (Gemini-only thinking gate), 13 (cache
threshold miss), 30 (citations × SO guard), 31 (citations on
filings).

### TIER 4 — HARDENING / DATA INTEGRITY (11 items)

14 BigQuery MCP drift, 15 prune stubs, 16 sub-agent descriptions
(DONE via MF-40/41), 17 claude.yml pin, 18 SDK bump 0.87→0.96,
19 cron_budget aspirational, 32 search_result blocks, 33 Files
API, 34 PDF native, **MF-50 claude.yml permissions read-only**,
MF-38 evaluator_agent mock path silent fallback.

### TIER 5 — HOUSEKEEPING (11 items)

20 Haiku 3 CI assert, 22 latency instrumentation, 23 harmlessness
pre-screen, 24 prompt-leak defenses, 25 "I don't know" permission,
v3 items 21/22/23 (PreToolUse + ConfigChange + InstructionsLoaded
hooks + .bak files cleanup), **MF-48 cache-write premium missing**,
**MF-49 dead Bash allow rules**.

### FIXED THIS CYCLE (8 items — 5 prior + 3 closed 2026-04-18 via phase-4.14 T1 batch)

| MF-# | What was fixed | Closed by |
|------|----------------|-----------|
| MF-40 | `permissionMode: plan` added to both merged agents | prior |
| MF-41 | qa.md `NEVER Edit or Write` constraint rewritten with Bash allow-list | prior |
| MF-42 | `SubagentStop` hook added to `.claude/settings.json` | prior |
| MF-43 | Separation-of-duties note added to CLAUDE.md | prior |
| MF-44 | Session-restart requirement noted in CLAUDE.md | prior |
| MF-29 | Opus 4.7 thinking-API gate: `ClaudeClient._call_claude` routes 4.7/4.6/sonnet-4.6/haiku-4.5 to `{"type":"adaptive"}`, legacy models to `{"type":"enabled","budget_tokens":N}`, `temperature=1` forced on both paths (`llm_client.py:638-653`) | phase-4.14.0 (2026-04-18) |
| MF-1 | `MODEL_PRICING` now correct for all 7 current Claude 4-family models + retired Haiku 3.5 / Sonnet 3.7 rows removed (`cost_tracker.py:20-76`). Cost-dashboard no longer under-reports by 50-187×. | phase-4.14.1 (2026-04-18) |
| MF-2 | Settings hardening: `.claude/settings.json:permissions.deny` now blocks `mcp__alpaca__{place_order,cancel_order,replace_order,close_position,close_all_positions}` + `mcp__bigquery__execute_sql` + Bash safety denies. `.claude/settings.local.json` contradiction removed. | phase-4.14.2 (2026-04-18) |
| MF-28 (partial) | `output_config.effort` pass-through wired in `ClaudeClient` with per-agent-class defaults (`model_tiers.py::EFFORT_DEFAULTS` + `resolve_effort` + `resolve_effort_by_model`), xhigh-Opus-4.7-only guard, and model-support allowlist. Effort is now applied independent of thinking (the MF-29 bug was effort-only-when-thinking-on). FULL closure of MF-28 requires MF-35 consolidation (5 callsites still bypass ClaudeClient: `multi_agent_orchestrator.py:165`, `planner_agent.py`, `planner_enhanced.py`, `services/autonomous_loop.py:419`, `ticket_queue_processor.py:178`). | phase-4.14.3 (2026-04-18) |

### NEW THIS CYCLE (1 follow-up)

| MF-# | Finding | Tier |
|------|---------|------|
| MF-51 | Phase-4.14.0/1/2 verification commands are weak (4.14.0 short-circuits on the word "adaptive" anywhere in the file; 4.14.1 checks key presence not values; 4.14.2 asserts only `alpaca__place_order` substring). Harden to AST-based / value-based assertions. Raised by Q/A during phase-4.14 T1 close. | Tier 5 (housekeeping) |

## Decisions required (expanded from v4's 11 to 14)

1. Approve all 50 MUST-FIX items (or prune / reprioritize)
2. Pick 3-5 NICE-TO-HAVE clusters for phase-4.16+ scope
3. Routines-vs-cron for phase-10.7 Meta-Evolution
4. Agent SDK: port planner + evaluator only (MF-35 + cluster E1)?
5. Managed Agents: submit access form with narrow pilot scope?
6. Structured Outputs migration blocking go-live?
7. Rename `skills/` → `prompt_templates/`?
8. Eval suite (Cluster K) phasing
9. ZDR compliance target declared yes/no (gates Files API + Batches)
10. MF-29 same-day hotfix?
11. MF-45/MF-46 same-day hotfix (HAIKU 3.5 RETIRES TOMORROW)?
12. **(NEW) MF-35 `ClaudeClient` consolidation**: priority before
    any other MF because it's the propagation path
13. **(NEW) MF-36 wire HMAC capability tokens to FastMCP**: design
    decision needed (decorator vs middleware vs request-context var)
14. **(NEW) MF-39 `temperature=1` in MAS tool loop**: silent
    currently-live-bug on Opus 4.6 — hotfix?

## Running file count

- `docs/audits/compliance-*.md` — 13 files (one per topic)
- `docs/audits/COMPLIANCE_MATRIX.md` — summary
- `docs/audits/GAP_REPORT.md` — this file
- `handoff/current/phase-4.15.*` — 13 contracts + 13
  experiment-results + 13 evaluator-critiques + harness_log
  appends

All under `handoff/archive/phase-4.15/` on next masterplan flip
triggering the `archive-handoff` PostToolUse hook.

## Next phase

- Phase-4.14 Implementation (gate already approved, 29 steps)
- Phase-4.16 or extend phase-4.14 to absorb the 16 new MF items
- Session restart required to dispatch merged `qa` agent
