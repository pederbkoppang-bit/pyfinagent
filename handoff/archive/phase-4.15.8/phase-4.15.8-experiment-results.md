# Experiment Results — Cycle 4.15.8

Step: phase-4.15.8 Tool-use primitives compliance

## What was built

`docs/audits/compliance-tool-use.md` (~2000 words, 25+ patterns).

## Zero-usage confirmed via live grep (all 8 checks = 0)

All 8 Anthropic versioned tool primitives absent from pyfinagent:
- `advisor_20260301` — 0
- `memory_20250818` — 0
- `web_search_20260209` / `web_search_20250305` — 0
- `web_fetch_20260209` / `web_fetch_20250910` — 0
- `code_execution_20250825` / `20260120` — 0
- `computer_20251124` / `text_editor_20250728` / `bash_20250124` — 0
- `tool_search_tool_regex/bm25` + `defer_loading` — 0
- `allowed_callers` (programmatic tool calling) — 0
- `eager_input_streaming` / fine-grained tool streaming — 0

## Summary by finding type

**Genuine gaps (2 — map to existing MF):**

| # | Finding | Evidence | MF-# |
|---|---------|----------|------|
| 1 | `strict: true` missing on all 7 AGENT_TOOLS | `multi_agent_orchestrator.py:72-120` | MF-5 / prior phase-4.14 |
| 2 | `cache_control` missing on AGENT_TOOLS array | `multi_agent_orchestrator.py:944-954` | Cluster A1 (phase-4.14) |

**Adoption opportunity (1 — MF-23):**

- **Advisor tool (`advisor_20260301`)** Sonnet-executor + Opus-4.7-
  advisor pattern for MAS planner/evaluator. Already in phase-4.14
  NICE-TO-HAVE list. Requires Peder cost approval.

**Correct non-adoptions (confirmed, no action):**

- `memory_20250818` — custom BQ-backed memory is more appropriate
- `bash_20250124`, `text_editor_20250728` — Python native handles
- `computer_20251124` — no desktop automation workload
- `code_execution_20250825/20260120` — backtests run via direct
  Python (local compute sandboxed via .venv)
- `tool_search_tool_*` + `defer_loading` — only 7 tools, below
  ~10-tool threshold
- `allowed_callers` / PTC — no sandbox-routed multi-tool workflow
- Web-search/fetch via Messages API — harness correctly uses
  Claude Code's built-in tools (different product)

**Tool loop mechanics — correct:**

- `stop_reason == "tool_use"` → execute → re-send: correct
- `ThreadPoolExecutor` parallel tool execution at L968-983
- `tool_choice` not set (implicit `"auto"`, correct)
- Hand-rolled loop vs `client.beta.messages.tool_runner()`: correct
  but more maintenance
- `betas=[]` absent — required if advisor/computer-use ever
  adopted; no impact today (overlaps MF-37 / cluster A4)

## Success criteria (from contract)

1. every_doc_pattern_status_evidenced — PASS (25+ patterns)
2. qa_runs_live_code_checks_not_review — PARTIAL (researcher ran
   live greps; Q/A verifies next)
3. deviations_cite_doc_page — PASS

## Artifact

- `docs/audits/compliance-tool-use.md`

## Novel findings

- No novel MF this cycle — findings align with existing MF-5,
  MF-23, MF-37, and cluster A1. This is a **confirmation audit**
  — live-verified zero usage of the 13 versioned primitives.
