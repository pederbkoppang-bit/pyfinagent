# Claude Docs Alignment -- Consolidated v4 (phase-4.13)

Produced: 2026-04-18. Scope: audit-only. Extends v3 with findings
from phase-4.13 (Messages-sidebar final sweep: 20 URLs, 17 valid
pages). This is a **diff**, not a rewrite -- all v2/v3 items still
apply unless explicitly refined below.

Source: `handoff/audit/phase-4.13/messages_sidebar_sweep.md`.

## Coverage update

- v3 running total: ~152 pages.
- v4 adds: **17 valid pages** (3 phantom 404s confirmed: `/build-
  with-claude/claude-api-skill`, `/build-with-claude/streaming-
  refusals`, `/agents-and-tools/agent-skills/skills-in-the-api` --
  content absorbed into adjacent pages; no findings lost).
- **Running total: ~169 pages** verified read-in-full across
  phase-4.10 + 4.11 + 4.12 + 4.13.

## Most urgent finding of the whole audit series

**`backend/agents/llm_client.py:622-628` silently 400s on Opus 4.7.**
The effort doc is unambiguous: manual `thinking:{type:"enabled",
budget_tokens:N}` is **no longer supported** on Opus 4.7. If ANY
production call currently targets `claude-opus-4-7` with
`ENABLE_THINKING=true`, it's failing. This is the single
highest-priority fix in the entire audit series (MF-29 below).

## Promotions from NICE-TO-HAVE to MUST-FIX

Three v2/v3 nice-to-haves reclassified based on the Messages-sidebar
sweep:

1. **Effort parameter adoption** (v2 cluster B / v3 refinement)
   -> **MF-28 + MF-29**. Blocks Opus 4.7 usage.
2. **Files API for SEC filings** (v2 cluster C1)
   -> **MF-33**. 500 MB cap + free upload + no-regret `file_id`
   reuse across pipeline makes current "re-encode filing PDF on
   every synthesis call" pattern inexcusable.
3. **Citations on SEC/earnings documents** (v2 cluster C1)
   -> **MF-31**. `cited_text` is free (not counted as input OR
   output tokens); cost + auditability win too large to defer.

## Withdrawal

**"Prefill for JSON coercion" (v3 NICE-TO-HAVE)** -- WITHDRAWN.
The `working-with-messages` page explicitly rejects prefill on
Opus 4.7 / 4.6 / Sonnet 4.6 (all current models). Correct
replacement is `output_config.format` structured outputs (already
MF-21 in v3).

## Additions to MUST FIX (9 new items: MF-26 to MF-34)

Continuing numbering from v3's item 25.

**MF-26: Full `stop_reason` dispatch in `ClaudeClient` +
`assistant_handler`.** Today we handle only `tool_use`
(`multi_agent_orchestrator.py:962`). Seven values exist:
`end_turn`, `max_tokens`, `stop_sequence`, `tool_use`,
`pause_turn`, `refusal`, `model_context_window_exceeded`. Grep
on `backend/slack_bot/assistant_handler.py` returns zero hits on
`stop_reason`, `refusal`, `pause_turn`. If the Slack bot hits a
refusal, user sees empty message. Add explicit log + UX behavior
per value; surface `refusal` with "Claude declined this request"
+ suggest Haiku 4.5 retry (different safety filters per doc tip).
Effort: M.

**MF-27: Auto-retry on `max_tokens` with incomplete `tool_use`
tail.** Per stop-reasons doc, truncated `tool_use` blocks are
unusable. Retry with 2× `max_tokens` on detection. Cheap;
prevents silent data loss. Effort: S.

**MF-28: Add `output_config.effort` pass-through to
`ClaudeClient`.** Default: `high` for synthesis/debate, `medium`
for enrichment, `low` for bias/conflict. Opus 4.7 guidance from
the doc: start at `xhigh` for coding/agentic; Sonnet 4.6 default
should be `medium`, NOT `high` (avoids unexpected latency).
Effort: S.

**MF-29: Gate `thinking:{type:"enabled",budget_tokens:N}` on
Opus 4.5 and older.** `llm_client.py:622-628` will 400 on Opus
4.7. Opus 4.7 path = adaptive + effort (MF-28). Effort: S.
**HIGHEST PRIORITY.**

**MF-30: Error-fast when citations + structured outputs both
set.** API returns 400; we should reject client-side with a
clearer message. Effort: XS.

**MF-31: Wrap SEC filings + earnings transcripts as `document`
blocks with `citations.enabled=true`.** `backend/tools/
sec_insider.py` and `earnings_tone.py` currently pass filings as
plain text. Wrapping gives per-sentence citations in synthesis
output + `cited_text` is free (no input/output token cost).
Effort: M.

**MF-32: Wrap BigQuery RAG rows as `search_result` content
blocks.** When `backend/agents/orchestrator.py` Step 3 RAG
fetches macro events, insider tx, price context, wrap each row
`{type:"search_result", source:"sunny-might-477607-p8://{ds}/
{table}/{id}", title, content:[{type:"text",text:...}]}`. Closes
"where did the model get this number?" hole with web-search-
quality citations. Supported on all current models. Effort: M.

**MF-33: Files API for large (>32 MB) SEC filings with `file_id`
reuse.** Beta header `files-api-2025-04-14`. 500 MB per file,
500 GB per org, free uploads/downloads/list/delete. Reusable
across enrichment -> synthesis -> citations without re-uploading.
Not ZDR-eligible; gate on retention policy. Effort: M.

**MF-34: PDF-native ingestion with `cache_control:"ephemeral"`
for earnings decks and 10-Ks.** `earnings_tone.py` currently
converts PDF -> text, losing charts + tables. 32 MB / 600 pages
cap; 1.5-3k tokens/page text + image. Pair with caching for
repeated queries on same PDF; pair with Batches for high-volume.
For filings >32 MB combine with MF-33. Effort: M.

## Additions to NICE TO HAVE (4 new items: NTH-10 to NTH-13)

**NTH-10: `task_budget` per-cycle harness spend cap (Opus 4.7,
beta).** Beta header `task-budgets-2026-03-13`.
`output_config.task_budget = {type:"tokens", total:N}`. Advisory
(not enforced) but cleaner than our `consecutive_fails` counter.
Minimum 20k total; don't mutate `remaining` (invalidates cache).
Wait for GA before harness depends on it.

**NTH-11: Stream Opus 4.7 synthesis into Slack** via SDK
`client.messages.stream(...).text_stream` (don't hand-roll SSE).
Long Opus 4.7 syntheses (30s+) feel responsive. `stop_reason`
arrives in `message_delta`, NOT `message_stop` -- don't
mis-parse.

**NTH-12: Vision on embedded filing figures.** 100 images/req
on 200k-ctx, 600/req on 1M. 8000x8000 px max. Chart OCR on SEC
filings. Low ROI today; defer.

**NTH-13: `pptx` Skill for "export analysis as deck".** Pre-
built Anthropic skill; invoked via `container.skills=[{type:
"anthropic", skill_id:"pptx"}]` + `code_execution_20250825`.
Not ZDR-eligible. Gate behind retention constraints.

## ZDR eligibility reconciliation

v4 confirms a feature-level ZDR matrix that intersects our
roadmap. If Peder ever asks for ZDR compliance, the following
MUST-FIX/NICE-TO-HAVE items **become unavailable**:

- **Not ZDR-eligible**: Batch processing (v2 A2), Files API
  (MF-33), Agent Skills (NTH-13), MCP connector, Code execution.
- **ZDR-eligible**: Adaptive thinking, effort, prompt caching
  (5m + 1h), citations (MF-31), structured outputs (qualified:
  schemas cached 24h), PDF support (MF-34), search results
  (MF-32), web search + web fetch (unless `dynamic_filtering`
  on).

Recommendation: **flag ZDR status BEFORE adopting** Files API
or Batches or Skills. Document in ARCHITECTURE.md.

## Refined understanding of the three phantom URLs

- `/build-with-claude/claude-api-skill` -- does not exist under
  platform.claude.com; best match is the Agent Skills API. No
  content lost.
- `/build-with-claude/streaming-refusals` -- refusal streaming is
  folded into `streaming` + `handling-stop-reasons` pages. During
  a stream, `message_delta` emits `stop_reason:"refusal"`; no
  special client logic needed beyond MF-26's dispatch table.
- `/agents-and-tools/agent-skills/skills-in-the-api` -- content
  merged into skills quickstart + `skills-guide`. The quickstart
  already covers `container.skills` / `skill_id` invocation from
  Messages API.

## Updated MUST-FIX grand total

v2 (items 1-20) + v3 (items 21-25) + v4 (items 26-34) = **34
MUST-FIX items**.

With promotions, the effective priority tiers are:

### TIER 1 -- HIGHEST PRIORITY (3 items; blocks Opus 4.7 + cost
correctness)

- **MF-29**: Gate `thinking:{type:"enabled",budget_tokens}` on
  Opus 4.5 and older -- Opus 4.7 silently 400s today
- **v3 MF-1**: Fix `MODEL_PRICING` -- Opus calls under-reported
  ~50-187x
- **v3 MF-2**: MCP Alpaca write-tool deny + resolve allowlist
  contradiction

### TIER 2 -- GO-LIVE BLOCKERS (6 items)

- **MF-28**: Add `output_config.effort` pass-through
- **MF-26 + MF-27**: stop_reason dispatch + max_tokens retry
- **v3 MF-21**: Migrate Claude JSON to `output_config.format`
- **v3 MF-3 + MF-4**: Permission mode from bypass -> acceptEdits/
  auto + enable sandboxing
- **v3 MF-5 + MF-6**: Opus 4.7 sampling-param + retired-models
  removal

### TIER 3 -- CORRECTNESS (8 items)

- MF-30 (citations x structured outputs guard)
- v3 MF-7 (autonomous_loop stale snapshot), MF-9
  (retry-after), MF-10 (exception classes), MF-11 (request_id),
  MF-12 (Gemini-only thinking gate), MF-13 (cache threshold miss)
- v4 MF-31 (citations)

### TIER 4 -- HARDENING / DATA INTEGRITY (9 items)

- MF-32 (search_result blocks), MF-33 (Files API), MF-34
  (PDF native)
- v3 MF-14 (BigQuery MCP doc drift), MF-15 (prune stubs),
  MF-16 (subagent descriptions), MF-17 (claude.yml pins),
  MF-18 (SDK bump), MF-19 (cron_budget.yaml)

### TIER 5 -- HOUSEKEEPING (8 items)

- MF-20 (Haiku 3 CI assert), v3 MF-22 (latency instrumentation),
  MF-23 (harmlessness pre-screen), MF-24 (prompt-leak defences),
  MF-25 ("I don't know" permission), plus v3 items 21-23
  (PreToolUse hooks, ConfigChange hooks, bak file cleanup).

## Decisions now required (revised from v3's 8 to 11)

1. Approve the 34 MUST-FIX items (or prune / reprioritize)
2. Pick 3-5 NICE-TO-HAVE clusters for phase-4.14+ scope
3. Routines-vs-cron for phase-10.7 (unchanged from v3)
4. Agent SDK: port planner + evaluator only, or defer?
5. Managed Agents pilot: submit access form with v3 M2 narrow
   scope? (unchanged from v3)
6. Structured Outputs migration blocking go-live? (v3 #6 --
   reaffirmed)
7. Rename `skills/` -> `prompt_templates/`? (v3 #7)
8. Eval suite (Cluster K) phasing (v3 #8)
9. **(NEW) ZDR compliance target -- declare yes/no BEFORE
   committing to Files API / Batches / Skills adoption?**
10. **(NEW) Do we accept the cost/auditability tradeoff of using
    PDF-native ingestion (MF-34) -- paired with the ZDR
    decision?**
11. **(NEW) Is MF-29 (Opus 4.7 thinking-API gate) a same-day
    hotfix or can it wait for the phase-4.14 batch?**

## Coverage audit -- final

All 169 pages verified read-in-full across four audit phases.
Structure of audit records:

- `handoff/audit/phase-4.10/` -- 6 topic audits (extended
  thinking, adaptive thinking, sub-agents, agent teams, MCP,
  platform overview) + CONSOLIDATED_REPORT.md.
- `handoff/audit/phase-4.11/` -- 8 deep audits (Agent SDK 29,
  Managed Agents 9, Tool primitives 14, Skills/prompting/context
  12, Claude Code core 17, Claude Code surfaces 33, API + SDKs
  23, Models/pricing/admin 34) + CONSOLIDATED_REPORT_v2.md.
- `handoff/audit/phase-4.12/` -- 3 gap audits (Evals + guardrails
  7, Prompting/Skills tail 7, Managed Agents sub-pages 8) +
  CONSOLIDATED_REPORT_v3.md.
- `handoff/audit/phase-4.13/` -- 1 final sweep (Messages sidebar
  17 valid + 3 phantom) + this v4 report.

Grand total: 14 topic audits + 4 consolidated reports. Every
sidebar entry from the screenshots supplied is mapped to at least
one audit file.

All implementation still blocked pending approval per audit-only
mandate.

## References

- v3 report (latest consolidated): `handoff/audit/phase-4.12/
  CONSOLIDATED_REPORT_v3.md`
- v2 report: `handoff/audit/phase-4.11/CONSOLIDATED_REPORT_v2.md`
- v1 report: `handoff/audit/phase-4.10/CONSOLIDATED_REPORT.md`
- Phase-4.13 detail: `handoff/audit/phase-4.13/
  messages_sidebar_sweep.md`

Documentation accessed 2026-04-18 across platform.claude.com,
code.claude.com, and docs.anthropic.com.
