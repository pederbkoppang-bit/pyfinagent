# Skills + Prompt Engineering + Context Mgmt Deep Audit (phase-4.11.3)

Auditor: Opus 4.7 (1M). Scope: Anthropic docs on Agent Skills, prompt engineering, prompt caching, compaction, context editing, context windows, token counting, embeddings, effort, search-results, Claude Code skills, create-skill API; compared against `backend/agents/skills/*.md` (28 files), `backend/agents/llm_client.py`, `backend/config/prompts.py`, and `backend/agents/memory.py`.

## URL coverage

All twelve required pages fetched in full (no abstract-only reads):

1. `agents-and-tools/agent-skills/overview` — 3-level progressive disclosure, YAML frontmatter, security, platform availability
2. `build-with-claude/prompt-engineering/overview` — entry point; canonical content now lives in `claude-prompting-best-practices`
3. `build-with-claude/prompt-engineering/claude-prompting-best-practices` — single reference for latest models (Opus 4.7, Sonnet 4.6, Haiku 4.5)
4. `build-with-claude/prompt-caching` — min tokens, TTLs, hierarchy, invalidation
5. `build-with-claude/compaction` — `compact_20260112`, `pause_after_compaction`
6. `build-with-claude/context-editing` — tool-result and thinking-block clearing
7. `build-with-claude/context-windows` — 1M on Opus 4.7/4.6 + Sonnet 4.6, context awareness (4.5+)
8. `build-with-claude/token-counting` — estimate endpoint, separate rate limits
9. `build-with-claude/embeddings` — Anthropic has no embedding model; recommends Voyage
10. `build-with-claude/effort` — `low/medium/high/xhigh/max`, replaces `budget_tokens` on Opus 4.6+
11. `build-with-claude/search-results` — citation-capable content blocks for RAG
12. `code.claude.com/docs/en/skills` — Claude Code filesystem skills, `allowed-tools`, `context: fork`
13. `api/skills/create-skill` — `POST /v1/skills`, beta `skills-2025-10-02`
14. `agent-skills/best-practices` — authoring rules (concise, 500 lines, third-person descriptions)

## Per-page digests

**Agent Skills overview.** Skills are filesystem dirs with `SKILL.md` + optional scripts/resources. Progressive disclosure: L1 metadata (name+description, ~100 tok) pre-loads in system prompt; L2 body (<5k tok) loads on match; L3 resources read via bash on demand, scripts executed (only stdout enters context). `description ≤1024 chars`, `name ≤64 chars` lowercase/hyphens. Surfaces: API (betas `skills-2025-10-02`, `code-execution-2025-08-25`, `files-api-2025-04-14`), Claude Code (filesystem), claude.ai (zip).

**Prompt engineering best practices.** Opus 4.7 self-calibrates length; over-prompting verbosity is top 4.7 migration issue. XML tags remain canonical structural primitive. Multishot examples, CoT, prefill, chain-prompts still recommended. Context-awareness on 4.5/4.6 exposes `<budget:token_budget>` + `<system_warning>` — tell model to budget rather than panic-truncate. For long-horizon, use "persist until done" language + checklists.

**Prompt caching.** Min cacheable prefix: **Opus 4.7/4.6/4.5/Haiku 4.5 = 4,096 tok**; Sonnet 4.6 = 2,048; Sonnet 4.5/4+Opus 4.1/4 = 1,024. Max 4 `cache_control` breakpoints. Order: tools → system → messages; any change invalidates that level + below. Writes 1.25× (5-min) or 2× (`ttl:"1h"`); reads 0.1×. Lookback 20 blocks. Thinking blocks cached indirectly.

**Compaction.** Beta `compact-2026-01-12` on Opus 4.7/4.6 + Sonnet 4.6. Trigger default 150k, min 50k. Emits `compaction` block that must be echoed back. Add `cache_control` to system prompt to survive compactions. `pause_after_compaction:true` lets caller inject preserved recent messages.

**Context editing.** `tool_result_clear` + `thinking_block_clear`. Use when bottleneck is tool-noise vs pure message growth.

**Context windows.** 1M on Opus 4.7/4.6 + Sonnet 4.6. Prior-turn thinking auto-stripped. Context awareness on 4.5+.

**Token counting.** Free endpoint, separate rate limit, no cache logic.

**Embeddings.** Anthropic ships no embedding model; recommends Voyage (`voyage-3-large`, `voyage-finance-2`, `voyage-code-3`). `input_type="query"` vs `"document"` matters. Int8/binary quantization available.

**Effort.** On Opus 4.7, effort **replaces** `budget_tokens` (unsupported). Values `low|medium|high|xhigh|max`; `xhigh` recommended for coding/agentic. Controls tool-call verbosity and count, not just thinking depth.

**Search-results.** Content block `{title, url, content}` for RAG with web-search-grade citations. Available Opus 4.7 + back. Direct fit for "BigQuery row as citable snippet".

**Claude Code skills.** Filesystem under `~/.claude/skills/` or `.claude/skills/`. Extended frontmatter: `disable-model-invocation`, `user-invocable`, `allowed-tools`, `model`, `effort`, `context:fork`, `agent`, `hooks`, `paths`, `argument-hint`. `$ARGUMENTS`/`$N`/`${CLAUDE_SKILL_DIR}` substitutions. Content enters conversation once on invoke; auto-compaction keeps first 5k tok (25k shared budget).

**Create-skill API.** `POST /v1/skills` with `anthropic-beta: skills-2025-10-02`. Returns `{id, type, source, display_title, latest_version, ...}`. Files uploaded via separate version endpoints.

**Authoring best practices.** <500 lines per SKILL.md. Third-person descriptions ("Processes X" not "I help you"). Gerund names. References one level deep. Match freedom to fragility. Evaluation-driven iteration (Claude A writes, Claude B tests).

## Our 28 skills vs native Skills: adoption analysis

Our current pattern (`backend/agents/skills/*.md` with `## Prompt Template` section injected via `{{var}}` placeholders from `backend/config/prompts.py:24-63`) is **not** a native Agent Skill. It is a custom prompt-template registry. Key divergences:

| Dimension | Our files | Native Skills |
|---|---|---|
| Discovery | Hardcoded `load_skill(name)` per pipeline step | Description-matched by Claude at runtime |
| Loading | Always loaded for every call to that agent | Level 1 metadata always; Level 2 body on match |
| Frontmatter | Absent (freeform `## Goal / ## Identity / ...`) | YAML `name` + `description` required |
| Invocation surface | Python-side Gemini/Anthropic calls | Claude Code CLI, Claude API skill_id, claude.ai |
| Scripts | None — pure prompt | `scripts/*.py` executable via bash |
| Cross-product reuse | Impossible | Portable across Claude Code + API + claude.ai |

**Verdict: do NOT migrate wholesale.** These are deterministic-slot prompts invoked by our orchestrator in a fixed 15-step pipeline; they don't benefit from Claude choosing when to load them, we already know. Native Skills shine when the *agent* picks capabilities — our harness does that itself. Productive borrows: (a) authoring discipline (third-person, consistent terms, one-level refs, <500-line bodies — we already comply on length); (b) put our 4 Claude MAS agents (`planner_agent.py`, `evaluator_agent.py`, harness verifier/qa) into Claude Code skills so `/planner`, `/evaluator`, `/harness-verifier`, `/qa-evaluator` become first-class, matching `.claude/agents/per-step-protocol.md`.

## Prompt engineering gaps (per our skill prompts)

Sampled `SKILL_TEMPLATE.md`, `bull_agent.md`, `quant_model_agent.md` (patterns recur across all 28):

1. **No XML tags.** Zero occurrences of `<instructions>`, `<example>`, `<context>` across 28 skill files (grep returned 0). Anthropic's best-practices page explicitly calls XML tags the primary structural primitive for long, multi-section prompts. Our skills rely on `--- SECTION ---` ASCII separators. Gemini tolerates this, Claude works better with XML. Since Layer 2 MAS runs Claude, this costs accuracy there.
2. **No prefill.** We never pre-seed the assistant turn (e.g. `{"role":"assistant","content":"{\n  \"thesis\":"}`). For JSON schema outputs this is the single highest-leverage technique to suppress preamble.
3. **No multishot examples.** Output format is described, not demonstrated. Bull/Bear/etc. would benefit from 1-2 realistic JSON examples inlined (the best-practices guide lists this as first-order technique).
4. **No chain-of-thought scaffold.** Several agents have list-form "Skills & Techniques" but no explicit `<thinking>...</thinking>` prompt. With adaptive thinking on Opus 4.7 this is auto-handled when we set `effort`; but for Gemini agents we're leaving reasoning quality on the table.
5. **First-person voice.** SKILL_TEMPLATE.md line 17: "What You CANNOT Modify". Descriptions read from the agent's POV ("You are the Bull Agent..."). Native Skills require third-person descriptions; our SKILL_TEMPLATE.md should flip to "Builds aggressive long thesis for {{ticker}}..." in frontmatter-equivalent position.
6. **No frontmatter.** Adding a YAML block with `name:` + `description:` (third-person, ≤1024 chars) costs nothing in the Gemini path and would make these files portable to both Claude Code (via `/bull-agent`) and the API create-skill endpoint if we ever want runbooks to run as skills.
7. **`{{var}}` syntax** is fine and consistent; no change needed. Variable substitution is the same semantic as `$ARGUMENTS`.

## Context management opportunities

Audited `backend/agents/llm_client.py:536-670` and `backend/agents/memory.py`.

- **Prompt caching wired but dormant** (`llm_client.py:602-609`): `cache_control:{type:ephemeral}` is applied to the system block, but the system block is typically only `"You are a financial analysis AI."` + optional JSON schema. Opus 4.7's 4,096-tok minimum means most of our calls **never cache**. Fix: have `format_skill()` split the rendered prompt into `{static_skill_body+schema+FACT_LEDGER}` cached in `system` and `{dynamic_vars}` in `user`. 28 skill bodies average ~450 tok each; must bundle with FACT_LEDGER to clear the floor.
- **1h cache TTL** (`ttl:"1h"`) fits harness cycle time better than default 5 min. Grid searches over the same prompt often expire mid-grid.
- **Compaction** relevant to MAS tool loops (`multi_agent_orchestrator.py`) and harness cycles past 150k input. Not relevant to Layer 1 (single-shot, <30k). Enable on harness Claude loop; add `cache_control` to the compaction block.
- **Context editing `tool_result_clear`** directly applies to Bash-heavy harness cycles accumulating grep/read results.
- **Context awareness** auto-on for Sonnet 4.6 — we're getting this already in sub-agents. Log `system_warning` to verify.
- **Effort.** `llm_client.py:622-628` still uses `thinking:{type:enabled,budget_tokens:N}`. **This is unsupported on Opus 4.7** per the effort doc: "Manual extended thinking is no longer supported on Opus 4.7; use adaptive thinking with effort instead." Critic 8192 / Synthesis 4096 budgets in `.claude/rules/backend-agents.md` should map to `output_config:{effort:"xhigh"}` on Opus 4.7, `"high"` on Sonnet 4.6 critic, `"medium"` on enrichment.
- **Embeddings.** `text-embedding-005` (Vertex) in `nlp_sentiment.py:15` and BM25 in `memory.py:18` are legacy. Anthropic recommends **`voyage-finance-2`** for financial RAG — direct upgrade path for Step 3 RAG + FinancialSituationMemory. Needs cost comparison vs existing Vertex.
- **Search-results primitive** fits synthesis/debate step: each alt-data BigQuery row as `{type:"search_result", title, url, content}` gives web-search-grade citations. Today we stringify JSON into user message and lose citation fidelity.

## Findings

1. Our "skills" are prompt templates, not Agent Skills, and that's correct for Layer 1/2. Do not migrate the 28 agents. Do consider wrapping our 4 MAS Claude sub-agents as Claude Code skills for dev ergonomics.
2. Prompt caching is nominally on but effectively **dormant**: the cached block is too small (<4,096 tok for Opus 4.7) to ever hit the minimum. Concrete fix: cache the skill body + schema, not just the stub system string.
3. Our skill prompts are missing three zero-cost prompt-engineering wins: XML tags, assistant prefill for JSON, and 1-shot examples in the output-format section.
4. `thinking: {type: enabled, budget_tokens}` in `llm_client.py` is **unsupported on Opus 4.7** — silently broken on our most-used model. Replace with `effort`.
5. Compaction and `tool_result_clear` belong on the harness loop (`scripts/harness/run_harness.py`), not on the pipeline.
6. `voyage-finance-2` is the Anthropic-recommended embedding path for finance RAG; Vertex `text-embedding-005` + BM25 is legacy.
7. Search-results content block would give us first-class citations in the synthesis agent for alt-data rows.

## MUST FIX

- **Replace `thinking.budget_tokens` with `effort`** in `backend/agents/llm_client.py:622-628` for Opus 4.7/4.6/Sonnet 4.6 paths. This is a silent breakage, not a nice-to-have.
- **Move skill body into system block + `cache_control`** in `backend/config/prompts.py` so the cached prefix clears Opus 4.7's 4,096-tok minimum. Add `ttl: "1h"` to match harness cycle time.
- **Update SKILL_TEMPLATE.md** to emit YAML frontmatter (`name:`, `description:` in third person, ≤1024 chars). Even if we never ship to claude.ai, this enforces discipline.
- **Remove dead Level-2 prompt-caching phase-4.10 claim** if the above isn't done — the audit currently says caching works; on Opus 4.7 it doesn't.

## NICE TO HAVE

- Add XML tag scaffolding to the 4 Claude MAS agent skills (`bull_agent.md`, `bear_agent.md`, `devils_advocate_agent.md`, `moderator_agent.md`) — highest ROI because those run on Claude.
- Add assistant prefill for JSON-schema agents (`bull`, `bear`, `critic`, `risk_judge`) — 1 line change in the client wrapper.
- Pilot `voyage-finance-2` for Step 3 RAG and the FinancialSituationMemory, benchmarked vs current Vertex + BM25.
- Enable `compact_20260112` on the harness Claude loop (Opus 4.7) with 150k trigger; bake `cache_control` onto the compaction block.
- Migrate Step 3 RAG row passing to `search_result` content blocks for citation fidelity.
- Wrap the 4 MAS sub-agents (planner, evaluator, qa-evaluator, harness-verifier) as Claude Code skills in `.claude/skills/` so they become `/planner` etc. — matches `.claude/agents/per-step-protocol.md`.

## References

- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/llm_client.py:536-670`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/config/prompts.py:24-63, 106-265`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/memory.py:1-84`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/tools/nlp_sentiment.py:1-81`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/skills/SKILL_TEMPLATE.md`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/skills/bull_agent.md`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/skills/quant_model_agent.md`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/cost_tracker.py:87-220`
- `.claude/rules/backend-agents.md` (quoted in-session)
- Anthropic docs cited verbatim above (12 URLs)
