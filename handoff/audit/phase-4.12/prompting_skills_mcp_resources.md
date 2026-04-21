# Prompting / Skills / MCP / Resources Tail (phase-4.12.1)

Audit-only pass over 7 Claude-platform docs previously missed during phase-4.10/4.11. Scope: what we use vs. what the page says, with MUST-FIX / NICE-TO-HAVE fixes. No code changed.

## URL coverage

| # | URL | Status |
|---|-----|--------|
| 1a | platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompt-generator | CHECKED (200; same page as 1b — anchors) |
| 1b | .../prompt-engineering/prompt-improver | CHECKED (200; same page) |
| 2  | .../agents-and-tools/agent-skills/best-practices | CHECKED (200) |
| 3  | .../agents-and-tools/agent-skills/enterprise | CHECKED (200) |
| 4a | .../agents-and-tools/mcp-connector | CHECKED (200) |
| 4b | .../agents-and-tools/remote-mcp-servers | CHECKED (200; index page, sibling to 4a) |
| 5a | .../resources/glossary | CHECKED (200) |
| 5b | .../about-claude/glossary | CHECKED (200; redirects to same content) |
| 6  | .../about-claude/use-case-guides/overview | CHECKED (200) |
| 7a | .../release-notes/claude-platform | FAILED (404 via this exact slug) |
| 7b | .../release-notes/overview | CHECKED (200; renders Claude Platform release notes) |

All sibling paths tried; 7a returns a generic "Not Found" shell, 7b is canonical.

## Per-page digests

### 1. Console prompting tools (prompt-generator + prompt-improver)
- **Prompt generator** — Console drafts a prompt template from a task description, auto-inserts `{{double-bracket}}` variables, follows Anthropic's best practices. Colab notebook also available.
- **Prompt improver** — 4-step pipeline: example identification → initial draft with XML-tag sections → CoT refinement → example enhancement. Adds explicit CoT, XML organization, prefills. For complex/high-accuracy tasks; not recommended for latency-sensitive prompts. Supports Test-Case Generator.

### 2. Skills best practices (authoring)
- Only SKILL.md metadata is pre-loaded; body loads on trigger; reference files load on demand.
- "Default assumption: Claude is already very smart" — strip explanations it already knows.
- **Degrees of freedom** match instruction specificity to task fragility: high (text) for open, medium (pseudocode) for preferred-pattern, low (exact scripts "do not modify") for fragile (migrations).
- **Test across every model you plan to use** (Haiku / Sonnet / Opus). Opus-tuned instructions may under-specify for Haiku.
- **Frontmatter**: `name` ≤64 chars, lowercase/digits/hyphens, no `anthropic`/`claude`; `description` ≤1024 chars, **third person**, specific triggers.
- **Naming**: gerund form preferred (`processing-pdfs`); avoid `helper`/`utils`.
- **Progressive disclosure**: SKILL.md body < 500 lines; split into `REFERENCE.md`/`FORMS.md` **one level deep** (Claude only partially reads nested references via `head -100`).

### 3. Skills for enterprise
- **Risk tiers** per skill: code exec, instruction manipulation, MCP references, network access, hardcoded creds, FS scope, tool invocations.
- **Review checklist**: read all content, sandbox scripts, check adversarial instructions, audit network calls, scan creds, enumerate tool calls, verify redirects, exfiltration patterns. Treat untrusted skills like installing production software.
- **Approval gates**: triggering accuracy / isolation / coexistence / instruction-following / output quality. Require 3–5 query eval suites per skill (fire / no-fire / ambiguous).
- **Hard cap**: max 8 skills per API request.
- **Lifecycle**: plan → create+review → test → deploy (Skills API, workspace-scoped) → monitor → iterate/deprecate. Separation of duties: authors ≠ reviewers.
- **Cross-surface**: API-uploaded skills **not** available on claude.ai or Claude Code; Git is source of truth; BYO sync.

### 4. MCP connector + Remote MCP servers
- Current beta header `mcp-client-2025-11-20` (2025-04-04 deprecated). Not ZDR-eligible. Not on Bedrock/Vertex.
- HTTPS only (Streamable HTTP or SSE); **local STDIO servers cannot be connected via the connector** — use TypeScript `mcpTools`/`mcpMessages`/`mcpResourceToContent`/`mcpResourceToFile` helpers instead.
- `mcp_servers[]` holds `type:"url"`, `url`, `name`, optional `authorization_token`. `tools[]` needs one `mcp_toolset` per server; per-tool `enabled`/`defer_loading` with inheritance from `default_config`.
- Response: `mcp_tool_use` / `mcp_tool_result` blocks. OAuth handled by caller; inspector CLI provides quick-OAuth flow.
- Remote MCP index: third-party servers (Asana, Atlassian, CloudFlare, Intercom, Linear, PayPal, Sentry, Slack, Square, Workato, Zapier, etc.) — not endorsed by Anthropic, BYO auth.

### 5. Glossary
13 terms: context window, fine-tuning, HHH, latency, LLM, MCP, MCP connector, pretraining, RAG, RLHF, temperature, TTFT, tokens.

### 6. Use-case guides overview
Exactly 4 guides: **ticket routing, customer support agent, content moderation, legal summarization**. No RAG/search/agent-harness pages.

### 7. Release notes (Claude Platform) — deltas
- **2026-04-16**: Opus 4.7 GA at $5/$25 MTok; API breaking changes vs 4.6; new tokenizer.
- **2026-04-14**: Sonnet 4 / Opus 4 deprecated, retire 2026-06-15.
- **2026-04-09**: Advisor tool public beta (`advisor-tool-2026-03-01`) — fast executor + smart advisor mid-generation.
- **2026-04-08**: Claude Managed Agents public beta + `ant` CLI.
- **2026-03-30**: `max_tokens` 300k on Batches for 4.6; 1M-context beta retiring for Sonnet 4.5/4 on 2026-04-30.
- **2026-02-19**: Automatic cache-control (no manual breakpoints).
- **2026-02-17**: Code execution free with web search/fetch; web search + programmatic tool calling GA; memory/tool-search/tool-use-examples GA.
- **2026-01-12**: console.anthropic.com → platform.claude.com.
- **2026-02-05**: Sonnet 4.6 GA; compaction API beta; `inference_geo` data residency.

## pyfinagent audit

- `backend/agents/skills/*.md` — 28 files, 62–239 lines each, 2.8k total. **No YAML frontmatter.** Headers: `# Title`, `## Goal`, `## Identity`, `## What You CAN/CANNOT Modify`, `## Data Inputs`. Variables use `{{ticker}}`, `{{signals_json}}` — matches Anthropic `{{double-bracket}}`.
- Loaded by `orchestrator.py` via `load_skill()` / `format_skill()` — these are *prompt templates*, not Anthropic Agent Skills (no SKILL.md shape, no Skills API upload, no `name`/`description` frontmatter).
- `backend/config/prompts.py` — 699-line monolithic prompt module for Gemini Layer-1. No evidence of being run through Console prompt-improver.
- MCP surface: `backend/mcp/*.py`, `backend/agents/mcp_servers/*.py`, `backend/slack_bot/mcp_tools.py` — in-process FastMCP + local stdio (signals, backtest, data, risk, slack, alpaca). No HTTPS-exposed remote MCP; MCP connector on Messages API not used.
- Terminology: `CLAUDE.md` and `.claude/rules/` correctly use MCP / context window / RAG / tokens. Calling local prompt files "skills" collides with Anthropic's "Agent Skill" — mild reviewer-confusion risk.
- Release-notes delta: Opus 4.7 in CLAUDE.md already; not yet using automatic cache-control, advisor tool, managed agents, programmatic tool calling, tool-search.

## Findings

1. **Prompt-generator / improver never used.** The 4 MAS skills (bull/bear/devils_advocate/moderator) are hand-written. Prompt-improver explicitly targets complex-reasoning, accuracy-over-latency prompts — a perfect match. One-time Console pass would add CoT scaffolding + XML-tagged output sections without code change.
2. **Skill files don't conform to Anthropic Agent Skill shape** (no frontmatter, not SKILL.md). Fine because we're not using the Skills API, but:
   - Nomenclature collides with docs. Consider renaming to "prompt-templates" or annotating in CLAUDE.md.
   - If phase-5 ever uploads via Skills API, we need frontmatter + third-person descriptions + kebab names + enterprise security review.
3. **A few skills approach the 500-line ceiling.** `quant_strategy.md` (239) and `synthesis_agent.md` (179) are candidates for progressive-disclosure split (one-level `REFERENCE.md`).
4. **No cross-model testing record.** Best-practices mandates testing across Haiku/Sonnet/Opus. `skill_optimizer.py` evaluates only the currently-selected provider.
5. **MCP connector unused (correctly for now).** All MCP is local stdio / FastMCP in-process. Document as intentional before go-live review.
6. **Remote MCP servers index** lists Slack/Linear/Sentry/Zapier as hosted options — noting as future simplification, not a change.
7. **8-skills-per-request cap** is load-bearing if we ever upload. 28 files would require consolidation.
8. **Adoptable release-notes items**:
   - Automatic cache-control (2026-02-19) — drop manual breakpoints in `llm_client.py`.
   - Advisor tool (2026-04-09) — Haiku executor + Opus advisor on MAS Moderator = cost win.
   - Tool-search tool GA (2026-02-17) — relevant if we exceed the 8-tool ceiling.
   - `ant` CLI + Managed Agents (2026-04-08) — awareness only; we have our own harness.

## MUST FIX

(None — this was an audit pass; no regressions found.)

## NICE TO HAVE

- **N1** Run the 4 Claude MAS skills (`bull_agent.md`, `bear_agent.md`, `devils_advocate_agent.md`, `moderator_agent.md`) through Console prompt-improver; paste XML-tagged/CoT version and A/B via `skill_optimizer`. Expected lift on debate conviction + moderator consensus calibration.
- **N2** Rename the `skills/` directory to `prompt_templates/` (or clarify in `CLAUDE.md` that these are local prompt templates, not Anthropic Agent Skills) before any go-live review.
- **N3** Split `quant_strategy.md` (239 L) and `synthesis_agent.md` (179 L) into SKILL.md + one-level-deep reference files, per progressive-disclosure pattern.
- **N4** Add a cross-model evaluation matrix in `skill_optimizer.py` (Haiku / Sonnet / Opus rows) per Best-Practices "Test with all models" guidance.
- **N5** File a phase-5 ticket to pilot **Advisor Tool** on the MAS Moderator path (Opus advisor + Sonnet executor), targeting the cost/quality goal.
- **N6** File a phase-5 ticket to adopt **automatic cache-control** (`cache_control` without manual breakpoints) in `llm_client.py` — reduces maintenance and handles growing conversations.
- **N7** Document "MCP connector intentionally unused; all MCP servers local" in `.claude/rules/backend-agents.md` or `ARCHITECTURE.md` so go-live reviewers don't flag it.

## References

- Console prompting tools — https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompt-generator (anchors: prompt-generator, prompt-improver)
- Skill authoring best practices — https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices
- Skills for enterprise — https://platform.claude.com/docs/en/agents-and-tools/agent-skills/enterprise
- MCP connector — https://platform.claude.com/docs/en/agents-and-tools/mcp-connector
- Remote MCP servers — https://platform.claude.com/docs/en/agents-and-tools/remote-mcp-servers
- Glossary — https://platform.claude.com/docs/en/resources/glossary
- Use-case guides overview — https://platform.claude.com/docs/en/about-claude/use-case-guides/overview
- Release notes (Claude Platform) — https://platform.claude.com/docs/en/release-notes/overview  (canonical; `/release-notes/claude-platform` returned 404)
- Local files audited: `backend/agents/skills/*.md` (28 files), `backend/config/prompts.py`, `backend/mcp/*.py`, `backend/agents/mcp_servers/*.py`, `.claude/rules/backend-agents.md`, `CLAUDE.md`.
