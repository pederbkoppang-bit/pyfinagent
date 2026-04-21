# Sub-agents Audit — Claude Doc Alignment (phase-4.10.2)

## Documentation summary

Authoritative page: https://code.claude.com/docs/en/sub-agents (accessed 2026-04-18).

**File format.** Sub-agents are Markdown files under `.claude/agents/` (project), `~/.claude/agents/` (user), or plugin `agents/` dirs, with YAML frontmatter + Markdown body as the system prompt. Priority order: managed settings (1) > `--agents` CLI flag (2) > project (3) > user (4) > plugin (5). Loaded at session start; file edits need a restart or `/agents`.

**Required frontmatter fields (quote):** "Only `name` and `description` are required."

**Full supported frontmatter keys** (from the docs table):

| Field | Notes |
|---|---|
| `name` | required. "Unique identifier using lowercase letters and hyphens." |
| `description` | required. "When Claude should delegate to this subagent." |
| `tools` | allowlist, comma-separated string (`Read, Grep, Glob`) or YAML list. Inherits all if omitted. |
| `disallowedTools` | denylist, applied before `tools`. |
| `model` | `sonnet` \| `opus` \| `haiku` \| full ID (e.g. `claude-opus-4-7`) \| `inherit`. Default `inherit`. |
| `permissionMode` | `default` \| `acceptEdits` \| `auto` \| `dontAsk` \| `bypassPermissions` \| `plan`. |
| `maxTurns` | integer cap on agentic turns. |
| `skills` | list of skills pre-loaded into context. |
| `mcpServers` | inline or referenced MCP server configs. |
| `hooks` | lifecycle hooks (PreToolUse, PostToolUse, Stop → SubagentStop). |
| `memory` | `user` \| `project` \| `local`; auto-enables Read/Write/Edit for memory dir. |
| `background` | boolean; run concurrently. |
| `effort` | `low` \| `medium` \| `high` \| `xhigh` \| `max` (model-dependent). |
| `isolation` | `worktree` for isolated repo copy. |
| `color` | `red`\|`blue`\|`green`\|`yellow`\|`purple`\|`orange`\|`pink`\|`cyan`. |
| `initialPrompt` | only used when agent runs as main via `--agent`. |

**Tool restriction syntax.** Either comma-separated inline (`tools: Read, Grep, Glob, Bash`) or YAML list. `Agent(worker, researcher)` is an allowlist that restricts which sub-agents a main-thread agent can spawn; subagents cannot spawn other subagents so this has no effect inside a subagent file. Omitting `tools` inherits everything including MCP tools.

**Delegation.** "Claude uses each subagent's description to decide when to delegate… include phrases like 'use proactively' in your subagent's description field." Explicit invocation via natural language, `@agent-<name>`, or `--agent <name>` for a whole session.

**Best-practice quote:** "Design focused subagents: each subagent should excel at one specific task"; "Write detailed descriptions… Claude uses the description to decide when to delegate"; "Limit tool access: grant only necessary permissions."

Linked sub-pages scanned in the same fetch: context-window, agent-teams, model-config, tools-reference, permissions, permission-modes, hooks, skills, plugins, settings, env-vars, cli-reference, headless, mcp. All referenced keys above are consistent with that fetch.

## Codebase audit

### Inventory of sub-agents

| Agent file | Name | Description (first 80 chars) | Tools | Model | Status |
|------------|------|-------------|-------|-------|--------|
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/agents/qa-evaluator.md` | `qa-evaluator` | "Independent QA evaluator for cross-verification of completed work…" | `Read, Bash, Glob, Grep` | `opus` | Drift (unknown keys + weak trigger) |
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/agents/harness-verifier.md` | `harness-verifier` | "Cross-verifies step completion by checking harness results and evaluator critiques…" | list form: Bash/Read/Glob/Grep | `sonnet` | Drift (unknown keys + weak trigger) |
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/agents/researcher.md` | `researcher` | "Research specialist for literature search, technical analysis, and evidence gathering…" | `Read, Grep, Glob, Bash, WebSearch, WebFetch` | `sonnet` | Drift (unknown keys) |
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/agents/per-step-protocol.md` | (none — no frontmatter) | Operator runbook, not an agent | — | — | Not an agent (correctly excluded by loader) |

### Detailed findings per agent

**qa-evaluator.md** — Frontmatter declares `maxTurns: 10` (valid), `effort: medium` (valid), `memory: project` (valid), `color: green` (valid). All four line up with the doc schema, good. Tools are least-privilege (read-only set, no Write/Edit), which matches the "NEVER modify files" constraint in the prompt body — compliant. Model `opus` is an allowed alias. `description` is informative but *not action-oriented for automatic delegation*: it describes what the agent is ("Independent QA evaluator…") rather than when to invoke ("Use proactively after…"). Docs explicitly recommend phrases like "use proactively" / "MUST BE USED" — missing here. Per-step-protocol does manually spawn it, so automatic delegation is not relied on, but the weak description reduces future reusability.

**harness-verifier.md** — Same shape as qa-evaluator with `model: sonnet`, `maxTurns: 10`, `effort`, `memory`, `color` — all valid. Tools given as YAML list (both inline CSV and list forms are accepted per docs). Tool scope is correctly read-only (Bash included, which is needed for syntax checks + dry-run). Description "Read-only — never modifies files." is factual but again passive; a trigger phrase like "Use proactively after any implementation step" would improve auto-delegation. The prompt body mandates *always* co-running with qa-evaluator — this is a human-orchestrated constraint, not enforceable via frontmatter. Acceptable.

**researcher.md** — Tools include `WebSearch, WebFetch` (needed — fine). `model: sonnet`, `memory: project`, `maxTurns: 15`, `effort: medium`, `color: cyan` — all valid. Description is weak for automatic delegation (describes the specialty, not the trigger). Body contains tier system (simple / moderate / complex) enforced by caller — not a frontmatter concern. Note: `memory: project` means Claude auto-enables Read/Write/Edit for the memory dir (per docs) — this *slightly expands* the declared read-only-ish tool list, but it's scoped to `.claude/agent-memory/researcher/` and is documented behavior.

**per-step-protocol.md** — No YAML frontmatter. This is correct — it is an orchestrator runbook, not a sub-agent. Claude Code's loader requires frontmatter to register an agent, so this file is ignored by `/agents` and correctly lives in the same dir only for convenience. Recommend renaming or relocating to avoid confusion (see "Nice to have").

### Unknown frontmatter keys

The docs' supported-fields table does **not** list `effort` at the level used here… actually, re-check: the docs do list `effort` as a valid field ("Options: `low`, `medium`, `high`, `xhigh`, `max`") — OK, compliant. `maxTurns`, `memory`, `color` are all documented. **No unknown keys remain** after re-reading the table. Revising earlier "unknown keys" flag downward — all three agents use only documented keys.

### Cross-references in code

Grep across the repo finds `qa-evaluator`, `harness-verifier`, `researcher` referenced from:
- `CLAUDE.md` (Critical Rules + Harness Protocol sections)
- `.claude/agents/per-step-protocol.md` (operator sequence)
- `scripts/harness/run_harness.py` lines 287–316, 1044–1066 (programmatic `spawn_researcher` shim; no qa-evaluator/harness-verifier direct invocation — those rely on the main-session orchestrator spawning them in parallel)
- `handoff/harness_log.md` and `handoff/archive/phase-*/` (log entries, evidence of use)

No dead agents. No duplicated responsibilities (qa-evaluator does semantic/LLM-level review; harness-verifier reproduces the deterministic command; researcher gathers external evidence).

## Findings

| Aspect | Status | Evidence | Notes |
|--------|--------|----------|-------|
| Required fields present (`name`, `description`) | Correct | all three .md files, lines 2–3 | — |
| Only documented frontmatter keys used | Correct | qa-evaluator:5-9, harness-verifier:9-13, researcher:5-9 | `effort`, `memory`, `maxTurns`, `color` are all documented. |
| Tool allowlist follows least-privilege | Correct | qa-evaluator:5 (read-only); harness-verifier:4-8 (read-only); researcher:5 (incl. WebSearch/WebFetch as needed) | No Write/Edit on read-only reviewers. |
| Description is action/trigger oriented (auto-delegation) | Weak | qa-evaluator:3, harness-verifier:3, researcher:3 | None contain "use PROACTIVELY" / "MUST BE USED" / "Use immediately after…". |
| Model override present & valid | Correct | `opus` / `sonnet` / `sonnet` | All valid aliases. |
| No duplicated responsibilities | Correct | see Cross-references | qa-evaluator vs harness-verifier are deliberately complementary (LLM vs deterministic). |
| Per-step-protocol.md excluded from agent loader | Correct | file has no frontmatter | Treated as docs, not as an agent. |
| Dual-evaluator rule enforceable via frontmatter | Not possible | CLAUDE.md + per-step-protocol.md §4 | Enforced by orchestrator discipline only; docs do not offer a "must-co-run" frontmatter primitive. |
| `memory: project` side-effect (Read/Write/Edit on memory dir) documented | Acknowledged | researcher.md, qa-evaluator.md, harness-verifier.md | Per docs, memory auto-enables Read/Write/Edit scoped to the memory dir. Not a least-privilege violation in practice. |
| Anthropic "stress-test doctrine" (prune scaffolding) | Not tracked | CLAUDE.md §Stress-test doctrine | Doctrine documented but no audit record shows re-run without harness on latest Opus 4.7 release. |

## Gaps & Opportunities

- **MUST FIX (low effort, high signal):** Strengthen `description` fields for auto-delegation. Add explicit trigger verbs and "use proactively" phrasing, e.g. qa-evaluator → "Use proactively after any GENERATE step, immediately before marking a masterplan step done." harness-verifier → "MUST BE USED in parallel with qa-evaluator after every implementation step." researcher → "Use proactively before PLAN, whenever new external sources or unresolved research claims are needed." Without this, future sessions that *don't* invoke via the runbook risk mis-delegation.
- **NICE TO HAVE:** Move `per-step-protocol.md` out of `.claude/agents/` (e.g. to `.claude/docs/per-step-protocol.md` or `.claude/runbooks/`). Today it sits among agent files and the loader silently ignores it; a human reader cannot tell at a glance which files are agents vs docs.
- **NICE TO HAVE:** Add a `hooks:` block on qa-evaluator and harness-verifier with a `Stop` (→ `SubagentStop`) hook that refuses to exit `ok: true` unless the sibling evaluator also ran in this cycle. Today the dual-evaluator rule relies entirely on orchestrator discipline (CLAUDE.md warns it has been violated in past cycles per auto-memory `feedback_harness_rigor.md`).
- **NICE TO HAVE:** Consider `permissionMode: plan` on qa-evaluator and harness-verifier to guarantee read-only exploration at the permission layer, belt-and-suspenders against the existing allowlist.
- **NICE TO HAVE:** Add a new `code-reviewer` or `simplify` subagent matching the pattern from the docs' example section — today the orchestrator handles diffs/review manually and the `simplify` skill runs in the main context.
- **NICE TO HAVE (doctrine compliance):** Record a dated "stress-test ran vs Opus 4.7" entry each time the model ships; CLAUDE.md mandates the re-run but there is no artifact in `handoff/` confirming it has happened since 4.7 became current on this session.
- **MUST FIX (minor):** `researcher.md:66` cites Anthropic's Multi-Agent Research System as "2024" — fine — but CLAUDE.md asks for a periodic re-read; add a last-reviewed date comment to each agent file so drift against upstream docs is visible.

## References

- Claude Code — Create custom subagents, https://code.claude.com/docs/en/sub-agents (accessed 2026-04-18)
- Claude Code — Agent teams, https://code.claude.com/docs/en/agent-teams (linked from above, accessed 2026-04-18)
- Claude Code — Hooks, https://code.claude.com/docs/en/hooks (linked, accessed 2026-04-18)
- Claude Code — Model config, https://code.claude.com/docs/en/model-config (linked, accessed 2026-04-18)
- Claude Code — Permission modes, https://code.claude.com/docs/en/permission-modes (linked, accessed 2026-04-18)
- Anthropic — "Harness Design for Long-Running Apps," https://www.anthropic.com/engineering/harness-design-long-running-apps (cited by every agent's body)
- Anthropic — "How We Built Our Multi-Agent Research System," https://www.anthropic.com/engineering/built-multi-agent-research-system (cited by researcher.md)
- Source files audited:
  - `/Users/ford/.openclaw/workspace/pyfinagent/.claude/agents/qa-evaluator.md`
  - `/Users/ford/.openclaw/workspace/pyfinagent/.claude/agents/harness-verifier.md`
  - `/Users/ford/.openclaw/workspace/pyfinagent/.claude/agents/researcher.md`
  - `/Users/ford/.openclaw/workspace/pyfinagent/.claude/agents/per-step-protocol.md`
  - `/Users/ford/.openclaw/workspace/pyfinagent/scripts/harness/run_harness.py` (lines 252–320, 1001–1066)
  - `/Users/ford/.openclaw/workspace/pyfinagent/CLAUDE.md` (Harness Protocol section)
