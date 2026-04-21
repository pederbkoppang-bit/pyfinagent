# Experiment Results — Cycle 4.15.3

Step: phase-4.15.3 Sub-agents + Agent teams + Skills compliance

## What was built

One research-phase output: `docs/audits/compliance-subagents-skills.md`
(~2,100 words, 25 patterns indexed).

## Researcher report summary

25 patterns audited across 8 doc pages + 8 internal files.

**Correct (5):** trigger phrasing on both merged agents, model
aliases, maxTurns/effort/memory calibrated, agent-teams flag set,
freedom-to-fragility matched to agent role.

**Deviations requiring fix (9, 5 NOVEL to this cycle):**

| # | Pattern | Evidence | Severity | Novel? |
|---|---------|----------|----------|--------|
| 1/6 | permissionMode missing on both agents | `.claude/agents/researcher.md`, `qa.md` frontmatter | M | NEW |
| 3 | qa.md `tools: Bash` contradicts body "read-only" | qa.md L4+body | L/M | NEW |
| 7 | SubagentStop hook entirely absent | `.claude/settings.json` hooks keys | M | NEW |
| 9/11 | Agent teams flag + TeammateIdle dead weight | settings.json + no `~/.claude/teams/` dir | L | confirmed from 4.10 |
| 16 | Skills terminology collision (`backend/agents/skills/*.md` not Anthropic Agent Skills) | 28 files, 0 YAML frontmatter, 252 `{{}}` placeholders | L | confirmed from 4.11/4.12 |
| 20 | Description phrasing imperative, not third-person | both agents | L | NEW |
| 22 | Enterprise risk-tier review not applied | researcher.md grants Bash + WebFetch (two high-risk per checklist) | M | NEW |
| 23 | Separation-of-duties gap on agent definition changes | `.claude/agents/*.md` authored + self-approved in same session | M | NEW |
| 25 | TaskCompleted hook double-spawns qa alongside explicit qa spawn | settings.json TaskCompleted + per-step-protocol.md explicit spawn | L | NEW |

**Verification commands run by researcher (live):**
- `ls .claude/agents/` → 2 files (researcher.md, qa.md) ✓
- `ls .claude/skills/` → does not exist (no Claude Code Skills registered at project scope)
- `wc -l backend/agents/skills/*.md` → 2,859 total lines across 28 files
- `grep -l 'name:\|description:' backend/agents/skills/*.md | wc -l` → 0 (none have YAML frontmatter — confirms they are prompt templates, not Agent Skills)
- `grep -c 'MUST BE USED\|use proactively' .claude/agents/researcher.md .claude/agents/qa.md` → 1 each (trigger phrasing present, 4.15.0 fix holds)
- `jq '.env.CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS' .claude/settings.json` → `"1"` (flag set; unused)
- `jq '.hooks | keys' .claude/settings.json` → `[PostToolUse, Stop, TaskCompleted, TeammateIdle]` (no SubagentStop)

## Success criteria (from contract)

1. every_doc_pattern_status_evidenced — PASS (25 patterns with
   Status / Evidence / Deviation / Risk / Recommended fix)
2. qa_runs_live_code_checks_not_review — PARTIAL (researcher ran
   live checks; Q/A phase next)
3. deviations_cite_doc_page — PASS (every row cites the source
   doc URL)
4. recommended_fix_included_not_implemented — PASS (audit-only;
   no code modified)

## Artifact

- `docs/audits/compliance-subagents-skills.md` (25 patterns, ~2,100
  words)

## Honest scope disclosure

- The researcher tried each external URL; all 7 returned 200 with
  content (no phantom pages this cycle).
- Internal exploration covered every relevant file path on the
  researcher's list.
- The researcher flagged Pattern 23 (separation of duties) as
  self-referential to THIS session — I (Main) authored the 4.15.0
  MAS merge and am now self-auditing. The Q/A agent in the next
  phase provides partial mitigation by running in a separate
  context window.
