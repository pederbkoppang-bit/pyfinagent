# Contract — Cycle 4.15.3 — Sub-agents + Agent teams + Skills compliance

Step: phase-4.15.3 Sub-agents + Agent teams + Skills
Ran at: 2026-04-18 (UTC)

## Research gate (3-agent MAS)

Spawning `researcher` (merged) to cover BOTH halves in one session:
- External: Claude Code sub-agents doc, agent teams doc, platform
  Agent Skills docs, Skills best practices, Skills for enterprise,
  Skills quickstart
- Internal: `.claude/agents/*.md` (the 2 remaining post-4.15.0),
  `backend/agents/skills/*.md` (28 files), `backend/agents/
  agent_definitions.py` (AGENT_CONFIGS), and the Claude Code
  session-level MAS wiring in `.claude/settings.json`.

## Hypothesis

Claim: after phase-4.15.0's MAS restructure, our sub-agent
configuration is doc-compliant on the schema layer but may still
drift on:
- Skills vs prompt-templates terminology (our 28 skills/*.md files
  are NOT Anthropic Agent Skills)
- Trigger-phrasing proactive-delegation on the remaining 2 agents
- Agent teams flag enabled but unreachable (phase-4.10 finding)
- Skills-for-enterprise review checklist not applied

## Success criteria (immutable, from masterplan)

1. every_doc_pattern_status_evidenced
2. qa_runs_live_code_checks_not_review
3. deviations_cite_doc_page
4. recommended_fix_included_not_implemented

## Scope

Write `docs/audits/compliance-subagents-skills.md` with a
pattern-per-row table covering EVERY pattern in:
- Claude Code / Sub-agents
- Claude Code / Agent teams
- Platform / Agent Skills overview + quickstart + best-practices +
  enterprise

Each row: Status / Evidence (file:line anchors) / Deviation / Risk
/ Recommended fix / MF-# mapping.

## Anti-patterns guarded

- Do NOT re-spawn `Explore` subagent — it was merged into
  researcher in 4.15.0.
- Do NOT re-spawn `harness-verifier` — merged into `qa`.
- Do NOT skim Skills-for-enterprise — phase-4.10 never read it.

## Out of scope

- Implementation (audit-only per CLAUDE.md)
- Managed Agents Skills (covered in phase-4.15.14)

## Risk

- Skills terminology collision may confuse go-live reviewers
  (phase-4.12 flagged; this cycle should verify still present)
- Unused `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` flag +
  `TeammateIdle` hook may be dead weight per stress-test doctrine

## References

- Phase-4.10 sub_agents.md + agent_teams.md (prior audits)
- Phase-4.11 skills_prompting_context.md
- Phase-4.12 prompting_skills_mcp_resources.md (enterprise skills)
