# Contract — Cycle 4.15.12 — Claude Code core compliance

## Research gate
Merged researcher: hooks / permissions / sandboxing / ZDR /
settings / memory / plugins / claude-directory docs + the POST
4.15.0 state of `.claude/settings.json`, `.claude/hooks/*`,
CLAUDE.md, `.claude/agents/`, and `docs/runbooks/`.

## Hypothesis
After the 4.15.0 MAS merge + recent 4.15.3 + MF-40-44 fixes, the
Claude Code configuration should be tighter than phase-4.10/4.11
captured. Verify:
- `permissionMode: plan` on both merged agents (just added)
- `SubagentStop` hook now present
- Still missing: `InstructionsLoaded`, `PreToolUse` deny on
  dangerous bash, `ConfigChange`, `SessionStart`, sandboxing
- `bypassPermissions` default still live (not yet fixed)

## Success criteria
1. every_doc_pattern_status_evidenced
2. qa_runs_live_code_checks_not_review
3. deviations_cite_doc_page

## Scope
`docs/audits/compliance-claude-code-core.md` covering:
- Sub-agents (schema recheck post-4.15.0)
- Hooks (7 documented events, which wired)
- Permission modes (bypass -> acceptEdits path)
- Sandboxing (macOS Seatbelt)
- ZDR + data usage
- Settings + server-managed-settings
- Memory (CLAUDE.md + auto memory + .claude/agent-memory/ custom)
- Plugins (strictKnownMarketplaces)
- Claude-directory layout

## References
Phase-4.10 sub_agents + agent_teams audits, phase-4.11
claude_code_core.md, phase-4.15.3 subagents-skills.md.
