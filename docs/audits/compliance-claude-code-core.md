# Compliance Audit — Claude Code Core
**Phase:** 4.15.12 | **Date:** 2026-04-18 | **Effort:** moderate
**Baseline:** phase-4.11 claude_code_core.md | **Auditor:** researcher (merged)

---

## Scope

Hooks / permissions / sandboxing / ZDR / settings / memory / plugins /
claude-directory layout — POST-4.15.0 + MF-40-44 state.

---

## Pattern Table (25 rows)

### HOOKS

| # | Pattern | Status | Evidence | Deviation / Risk | Fix | MF# |
|---|---------|--------|----------|-----------------|-----|-----|
| 1 | **PostToolUse — git commit** wired to `post-commit-changelog.sh` | COMPLIANT | `settings.json` L3-15; `if: Bash(git commit *)` matcher correct | None | — | MF-41 |
| 2 | **PostToolUse — masterplan.json write** wired to 3 hooks (archive-handoff, masterplan-memory-sync, commit-reminder) | COMPLIANT | `settings.json` L16-35; chain order logical | None | — | MF-41 |
| 3 | **TaskCompleted** wired to agent cross-verifier | COMPLIANT | `settings.json` L37-48; `asyncRewake: true`; 60s timeout. Agent prompt reads masterplan + runs deterministic checks first | timeout is 60s, doc recommends leaving buffer vs 55s Stop hook; risk of TaskCompleted agent racing Stop agent | Align to 55s or document the intentional 5s delta | MF-40 |
| 4 | **Stop** wired to agent masterplan verifier | COMPLIANT | `settings.json` L60-71; loop-prevention guard (`stop_hook_active`) present; 55s timeout | Stop agent re-reads masterplan via Bash — if masterplan.json write is in-flight this could read stale state | Acceptable; race window is narrow | MF-40 |
| 5 | **SubagentStop** wired (NEW post-4.15.0) | NEWLY COMPLIANT | `settings.json` L72-82; loop-prevention bash guard. Resolves MF-43 gap | SubagentStop uses a command hook (not agent); only echoes `ok:true` — does no real verification. Exists to satisfy the hook slot, not to add safety | Acceptable as a gate stub; document that it is intentionally thin | MF-43 |
| 6 | **TeammateIdle** wired to `teammate-idle-check.sh` | COMPLIANT | `settings.json` L49-58; script exits 2 when in-progress steps exist | None | — | — |
| 7 | **InstructionsLoaded** NOT wired | MISSING | `jq '.hooks | keys'` returns `[PostToolUse, Stop, SubagentStop, TaskCompleted, TeammateIdle]` — no InstructionsLoaded key | Per per-step-protocol.md L186-189: "Fix: InstructionsLoaded hook reloads this rule every session start". Without it, Research-gate drift (7/9 phase-4.8 misses) has no mechanical guard | Add InstructionsLoaded hook that appends research-gate rule text to additionalContext | MF-44 |
| 8 | **SessionStart** NOT wired | MISSING | Not in keys list | Low-risk gap: SessionStart is command-only (no agent type), so limited value. But it could set `CLAUDE_PROJECT_DIR` or echo startup banner | Optional; lower priority than InstructionsLoaded | — |
| 9 | **PreToolUse deny on dangerous Bash** NOT wired | MISSING | No PreToolUse key in hooks | Docs: "PreToolUse can deny the tool call before it runs" — the bypassPermissions default makes this the only runtime guard against dangerous commands like `git push --force`, `rm -rf`, or redirect `>>` to protected files | Add PreToolUse hook matching `Bash(rm -rf *)`, `Bash(git push --force *)`, `Bash(git reset --hard *)` with exit 2 | — |
| 10 | **ConfigChange** NOT wired | MISSING | Not in keys list | ConfigChange can block settings.json mutations mid-session. Without it, a subagent could silently overwrite `defaultMode` | Low priority (bypassPermissions already grants full access, so ConfigChange only guards against accidental writes) | — |

### PERMISSIONS

| # | Pattern | Status | Evidence | Deviation / Risk | Fix | MF# |
|---|---------|--------|----------|-----------------|-----|-----|
| 11 | **defaultMode: bypassPermissions** | RISK — NOT FIXED | `settings.json` L88; jq confirms `"bypassPermissions"` | Docs (permissions.md): "Only use this mode in isolated environments like containers or VMs where Claude Code cannot cause damage." pyfinagent runs on a developer Mac, not a VM | Move to `acceptEdits`; add explicit `Bash(*)` allow for commands that need it; add deny rules for destructive commands | — |
| 12 | **allow list present** (`Bash, Read, Write, Edit, Glob, Grep, Agent, WebSearch, WebFetch`) | PARTIAL | `settings.json` L89-101. Allow list is present but rendered moot by bypassPermissions | Once bypassPermissions is removed, the allow list is the correct shape to whitelist needed tools. No deny list exists | Add deny rules for `Bash(git push *)`, `Bash(rm -rf *)`, `Bash(git reset --hard *)` alongside the allow list | — |
| 13 | **per-agent permissionMode: plan** on researcher + qa | NEWLY COMPLIANT | `researcher.md` L10; `qa.md` L10, both confirmed. Resolves MF-42 gap | `plan` mode means agents cannot Edit/Write by design — qa.md §Constraints already documents "NEVER Edit or Write" | None | MF-42 |
| 14 | **disableBypassPermissionsMode not set** | MISSING | No such key in settings.json or settings.local.json | Docs: admins can set `permissions.disableBypassPermissionsMode: "disable"` to prevent the mode. Without it, any agent can re-enable bypassPermissions | Set in project settings now; move to managed settings once org-level deployment is in place | — |
| 15 | **settings.local.json scope discipline** | PARTIAL | `.claude/settings.local.json` contains 3 one-time migration Bash allow rules from the 4.15.0 re-org (`mkdir -p docs/runbooks`, `mv .claude/agents/per-step-protocol.md`, `rm .claude/agents/qa-evaluator.md ...`). These are dead rules — the migration is complete | Dead allow rules add surface area. No active risk but should be pruned after confirming the commands will not be re-needed | Remove the 3 stale Bash allow entries from settings.local.json | — |

### SANDBOXING

| # | Pattern | Status | Evidence | Deviation / Risk | Fix | MF# |
|---|---------|--------|----------|-----------------|-----|-----|
| 16 | **sandbox block absent** | MISSING | `jq '.sandbox' .claude/settings.json` returns `null` | Docs: macOS uses Seatbelt, works out of the box, no packages needed. With bypassPermissions + no sandbox, a compromised or hallucinating agent can write anywhere on the host filesystem and exfiltrate to any network domain | Enable sandbox (`"sandbox": {"enabled": true}`) in settings.json; run `/sandbox` to confirm Seatbelt activates | — |
| 17 | **sandbox.failIfUnavailable not set** | INFO | N/A — sandbox not enabled | If sandbox is later enabled without `failIfUnavailable: true`, a silent sandbox-startup failure would leave the session unguarded | Set `failIfUnavailable: true` alongside `enabled: true` | — |

### ZDR / DATA USAGE

| # | Pattern | Status | Evidence | Deviation / Risk | Fix | MF# |
|---|---------|--------|----------|-----------------|-----|-----|
| 18 | **ZDR ineligibility** | OPERATOR DECISION | Claude Code API with default data usage. Project handles financial signals + live portfolio data (PMS dataset in BigQuery) | ZDR requires an enterprise agreement. pyfinagent uses API tier without explicit ZDR opt-in. Financial data (positions, paper trades) passes through Claude. Prior audits flagged this as an owner decision, not a blocker | Owner (Peder) must evaluate ZDR requirement against regulatory obligations for financial data in AI pipelines | — |

### SETTINGS + SERVER-MANAGED SETTINGS

| # | Pattern | Status | Evidence | Deviation / Risk | Fix | MF# |
|---|---------|--------|----------|-----------------|-----|-----|
| 19 | **CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS env var** | COMPLIANT — WATCH | `settings.json` L85; `"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"`. Required for TeammateIdle hook to work | This is an experimental flag. When agent teams GA, the flag may change or become a no-op; watch release notes | Monitor Claude Code release notes per-sprint | — |
| 20 | **No managed settings deployed** | INFO / GAP | No `/Library/Application Support/ClaudeCode/` managed settings detected | Without managed settings, `disableBypassPermissionsMode` and `strictKnownMarketplaces` cannot be enforced at the machine level. Fine for solo dev; becomes a gap at team scale | Deploy managed settings before team onboarding | — |
| 21 | **Stray .bak files in .claude/** | LOW RISK | `find .claude -name '*.bak-*'` returns `.claude/settings.json.bak-harness-ABCD` and `.claude/masterplan.json.bak-phase4.5` | Stale snapshots checked into the working tree alongside live config files. Risk: confusing diffs; no security risk (they are not loaded) | Delete or move to `handoff/archive/`; gitignore `*.bak-*` pattern | — |

### MEMORY

| # | Pattern | Status | Evidence | Deviation / Risk | Fix | MF# |
|---|---------|--------|----------|-----------------|-----|-----|
| 22 | **Two parallel memory systems still coexist** | UNRESOLVED | System 1: `.claude/agent-memory/researcher/` (custom project-scoped memory, 6 files). System 2: `~/.claude/projects/-Users-ford--openclaw-workspace-pyfinagent/memory/` (auto-memory, MEMORY.md index + topic files). Prior phase-4.15.3 audit flagged "pick one" | Dual systems create drift risk: researcher agent reads `.claude/agent-memory/researcher/MEMORY.md`; main agent reads the auto-memory MEMORY.md via system-reminder injection. If both write about the same topic (e.g. PSR/DSR formulas), future agents may act on the older or the newer version without knowing | Decision needed: (a) deprecate `.claude/agent-memory/` and migrate to auto-memory, or (b) document the split as intentional with clear ownership boundaries per system | — |
| 23 | **session-restart doctrine documented** | NEWLY COMPLIANT | `CLAUDE.md` L46-47: agent definition changes require session restart + separation-of-duties rule. Grep confirms 2 hits. Resolves MF-44 documentation gap | CLAUDE.md documents the rule but no hook enforces it. A SubagentStart hook could warn when an agent file was modified within the current session | Acceptable as documentation-only for now; add SubagentStart hook when agents are edited more frequently | MF-44 |

### PLUGINS + MARKETPLACES

| # | Pattern | Status | Evidence | Deviation / Risk | Fix | MF# |
|---|---------|--------|----------|-----------------|-----|-----|
| 24 | **strictKnownMarketplaces not set** | INFO | Not present in settings.json or settings.local.json | Docs: when undefined, "No restrictions. Users can add any marketplace." Currently no plugins are installed, so no active risk. If team onboards and installs community plugins, any marketplace can be added | Set `strictKnownMarketplaces: []` in managed settings (or project settings as interim) to block unapproved marketplaces until an approved list exists | — |
| 25 | **No plugins installed; enabledPlugins absent** | COMPLIANT (by absence) | No `.claude-plugin/` directory; no `enabledPlugins` key | pyfinagent uses MCP servers (slack, alpaca) not the plugin system. MCP is configured in `.mcp.json` with correct env-var substitution. No plugin surface area to audit | None — reassess when plugin adoption begins | — |

---

## Summary: Newly Compliant post-4.15.0 + MF-40-44

| Item | MF# | What changed |
|------|-----|-------------|
| SubagentStop hook added | MF-43 | Was missing in prior audit; now wired (thin gate stub) |
| permissionMode: plan on researcher + qa | MF-42 | Both agents now cannot Edit/Write; resolves the write-permission gap on subagents |
| Session-restart + Separation-of-duties doctrine | MF-44 | Documented in CLAUDE.md L46-47 |
| TaskCompleted + Stop hooks (existed, verified) | MF-40 | Still compliant; timeout alignment note added |

## Still Missing (priority order)

1. **InstructionsLoaded hook** — highest priority; only mechanical guard against research-gate skip
2. **Sandboxing** — macOS Seatbelt is zero-config; just needs `"sandbox": {"enabled": true}`
3. **bypassPermissions -> acceptEdits migration** — pyfinagent is not a VM; this is a standing risk
4. **PreToolUse deny on destructive Bash** — `rm -rf`, `git push --force`, `git reset --hard`
5. **disableBypassPermissionsMode** — should be set in project settings now, managed settings later
6. **Dead allow rules in settings.local.json** — prune 3 one-time migration entries
7. **Dual memory system** — resolve ownership between `.claude/agent-memory/` and auto-memory
8. **strictKnownMarketplaces** — block open marketplace installs (low urgency, no plugins active)
9. **Stray .bak files** — cosmetic; delete and gitignore

## ZDR Status

Ineligible at current API tier without explicit enterprise ZDR agreement.
Financial signal + portfolio data flows through Claude API calls.
Owner decision required — not a code fix.

---

*Sources: Claude Code docs — hooks (28 events confirmed), permissions, sandboxing, memory, plugin-marketplaces; live state read from `.claude/settings.json`, `.claude/settings.local.json`, `.claude/agents/researcher.md`, `.claude/agents/qa.md`, `docs/runbooks/per-step-protocol.md`, `CLAUDE.md`.*
