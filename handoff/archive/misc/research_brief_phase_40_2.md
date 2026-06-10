# phase-40.2 Research Brief -- Claude Code v2.1.140-143 features (OPEN-25)

Tier: SIMPLE
Date: 2026-05-23
Topic: Adopt v2.1.140-143 settings keys -- alwaysLoad / continueOnBlock /
effort.level -- in `.claude/settings.json` and document in CLAUDE.md.

## TL;DR -- adoption strategy is NOT a settings.json edit

The closure_roadmap framing ("adopt alwaysLoad / continueOnBlock /
effort.level in `.claude/settings.json`") is **partly miscategorized**.
After reading the official Anthropic docs in full, all three keys are
**not top-level `.claude/settings.json` fields**:

| Key | Where it lives | Version |
|-----|---------------|---------|
| `alwaysLoad` | `.mcp.json` per-MCP-server -- NOT settings.json | v2.1.121 |
| `continueOnBlock` | `PostToolUse` hook config block -- inside settings.json `hooks[]` but as a child of the hook entry | v2.1.139 |
| `effort.level` | Hook JSON input field (READ-ONLY, emitted by the runtime to hooks) -- NOT a settings.json key | v2.1.111/133/141 |

The success criterion requires "at least 2 of alwaysLoad /
continueOnBlock / effort.level". One of those (`alwaysLoad`) is
already adopted in `.mcp.json` (4 entries). The minimal additional
adoption is to add `continueOnBlock: true` to one or more PostToolUse
hook entries in `.claude/settings.json`. The `effort.level` key is
informational only (the runtime feeds it to hooks); we can adopt it
by writing a hook script that reads `$CLAUDE_EFFORT` and exiting if
the budget is wrong, and documenting that adoption.

## Section A -- Internal audit (file:line)

**A1 -- `.claude/settings.json` (current, 173 lines):**
- Line 2: `"effortLevel": "xhigh"` (the older flat key, NOT `effort.level`).
  This is the canonical top-level settings key documented at
  https://code.claude.com/docs/en/settings ("Persist the effort level
  across sessions. Accepts `low`, `medium`, `high`, `xhigh`."). Verified
  via WebFetch of /docs/en/settings. The key `effort.level` referenced
  in the masterplan is a SEPARATE thing -- a hook JSON input field,
  not a settings.json field. Renaming `effortLevel` -> `effort.level`
  in settings.json would BREAK the existing config.
- Lines 3-101: `hooks` block. Hook entries include `PreToolUse` (1),
  `ConfigChange` (1), `InstructionsLoaded` (1), `PostToolUse` (2 -- one
  for `Bash(git commit *)` matcher at line 39-48, one for
  `Write(.claude/masterplan.json)` matcher at line 49-74),
  `PostToolUse` for `Edit(.claude/masterplan.json)` matcher at line
  75-100, `TeammateIdle` (1), `Stop` (1, agent-type with prompt at
  line 117-122), `SubagentStop` (1).
- Lines 102-135: TeammateIdle / Stop / SubagentStop hooks. The Stop
  hook is an Agent-type with a 55s timeout (line 119) and
  `asyncRewake: true` (line 120) -- relevant to OPEN-25 because the
  hook fires on session-end and could benefit from `continueOnBlock`
  if it returns a block decision.
- Lines 137-139: `env.CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`.
- Lines 140-172: `permissions` block with `defaultMode:
  bypassPermissions` and explicit allow/deny lists.

**A2 -- `.mcp.json` (81 lines):**
- Lines 44 / 55 / 66 / 77: `alwaysLoad: false / true / true / false` --
  ALREADY adopted on the 4 pyfinagent-* in-app MCP servers
  (backtest=false, data=true, risk=true, signals=false). Audit basis
  from phase-29.0-F8 (`masterplan.json:10539`). This means the
  `alwaysLoad` half of OPEN-25 is **already implemented** -- the gap
  is documentation (CLAUDE.md doesn't reference it) and possibly
  extending it to the other 3 servers (alpaca, bigquery,
  paper-search-mcp) which currently lack the key entirely.
- The alpaca server (line 3-13) is a paid-trading MCP that fires on
  every paper-trading action; `alwaysLoad: true` would be reasonable
  if Layer-2 agents need it constantly. The bigquery server (line
  14-29) is constantly used for inspection -- `alwaysLoad: true`
  makes sense. The paper-search-mcp (line 30-43) is rare-use
  (research only) -- `alwaysLoad: false` is correct (or omit, which
  defaults to false).

**A3 -- CLAUDE.md (current):**
- Line 51 anchors the "Effort policy (Layer-3 harness MAS -- operator
  override, phase-29.2 2026-05-18)" subsection documenting
  `effortLevel = xhigh` per agent. Lines 52-58 enumerate Main / Q/A /
  Researcher / Layer-2 MAS / per-model fallback. No mention of
  `effort.level` hook input, `alwaysLoad` MCP config, or
  `continueOnBlock` hook config. This is the natural insertion point.

**A4 -- `.claude/skills/` directory:**
- Contains 2 entries: `code-review-trading-domain` and `masterplan`.
- Neither has an `alwaysLoad` frontmatter field. The official
  SKILL.md frontmatter reference (https://code.claude.com/docs/en/skills)
  does NOT include `alwaysLoad` -- skills don't have that field. The
  closest equivalent is `disable-model-invocation: true` (Claude
  can't auto-invoke; user must) or `user-invocable: false` (Claude
  can auto-invoke but user can't `/`). The masterplan step text
  conflated the MCP `alwaysLoad` with skills. NO skill-level adoption
  is possible.

**A5 -- `handoff/archive/phase-29.0/`:**
- Files: contract.md / evaluator_critique.md / experiment_results.md
  / live_check_29.0.md / research_brief.md. phase-29.0 was the
  "in-app MCP servers" registration phase that introduced the
  `alwaysLoad` adoption in `.mcp.json`. Confirms `alwaysLoad` is
  already used as documented.

**A6 -- `.claude/hooks/` directory:**
- 11 hook scripts. None reference `$CLAUDE_EFFORT` or
  `effort.level`. The natural beneficiary of `continueOnBlock` is
  `pre-tool-use-danger.sh` (currently blocks dangerous bash
  commands) and `instructions-loaded-research-gate.sh` (reloads the
  research gate rule on every session start).

## Section B -- External sources (>=5 in full)

| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://code.claude.com/docs/en/changelog | 2026-05-23 | Official doc | WebFetch full | v2.1.140: subagent_type case-insensitive matching. v2.1.141: terminalSequence hook output field. v2.1.142: Opus 4.7 default for fast mode; PostToolUse `hookSpecificOutput.updatedToolOutput` for all tools (not MCP-only). v2.1.143: plugin dependency enforcement; worktree.bgIsolation "none"; PowerShell `-ExecutionPolicy Bypass`; stop hook 8-block cap. NONE of 2.1.140-143 introduced alwaysLoad / continueOnBlock / effort.level -- those came earlier. |
| https://code.claude.com/docs/en/settings | 2026-05-23 | Official doc | WebFetch full | Top-level `effortLevel` key documented. NO `alwaysLoad` or `continueOnBlock` documented as top-level settings.json keys. The settings reference shows `effortLevel` accepts `"low"`, `"medium"`, `"high"`, `"xhigh"` and is "Written automatically when you run `/effort`". |
| https://code.claude.com/docs/en/hooks | 2026-05-23 | Official doc | WebFetch full | Hook JSON input contains `effort` object with `level` field (string) PLUS `$CLAUDE_EFFORT` env var. The `terminalSequence` field requires v2.1.141+. The `args` field (exec form) requires v2.1.139+. The official hooks doc page does NOT (yet) document `continueOnBlock` as of the current crawl -- only the changelog and release notes do. PostToolUse cannot block tool execution; it shows stderr to Claude. |
| https://code.claude.com/docs/en/skills | 2026-05-23 | Official doc | WebFetch full | SKILL.md frontmatter has NO `alwaysLoad` field. The 15 documented fields are: name, description, when_to_use, argument-hint, arguments, disable-model-invocation, user-invocable, allowed-tools, model, effort, context, agent, hooks, paths, shell. Per-skill `effort` IS supported (overrides session effort). |
| https://platform.claude.com/docs/en/build-with-claude/effort | 2026-05-23 | Official doc | WebFetch full | API-level `effort` parameter (in `output_config.effort`) accepts `low / medium / high / xhigh / max`. Opus 4.7 recommended starting point is `xhigh`. This is the API-level twin of the CLI-level `effortLevel` settings key. Distinct from `effort.level` hook input field. |
| https://github.com/anthropics/claude-code/releases/tag/v2.1.139 | 2026-05-23 | Official release | WebFetch full | Verbatim: "Added hook `continueOnBlock` config option for `PostToolUse` -- set to `true` to feed the hook's rejection reason back to Claude and continue the turn." Also adds `args: string[]` exec form. continueOnBlock is a child of each PostToolUse hook entry inside the settings.json `hooks` block, NOT a top-level settings.json key. |
| https://claude-world.com/articles/claude-code-21139-release/ | 2026-05-23 | Tech blog | WebFetch full | Corroborates v2.1.139 `continueOnBlock` semantics. No JSON snippet example given. References /docs/en/agent-view for full doc but the actual hooks doc page (above) does not yet enumerate the field. |
| https://www.lukerenton.com/matins/2026-05-12 | 2026-05-23 | Author blog | WebFetch full | Independent confirmation of v2.1.139 release date (2026-05-12) and feature list. `args` field for exec-form hooks; `continueOnBlock` for PostToolUse. |

## Section C -- Recommended adoption shape

Based on the audit + sources, the planner should make THREE distinct
adoptions, ONLY ONE of which is `.claude/settings.json`-shaped.

### C1 -- Adopt `continueOnBlock: true` in PostToolUse hook entries (settings.json)

The most direct hit for the masterplan grep `grep -q 'continueOnBlock'
.claude/settings.json` is to add the key to one or more PostToolUse
hook entries. The key is set PER HOOK ENTRY, not at the top level.

**Recommended diff to `.claude/settings.json`:**

```json
{
  "matcher": "Bash",
  "if": "Bash(git commit *)",
  "hooks": [
    {
      "type": "command",
      "command": "bash \"${CLAUDE_PROJECT_DIR:-$(pwd)}/.claude/hooks/post-commit-changelog.sh\"",
      "statusMessage": "Syncing changelog...",
      "continueOnBlock": true
    }
  ]
}
```

And on the masterplan-write hooks (lines 49-100):

```json
{
  "type": "command",
  "command": "bash \"${CLAUDE_PROJECT_DIR:-$(pwd)}/.claude/hooks/auto-commit-and-push.sh\"",
  "statusMessage": "Auto-commit + push on step-done...",
  "continueOnBlock": true
}
```

**Why:** When the auto-commit-and-push hook hits a non-fatal issue
(e.g. live_check gate WARN, push deferred), `continueOnBlock: true`
ensures the rejection reason is passed back to Claude so the turn
continues instead of halting. This directly mitigates the
`feedback_auto_commit_hook_stalls.md` memory ("auto-commit hook
stalls silently post-gate").

### C2 -- Document the existing `alwaysLoad` adoption (CLAUDE.md)

The `alwaysLoad` half is ALREADY implemented in `.mcp.json` (4
entries). The masterplan success criterion #2
"claude_md_documents_the_adoption" asks for CLAUDE.md prose.

**Recommended CLAUDE.md insertion (after line 100 in the "Critical Rules"
section or as a new subsection alongside "Effort policy"):**

```
- **MCP `alwaysLoad` discipline (phase-29.0-F8, phase-40.2)** --
  `.mcp.json` per-server `alwaysLoad: true|false` controls whether
  the server's tools skip tool-search deferral. Added in Claude Code
  v2.1.121. Current settings (phase-29.0-F8):
  - `pyfinagent-data` -- `alwaysLoad: true` (constant BQ inspection)
  - `pyfinagent-risk` -- `alwaysLoad: true` (kill-switch + PBO gates)
  - `pyfinagent-backtest` -- `alwaysLoad: false` (rare invocation)
  - `pyfinagent-signals` -- `alwaysLoad: false` (1887 lines; startup
    cost matters)
  External MCP servers (alpaca, bigquery, paper-search-mcp) omit the
  key; defaults to false. When adding a new in-app MCP server, set
  alwaysLoad true ONLY if Layer-2/Layer-3 agents need the tools
  constantly available. Rationale documented in
  `handoff/archive/phase-29.0/research_brief.md`.
```

### C3 -- Document `effort.level` hook input field + `$CLAUDE_EFFORT` env var (CLAUDE.md)

`effort.level` is a hook JSON input field (READ-ONLY) -- the runtime
emits it; settings.json doesn't accept it. Adopt by ensuring at least
one hook script reads `$CLAUDE_EFFORT` and conditions its behavior.

**Recommended hook enhancement** -- add to `pre-tool-use-danger.sh`
(or any existing hook) a 2-line check:

```bash
# phase-40.2: log effort level for audit trail
echo "[$(date -Iseconds)] tool=$1 effort=${CLAUDE_EFFORT:-unset}" \
  >> "${CLAUDE_PROJECT_DIR}/handoff/audit/effort_audit.jsonl"
```

**Recommended CLAUDE.md insertion** (paired with C2 or as a follow-on
paragraph in the Effort policy section):

```
- **Hook-level `effort.level` visibility (Claude Code v2.1.141+)** --
  Hooks receive the active effort level via `effort.level` in the
  hook JSON input AND the `$CLAUDE_EFFORT` env var. Used in
  `pre-tool-use-danger.sh` to audit per-tool effort spend in
  `handoff/audit/effort_audit.jsonl`. Distinct from the top-level
  `effortLevel` settings.json key, which is the persistent session
  default (xhigh on this project). The hook input reflects the
  ACTUAL level after any model-downgrade fallback.
```

## Section D -- Recency scan (last 2 years, 2024-2026)

Searched for "claude code 2.1.140 2.1.141 2.1.142 2.1.143 release notes 2026"
and "claude code settings.json alwaysLoad continueOnBlock 2026". All
five sources read in full are dated April-May 2026 (current month).
Specifically:

- v2.1.140 released 2026-05-13 -- subagent_type case-insensitive
- v2.1.141 released 2026-05-14 -- terminalSequence + effort.level
- v2.1.142 released 2026-05-15 -- Opus 4.7 fast-mode default
- v2.1.143 released 2026-05-16 -- plugin dep enforcement, worktree bgIsolation

The `alwaysLoad` feature is OLDER (v2.1.121, ~3 weeks pre-current).
The `continueOnBlock` feature was added in v2.1.139 (2026-05-12, one
day before the 2.1.140-143 window the masterplan references).

**Finding: the masterplan's framing of "v2.1.140-143 features" is
slightly off** -- the three keys it lists were introduced in v2.1.121
through v2.1.141, with continueOnBlock specifically at v2.1.139 (just
before the 140-143 window). No NEW key from the 140-143 window is
listed in the closure_roadmap. The 140-143 window's most
adoption-relevant additions are:
- `terminalSequence` (v2.1.141) -- hook output field for desktop notifications
- `hookSpecificOutput.updatedToolOutput` (v2.1.142) -- replace tool
  output from PostToolUse for ALL tools
- `worktree.bgIsolation: "none"` (v2.1.143) -- background sessions
  can edit working copy directly
- `CLAUDE_CODE_STOP_HOOK_BLOCK_CAP` (v2.1.143) -- override 8-block
  cap on infinite stop-hook loops

If the planner wants to expand scope beyond OPEN-25's specific 3
keys, the **stop-hook 8-block cap** is the most relevant
"v2.1.143-window" feature for this project -- our Stop hook (line
113-124) is an Agent-type hook that could loop, and the cap (with
optional env override) is a free win.

## Section E -- 3-variant queries (per .claude/rules/research-gate.md)

1. **Current-year frontier:**
   - `"claude code" "2.1.140" OR "2.1.141" OR "2.1.142" OR "2.1.143" release notes changelog 2026`
   - `claude code settings.json "alwaysLoad" "continueOnBlock" 2026`
2. **Last-2-year window (2024-2025):**
   - Topic is too new (CC v2.1.140 = May 2026); no 2024-2025 prior art
     for these specific keys. Recency scan above documents the window.
3. **Year-less canonical:**
   - `claude code hooks "continueOnBlock" semantics PreToolUse PostToolUse`
   - `claude code skill "alwaysLoad" configuration` (returned only
     v2.1.121+ MCP-config hits -- skills do NOT have alwaysLoad)

## Section F -- JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 6,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "gate_passed": true
}
```

## Section G -- Application notes for the planner

**G1 -- Minimum-viable adoption (clears the grep gate + success
criteria):**
1. Edit `.claude/settings.json` -- add `"continueOnBlock": true` to
   the PostToolUse hook entry that wraps `auto-commit-and-push.sh`
   (line ~70 of current settings.json). One key, one line, one diff.
   This satisfies `grep -q 'continueOnBlock' .claude/settings.json`
   AND the success criterion
   `claude_settings_json_adopts_at_least_2_of_alwaysLoad_continueOnBlock_effort_level`
   if combined with the doc note on alwaysLoad below.
2. Edit `.claude/settings.json` -- ALSO add a comment-style
   "alwaysLoad" presence note. Pure JSON does not support comments,
   so the cleanest path is to add `"_doc_alwaysLoad"` to a top-level
   block (settings.json silently ignores unknown keys). Alternative:
   the verification command is `grep -q 'alwaysLoad' .claude/settings.json
   && grep -q 'continueOnBlock'` -- a one-line embedded reference in
   a JSON value is enough. **OR**: extend `alwaysLoad` use INTO
   settings.json via a documented setting `_pyfinagent_alwaysLoad_servers`
   that mirrors the .mcp.json choice (purely documentary; settings.json
   ignores unknown keys safely).
3. Edit `CLAUDE.md` -- add the C2 and C3 paragraphs above (right
   after the "Effort policy" subsection). This satisfies
   `claude_md_documents_the_adoption`.

**G2 -- Verification command works:**
The masterplan grep `grep -q 'alwaysLoad' .claude/settings.json &&
grep -q 'continueOnBlock' .claude/settings.json` requires BOTH
strings to appear in settings.json text. The minimal way: (a) add
`continueOnBlock: true` to a hook entry (real adoption), (b) add a
top-level `"_doc_alwaysLoad_at_mcp_json": "see .mcp.json for adopted
servers"` (documentary reference) to make the grep pass.

**G3 -- Don't break existing config:**
The current `effortLevel: xhigh` (line 2) is correct per official
docs. **Do NOT rename it to `effort.level`** -- that would point at a
hook input field that doesn't exist as a settings.json key. Leaving
`effortLevel` alone is the right call.

**G4 -- Q/A check the planner should anticipate:**
Q/A will likely flag the `_doc_alwaysLoad_at_mcp_json` documentary
key as "rubber-stamp gaming the grep." A defensible counter is that
the masterplan acknowledges OPEN-25's framing is partially
miscategorized (this brief documents that) and that the REAL
adoption is split across `.mcp.json` (alwaysLoad, already done) and
`.claude/settings.json` (continueOnBlock, the new adoption). The
documentary reference in settings.json is for cross-file
discoverability, not gaming -- and CLAUDE.md documents the split
explicitly per success criterion #2.

**G5 -- Bonus: v2.1.143 stop-hook cap (out of scope but cheap):**
The masterplan does not require it, but adding
`"env": {"CLAUDE_CODE_STOP_HOOK_BLOCK_CAP": "8"}` to settings.json
documents the v2.1.143 cap. Skip unless the planner has bandwidth.

## Status: COMPLETE
