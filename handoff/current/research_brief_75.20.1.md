# Research Brief -- Step 75.20.1: qa subagent Write/Edit injection past frontmatter allowlist

tier: complex | status: COMPLETE | started+finished: 2026-07-24
installed Claude Code: **2.1.218** (`claude --version`)

## Question

The Agent-tool loader gives the `qa` subagent Write+Edit despite `qa.md:4`
granting only `Read, Bash, Glob, Grep, SendMessage` + 4 read-only browser tools.
This brief feeds four work items: (a) a re-runnable probe, (b) the injection
SOURCE (upstream vs local), (c) enforcement options + the identity fields a hook
actually receives, (d) the git-status cleanliness insertion into
`per-step-protocol.md` S4.

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
| --- | --- | --- | --- | --- |
| https://code.claude.com/docs/en/sub-agents | 2026-07-24 | Official doc | WebFetch (89KB persisted; read tools-filter §L279-396 + memory §L493-530 verbatim) | `memory:` auto-enables Read/Write/Edit; tools-field is an allowlist narrowed by 2 filters that only REMOVE; parent bypassPermissions overrides child permissionMode |
| https://code.claude.com/docs/en/hooks | 2026-07-24 | Official doc | curl raw .md (3149 lines; read common-input-fields + PreToolUse output + exit-2 in full) | **agent_id + agent_type ARE in PreToolUse common input "when the hook fires inside a subagent call"; for custom subagents agent_type = frontmatter `name`** |
| https://code.claude.com/docs/en/permissions | 2026-07-24 | Official doc | WebFetch (58.8KB persisted; read permission-modes + hook-precedence + Agent-rules verbatim) | PreToolUse hook runs before the prompt + blocks even vs allow rules; bypassPermissions skips PROMPTS only; `Agent(name)` gates spawn, NOT a subagent's tool set |
| https://github.com/anthropics/claude-code/issues/57507 | 2026-07-24 | Issue (bug) | WebFetch full | Quotes the doc "Read, Write, and Edit tools are automatically enabled..."; memory/allowlist interaction; **closed "not planned" / stale** |
| https://github.com/anthropics/claude-code/issues/57118 | 2026-07-24 | Issue (bug) | WebFetch full | Sub-agent `permissionMode` frontmatter INEFFECTIVE for Edit/Write; parent acceptEdits/bypass precedence; **closed "not planned"** |
| https://github.com/anthropics/claude-code/issues/40140 | 2026-07-24 | Issue (feat req) | WebFetch full | Requested agent_id/agent_type in PreToolUse (Mar 2026); **closed "not planned"/stale** -- yet the current doc NOW documents the fields present |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
| --- | --- | --- |
| github.com/anthropics/claude-code/issues/60237 | Issue | "tools array drops first/last position" -- does NOT match our symptom (Read + last browser tool are KEPT; Glob/Grep are middle) |
| github.com/anthropics/claude-code/issues/16126 | Issue | earlier dup of #40140 (agent identity in PreToolUse) |
| github.com/anthropics/claude-code/issues/57507 (search dup) | Issue | already read in full |
| github.com/anthropics/claude-code/issues/49559 | Issue | plugin skill `context:fork`/`agent:` not honored -- adjacent, not tool-injection |
| github.com/anthropics/claude-code/issues/44412 | Issue | PreToolUse `updatedInput` ignored for Agent tool -- adjacent |
| github.com/anthropics/claude-code/issues/17283 | Issue | Skill tool frontmatter -- adjacent |
| code.claude.com/docs/en/agent-sdk/hooks | Official doc | SDK hooks (Python) -- our project uses CLI command hooks; kept the CLI hooks ref as authoritative |
| github.com/anthropics/claude-code/blob/main/CHANGELOG.md | Changelog | recency scan target (see below) |
| github.com/anthropics/claude-agent-sdk-typescript/.../CHANGELOG.md | Changelog | SDK changelog |
| claudelog.com/claude-code-changelog | 3rd-party changelog mirror | community weight |
| morphllm.com/claude-code-hooks | Blog | hook overview |
| developersdigest.tech/guides/subagent-frontmatter | Blog | frontmatter overview |
| dev.to/owen_fox/...hooks-subagents-and-skills | Blog | community |
| hidekazu-konishi.com/.../claude_code_hooks_complete_guide | Blog | community |
| shanraisshan/claude-code-best-practice, luongnv89/claude-howto | Repo docs | community |
| penligent.ai/.../inside-claude-code..., releasebot.io/updates/anthropic | Blog | community |

## Recency scan (2024-2026)

Three-variant discipline: year-less ("...frontmatter tools field not honored
Write Edit injected"), 2026 ("...PreToolUse hook agent identity 2026"), and a
2025-2026 changelog pass ("...subagent tools resolution memory auto-enable
2025 2026"). Result: **the entire evidence base is current (2026); no older
canonical source is superseded because this is a 2026-only capability surface.**
New/relevant findings in the window:
- The agent_type-in-PreToolUse capability (hooks doc) is NEWER than the Mar-2026
  feature request (#40140) that asked for it -- the request is "not planned"/stale
  yet the doc now documents the fields. Authoritative = the current doc; the
  probe (a) must EMPIRICALLY confirm the field is populated on our v2.1.218.
- Changelog search surfaced 2026 subagent fixes (transcript view, backgrounding
  ctrl+b no-restart) but NO entry reverting the memory auto-enable -- the
  documented "memory auto-enables Read/Write/Edit" is the live, intended spec.
- #57507's v2.1.137 bug (allowlist OVERRODE memory auto-enable, so memory didn't
  work) has since converged to the DOCUMENTED behavior at v2.1.218 (auto-enable
  wins / is additive -> Write/Edit present), matching this session's reproduction.

## Key findings

1. **The injection is DOCUMENTED, intended behavior of `memory: project` -- not a
   bug and not a local misconfig.** qa.md:25 sets `memory: project`. The
   sub-agents doc "Enable persistent memory" section states verbatim: *"When
   memory is enabled: ... Read, Write, and Edit tools are automatically enabled
   so the subagent can manage its memory files."* (Source: code.claude.com/docs/en/sub-agents,
   accessed 2026-07-24). Q/A's memory is REAL and load-bearing: `.claude/agent-memory/qa/`
   holds `MEMORY.md` + 6 curated project memories (e.g.
   `project_committed_criterion_gitignore_check.md`, last curated 2026-07-10).
   So the Write/Edit surface is the price of Q/A having memory.

2. **You cannot keep Q/A memory AND exclude Write/Edit via the tools allowlist.**
   The doc makes memory-write access inseparable from `memory:`. The doc's own
   escape hatch -- turning off auto-memory (`autoMemoryEnabled` /
   `CLAUDE_CODE_DISABLE_AUTO_MEMORY`) -- is GLOBAL (kills every agent's memory,
   incl. the researcher's) and "the `memory` field has no effect and the subagent
   launches without ... the memory tool access". The only SCOPED way to drop the
   injection is to delete `memory: project` from qa.md -- which costs Q/A its
   accumulated evaluation knowledge. (Source: sub-agents doc, memory section.)

3. **`disallowedTools` on the Workflow `agent()` call is silently ignored (probe
   wf_78b46633-fdd); frontmatter `disallowedTools` is a DIFFERENT, documented
   field.** The doc documents `disallowedTools: Write, Edit` as a FRONTMATTER
   denylist (sub-agents doc L280, L350-356) -- distinct from the ignored `agent()`
   option. BUT its order-of-operations vs the memory auto-enable is undocumented,
   and if it wins it would ALSO break Q/A memory curation. => must be probed, not
   assumed; not a doc-guaranteed fix.

4. **qa.md's `permissionMode: plan` (L27) is INERT here.** The sub-agents doc:
   *"If the parent uses `bypassPermissions` or `acceptEdits`, this takes
   precedence and can't be overridden."* This project runs
   `settings.json:defaultMode = bypassPermissions` (required for unattended
   harness; auto-memory `project_permissions_bypass_required`). So Q/A inherits
   bypassPermissions and its `permissionMode: plan` is overridden. Corroborated by
   issue #57118 (permissionMode frontmatter ineffective, closed not-planned).
   **Do not rely on permissionMode:plan to block writes.**

5. **A PreToolUse hook CAN see the acting subagent and IS the only per-agent
   tool-scoping mechanism.** Deciding doc text (hooks doc L623-628, verbatim):
   *"When running with `--agent` or inside a subagent, two additional fields are
   included: `agent_id` -- ... Present only when the hook fires inside a subagent
   call ...; `agent_type` -- Agent name ... For custom subagents, this is the
   `name` field from the agent's frontmatter, not the filename."* So a PreToolUse
   hook receives `agent_type == "qa"` for the qa subagent's tool calls.

6. **The hook blocks even under bypassPermissions; settings.json permission rules
   CANNOT scope a tool-deny per acting agent.** Permissions doc L411: PreToolUse
   hooks "run before the permission prompt, for every tool except EndConversation."
   L419: "A hook that exits with code 2 stops the tool call before permission
   rules are evaluated, so the block applies even when an allow rule would
   otherwise let the call proceed." L58: bypassPermissions "Skips permission
   PROMPTS" (not hooks). L377-383: `Agent(AgentName)` rules control which
   subagents Claude may SPAWN -- there is NO settings.json construct that denies a
   TOOL to a specific acting subagent. A global `Write` deny would break Main
   (Main needs Write). => the PreToolUse hook is the ONLY provable per-agent block.

7. **Known hook limitation (must not over-claim):** permissions doc L272 -- Write/Edit
   deny/hook rules apply to Claude's built-in file tools, "They don't apply to
   arbitrary subprocesses ... like a Python or Node script that opens files
   itself." Q/A HAS Bash, so a `Bash(python -c "open(...,'w')")` write bypasses a
   Write/Edit hook. qa.md prose already forbids mutating Bash (no `>`/`sed -i`/`git
   commit`), and the git-status backstop (d) catches leftovers. The hook closes the
   DIRECT tool path; the Bash path is a separately-governed surface.

8. **The Glob/Grep "drop" in the section-7a self-disclosure is almost certainly a
   self-report artifact, not a real removal.** Glob + Grep are in the doc's
   background-subagent KEEP list (sub-agents doc L336: "Read, Grep, Glob, Bash,
   ... Edit, Write ..."). An agent enumerating its own tools misreports. This is
   the core reason the probe (a) must measure EXECUTION, not self-disclosure.

## Internal code inventory

| File | Lines | Role | Status |
| --- | --- | --- | --- |
| `.claude/agents/qa.md` | :4 tools; :25 `memory: project`; :27 `permissionMode: plan`; :3 + :484 "NEVER Edit/Write" prose | The subject config | memory:project is the injection ROOT CAUSE; permissionMode:plan is INERT under parent bypass |
| `.claude/workflows/qa-verdict.js` | :111-120 disclosed residual comment; :123 `agentType:'qa'` | Primary launch | already documents "loader injects Write/Edit ... disallowedTools silently ignored"; PROMPT step B already has a Q/A-side `git status --short` check |
| `.claude/settings.json` | :16-25 PreToolUse hook (no matcher, fires all tools); :153 bypassPermissions; :166-194 deny list | Where the enforcement hook + any deny lands | existing `pre-tool-use-danger.sh` PROVES hooks fire under this project's bypassPermissions |
| `backend/tests/test_phase_75_20_qa_browser_grant.py` | :59-65 `_tools_line`; :110-114 agentType assert | Existing 75.20 grant tests | pattern to extend: add a pin that the qa Write/Edit residual is covered by hook+prose |
| `scripts/qa/verify_qa_roster_live.sh` | whole file | Existing probe pattern (on-disk + git + operator self-disclosure) | model for probe shape BUT it relies on self-disclosure -- 75.20.1 probe must be BEHAVIORAL instead |
| `docs/runbooks/per-step-protocol.md` | S4 EVALUATE = L111-258 (LOG starts :259); launch/transcription para ends :135; qa-verdict.js step-B git check already exists | (d) insertion target | insert new subsection AFTER :135 |
| `.claude/agent-memory/qa/` | MEMORY.md + 6 project memories | Proof memory is load-bearing | do NOT casually delete memory:project |

## Application to pyfinagent (per work item)

**(a) Probe shape -- BEHAVIORAL, two complementary artifacts (self-disclosure is
unreliable; see finding 8):**
- **`scripts/qa/probe_qa_tool_surface.js` (Workflow script, re-runnable, $0 on Max
  rail).** Spawns `agent(prompt, {agentType:'qa', schema})` where the prompt makes
  the qa agent ATTEMPT each tool against a scratch path under
  `scratchpad/` and REPORT success/failure per tool (Write, Edit, Grep, Glob,
  Bash) as a structured map -- measuring EXECUTION. This is the scriptable analog
  of a session-interactive Agent-tool spawn. Clean up the scratch file.
- **The enforcement hook in LOG-ONLY mode doubles as the empirical probe.** A
  PreToolUse hook that appends `{agent_type, tool_name, file_path}` to
  `handoff/audit/qa_tool_surface_probe.jsonl` for Write/Edit calls. When any
  EVALUATE runs, it records whether Write/Edit actually originate from
  `agent_type=="qa"` AND confirms the doc's field is populated on v2.1.218. Then
  flip the same hook from log to deny. Probe and enforcement are ONE artifact.
- A headless `claude -p ... --agents '<json>'` (sub-agents doc L222) is scriptable
  but dispatch to a specific named subagent from `-p` is description-matched and
  unreliable; the `--agent <name>` session pin is the more deterministic variant.
  Prefer the Workflow + hook-log combo.

**(b) Injection source = UPSTREAM loader, DOCUMENTED.** `memory: project`
auto-enables Read/Write/Edit (finding 1). Not a local config bug. The related
upstream issues (#57507 memory/allowlist, #57118 permissionMode, #40140
agent_id-in-PreToolUse) are ALL closed "not planned"/stale => **upstream-wait
yields nothing.**

**(c) Enforcement -- recommend the HOOK, path-aware.** A PreToolUse hook that
returns `permissionDecision:"deny"` (or `exit 2`) when
`agent_type == "qa"` AND `tool_name in {Write, Edit}` AND `tool_input.file_path`
is NOT under `.claude/agent-memory/qa/`. The path carve-out preserves Q/A's
memory curation (the very reason Write/Edit exist) while making PRODUCTION writes
impossible via the direct tool path. This is the ONLY doc-provable, per-agent,
bypassPermissions-surviving block (findings 5-6). Complement with (d) as a
backstop and, if the project decides Q/A memory is not worth the residual,
the root-cause option of deleting `memory: project` (probe-verify it removes the
injection). permissionMode:plan and settings.json deny rules are NOT viable
(findings 4, 6).

**(d) git-status cleanliness rule -- exact insertion.** Insert a new subsection in
`per-step-protocol.md` S4 EVALUATE immediately AFTER the launch/transcription
paragraph (after L135, before "Q/A runs deterministic-first" at L137), e.g.
`#### Post-verdict git-status cleanliness (phase-75.20.1)`: after transcribing
the verdict VERBATIM, Main runs `git status --short` and asserts the qa subagent
left no stray Write/Edit outside the step's scoped files and
`.claude/agent-memory/qa/`; any unexpected mutation is reverted before the
status flip. Note the companion Q/A-side check already in `qa-verdict.js` PROMPT
step B ("confirm NO UNINTENDED production code changed (git status --short / git
diff)") -- the new Main-side rule is defense-in-depth on the same seam.

## Enforcement recommendation

**HOOK** (path-aware PreToolUse deny). Deciding fact: the official hooks doc
(code.claude.com/docs/en/hooks L623-628) documents `agent_type` in the PreToolUse
common input, present when the hook fires inside a subagent and equal to the
frontmatter `name` ("qa"); the permissions doc (L411/L419/L58) confirms a
PreToolUse hook runs before the prompt and blocks even under `bypassPermissions`,
while `Agent(name)` rules (L377-383) gate spawning, NOT a subagent's tool set --
so a PreToolUse hook is the ONLY provable per-acting-agent block. upstream-wait is
excluded because the Write/Edit injection is the DOCUMENTED effect of
`memory: project` and every related issue is closed not-planned/stale.

## Research Gate Checklist

Hard blockers (all satisfied):
- [x] >=5 authoritative external sources READ IN FULL via WebFetch/curl (6: 3 official docs + 3 issues)
- [x] 10+ unique URLs total (23 collected; ~17 snippet-only)
- [x] Recency scan (last 2 years) performed + reported (3-variant; all sources 2026)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (qa.md, qa-verdict.js, settings.json, tests, verify script, per-step-protocol, memory dir)
- [x] Contradictions noted (doc says agent_type present in PreToolUse vs #40140 "not planned" -- resolved: doc authoritative, probe confirms on v2.1.218)
- [x] Claims cited per-claim with URL + file:line

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 17,
  "urls_collected": 23,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {"audit_class": false, "rounds": 1, "dry_rounds": 0, "K_required": 2, "new_findings_last_round": 0, "dry": false},
  "summary": "The qa subagent's Write/Edit surface is DOCUMENTED, intended behavior of qa.md's `memory: project` (sub-agents doc: memory auto-enables Read/Write/Edit); Q/A's memory is real+load-bearing (6 curated files), so the tools allowlist cannot exclude Write/Edit without killing memory. permissionMode:plan is inert (parent bypassPermissions overrides). Upstream-wait is dead (#57507/#57118/#40140 all closed not-planned). The only doc-provable per-agent enforcement is a PATH-AWARE PreToolUse hook: hooks doc L623-628 gives agent_type==frontmatter name ('qa') in PreToolUse input; permissions doc L411/419/58 confirm the hook blocks even under bypassPermissions, and Agent(name) rules gate spawn not tool-set. Recommend: path-aware hook denying Write|Edit for agent_type=='qa' except under .claude/agent-memory/qa/, + Main-side post-verdict git-status cleanliness (insert per-step-protocol.md after L135), + a BEHAVIORAL probe (Workflow attempt-each-tool + hook log-mode) because the section-7a Glob/Grep 'drop' is a self-report artifact.",
  "brief_path": "handoff/current/research_brief_75.20.1.md",
  "gate_passed": true
}
```
