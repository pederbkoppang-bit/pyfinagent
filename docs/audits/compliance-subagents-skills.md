# Compliance: Sub-agents + Agent teams + Skills (phase-4.15.3)

Audit date: 2026-04-18
Researcher: merged researcher agent (external + internal in one session)

---

## URL coverage

| URL | Status |
|-----|--------|
| https://code.claude.com/docs/en/sub-agents | CHECKED |
| https://code.claude.com/docs/en/agent-teams | CHECKED |
| https://code.claude.com/docs/en/agent-sdk/subagents | CHECKED |
| https://code.claude.com/docs/en/skills | CHECKED |
| https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview | CHECKED |
| https://platform.claude.com/docs/en/agents-and-tools/agent-skills/quickstart | NOT READ (quickstart; redundant with overview) |
| https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices | CHECKED |
| https://platform.claude.com/docs/en/agents-and-tools/agent-skills/enterprise | CHECKED |

---

## Per-pattern compliance matrix

### Pattern 1: Sub-agent frontmatter — required fields

- **Doc source:** code.claude.com/docs/en/sub-agents (frontmatter schema)
- **Doc says:** Required fields are `name` and `description`. Optional: `tools`, `model`, `maxTurns`, `effort`, `memory`, `color`, `permissionMode`, `skills`, `mcpServers`.
- **Status:** Partial — both agents have all documented required fields; optional `permissionMode`, `skills`, `mcpServers` are absent.
- **Evidence:**
  - `.claude/agents/researcher.md` lines 2-10: has `name`, `description`, `tools`, `model`, `maxTurns`, `effort`, `memory`, `color`.
  - `.claude/agents/qa.md` lines 2-10: same set.
  - `grep permissionMode .claude/agents/*.md` — no matches.
  - `grep 'skills\b' .claude/agents/*.md` — no matches.
  - `grep mcpServers .claude/agents/*.md` — no matches.
- **Deviation:** `permissionMode` is not set on either agent. The project-level `settings.json` sets `defaultMode: bypassPermissions` globally, which flows to all subagents, but doc recommends explicit per-agent `permissionMode` for principle of least privilege.
- **Risk:** `bypassPermissions` is inherited by both agents. If researcher or qa can access destructive tools (Bash is listed in researcher's `tools`), there is no per-agent permission gate. A buggy researcher could write files.
- **Recommended fix:** Add `permissionMode: default` (or `restricted`) to `researcher.md` and `qa.md` frontmatter; move `bypassPermissions` only to steps where Main requires it.
- **MF-# mapping:** New finding (not in phase-4.10-4.13).

---

### Pattern 2: Trigger phrasing — MUST BE USED / use proactively

- **Doc source:** code.claude.com/docs/en/sub-agents — "Claude uses the description to decide when to delegate"
- **Doc says:** Descriptions should be clear so Claude knows when to delegate. The docs do not mandate specific phrasing, but the project's own convention (seen in both agent files) uses "MUST BE USED" and "use proactively".
- **Status:** Correct — both agents use the required trigger phrases.
- **Evidence:**
  - `grep -c 'MUST BE USED\|use proactively' .claude/agents/researcher.md .claude/agents/qa.md` — returns 1 match each (one combined in description line).
  - `researcher.md` line 3: "MUST BE USED before every PLAN phase … Use proactively at the start of any masterplan step".
  - `qa.md` line 3: "MUST BE USED in every EVALUATE phase … Use proactively after any GENERATE step".
- **Deviation:** None.
- **Risk:** None.
- **Recommended fix:** N/A.

---

### Pattern 3: Tool allowlist least-privilege

- **Doc source:** code.claude.com/docs/en/agent-sdk/subagents — "tools restricts what the subagent can do (read-only here)"
- **Doc says:** Specify `tools` to limit subagent access; read-only agents should list only `Read`, `Grep`, `Glob`; agents that must not modify should exclude `Write`, `Edit`, `Bash`.
- **Status:** Partial — `qa.md` specifies `tools: Read, Bash, Glob, Grep` which includes Bash despite being declared "NEVER modify files" in its body. `researcher.md` specifies `tools: Read, Grep, Glob, Bash, WebSearch, WebFetch` which is reasonable for its exploration role.
- **Evidence:**
  - `qa.md` line 5: `tools: Read, Bash, Glob, Grep`.
  - `qa.md` line 150: "NEVER modify files. Read-only tools only."
  - Contradiction: Bash is in `tools` allowlist but the body says read-only. Bash is used only for `source .venv/bin/activate && <cmd>` and dry-run, but it could in principle write.
- **Deviation:** `qa.md` grants Bash but claims read-only intent. The allowlist should be scoped to `Bash(python *)` or narrower patterns to enforce the invariant.
- **Risk:** qa agent could accidentally write files or run destructive commands if its prompt is manipulated or hallucinates a bash write.
- **Recommended fix:** Restrict `tools` in `qa.md` to `Read, Glob, Grep, Bash(python *), Bash(source *), Bash(pytest *)` or use `permissionMode: restricted` with explicit allow rules.
- **MF-# mapping:** New finding.

---

### Pattern 4: Model aliases

- **Doc source:** code.claude.com/docs/en/agent-sdk/subagents — `model: 'sonnet' | 'opus' | 'haiku' | 'inherit'`
- **Doc says:** Use model aliases (not full model IDs) in frontmatter.
- **Status:** Correct.
- **Evidence:**
  - `researcher.md` line 7: `model: sonnet`.
  - `qa.md` line 7: `model: opus`.
  - Both use documented alias strings.
- **Deviation:** None.
- **Risk:** None.
- **Recommended fix:** N/A.

---

### Pattern 5: maxTurns, effort, memory fields

- **Doc source:** code.claude.com/docs/en/sub-agents (optional frontmatter fields)
- **Doc says:** `maxTurns` caps turns; `effort` sets thinking level; `memory` selects memory source.
- **Status:** Correct — all three present on both agents.
- **Evidence:**
  - `researcher.md`: `maxTurns: 20`, `effort: medium`, `memory: project`.
  - `qa.md`: `maxTurns: 12`, `effort: medium`, `memory: project`.
  - Values are deliberate and match the agents' workload (researcher heavier, qa faster).
- **Deviation:** None.
- **Risk:** None.
- **Recommended fix:** N/A.

---

### Pattern 6: permissionMode field (explicit)

- **Doc source:** code.claude.com/docs/en/sub-agents — `permissionMode` optional field
- **Doc says:** Set per-agent `permissionMode` to override global setting; enables per-agent least-privilege.
- **Status:** Missing — neither agent sets `permissionMode`.
- **Evidence:** `grep permissionMode .claude/agents/*.md` — no output. `settings.json` line 78: `"defaultMode": "bypassPermissions"`.
- **Deviation:** Global `bypassPermissions` flows to all agents unconstrained.
- **Risk:** Medium — researcher and qa inherit dangerously permissive mode.
- **Recommended fix:** Add `permissionMode: default` to both agent files; reserve `bypassPermissions` only for Main session.
- **MF-# mapping:** New finding (phase-4.10 did not flag permissionMode).

---

### Pattern 7: SubagentStop hook enforcement

- **Doc source:** code.claude.com/docs/en/sub-agents (hooks section) — `SubagentStop` hook runs when a subagent is about to stop; exit code 2 sends feedback and keeps it running.
- **Doc says:** Use `SubagentStop` to enforce quality gates before a subagent exits.
- **Status:** Missing — no `SubagentStop` hook is configured anywhere in `settings.json` or agent frontmatter.
- **Evidence:** `jq '.hooks | keys' .claude/settings.json` returns `["PostToolUse","Stop","TaskCompleted","TeammateIdle"]` — no `SubagentStop`.
- **Deviation:** No per-subagent stop gate. The `Stop` hook covers the main session but does not apply to spawned subagents.
- **Risk:** A researcher or qa agent could complete and return without having produced a complete research gate checklist or verification JSON. The Stop hook only catches the orchestrator session.
- **Recommended fix:** Add a `SubagentStop` hook in `settings.json` that checks for gate completeness (e.g., confirms `gate_passed` field in researcher output, `ok` field in qa output) before allowing exit.
- **MF-# mapping:** New finding. Phase-4.10 flagged the Stop hook; SubagentStop is distinct.

---

### Pattern 8: Agent teams — enable flag

- **Doc source:** code.claude.com/docs/en/agent-teams — "Enable by setting CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1"
- **Doc says:** Disabled by default; must be explicitly enabled in settings.json env block.
- **Status:** Correct — flag is set.
- **Evidence:** `jq '.env.CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS' .claude/settings.json` returns `"1"`.
- **Deviation:** None on the enable step. See Pattern 9 for the reachability concern.
- **Risk:** None from the flag itself.
- **Recommended fix:** N/A.

---

### Pattern 9: Agent teams — team-lead / teammate split

- **Doc source:** code.claude.com/docs/en/agent-teams — "One session is the lead; teammates work independently"
- **Doc says:** The lead creates the team, assigns tasks, synthesizes results. Teammates have their own context windows and communicate via mailbox and shared task list. Only the lead manages team lifecycle (clean up).
- **Status:** N/A in practice — our 3-agent MAS uses sub-agents (sequential spawn-and-return), not agent teams. The flag is enabled but no team-lead/teammate pattern has been coded.
- **Evidence:**
  - `.claude/agents/researcher.md` body: instructs researcher to run external + internal work in one session and return a text report. That is a subagent pattern, not a teammate.
  - `settings.json` has `TeammateIdle` hook (Pattern 12) but the hook triggers on subagent idle, not a true teammate idle from a separate team session.
  - No `~/.claude/teams/` directory observed.
- **Deviation:** `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` is set but no code path creates a team. The hook wiring references teammate semantics without instantiating a real team.
- **Risk:** Dead flag. Adds token overhead from the experimental flag and may confuse future maintainers who see `TeammateIdle` wired up without a team structure. Doc warns the flag has "known limitations around session resumption, task coordination, and shutdown behavior".
- **Recommended fix:** Either remove the flag and `TeammateIdle` hook (and rename the hook handler to a generic post-agent check), or document the intent to convert the MAS to a proper team lead/teammate topology.
- **MF-# mapping:** Phase-4.10 originally flagged as "unreachable". Confirmed still present.

---

### Pattern 10: Agent teams — mailbox and shared task list

- **Doc source:** code.claude.com/docs/en/agent-teams — Architecture table: Team config at `~/.claude/teams/`, task list at `~/.claude/tasks/`, mailbox for inter-agent messages
- **Doc says:** Teammates share a task list, claim work, and message each other directly. Team config is auto-generated; do not pre-author it.
- **Status:** N/A — no team instantiated; no mailbox or shared task list.
- **Evidence:** `ls ~/.claude/teams/` — directory does not exist. Task management is done via `handoff/current/` files (file-based, not the agent teams task protocol).
- **Deviation:** Our task coordination uses CLAUDE.md-documented handoff files. That is a valid sub-agent pattern but is not the agent teams mailbox.
- **Risk:** None, since teams are not used. If teams are adopted later, the handoff protocol must be reconciled with the task list API.
- **Recommended fix:** No action needed unless teams are adopted. Document the decision in CLAUDE.md.

---

### Pattern 11: Agent teams — TeammateIdle hook

- **Doc source:** code.claude.com/docs/en/agent-teams — "`TeammateIdle`: runs when a teammate is about to go idle. Exit code 2 to send feedback and keep working."
- **Doc says:** The hook enforces quality gates when a teammate finishes its current task.
- **Status:** Incorrect use — the hook is wired but it runs `teammate-idle-check.sh` which checks masterplan for active steps, not teammate quality gates.
- **Evidence:**
  - `settings.json` lines 49-58: `TeammateIdle` hook runs `bash .claude/hooks/teammate-idle-check.sh`.
  - Our MAS is not a team; there are no real teammates. The hook fires when any subagent-like idle event occurs.
  - The shell script is named as a teammate check but its purpose (check masterplan active steps) is closer to a Stop hook concern.
- **Deviation:** The hook name implies team coordination; the implementation is a masterplan state check. This is semantic confusion, not an active bug, but misleads future auditors.
- **Risk:** Low today. If agent teams are ever adopted, the `TeammateIdle` hook will conflict with or shadow the real teammate-idle intent.
- **Recommended fix:** Rename the hook or add a comment clarifying it is repurposed from the agent-teams event into a general idle checkpoint.

---

### Pattern 12: Agent teams — spawn pattern and best practices

- **Doc source:** code.claude.com/docs/en/agent-teams — "Start with 3-5 teammates; 5-6 tasks per teammate; avoid same-file edits; give context in spawn prompt"
- **Doc says:** Teams work best for parallel exploration; lead assigns tasks; teammates self-claim from shared list.
- **Status:** N/A — no team used.
- **Evidence:** See Pattern 9-10.
- **Deviation:** N/A.
- **Risk:** N/A.

---

### Pattern 13: Subagent definitions used as teammates

- **Doc source:** code.claude.com/docs/en/agent-teams — "When spawning a teammate, reference a subagent type. The teammate honors that definition's `tools` allowlist and `model`. Note: `skills` and `mcpServers` frontmatter fields are NOT applied when the definition runs as a teammate."
- **Doc says:** Subagent `skills` and `mcpServers` fields are silently dropped when the agent runs as a teammate.
- **Status:** N/A currently (not using teams). However, both agent files omit `skills` and `mcpServers`, so no data would be dropped even if teammates were adopted.
- **Evidence:** `grep -n 'skills\b\|mcpServers' .claude/agents/*.md` — no matches.
- **Deviation:** None (omission is neutral, not incorrect).
- **Risk:** If teams are adopted and `skills` are later added to agent definitions, the teammate silently drops them. This is a latent risk to document.
- **Recommended fix:** Note in agent file comments that `skills` and `mcpServers` do not apply in teammate mode.

---

### Pattern 14: Claude Code Skills — SKILL.md shape and directory

- **Doc source:** code.claude.com/docs/en/skills — "Each skill is a directory with SKILL.md as the entrypoint. Personal: `~/.claude/skills/`; Project: `.claude/skills/`."
- **Doc says:** Skills live in a named directory; SKILL.md is required; supporting files (scripts, reference docs) are optional; skills are auto-discovered.
- **Status:** Missing at both levels — no `.claude/skills/` project directory and no `~/.claude/skills/` user directory exist.
- **Evidence:**
  - `ls .claude/skills/ 2>/dev/null` — "NO PROJECT SKILLS DIR".
  - `ls ~/.claude/skills/ 2>/dev/null` — "NO USER SKILLS DIR".
  - `.claude/agents/` contains only `researcher.md` and `qa.md`; no SKILL.md present.
- **Deviation:** We have no Claude Code Skills registered. Our `backend/agents/skills/*.md` (28 files) are in-app prompt templates — a wholly different concept (see Pattern 16).
- **Risk:** If we ever want Claude Code to auto-load domain knowledge (e.g., the pyfinagent backtest protocol, DSR formula reference), we have no skill infrastructure in place.
- **Recommended fix:** No immediate action needed since harness context is delivered via CLAUDE.md. If in-session auto-loading of large reference blobs is desired, create one or more skills under `.claude/skills/`.

---

### Pattern 15: Claude Code Skills — frontmatter fields

- **Doc source:** code.claude.com/docs/en/skills — frontmatter reference table
- **Doc says:** Optional fields include `name`, `description` (recommended), `disable-model-invocation`, `user-invocable`, `allowed-tools`, `model`, `effort`, `context`, `agent`, `hooks`, `paths`, `when_to_use`, `argument-hint`, `shell`. `description` + `when_to_use` capped at 1,536 chars in skill listing.
- **Status:** N/A — no project skills exist.
- **Evidence:** See Pattern 14.
- **Deviation:** N/A.
- **Risk:** N/A (latent: if skills are created, authors must know these fields).
- **Recommended fix:** If skills are created, reference this table; do not use Platform Agent Skills frontmatter (`name` must be lowercase letters/numbers/hyphens, max 64 chars, no XML, no reserved words).

---

### Pattern 16: Skills terminology collision

- **Doc source:** platform.claude.com/docs/en/agents-and-tools/agent-skills/overview — "Skills are modular capabilities that extend Claude's functionality via SKILL.md files with YAML frontmatter."
- **Doc says:** "Agent Skills" = filesystem SKILL.md files loaded by Claude Code or the API. They are NOT prompt templates.
- **Status:** Incorrect terminology in codebase — `backend/agents/skills/*.md` are labeled "skills" but are Jinja-style prompt templates, not Agent Skills.
- **Evidence:**
  - `ls backend/agents/skills/*.md` — 28 files, no YAML frontmatter (`grep -l 'name:\|description:' backend/agents/skills/*.md` returns 0).
  - `grep -c '{{' backend/agents/skills/*.md | awk '{t+=$2} END{print t}'` — 252 variable placeholders.
  - `backend/agents/skills/SKILL_TEMPLATE.md` uses `# {Agent Name}` notation — a fill-in template, not an Agent Skill.
  - Per `.claude/rules/backend-agents.md`: "Agent prompts in `skills/*.md`, loaded via `load_skill()` + `format_skill()` with `{{variable}}` placeholders."
- **Deviation:** The 28 files are prompt templates loaded at runtime by Python. They are unrelated to Anthropic Agent Skills and will confuse any auditor or new contributor who looks at the directory name.
- **Risk:** Go-live reviewers and new engineers will conflate the two systems. The directory name `backend/agents/skills/` looks like Agent Skills when it is not.
- **Recommended fix:** Rename `backend/agents/skills/` to `backend/agents/prompts/` and update all references in `load_skill()`, `format_skill()`, `backend/agents/orchestrator.py`, and `backend/agents/skill_optimizer.py`. Update CLAUDE.md and `backend-agents.md` rule accordingly. Do NOT implement — audit only.
- **MF-# mapping:** Phase-4.12 flagged this collision. Still present.

---

### Pattern 17: Platform Agent Skills — YAML frontmatter `name` and `description` constraints

- **Doc source:** platform.claude.com/docs/en/agents-and-tools/agent-skills/overview#skill-structure
- **Doc says:** `name` max 64 chars, lowercase letters/numbers/hyphens only, no XML, no reserved words ("anthropic", "claude"). `description` max 1,024 chars, non-empty, no XML.
- **Status:** N/A — no Platform Agent Skills registered. Our subagent `.md` frontmatter uses different fields (it is a Claude Code sub-agent, not a Platform Skill).
- **Evidence:** Confirmed above.
- **Deviation:** N/A now. But if someone tries to register our `researcher.md` as a Platform Skill, `model: sonnet` and other Claude-Code-specific fields would be rejected or ignored.
- **Risk:** Latent cross-contamination if frontmatter conventions are mixed.
- **Recommended fix:** Keep the two naming systems in separate documentation sections. Do not mix Platform Skill frontmatter with Claude Code subagent frontmatter.

---

### Pattern 18: Skills best practices — SKILL.md body under 500 lines

- **Doc source:** platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices — "Keep SKILL.md body under 500 lines for optimal performance."
- **Doc says:** If content exceeds 500 lines, split into supporting files using progressive disclosure.
- **Status:** N/A for Agent Skills (none registered). For reference: `backend/agents/skills/synthesis_agent.md` is 179 lines; total across all 28 files is 2,859 lines (average 102 lines each). Were any of these converted to Agent Skills, all would be under the 500-line limit.
- **Evidence:** `wc -l backend/agents/skills/*.md | tail -1` — 2,859 total.
- **Deviation:** N/A (these are not Agent Skills).
- **Risk:** N/A.

---

### Pattern 19: Skills best practices — match freedom to fragility

- **Doc source:** platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices — "Match the level of specificity to the task's fragility and variability."
- **Doc says:** High-freedom (text-based guidance) for multi-path tasks; low-freedom (exact script/command) for fragile operations.
- **Status:** Applied correctly in subagent files (different concept but same principle).
- **Evidence:**
  - `researcher.md`: high freedom — prose instructions with a numbered protocol. Correct, as research has many valid paths.
  - `qa.md`: low freedom — exact JSON output schema, `violation_type` must be one of SAVeR taxonomy, exact field names. Correct, as verification must be deterministic.
- **Deviation:** None.
- **Risk:** None.

---

### Pattern 20: Skills best practices — third-person description

- **Doc source:** platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices — "Always write description in third person. Good: 'Processes Excel files'. Avoid: 'I can help you process'."
- **Doc says:** Third-person descriptions avoid system prompt point-of-view conflicts.
- **Status:** Partial — both agent descriptions use second-person imperative ("MUST BE USED") rather than third-person declarative.
- **Evidence:**
  - `researcher.md` line 3: "MUST BE USED before every PLAN phase" — imperative/second-person.
  - `qa.md` line 3: "MUST BE USED in every EVALUATE phase" — same.
  - The doc guideline applies to Platform Agent Skills specifically. For Claude Code subagent descriptions the convention is less strict, but the imperative phrasing is non-standard.
- **Deviation:** Descriptions use first-person imperative, not third-person declarative.
- **Risk:** Low for Claude Code subagents. If these files are ever used as Platform Agent Skills, descriptions would be non-compliant.
- **Recommended fix:** Optionally rephrase: "Performs external literature research and internal codebase exploration. Invoked before every PLAN phase." Preserves the mandate without imperative phrasing.

---

### Pattern 21: Skills best practices — cross-model evaluation

- **Doc source:** platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices — "Test with all models you plan to use it with (Haiku, Sonnet, Opus)."
- **Doc says:** Skill effectiveness varies by model; evaluate across the full model set before deployment.
- **Status:** Partial — agent model assignments are fixed (researcher=sonnet, qa=opus) but no documented cross-model evaluation exists for the agent prompts.
- **Evidence:** No `evals/` directory for agent prompts. The harness cycles stress-test indirectly via PASS/FAIL verdicts, but this is not a controlled cross-model evaluation.
- **Deviation:** No formal eval suite for researcher/qa prompts across model tiers.
- **Risk:** Prompt drift — a sonnet-tuned researcher prompt may underperform if the model assignment is changed.
- **Recommended fix:** Create 3-5 eval queries per agent in a `tests/agent_evals/` directory, run them against at least sonnet + opus, and gate model changes on eval PASS.

---

### Pattern 22: Enterprise Skills — risk tier assessment

- **Doc source:** platform.claude.com/docs/en/agents-and-tools/agent-skills/enterprise#risk-tier-assessment
- **Doc says:** Evaluate each skill against risk indicators: code execution scripts, instruction manipulation, MCP server references, network access patterns, hardcoded credentials, filesystem access scope, tool invocations.
- **Status:** Not formally applied — no security review checklist for subagent files exists in the repo.
- **Evidence:** No `docs/audits/subagent-security-review.md` or equivalent. The `backend/agents/skills/*.md` prompt templates are not audited for adversarial instruction patterns.
- **Deviation:** Enterprise review checklist has not been run on our two subagent definitions.
- **Risk:** Medium. `researcher.md` grants Bash + WebFetch/WebSearch — two High-risk indicators (code execution + network access). No review confirms the prompt does not have instruction-manipulation patterns.
- **Recommended fix:** Apply the enterprise review checklist to `researcher.md` and `qa.md` before go-live: read all files, verify script behavior (N/A here), check for adversarial instructions, confirm no hardcoded credentials, list all tool invocations and their risk.

---

### Pattern 23: Enterprise Skills — separation of duties

- **Doc source:** platform.claude.com/docs/en/agents-and-tools/agent-skills/enterprise — "Skill authors should not be their own reviewers. Establish separation of duties."
- **Doc says:** The person who writes a Skill must not be the one who approves it for production deployment.
- **Status:** Violated — the harness orchestrator (Main) both writes agent definitions and self-approves them via masterplan step transitions.
- **Evidence:** The harness protocol in CLAUDE.md states that the qa agent evaluates the work, but the qa agent's definition was authored by the same session that writes masterplan steps. There is no external reviewer gate for changes to `.claude/agents/*.md` files.
- **Deviation:** Agent definition changes bypass a separation-of-duties gate.
- **Risk:** A malicious or buggy change to `qa.md` could weaken the verification gate without detection.
- **Recommended fix:** Require that changes to `.claude/agents/*.md` files be reviewed in a GitHub PR with Peder's approval before merge, separate from the harness PASS/FAIL cycle.
- **MF-# mapping:** New finding; not in prior audits.

---

### Pattern 24: Enterprise Skills — 8-skill cap per API request

- **Doc source:** platform.claude.com/docs/en/agents-and-tools/agent-skills/enterprise#recall-limits — "API requests support a maximum of 8 Skills per request."
- **Doc says:** Limit simultaneous Skills to maintain recall accuracy. If a role needs more, consolidate or route to different Skill sets.
- **Status:** N/A — we use zero Platform Agent Skills. Subagents are not subject to this cap.
- **Evidence:** No Platform Skills registered; cap does not apply.
- **Deviation:** N/A.
- **Risk:** Latent. If Platform Skills are adopted in phase-4.15.14 (Managed Agents Skills), the 8-skill cap applies.
- **Recommended fix:** Note the cap as a constraint in any future Skills adoption plan.

---

### Pattern 25: TaskCompleted hook — type: agent vs type: command

- **Doc source:** code.claude.com/docs/en/agent-teams — `TaskCompleted` hook; agent hooks run a new Claude instance; command hooks run a shell command.
- **Doc says:** `type: agent` in `TaskCompleted` spawns a verification agent asynchronously via `asyncRewake`.
- **Status:** Correct in schema; potential over-triggering concern.
- **Evidence:**
  - `settings.json` lines 37-46: `TaskCompleted` hook has `type: agent`, `timeout: 60`, `asyncRewake: true`.
  - The hook prompt asks Claude to verify the masterplan step, which is functionally the qa agent's job. This duplicates verification: qa is already spawned by Main, and this hook fires again on task completion.
- **Deviation:** Double-verification: qa agent is spawned explicitly by Main during EVALUATE phase, AND the TaskCompleted hook spawns another verification agent automatically. This creates redundant token spend and potential race conditions if both return verdicts simultaneously.
- **Risk:** Token cost inflation; potential for conflicting PASS/FAIL verdicts from two concurrent agents.
- **Recommended fix:** Either remove the `TaskCompleted` hook agent (rely solely on Main spawning qa explicitly) or repurpose it as a lightweight `type: command` that just logs the step completion without invoking a full Claude agent.
- **MF-# mapping:** New finding.

---

## Novel findings (not in phase-4.10-4.13)

1. **permissionMode not set per-agent (Pattern 1, 6):** Global `bypassPermissions` flows unconstrained to both researcher and qa. Prior audits flagged the overall permission mode; this cycle identifies the per-agent missing field specifically.

2. **Bash in qa tools vs read-only claim (Pattern 3):** `qa.md` contradicts itself: body says "NEVER modify files, read-only tools only" but `tools` list includes unrestricted `Bash`. Prior audits did not inspect the tools list vs body contradiction.

3. **SubagentStop hook entirely absent (Pattern 7):** `Stop` and `TaskCompleted` hooks are wired, but `SubagentStop` (which gates each subagent before it exits) is missing. Prior audits reviewed Stop; SubagentStop is a distinct hook.

4. **TaskCompleted agent hook duplicates qa spawn (Pattern 25):** Two Claude agents (qa explicit spawn + TaskCompleted hook agent) both verify the same step. This is novel and creates token waste and potential verdict conflicts.

5. **Separation of duties not enforced on agent definitions (Pattern 23):** Changes to `.claude/agents/*.md` bypass any external review. Not flagged before because prior phases did not audit the agent lifecycle governance itself.

---

## Agent-teams flag: kill or commit?

**Recommendation: remove the flag and the TeammateIdle hook.**

Evidence:
- `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` is set but no team has ever been instantiated (no `~/.claude/teams/` directory, no team-lead pattern in code).
- The doc explicitly labels this "experimental" with known limitations (no session resumption, task status lag, one team per session).
- Our 3-agent MAS (Main + Researcher + QA) is a subagent topology, which is the correct pattern for our sequential Plan → Generate → Evaluate loop. Agent teams are for parallel exploration with inter-agent communication.
- The `TeammateIdle` hook is repurposed as a masterplan check — that semantic mismatch will accumulate tech debt.
- Stress-test doctrine (CLAUDE.md): "Every component encodes an assumption about what the model can't do on its own … Stale scaffolding is dead weight — prune it."

The flag costs tokens (experimental features load additional coordination metadata) and adds confusion. Remove it unless a concrete team topology is designed (e.g., phase-4.15.x adopts parallel fact-checking teammates).

---

## Skills terminology collision: recommended rename

The 28 files in `backend/agents/skills/*.md` must not be called "skills." They are Jinja-style prompt templates loaded by `load_skill()` / `format_skill()` into the 15-step Gemini pipeline. The term "skill" now has a precise Anthropic-platform meaning (filesystem SKILL.md + YAML frontmatter + progressive disclosure loading) that these files do not implement.

**Recommended rename path (audit only, do not implement):**

| Current | Renamed |
|---------|---------|
| `backend/agents/skills/` | `backend/agents/prompts/` |
| `load_skill()` | `load_prompt()` |
| `format_skill()` | `format_prompt()` |
| `SkillOptimizer` class | `PromptOptimizer` |
| `.claude/rules/backend-agents.md` "Skills System" section | "Prompts System" |

This rename removes ambiguity before go-live and before any Platform Agent Skills are adopted in later phases.

---

## References

- code.claude.com/docs/en/sub-agents (accessed 2026-04-18)
- code.claude.com/docs/en/agent-teams (accessed 2026-04-18)
- code.claude.com/docs/en/agent-sdk/subagents (accessed 2026-04-18)
- code.claude.com/docs/en/skills (accessed 2026-04-18)
- platform.claude.com/docs/en/agents-and-tools/agent-skills/overview (accessed 2026-04-18)
- platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices (accessed 2026-04-18)
- platform.claude.com/docs/en/agents-and-tools/agent-skills/enterprise (accessed 2026-04-18)
- `.claude/agents/researcher.md` (post-4.15.0 merged agent)
- `.claude/agents/qa.md` (post-4.15.0 merged agent)
- `.claude/settings.json` (hooks, env, permissions)
- `.claude/settings.local.json` (MCP, temp allow rules)
- `backend/agents/agent_definitions.py` (AGENT_CONFIGS Layer-2 MAS)
- `backend/agents/skills/*.md` (28 prompt templates)
- `docs/runbooks/per-step-protocol.md` (post-4.15.0 runbook)
