# Contract -- 67.5 Claude Code feature adoption (fallback chain, revert tripwire, doc truth, MCP approval fix)

Step: masterplan phase-67 / 67.5 (P1). Research gate: PASSED (moderate;
research_brief_67_5.md -- 6 official sources read in full, 26 URLs, recency scan,
all internal claims anchored).

## Research-gate summary

- EMPIRICAL FIND (criterion 4): `claude mcp list` from the repo cwd shows ALL project
  .mcp.json servers "Pending approval" (bigquery, paper-search-mcp, 4x pyfinagent-*,
  playwright; alpaca rejected via settings.local.json). ~/.claude.json project entry:
  hasTrustDialogAccepted:true but enabledMcpjsonServers:[] -- no approvals recorded.
  Away/headless sessions (launchd -> scripts/away_ops/run_away_session.sh:160-165,
  `claude -p --dangerously-skip-permissions ... --output-format json`, same HOME)
  currently attach ZERO pyfinagent stdio servers. v2.1.196 honors approvals from the
  UNTRACKED settings.local.json; the committed settings.json is exactly what it
  distrusts. CORRECTIONS from the audit prompt: .mcp.json has 8 servers (not 9);
  settings.local.json enabledMcpjsonServers=["slack"] is STALE (no such server).
- fallbackModel (criterion 1): array of strings, max 3 after dedup, no cross-file
  merge; goes in PROJECT settings.json (a local-settings copy would shadow it).
  Fable is the PRIMARY (saved default + agent pins) so it must NOT be in the chain.
  HONEST CAVEAT: triggers on overload/unavailable/non-retryable-5xx ONLY -- docs:
  rate-limit errors "never trigger a switch". So it insures the overload-class
  failure mode; usage-limit cutoffs are instead mitigated by v2.1.199 partial-work
  retention. Do not oversell in docs.
- SessionStart (criterion 2): matchers startup/resume/clear/compact; command hooks
  emit {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext":
  "..."}}; plain stdout ALSO becomes context (double fail-open); SessionStart cannot
  block startup by design. Grep target verified: `model: fable` at researcher.md:5 +
  qa.md:5.
- Doc refresh (criterion 3): stale "local 2.1.172" at researcher.md:27 + qa.md:28
  (only 2 hits repo-wide in .claude/agents/); researcher.md deep-tier fork section
  (~:241-250) reword for subagent self-nesting (v2.1.172+, 5 levels) -- NOTE: doc
  wording only; the researcher's tools list does NOT include Agent, so self-forking
  requires the caller to grant it per-spawn -- the reword must say that, not imply a
  standing capability. Runbook gains a subagent-runtime-semantics note (~:264).
  Both legacy agent-memory dirs confirmed empty -- rmdir safe.

## Hypothesis (falsifiable)

Recording MCP approvals in the untracked local settings restores stdio-server attach
for headless sessions (provable by `claude mcp list` before/after); a SessionStart
tripwire makes the 67.4 revert self-enforcing (provable by a dry-run under a faked
date emitting the warning); the fallback chain + doc refresh close the remaining
adoption gaps without touching the 3-agent doctrine.

## Success criteria (verbatim from .claude/masterplan.json 67.5 -- IMMUTABLE)

1. ".claude/settings.json carries a fallbackModel chain (fable-first falling to
   opus-4-8), shape verified against the running Claude Code version's documentation
   at research-gate time"
2. "A SessionStart hook injects a loud warning (additionalContext) when the session
   date is on/after 2026-07-12 AND either agent file still pins model: fable, naming
   masterplan 67.4 as the top P0; hook is fail-open like every hook in this project"
3. "Runbook + researcher.md reflect current subagent runtime semantics
   (background-by-default, partial-work retention, 5-level self-nesting for the
   deep-tier fork); stale local-version comments refreshed to the running version;
   the two empty legacy agent-memory dirs are removed"
4. "Headless MCP attach verified: a fresh non-interactive session lists the
   pyfinagent stdio servers, or the v2.1.196 approval posture is documented and
   fixed via enabledMcpjsonServers so away sessions do not silently lose servers"
5. "Fresh Q/A PASS on the diff; harness stays exactly 3 agents (no doctrine
   violation)"

Criterion-1 note (honest interpretation, decided at PLAN time): "fable-first falling
to opus-4-8" describes the SESSION's model order -- fable is the primary model (saved
default + pins), so the fallbackModel VALUE is the fall-to chain
["claude-opus-4-8", "claude-sonnet-5"]; putting fable inside the chain itself would
be wrong (it is the model being fallen FROM). The research-gate doc verification is
recorded in the brief.

## Design

1. `.claude/settings.json`: add `"fallbackModel": ["claude-opus-4-8",
   "claude-sonnet-5"]` (project file; NOT settings.local.json which would shadow);
   add the SessionStart hook entry (house pattern: command type, ${CLAUDE_PROJECT_DIR}
   resolution, statusMessage).
2. NEW `.claude/hooks/session-start-fable-tripwire.sh` (~26 lines): date >=
   2026-07-12 (TRIPWIRE_FAKE_TODAY env knob for the dry-run) AND `grep -q
   "^model: fable"` in either agent file -> emit hookSpecificOutput.additionalContext
   naming 67.4 top P0; all error paths exit 0 silently (fail-open; SessionStart
   cannot block startup anyway).
3. `.claude/settings.local.json`: enabledMcpjsonServers = the 7 non-alpaca server
   names (bigquery, paper-search-mcp, pyfinagent-backtest, pyfinagent-data,
   pyfinagent-risk, pyfinagent-signals, playwright); drop the stale "slack" entry;
   alpaca stays in disabledMcpjsonServers. (Untracked file -- the approval surface
   v2.1.196 honors.)
4. Doc refresh: researcher.md:27 + qa.md:28 "local 2.1.172" -> "local 2.1.205";
   researcher.md deep-tier fork section reword (self-nesting v2.1.172+, 5-level cap,
   caller must grant the Agent tool per-spawn); per-step-protocol.md new short
   "Subagent runtime semantics (v2.1.198+)" note: background-by-default spawns,
   partial-work retention on rate-limit/error cutoffs, fallbackModel covers
   overload-class only. rmdir .claude/agent-memory/{harness-verifier,qa-evaluator}.
5. live_check_67.5.md: (a) tripwire dry-run TRIPWIRE_FAKE_TODAY=2026-07-13 output
   verbatim; (b) `claude mcp list` BEFORE (pending approval) and AFTER (connected)
   -- the headless-equivalent approval surface; (c) Q/A verdict JSON.

## Anti-patterns guarded

- Doctrine: no new agents, no tool-list changes (fork reword is caller-grants wording).
- Shadowing: fallbackModel only in project settings.json (local would outrank).
- Overselling: fallback chain documented as overload-class insurance, NOT a stall/
  rate-limit cure (the brief's verbatim doc quote).
- Fail-open: tripwire exits 0 on every error path; SessionStart cannot block.
- Editing agent files this session changes NOTHING for this session (roster
  snapshot); noted for the log.

## Out of scope

Adding the Agent tool to researcher/qa tool lists; any Workflow adoption; fast mode;
MessageDisplay/statusline/output styles; the alpaca server posture (stays rejected).

## Risk

- v2.1.196 approval semantics could differ headless vs interactive -> criterion-4
  wording covers both branches (verify OR document+fix); live_check records the
  actual `claude mcp list` state transition.
- fallbackModel chain includes sonnet-5 (never yet run in-app) -> safe: Claude Code
  rail only (Max), and 67.6 already de-mined sonnet-5 request shapes on the API rail.
- A future Claude Code version changing hook JSON shape -> tripwire ALSO prints a
  plain-stdout line (doubles as context per docs; survives schema drift).
