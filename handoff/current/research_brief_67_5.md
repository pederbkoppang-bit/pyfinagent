# Research Brief — phase-67.5 (Claude Code v2.1.205 runtime adoption)

Tier: moderate (caller-specified). Status: IN PROGRESS — written incrementally.
Started: 2026-07-09.

## Question

Adopt Claude Code v2.1.205 runtime features: (1) `fallbackModel` in settings.json
(v2.1.166, up to 3 fallbacks); (2) `SessionStart` hook (v2.1.152) as Sunday-revert
tripwire (warn if date >= 2026-07-12 AND `model: fable` still pinned in
.claude/agents/{researcher,qa}.md, naming masterplan 67.4 top P0); (3) doc refresh
for new runtime semantics; (4) headless MCP attach verification (v2.1.196 approval
requirement for .mcp.json servers).

## Immutable success criteria (verbatim from .claude/masterplan.json phase-67 step 67.5)

1. ".claude/settings.json carries a fallbackModel chain (fable-first falling to opus-4-8), shape verified against the running Claude Code version's documentation at research-gate time"
2. "A SessionStart hook injects a loud warning (additionalContext) when the session date is on/after 2026-07-12 AND either agent file still pins model: fable, naming masterplan 67.4 as the top P0; hook is fail-open like every hook in this project"
3. "Runbook + researcher.md reflect current subagent runtime semantics (background-by-default, partial-work retention, 5-level self-nesting for the deep-tier fork); stale local-version comments refreshed to the running version; the two empty legacy agent-memory dirs are removed"
4. "Headless MCP attach verified: a fresh non-interactive session lists the pyfinagent stdio servers, or the v2.1.196 approval posture is documented and fixed via enabledMcpjsonServers so away sessions do not silently lose servers"
5. "Fresh Q/A PASS on the diff; harness stays exactly 3 agents (no doctrine violation)"

Verification command (immutable):
`bash -c 'grep -q "fallbackModel" .claude/settings.json && grep -q "SessionStart" .claude/settings.json && test -f .claude/hooks/session-start-fable-tripwire.sh && ! grep -rq "local 2.1.172" .claude/agents/ && test ! -d .claude/agent-memory/harness-verifier && test ! -d .claude/agent-memory/qa-evaluator'`

live_check: "live_check_67.5.md with (a) tripwire dry-run output under a faked date >= 2026-07-12 showing the warning text verbatim, (b) the headless MCP attach listing, (c) the fresh Q/A verdict JSON"

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://code.claude.com/docs/en/settings.md | 2026-07-09 | official doc | WebFetch full | `fallbackModel` = ARRAY of strings, "Chains are capped at three models; extra entries are ignored"; "the highest-precedence file that defines it supplies the entire chain" (no merge across files); CLI `--fallback-model`; `"default"` expands to default model. `enabledMcpjsonServers` = "List of specific MCP servers from .mcp.json files to approve"; v2.1.196+: `claude mcp list`/`get` honor the key in an untrusted folder only from settings files NOT checked into the repo. `enableAllProjectMcpServers: true` auto-approves all project .mcp.json servers. |
| https://code.claude.com/docs/en/hooks.md | 2026-07-09 | official doc | WebFetch full | SessionStart matchers: `startup`/`resume`/`clear`/`compact`. stdin: session_id, transcript_path, cwd, hook_event_name, source, model (MAY BE OMITTED after /clear or recovery — check before reading), agent_type, session_title. Command hooks CAN return JSON: `{"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "...", "sessionTitle": "..."}}`; plain stdout also becomes context. `initialUserMessage` applies in headless `-p`. Default timeout 600s. SessionStart CANNOT block startup; exit 2 = non-blocking notice only (fail-open by construction). |
| https://code.claude.com/docs/en/model-config.md | 2026-07-09 | official doc | WebFetch full (74.6KB persisted + read) | "Fallback model chains" section: triggers when primary "is overloaded, unavailable, or returns another non-retryable server error. Authentication, billing, rate-limit, request-size, and transport errors never trigger a switch". "The switch lasts for the current turn only". "Chains are capped at three models after duplicate removal". `--fallback-model sonnet,haiku` (comma list) takes precedence over setting. Each element = model name or alias; `"default"` expands to default model. Skips: unavailable/retired models; allowlist-excluded elements dropped. SEPARATE mechanism: "Automatic model fallback" = content-based Fable-5 classifier fallback to default Opus (cyber/bio); in non-interactive mode a flagged request "ends the turn with a refusal instead". `fable` alias table row confirmed; effort default `high` on Fable 5/Opus 4.8. Non-interactive `/effort` reports Not applied on Fable/4.8/4.7. |
| https://code.claude.com/docs/en/sub-agents.md | 2026-07-09 | official doc | WebFetch full (72.9KB persisted + read) | Background-by-default: "As of v2.1.198, subagents run in the background by default. Claude runs a subagent in the foreground when it needs the result before continuing"; `background: true` frontmatter forces it. Partial-work retention (v2.1.199): foreground cut-off "returns that partial output with a note that the subagent was cut off"; background "marked failed, and the message Claude receives...includes the subagent's last output, so partial work isn't lost"; v2.1.200: no-text subagent fails with "Agent terminated early due to an API error". Nesting: "As of Claude Code v2.1.172, a subagent can spawn its own subagents...A subagent at depth five doesn't receive the Agent tool and can't spawn further. The limit is fixed and not configurable." In a SUBAGENT definition, `Agent(type)` parenthesized allowlist is IGNORED — listing `Agent` in tools enables nesting, type-list only works for `claude --agent` main-thread. Model resolution order: CLAUDE_CODE_SUBAGENT_MODEL > per-invocation param > frontmatter > inherit. IMPORTANT: fallbackModel chain doc lives in model-config; subagent frontmatter `model:` is a pin, availability-fallback applies at the session request layer. |

| https://code.claude.com/docs/en/mcp.md | 2026-07-09 | official doc | WebFetch full (63.7KB persisted + grep-read) | v2.1.196: ".claude mcp list and claude mcp get read .mcp.json approvals only from settings files that aren't checked into the repository until you trust the workspace by running claude in it and accepting the workspace trust dialog. A cloned repository can't approve its own servers: enableAllProjectMcpServers or enabledMcpjsonServers committed to the project's .claude/settings.json is ignored in an untrusted folder, and the server stays at Pending approval". Approvals that DO apply in an untrusted folder: `~/.claude/settings.json`, managed settings, `--settings`, and `.claude/settings.local.json` "as long as git doesn't track it". "A disabledMcpjsonServers entry in any settings file still rejects the server." Pending servers show as "⏸ Pending approval" in `claude mcp list`. "Claude Code prompts for approval before using project-scoped servers from .mcp.json files... reset via claude mcp reset-project-choices". Headless auth (v2.1.196): during `claude -p` with tool search, unauthenticated servers are reported to Claude as "unavailable until you authorize it". |

**KEY IMPLICATION for criterion 1**: `fallbackModel` triggers on overload/unavailability/non-retryable server errors — NOT on rate-limit errors ("rate-limit... never trigger a switch"). The 2026-07-09 Fable Q/A stalls: whether the chain would have fired depends on whether the stall was an overload/5xx (fires) vs a usage-limit cutoff (does NOT fire; that path is instead mitigated by the v2.1.199 partial-work retention). The audit_basis's "would have auto-fallen the two Fable Q/A stalls to Opus" is TRUE for overload-class errors only; adopt the chain anyway (cheap, correct-shaped), but do not claim it covers usage-limit cutoffs. Also note: chain switch lasts the CURRENT TURN only, and the SEPARATE content-classifier fallback (Fable→Opus) already exists independent of this setting.

## Identified but snippet-only

| URL | Kind | Why not fetched in full |
|---|---|---|

## EMPIRICAL FINDING (criterion 4 — measured 2026-07-09/10 on this machine)

`claude mcp list` run non-interactively from the repo cwd (v2.1.205):
ALL 8 project `.mcp.json` servers report `⏸ Pending approval (run \`claude\` to approve)`:
bigquery, paper-search-mcp, pyfinagent-backtest, pyfinagent-data, pyfinagent-risk,
pyfinagent-signals, playwright (7 listed; alpaca not shown = rejected by
`disabledMcpjsonServers: ["alpaca"]`). The claude.ai session connectors (Figma, Slack,
Spotify connected; S&P/BigQuery/Drive need auth) are a separate mechanism.

Local approval state: `~/.claude.json` project entry has `hasTrustDialogAccepted: true`
but `enabledMcpjsonServers: []` and `disabledMcpjsonServers: []` (both EMPTY — no
per-server approvals ever recorded there). `~/.claude/settings.json` (user) has NO mcp
keys. `.claude/settings.local.json` (verified gitignored via `git check-ignore`) has
`enabledMcpjsonServers: ["slack"]` — STALE: no server named "slack" exists in `.mcp.json`
(8 servers: alpaca, bigquery, paper-search-mcp, pyfinagent-backtest, pyfinagent-data,
pyfinagent-risk, pyfinagent-signals, playwright) — and `disabledMcpjsonServers: ["alpaca"]`
(intentional: alpaca order tools are also in the settings.json deny list; keep rejected).

CONCLUSION: the v2.1.196 posture means away/headless sessions (and `claude mcp list`)
currently see ZERO pyfinagent stdio servers approved. The fix is exactly what the
criterion names: `enabledMcpjsonServers` listing the 7 non-alpaca servers, placed in
`.claude/settings.local.json` (untracked -> honored per docs even in untrusted folders;
matches the v2.1.196 security intent that "a cloned repository can't approve its own
servers"). Committed `.claude/settings.json` would also work in THIS trusted folder but
is the shape v2.1.196 specifically distrusts; prefer the local file + a documented note.

## Sixth full read

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://raw.githubusercontent.com/anthropics/claude-code/main/CHANGELOG.md | 2026-07-09 | official changelog | WebFetch full | v2.1.152: "SessionStart hooks can now set the session title via hookSpecialOutput.sessionTitle on startup" (note: SessionStart EVENT itself predates this; 2.1.152 added sessionTitle). v2.1.166: "Added fallbackModel setting to configure up to three fallback models tried in order". v2.1.172: "Sub-agents can now spawn their own sub-agents (up to 5 levels deep)". v2.1.196: "Security: claude mcp list/get no longer spawn .mcp.json servers that a repo self-approved". v2.1.198: "Subagents now run in the background by default"; "Fixed subagents cut off by a rate limit before producing any text output returning an empty [result]". v2.1.199: "Fixed subagents cut off by a rate limit or server error silently failing"; "Fixed streaming responses being discarded when the API emits a mid-stream overloaded/server error". |

## Query variants run (research-gate.md discipline)

1. Current-year: "Claude Code SessionStart hook fallbackModel settings 2026"
2. Last-2-year: "Claude Code hooks settings.json best practices 2025"
3. Year-less canonical: "Claude Code MCP server approval enabledMcpjsonServers headless"

## Recency scan (2025-2026)

Performed 2026-07-09/10. Findings: the feature set under adoption IS the 2026 frontier
(all four features shipped May-July 2026 per changelog). Third-party 2026 coverage
corroborates the official docs with no contradictions: digitalapplied.com
"Claude Code Adds Safe Mode and Fallback Model Chains" and aiforanything.io
"fallbackModel: Fix 529 Overload Errors" (both 2026) confirm the overload-error
(529) trigger scope; claudefa.st "Session Lifecycle Hooks" (2026) confirms
SessionStart stdout-as-context + only-SessionStart-receives-model +
model-not-guaranteed-present; community best practice (2025-2026, shanraisshan /
claudefa.st / vincentqiao) converges on `$CLAUDE_PROJECT_DIR`-prefixed hook paths,
small security-first scripts, project-settings for team guardrails — all already the
house pattern here. One 2026 addition noted: `async: true` on hooks (Jan 2026) runs
a hook without blocking — NOT needed for the tripwire (SessionStart additionalContext
must be synchronous to land before the first prompt). No finding supersedes the
official docs read above.

## Key findings (external)

1. **`fallbackModel` exact schema**: array of strings, max 3 after dedup, in
   project/user settings; per-session override `--fallback-model a,b` (comma list)
   takes precedence; `"default"` expands to the account default; NO merge across
   settings files — highest-precedence file supplies the entire chain.
   (Source: settings.md + model-config.md "Fallback model chains", accessed 2026-07-09)
2. **Trigger scope is narrow**: overloaded / unavailable / non-retryable server
   errors only. "Authentication, billing, rate-limit, request-size, and transport
   errors never trigger a switch." Switch lasts the current turn only.
   (model-config.md) → the 2026-07-09 Fable Q/A stalls are covered only if they were
   overload/5xx; usage-limit cutoffs are instead mitigated by v2.1.199 partial-work
   retention. Both adoptions are complementary, neither is redundant.
3. **Content-classifier fallback is separate**: Fable 5 cyber/bio classifier re-runs
   the request on default Opus automatically — exists regardless of `fallbackModel`;
   in non-interactive mode a flagged request "ends the turn with a refusal".
   (model-config.md "Automatic model fallback")
4. **SessionStart contract**: matchers startup/resume/clear/compact; stdin JSON has
   `source`, optional `model` (may be absent after /clear — check before reading),
   `session_title`; command hooks may print either plain text (becomes context) or
   `{"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "..."}}`;
   `sessionTitle` applies only on startup/resume; `initialUserMessage` works in `-p`
   headless; default timeout 600s; SessionStart CANNOT block session start — exit 2 is
   a non-blocking transcript notice, so the hook is fail-open by construction.
   (hooks.md, accessed 2026-07-09)
5. **v2.1.196 MCP approval**: committed `.claude/settings.json` approval keys are
   ignored in untrusted folders ("A cloned repository can't approve its own servers");
   approvals honored from user settings, managed settings, `--settings`, and an
   untracked `.claude/settings.local.json`. `disabledMcpjsonServers` in ANY file still
   rejects. Pending servers show `⏸ Pending approval` in `claude mcp list`.
   (mcp.md + settings.md)
6. **Subagent runtime semantics to document** (sub-agents.md):
   background-by-default since v2.1.198 ("Claude runs a subagent in the foreground
   when it needs the result before continuing"); partial-work retention since
   v2.1.199 (foreground: partial output + cut-off note; background: marked failed but
   last output included, "so partial work isn't lost"; v2.1.200: zero-text cutoffs
   fail with "Agent terminated early due to an API error"); nesting since v2.1.172
   (5 levels; depth-5 subagent doesn't receive Agent tool; limit fixed). In a
   SUBAGENT definition, `Agent(type-list)` parentheses are IGNORED — plain `Agent`
   in `tools` is what enables nesting (type allowlists only work for `claude --agent`
   main-thread). Subagent model resolution: `CLAUDE_CODE_SUBAGENT_MODEL` >
   per-invocation param > frontmatter > inherit.

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| .claude/settings.json | hooks block (whole file) | House hook pattern: every command entry is `bash "${CLAUDE_PROJECT_DIR:-$(pwd)}/.claude/hooks/<script>.sh"` + a `statusMessage` string; events wired: PreToolUse, ConfigChange, InstructionsLoaded, PostToolUse (matcher+if), TeammateIdle, Stop (agent-type), SubagentStop | NO SessionStart entry; NO fallbackModel key (grep confirms) |
| docs/runbooks/per-step-protocol.md | 264-269 | "Hook sanity" section documents the `${CLAUDE_PROJECT_DIR:-$(pwd)}` resolution idiom | New SessionStart entry must match; runbook has NO subagent-runtime-semantics section (grep for background/nest/partial: no hits) |
| scripts/away_ops/run_away_session.sh | 20, 141-143, 160-165 | REAL headless invocation: `CLAUDE_BIN=/Users/ford/.local/bin/claude`; main launch `claude -p --dangerously-skip-permissions --model claude-opus-4-8 --max-turns N --output-format json < prompt > out.json`; auth probe at :141-143 same shape | MCP attach for away sessions rides this exact path |
| ~/Library/LaunchAgents/com.pyfinagent.away-session-{am,pm}.plist | ProgramArguments | `/bin/bash run_away_session.sh {am,pm}`; env: CLAUDE_CODE_OAUTH_TOKEN (secret, redacted), CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1, HOME, PATH; AM 07:30 cap 14400s/250 turns, PM 22:00 cap 7200s/120 turns; WorkingDirectory = repo | Same user + HOME → same ~/.claude.json + settings.local.json as interactive sessions |
| .claude/settings.local.json | whole file | `enabledMcpjsonServers: ["slack"]` (STALE — no such server in .mcp.json), `disabledMcpjsonServers: ["alpaca"]` (intentional, keep) | Verified gitignored (`git check-ignore` passes) → honored per v2.1.196 rules |
| .mcp.json | mcpServers keys | 8 servers: alpaca, bigquery, paper-search-mcp, pyfinagent-backtest, pyfinagent-data, pyfinagent-risk, pyfinagent-signals, playwright (caller said 9; actual count is 8) | 7 of 8 `⏸ Pending approval` empirically; alpaca rejected |
| ~/.claude.json (project entry) | projects["...pyfinagent"] | `hasTrustDialogAccepted: true`, `enabledMcpjsonServers: []`, `disabledMcpjsonServers: []`, `mcpServers: {}` — NO per-server approvals recorded despite trust | Explains the Pending state |
| ~/.claude/settings.json (user) | keys | env, permissions, model, effortLevel, tui, skipDangerousModePermissionPrompt, agentPushNotifEnabled — NO mcp keys | Cannot be the approval source |
| .claude/agents/researcher.md | 5-6, 27, 42, 241-250 | `model: fable` (:5), `maxTurns: 40` (:6), stale "requires Claude Code v2.1.170+, local 2.1.172" comment (:27), `effort: max` (:42), deep-tier "Multi-subagent fork option" §4 (:241-250) says "Main may spawn 2-3 parallel deep-tier subagents" | :27 comment stale (local is 2.1.205); :241-250 predates v2.1.172 self-nesting |
| .claude/agents/qa.md | 5-6, 28, 44 | `model: fable` (:5), `maxTurns: 30` (:6), same stale version comment (:28), `effort: max` (:44) | :28 comment stale |
| .claude/agent-memory/harness-verifier/, .claude/agent-memory/qa-evaluator/ | dir listing | Both contain ONLY `.` and `..` (ls -la verified) — completely empty | Safe to rmdir (git doesn't track empty dirs; no data loss) |
| .claude/hooks/ | dir listing | 10 scripts + lib/; no session-start script exists yet | Tripwire script is net-new: session-start-fable-tripwire.sh |
| .claude/masterplan.json | phase-67 step 67.5 | Verbatim criteria + immutable verification command copied above | Verification greps `.claude/settings.json` for both keys; requires tripwire file; requires absence of "local 2.1.172" in .claude/agents/; requires both legacy dirs gone |

## Application to pyfinagent — concrete recommendations

### 1. `fallbackModel` diff (criterion 1)

Add to `.claude/settings.json` (project — satisfies the immutable grep; no merge
across files, so this one file carries the whole chain):

```json
"fallbackModel": ["claude-opus-4-8", "claude-sonnet-5"]
```

Rationale: the criterion's "fable-first falling to opus-4-8" describes the SYSTEM
posture — primary = Fable (operator's saved `/model fable` default + the two agent-file
`model: fable` pins at researcher.md:5 / qa.md:5), chain = what to fall TO. The chain
itself must NOT contain fable (it's the primary; duplicates are removed anyway). Pin
full IDs, not aliases, so the chain stays deterministic after the 67.4 Sunday revert
flips primaries back to opus. `claude-sonnet-5` as element 2 gives a second rung
(requires v2.1.197+; local is 2.1.205 — OK). Do NOT add `"default"` as a third
element: on this Max account it expands to Opus 4.8 = duplicate of element 1, removed.
HONEST CAVEAT for the contract: docs scope the chain to requests the session makes;
they do not explicitly promise coverage of subagent-pinned models, and rate-limit
errors NEVER trigger the chain (that class is handled by v2.1.199 partial-work
retention instead). Do not oversell in experiment_results.md.

### 2. SessionStart tripwire (criterion 2)

settings.json entry (house pattern — statusMessage + ${CLAUDE_PROJECT_DIR} idiom per
per-step-protocol.md:264-269; no matcher = fires on startup/resume/clear/compact,
maximally loud):

```json
"SessionStart": [
  {
    "hooks": [
      {
        "type": "command",
        "command": "bash \"${CLAUDE_PROJECT_DIR:-$(pwd)}/.claude/hooks/session-start-fable-tripwire.sh\"",
        "statusMessage": "Checking Fable Sunday-revert tripwire (67.4)..."
      }
    ]
  }
]
```

Script sketch `.claude/hooks/session-start-fable-tripwire.sh` (~26 lines, fail-open —
every path exits 0; SessionStart cannot block startup anyway per hooks.md):

```bash
#!/usr/bin/env bash
# session-start-fable-tripwire.sh -- phase-67.5. Warns via additionalContext
# when today >= 2026-07-12 AND an agent file still pins model: fable.
# Fail-open: every path exits 0. Dry-run: TRIPWIRE_FAKE_TODAY=20260713.
set -u
DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
DEADLINE=20260712
TODAY="${TRIPWIRE_FAKE_TODAY:-$(date +%Y%m%d 2>/dev/null)}"
case "$TODAY" in ''|*[!0-9]*) exit 0 ;; esac
[ "$TODAY" -ge "$DEADLINE" ] || exit 0
PINNED=""
for f in researcher qa; do
  grep -qE '^model:[[:space:]]*fable[[:space:]]*$' \
    "$DIR/.claude/agents/$f.md" 2>/dev/null && PINNED="$PINNED $f.md"
done
[ -n "$PINNED" ] || exit 0
WARN="WARNING -- masterplan 67.4 is the TOP P0: today ($TODAY) is on/after 2026-07-12 but$PINNED still pin 'model: fable'. The Fable-5 evaluation window is over; revert the Layer-3 pins to 'model: opus' (masterplan step 67.4) BEFORE spawning researcher/qa, then restart the session (roster snapshot)."
python3 -c 'import json,sys; print(json.dumps({"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":sys.argv[1]}}))' "$WARN" 2>/dev/null \
  || printf '%s\n' "$WARN"
exit 0
```

Design notes: (a) the `|| printf` fallback still lands as context because plain
SessionStart stdout is added as context per hooks.md — double fail-open; (b) grep
targets the exact frontmatter `model: fable` at researcher.md:5 / qa.md:5; after 67.4
flips them to `model: opus` the hook goes silent with zero maintenance; (c) live_check
(a) = `TRIPWIRE_FAKE_TODAY=20260713 bash .claude/hooks/session-start-fable-tripwire.sh`
showing the JSON verbatim; also run WITHOUT the fake date pre-Sunday to show silence.

### 3. Doc refresh list (criterion 3)

1. `.claude/agents/researcher.md:27` + `.claude/agents/qa.md:28` — replace
   "requires Claude Code v2.1.170+, local 2.1.172" with "requires Claude Code
   v2.1.170+, local 2.1.205 at phase-67.5" (the immutable grep demands NO
   "local 2.1.172" anywhere under .claude/agents/).
2. `.claude/agents/researcher.md:241-250` §4 minimal rewording (self-managed nesting):
   "**Multi-subagent fork option.** If the caller requests it, OR if the topic has
   >=3 clearly separable sub-questions that each warrant a `complex`-tier session on
   their own, the researcher MAY fork ITSELF: spawn 2-3 parallel deep-tier
   sub-researchers as nested subagents (supported since Claude Code v2.1.172, max 5
   levels; requires `Agent` in this agent's tool list — note a type-allowlist in
   parentheses is ignored in subagent frontmatter) and merge their read-in-full
   tables; Main-level forking remains a fallback when nesting is unavailable. Each
   sub-researcher must meet the >=20-source floor INDEPENDENTLY. The merged brief
   must deduplicate URLs and label sources by subagent origin. Estimated cost: ~1
   Claude Max 5-hour rolling window per subagent; confirm with caller before forking."
3. `docs/runbooks/per-step-protocol.md` — add a short "Subagent runtime semantics
   (v2.1.205, phase-67.5)" subsection near the Hook-sanity section (:264): background-
   by-default (2.1.198) — Main should NOT assume a spawn blocks; partial-work
   retention (2.1.199/2.1.200) — a rate-limited Q/A now returns partial output + an
   API-error note instead of silently failing (directly mitigates the 2026-07-09
   Fable Q/A stall class — the stall evidence is no longer lost); 5-level nesting
   (2.1.172) — the researcher deep-tier fork is self-managed. Cross-link researcher.md §4.
4. `rmdir .claude/agent-memory/harness-verifier .claude/agent-memory/qa-evaluator`
   (verified empty; git does not track empty dirs — no history loss; satisfies the
   two `test ! -d` legs).
5. `.claude/settings.local.json` — drop stale `"slack"` from enabledMcpjsonServers
   while adding the real 7 (below).

### 4. Headless MCP attach fix + verification (criterion 4)

FIX (do first): set in `.claude/settings.local.json` (untracked → honored even in
untrusted folders; conforms to the v2.1.196 security intent — a committed
self-approval is exactly what 2.1.196 distrusts):

```json
"enabledMcpjsonServers": ["bigquery", "paper-search-mcp", "pyfinagent-backtest", "pyfinagent-data", "pyfinagent-risk", "pyfinagent-signals", "playwright"],
"disabledMcpjsonServers": ["alpaca"]
```

Keep alpaca REJECTED (its order-placing tools are individually denied in
settings.json, but rejection at spawn is defense-in-depth and current posture).

VERIFY (live_check (b)): from the repo cwd, `claude mcp list` — expect the 7 servers
to flip from `⏸ Pending approval (run \`claude\` to approve)` (verbatim current state,
measured 2026-07-09/10) to `✔ Connected` (or at worst a health-check failure, which is
a server bug, not an approval gap). This is the documented approval-state surface and
costs no tokens. The REAL away path (`run_away_session.sh:160-165`,
`claude -p --dangerously-skip-permissions --model claude-opus-4-8 --output-format json`)
runs as the same user with the same HOME → reads the same settings.local.json;
optionally prove end-to-end with a 1-turn `claude -p` asking for a ToolSearch of
`mcp__pyfinagent-data` tools (token cost ~cents — flag for operator since metered
$0 policy applies to away infra).

### Pitfalls (from literature + local evidence)

- fallbackModel does NOT cover rate-limit/auth errors (model-config.md) — don't claim
  it fixes the credential-death class (66.4) or usage-limit stalls.
- The chain is read from ONE settings file (no merge) — a future user-settings
  fallbackModel would silently shadow the project one... precedence is
  managed > local > project > user, and `.claude/settings.local.json` outranks
  project settings.json: keep the chain OUT of settings.local.json.
- SessionStart `model` stdin field may be absent (post-/clear) — tripwire ignores
  stdin entirely by design.
- Hook stdout that is not valid JSON still becomes context (SessionStart-specific) —
  benign here, by design in the fallback branch.
- `.mcp.json` edits mid-session don't respawn connected stdio servers (CLAUDE.md
  Playwright note) — approval flips likewise need a fresh session/`/mcp` reconnect.
- Caller's prompt said "9 .mcp.json servers"; actual count is 8 (alpaca, bigquery,
  paper-search-mcp, 4x pyfinagent-*, playwright) — contract should say 8/7.

## Consensus vs debate

No contradictions found between official docs, changelog, and 2026 third-party
coverage. Sole nuance worth recording: community posts loosely say fallbackModel
"fixes 529 overload errors" while the official doc enumerates the full trigger set
(overloaded, unavailable, non-retryable server errors) and the exclusions
(auth/billing/rate-limit/request-size/transport) — use the official wording in docs.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6: settings.md, hooks.md, model-config.md, sub-agents.md, mcp.md, CHANGELOG.md — all official Anthropic)
- [x] 10+ unique URLs total (6 full + 20 snippet-only = 26)
- [x] Recency scan (2025-2026) performed + reported
- [x] Full pages read (not snippets) for the read-in-full set (two persisted-to-file at 72-75KB and read back; mcp.md grep-extracted from full persisted fetch)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module the step touches (settings x3 scopes, ~/.claude.json, .mcp.json, launchd plists, launcher script, both agent files, runbook, hooks dir, legacy memory dirs)
- [x] Contradictions/consensus noted
- [x] Claims cited per-claim

## Snippet-only sources (do not count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://code.claude.com/docs/en/hooks-guide | official doc | hooks.md already covered the contract |
| https://www.digitalapplied.com/blog/claude-code-safe-mode-fallback-models-production-resilience-guide | blog 2026 | corroborates official doc; snippet sufficient |
| https://www.aiforanything.io/blog/claude-code-fallback-model-overload-fix-2026 | blog 2026 | corroborates 529-trigger scope |
| https://claudefa.st/blog/tools/hooks/session-lifecycle-hooks | blog | corroborates SessionStart model-field caveat |
| https://claudefa.st/blog/tools/hooks/hooks-guide | blog | general hooks overview |
| https://claudefa.st/blog/guide/settings-reference | blog | settings overview |
| https://blakecrosley.com/guides/claude-code | blog | general guide |
| https://developertoolkit.ai/en/claude-code/version-management/changelog/ | community changelog mirror | official changelog read instead |
| https://angelo-lima.fr/en/claude-code-cheatsheet-june-2026-update/ | blog 2026 | cheatsheet |
| https://www.sitepoint.com/claude-code-june-2026-10-new-features-devs-need-to-know/ | blog 2026 | feature roundup |
| https://hidekazu-konishi.com/entry/claude_code_hooks_complete_guide.html | blog | hooks guide |
| https://github.com/disler/claude-code-hooks-mastery | community repo | example hooks |
| https://github.com/shanraisshan/claude-code-best-practice/blob/main/best-practice/claude-settings.md | community repo | settings best practice |
| https://github.com/shanraisshan/claude-code-best-practice/blob/main/.claude/settings.json | community repo | example |
| https://smartscope.blog/en/generative-ai/claude/claude-code-hooks-guide/ | blog | hooks guide |
| https://blog.vincentqiao.com/en/posts/claude-code-settings-hooks/ | blog | hooks guide |
| https://explainx.ai/blog/claude-code-settings-json-complete-reference-2026 | blog 2026 | settings reference |
| https://blog.promptlayer.com/understanding-claude-code-hooks-documentation/ | blog | hooks overview |
| https://github.com/anthropics/claude-code/issues/3106 | GitHub issue | mcp list disabled-server reporting |
| https://www.builder.io/blog/claude-code-mcp-servers | blog | MCP setup guide |

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 20,
  "urls_collected": 26,
  "recency_scan_performed": true,
  "internal_files_inspected": 13,
  "report_md": "handoff/current/research_brief_67_5.md",
  "gate_passed": true
}
```
