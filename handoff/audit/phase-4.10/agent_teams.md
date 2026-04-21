# Agent Teams Audit — Claude Doc Alignment (phase-4.10.3)

Source: https://code.claude.com/docs/en/agent-teams (fetched 2026-04-18).
Audit is read-only; no pyfinAgent source was modified.

## Documentation summary

**What an "agent team" is.** "Agent teams let you coordinate multiple
Claude Code instances working together. One session acts as the team
lead, coordinating work, assigning tasks, and synthesizing results.
Teammates work independently, each in its own context window, and
communicate directly with each other." It is explicitly contrasted
with **subagents**: "Subagents only report results back to the main
agent and never talk to each other. In agent teams, teammates share a
task list, claim work, and communicate directly with each other."

**Activation.** "Agent teams are experimental and disabled by default.
Enable them by adding `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` to your
settings.json or environment." Requires Claude Code v2.1.32+.

**Architecture (verbatim table from doc).**
| Component | Role |
|---|---|
| Team lead | Main Claude Code session; creates team, spawns teammates, coordinates |
| Teammates | Separate Claude Code instances, each on assigned tasks |
| Task list | Shared list; teammates claim + complete |
| Mailbox | Messaging system between agents |

**How a team is defined.** Claude itself spawns the team in response
to a natural-language request ("Create an agent team to …"). State
lives at `~/.claude/teams/{team-name}/config.json` and
`~/.claude/tasks/{team-name}/`; the doc is emphatic that this file is
**auto-managed**: "don't edit it by hand or pre-author it: your
changes are overwritten on the next state update. There is no
project-level equivalent of the team config. A file like
`.claude/teams/teams.json` in your project directory is not recognized
as configuration; Claude treats it as an ordinary file."

**Teammate roles** reuse subagent definitions: "you can reference a
subagent type from any subagent scope … The teammate honors that
definition's `tools` allowlist and `model`, and the definition's body
is appended to the teammate's system prompt." Note: `skills` and
`mcpServers` frontmatter are **not** applied to teammates.

**Messaging.** Two primitives: `message` (one teammate) and
`broadcast` (all). "Team coordination tools such as `SendMessage` and
the task management tools are always available to a teammate even
when `tools` restricts other tools." Delivery is automatic; the lead
does not poll.

**Lifecycle hooks.** `TeammateIdle`, `TaskCreated`, `TaskCompleted`
can exit code 2 to block or feedback-loop.

**Hard limits.** One team per session, no nested teams, lead is fixed
for team lifetime, permissions set at spawn.

## Codebase audit

### Team-like structures in pyfinAgent

1. **Layer 1 — Gemini analysis pipeline** (`backend/agents/orchestrator.py`,
   1495 lines). A linear 15-step pipeline over ~28 skill prompts
   (`backend/agents/skills/*.md`). Coordinator is `orchestrator.py`;
   workers are stateless Gemini calls via `LLMClient`. **Not a team**
   by Claude's definition — no inter-worker messaging, no shared task
   list, no Claude Code instances; just a DAG of LLM calls.

2. **Layer 2 — Claude MAS** (`backend/agents/multi_agent_orchestrator.py`
   L1–L1234; `agent_definitions.py` L122+ `AGENT_CONFIGS`). Four
   roles: Communication (Sonnet), Ford/Main (Opus), QA/Analyst (Opus),
   Researcher (Sonnet). Delegation is expressed as a static
   `can_delegate_to` graph (`agent_definitions.py:132,182,230,276`).
   Parallel fan-out at `multi_agent_orchestrator.py:446–467` uses
   `asyncio.gather` over `loop.run_in_executor(..., _call_agent_with_tools)`
   — i.e., **parallel `anthropic.Anthropic()` API calls**, not
   spawned Claude Code sessions. No mailbox, no shared task list, no
   teammate-to-teammate messaging: findings flow back through Ford as
   the synthesizer (`_synthesize`, L515). This is the classic
   **subagent** pattern, not an agent team.

3. **Layer 3 — Harness** (`scripts/harness/run_harness.py`, 1206 lines).
   Planner → Generator (`QuantStrategyOptimizer`) → Evaluator is a
   sequential loop with file-based handoffs (`handoff/current/*.md`
   confirmed: contract, experiment_results, evaluator_critique). Again
   direct Python dispatch, not Claude Code sessions; no SendMessage.

4. **Claude Code sub-agents** (`.claude/agents/*.md`: qa-evaluator.md,
   harness-verifier.md, researcher.md, per-step-protocol.md). These
   **are** correctly defined subagents (frontmatter: `name`, `tools`,
   `model`). `per-step-protocol.md` §4 spawns qa-evaluator +
   harness-verifier "in parallel" via two `Agent` tool calls in one
   message — this is the Claude Code subagent pattern, not a team.
   They report to the orchestrator only; they do not message each
   other, and no shared task list exists.

5. **Agent SDK usage.** `grep claude-agent-sdk` / `@anthropic-ai/claude-agent-sdk`:
   **zero hits** in source. No SDK Teams API is used anywhere.

6. **Settings.** `.claude/settings.json:74` sets
   `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS: "1"` and `.claude/settings.json:49`
   registers a `TeammateIdle` hook
   (`.claude/hooks/teammate-idle-check.sh`) — so the flag is *enabled*
   but nothing in the repo actually asks Claude Code to create a team.

## Findings

| Aspect | Status | Evidence | Notes |
|---|---|---|---|
| Team-lead / teammates split (separate Claude Code instances) | Missing | `backend/agents/multi_agent_orchestrator.py:446–467` | Layer 2 uses parallel API calls, not spawned sessions. |
| Shared task list | Missing | — | No `~/.claude/tasks/{team}/` used; task state lives in `handoff/current/*.md` + `.claude/masterplan.json`. |
| Mailbox / SendMessage / broadcast | Missing | — | Subagents in `.claude/agents/` cannot message each other; findings aggregate via Ford (`_synthesize`, L515). |
| Subagent-definition reuse as teammates | Partial | `.claude/agents/{qa-evaluator,harness-verifier,researcher}.md` | Valid subagent frontmatter — would work as teammate definitions if a team were ever spawned. |
| `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` enabled | Correct | `.claude/settings.json:74` | But unused — no prompt in repo asks Claude to create a team. |
| `TeammateIdle` hook wired | Correct (stub) | `.claude/settings.json:49`, `.claude/hooks/teammate-idle-check.sh` | Fires only if a team exists; today it never does. |
| `TaskCreated` / `TaskCompleted` hooks | Partial | `.claude/settings.json:37` has `TaskCompleted` | Used for masterplan cross-verification, not for team task gating. |
| File-based handoffs | Correct for our pattern | `handoff/current/{contract,experiment_results,evaluator_critique}.md`, `handoff/harness_log.md` | Matches Anthropic "harness design" post; orthogonal to agent-teams. |
| Parallel qa-evaluator + harness-verifier spawn | Correct (as subagents) | `.claude/agents/per-step-protocol.md` §4 | Both invoked in one `Agent` tool batch. Not a team — each reports back, doesn't message the other. |
| Coordinator/worker clarity | Correct | `multi_agent_orchestrator.py:12–17`, `agent_definitions.py:10–16` | Ford/Main is clearly the LeadResearcher. |
| No nested teams constraint | N/A | — | Harness could theoretically hit this if migrated; keep in mind. |
| `SendMessage` tool access in subagent frontmatter | Not applicable | `.claude/agents/*.md` | Only meaningful inside a team session; our subagents are not teammates. |

## Gaps & Opportunities

**MUST FIX — none.** pyfinAgent's MAS is an intentional
subagent-plus-file-handoff pattern aligned with Anthropic's
"Harness design for long-running apps" and "How We Built Our
Multi-Agent Research System." The agent-teams feature is experimental
and explicitly discouraged for "sequential tasks, same-file edits, or
work with many dependencies" (most masterplan steps). No correctness
bug stems from the current choice.

**NICE TO HAVE — targeted migration candidates.**

1. **phase-4.10 research-gate cycles** are the one place a team fits:
   adversarial-hypothesis debugging and multi-angle research gates
   map directly onto the doc's "Investigate with competing
   hypotheses" and "parallel code review" use cases. An experiment
   would ask the lead to "Create an agent team … researcher
   (`.claude/agents/researcher.md`), qa-evaluator, harness-verifier"
   — our existing subagent definitions are already valid teammate
   roles.

2. **Formalize tool scoping by moving MAS roles to
   `.claude/agents/*.md`.** Today `AGENT_CONFIGS` in
   `agent_definitions.py:122` carries its own ad-hoc
   `can_delegate_to` graph and `AGENT_TOOLS` list
   (`multi_agent_orchestrator.py:72–120`). If we want agent-teams
   semantics, we'd define `ford-lead.md`, `analyst.md`, etc. with
   `tools:` allowlists and let the lead spawn them as teammates.

3. **Remove dead weight.** `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`
   and the `TeammateIdle` hook are enabled but unreachable. Either
   commit to an agent-teams pilot (opportunity 1) or strip both from
   `.claude/settings.json` to match the stress-test doctrine in
   CLAUDE.md §Stress-test.

4. **Do NOT pre-author team configs.** The doc warns that
   `.claude/teams/teams.json` is "not recognized as configuration;
   Claude treats it as an ordinary file." If anyone drafts such a
   file later, flag it — this is the easiest foot-gun.

5. **No Agent SDK Teams API in use** and no need: the SDK documents
   no stable Teams surface today, and our file-based harness is
   version-controlled and testable without it.

## References

- https://code.claude.com/docs/en/agent-teams (primary, quoted above)
- https://code.claude.com/docs/en/sub-agents (comparison table source)
- https://www.anthropic.com/engineering/built-multi-agent-research-system
- https://www.anthropic.com/engineering/harness-design-long-running-apps
- Repo: `backend/agents/multi_agent_orchestrator.py:446–526`,
  `backend/agents/agent_definitions.py:122–276`,
  `scripts/harness/run_harness.py:1–60`,
  `.claude/agents/{qa-evaluator,harness-verifier,researcher,per-step-protocol}.md`,
  `.claude/settings.json:37,49,74`.
