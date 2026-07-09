# Goal Prompt -- goal-phase67-fable-window

Set: 2026-07-09 (operator in-session, via /goal with active Stop hook). Runs alongside
the 66.2 close (deferred from Cycle 76) and ahead of the parked 61.2-61.5 work.

Operator directive (verbatim, from this session):
- "we shall only use fable for program development while the in app tiers should have
  the highest return of token based RIO [ROI]"
- "we only have fable 5 until this sunday where i would like our harness and mas agents
  to be at top notch using fable 5 reasoning and other features"
- /goal (2026-07-09): "use Fable — the most capable model — as an UPGRADE ENGINE to make
  DURABLE improvements to the harness and MAS agents, so that after we revert to
  Opus 4.8 the system is permanently better at: developing to spec, researching
  correctly, coding without bugs, and verifying that code actually works. The
  deliverable is improved agent definitions / skills / protocols that persist on
  Opus 4.8 -- NOT a permanent Fable pin."

## Why (north-star linkage)

N* = Profit - (Risk Exposure + Compute Burn). Anthropic renewed FREE Fable 5 access on
the Max plan through ~Sunday 2026-07-12. Fable is the upgrade ENGINE, not the runtime
model: it audits and rewrites the harness/MAS artifacts themselves during the free
window; the improvements persist when the pins revert to Opus 4.8. The harness builds
and self-improves a live trading system, so better scaffolding compounds every cycle at
zero marginal burn.

## Audit findings (2026-07-09, all tool-verified this session)

PRIMARY (verified against repo HEAD + live venv):
1. **Dead 55s Q/A cap.** `.claude/agents/qa.md` "Maximum runtime: 55 seconds (leave
   buffer for hook timeout)" is calibrated to the TaskCompleted hook RETIRED in
   phase-23.8.2 (per-step-protocol.md:227-229; audit R-2). No hook spawns Q/A today
   (spawns are manual Agent-tool, maxTurns 30, observed 20-26 tool uses per
   evaluation). The cap makes the harness dry-run "optional if time permits" and full
   pytest runs impossible -- it actively caps verification depth. Stress-test doctrine:
   stale scaffolding, prune.
2. **No Python lint gate anywhere.** ruff/pyflakes/flake8 all absent from .venv
   (verified). Q/A's deterministic checks (`ast.parse`) cannot see undefined names.
3. **Live NameError in the Layer-2 MAS** (found BY this audit, repro verbatim):
   `backend/agents/agent_definitions.py:396` `except (json.JSONDecodeError, ...)` --
   module never imports `json`. Repro: `parse_llm_classification('not json {')` ->
   `NameError: name 'json' is not defined`. The graceful-default-to-Main fallback
   (built precisely for malformed Communication-agent output) is dead code.
4. **CONDITIONAL-recovery contradiction.** qa.md:237-239 + per-step-protocol.md
   anti-pattern #5 mandate "SendMessage back to the SAME agent"; CLAUDE.md canonical
   cycle-2 flow + per-step-protocol.md §4 retry loop mandate a FRESH Q/A spawn after
   evidence changes (SendMessage documented unreliable -- dormant agents don't
   auto-replay; re-confirmed by the Cycle-76 stall where a SendMessage nudge failed).
   An executing Main is told both things.
5. **stop_hook_active auto-PASS backdoor.** qa.md returns `{"ok": true}` ("loop
   prevention") when stop_hook_active is set -- an unconditional PASS path in the
   evaluator, newly risky now that /goal Stop hooks are in active use.
6. **Researcher unbounded-read failure mode.** 2026-05-16 incident: 132K tokens read,
   ZERO brief written. The write-first fix lives only in operator memory (prompts must
   remember to say it) -- not codified in researcher.md.
7. **Reasoning-echo scan (Fable reasoning_extraction risk): CLEAN.** No instruction in
   researcher.md / qa.md / SKILL.md tells an agent to echo its chain of thought as
   output text; verdict "reason" fields demand evidence citations, not reasoning
   transcripts. No change needed.
SECONDARY:
8. Operator's recurring shipped-bug class (memory, 2026-05-26 complaint): interface /
   flag / output-shape changes with consumers not all updated (argv-vs-stdin,
   --max-tokens SDK-vs-CLI, Recent Reports alpha/casing). No code-review heuristic
   covers it.
9. Stale scaffolding: runbook diagram labels Researcher "(sonnet)" (it runs opus/fable);
   researcher.md + agent_definitions.py carry undated point-in-time metrics
   ("Sharpe 1.1705", "May 2026 go-live target", "Opus 4.6" comments).

## The chosen changes (2-3 per the goal; ranked by leverage)

- **67.1 Q/A verification depth ("does it actually run")** -- retire the 55s relic ->
  tiered budget; add deterministic backend gates: Python lint (undefined-name class,
  ruff/pyflakes) + runtime smoke (venv import of changed modules; exercise live paths
  against the running app) mirroring the phase-23.2.24 ESLint-gate precedent; remove
  the stop_hook_active auto-PASS; reconcile the recovery contradiction to the CLAUDE.md
  canonical flow (fresh spawn on changed evidence).
- **67.2 Bug-catching** -- add the consumer-contract-break heuristic to
  code-review-trading-domain SKILL.md (the operator's #1 shipped-bug class) and fix the
  verified NameError with a behavioral test. The fix is the living proof the 67.1 lint
  gate catches real bugs.
- **67.3 Research correctness** -- codify WRITE-FIRST / incremental-brief discipline in
  researcher.md (partial brief + honest gate_passed:false envelope even on failure);
  prune stale labels/metrics. NO research-gate floor is weakened (>=5 full reads,
  >=10 URLs, recency scan, envelope: all non-negotiable per operator).
- **67.4 Post-window revert (scheduled Sunday 2026-07-12)** -- pins back to Opus 4.8
  (unless operator token `FABLE PERMANENT: AUTHORIZE`); KEEP every artifact
  improvement; CLAUDE.md reflects the steady state.

## Setup (operator-directed, precedent: 07-08 burn-down repin via /goal item 4)

`model: fable` in researcher.md + qa.md frontmatter with REVERT-BY 2026-07-12 comment;
Main session on `/model fable` (done by operator). Roster-snapshot caveat: pins take
effect NEXT session; THIS session's subagent spawns run the Opus snapshot -- which is
the desired separation anyway (Fable authors, Opus-snapshot Q/A verifies, and the
evaluating Q/A instances do NOT run the definitions being modified: no bootstrap).

## Boundaries (binding -- violating any is an automatic FAIL)

- $0 METERED. Fable only via the free Max rail (Claude Code session + `model: fable`
  frontmatter). No in-app / api.anthropic.com Fable pins in this phase at all -- the
  in-app tiers stay on the token-ROI doctrine (Opus 4.8 / Sonnet 4.6 as pinned).
- HARNESS STAYS EXACTLY 3 AGENTS. Improve definitions; never add/split agents.
- FULL HARNESS PROTOCOL per step: research gate -> contract -> generate -> fresh Q/A ->
  log-last -> live_check -> flip. No self-evaluation. No verdict-shopping.
- Research-gate floors are IMMUTABLE (operator: ">=3 authoritative sources + >=10 URLs
  + read-in-full is non-negotiable"; the >=5 floor in researcher.md stays).
- DO NO HARM: trailing-stop engine, sector caps, kill-switch, gate thresholds untouched.
- Every progress claim audited against a tool result from the session.
- A change ships only if a fresh Q/A (or the directive_review.py judge gate, fail-closed
  mean >= 0.70, for agent-definition rewrites) confirms it makes the harness BETTER.

## Definition of done (whole goal)

67.1-67.3 all PASS with the five-file protocol per step; 66.2 closed (rider from
Cycle 76); Fable pins reverted on schedule (67.4) with all artifact improvements
retained; $0 metered throughout; harness_log carries the full trail.
