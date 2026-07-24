# Design pack — phase-71 (Harness + MAS upgrade)

Step 71.0 GENERATE. Offline, $0, **no production code changed** — this is the implementation design for the
downstream steps 71.1–71.6, converting the 2026-07-16 max-effort ultracode self-audit register
(`handoff/current/harness_proposals.json`, 17 kept / 15 rejected) into grounded, constraint-respecting changes.
Research basis: `research_brief_71.0.md` (7 sources read in full; every register fact re-verified on HEAD 7d54d30d).

## Binding constraints (apply to EVERY phase-71 step — verbatim)

- **Layer-3 harness stays EXACTLY 3 agents** (Main + Researcher + Q/A). No re-splitting Explore / harness-verifier.
  Adversarial / red-team checks live WITHIN the single Q/A role, not as a fourth agent.
- **Self-evaluation by Main stays forbidden;** Q/A independence is load-bearing; verdicts are transcribed VERBATIM,
  never authored by Main.
- **$0 metered on the Layer-3 rail** (first-party Claude Code on the Max plan). Layer-2 is per-ticker
  cost-sensitive — the effort proposals RECONCILE/REDUCE, never inflate.
- **Local-only** (Peder's Mac). Any scheduled self-audit is a local `/loop` or a report-only routine; never an
  assumed cloud fleet, never auto-apply.
- **File-based 5-file handoff protocol stays** the durable-state backbone.
- **Separation of duties + roster snapshot** for the steps that edit `.claude/agents/*.md` (71.1, 71.3, 71.4,
  71.5, 71.6 — i.e. EVERY downstream step EXCEPT the pure Layer-2 backend step 71.2):
  the session that edits an agent `.md` cannot self-evaluate work depending on it; agent-file edits take effect
  only at the NEXT session start (roster snapshot) — each such step must (a) leave a `harness_log` note requesting
  Peder review before a later step depends on the new wording, and (b) note that `scripts/qa/verify_qa_roster_live.sh`
  must confirm the new roster is live next session.
- **historical_macro FROZEN; hysteresis BANNED;** every grounding must be a REAL current feature/doc (URL cited).

## Per-step design (grounded)

### 71.1 (P1) — Workflow structured-output as the first-class Q/A + Researcher launch  [kept #1/#2/#3/#10]
**Grounding:** https://code.claude.com/docs/en/workflows — "The runtime tracks each agent's result as the run
progresses, which is what makes a run **resumable within the same session**"; the script "executes … in an isolated
environment, separate from your conversation. Intermediate results stay in **script variables**"; the
`agent(prompt, {schema})` primitive (v2.1.154+); scripts saved under `.claude/workflows/`; "Every agent in a
workflow uses your session's model unless … `CLAUDE_CODE_SUBAGENT_MODEL` is set." Plus
https://www.anthropic.com/engineering/multi-agent-research-system (resume-not-restart) +
https://www.anthropic.com/engineering/harness-design-long-running-apps (separate doer/judge).
**Change:** save a reusable `.claude/workflows/` script that runs the Q/A (and Researcher) role and returns a
**structured verdict = the captured return value** — immune to the 6× end-flush stall (auto-memory
`feedback_workflow_qa_when_subagents_stall`; empirically the path this whole session used at $0). The Agent-tool
spawn stays as a documented FALLBACK. One checked-in `VERDICT_SCHEMA` / `ENVELOPE_SCHEMA`. Pin `model: opus`
(session/frontmatter model-inheritance nuance — the workflow inherits Main's model unless overridden). Update
`docs/runbooks/per-step-protocol.md` + `qa.md` + `researcher.md` + CLAUDE.md. **GUARDRAIL:** the protocol MUST
require Main to transcribe the returned verdict VERBATIM into `evaluator_critique.md`. Preserves the
single-Q/A-per-step rule + the file-based fresh-respawn cycle-2 pattern. **RIDER TRAPS to NOT adopt** (they ride
on this Workflow path but are rejected): R1 `/goal`-to-PASS auto-fix loop (erodes evaluator independence), R4
model-swap on stall (the stall is model-agnostic), R11 Monitor mtime watchdog (contradicts the do-not-poll rule).

### 71.2 (P1) — Layer-2 structured outputs + kill silent-failure classes  [kept #4/#9/#17]
**Grounding:** https://platform.claude.com/docs/en/build-with-claude/structured-outputs — "guarantee
schema-compliant responses through **constrained decoding**"; "No more JSON.parse() errors"; GA on Opus 4.8 /
Sonnet 4.6 / Haiku 4.5; `output_config.format={type:"json_schema"}`; grammar cached 24h. Plus
https://www.anthropic.com/engineering/building-effective-agents (evaluator-optimizer).
**Change:** adopt `output_config.format` on the two highest-frequency Claude JSON sites (the quality gate + the
classifier in `multi_agent_orchestrator.py`). Fix the **clobber** bug at `multi_agent_orchestrator.py:883-885`
(re-verified on HEAD: `:883` "if we can't parse, treat non-PASS as improvement" → `:885 return gate_response`,
substituted into the user answer at `:461-462`) — fail-SAFE: on a parse failure `return None` so the original
analyst answer is preserved, never the gate response. Delete the fabricated `evaluator_agent.py:_run_spot_checks`
(`:513-515` hardcoded 1.02/0.95/0.99, `def :496`) or wire it to a real backtest — never fabricated numbers. Insert
an independent evaluator between `skill_optimizer.propose_skill_modification` (`:300`) and `apply_modification`
(`:399`, mechanical-only today). **Gemini paths keep `response_schema`, NOT `output_config`** (the Vertex API).
All Layer-2 (metered) — structured output is a **$0-delta correctness** upgrade, no effort bump.

### 71.3 (P2) — Q/A rubric hardening (WITHIN the single Q/A role)  [kept #6 + #8b; DROP #8a]
**Grounding:** multi-agent-research judge rubric ("**completeness** — are all requested aspects covered?") +
harness-design (separate doer/judge).
**Change:** add a **contract-completeness** dimension (does `experiment_results.md` cover every contract
criterion?) + an **adversarial red-team leg** to `qa.md`. **DROP the worst-of-N self-consistency (#8a):** the
grounding was oversold — the 345,968-NAV bug was a *deterministic* catch, and a `money_path` tag is invented; N
identical samples add cost without independent signal. Keep it inside the single Q/A role (no re-split).

### 71.4 (P2) — coverage-ledger research gate + audit-class loop-until-dry critic  [kept #7/#11]
**Grounding:** multi-agent-research ("decides whether more research is needed … exits the research loop") +
building-effective-agents ("stopping conditions (such as a maximum number of iterations)").
**Change:** turn the fixed ≥5-source floor into a **coverage-ledger** gate (add a `coverage` field to the JSON
envelope; keep the ≥5 floor as a hard floor; collapse when 1 sub-question remains). Add an **audit-class
loop-until-dry** completeness critic (3-pass ceiling, empty-delta stop, say "evidence" not "proof", scope only to
unknown-denominator audits). Cross-link the rules files; do not duplicate. Because the "JSON envelope (always
emit)" directive is echoed in `researcher.md` (not only `.claude/rules/research-gate.md`), adding the `coverage`
field edits that agent file → **separation-of-duties + roster-snapshot handling applies** (see Binding
constraints). Keep the canonical field spec in `research-gate.md`; `researcher.md` references it.

### 71.5 (P3, OPERATOR SIGN-OFF) — effort/model config HYGIENE  [kept #5/#13/#14/#15]
**Grounding:** https://platform.claude.com/docs/en/build-with-claude/effort ("Reserve max … relatively small
quality gains … overthinking"; Opus 4.8 start `xhigh`) + https://code.claude.com/docs/en/model-config
(`CLAUDE_CODE_EFFORT_LEVEL` env precedence; `settings.effortLevel` accepts `xhigh` not `max`; fallback capped at
three after dedup, example `[claude-sonnet-5, claude-haiku-4-5]`).
**Change (framed as CONFIG HYGIENE, NOT the rejected "cost-analysis-of-dead-config"):** revert the DEAD
`model_tiers.py::EFFORT_DEFAULTS` `max` override (`:261-265`, whose own comment `:255-260` says "revert after
closure"; stale Sonnet-4.6 comment `:251` on the now-Opus `mas_qa`) + update the phase-59.1 test; pin Main's tier
`xhigh` deterministically (env; strip `max` which `settings.effortLevel` rejects); tighten the Layer-3
`fallbackModel` chain → `[sonnet-5, haiku-4-5]`; prune the ~74 stale Fable-window comment lines in
`researcher.md`/`qa.md` **but PRESERVE the phase-29.2-permanent `model: opus` + `effort: max` pins** (not a Fable
artifact). Operator sign-off because it touches the documented effort policy. Edits `.claude/agents/*.md` →
separation-of-duties + roster-snapshot handling applies.

### 71.6 (P3) — subagent context hygiene + automated stress-test  [kept #16 + #12-descoped]
**Grounding:** multi-agent-research ("pass **lightweight references** back to the coordinator") + harness-design
(stress-test doctrine, "stripping away pieces … no longer load-bearing").
**Change:** `researcher.md`/`qa.md` instruct the subagent to return an **envelope + ≤200-word summary + brief_path**
(not the full brief/critique through Main's context). Delete the stale self-evaluating driver
(`scripts/mas_harness/cycle_prompt.md` + `.bak` plists) after the stress-test confirms it's dead weight; add
SessionStart grep asserts. **NO new autonomous loop; local-only; do NOT touch the live
`run_harness.py::_default_spawn_researcher` (`:1122`).** #12 (weekly scheduled self-audit) is DESCOPED to
report-only / manual (honors the background-agent-unauthorized-action memory). The envelope-return change edits
`researcher.md`/`qa.md` → **separation-of-duties + roster-snapshot handling applies** (see Binding constraints).

## The 15 REJECTED proposals — acknowledged (do NOT re-introduce)

Per the register + `research_brief_71.0.md §6`, each rejected proposal is disqualified for a specific reason; the
design MUST NOT silently re-introduce any:
- **R1** `/goal`-wrapped cycle-2 auto-fix — erodes evaluator independence.
- **R2** effort-gated hooks for conditional verification — weakens the live_check gate.
- **R3** worktree-isolated Q/A — reads STALE COMMITTED evidence, not the working tree.
- **R4** fallback model auto-downgrade on stall — the stall is model-agnostic.
- **R5** MCP `alwaysLoad` tuning for Q/A — zero-delta.
- **R6** hooks `continueOnBlock` for live_check retry — self-clears the audit gate.
- **R7** generator self-review punch-list — INVERTED grounding; re-imports banned self-eval.
- **R8** parallel researcher fan-out as default — 1M context doesn't exceed the window.
- **R9/R10** context-reset / compaction rituals — Opus 4.5+ already dropped resets.
- **R11** Monitor mtime stall-watchdog — contradicts the do-not-poll-transcripts rule.
- **R12** prompt caching on Layer-2 — no-ops below the 4096/2048 token floor.
- **R13** Layer-4 directive constrained-decoding + routing — dormant Layer-4 code.
- **R14** reconcile/cost-justify the Layer-2 effort=max drift — it's DEAD config, no live spend to remove
  (71.5 does the HYGIENE revert, NOT this cost framing).
- **R15** repeatable xhigh-vs-max effort eval — moot on the flat-fee rail.

RIDER-TRAP note: R1/R4/R11 ride on the KEPT 71.1 Workflow path, and R13/R14/R15 ride on the effort/structured
themes — 71.1 adopts the Workflow launch WITHOUT R1/R4/R11, and 71.5 does effort HYGIENE WITHOUT R14/R15's framing.

## Sequencing note
71.1 + 71.2 are the P1 wins (codify the stall-immune Q/A path; kill the Layer-2 silent-failure classes). 71.5
needs operator sign-off (effort policy). 71.1/71.3/71.4/71.5/71.6 ALL edit `.claude/agents/*.md` (only 71.2 is a
pure Layer-2 backend step — no agent-file edit) → each carries the separation-of-duties + roster-snapshot
handling above.
