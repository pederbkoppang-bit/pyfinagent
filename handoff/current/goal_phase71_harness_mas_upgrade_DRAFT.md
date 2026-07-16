# Goal — phase-71: Harness + MAS upgrade (stress-test doctrine, max-settings self-audit)

_Operator prompt 2026-07-16 (verbatim intent): "ultracode loops within Claude Code and we can
even better our harness and MAS agents — do this on max settings and then add it to our masterplan
as next steps."_

This is the CLAUDE.md **stress-test doctrine** turned on ourselves: "every component in a harness
encodes an assumption about what the model can't do; those assumptions are worth stress-testing as
models improve." A max-effort ultracode self-audit (2026-07-16; 37 agents, 5 research/audit
dimensions × adversarial critique; Opus 4.8 at `effort: max`; $0 on the Max rail) evaluated our
Layer-3 harness + Layer-2/4 MAS against the LATEST Claude Code + Anthropic capabilities and kept
**17 of 32 proposals** (15 rejected as cargo-cult / constraint-violating / mis-grounded). Register:
`handoff/current/harness_proposals.json`.

## The through-line

The single strongest signal — surfaced independently by 3 of the 5 dimensions — is that our harness
still launches Q/A (and Researcher) via the **Agent tool, which stalled 6× on end-flush 2026-07-11**
(model-agnostic), blocking unattended step closure. We already discovered the fix (run the role via
the **Workflow tool's structured-output**, where the verdict is the captured return value, immune to
the file-write hang), but it is **not codified** into the protocol or a saved workflow. Codifying it
is the P1 win and directly de-risks the away-ops autonomy (phases 62–65). The rest of the audit
hardens honesty and structure: guaranteed structured outputs (Anthropic GA feature) to kill silent
JSON-parse corruption on the Claude paths, independent adversarial evaluation before self-improvement
edits, a machine-readable verdict schema, and an effort/model posture that is truly "max where it
helps, cost-appropriate elsewhere."

## What the audit verified (all code claims confirmed by the critique agents)

- **Q/A/Researcher Agent-tool stall** — real, empirically diagnosed (auto-memory
  `feedback_workflow_qa_when_subagents_stall`); Workflow structured-output is the proven, documented
  fix (`code.claude.com/docs/en/workflows.md`: the runtime tracks each agent's result → resumable).
- **Silent JSON clobber** — `multi_agent_orchestrator.py:883-885` returns the gate response as the
  user-facing answer on a parse failure (real corruption bug); Claude paths hand-roll parsing while
  the Gemini debate paths already use `response_schema`.
- **Fabricated evaluator spot-checks** — `evaluator_agent.py:_run_spot_checks` returns HARDCODED
  `1.02/0.95/0.99` and can flip CONDITIONAL→PASS (latent footgun in a gate whose whole job is honest
  verification).
- **Un-reviewed self-improvement** — `skill_optimizer.apply_modification` writes a proposed prompt
  diff with only a mechanical uniqueness check; no independent LLM reviewer (evaluator-optimizer gap).
- **Stale effort override** — `model_tiers.py::EFFORT_DEFAULTS` pins mas_* to `max` via a comment
  that itself says "revert after closure"; config-vs-runtime drift on the MAS path.

## Definition of done

The 17 kept upgrades are implemented or explicitly dispositioned across the phase-71 steps, with:
1. **Harness stall eliminated:** the Workflow structured-output path is a documented, first-class,
   saved (`.claude/workflows/`) launch mode for Q/A (and Researcher), with a **verbatim-transcription**
   guardrail so the no-self-eval guarantee stays airtight; the Agent-tool path remains a fallback.
2. **Silent-failure classes closed:** guaranteed structured outputs on the highest-frequency Claude
   JSON sites; the quality-gate clobber bug fixed (parse failure preserves the original answer); the
   fabricated spot-check stub deleted or wired to a real backtest (never fabricated numbers).
3. **Judgment hardened:** Q/A rubric gains a contract-completeness dimension + an adversarial
   self-consistency check for P0/P1 money-path steps (WITHIN the single Q/A role — no re-split), and
   emits a machine-readable verdict JSON alongside the markdown.
4. **Self-improvement gated:** an independent evaluator reviews prompt/skill diffs before they land;
   audit-class steps get a loop-until-dry completeness-critic pass.
5. **Effort posture reconciled** against the documented CLAUDE.md policy (not a blind revert), Main's
   tier pinned deterministically, fallback chain tightened, stale Fable-window frontmatter comments
   pruned.

## Binding constraints (a step that violates any is out of scope)

- **Layer-3 harness stays EXACTLY 3 agents** (Main + Researcher + Q/A). No re-splitting Explore /
  harness-verifier. Adversarial/worst-of-N checks live WITHIN the single Q/A role, not as new agents.
- **$0 metered / Max-plan first-party rail** for the harness; **Layer-2 is per-ticker cost-sensitive**
  — no effort bump there without a cost analysis (the audit's effort proposals REDUCE or reconcile,
  they don't inflate).
- **Local-only** (Peder's Mac) — the scheduled self-audit uses local `/loop`/cron or a report-only
  routine, never assumed cloud fleet.
- **Self-evaluation by Main stays forbidden;** Q/A independence is load-bearing; verdicts are
  transcribed verbatim, never authored by Main.
- **File-based 5-file handoff protocol stays** the durable-state backbone; agent-file edits take effect
  only at NEXT session start (roster snapshot) and require separation-of-duties review.
- **Every change grounded in a REAL current feature/recommendation** (each step cites it); prefer
  high-leverage over complexity. The 15 rejected proposals are logged in the register for auditor trust.

## Step map (all 17 kept proposals covered)

- **71.0** — Design pack (research basis = the 2026-07-16 audit; design the codification + guardrails). Offline, $0.
- **71.1 (P1)** — Codify the Workflow structured-output path as the first-class Q/A + Researcher launch (saved `.claude/workflows/` script + protocol/qa.md/researcher.md/CLAUDE.md updates + verbatim-transcription guardrail). [merges the 3 convergent P1 findings]
- **71.2 (P1)** — Guaranteed structured outputs on the Claude JSON paths + fix the quality-gate clobber bug + delete/guard the fabricated evaluator spot-check stub. [Layer-2 honesty]
- **71.3 (P2)** — Q/A rubric hardening: contract-completeness dimension + adversarial self-consistency for P0/P1 money-path steps + machine-readable verdict JSON alongside markdown.
- **71.4 (P2)** — Evaluator-optimizer + coverage: independent adversarial evaluator before `skill_optimizer.apply_modification` + loop-until-dry completeness-critic for audit-class steps and the research gate.
- **71.5 (P3)** — Effort/model posture reconciliation: reconcile the stale EFFORT_DEFAULTS `max` override vs the CLAUDE.md policy, pin Main's tier deterministically, tighten the fallbackModel chain, prune stale Fable-window frontmatter comments.
- **71.6 (P3)** — Automated stress-test + context hygiene: a scheduled weekly report-only harness self-audit (this workflow, saved + scheduled) + subagents return envelope+summary+path (not full brief through Main's context) + prune the dormant self-evaluating driver.
