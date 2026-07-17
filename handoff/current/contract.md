# Contract — step 71.0 (Harness + MAS upgrade design pack)

**Phase:** phase-71 | **Step:** 71.0 (phase opener) | **Priority:** P1 | harness_required: true
**Cycle:** 1 | Date: 2026-07-17 | **Type:** design + research ONLY (offline, $0, NO production code). live_check: none.

## Research-gate summary (gate PASSED)

Researcher via Workflow structured-output (Opus 4.8, $0). Envelope: **gate_passed=true**, tier=complex,
**7 external sources read in full**, 13 snippet-only, 20 URLs, recency scan performed, 10 internal files
re-anchored on HEAD 7d54d30d. Brief: `research_brief_71.0.md`. It re-validated every grounding (with URLs) AND
confirmed every register fact still holds on HEAD, AND refined the design.

## Hypothesis / design (to be written into design_harness_mas_71.md)

The 2026-07-16 self-audit register (`harness_proposals.json`, 17 kept / 15 rejected) is converted into a design
pack that grounds each phase-71 change in a REAL current feature/doc:
- **71.1 (P1)** — codify the Workflow structured-output path as the first-class Q/A (+Researcher) launch (verdict =
  captured return value, immune to the 6x end-flush stall), Agent-tool as fallback, checked-in verdict schema, pin
  `model: opus`, Main transcribes VERBATIM. Grounds: `code.claude.com/docs/en/workflows` (runtime tracks each agent
  result → resumable in-session) + multi-agent-research (resume-not-restart) + harness-design (separate doer/judge).
- **71.2 (P1)** — `output_config.format` structured outputs on the two Claude JSON sites; fail-SAFE clobber fix at
  `multi_agent_orchestrator.py:883-885` (return None, preserve the analyst answer); delete the fabricated
  `evaluator_agent.py:_run_spot_checks` (513-515 hardcoded 1.02/0.95/0.99); independent evaluator before
  `skill_optimizer.apply_modification`. Grounds: structured-outputs GA + evaluator-optimizer.
- **71.3 (P2)** — Q/A rubric: contract-completeness dimension + adversarial red-team leg. DROP the worst-of-N (#8a,
  grounding oversold). Grounds: multi-agent-research judge rubric + harness-design.
- **71.4 (P2)** — coverage-ledger research gate (add a coverage field; keep the ≥5 floor) + audit-class
  loop-until-dry completeness critic (3-pass ceiling, empty-delta stop). Grounds: multi-agent-research + building-effective-agents stopping conditions.
- **71.5 (P3, OPERATOR SIGN-OFF)** — effort/model config HYGIENE: revert the DEAD `EFFORT_DEFAULTS` max override;
  pin Main `xhigh` (env, strip max); fallback → `[sonnet-5, haiku-4-5]`; prune ~74 stale Fable comment lines but
  PRESERVE the phase-29.2-permanent `opus`+`max` pins. Grounds: effort doc + model-config doc.
- **71.6 (P3)** — subagent context hygiene (return envelope + ≤200w summary + brief_path) + delete stale
  self-evaluating scripts + SessionStart grep asserts (NO new autonomous loop, local-only; do NOT touch the live
  `_default_spawn_researcher`). Grounds: multi-agent-research lightweight-references + harness-design stress-test doctrine.

## Immutable success criteria (verbatim from masterplan.json 71.0)

1. design_harness_mas_71.md cites, for every phase-71 change, the specific Claude Code feature or Anthropic doc
   that grounds it (Workflow structured-output resumability; structured-outputs GA; evaluator-optimizer; effort
   guidance) -- with URLs
2. The design preserves every binding constraint verbatim (exactly-3-agents at L3, no self-eval, $0 metered L3
   rail, Layer-2 cost-sensitivity, local-only) and calls out the separation-of-duties + roster-snapshot handling
   for the steps that edit .claude/agents/*.md
3. The register's 15 REJECTED proposals are acknowledged so the phase does not silently re-introduce a rejected idea

Verification command (immutable):
`bash -c 'test -f handoff/current/harness_proposals.json && test -f handoff/current/design_harness_mas_71.md && grep -Eqi "structured.?output|workflow" handoff/current/design_harness_mas_71.md && grep -Eqi "verbatim|transcrib" handoff/current/design_harness_mas_71.md && grep -Eqi "clobber|883|structured" handoff/current/design_harness_mas_71.md'`

## Plan
2 (this contract). 3. GENERATE: write handoff/current/design_harness_mas_71.md (grounded per-step design +
constraints + rejected-ack). 4. write experiment_results.md. 5. Q/A (Workflow). 6. LOG. 7. FLIP.

## Boundaries (binding)
offline/$0; design-only (NO production code); exactly-3-agents at L3; no self-eval; Layer-2 cost-sensitivity;
local-only; the 15 rejected NOT re-introduced; every grounding a REAL current feature/doc with a URL; the
agent-file-editing steps (71.1/71.3/71.4/71.5/71.6 — all but the pure Layer-2 step 71.2) get
separation-of-duties + roster-snapshot handling; historical_macro FROZEN; harness stays 3 agents.

## References
research_brief_71.0.md; harness_proposals.json (the register); goal_phase71_harness_mas_upgrade_DRAFT.md.
