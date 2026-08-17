# Contract -- step 86.72

**Step:** 86.72 -- the re-research leg of the harness loop never fires on the
rail that actually runs -- a failed fix is re-attempted from Main's own
reasoning, never sent back for more documentation.
**Date:** 2026-08-17 · **Author:** Main (operator-attended harness-repair session)
**Research gate:** PASSED (recomputed) -- `research_brief_86.72.md`, 8 sources
in full / 28 URLs / recency scan / brief COMPLETE (37,369 chars), run
`wf_bab18828-6a8`.

## Research-gate summary (what changes the plan)

- **This is an EXTENSION of Anthropic's harness design, not a restoration**:
  harness-design documents only a QA->Generator backward edge; the QA->RESEARCH
  edge's prior art is SEMA-RAG's E-Agent -- a sufficiency flag `s_t` + gap
  description `g_t` + follow-up query set, STRUCTURALLY SEPARATE from
  answer-correctness (ablating it costs -6.37 to -8.40 points).
- **Bound the loop THREE ways** (sufficiency reached; Tmax; stagnation), with
  Tmax=2 -- the critique-revision literature independently converges on 2-3
  rounds.
- **Schema evolution**: `research_needed` lands as an OPTIONAL field in
  VERDICT_SCHEMA (FULL-compatible per Confluent; REQUIRED would break BACKWARD
  compatibility -- criterion 6 satisfied by construction).
- **Route on the signal OUTSIDE the judge**, in the shape of
  `enforceEscalation`: telling a judge what its verdict triggers causes
  leniency (measured 58/72 cells; the 86.78 doctrine).
- **Difficulty tier stays CALLER-SUPPLIED**: the literature argues against
  self-assessment (Anthropic: "agents struggle to judge appropriate effort";
  Triage: self-allocation worse than random when binding). Justified in
  writing; the deeper policy question remains step 86.73's.
- **Adversarial-internal, weighed**: the repo's own stress test (86.100
  material) leaned toward PRUNING mechanisms; this step adds the SMALLEST
  mechanism that closes the measured gap (9 qa runs / 0 researcher on the worst
  step) rather than an orchestration layer.

## Hypothesis

An optional `research_needed` + 4-key `research_brief_spec` in the Q/A verdict
envelope, surfaced by the launch script beside the escalation envelope and
enforced by Main's protocol (research-gate before the next GENERATE when set),
gives the live rail the F2 leg run_harness.py already has -- bounded at Tmax=2,
with no floor weakened and no verdict semantics touched.

## Immutable success criteria (copied verbatim from .claude/masterplan.json)

1. the absence of research_needed from the Workflow rail is re-verified with a positive-controlled search that names its control string, and any disagreement with this audit_basis is reported rather than silently adopted
2. the per-step run split by role is INDEPENDENTLY re-derived over the wf_* corpus with the population rule stated, and the claim that high-run steps show zero researcher re-engagement is confirmed or corrected
3. a mechanism exists by which a Q/A verdict can require MORE RESEARCH rather than only more fixing, and it is proven by DRIVING it end to end -- show a verdict carrying that signal causes a researcher spawn before the next GENERATE, and show a verdict without it does not
4. difficulty assessment is addressed explicitly: either the researcher assesses tier itself, or the caller-supplied tier is justified in writing against the operator's stated design, and whichever is chosen is demonstrated rather than asserted
5. the 'deep' tier is NOT enabled unilaterally -- it is an open operator decision recorded at research-gate.js:190-200 and must be raised as a numbered ask with its measured cost, not closed by this step
6. no research floor is weakened to make any of this pass: the >=5 sources-read-in-full and >=10 URL floors and the recency scan remain exactly as they are, and the step shows they are unchanged
7. verdict semantics are UNCHANGED: nothing here may turn a FAIL into a PASS, and that is demonstrated rather than asserted
8. mutation-test every new guard: revert it and show the check goes red, with the control observed GREEN first and a byte-identical restore

## Plan

1. **Criteria 1+2** (banked pre-design, restated with commands): control string
   `research_needed` hits `scripts/harness/run_harness.py` 5x (the F2 leg) and
   both rail scripts 0x; per-step split re-derived (36.8: 9 qa / 0 researcher;
   75.5: 7/0; 78.2: 6/0 -- confirmed).
2. **VERDICT_SCHEMA**: add OPTIONAL `research_needed` (boolean) and
   `research_brief_spec` (object: objective / output_format / tool_scope /
   task_boundaries -- the F2 4-key shape) to qa-verdict.js; the launch script
   surfaces them in its return beside `escalation`, computed OUTSIDE the
   judge's view of consequences.
3. **Protocol seam**: the qa-verdict.js return carries
   `next_action_on_research_needed` guidance text (deterministic, caller-facing)
   instructing the research-gate spawn before the next GENERATE, bounded: at
   most 2 re-research rounds per step (Tmax=2, stagnation = a round adding no
   new read-in-full sources ends the loop).
4. **Drive (criterion 3)**: a fixture-driven qa-verdict launch whose evaluator
   returns research_needed=true (checker-level drive through the stubbed
   runtime, per the house verify_prompt_render technique) -> the surfaced
   signal is present and well-formed; a verdict without it -> absent. Then ONE
   real end-to-end demonstration documented on the next live step that earns
   the signal (or the checker drive stands as the executed proof if no live
   verdict sets it during this step -- stated honestly either way).
5. **Criteria 4+5**: caller-supplied tier justified (research above); the deep
   tier raised as ASK-1 with its measured cost, not enabled.
6. **Criterion 6**: research-gate.js floors untouched -- shown by diff scope.
7. **Criterion 8**: mutation cells in the existing checker family -- signal
   dropped by the schema, guidance not surfaced, Tmax bound removed -- control
   green first.

## Numbered operator asks

- **ASK-1 (deep tier)**: research-gate.js's deep tier stays OFF. Cost basis:
  moderate-tier gates measured ~190-210K tokens/run this session; a deep tier
  (more sources + rounds) plausibly 1.5-2.5x that. Enable only on operator
  sign-off at 86.73.

## References

`research_brief_86.72.md` (SEMA-RAG arXiv 2606.06324; Confluent schema
evolution; arXiv 2606.27416, 2605.13414, 2605.17101; Anthropic built-multi-agent
+ harness-design; emergentmind critique-revision survey); run_harness.py F2
leg; qa-verdict.js VERDICT_SCHEMA + enforceEscalation; 86.73 (open policy);
research-gate.js:190-200 deep-tier note.
