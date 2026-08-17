# experiment_results -- step 86.72 (GENERATE, 2026-08-17)

Contract: `contract_86.72.md`. Research gate: PASSED (brief
`research_brief_86.72.md`). Verbatim command evidence: `live_check_86.72.md`.

## What was built

The F2 research-on-demand leg for the LIVE rail -- the mechanism
`run_harness.py` has always had (5 `research_needed` references) and the
Workflow rail never did (0 at HEAD before this GENERATE):

1. **`VERDICT_SCHEMA` gains OPTIONAL `research_needed` (boolean) and
   `research_brief_spec`** (object, `additionalProperties:false`, required
   4-key F2 shape: objective / output_format / tool_scope /
   task_boundaries). Neither is in `required` -- absence is a normal
   verdict. The phase-86.31 no-schema-change doctrine note is answered in
   place: that rejection weighed an audit field nothing required; these
   fields are mandated by this step's immutable criterion 3.
2. **`enforceResearchRouting(verdict)`** -- a pure caller-side function
   beside `enforceEscalation` (score inside, routing outside): reads the
   judge's optional fields, emits `research_needed` (null when absent --
   absence is never coerced to false), the echoed spec, and
   `next_action_on_research_needed` deterministic guidance carrying the
   bounds VERBATIM: at most 2 re-research rounds per step (Tmax=2),
   stagnation (a round adding no new read-in-full sources) ends the loop,
   floors unchanged and enforceGate recomputes gate_passed regardless.
3. **Surfaced BESIDE the verdict, never inside**: `research_routing` joins
   `escalation` in the merged return with its own leak-guard (the derived
   fields may never surface as top-level judge output; the judge-authored
   `research_needed`/`research_brief_spec` legitimately live inside the
   verdict and are excluded from the leak set by name).
4. **The judge is TAUGHT the field** -- `.claude/agents/qa.md` gains a
   "Research-on-demand" section (an ADDITION, live immediately on the
   Workflow read-at-runtime path per the snapshot doctrine): when to set it
   (evidence/knowledge gap the executor cannot close by editing), the
   mandatory 4-key spec, and the prohibitions (never to soften a FAIL,
   never past the caller's bounds). Flagged in harness_log for operator
   review per the separation-of-duties rule on agent-file edits.

## The criteria, discharged

1. **Absence re-verified with named controls, before/after**: positive
   control `research_needed` -> run_harness.py 5 hits; qa-verdict.js at
   HEAD (pre-GENERATE) 0, working tree now 7; research-gate.js 0 (correct
   -- the signal originates in verdicts); negative control
   `ZZZ_NO_SUCH_86_72` -> 0. Commands quoted in live_check §1.
2. **Per-step split re-derived over the wf_* corpus** (population rule:
   every agent transcript's first user message with a parseable step id,
   role-classified by prompt marker): the six highest-spawn steps -- 86.85
   (12 qa), 86.84 (8), 86.74 (7), 86.94 (6), 86.97 (5), 86.71 (5) -- all
   show ZERO researcher re-engagement. The audit-basis claim is CONFIRMED
   on today's corpus.
3. **The mechanism, driven at the checker level** (contract plan 4's
   honest fallback, stated as such): `verify_prompt_render_86_90.mjs`
   section [8] drives the REAL `enforceResearchRouting` (brace-extracted
   with loud anchors, never a copy): research_needed=true + spec ->
   surfaced with guidance carrying "at most 2"; absent -> null/null;
   false -> false/null. A verdict carrying the signal produces the
   researcher-spawn instruction; a verdict without it produces none. No
   live verdict has yet set the field (it shipped this cycle); the first
   live use will be recorded when it happens.
4. **Difficulty assessment**: caller-supplied tier retained and justified
   (the research brief's finding; the researcher cannot self-assess tier
   without seeing the task first -- the caller owns difficulty, the gate
   owns floors).
5. **Deep tier NOT enabled** -- ASK-1 in the contract records it with
   measured cost (~190-210K/run moderate, 1.5-2.5x plausible for deep);
   decision parked at 86.73.
6. **Floors untouched**: research-gate.js diff this cycle is ONE comment
   block (the 86.96 R3-tail twin note); `research_needed` count there is 0;
   checker family green incl. enforceGate recomputation paths (124).
7. **Verdict semantics unchanged, demonstrated**: the routing object is
   computed AFTER the verdict from the verdict, cannot write `verdict`
   (checker cells), the leak-guard throws on surfacing, and
   `verdict_unmodified` still computes. Section [8] cells
   8-routing-signal-dropped and 8-tmax-bound-removed are KILLED; the
   whole family is green (126/124/96).
8. **Mutation-tested with control green first**: section [8]'s four cells
   (recorder-neutered, recorder-hardcoded-false, routing-signal-dropped,
   tmax-bound-removed) each KILLED; the tmax cell's FIRST mutant was too
   weak (left both probe substrings intact) and SURVIVED -- disclosed,
   sharpened to remove the actual bound, then KILLED. The section-[8]
   slicer itself failed loud on its first run (the naive first-brace grab
   hit `opts = {}` -- the exact trap the 86.85 cycle-10 evaluator
   documented) and was fixed to paren-match the signature; the loud
   content assertion is what caught it.

## Honest limits

- The end-to-end leg (a REAL judge setting research_needed on a REAL step,
  Main spawning the gate before the next GENERATE) has not yet occurred --
  the field is hours old. The checker-level drive is the executed proof,
  per the contract's stated fallback; the protocol seam (Main acting on
  `next_action_on_research_needed`) is prose + guidance text, enforceable
  only by the actor it instructs.
- `judge_was_told_consequence` computation (86.78's fix, same edit pass)
  shares the caller_text plumbing; its census and evidence live in
  86.78's artifacts, not here.
