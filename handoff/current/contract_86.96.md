# Contract -- step 86.96

**Step:** 86.96 -- the Q/A rail's args arrive as a JSON STRING on most launches
and can fail to re-parse, killing the spawn before any agent runs -- and the
workaround altered immutable criteria text in transit.
**Date:** 2026-08-17 · **Author:** Main (operator-attended harness-repair session)
**Research gate:** PASSED (recomputed) -- `research_brief_86.96.md`, 8 sources
read in full / 30 URLs / recency scan performed / brief COMPLETE on disk
(32,971 chars), run `wf_9e9ef2b7-70b`.

## Research-gate summary (what changes the plan)

The brief settled the diagnosis before any build: the minimal failing input is
ONE character -- a `}` closing the `[` opened at `extra.judge_these_specifically`
-- and it repairs under SUBSTITUTION but not insertion, so the bracket is WRONG,
not missing (rules out delta-boundary shear; §1, §6). Run-record args are
sha1-identical to the payloads (transport exonerated, §2). `classifyArgs`
already fails FAST and LOUD at `qa-verdict.js:75-99` -- the step is a DIAGNOSIS
and GUARD gap, not a policy gap. The census found FOUR failures, not two (2
bracket, 2 truncation) across both scripts; strings are 81.4% of launches with
4/394 failing vs 0/90 objects (§3-4). One pre-hardening run completed BLIND and
silently dropped `audit_class:true` (§5). Mechanism: idiom priming -- the
dict-valued field preceding the list reuses its `"},` close (§6). Two stale
premises corrected: `verify_workflow_args_boundary.mjs` is GREEN 96/0 (86.92
closed at e45c1bf6); zero failures are escaping defects.

## Hypothesis

The fail-fast is correct and stays; what is missing is (a) the executed
diagnosis trail the criteria demand, (b) a mutation-tested guard pinning
byte-verbatim round-trip of adversarial criteria through BOTH args shapes and
pinning the fail-fast against silent-fallback regression, and (c) the
documented object-first launch contract so no caller ever again needs the
workaround that mangled criteria text.

## Immutable success criteria (copied verbatim from .claude/masterplan.json)

1. the failure is REPRODUCED by execution before anything is changed, with the verbatim error, and the reproduction is shown to be deterministic rather than intermittent -- if it does not reproduce, say so and re-scope rather than building for a defect that is not there
2. the trigger is BISECTED from the two failing payloads rather than guessed: shrink and mutate them until the minimal failing input is isolated, and state explicitly that size and escaped quotes were each tested and either confirmed or ruled out, because both are contradicted by the cycle-2 payloads that parsed
3. the layer is localised -- caller serialisation, the Workflow args marshalling, or the script's own JSON.parse -- with evidence per layer, not by elimination
4. whatever the cause, a caller must be able to transmit the immutable criteria BYTE-VERBATIM, demonstrated by round-tripping a criteria array containing backticks, double quotes, newlines and non-ASCII, and asserting byte equality at the receiving end
5. the args-shape census is re-derived at execution time rather than copied (string vs object counts), and any figure quoted carries the command that produces it
6. a regression guard is added that would go RED if a verbatim-critical payload silently fails to round-trip, and it is mutation-tested with the control observed GREEN first
7. verdict semantics are UNCHANGED: nothing here may turn a non-PASS into a PASS

## Plan

1. **Reproduce (crit 1):** drive the shipped `classifyArgs` on both stored
   payloads via a node slice; verbatim error; run twice for determinism; also
   `python3 json.loads` positions 4939/5536 as the cross-parser control.
2. **Bisect (crit 2):** single-char substitution/insertion differentials at the
   failure offsets; SIZE ruled out by parsing an equal-length valid payload;
   ESCAPED QUOTES ruled out by naming parsed production payloads containing
   them. Commands beside every figure.
3. **Localise (crit 3):** per-layer table with executed evidence: caller
   (bracket-idiom analysis, both spawns identical shape), marshalling (sha1
   record==payload), script parse (node+python agree; refusal is correct).
4. **Round-trip (crit 4):** adversarial criteria array (backticks, double
   quotes, newlines, non-ASCII incl. nb-NO + CJK) driven through the script's
   own prompt-render boundary as OBJECT args and as a JSON-STRING args, byte
   equality asserted (sha256) at the received end.
5. **Census (crit 5):** re-derive object/string/failure counts over the live
   run-record corpus with the population rule stated.
6. **Guard (crit 6):** extend the existing args-boundary/prompt-render checker
   family with a round-trip section (fit decided at GENERATE against checker
   sprawl); mutation cells: silent-fallback-instead-of-throw on classifyArgs,
   round-trip assertion deleted, adversarial corpus neutered -- control GREEN
   first, byte-identical restore.
7. **Docs (crit 7 + step name):** comment at classifyArgs recording the
   four-event class, idiom-priming mechanism, and object-first launch contract.
   No behavioural change to either workflow script; criterion 7 demonstrated by
   the diff being comment-only there plus the semantics checks staying green.

## References

- `handoff/current/research_brief_86.96.md` (all sections; envelope COMPLETE)
- The two stored payloads (scratchpad, sha1-pinned in the brief §2)
- `.claude/workflows/qa-verdict.js:75-99` (classifyArgs), `research-gate.js` twin
- `scripts/qa/verify_workflow_args_boundary.mjs` (96/0), `verify_prompt_render_86_90.mjs` (95/0)
- RFC 8259, RFC 9413 (postel's-law caution), fine-grained tool streaming doc,
  claude-code#69085, arXiv 2501.10868 + 2605.02363, dasroot structured-output
  failure taxonomy, fsharpforfunandprofit property-based round-trip testing
  (read-in-full set, brief §8)
