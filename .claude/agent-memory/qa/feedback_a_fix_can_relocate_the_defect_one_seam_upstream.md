---
name: a-fix-can-relocate-the-defect-one-seam-upstream
description: Cycle-2 fixed "the rail is read from a flag" by deriving it from model_name -- and built every new guard at the OLD seam; a mutant that hardcodes the model at the CALL SITE reinstated the identical misattribution and survived 29/29 (86.108 c3)
metadata:
  type: feedback
---

When cycle N+1 fixes "X was computed from the wrong INPUT", the fix introduces a
new input-producing seam. **Grade the new seam, not the repaired one.** The
author's new guards will cluster where the bug was found, because that is where
attention was; the relocated bug lives one call frame up.

**Why:** phase-86.108. Cycle 1: `current_rail()` read only
`paper_use_claude_code_route`, so every Gemini-served parse failure was stamped
`claude_code`. Cycle 2 replaced it with `resolve_rail(model_name)` mirroring the
client's real `startswith("claude-") AND flag` predicate, threaded `model_name`
through 4 emit sites via a NEW `_effective_model_name`, and added a 5-cell truth
table, a discriminating flag-only test, an emit-site threading test and mutation
cells M13/M14. All of that guards `_parse_json(model_name=..) -> record ->
resolve_rail`. **Nothing guards `run_debate -> _parse_json(model_name=???)`.**
Measured, in-memory via `pytest.main(plugins=[...])`, control green first:

    _effective_model_name -> None                : SURVIVED 29/29
    _effective_model_name -> "claude-opus-4-8"   : SURVIVED 29/29

The second is the cycle-1 defect verbatim, one seam upstream: every
Gemini-served debate failure stamped `claude_code` again.

**How to apply:** ask "what produces the corrected input, and is THAT under
test?" Then run the discriminating pair, because the two answers differ in
severity: an input that degrades to a stated `unknown` is a near-equivalent
mutant (honest), while a WRONG-but-in-vocabulary input is the finding. Also run
the converse probes to scope it fairly -- patching the resolver's return and
patching the forwarding both went red here, which is what proved the record seam
genuine and confined the finding to the call site. Severity WARN (CONDITIONAL),
not BLOCK, when a real behavioural guard coexists: qa.md 4c reserves BLOCK for
sole-coverage vacuity.

Related: [[class-guard-bound-to-the-helper-not-the-call-site]],
[[feedback-enum-membership-guard-passes-every-wrong-value]],
[[feedback-survivor-needs-behavioural-differential]].
