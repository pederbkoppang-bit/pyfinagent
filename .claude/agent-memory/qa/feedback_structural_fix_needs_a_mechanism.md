---
name: structural-fix-needs-a-mechanism
description: When an author says a fix is "structural" / "generated" / "cannot happen again", grep for the mechanism; absent a checked-in script or hook it is a one-off act and the durability claim is an Unjustified_Inference
metadata:
  type: feedback
---

When remediation prose claims a fix is **structural** -- "it is GENERATED from
X", "it cannot silently lag again", "this is no longer hand-maintained" --
search the whole tree for the mechanism before accepting it. If no checked-in
script, hook, or test performs the regeneration, the claim describes a one-off
act performed in-session, and the durability consequent does not follow.

**Why:** phase-82.0 cycle 6. Cycle 4's worst finding was that a Q/A verdict went
untranscribed *after* the same finding (F5) had been declared FIXED. Main's
cycle-5 remedy said `evaluator_critique_82.0.md` "is GENERATED from the persisted
returns, so it cannot silently lag them again". The factual half was true and I
verified it byte-exact (all 4 cycle reasons matched their JSON at
1742/3687/5255/6038 chars). But a whole-tree search found **no generator** in
`scripts/`, `.claude/hooks/`, or `.claude/workflows/` -- only the file's own
self-description. The lag mechanism was already observable: a newer
`_cycle5_ERRORED.json` postdated the .md and was absent from it. A claimed
structural fix with no artifact is the same shape as the regression it repairs.

**How to apply:** split the claim in two and grade each half. (a) Did the
described state actually get produced? -- verify by byte-comparison / re-derivation.
(b) Is there a mechanism that reproduces it? -- `grep -rl` for code that reads the
inputs and writes the output, excluding `.venv`/`node_modules`/`.git`. Half (a)
passing does not carry (b). Look for a newer input that the output does not yet
reflect -- that is the cheapest positive proof the lag is still possible. Severity
is WARN, not blocking, when the concrete harm is closed for everything that
currently exists; the fix is a queued step, not a revert. Related:
[[verify-own-completed-action-claims]], [[rederive-the-label-not-just-the-number]].
