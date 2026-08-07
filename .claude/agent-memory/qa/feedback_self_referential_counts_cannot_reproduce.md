---
name: self-referential-counts-cannot-reproduce
description: a count over handoff/** grows from the act of writing the artifact that states it, so it can never reproduce; grade it NOTE only if the committed measurement reproduces
metadata:
  type: feedback
---

Before escalating a non-reproducing number under qa.md section 4b, ask whether the
POPULATION it counts includes the artifact stating the number. A grep across
`handoff/**` (.md/.json) is self-referential: writing the sentence, transcribing
the prior verdict, and dropping the raw Q/A return into `qa_returns/` all ADD
matches. Such a number is stale the instant it is saved -- the author cannot win.

**Grading rule.** Split the claim in two:
- the COMMITTED measurement (the one the criterion names) -- must reproduce
  EXACTLY, no tolerance. If it does not, that is the finding.
- a parenthetical, explicitly-designated NON-committed figure over a
  non-stationary population -- report the drift with your measured value and the
  mechanism, grade NOTE, do not cap the verdict.

**Why:** phase-83.0.3 cycle 3 (2026-08-07). `experiment_results:59` said the
unqualified grep "returns 297"; I measured 303. The committed qualified form
(`grep -rnP --include='*.py' ...` -> 51) reproduced exactly, as did the AST census
(5 sites, EMPTY symmetric difference, HEAD-state 4) and the byte-diff of the
criterion-6 block. The +6 was fully explained by artifacts born after the cycle-2
measurement (`qa_returns/wqwtohg6s.output.json` alone = 3). Escalating that to
WARN would have auto-FAILed a step under the 3rd-CONDITIONAL rule over a number
that is structurally unstatable. Contrast the cycle-2 WARN, correctly capped: a
stale `:86` line reference made the EVIDENCE UN-FINDABLE -- a pointer, not a
parenthetical.

**Discriminator to apply:** does the number POINT AT evidence (line ref, file
path, member list, the criterion's own deliverable)? Cap it. Or is it colour
commentary over a growing corpus? NOTE it.

**Always rule out self-contamination first** ([[probe-self-contamination-shared-module]]):
grep `handoff/audit/*.jsonl` for your own search term -- the PreToolUse hook logs
tool inputs, so your own probe commands can inflate the count you are auditing.
On 83.0.3 those files had 0 matches, so the drift was real, not mine.

**Residual to still call out:** a blanket "every number here was re-derived after
the final edit -- nothing carried forward" is falsified by ONE carried number.
Name it, even at NOTE severity. See [[measure-the-capture-you-didnt-take]] and
[[rederive-the-label-not-just-the-number]].
