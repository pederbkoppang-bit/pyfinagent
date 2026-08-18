---
name: project-86-84-workflow-turn-cap-drops
description: StructuredOutput rail drops trace to frontmatter maxTurns; the error message is cause-blind by construction and a cap cannot be sized from capped runs
metadata:
  type: project
---

Step 86.84 researched the remedy for `agent({schema}): subagent completed without
calling StructuredOutput`. Root cause was given by Main (measured: all 39 dropped
`qa` agents ran exactly 30 turns = `qa.md:6`; all 9 `researcher` ran exactly 40 =
`researcher.md:6`; uncapped types 0 drops).

**Why the message misleads.** Decompiling 2.1.232: the LOCAL workflow path's result
object has no `resultSubtype` field, so it throws the same "never called
StructuredOutput" text whether the model ended in prose or the runtime cut the loop
at `maxTurns`. The `isolation:'remote'` path DOES read `resultSubtype` and names the
cause. **The message is cause-blind by construction** — never read it as evidence the
model chose prose.

**The sizing rule.** A run that used exactly N turns under a cap of N is a
RIGHT-CENSORED observation: it proves the requirement was >=N, never that N sufficed.
Phase-59.1 raised qa 12->30 and researcher 30->40 for this same class and it recurred,
because each new cap was fit to a distribution the previous cap created. The only
uncensored data are the uncapped types at 63 and 56 turns — both ABOVE 40.

**Why not "reserve the last turn".** StructuredOutput is a tool call, so it needs a
spare tool-use turn (`max_turns` "counts tool-use turns only"), but there is no
per-call turn budget in `agent()` opts and no primitive to force the terminal call
(feature request #20625 closed as not planned). Prompt-level self-rationing is
refuted by BAGEN arXiv:2606.00198 — models predict feasibility >70% after 60% of
budget is spent; the alarm fires only in the final 20%.

**Why:** the recommended remedy is to REMOVE the cap (documented default for an
absent `maxTurns` is literally "No limit"), not raise it again, and to bound cost
with token/spend budgets instead.

**How to apply:** if a future step proposes a bigger `maxTurns`, say what uncensored
data sized it. If a step cites GitHub #65500, version-check first — its nudge count
(2) and its "not catchable" claim are both stale against 2.1.232.

Technique used: [[grep-the-installed-cc-runtime-binary]]. See also
[[measure-dont-assert-claims]].
