---
name: replacement-stops-at-the-line-not-the-sentence
description: A "REPLACED the false sentence" claim whose diff is one line, while the sentence wrapped across two source lines -- the orphan half survives, still offering the command the replacement just called vacuous (86.79 c6)
metadata:
  type: feedback
---

When an artifact claims a false sentence was **REPLACED**, diff the hunk against the
sentence's FULL SOURCE SPAN, not against its first line. Prose in markdown wraps, and
a string-anchored replace matches one line.

**Why:** 86.79 cycle 6 shipped "F5 completed by REPLACEMENT". `git show <sha> -- <file>`
was `2 +-` -- one line in, one out. The sentence was:

```
So this file is the ask. **Nothing in `.claude/agents/qa.md` was modified by step
86.79** — verify with `git diff --stat .claude/agents/qa.md`.
```

Only line 1 was replaced. Line 2 survives as an orphan fragment that STILL offers the
working-tree diff the new note on line 1 explicitly names VACUOUS ("a working-tree diff
on a committed tree can never dissent"). Fourth correction of that one sentence; each
prior pass fixed the headline and left it. Related: [[a-correction-must-replace-not-accompany]],
[[regenerated-label-is-a-claim-check-the-diff]].

**Companion finding from the same cycle, same shape:** the fix for "a present-tense count
that does not reproduce" introduced THREE NEW instances of exactly that class -- cycle-6
anti-staleness marks reading "60 checks / floor 59" and "60/59 today", committed in the
SAME commit that raised the floor to 61 and the run to 62. See
[[carried-forward-residuals-go-stale]].

**How to apply:** on any REPLACED/REGENERATED/COMPLETED-FOR-THE-CLASS claim --
(1) `git show <sha> -- <file>` and read the removed line's TAIL: if it ends mid-sentence
(no terminal punctuation, an unclosed `**`), the next source line is an orphan;
(2) grep the file for the replaced text's distinctive tail token, not its head;
(3) if the correction note states a CURRENT number, re-derive that number at the tree the
commit produced -- a mark written in the commit that moved the number is stale on arrival.
Both are evidence-quality, not criterion misses, when the underlying product is verified
independently; say so and let them queue.
