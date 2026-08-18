---
name: mutate-each-duplicated-site-individually
description: A matrix cell that mutates ALL N copies of a duplicated fix at once cannot detect a regression in ONE copy; mutate each site alone -- that is what finds the unguarded twin
metadata:
  type: feedback
---

When a fix is applied at N duplicated call sites, the author's matrix cell is
almost always "drop/break the thing" applied to **all N at once**. That cell
KILLS, so it looks like coverage. It is the weakest cell in the matrix: it
cannot distinguish "both copies are guarded" from "one copy is guarded and the
other is dead weight".

**Mutate each site INDIVIDUALLY.** The single-site mutant is the one that finds
the unguarded twin.

**Why:** phase-86.88 cycle 3 threaded a `risk_assessment_provenance` key into
the lite `full_report` on BOTH the Claude and Gemini paths
(`autonomous_loop.py:3302` and `:3544`, byte-identical blocks). The author's
cell M12 dropped it from BOTH -> KILLED, 1 failed. Measured:

```
M12  drop from BOTH persisted blobs      KILLED  1 failed
IND1 drop from CLAUDE blob ONLY          KILLED  1 failed
IND2 drop from GEMINI blob ONLY          *** SURVIVED *** 77 passed
IND3 pin False on BOTH                   KILLED  1 failed
IND4 pin False, GEMINI ONLY              *** SURVIVED *** 77 passed
```

Every kill came from ONE test that drives the Claude route only. The Gemini
half had zero coverage while the matrix read 12/12.

**Corollary -- "I measured the literal count == 2" is not a guard.** The
artifact credited the both-paths safety to having *measured* the count once.
`grep -rn "risk_assessment_provenance" backend/ scripts/` returned exactly the
2 PRODUCTION lines and zero assertions. A one-time census is evidence about a
moment; only an assertion is a guard. When an author says a count is "asserted",
grep for the assertion before believing the second site is protected.

**How to apply:** any diff that repeats the same block at N sites (both LLM
paths, both handlers, both routes). Enumerate the sites from source
(`grep -c` the literal), then run N single-site mutants plus the all-sites one.
Also check the guard's own breadth: a guard pinned on ONE key/instance
(`_lite_judge_produced_no_verdict` exactness pinned only via `reasoning`) leaves
the other keys surviving -- see [[guard-from-instance-not-class]] and
[[enumerate-every-position-at-a-recidivist-call-site]].
