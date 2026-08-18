---
name: boundary-on-elements-not-the-container
description: A render/validation boundary applied to a collection's ELEMENTS leaves the CONTAINER type-guard upstream of it, where a wrong-shaped value is silently swapped for a default
metadata:
  type: feedback
---

When a step adds a "render-or-throw" boundary to caller-supplied fields, check the
**type guard that decides whether the boundary is entered at all**. It sits
upstream and usually still coerces silently.

**Why:** 86.90 fixed `[object Object]` by routing every prose field through
`renderArgField`. The diff's own removed line shows the author edited
`criteria.map((c,i) => ... + c)` into `... + renderArgField('criteria['+i+']', c, null)`
-- the ELEMENTS. One line above, `const criteria = Array.isArray(a.criteria) ? a.criteria : []`
was left alone, so a `criteria` passed as a **string or object** is discarded before
the boundary and replaced by the "(none passed in args -- read them from
masterplan.json)" placeholder. Zero throw, zero log, agent spawned anyway -- on the
field qa.md calls "the rubric". Unlike the six exotic holes that cycle found
(non-enumerables, Proxy), this one is **trivially JSON-reachable**.

**How to apply:** grep every `a.<field>` the script reads and classify each as
ROUTED / GUARDED-THEN-DROPPED / UNREAD. Drive the real script with the field in
each wrong shape and read the prompt actually handed to `agent()`, **control
(right shape) first**. Then bound the finding: census the corpus for real
occurrences (here 384 array / 1 absent / 0 wrong-shaped -> latent, WARN not BLOCK)
and check `git diff` whether the guard is pre-existing or a regression.
Companion checks: [[feedback_mutate_without_touching_the_tree]] (data:-URL module
import, no repo write), [[feedback_derived_scope_lint_use_xargs]].
