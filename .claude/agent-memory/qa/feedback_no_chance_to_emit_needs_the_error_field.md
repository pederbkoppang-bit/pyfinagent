---
name: no-chance-to-emit-needs-the-error-field
description: A "completed but never emitted" counter must exclude every no-chance-to-succeed cause, and the run record's per-agent `error` field names them (529, quota) -- excluding only `killed` fixes the instance and leaves the class, and on an append-only corpus the resulting red is PERMANENT
metadata:
  type: feedback
---

A guard that counts "ran to completion WITHOUT emitting the schema call" as
evidence of a NEW loss mechanism must first subtract every spawn that never had
the chance to emit. `killed` is one such cause. It is not the class.

**Measured, 86.84 cycle 6 (2026-08-17).** The step's immutable command went RED:
`POST-REMOVAL NON-EMITTER: 2 uncapped qa spawn(s) never emitted StructuredOutput`.
Both flagged spawns carried, in the *same* `workflowProgress` entry the collector
already reads four keys from, `error: "API Error: 529 Overloaded ..."`. One had
run 10 turns, one 38 -- neither was the exhaustion the guard exists to detect.
`collect()` read `agentId/agentType/model/toolCalls/tokens` and never `error`.
The previous cycle had closed exactly this shape for `killed`, on a Q/A finding,
and wrote a comment calling non-emission "the one shape that genuinely signals a
new loss mechanism" -- the instance was fixed, the class was not.
Corpus census of the field: 9 errored agent entries, **5 environmental**
(2x 529, 3x weekly/session/credit limit). The class was already on disk.

**Why it matters more than a normal false positive:** the population is the
append-only `~/.claude/projects` run-record corpus, and it INCLUDES the step's
own Q/A evaluation spawns. So (a) the red never heals -- those records are
permanent -- and (b) the gate is self-referential: every dropped grading cycle
reddens the thing being graded. The mutation harness refused to score a single
cell ("CONTROL IS RED -- the matrix is meaningless"), so the criterion demanding
a mutation matrix had zero live evidence.

**How to apply:** when a guard counts an ABSENCE (no emission, no row, no call),
(1) enumerate every reason the absence can occur and subtract the ones outside
the mechanism under test -- census the raw cause field, do not list from memory;
(2) ask whether the population can contain artifacts of the evaluation itself,
and whether it is append-only, because those two together make a transient
external fault a permanent gate failure; (3) re-run the immutable command
yourself even when the artifact quotes exit 0 -- a corpus-derived gate can be
true at capture and false an hour later.
Related: [[feedback_run_status_is_not_agent_outcome]],
[[feedback_guard_from_instance_not_class]],
[[feedback_self_referential_counts_cannot_reproduce]],
[[feedback_the_guard_carries_the_defect_it_guards]].
