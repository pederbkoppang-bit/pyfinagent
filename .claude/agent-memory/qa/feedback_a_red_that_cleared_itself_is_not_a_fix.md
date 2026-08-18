---
name: a-red-that-cleared-itself-is-not-a-fix
description: When a gate was RED last cycle and is GREEN now, run the PRE-fix source on today's data before crediting the fix; 86.84 c8's guard was inert and the red cleared because the corpus rewrote itself
metadata:
  type: feedback
---

Before crediting a fix with turning a gate green, run the **pre-fix source against
today's data**, and delete the new guard from the post-fix source and run it again.
Two commands; they separate "the fix worked" from "the input changed".

**Why:** 86.84 cycle 8. Cycle 7 FAILED because the immutable command was red: two
529-errored `qa` spawns counted as post-removal non-emitters. Cycle 8 added
`errored=bool(entry.get("error"))` + an exclusion, and reported "the two 529 entries
are now counted under `errored_n` and excluded from the loss signal, correctly."
Measured: **`errored_n = 0`**. HEAD's pre-fix source was equally green. Removing the
new clause left it green. The corpus had a total of 7 error-bearing entries, all
2026-07-18..08-06, none `qa`, none 529.

What actually cleared it: a **same-runId re-dispatch REWROTE the run record**
(`birth 10:29:44Z -> mtime 10:44:12Z`, 1 `workflow_agent` entry but **2 transcripts on
disk**). The failed 38-turn and 10-turn attempts still sit on disk carrying
`API Error: 529 Overloaded`; their entries are gone from `workflowProgress`, which is
what the collector iterates. Contrast the in-script 86.81 retry
(`wf_078f4125-57a`): `birth == mtime`, **2 entries**, failed attempt visible. **Retry
appends; re-dispatch replaces.** So the artifact's premise "the corpus is append-only,
the red is PERMANENT" — which also shipped in a production code comment — was false,
and the guard was never demonstrated on real data (only on an injected fixture).

**How to apply:** any time a prior cycle's blocker is reported closed, (1) `git show
HEAD:<file>` into a temp module and run the gate; (2) source-mutate the new clause out
and run it again; (3) if both stay green, the guard is INERT and the credit is
misplaced. Then ask what changed in the INPUT — `stat` birth vs mtime on the data
files, and count records-vs-transcripts. A record store you assumed was append-only is
the first thing to check: it also means the gate has a silent **false-negative**
channel, because the evidence it would fire on can be deleted out from under it.
Related: [[feedback_a_correct_observation_can_credit_the_wrong_mechanism]],
[[feedback_stub_fallback_is_not_a_production_default]],
[[feedback_run_status_is_not_agent_outcome]].
