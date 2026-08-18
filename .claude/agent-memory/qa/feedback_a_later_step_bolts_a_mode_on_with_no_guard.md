---
name: a-later-step-bolts-a-mode-on-with-no-guard
description: A flag/mode added by a LATER step to an EARLIER step's file inherits ZERO coverage -- the self-test predates it and nobody owns it; grep the test for the flag name before trusting a green suite
metadata:
  type: feedback
---

When step B edits step A's product file to add a mode, A's self-test does not
grow to cover it. The suite stays green, the mutation matrix stays 16/16, and the
new mode has **no guard at all**. Grep the test body for the flag name.

**Why (86.21 cycle 7, measured).** `scripts/qa/verdict_history_86_21.py` gained
`--evidence-only` from **phase-86.78** (commit `9b4d5281`), whose entire purpose
is a bias control: withhold the threshold from the judge. `self_test()` calls
`_report(...)` five times and **never once with `evidence_only=True`**. I deleted
the early return in that branch and ran it:

- self-test rc **0** (green), CLI exit codes unchanged (0/0);
- the mutant printed `consecutive     : 2` and
  `auto-FAIL armed : True  (a further CONDITIONAL would be the 3rd)` in the
  judge-facing mode -- the exact payload 86.78 exists to suppress.

Same shape one seam over: mutating `main()` to pass `evidence_only=False` also
survived. Two survivors, one bias control, zero guards.

**A second tell from the same run:** the artifact's pasted matrix header carried
`md5 : 142f6be...` while the shipped file is `b8c0370a...` -- because B edited
A's file after A's last capture. **An md5 line in a pasted capture is a free
staleness detector: recompute it.** It told me which commit the capture came
from and, from there, that a later step owned the untested code.

**How to apply.** For every product file: `git log --format='%h %cI %s' -- <file>`
and look for commits whose subject names a DIFFERENT step. Anything those commits
added is presumptively unguarded. Then confirm by grepping the self-test for the
new symbol before you accept a green run.

Related: [[feedback_matrix_oracle_inherits_selftest_blindspots]],
[[feedback_enumerate_entry_points_not_the_main_path]].
