---
name: verbatim-paste-drift-arithmetic
description: Verbatim test-count pastes drift from committed files; cross-suite count arithmetic adjudicates transcription-error vs concealment (66.3)
metadata:
  type: project
---

"Verbatim" pytest pastes in experiment_results can be stale relative to the
COMMITTED test file (phase-66.3: docs + commit msg said "11 passed"/"11 new
tests"; committed file had 10 `def test_`, command yields `10 passed`).

**Why:** authors edit/merge tests after capturing the standalone paste and only
re-run the combined suite. Distinguish benign staleness from concealment with
arithmetic: if the doc's COMBINED total (32) equals actual-suite-counts summed
(10 + 9 sentinel + 13 observability), the final code was tested post-change and
the standalone number is a transcription artifact -- NOTE severity, not
CONDITIONAL. If the combined total only works with the CLAIMED count, suspect
an untested late change -- escalate.

**How to apply:** always `grep -c "def test_"` on BOTH worktree and
`git show <commit>:<testfile>`, re-run the immutable command yourself, then
check the doc's own totals for internal consistency before choosing severity.
Also expect live-day $ figures to drift between capture and evaluation
(66.3: $0.0013 -> $0.0025, a second ticket call) -- verify per-call math at
each capture time instead of demanding equality. See
[[scheduled-job-fix-evidence]] for the sibling lesson on evidence timing.
