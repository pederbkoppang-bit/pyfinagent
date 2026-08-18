---
name: recheck-head-before-returning-a-scoped-grade
description: A commit range in the tasking prompt goes stale mid-evaluation -- re-read git log before returning, and check whether A..B excludes the commit holding the work
metadata:
  type: feedback
---

Before returning a grade scoped to a commit range, re-run `git log --oneline -6`
and state the exact HEAD you graded. Also check the range's endpoints: `A..HEAD`
EXCLUDES A.

**Why:** in one 86.74 grade both traps fired. `cba60c0b..HEAD` was docs-only
because the guard and the mutation cell lived IN `cba60c0b`. And two commits
landed while I was working (`38ba13ad`, `a33a5117`) which RETRACTED the very
claim the prompt asked me to attack -- the author had refuted themselves after
sending the task. Grading the range as given would have attacked a withdrawn
claim and missed the real state.

**How to apply:** open with `git log --oneline` over the range AND `--stat`; if
the range is docs-only but the task describes code work, the work is in the
excluded endpoint. Close by re-reading HEAD. Grading the newer state is correct
when the movement made a claim more conservative and is recorded -- but say so,
and name the sha. Related: [[recheck-prior-remediation-list]],
[[pin-invariants-to-a-file-list]].

**Same trap, second shape (86.90): HEAD IS NOT A PRE-EXISTING BASELINE.** Twice
in that step -- in `experiment_results` §9.1 and in a masterplan `audit_basis` --
"this RED is not my change" was justified by `git worktree add --detach <path>
HEAD`. HEAD already contained the work commit, so that worktree excludes only
UNCOMMITTED edits; it cannot exclude the step. The conclusion happened to be
true, which is what makes it dangerous. What actually settles it, with no
worktree and no writes: `git log -S'<the failing rule text>' -- <file>` to date
the rule, plus `git diff <base> HEAD -- <file> | grep -c '<the symbol>'` to show
the diff never touched it. A conclusion that is right for a reason that does not
establish it is still a finding.
