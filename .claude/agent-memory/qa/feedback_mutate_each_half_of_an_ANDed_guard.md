---
name: mutate-each-half-of-an-anded-guard
description: When a fix ANDs a second predicate onto an existing guard, mutate EACH half separately -- the stronger half masks the weaker one's absence and every existing fixture is subsumed by it (86.85 c9 shape-half survivor; 86.110 c2 prefix-half survivor)
metadata:
  type: feedback
---

When a remediation turns a one-predicate guard into `A and B`, the matrix
almost always gets a cell for the NEW predicate and none for the OLD one.
Build the missing cell: neuter `A` and leave `B` live, then neuter `B` and
leave `A` live. **A cell that removes the WHOLE call (both halves at once)
does not cover either half.**

**Why:** phase-86.85 cycle 9 fixed a date guard by ANDing the shape regex with
`datetime.date.fromisoformat`. Cell M21 covered the calendar half. I built the
shape-half cell (`if not ISO_DATE_RE.match(s): return False` -> `if False:`)
and it **SURVIVED** the 31-check self-test AND 31 pytest regressions, control
GREEN first in both. Cause, measured: the only shape fixture anywhere was
`'2026-8-10'`, and `date.fromisoformat` rejects that too -- the new half
**subsumed every fixture the old half had**, so nothing could tell them apart.
The differential was real, not equivalent: `fromisoformat` accepts `'20260810'`
and `'2026-W32-1'`, both of which sort AFTER every hyphenated date, so a
backfilled older PASS lands LAST and takes the sequence from
`[C,C]` to `[C,C,PASS]` -- consecutive-CONDITIONAL 2 -> 0, the
escalation-CLEARING direction the guard existed to stop.

**Second instance, opposite polarity -- the REDUNDANT half is the tested one
(86.110 cycle 2).** A concurrency-tolerance rule shipped as
`return after.startswith(before) and len(after) > len(before)`. Its cell
mutated the WHOLE return to `True` and scored KILLED. I mutated only the
prefix half away (`return len(after) > len(before)`) and the entire 13-test
suite stayed **GREEN**. Cause: all three fixtures in the unit test are
length-only discriminable -- the "a REWRITE is not an append" case supplied 20
bytes against a 42-byte snapshot, so it varies TWO properties at once
(not-a-prefix AND shorter) and cannot attribute; truncation is `b""` and
no-growth is the snapshot itself. And the length clause is **redundant** given
startswith (equal content is short-circuited upstream by a sha compare), so the
suite exercised only the half that does nothing. **A fixture must vary ONE
property**: add a non-prefix rewrite that is LONGER than the snapshot.

**How to apply:** on any `and`/`&&` added to a guard, ask "which inputs does
ONLY the old half refuse?" and check a fixture exists for one of them. If the
new predicate is strictly stronger on every fixture on disk, the old half is
untested by construction. Also note the AST coverage checkers miss this
entirely: `return False` is not a failure code, so a two-branch helper can
report `guards: 21 covered: 21 uncovered: 0` with a whole branch unmutated.
Related: [[feedback-mutate-the-flag-read-not-just-the-guard]],
[[feedback-survivor-needs-behavioural-differential]],
[[feedback-the-guard-carries-the-defect-it-guards]].
