---
name: survivor-needs-behavioural-differential
description: A surviving mutant is only a finding if it's a real regression — adjudicate every survivor with a behavioural differential, and assert the baseline row isn't an error object
metadata:
  type: feedback
---

A mutation that survives the suite is NOT automatically a vacuity finding. Before
reporting it, run a **behavioural differential**: same input classes, baseline vs
mutant, compare the returned values. Equivalent mutants (no behavioural diff on any
class) must be reported as equivalent, not as coverage gaps — and when they are
equivalent, name which guard *actually* does the work (qa.md §4c #11,
mis-attributed kill mechanism).

**Why:** phase-80.27 cycle 2 produced 3 survivors out of 17. Two of them (removing
the `sector_analysis` ladder guard entirely, narrowing it to one of three operands)
were EQUIVALENT — the later payload-completeness guard fully subsumes the ladder
guard, so the code comments crediting the ladder guard with the fail-safe are
inaccurate but nothing is unpinned. Only the third (narrowing the payload scan to
`stock_returns`) was a real regression. Reporting all three as findings would have
been three-quarters noise and would have blocked a correct P0 fix.

**Also:** my first differential run was itself vacuous — a `SyntaxError` in the probe
made every run return an identical error dict, so all three mutants read "EQUIVALENT".
Always print and eyeball the BASELINE row before trusting any diff; a comparison whose
two sides are both failures cannot fail. Same shape as the defects being audited.

**How to apply:** any time a mutation survives, before writing it up: (1) run the
mutant and the baseline over 4-6 input classes, (2) assert the baseline output is a
real result and not an exception/empty, (3) only classify as a finding if some class
diverges. See also [[mutate-the-flag-read-not-just-the-guard]] and
[[derived-scope-lint-use-xargs]] — same family: verify the instrument before trusting
what it reports.
