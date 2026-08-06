---
name: two-mutant-forms-separate-artifact-from-kill
description: Build TWO differently-constructed mutants for the same property to tell a genuine kill from a construction artifact; and re-measure matrix rows on the immutable command's OWN scope, comparing named tests not counts
metadata:
  type: feedback
---

When auditing an author's mutation matrix, build a SECOND mutant for the same
property using a different construction, and compare the kill SETS. A single
mutant conflates "the property is guarded" with "my mutant happened to break
something else".

**Why:** phase-82.27 cycle 3. Main's M5 (criterion 2: `generate_report` must emit
no pbo) was built by renaming `generate_report` -> `_orig_generate_report` and
adding a `def generate_report(*a, **k)` wrapper. It killed 2 tests and Main
disclosed a caveat that the second kill was an artifact of the wrapper's `*args`
parameter list. I could not confirm that by reading -- I confirmed it by building
M5b, a signature-PRESERVING injection (`report.setdefault("analytics",{})["pbo"]=0.0`
just before `return report`). M5b killed exactly ONE test: criterion 2's real
behavioural guard. So the caveat was correct, and on the wider scope it was
UNDERSTATED -- M5's third kill (`test_generate_report_still_does_not_emit_a_pbo`
in the paired 82.23 suite) is ALSO a signature assertion, i.e. two of three kills
were the same artifact. Reading the mutant would never have settled it.

**Second half of the lesson: a kill count is SCOPE-DEPENDENT.** Main's rows were
measured on the step's own 19-test suite; the immutable verification command spans
TWO files (40 tests). Three rows (M5, M9, M10) each kill one extra test in the
paired suite. Every row was still correct -- the under-report was conservative and
no named test was wrong -- but a reviewer comparing counts against the immutable
command's scope would read a mismatch as a defect. **Compare the NAMED TESTS, not
the counts, and state which scope you measured on.**

**How to apply:** in any cycle whose criterion is "the matrix names, for each
guard, a mutation that makes it fail", re-run every row yourself with
`pytest -rf` over the immutable command's full scope, tag each failing node by
file, and diff the SET against the row's claim. Where the author flags a caveat
about their own mutant, build the alternate construction rather than reasoning
about it. Run mutants in-memory via sys.modules injection so the tree is never
written (see [[mutate-without-touching-the-tree]]), assert the target substring
occurrence count BEFORE mutating, and `ast.parse` every mutant so a
syntactically-broken file cannot masquerade as a strong kill.

Related: [[killed-mutant-needs-behavioural-differential]],
[[survivor-needs-behavioural-differential]],
[[enumerate-every-position-at-a-recidivist-call-site]].
