---
name: run-a-null-mutant-through-every-matrix
description: Run a comment-only NULL mutant through the matrix before believing any cell -- a tempdir-relocated mutant breaks __file__-relative sys.path imports, so 6/6 KILLED measured relocation, not mutation (86.71)
metadata:
  type: feedback
---

Before crediting a single cell of an author's mutation matrix, push a **NULL
MUTANT** through it: change one comment, nothing else. If the matrix scores that
as KILLED, every number in it is an artifact and the "0 survivors" claim is
unfalsifiable.

**Why:** measured on `scripts/qa/mutation_matrix_86_71.py` (86.71, cycle 1). It
wrote each mutant to a `TemporaryDirectory` and ran it as a subprocess -- correct
discipline, the real tree was never touched, md5 restore verified. But the subject
`scripts/harness/attempt_gate.py` opens with
`REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, REPO/"scripts"/"harness")`
and then `from attempt_budget import ...`. Relocated, `REPO` resolved to
`/private/var/folders/n4/...`, the import raised `ModuleNotFoundError`, the process
exited rc=1 before any gate logic ran, and every behavioural check failed. All six
cells scored KILLED; so did an unmutated copy; so did a comment. The filed claim
was "6/6 KILLED, real survivors=0".

The tell was in the author's own stdout and was edited out of the pasted evidence:
**all six cells reported the identical `by: below-ceiling launch is ALLOWED`**,
including cells that mutate only the at-ceiling branch and cannot change
below-ceiling behaviour. One kill reason repeated across cells with disjoint blast
radii is the signature -- see [[a-correct-observation-can-credit-the-wrong-mechanism]].

Sibling of [[a-mutant-that-cannot-build-scores-as-a-kill]] (mutant never parsed)
and [[pytest-exit-5-scores-as-a-kill]] (subject was empty). Here the mutant parsed
fine and the subject existed; **the environment moved out from under it**.

**How to apply:**
1. NULL mutant first. KILLED means stop -- report the matrix as non-discriminating,
   not the guards as sound.
2. Then REPAIR the harness and re-run, so you separate an evidence defect from a
   product defect. For a Python subject, `PYTHONPATH=<repo>/scripts/harness:<repo>/scripts/qa`
   in the subprocess env is enough; no repo write needed, so it stays inside the
   read-only rail. Under the repair, 5 of the 6 cells killed genuinely -- and G4
   **survived**, because `drive()` seeded only well-formed ledger rows so the
   corrupt-row branch it mutated was never exercised. A repaired re-run is how you
   find the survivor the filed matrix hid.
3. Suspect this whenever the subject does `sys.path` surgery off `__file__`,
   imports a sibling module, or reads a repo-relative path -- relocation silently
   changes all three. Check the interpreter both ways: the defect reproduced
   under `/usr/bin/python3` and `.venv/bin/python` alike.
