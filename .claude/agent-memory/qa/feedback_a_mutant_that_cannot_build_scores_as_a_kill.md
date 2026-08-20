---
name: a-mutant-that-cannot-build-scores-as-a-kill
description: Mutation harnesses that wrap expect() in try/catch score a mutant that fails to PARSE as KILLED -- instrument every cell with an unmutated control AND a threw/returned distinction
metadata:
  type: feedback
---

A mutation cell proves nothing unless the mutant BUILDS, RUNS, and the guard then
goes red. Harnesses written as
`try { survived = !(await m.expect(mutated)) } catch (_e) { survived = false }`
score a mutant that throws -- including one that never parsed -- as **KILLED**.

**Why:** measured on `scripts/qa/verify_prompt_render_86_90.mjs` cell M3. Its
replacement text ended `void ('` followed by a newline, i.e. an unterminated
single-quoted string, so importing the mutant raised `SyntaxError: Invalid or
unexpected token`; the catch converted the crash into a kill. Doubly inert: even
had it parsed, the injected `return '(unrenderable)'` sat AFTER the `throw` it
was meant to replace -- dead code. The matrix advertised "5 cells, all KILLED";
4 were genuine. Same family as [[pytest-exit-5-scores-as-a-kill]], different
mechanism: there the SUBJECT was empty, here the MUTANT was broken.

**Second instance -- PARSES IS NOT RUNS (phase-90.1 cycle 2, 2026-08-20, Python).** The
cycle-1 Q/A caught this class in `scripts/qa/mutation_matrix_90_1.py` and the fix added
`ast.parse(mutated)` before scoring: unparseable now scores ERROR. That closes only the
SyntaxError subset. I anchored three mutants on a unique module-level line and made each
parse cleanly but fail at import -- `raise RuntimeError(...)` at module scope, a NameError
(`X = __undefined__`), and an `import __no_such_module__`. **All three scored KILLED**; a
SyntaxError control scored ERROR. Mechanism: the drives run the mutant through
`subprocess.run`, which does NOT raise on a non-zero exit, so every check fails and the
cell is credited. `ast.parse` is a compile-time gate on a runtime failure.
**Fix, and the check that bounds the damage:** smoke-import the mutant
(`subprocess.run([sys.executable, '-c', f'import {mod}'])`) and score a non-zero import as
ERROR. Then apply EVERY shipped cell to a temp copy and import each one -- in 90.1 all 15
imported, so no reported kill was false and the finding stayed WARN instead of BLOCK.
That measurement is what separates "latent gap in the gate" from "corrupted result".

**How to apply:** re-run each cell yourself and record three states, never two --
`expect()` RETURNED false / RETURNED true / THREW. Run the cell's own `expect()`
against the UNMUTATED source first: it must return false, or the cell was never
discriminating. A THREW is an ARTIFACT-KILL: report it and name the fix. Then
build a second, differently-constructed mutant of the same guard
([[two-mutant-forms-separate-artifact-from-kill]]) to decide whether the GUARD is
vacuous or only the CELL is -- in 86.90 a valid-syntax reachable substitution did
turn the section red, so the guard was sound and only the cell was WARN-level.
Anchor-uniqueness checks do NOT cover this: the anchor was unique and the
replacement still could not run.
