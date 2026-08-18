---
name: matrix-oracle-inherits-selftest-blindspots
description: A mutation matrix that scores cells by "the module's own --self-test goes RED" can never catch a mutant the self-test is blind to; and a filter fixture whose two ids are unrelated under the relation being enforced cannot test broadening.
metadata:
  type: feedback
---

When a mutation matrix's ORACLE is the module's own `--self-test`, the matrix
inherits **every** blind spot of that self-test. "22 cells, 22 KILLED, 0 SURVIVED"
then licenses only "22 mutations the self-test can see were killed" -- it says
nothing about the mutants it cannot see, and adding cells cannot fix it because
every new cell is scored by the same blind oracle. Look for a second, independent
oracle (pytest, a behavioural differential) before crediting a full-kill matrix.

Companion shape, same incident: **a filter/selector fixture must instantiate the
relation the guard enforces.** A test named "sequence filters by step" whose fixture
uses ids `4.1` and `4.2` cannot fail when exact equality is broadened to
`startswith` / `in` -- because `4.2` is not a prefix of `4.1`. The fixture has to
carry a PREFIX-RELATED pair (`4.1` + `4.10`, `86.9` + `86.90`) or the assertion is
inert on the only regression direction that matters.

**Why:** measured on step 86.85 cycle 11 (2026-08-17). `verdict_ledger_write.py:332`
filters ledger rows by exact step_id. Its sole coverage was the self-test check
"sequence filters by step" (fixture `99.4`/`99.2`) and `test_sequence_filters_by_step`
(fixture `4.1`/`4.2`). TWO independently-constructed mutants -- `.startswith(step_id)`
and `step_id not in ...` -- SURVIVED all 32 self-test checks, all 31 pytest tests, and
were structurally invisible to the 22-cell matrix. The differential was material and
reachable: `emit_sequence("86.9")` went `[C,C,C]` -> `[C,C,C,PASS]` by sweeping in
step 86.90's row, and through the real `enforceEscalation` that is
`n=3 would_auto_fail=true` -> `n=0 would_auto_fail=false`: a FOREIGN step's PASS
clearing a real escalation. Walking `.claude/masterplan.json` gives **869 strict-prefix
pairs among 1413 step ids**, and `attempt_gate.py` calls `emit_sequence` on the
production ledger at every Workflow launch -- so the shape is pervasive, not exotic.

The same file had already been FAILed at cycle 1 for a **palindromic** ordering
fixture and had since added explicit anti-vacuity assertions for the ORDER axis
("order fixture is NOT palindromic") and the DATE axis ("distinct event dates").
Nobody added one for the FILTER axis. Anti-vacuity assertions get added per-axis, by
incident -- so when you find one, enumerate the OTHER axes the same file selects on.

**How to apply:** on any step shipping a mutation matrix, (1) ask what the cells'
oracle actually is and find a mutant outside its reach; (2) for every selector,
comparator or key in the code under test, name the relation it enforces (equality,
prefix, membership, ordering) and check the fixture instantiates a pair that BREAKS
under a broadened relation; (3) run two differently-constructed mutants so survival
is not a construction artifact; (4) drive the survivor's differential through the
real downstream consumer before calling it material -- and through it before calling
it equivalent. Links: [[palindromic-fixture-cannot-test-order]],
[[mutate-each-half-of-an-ANDed-guard]], [[two-mutant-forms-separate-artifact-from-kill]].
