# Contract -- phase-86.89

**Step:** `86.89` (P2) · **Cycle:** 1 · **Written:** 2026-08-16, AFTER the gate returned.
**Title:** the derived mutation-coverage gate is blind to BEHAVIOURAL guards:
known-member recall against the three FAILs that motivated it is 1 of 4

---

## 1. Research gate -- PASSED (enforced)

| Field | Value |
|---|---|
| Rail | `.claude/workflows/research-gate.js` by **scriptPath** · run `wf_abfa4db8-f13` |
| Brief | `handoff/current/research_brief_86.89.md` (71,227 chars, `COMPLETE`) |
| Sources read in full | **22** (floor 5) · URLs **45** (floor 10) |
| Audit-class | YES -- `coverage.dry` after 2 dry rounds over **8** rounds |
| `gate_passed` | **true**, RECOMPUTED; self-report agreed |

---

## 2. Criterion 1 -- REPRODUCED, and the probe took three attempts

```
CONTROL: exit 0 | RESULT: OK -- every enumerated guard is touched by at least one cell.

PROBE POSITIVE CONTROL -- drop M4: exit 1 -> RED (probe LIVE)

  M6   (ordering              ): GREEN -> NOT demanded (INVISIBLE)
  M8   (fail-loud I/O         ): RED  -> DEMANDED
  M9   (step_id-in-dedup-key  ): GREEN -> NOT demanded (INVISIBLE)
  M11  (cycle-fallback (a)    ): GREEN -> NOT demanded (INVISIBLE)
  M12  (cycle-fallback (b)    ): GREEN -> NOT demanded (INVISIBLE)

KNOWN-MEMBER RECALL: 1 of 5   demanded=['M8']
restore byte-identical: True
```

**The figure REPRODUCES as filed.** The step says **1 of 4** because it counts
cycle-fallback as ONE member (M11+M12 target the same guard); by members that is
ordering / fail-loud-I/O / step_id-in-key / cycle-fallback, and only
fail-loud-I/O is demanded. Both framings are above and the denominator
convention is stated rather than left implicit.

**The probe's first two versions would have LIED.** Attempt 1 assumed the cells
were dicts (they are tuples) and printed **"RECALL: 0 of 5"** -- a number about my
regex. Attempt 2 used `\s*\n` where `\s*` already eats the newline. Attempt 3
used paren-depth counting, which the unbalanced `LedgerError(` inside the
mutation strings defeats. Only a line-based span worked. **The positive control
is what makes the final number mean anything**: dropping `M4` -- a guard the rule
genuinely sees -- turns the gate RED, so GREEN for the others is a fact about the
gate rather than a dead probe.

---

## 3. THE RESEARCH REFRAMES THE STEP, and the plan follows the reframing

The step's own `audit_basis` proposes "a declared-and-verified register" as the
likely mechanism. The gate returned something better, and it changes the design:

**1. This is a missing VACUITY check, not a coverage bug.** Kupferman
(CONCUR 2006): *"in vacuity, mutations are in the specifications, whereas in
coverage, mutations are in the system."* Both shipped artefacts --
`mutation_matrix_86_85.py` and `verify_matrix_coverage_86_85.py` -- mutate **the
system**. **Nothing mutates the CELLS.** ~20% of specifications pass vacuously,
and *"vacuous passes always point to a real problem"*. The one-off drop-a-cell
probe that caught the `ast.Try` over-crediting **should be standing, not a
comment** -- and it is exactly the probe I had to hand-rebuild three times above.

**2. Enriching the AST rule is the WRONG fix.** The misses are StaAgent **Type-2
(inadequate)** -- not fixable by broadening the pattern. Corroborated: SpecFuzzer
filtering raised precision 67.83% -> 74.17% while recall stayed **54.57%,
unchanged**. So widening `raise LedgerError` / `ast.If` to cover ordering and
dedup-key composition would buy precision and no recall.

**3. Over-crediting is detected by DISCRIMINATION, not by recall** -- paired
safe/unsafe probes (NIST). Recall says "did we find the known ones"; only a
paired probe says "does a cell that should NOT cover this guard actually fail to
cover it".

**4. "1 of 4" is `Recall_SD`, not population recall.** ISSTA'24 Thm 3.5: a
sample-derived recall is not convertible to population recall without
`Recall_DG`. `qa.md` 4b already demands an author-independent known set, and the
**three FAIL ledger rows supply exactly that** -- they were chosen by history,
not by me.

**5. Adversarial, and it moderates criterion 3.** Google went 15% -> 89%
productive using **hand-declared, unsound** rules. So *"never declare"* is too
strong; the defensible rule is **"never let a declaration be the DENOMINATOR"** --
a register may add obligations, but recall must still be measured against the
author-independent set.

**6. CORRECTION the gate confirms:** 5 of 14 cells cover ZERO guards and no guard
is covered by more than one cell -- matching the step's `audit_basis`, and
confirming those cells are **invisible**, not redundant.

---

## 4. Immutable success criteria (VERBATIM)

1. the 1-of-4 recall is REPRODUCED first, with the command and its verbatim output, before anything is designed -- and if the figure does not reproduce, say so and re-scope rather than building for a number that is wrong
2. a mechanism is proposed for BEHAVIOURAL guards and its recall is MEASURED against the same known-member set (the three 86.85 FAILs plus the 86.86 caller-side mutant), stating the recall figure rather than asserting improvement
3. if the chosen mechanism is a declared register rather than a derivation, the declaration must itself be VERIFIED against behaviour -- a hand-declared list that nothing checks is the very failure this whole series is about, and shipping one silently is a scope breach
4. the gate must be shown to go RED per member: dropping the cell for each known member individually must turn it red, demonstrated by execution
5. over-crediting is tested for explicitly, because it is the dangerous direction and it already happened once here (including ast.Try as an ancestor let one anchor cover every guard in main's body, and the tell was that dropping a cell left the gate GREEN)
6. no claim of completeness is made that the measured recall does not support; the licence sentence must state what the mechanism does NOT cover
7. verdict semantics are UNCHANGED: nothing here may turn a non-PASS into a PASS
8. mutation-test every new guard with the control observed GREEN first and a byte-identical restore

**Immutable command** (green now):
`bash -c 'source .venv/bin/activate && python scripts/qa/verify_matrix_coverage_86_85.py'`

---

## 5. Plan

**P1 -- make the drop-a-cell probe STANDING (the vacuity check).** Ship
`scripts/qa/verify_cell_vacuity_86_89.py`: for **every** cell in the matrix, drop
it and require the gate to change. A cell whose removal leaves the gate GREEN is
either covering nothing or covering something already covered -- both are
reportable, and today **5 of 14 cells** are in that state. This is the mechanism
the research names, it is a DERIVATION rather than a declaration, and it is the
probe I rebuilt by hand three times.

**P2 -- measure its recall against the AUTHOR-INDEPENDENT known set**
(criterion 2): the three 86.85 FAILs plus the 86.86 caller-side mutant. State the
figure; do not assert improvement.

**P3 -- criterion 5, over-crediting, by DISCRIMINATION not recall.** Paired
probes: for each cell, a guard it SHOULD cover and a guard it should NOT. The
`ast.Try`-ancestor failure is the worked example -- one anchor appearing to cover
every guard in a function body.

**P4 -- criterion 4 per member**, by execution, with the positive control shown
live so a GREEN is never mistaken for a dead probe.

**P5 -- criterion 6, the licence sentence.** It must state what this does NOT
cover: a standing vacuity check finds cells that cover nothing; it does **not**
discover guards nobody wrote a cell for. That is a different (and unsolved)
problem, and the `Recall_SD` vs `Recall_DG` distinction must appear so the figure
is not read as population recall.

**P6 --** criterion 8 on every new guard, control GREEN first, byte-identical
restores; then handoff, Q/A, log.

---

## 6. Out of scope (named)

- Widening the AST enumeration rule. The research says the misses are Type-2 and
  broadening buys precision, not recall. Recorded as a REJECTED alternative with
  its evidence, not silently skipped.
- A declared behavioural register **as the denominator**. Permitted only as an
  additive obligation list, per finding 5.
- The escalated `86.90`/`86.91` and the in-flight `86.88` are untouched.

## 7. References

- `handoff/current/research_brief_86.89.md` (run `wf_abfa4db8-f13`)
- Kupferman, *vacuity vs coverage* (CONCUR 2006; CHARME 2003)
- ISSTA'24 TotalRecall -- `Recall_SD` vs `Recall_DG`, Thm 3.5
- NIST *Evaluating Bug Finders* -- paired safe/unsafe discrimination
- SpecFuzzer / Daikon -- filtering raises precision, not recall
- Inozemtseva & Holmes (ICSE'14) -- coverage is not correlated with effectiveness
- Google's 15% -> 89% productive-rule result -- declarations may add, never denominate
