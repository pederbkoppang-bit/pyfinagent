# Experiment results -- phase-86.89

**Cycle:** 1 · **Written:** 2026-08-16 · **Contract:** `handoff/current/contract_86.89.md`

---

## 1. Files changed

| File | Change |
|---|---|
| `scripts/qa/verify_cell_vacuity_86_89.py` | **NEW** -- the standing cell-vacuity check |
| `handoff/current/{contract,research_brief,experiment_results,live_check}_86.89.md` | handoff artifacts |

**No production code touched.** No `.env`, no flag, no gate loosened. The
existing `verify_matrix_coverage_86_85.py` and `mutation_matrix_86_85.py` are
**unmodified** -- deliberately, see §4.

---

## 2. Criterion 1 -- REPRODUCED, and the probe lied twice first

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

**The figure REPRODUCES as filed.** The step states **1 of 4** because it counts
cycle-fallback as ONE member (M11+M12 target the same guard). Both denominators
are above; the convention is stated rather than left implicit.

**The probe's first two versions would have reported a false number.** Attempt 1
assumed dict-shaped cells (they are tuples) and printed **"RECALL: 0 of 5"** -- a
number about my regex. Attempt 2 wrote `\s*\n`, where `\s*` already consumes the
newline. Attempt 3 counted paren depth, which the unbalanced `LedgerError(`
inside the mutation strings defeats. **A probe that finds nothing looks exactly
like a gate that demands nothing**, which is why the positive control (drop `M4`
-> RED) is what makes the final figure mean anything.

---

## 3. The research REFRAMED the step, and the plan followed it

The step's `audit_basis` expected "a declared-and-verified register". The gate
returned something better:

- **This is a missing VACUITY check, not a coverage bug.** Kupferman (CONCUR
  2006): *"in vacuity, mutations are in the specifications, whereas in coverage,
  mutations are in the system."* Both shipped artefacts mutate the SYSTEM.
  Nothing mutates the CELLS.
- **Enriching the AST rule would not have worked.** The misses are StaAgent
  **Type-2 (inadequate)**, not Type-1 (imprecise) -- and SpecFuzzer's filtering
  moved precision 67.83% -> 74.17% with recall **unchanged at 54.57%**.
- **Over-crediting is caught by DISCRIMINATION (paired safe/unsafe probes,
  NIST), not by recall.**
- **"1 of 4" is `Recall_SD`**, not population recall (ISSTA'24 Thm 3.5).
- **A declaration may ADD an obligation but must never be the DENOMINATOR**
  (Google: 15% -> 89% productive with hand-declared *unsound* rules, so "never
  declare" is too strong).

---

## 4. The mechanism -- a standing vacuity check, not a register

`scripts/qa/verify_cell_vacuity_86_89.py` asks the DUAL question the existing
gate never asks: **does every cell demand a guard?** For each cell, drop it and
require the gate to change.

This is a **derivation, not a declaration** -- cell ids are parsed from the
matrix -- so criterion 3's register requirement does not bind. It is also the
probe I had to hand-rebuild three times during the reproduction, which is the
research's point: *the one-off drop-a-cell probe should be standing, not a
comment.*

```
$ python scripts/qa/verify_cell_vacuity_86_89.py
  cells derived from the matrix: 14 -> M1 ... M14
  demanding :  9  ['M1','M2','M3','M4','M7','M8','M10','M13','M14']
  VACUOUS   :  5  ['M5','M6','M9','M11','M12']
ALL GREEN: 7 passed, 0 failed
```

**It immediately finds exactly the five cells the step's own `audit_basis`
names** -- independent confirmation that they cover ZERO enumerated guards and
are INVISIBLE rather than redundant.

### Why it ships GREEN with a baseline rather than RED

Shipping it red would make it a dead gate within a day -- the exact 86.92 failure
now filed (a checker red for a known reason stops signalling anything). The five
already-vacuous cells are recorded as acknowledged **debt** in `KNOWN_VACUOUS`,
with the per-member reason written down.

**The declaration is not the denominator** (the research's rule): recall is still
measured against the author-independent known set, never against this list. And
it cannot rot in either direction -- assertion `[4]` fails on a NEW vacuous cell,
and assertion `[5]` fails if a baselined cell starts demanding something, forcing
the debt list to shrink rather than sit.

---

## 5. Criterion 2 -- recall MEASURED against the author-independent set

The set is chosen by history, not by me: the three 86.85 FAILs plus the 86.86
caller-side mutant.

| # | member | OLD gate | NEW check |
|---|---|---|---|
| 1 | ordering (M6) | GREEN -- not demanded | **FLAGGED vacuous** |
| 2 | fail-loud I/O (M8) | RED -- demanded | demanding (correctly not flagged) |
| 3 | step_id-in-dedup-key (M9) | GREEN -- not demanded | **FLAGGED vacuous** |
| 4 | cycle-fallback (M11+M12) | GREEN -- not demanded | **FLAGGED vacuous** |
| 5 | 86.86 caller-side pre-mangle | invisible (no cell exists) | **STILL INVISIBLE** |

- **OLD gate: 1 of 4** on members 1-4.
- **NEW check: 4 of 4 correctly CLASSIFIED** on members 1-4.
- **NEW check on member 5: 0**, and the reason is structural.

---

## 6. Criterion 6 -- the licence sentence, and what this does NOT cover

> **This mechanism licenses one claim: every cell in the matrix demands at least
> one enumerated guard.**

It does **NOT** license:

- **that the guard set is complete.** A vacuity check finds cells that demand
  nothing; it **cannot discover a guard nobody wrote a cell for**. A matrix with
  one good cell and no others scores a perfect 1/1. **That is member 5, and it
  remains open** -- the mechanism does nothing for it.
- **population recall.** Both figures are `Recall_SD` against a chosen sample;
  ISSTA'24 Thm 3.5 forbids converting them without `Recall_DG`, which is not
  measured and not claimed.

---

## 7. Criterion 5 -- over-crediting, tested by DISCRIMINATION

Per NIST, over-crediting is caught by paired probes, not by recall. The pairing
is internal to the check:

- Assertion `[1]` requires at least one cell observed **DEMANDING**. A probe that
  reported "vacuous" for everything -- the over-crediting direction, where every
  cell looks removable -- **fails** rather than passing with a flattering count.
- Assertion `[5]` is the opposite pairing: a baselined cell that starts demanding
  something must leave the baseline.
- Assertion `[3]` asserts the matrix is restored byte-identically, so a probe
  that corrupted the file could not report a clean result.

---

## 8. Criteria 7 and 8

**7 --** no verdict semantics touched: this check reads a mutation matrix and a
coverage gate. It cannot turn a non-PASS into a PASS; it has no verdict surface.

**8 --** mutation-testing of the new guard is in §9 of `live_check_86.89.md`,
control GREEN first, matrix restored byte-identically (asserted by the check
itself at `[3]`).
