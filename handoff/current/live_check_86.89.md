# live_check -- phase-86.89

Evidence artifact for the `verification.live_check` gate. Verbatim output only.

## 1. Criterion 1 -- the 1-of-4 recall REPRODUCED, with a proven-live probe

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

Reproduces as filed. The step says 1 of 4 because it counts cycle-fallback
(M11+M12) as one member. Both denominators are shown.

**Two earlier versions of this probe reported a FALSE number** -- a dict-shaped
regex against tuple cells printed "RECALL: 0 of 5", and a `\s*\n` pattern matched
nothing because `\s*` already eats the newline. A probe that finds nothing looks
identical to a gate that demands nothing; the `M4` positive control is what
separates them.

## 2. The new mechanism, running

```
$ python scripts/qa/verify_cell_vacuity_86_89.py
phase-86.89 -- cell vacuity (mutating the SPECIFICATION, not the system)

  cells derived from the matrix: 14 -> M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14

  ok   [0] the matrix yields a non-empty cell list
  ok   [0] CONTROL: the gate is GREEN before any cell is dropped
  demanding :  9  ['M1', 'M2', 'M3', 'M4', 'M7', 'M8', 'M10', 'M13', 'M14']
  VACUOUS   :  5  ['M5', 'M6', 'M9', 'M11', 'M12']

  ok   [1] the probe DISCRIMINATES (at least one demanding cell observed)
  ok   [2] every cell is scorable
  ok   [3] the matrix is restored byte-identically
  ok   [4] no NEW vacuous cell
  ok   [5] the baseline has not rotted
  ok   [floor] 7 assertions ran (floor 7)

ALL GREEN: 7 passed, 0 failed
```

It finds exactly the five cells the step's own `audit_basis` names.

## 3. Criterion 2 -- recall against the AUTHOR-INDEPENDENT known set

| # | member | OLD gate | NEW check |
|---|---|---|---|
| 1 | ordering (M6) | not demanded | **FLAGGED vacuous** |
| 2 | fail-loud I/O (M8) | demanded | demanding (correctly not flagged) |
| 3 | step_id-in-dedup-key (M9) | not demanded | **FLAGGED vacuous** |
| 4 | cycle-fallback (M11+M12) | not demanded | **FLAGGED vacuous** |
| 5 | 86.86 caller-side pre-mangle | invisible (no cell exists) | **STILL INVISIBLE** |

OLD **1 of 4** -> NEW **4 of 4 correctly classified** on members 1-4, and **0**
on member 5. Both are `Recall_SD`, never population recall (ISSTA'24 Thm 3.5).

## 4. Criterion 8 -- the new guard is itself mutation-tested

```
CONTROL: exit 0 | ALL GREEN: 7 passed, 0 failed -> GREEN

  KILLED  V1 baseline swallows everything (KNOWN_VACUOUS = every cell)   exit 1
  KILLED  V2 the rot guard [5] removed                                   exit 1
  KILLED  V3 the discrimination guard [1] removed                        exit 1
  KILLED  V4 cell parser matches nothing (the "0 of 5" failure)          exit 1
  KILLED  V5 restore guard [3] removed and the matrix left mutated       exit 1

5/5 killed | checker restored byte-identical: True | matrix clean: True
```

**V2 and V3 SURVIVED on the first run**, and that is why the cardinality floor
exists: deleting an assertion from a checker cannot fail that checker, because a
shrinking checker reports the same `ALL GREEN` as a passing one. The floor
asserts the assertion COUNT, and both cells then killed.

**V1 is the over-crediting direction** -- a baseline that swallows everything --
and it fails on `[5]`, the rot guard, because baselined cells that demand
something must leave the list.

## 5. The licence, stated

> Licenses ONE claim: every cell in this matrix demands at least one enumerated guard.

Does NOT license that the guard set is complete -- a vacuity check **cannot
discover a guard nobody wrote a cell for**, which is member 5 and remains open.
Does NOT license population recall.
