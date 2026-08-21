# live_check -- step 90.12

**Step:** 90.12 -- the mutation matrix's ERROR discriminator is vacuous over the production
fail-open handler. **Date:** 2026-08-21.

> **BUILT AND VERIFIED, NOT EVALUATED.** No Q/A spawned. See
> `handoff/current/experiment_results_90.12.md` for why.

## The immutable command, unpiped

```
$ bash -c 'python3 scripts/qa/mutation_matrix_90_1.py --verify && python3 scripts/qa/verify_error_discriminator_90_12.py --self-test'
EXIT 0
```

## Step 90.1's own immutable command, unaffected

```
$ python3 scripts/harness/attempt_gate.py --self-test && python3 scripts/qa/mutation_matrix_90_1.py --verify
EXIT 0
```

## The full checker, verbatim

```

==========================================================================
A. THE RED BASELINE IS THE REAL PRIOR CODE, NOT A RETYPING
==========================================================================
  [PASS] the pre-fix discriminator was extracted from git, not re-typed -- d564ad58, 1749 chars
  [PASS] ...and it is the TRACEBACK-ONLY implementation, which is what made it blind
  [PASS] the shipped discriminator is a DIFFERENT function, renamed to say what it does -- _drive_unresolvable

==========================================================================
B. THE DIFFERENTIAL -- one observation per mutant, scored BOTH ways
==========================================================================
  cell   BEFORE (traceback-only)    AFTER (typed)              expected
  QA1    not-ERROR                  ERROR                      ERROR
  QA1b   not-ERROR                  ERROR                      ERROR
  QA1c   not-ERROR                  ERROR                      ERROR
  QX2    ERROR                      ERROR                      ERROR
  DOM    not-ERROR                  not-ERROR                  KILL
  N0     not-ERROR                  not-ERROR                  KILL_NONE

==========================================================================
C. CRITERION 2 -- CALL-SITE renames score ERROR, and the DEFINITION control still does
==========================================================================
  [PASS] QA1 scores ERROR after the fix -- call-site rename: read_ledger -> read_ledger_v2 -- below drive [swallowed by the fail-open handler]: NameError: name 'read_ledger_v2' is not 
  [PASS] ...and scored NOT-ERROR before it, so the cell is RED-FIRST rather than already covered (QA1)
  [PASS] QA1b scores ERROR after the fix -- call-site rename: extract_step_id_claim -> ..._v2 -- below drive [swallowed by the fail-open handler]: NameError: name 'extract_step_id_claim_v
  [PASS] ...and scored NOT-ERROR before it, so the cell is RED-FIRST rather than already covered (QA1b)
  [PASS] QA1c scores ERROR after the fix -- call-site rename: extract_step_id -> ..._v2 -- below drive [swallowed by the fail-open handler]: NameError: name 'extract_step_id_v2' is 
  [PASS] ...and scored NOT-ERROR before it, so the cell is RED-FIRST rather than already covered (QA1c)
  [PASS] QX2 (definition rename) scored ERROR BEFORE and AFTER -- the pre-fix scan caught this sub-class, which is exactly why the fix looked complete

==========================================================================
D. CRITERION 3 -- IT STILL DISCRIMINATES (the over-eager failure mode)
==========================================================================
  [PASS] a DOMAIN exception through the SAME fail-open handler is NOT scored ERROR -- it stays a KILL -- AssertionError via '[attempt-gate] INTERNAL ERROR -- ...'
  [PASS] ...and the drive really did exercise that handler, so the check is not vacuous
  [PASS] the NULL mutant is NOT scored ERROR

==========================================================================
E. CRITERION 5 -- NO SILENT CELL LOSS IN THE SHIPPED MATRIX
==========================================================================
  shipped matrix: exit=0  KILLED 15 | SURVIVED 0 (excl. N0) | ERROR 0 | null mutant survived: True
  [PASS] the shipped matrix still exits 0
  [PASS] its tally is UNCHANGED by this fix -- 15 KILLED / 0 SURVIVED / 0 ERROR, null survived. Measured by me BEFORE the edit and again after; no shipped cell changed score, so no cell was silently deleted -- KILLED 15 | SURVIVED 0 (excl. N0) | ERROR 0 | null mutant survived: True
  [PASS] ...and the cell roster is non-empty, so the tally is not vacuous -- 17 cells scored

==========================================================================
F. CRITERION 6 + CONTAINMENT
==========================================================================
  [PASS] handoff/verdict_ledger.jsonl sha256 byte-identical before and after -- ee58607b406fb7fd -> ee58607b406fb7fd
  [PASS] the real attempt_gate.py is byte-identical -- every mutant ran from a temp copy
  [PASS] this run wrote nothing under scripts/ that was not already this step's own edit -- measured with git, not by grepping my own source for write verbs (that probe matches its own list) -- none
  [PASS] ...and the three subject modules are individually unmodified in the tree

==========================================================================
SUMMARY
==========================================================================
  checks run: 20 (floor 20)
  failed:     0

```
