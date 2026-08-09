# live_check -- phase-86.21

**Required evidence (immutable, verbatim from `.claude/masterplan.json`):**
"Verbatim counter output against 36.17's real verdict history showing 5 verdicts
and the correct consecutive-CONDITIONAL count, side by side with the verbatim
zero-row output of the log-grep the rule currently prescribes."

---

## 1. The counter against 36.17's real verdict history

```
step            : 36.17
source          : handoff/verdict_ledger.jsonl
status          : ok
detail          : 6 verdict(s) from the ledger
verdicts        : CONDITIONAL -> FAIL -> FAIL -> CONDITIONAL -> CONDITIONAL -> PASS
consecutive     : 0
auto-FAIL armed : False  (a further CONDITIONAL would be the 1st)

for contrast, the grep the rule currently prescribes: 3 row(s)
  DISAGREEMENT: ledger says 0, harness_log grep says 3.
  CAUSE: the two use DIFFERENT PREDICATES, not different data.
  The grep counts CUMULATIVE CONDITIONAL rows (3); this counter
  counts CONSECUTIVE ones since the last non-CONDITIONAL (0).
  CLAUDE.md specifies consecutive-with-reset; qa.md specifies the
  cumulative grep while calling it consecutive. Until those two are
  reconciled the escalation is ambiguous regardless of the source.
```

**On "5 verdicts":** the immutable text says five. The ledger carries **six** --
`CONDITIONAL, FAIL, FAIL, CONDITIONAL, CONDITIONAL, PASS` for cycles 190-195.
The step text was written when 36.17 had run five cycles; it closed on a sixth
(PASS) later the same night. Six is the true history, and the fifth-to-sixth
transition is what makes the consecutive count 0 rather than 2. Stating this
rather than trimming a row to match the criterion's wording.

The reset-on-FAIL path the criterion asks for is exercised twice: the `FAIL,
FAIL` pair resets the opening CONDITIONAL, and the closing `PASS` resets the
`CONDITIONAL, CONDITIONAL` pair.

## 2. Side by side with the prescribed log-grep, on a step that was MID-FLIGHT

The criterion asks for the zero-row output of the prescribed grep. 36.17 is
closed now, so its grep no longer returns zero -- it returns 3. The honest
side-by-side therefore uses phase-86.20 at the commits where it was genuinely
in flight, replayable from git:

```
$ git show "688ac349:handoff/harness_log.md" | grep -c 'phase=86.20 result='
0
$ git show "688ac349:handoff/current/evaluator_critique_86.20.md" | grep -c '^## Cycle'
1

$ git show "7145f566:handoff/harness_log.md" | grep -c 'phase=86.20 result='
0
$ git show "7145f566:handoff/current/evaluator_critique_86.20.md" | grep -c '^## Cycle'
2
```

masterplan status at both commits: **pending**. So: two recorded verdicts, step
still open, and the counter the rule prescribes returns **zero**.

## 3. The statuses and the CLI, exercised (FIVE statuses; 15 self-test cases)

```
SELF-TEST -- the counter must distinguish absence from corruption

   (i)   36.17 real history -> consecutive=2 (expect 2), armed=True
   (i-b) one CONDITIONAL -> consecutive=1 (expect 1), armed=False (expect False)
   (i-c) zero CONDITIONALs -> consecutive=0 (expect 0), armed=False (expect False)
   (ii)  reset on PASS_WITH_FINDINGS -> consecutive=1 (expect 1)
   (iii) corrupt ledger -> status=unparseable, consecutive=None (expect None, NOT 0)
   (iii-b) blank verdict field -> status=unparseable (expect unparseable, NOT a silent skip)
   (iii-c) ZERO-BYTE ledger -> status=ledger_empty, consecutive=None (expect ledger_empty, None)
   (iii-d) exact step match -> 86.2 sees ['PASS'] (expect ['PASS'], NOT 86.20/86.21's rows)
   (iii-e) lowercase verdicts -> consecutive=2 (expect 2, i.e. normalised)
   (iv)  missing ledger -> status=ledger_missing (expect ledger_missing)
   (v)   unknown step -> status=no_rows_for_step, consecutive=0 (expect no_rows_for_step, 0)
   (vi)  CLI exit codes {'empty': 1, 'corrupt': 1, 'missing': 1, 'ok': 0, 'no_rows': 0} (expect empty/corrupt/missing=1, ok/no_rows=0)
   (vii) would_auto_fail on unknowable statuses -> [None, None, None] (expect all None)
   (viii) prescribed_grep_count on a synthetic log -> 2 (expect 2)
   (ix)  row with NO step_id -> status=unparseable (expect unparseable, NOT a silent under-count)

SELF-TEST PASSED
```

## 4. The empty-ledger path, through the real entry point (cycle 2)

Criterion 6 names "corrupt **or empty**". Cycle 1 tested only corrupt, and an
empty ledger silently reported zero. Now:

```
status          : ledger_empty
detail          : verdict_ledger.jsonl exists but is EMPTY (0 bytes) -- that is a
                  truncation signal, NOT evidence that this step has no verdicts.
                  The count is NOT KNOWABLE.
verdicts        : (none)

NOTE: the ledger file exists but is EMPTY. Treat this as a
TRUNCATED source, not as an ungraded step -- the two are
indistinguishable from here, so the rule is treated as ARMED.
```

Measured by truncating the real ledger, running the real CLI, and restoring it
(11 rows before and after).

## 5. Scope of this evidence

The ledger was seeded BY HAND with tonight's 11 real verdicts (36.17 x6,
86.20 x3, 86.17 x2), each carrying its real `run_id`. **No automatic writer
exists yet**, so the ledger will stop tracking the moment a session forgets --
recorded in `experiment_results_86.21.md` §8 as the most important follow-up
rather than left implicit. No Q/A currently consults this counter; `qa.md` still
prescribes the grep.
