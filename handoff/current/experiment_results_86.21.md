# Experiment results -- phase-86.21

**Step:** 86.21 (P2) -- the 3rd-CONDITIONAL counter is blind to any step in
flight, and fails open silently.
**Contract:** `handoff/current/contract_86.21.md` (written BEFORE any code).
**Research:** `handoff/current/research_brief_86.21.md` (gate PASSED, `wf_f916b683-d59`).

---

## 1. What was built

| File | Role |
|---|---|
| `handoff/verdict_ledger.jsonl` (NEW) | append-only, one row per Q/A VERDICT, written when Main transcribes it -- per CYCLE, not at close |
| `scripts/qa/verdict_history_86_21.py` (NEW) | the counter: reads the ledger, returns a STATUS as well as a count, and refuses to print a number it does not know |
| `scripts/qa/mutation_matrix_86_21.py` (NEW) | mutates the COUNTER and requires its self-test to go red |

**`harness_log.md` is untouched.** LOG-is-last is deliberate and the step text
forbids writing log rows mid-step; the ledger is a separate file precisely so
that ordering survives.

## 2. Criterion 1 -- REPRODUCED, and from git rather than asserted

The step text builds its reproduction on 36.17. **That no longer reproduces** --
36.17 has since closed and carries six rows. So the reproduction is built on
phase-86.20's real mid-flight state, replayable by anyone:

```
$ git show "688ac349:.claude/masterplan.json"            -> 86.20 status = pending
$ git show "688ac349:handoff/current/evaluator_critique_86.20.md" | grep -c '^## Cycle'
1
$ git show "688ac349:handoff/harness_log.md" | grep -c 'phase=86.20 result='
0

$ git show "7145f566:.claude/masterplan.json"            -> 86.20 status = pending
$ git show "7145f566:handoff/current/evaluator_critique_86.20.md" | grep -c '^## Cycle'
2
$ git show "7145f566:handoff/harness_log.md" | grep -c 'phase=86.20 result='
0
```

**Two recorded verdicts, status still `pending`, and the grep the rule prescribes
returns ZERO.** That is criterion 1 exactly. It is not a historical anecdote: both
Q/As at those commits stated in their own verdicts that the log carried no rows
for the step, and each was hand-fed its history by Main -- the party the rule
constrains.

> **A shell bug I hit and corrected rather than reported.** My first attempt at
> this produced zeros for BOTH counts. The cause was mine, not the data: in zsh,
> `git show $C:handoff/...` applies the `:h` history modifier to `$C`, yielding
> `.andoff/...` and a "unknown revision or path" error that `grep -c` faithfully
> reported as 0. Quoting as `"${C}:handoff/..."` fixes it. I nearly recorded a
> fabricated measurement produced by my own quoting.

## 3. Criterion 2 -- the source, and why not the obvious one

The ledger accumulates per CYCLE. `harness_log.md` was rejected as the source
because it is written at step CLOSE -- that IS the defect -- and writing rows
mid-step is explicitly forbidden.

`evaluator_critique_<id>.md` was also rejected as the primary source, and the
research is why: 17+ filename shapes exist, one-file-per-cycle is not an
invariant (36.17 ran six cycles and left one file), and -- measured directly --
a parser keyed on `## Cycle N verdict` returns 3 for 86.20 and 2 for 86.17 but
**0 for 36.17**, whose file uses heading depth 1. A silent zero over five real
verdicts is the defect recurring inside its own proposed fix.

## 4. Criterion 3 -- the correct count mid-flight, including the reset-on-FAIL path

Against 36.17's REAL six-verdict history, seeded into the ledger from the
recorded run ids:

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

The `CONDITIONAL -> FAIL -> FAIL -> CONDITIONAL -> CONDITIONAL -> PASS` sequence
exercises the reset twice: the FAIL pair resets the early run, and the closing
PASS resets the later one. Consecutive = 0, which is correct for a step that
ended on a PASS.

**The disagreement line is the interesting part**, and it is not the blindness
defect -- it is finding (1) from the research showing up live: the grep counts
CUMULATIVE conditionals (3), the counter counts CONSECUTIVE ones (0). My first
version of that message asserted the mid-flight cause in BOTH cases; it now
names the correct cause per case, because saying the wrong one is the same
claim-accuracy defect this project keeps hitting.

## 5. Criterion 4 -- independence, answered rather than preferred

**A Main-supplied count is ADVISORY, not authoritative.** The Q/A has no `Write`
tool and the Workflow runtime has no filesystem access, so Main or a hook is the
only possible writer -- the audited party writes the auditor's input, and no
amount of phrasing changes that.

What the design DOES buy is auditability rather than independence: the ledger is
append-only and git-committed, so a retro-edit appears as a diff in history
instead of vanishing into prose. That is a strictly weaker claim than
independence and is deliberately not dressed up as a stronger one. Genuine
independence needs a writer Main does not control -- a hook -- which is
recorded in §8 as not done.

## 6. Criterion 5 -- fail direction, asserted AND tested

Four distinct outcomes, because collapsing them is the defect:

| status | meaning | count | direction |
|---|---|---|---|
| `ok` | verdicts found | real | — |
| `no_rows_for_step` | ledger exists, step absent | 0 | fail-OPEN, and correct: it genuinely has no verdicts |
| `ledger_missing` | no ledger at all | 0 + a caution, non-zero exit | bootstrap state, NOT evidence of no history |
| `unparseable` | rows present, unreadable | **None** | fail-CLOSED: refuses to print a number, says treat the rule as ARMED |

The reasoning, not just the choice: an absent row for a step that has never been
graded is genuinely zero, so failing open there is correct. A corrupt ledger is
different in kind -- the count is unknown, and returning 0 for unknown is the
exact fail-open shape this step exists to remove (RFC 9413 §5.1).

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

## 7. Criterion 6 -- MUTATION, on the counter itself

The criterion asks for the SOURCE to be corrupted and the counter to notice
(self-test cases iii-v above). This closes the other half: it mutates the
COUNTER and requires the self-test to go RED. A self-test that cannot fail is
not a guard.

```
phase-86.21 criterion 6 -- mutation matrix (in-memory; repo never written)
target : verdict_history_86_21.py
md5    : 9ece5e79b6568feaaced32628fbfb144

[control] un-mutated self-test rc=0 (0 = PASSED)
  KILLED  | M1: unparseable/empty report 0 instead of None (the silent zero returns)
            self-test rc=1
  KILLED  | M2: reset becomes == 'PASS' (misses PASS_WITH_FINDINGS / PASS_AFTER_RETRY)
            self-test rc=1
  KILLED  | M3: corrupt rows are ignored instead of counted (fail-open restored)
            self-test rc=1
  KILLED  | M4: arming threshold drops to one CONDITIONAL (one-sided guard, Q/A's Q1)
            self-test rc=1
  KILLED  | M5: a present-but-EMPTY ledger reports a silent zero again (Q/A's finding 1)
            self-test rc=1
  KILLED  | M6: step matching becomes a PREFIX match (86.2 would swallow 86.20/86.21)
            self-test rc=1
  KILLED  | M7: verdict tokens stop being case-normalised (Q/A's Q3)
            self-test rc=1
  KILLED  | M8: a row with NO step_id is silently skipped again (fail-OPEN under-count)
            raised AttributeError: 'NoneType' object has no attribute 'strip'
  KILLED  | M9: prescribed_grep_count always returns 0 (Q/A's N1 -- contrast half unguarded)
            self-test rc=1
  KILLED  | M10: _report always exits 0 (Q/A's N2 -- the fail-CLOSED signal goes dark)
            self-test rc=1
  KILLED  | M11: would_auto_fail returns False instead of None when unknowable (Q/A's N4)
            self-test rc=1

[integrity] target md5 unchanged: True
ALL 11 MUTANTS KILLED -- every guard IN THIS MATRIX can fail.
```

M2 is the one worth reading twice: rewriting the reset as `== 'PASS'` looks
harmless and silently extends a "consecutive" run straight across a
`PASS_WITH_FINDINGS` -- and the corpus contains that token today.

## 8. Verification, and what I did NOT do

```
$ bash -c 'grep -c "^## Cycle" handoff/harness_log.md && ls handoff/current/evaluator_critique_*.md | head -3'
1189
handoff/current/evaluator_critique_36.12.md
handoff/current/evaluator_critique_36.13.md
handoff/current/evaluator_critique_36.17.md
```

exit **0**. `ruff --select F821,F401,F811` over the two new scripts: **All checks
passed** (it caught an unused `sys` import first, which is fixed).

**NOT done, stated plainly:**

- **Nothing writes the ledger automatically.** I seeded it with tonight's 11 real
  verdicts by hand. Until the transcription step or a hook appends to it, the
  ledger will silently stop tracking -- which is a fresh instance of the very
  class this step is about. It is the single most important follow-up.
- **The two contradictory rule statements are NOT reconciled in the docs.**
  `CLAUDE.md:358` and `qa.md:512` still disagree; the counter implements
  CLAUDE.md's predicate and SAYS so, but editing `qa.md` is a change to the Q/A's
  own instructions and I did not make it unilaterally at 02:00 with no operator
  awake. The counter reports the divergence rather than hiding it.
- **No Q/A has been asked to USE the counter.** Nothing in `qa.md` points at it,
  so today it informs a human reader, not the gate.
- **The ledger's history before tonight is absent.** Steps closed before
  2026-08-09 have no rows, so the counter reports `no_rows_for_step` for them --
  correctly, but that is not the same as knowing their real history.

---

## 9. Cycle 2 -- the Q/A found a silent zero I had left in, and a one-sided guard

**Cycle-1 verdict: CONDITIONAL (`wf_cb85c901-472`)**, verbatim in
`evaluator_critique_86.21.md`. Criteria 1-5 were MET and every one of them
re-derived independently -- including rebuilding the git reproduction,
reproducing the zsh `:h` trap under zsh 5.9 to confirm my disclosure, verifying
36.17's six-verdict history from `harness_log.md` (a source I did not supply),
and re-measuring every number in the artifact. All reproduced exactly.

It also **ruled for me on the six-vs-five question**, saying it would have
flagged the opposite choice -- trimming the closing PASS row to match the
criterion's wording would have been a fabricated measurement.

**Three findings, and the first is a genuine behavioural gap in a step whose
entire subject is "do not report a silent zero":**

1. **A present-but-ZERO-BYTE ledger reported `consecutive=0` at exit 0**, with
   the detail "it has genuinely not been graded yet" -- a confident and FALSE
   claim. Criterion 6 names "corrupt **or empty**" by word, and I had tested
   only corrupt. **This is the defect of the step, committed inside the fix for
   the defect of the step.**
2. **The arming threshold was one-sided.** `c >= 2` -> `c >= 1` SURVIVED: I
   asserted `would_auto_fail` is True at c=2 and never that it is False at c=1
   or c=0. The Q/A noted that its Q1 and Q2 mutate the SAME line in opposite
   directions and only one died -- which is the signature of a one-sided guard,
   and a better diagnostic than the finding itself.
3. **A blank/non-string `verdict` field was silently skipped** rather than
   counted malformed -- the field-level analogue of the line-level case I HAD
   guarded.

### What changed in cycle 2

- **New `ledger_empty` status.** A present-but-empty ledger now returns `None`,
  prints a truncation caution, and exits non-zero. It fails CLOSED, because a
  truncated ledger and an ungraded step are indistinguishable from the inside.
- **Threshold pinned from BELOW**: new self-test cases assert
  `would_auto_fail is False` at consecutive 1 and 0.
- **Blank verdict fields counted as malformed**, with a case that proves it.
- **Two more cases** the Q/A's other surviving mutants earned: exact step
  matching (86.2 must not swallow 86.20/86.21 -- these ids genuinely collide
  under a prefix rule) and case-normalisation of verdict tokens.
- **The matrix grew from 3 cells to 7**, and the four new ones are exactly the
  mutants the Q/A wrote that survived my cycle-1 matrix. They cannot survive
  again.

Adding the `ledger_empty` branch also broke M1's anchor. The harness reported
`anchor matched 0 time(s)` and refused to run the cell, rather than silently
mutating nothing -- the no-match-replace defence doing its job for the second
time tonight.

### The honest summary

My cycle-1 matrix mutated the three shapes I was already thinking about. The
Q/A's mutants were the ones I wasn't, and half of them lived. The step is about
a counter that must never report a silent zero, and I shipped one -- on the
exact input word the criterion names.

---

## 10. Cycle 3 -- my cycle-2 fix was the instance; this is the class

**Cycle-2 verdict: CONDITIONAL (`wf_8b188711-509`)**, verbatim in
`evaluator_critique_86.21.md`. All four of its cycle-1 survivors now died, and it
proved that **without using my harness** -- it built its own, on the explicit
reasoning that a kill seen only through the author's own construction can be an
artifact of that construction. Both harnesses agreed, and my 7-cell matrix
reproduced byte-for-byte including the md5.

**Then it found the fourth shape it had predicted, and it was mine.** Cycle 2
hardened the `verdict` FIELD. The `step_id` field, at the SAME call site, was
left unenumerated -- so a row with a missing, blank or null `step_id` was
indistinguishable from a row belonging to another step and took the silent-skip
path with `bad_lines=0` and `status=ok`.

Measured through the shipped entry point: a 3-row ledger with one such row
reported `consecutive=2, would_auto_fail=True` and printed *"a further
CONDITIONAL would be the 3rd"* — **when three consecutive CONDITIONALs had
already happened and the rule was already breached.** A silent UNDER-count, at
exit 0, in the fail-OPEN direction, on an escalation counter. That is the defect
of the step, for the second cycle running, one field to the left.

**Three more of its ten new mutants survived, and they clustered:**
`prescribed_grep_count`, `_report`, and `_report`'s exit-code map — the contrast
half and the entire CLI half — had **zero** automated coverage, because
`self_test()` never called them.

**And a contradiction on the shipped path:** `ledger_missing` printed *"treat the
rule as ARMED"* and `auto-FAIL armed : False` on the same screen. Prose
fail-closed, machine-readable properties fail-open — and the module's own
docstring says a caller that treats `None` as `0` has reintroduced the defect.

### What changed in cycle 3

1. **`step_id` is enumerated at the same call site as `verdict`** — absent, blank
   or null is MALFORMED (counted `bad`), while a row that legitimately names
   another step still skips. Fixed as a class this time: both fields at that
   call site, not just the one that was found.
2. **`LEDGER_MISSING` joins the not-knowable set**, so `consecutive` and
   `would_auto_fail` both return `None` and the prose stops contradicting the
   properties.
3. **The CLI half is now covered**: the self-test asserts `_report`'s exit codes
   (1/1/1/0/0), asserts `would_auto_fail is None` for every unknowable status,
   and drives `prescribed_grep_count` against a synthetic log.
4. **The `no_rows_for_step` detail stopped asserting a fact it cannot know.** It
   used to say "it has genuinely not been graded yet"; the Q/A ran the tool on
   **86.21 itself** and got that sentence while a cycle-1 verdict sat on disk.

Self-test 9 cases -> **15**. Matrix 7 cells -> **11**, and the four new ones are
exactly the Q/A's surviving mutants.

Two harness defects also fell out and were fixed: the M8 mutant CRASHES rather
than failing an assertion, and letting that propagate aborted the whole matrix
and reported nothing about the remaining cells — a crash is now scored as a kill.
And my cycle-3 edits broke the M1 and M6 anchors, which the harness reported as
`anchor matched 0 time(s)` rather than silently mutating nothing. That
no-match-replace defence has now earned its keep three times in one night.

### The honest summary

Two cycles running, the Q/A found the same defect class in my fix for that
class: cycle 2 fixed one field, cycle 3 fixed its sibling four lines away. The
lesson is not "harden fields" — it is that when a call site is found guilty once,
every position at that call site has to be enumerated, not just the guilty one.
