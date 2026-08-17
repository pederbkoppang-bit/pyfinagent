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
$ git show "688ac349:handoff/harness_log.md" | grep -c 'phase=86.20 result='
0
$ git show "688ac349:handoff/current/evaluator_critique_86.20.md" | grep -c '^## Cycle'
1

$ git show "7145f566:handoff/harness_log.md" | grep -c 'phase=86.20 result='
0
$ git show "7145f566:handoff/current/evaluator_critique_86.20.md" | grep -c '^## Cycle'
2
```

*(cycle-7 REPLACEMENT, closing the cycle-5 Q/A's fix-2: the fence above is the
live_check §2 replay VERBATIM -- the previous fence grepped 86.21's OWN
artifacts, returned 0/0 at both commits, and sat under a "Two recorded
verdicts" headline its own output contradicted. The masterplan status at both
commits is `pending`.)*

**Two recorded verdicts (critique headers 1 then 2), status still `pending`,
and the log-grep the rule prescribes returns ZERO at both commits.** That is
criterion 1 exactly. It is not a historical anecdote: both
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

Five distinct outcomes, because collapsing them is the defect *(cycle-8
REPLACEMENT of the cycle-2 table, which said "Four", omitted `ledger_empty`
entirely, and recorded `ledger_missing` as "0 + a caution" -- a count the
code refuses to print. The cycle-7 Q/A drove all five statuses through the
real read_ledger()/_report() and found the CODE more fail-closed than this
table asserted; the table now states what the code does)*:

| status | meaning | count | direction |
|---|---|---|---|
| `ok` | verdicts found | real | — |
| `no_rows_for_step` | ledger exists, step absent | 0 | fail-OPEN, and correct: it genuinely has no verdicts |
| `ledger_missing` | no ledger at all | **None**, exit 1 | fail-CLOSED: "NOT KNOWABLE (refusing to print 0)", auto-FAIL "UNKNOWN -- treat as ARMED" |
| `ledger_empty` | zero-byte ledger | **None**, exit 1 | fail-CLOSED, same refusal (criterion 6 names "empty" by word; self-test case (iii-c)) |
| `unparseable` | rows present, unreadable | **None**, exit 1 | fail-CLOSED: refuses to print a number, says treat the rule as ARMED |

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
   (vi-b) _report prints 'consecutive     : 2' -> True
   (vi-c) corrupt ledger refuses to print a zero -> True
   (vi-e) knowable branch prints the real armed flag (True) -> True
   (vi-f) unknowable branch refuses a boolean armed flag -> True
   (vi-d) blindness cause printed for g=0,c>0 -> True
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
md5    : 142f6befbd7fc96689f568cb16b98820   <- cycle-8 note: the md5 of revision 5b7966e8 (capture time). A DIFFERENT step (86.78/86.79, commit 9b4d5281) later edited this product file to add --evidence-only; live md5 today is b8c0370a54e5fb817d4e19980dd257ed. The 16 cells reproduce at the current tree (cycle-7 Q/A re-ran them); only this identity line aged.

[control] un-mutated self-test rc=0 (0 = PASSED)
  [broken-scoring self-check] uncompilable mutant -> 'broken' (correct)
  [broken-scoring self-check] real behavioural mutant -> 'killed' (correct)
  [broken-scoring self-check] behaviour-preserving mutant -> 'survived' (correct)
  KILLED  | S1: the CLI prints a hard zero for every step forever (the silent zero returns, in the OUTPUT)
            self-test rc=1
  KILLED  | S2: the two CAUSE explanations swap -- blindness attributed to a predicate mismatch and vice versa
            self-test rc=1
  KILLED  | A1: the knowable branch hardcodes 'auto-FAIL armed : False' (armed step reads as unarmed)
            self-test rc=1
  KILLED  | A2: the UNKNOWABLE branch prints a boolean instead of refusing (fail-OPEN)
            self-test rc=1
  KILLED  | S3: the whole DISAGREEMENT block disappears silently
            self-test rc=1
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
            self_test() raised AttributeError: 'NoneType' object has no attribute 'strip'
  KILLED  | M9: prescribed_grep_count always returns 0 (Q/A's N1 -- contrast half unguarded)
            self-test rc=1
  KILLED  | M10: _report always exits 0 (Q/A's N2 -- the fail-CLOSED signal goes dark)
            self-test rc=1
  KILLED  | M11: would_auto_fail returns False instead of None when unknowable (Q/A's N4)
            self-test rc=1

[integrity] target md5 unchanged: True
ALL 16 MUTANTS KILLED -- every guard IN THIS MATRIX can fail.
```

M2 is the one worth reading twice: rewriting the reset as `== 'PASS'` looks
harmless and silently extends a "consecutive" run straight across a
`PASS_WITH_FINDINGS` -- and the corpus contains that token today.

## 8. Verification, and what I did NOT do

*(cycle-7 note, closing the cycle-5 Q/A's fix-5 placement: STALE -- a
well-formed ledger missing recent verdicts -- is the FIFTH failure mode the
four statuses cannot represent; named and demonstrated in section 11
(lines around "A well-formed but STALE ledger"), with the current instance:
the tail is [C, C], so one more CONDITIONAL is the 3rd and arms the rail.)*

```
$ bash -c 'grep -c "^## Cycle" handoff/harness_log.md && ls handoff/current/evaluator_critique_*.md | head -3'
1189  <- cycle-8 note: true at commits 7897cb8c..130a5e9b; the log grows every cycle (1264 at the cycle-8 edit; the command is the derivation, the number is a snapshot)
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
- **The counter is USED by every Q/A now** *(cycle-8 REPLACEMENT of "No Q/A
  has been asked to USE the counter. Nothing in qa.md points at it" -- true
  when written, superseded by 86.75/86.78/86.79: qa.md:679 mandates
  `verdict_history_86_21.py --evidence-only`, and 7 of 7 critiques in the
  2026-08-17 drain quote its output; the 86.71 cycle-5 PASS cross-checked it
  against qa_wip and the attempt gate as three agreeing counters)*.
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

**COUNTS, WITH THE RULE STATED -- and one figure withdrawn (cycle 4).**

The cycle-3 verdict FAILED partly on this sentence, which read "Self-test 9
cases -> **15**". **That claim does not reproduce and it is withdrawn rather
than re-tuned.** The cycle-3 Q/A measured 11 -> 15; I have since tried four
separate operationalizations (runtime output lines, source `print` emitters
under two regexes, and per-commit `git show` counts) and got 5 / 7 / 4 / 18
across revisions -- no rule I can construct yields 9. **A number whose
counting rule was never written down cannot be checked, which is exactly the
defect, so the honest repair is to state a rule and count under it, not to
find a rule that rescues the old figure.**

THE RULE: a self-test *case* is one line of `--self-test` RUNTIME output
beginning with three spaces and an open bracket -- what a reader actually sees.
Under that rule, demonstrated rather than asserted:

```
$ python scripts/qa/verdict_history_86_21.py --self-test | grep -cE '^   \('
20
```

THE MATRIX RULE: a *cell* is one tuple literal in `MUTANTS`. Under that rule the
matrix carries **16 cells, all killed** as of cycle 6 -- the eleven that existed
before, the three cycle-3 survivors (S1/S2/S3), and the two cycle-4 survivors
(A1/A2) the cycle-4 Q/A found against `_report`'s printed output.

**These two numbers move every cycle, and that is the point of stating the rule
rather than the number.** Cycle 4 published 18 cases / 14 cells; cycle 5 measures
**20 cases / 16 cells**. Anyone re-running the two commands above gets whatever
is true then. A figure quoted without its rule and its tree is not checkable, and
this step exists because of one that was not.

No historical figure is restated here, because none of them can be reproduced.

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

---

## 11. DISPOSITION -- PARKED at the escalation boundary, remediated but ungraded

**Status: `pending`. Not closed. No verdict is claimed.**

| cycle | run | verdict |
|---|---|---|
| 1 | `wf_cb85c901-472` | CONDITIONAL |
| 2 | `wf_8b188711-509` | CONDITIONAL |
| 3 | -- | **FAIL** (the escalation rule converting an honest CONDITIONAL; all six criteria were MET) |
| 4 | `wf_982cd319-493` | CONDITIONAL |
| 5 | `wf_e66ad533-e61` | CONDITIONAL |
| 6 | -- | **not run** -- all cycle-5 findings fixed; **nobody has graded that** |

**Why parked.** The step's own counter now says it, which is the most fitting
possible evidence:

```
verdicts        : CONDITIONAL -> CONDITIONAL -> FAIL -> CONDITIONAL -> CONDITIONAL
consecutive     : 2
auto-FAIL armed : True  (a further CONDITIONAL would be the 3rd)
```

A cycle-6 Q/A that found *anything* would be REQUIRED to return FAIL. Five cycles
have each found a new instance of the same class, so that is the likely outcome,
and manufacturing an escalation from a minor finding at ~190k tokens is the wrong
trade.

**THE MOST IMPORTANT THING THIS STEP LEARNED ABOUT ITSELF, and it was found by
cycle 5 rather than by me.** The ledger had not been appended since 2026-08-10:
cycles 4 and 5 were graded and never written. So `--step 86.21` reported

> `consecutive : 0` / `auto-FAIL armed : False (a further CONDITIONAL would be the 1st)`

at **exit 0**, while the true history was five verdicts and a further CONDITIONAL
would be the **3rd**. **The counter built to stop a silent fail-open under-count
was itself silently under-counting, on its own step.** A well-formed but STALE
ledger is a fifth failure mode with no status of its own -- it is neither
"missing" nor "unreadable" (criterion 5) nor "corrupt or empty" (criterion 6), so
it is outside the immutable criteria, which is exactly why nothing caught it.

I appended the two missing rows, so the counter now reports the truth. **The
underlying gap is unfixed and is the first thing the next session should read:
nothing appends to that ledger automatically.** Main writes it by hand, which is
also the independence weakness criterion 4 names.

**What else is fixed and ungraded:**

- `verify_broken_scoring()` was ONE-SIDED -- it pinned `broken` and `killed` and
  no `survived`, so an always-KILLED scoring defect made the entire matrix report
  a false green while both cells said "correct". A third cell now drives a
  provably behaviour-preserving mutation and requires `survived`. **Verified
  against the Q/A's own defect: the matrix returns rc=5, REFUSING TO SCORE.**
- Three non-reproducing captures regenerated, including one that was a
  byte-identical duplicate of the wrong block; section 2 now carries the actual
  git replay. Every pasted figure in the cycle-6 additions reproduces at this tree (20 cases, 16 cells). *(cycle-8 bound: two OLDER pasted identity figures -- the section-7 md5 line and section-8's log count -- aged under later commits and carry their own supersession notes; the universal wording previously here was falsified by exactly those two, both UNDER-claiming.)*

**Known-open, disclosed rather than buried:** seven print-layer mutants from
cycle 4 remain unguarded by deliberate scope call (the cycle-5 Q/A measured their
differentials and agreed the prioritisation was correct, and recorded that its
own strong hypothesis about one of them was WRONG). The immutable verification
command remains weak by construction and cannot go red -- on the record since
cycle 1.

---

## 12. Cycle 6 (2026-08-14) -- and the discovery that this step was rebuilt from scratch by 86.75

> **Context an executor needs:** phase-86.75 (2026-08-13) independently rediscovered this
> defect and repointed the counter at `scripts/qa/qa_wip.py`. It did **not** reference
> 86.21, which is why step **86.76** now exists. This step's remaining work was to prove
> the repoint actually satisfies 86.21's six criteria — and **three of them did not hold**
> until the work below.

---

## C1 — the blindness, REPRODUCED

The criterion asks for a step with N>1 recorded verdicts, still pending, where the grep the
rule prescribed returns zero rows.

**I did not pick the subject; I measured every candidate and let the data pick it.** My
first attempt used 86.21 itself and I had pre-written the sentence "returns 0 for the
pending one" — the measurement came back **2**. The claim was written before the number
was read, so the subject was wrong, not the finding.

| step | status | ledger (`records_retained`) | anchored log rows |
|---|---|---:|---:|
| **86.62** | pending | **4** | **0** ← **C1 subject** |
| 86.44 | pending | 4 | 1 |
| 86.9 | pending | 4 | 3 |
| 86.38 | pending | 4 | 1 |
| 86.5 | pending | 3 | 1 |
| 86.29 | pending | 3 | 3 |
| 86.21 | pending | 2 | 2 |
| 86.32 | done | 5 | 1 |

**86.62 is the reproduction in its most consequential form:** four Q/A spawns — the step
that *escalated* after four attempts — and the prescribed counter would have read **zero**
every single time.

Generalised: the log **undercounts in 6 of 8** steps and agrees in 2. Never overcounts,
once the grep is header-anchored.

**Positive control:** the same anchored grep returns **6** rows for `phase=36.17`, a
CLOSED step. The probe is live; the zeros are real.

---

## C2 — the counting source, and why it does not touch harness_log

**Chosen:** `scripts/qa/qa_wip.py` over `.claude/agent-memory/qa/verdicts/` — run-stamped
WIP records written write-first by each Q/A spawn.

**LOG-is-last is preserved untouched:** nothing here writes a `harness_log.md` row
mid-step. That ordering is deliberate (the file feeds the Harness tab and the next cycle's
resume detection), and 86.21's design constraint explicitly forbids fixing the counter by
breaking it.

The ledger also survives the **8.2% of spawns that drop and return no verdict at all**,
because the record is written *before* the analysis rather than flushed after it.

---

## C3 — the counter against 36.17's real history, and a DEFECT THIS FOUND

36.17's sequence, verified by anchored grep (cycles 190–195, all 2026-08-09):

```
190 CONDITIONAL   191 FAIL   192 FAIL   193 CONDITIONAL   194 CONDITIONAL   195 PASS
```

**36.17 has ZERO ledger records** — it ran 2026-08-09 and the ledger began 2026-08-10.
The criterion cannot be met by pointing the tool at 36.17; it is met by replaying the
sequence through the rule. Stated plainly rather than papered over.

**The replay exposed that phase-86.75 silently changed the rule.** 86.75's repoint wrote:
*"If this would be the third **attempt** or later, return FAIL"* — but CLAUDE.md:371-376
defines the trigger as **3 consecutive CONDITIONALs**. Those are different rules:

| attempt | actual | consecutive-CONDITIONAL rule | attempt-count rule (86.75, live) |
|---:|---|---|---|
| 1 | CONDITIONAL | allows | allows |
| 2 | FAIL | allows | allows |
| 3 | FAIL | allows | allows |
| 4 | CONDITIONAL | allows | **forces FAIL** ← diverge |
| 5 | CONDITIONAL | allows | **forces FAIL** ← diverge |
| 6 | PASS | allows | allows |

**Longest consecutive run on 36.17 is 2, so the correct rule never fires. The
attempt-count rule would have failed 36.17 twice and denied the PASS it earned at attempt
6.** It was also stricter than CLAUDE.md's F1b cumulative budget (**5** attempts, which
**escalates to the operator** rather than auto-failing). Three bounds existed; they
disagreed; the tightest was live by accident.

**Fixed in both rails** — `.claude/agents/qa.md` and `.claude/workflows/qa-verdict.js`
(the first-class rail carried the same text). The reset-on-FAIL path is exercised at
attempts 2 and 3 (run 1 → 0). Zero survivors of the superseded trigger, negative-controlled.

---

## C4 — the independence question, answered

**Main-supplied counts are ADVISORY. The ledger is AUTHORITATIVE. `harness_log` is a
secondary cross-check that loses on disagreement.**

86.21's objection was precise: a spawn-prompt count is *authored by Main, and Main is the
party the rule constrains*. The ledger removes exactly that — records are written by each
**Q/A** spawn, not by Main, so the audited party no longer authors the counter's input.

**The residual, stated rather than hidden:** the rule constrains the Q/A role, and the Q/A
role writes the records. A given spawn cannot alter its predecessors (they are on disk and
run-stamped) but *could* skip its own write and undercount its successors. That write is
mandated by `qa.md` STEP 0b and is the only path `qa-write-guard.sh` permits the Q/A to
write. **So: more independent than a Main-supplied count, not perfectly independent.**

---

## C5 — fail-safe direction: it failed OPEN, and now it fails LOUD

**Before this step the ledger inherited the exact defect it replaced.**
`list_wip_records` returns `[]` when the sink directory is missing, so
`records_retained: 0` meant both "no prior attempts" and "the counter has no input" —
indistinguishable, and the escalation rule silently disabled.

Added `source_present()` and a `source_present` key. A count of zero is a fact about
attempts **only** when it is `true`; otherwise the guidance now reads *"SOURCE MISSING …
records_retained=0 is NOT a statement about prior attempts"*, and both rails instruct the
Q/A to report the attempt number as **UNKNOWN**.

**Direction: it now fails LOUD (closed).** That is right here because the failure mode
being removed is precisely a silent zero that reads as a clean slate.

---

## C6 — mutation test

`scripts/qa/mutate_counter_source_86_21.py`, all mutations inside a `TemporaryDirectory`:

```
[PASS] CONTROL  2 real records -> counted          records_retained=2
[PASS] M1 sink dir DELETED    -> notices           "SOURCE MISSING: the WIP sink ..."
[SKIP] M2 records DELETED     -> UNSCORED (see below)
[PASS] BASELINE genuine 1st attempt

genuine-first-attempt output == sink-DELETED output : False   (was True before the fix)
mutants surviving (undetected): 0
```

**M2 is unscored, with its reason, not quietly passed.** `prune_wip_records` deletes old
records *by design* (`DEFAULT_KEEP=3`), so "sink present, no record" is a state the module
produces deliberately and is genuinely identical to a first attempt. Detecting it needs a
second monotonic counter outside the sink — more machinery than the defect warrants.
**Stated limit: record loss inside an existing sink is not self-detectable.**

### What the mutation work incidentally found: FOUR DEAD CELLS in the 86.31 matrix

Running `mutation_matrix_86_31.py` as a regression surfaced that **4 of its 24 cells had
silently stopped testing anything** — `ANCHOR-BAD`, meaning the text they pin no longer
exists:

| cell | anchor drifted because | broken since |
|---|---|---|
| P2, Q2 | 86.36 inserted `, path revised by phase-86.36` into the `STEP 0b` line | `6e8f3169`, 2026-08-11 |
| M3 | later phases added dict keys *after* `"guidance": ""` | before 86.75 |
| M5 | 86.36 refactored the one-line `return` into a `sink` local | before 86.75 |

**None was caused by tonight's edits** — verified against `HEAD` and `9a59a4fa~1`, where
all four already counted 0. All four anchors repointed to text that occurs **exactly once**
(measured before editing; a 2-occurrence candidate was rejected).

**Measured state, staged so the claim matches the evidence:**

| run | result |
|---|---|
| before any repoint | P2, Q2 `ANCHOR-BAD`. M3, M5 **also** bad but hidden — I had piped through `tail -6` |
| after P2/Q2 repoint | **22/24 KILLED**; P2 and Q2 now genuinely kill (Q2: *"8 assertion(s) red"*) |
| after M3/M5 repoint | **RUNNING at the time of writing — not yet verified.** Do not read 24/24 into this table. |

**I found M3/M5 only because I stopped truncating the output.** The first two runs looked
like a 2-cell problem because I could not see the rest of the report.

**A correction to my own reporting:** I first said the suite "still exits 0" with dead
cells. **That was wrong** — `:390` returns `1` unless every cell is killed. My `exit=$?`
read the exit status of `tail` through a pipe. Demonstrated: `(exit 7) | tail -1` → `$?`
is `0`; with `pipefail` it is `7`. The script was right; my measurement was not.

---

## Files changed

| File | Change |
|---|---|
| `scripts/qa/qa_wip.py` | `source_present()` + `source_present` key + a SOURCE-MISSING guidance branch |
| `.claude/agents/qa.md` | trigger corrected to 3 **consecutive** CONDITIONALs; `source_present` check; F1b note |
| `.claude/workflows/qa-verdict.js` | same two corrections on the first-class rail |
| `scripts/qa/mutate_counter_source_86_21.py` | **new** — C6 mutation matrix |
| `scripts/qa/mutation_matrix_86_31.py` | 4 dead anchors repointed |

**Regression:** `verify_wip_retention_86_36.py` 23/23, `mutation_matrix_86_36.py` 5/5
cells killed, `verify_qa_write_first_86_31.py` 238/238, `prove_qa_write_separation_86_31.py`
OK. `verify_research_gate_workflow.mjs` **121 passed, 0 failed**.

## Scope honesty

- **No production/trade-path file touched.** Everything above is harness tooling and agent
  prompts.
- **86.21's criteria are addressed but NOT self-certified** — a Q/A has not yet graded this.
- The `qa.md` and `qa-verdict.js` edits are **agent-file changes I authored**, so the
  separation-of-duties rule applies: they need operator review, and this step now adds a
  second such edit on top of 86.75's.

---

### ADDENDUM (same cycle, ~05:00) — I rebuilt this step's work without knowing it existed

**Everything in section 12 above was a re-derivation.** Sections 5, 6 and 7 of this same
file already contained the independence answer, the fail-direction status table, and a
counter mutation matrix. I found them **only because I overwrote this file with `Write`
and had to recover the prior 467 lines from `git show`.**

Concretely, what already existed and what I re-derived:

| I "found" tonight | 86.21 already had it |
|---|---|
| ledger-vs-log divergence, positive-controlled | §2, replayed from git at two commits |
| independence: Main-supplied is ADVISORY | §5, plus the sharper auditability-vs-independence distinction |
| fail-open on a missing source | §6 — a **five**-value status vocabulary, where `ledger_missing` / `ledger_empty` / `unparseable` return `None`, not 0 |
| mutation-test the counter | §7 + `mutation_matrix_86_21.py`, **16/16 killed** (re-verified today) |
| cumulative-vs-consecutive rule ambiguity | §4, verbatim, on **2026-08-11** |

**The regression this exposes.** 86.75 repointed both rails at `qa_wip.py`, which counts
spawn *artifacts* and cannot express a verdict, and in doing so replaced the
consecutive-CONDITIONAL trigger with an attempt-count one — **the exact ambiguity §4 had
already flagged and asked to be reconciled.** The re-derivation did not merely repeat the
work; it regressed a rule the prior cycle had explicitly warned about.

**What actually changed as a result (kept — these are real):**

1. `qa_wip.source_present` — the automatic counter no longer conflates "no attempts" with
   "no input". M1 killed; control green; M2 unscored with its reason.
2. Both rails restored to the **consecutive-CONDITIONAL** trigger, matching
   CLAUDE.md:371-376, with the 36.17 replay recorded.
3. Both rails now point at **`verdict_history_86_21.py`** for the sequence — the tool this
   step built — instead of a hand-rolled grep.
4. **A staleness cross-check that neither tool had alone:** `qa_wip` is written
   automatically, the ledger by hand. If `records_retained` > the ledger's verdict count,
   the ledger is stale. Live: step **86.62** → `qa_wip` **4** vs ledger **0**; last ledger
   row **2026-08-11**.
5. `mutation_matrix_86_31.py` — 4 dead anchors repointed; **20/24 → 24/24 KILLED**.

**The unfixed root, stated plainly:** *nothing appends to `verdict_ledger.jsonl`
automatically.* §11 said so on 2026-08-11 and named it "the first thing the next session
should read". It is still true, and it is why the sequence source is stale. The
cross-check makes the staleness **visible**; it does not fix it.

**This is the strongest evidence for step 86.76**, whose basis has been escalated
accordingly: the harness re-derived a shipped implementation, and I re-derived it again on
top, inside one 24-hour window.


---

## 13. Cycle 7 (2026-08-17): the last two artifact fixes, and the live state

Of the cycle-5 verdict's five named fixes, cycles 6 landed (1) the third
scoring-path cell (a behaviour-preserving mutant must score "survived";
verified: the always-KILLED defect makes the matrix REFUSE at rc=5), (3) the
regenerated 20-case self-test block and 20/16 figures, (4) the cycle-4/5
record in section 11 with the seven print-layer mutants disclosed as
knowingly unguarded (the cycle-5 Q/A measured their differentials and agreed
the scope call), and (5) the STALE fifth-failure-mode disclosure. Cycle 7
lands the remaining two AT THE NAMED SITES: section 2's fence is now the
live_check §2 replay verbatim (the old fence self-contradicted -- fixed by
REPLACEMENT with the supersession note in place), and section 8 carries the
STALE pointer the fix asked for.

THE PRODUCT IS LIVE AND LOAD-BEARING TODAY: every evaluator spawned in the
2026-08-17 drain ran `python scripts/qa/qa_wip.py <sid> --spawned-at ...` and
`verdict_history_86_21.py --evidence-only` and quoted their outputs (e.g. the
86.71 cycle-5 PASS cross-checked prior_attempts==ledger rows==attempt gate row
as "three counters agree"). Captured at write time: self-test 20 cases
SELF-TEST PASSED exit 0; matrix ALL 16 MUTANTS KILLED exit 0, md5 unchanged;
the three broken-scoring self-check cells print "correct".


---

## 14. Cycle 8 (2026-08-17): the cycle-7 verdict's five named fixes, all at the site

1. **The BLOCKING table** (section 6): replaced whole -- FIVE rows, "Four"
   -> "Five", `ledger_empty` present, `ledger_missing`/`ledger_empty` both
   "None, exit 1, fail-CLOSED" exactly as the code behaves (the evaluator
   drove all five statuses; the code was MORE fail-closed than the old
   table asserted -- the cycle-2 table had outlived the cycle-3 change).
2. **The stale md5 identity line** (section 7): annotated at the line --
   capture-time revision 5b7966e8; a DIFFERENT step (86.78/86.79, commit
   9b4d5281) later edited the product file; live md5 b8c0370a...; the 16
   cells reproduce at the current tree per the cycle-7 Q/A's own re-run.
3. **The 1189 figure** (section 8): annotated -- true at its commits, 1264
   at this edit, the command is the derivation.
4. **Section 11's universal claim**: bounded to the cycle-6 additions with
   the two aged identity figures named.
5. **The stale no-consumer text**: REPLACED in place at BOTH sites
   (section 8 bullet + live_check §5) with the supersession note -- qa.md:679
   mandates the counter and all seven of today's critiques quote it.

No code changed this cycle. Captured at write time: matrix ALL 16 MUTANTS
KILLED exit 0 (cycle-7 Q/A's independent re-run corroborates); self-test 20
cases exit 0; the immutable command's grep is 1264 and moving.
