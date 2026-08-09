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
   (ii)  reset on PASS_WITH_FINDINGS -> consecutive=1 (expect 1)
   (iii) corrupt ledger -> status=unparseable, consecutive=None (expect None, NOT 0)
   (iv)  missing ledger -> status=ledger_missing (expect ledger_missing)
   (v)   unknown step -> status=no_rows_for_step, consecutive=0 (expect no_rows_for_step, 0)

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
md5    : 96f91ffb50a0a5a3fb68ab6df69c105a

[control] un-mutated self-test rc=0 (0 = PASSED)
  KILLED  | M1: unparseable reports 0 instead of None (the silent zero returns)
            self-test rc=1
  KILLED  | M2: reset becomes == 'PASS' (misses PASS_WITH_FINDINGS / PASS_AFTER_RETRY)
            self-test rc=1
  KILLED  | M3: corrupt rows are ignored instead of counted (fail-open restored)
            self-test rc=1

[integrity] target md5 unchanged: True
ALL 3 MUTANTS KILLED -- every guard IN THIS MATRIX can fail.
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
