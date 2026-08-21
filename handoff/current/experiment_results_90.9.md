# Experiment Results -- step 90.9

> **STATUS: BUILT AND SELF-VERIFIED, NOT EVALUATED. NOT CLOSEABLE.**
> No Q/A was spawned for this step. The operator stopped the cycle on 2026-08-21 on the
> grounds that running evaluation cycles while the harness's own known defects are still
> unfixed makes each cycle re-discover them at full token cost -- which is exactly what
> happened on 90.1 (5 attempts) and 90.2 (5 attempts) earlier the same day. That judgement
> is recorded here rather than argued with. **The immutable command exits 0 and the
> evidence below is real, but nothing in this file has been independently graded.**

**Step:** 90.9 -- the criteria are the loop's fuel; classify criterion SHAPE by execution
at filing time. **Date:** 2026-08-21. **Contract:** `handoff/current/contract_90.9.md`.
**Research gate:** PASSED (enforced) on the second attempt, `wf_a963cdc4-7d4`,
`handoff/current/research_brief_90.9.md`. The first attempt returned enforced
`gate_passed: false` on an over-claim; that is the gate working and is recorded, not
smoothed.

---

## 1. What was built

| File | Change |
|---|---|
| `scripts/qa/criteria_shape_90_9.py` | **NEW.** Classifies every criterion PRODUCT_BEHAVIOUR / EVIDENCE_APPARATUS by a rule PRINTED beside its output; takes a git-rev corpus pin; three modes with two distinct exit-code meanings. 34 self-test checks. |
| `scripts/qa/mutation_matrix_90_9.py` | **NEW.** 13 cells (1 null + 10 mutants + 2 ERROR controls), each a real subprocess drive of the classifier in a sandbox whose plan and ledger are COPIES, behind a containment guard. |

**Red-first baseline, captured before either file existed:** the immutable command exited
**2**. It now exits **0**. (Captured unpiped -- a piped capture returns the pipe's status,
not the command's, which is how a red baseline can silently read as green.)

## 2. Criterion-by-criterion evidence

### Criterion 1 -- the rule is printed; what reproduces and what does not

**The half that reproduces, exactly.** The step-inclusion rule -- *a node carrying an `id`,
a dict `verification` and a non-empty `verification.success_criteria`, with step 90.9 itself
excluded* -- reproduces the filed corpus **to the digit** at the filing commit:

```
  [PASS] the step-inclusion rule reproduces the filed step count at 252090a3 -- 155 vs filed 155
  [PASS] ...and the filed criterion count -- 980 vs filed 980
```

**The half that does not, and why it CANNOT.** The filed apparatus figure is 403 / 41.1%.
My rule gives **438 / 44.7%**. I did not tune toward 403, and the reason is decisive:
**the filing never recorded the rule that produced it.** It is absent from the masterplan
entry and absent from `research_brief_90.9.md`, which carefully records the step-inclusion
rule and all four unbounded-count regexes but not this one. So 403 is not *unmatched*, it
is *unrecoverable*. Criterion 1's clause "where a figure does not reproduce, the RULE is
corrected and the new figure printed with its step-inclusion rule stated" is the one that
governs, and the filed number is printed beside mine, unedited.

**The corpus moved TWICE, in BOTH directions, and that is measured rather than argued:**

```
  [PASS] the corpus genuinely MOVED, so a live-tree criterion could not have bound -- 155 -> 174 -> 159
```

`085c74e8` added 19 steps (86.127-86.145) **49 minutes after filing**; those same 19 ids
have since **left the 86..90 range entirely**, renumbered into phase-91. So criterion 1's
"by execution on the live tree" is unsatisfiable by construction, and taking its escape
hatch to "correct the rule" would have corrupted a correct rule to chase a moving corpus.
The classifier takes a git-rev pin (house idiom: `replay_changelog_rule_86_68.py:34`,
`sweep_absent_verification_paths.py:421`) and prints both censuses.

**The ratio range, collapsed -- and the finding that makes it worth having.** Criterion 1
asks that the filed 1.6x-1.9x range be collapsed "only by fixing the inclusion rule". At
the fixed inclusion rule and the fixed pin it collapses to **one value per rule**. The
sensitivity table is the point:

```
    NARROW  apparatus=163   16.6%   project= 7.5%   ratio=2.21x
    HOUSE   apparatus=290   29.6%   project=16.6%   ratio=1.79x
    MID     apparatus=438   44.7%   project=27.0%   ratio=1.65x  <- SHIPPED
    BROAD   apparatus=612   62.4%   project=34.4%   ratio=1.82x
```

**The apparatus LEVEL spans 16.6% -> 62.4% across four defensible rules; the RATIO spans
only 1.65x -> 2.21x.** So the thesis-bearing claim -- *phase-86..90 criteria are roughly
1.7x more apparatus-heavy than the project average* -- survives the choice of rule, and
the shipped rule's 1.65x sits inside the filed range. The claim *41.1% of them are
apparatus* does not survive it: that number is a property of a rule nobody wrote down.
**What the filed range was hiding is rule sensitivity, not corpus ambiguity.**

### The unbounded detector tests the PROPERTY, and it costs 12 steps to do so

The brief's v4 keyword proxy reproduces the filed **44** exactly. It is **not shipped**,
because the brief also measured that the only variant actually testing self-reference
returns **0 of 155**: the self-reference is carried by the word "new" plus the surrounding
sentence, never by an explicit "this step adds". A detector landing on the right number
from the wrong property would pass criterion 1 while measuring something else.

The shipped detector asks whether a universal quantifier governs **an artifact class the
step itself PRODUCES**, with no numeric bound in the clause -- because that is what makes
satisfaction grow with the work. It flags **56** steps at the pin against the proxy's 44;
22 property-only, 10 proxy-only, both lists printed. A concrete proxy false positive:
step **90.2** criterion 3 says "byte-identical before and after routing on **all** three
values ... over a **fixture** set" -- the proxy matches `all ... fixture` across 80
characters; the property detector correctly declines it, because the quantifier governs
`values`, a corpus population.

### Criterion 2 -- controls observed GREEN first, and both directions covered

```
  [PASS] the all-product-behaviour control scores 0% evidence-class -- ['PRODUCT_BEHAVIOUR', 'PRODUCT_BEHAVIOUR', 'PRODUCT_BEHAVIOUR']
  [PASS] ...and the gate exits 0 on it
  [PASS] the control whose terminal criterion says 'mutation-test every new guard this step adds' is flagged unbounded
  [PASS] ...and its FIRST criterion is NOT flagged, so the detector discriminates within one step
  [PASS] ...and the gate exits 2 on it
  [PASS] classify DISCRIMINATES: a criterion demanding a mutant + a GREEN control is EVIDENCE_APPARATUS
  [PASS] ...while a pure behaviour statement is PRODUCT_BEHAVIOUR
```

The last two exist because the all-product control alone only proves the classifier does
not OVER-classify. A classifier returning the constant `PRODUCT_BEHAVIOUR` would pass every
control above it -- and mutant **M2** does exactly that, and is KILLED by these.

### Criterion 3 -- exit-code sweep over every step at the pin

```
  swept 155 steps: exit 0 on 99, exit 2 on 56
  [PASS] every step was actually executed and yielded a captured exit code -- 155 of 155
  [PASS] every non-zero exit is attributable to an unbounded criterion, and nothing else -- 56 non-zero
  [PASS] every step carrying NO unbounded criterion exited 0 -- 99 zero-exit steps
  [PASS] the sweep is not vacuous: BOTH outcomes occur -- 99 / 56
```

**Two modes, two exit-code meanings, and they are not conflated.** `--step-file` is the
filing-time gate: exit 2 iff an unbounded criterion is present, exit 0 otherwise, and
classification alone NEVER fails. `--verify` is the self-test and exits 1 on a failed
check. The criterion speaks about the first.

### Criterion 4 -- it cannot mutate the plan, and the source scan is AST-level

```
  [PASS] sha256 of .claude/masterplan.json byte-identical across a FULL classification run over every step in the file
  [PASS] the source contains NO write-capable call, resolved at AST level -- none
  [PASS] ...and the AST scan is not vacuous: it FINDS writes in a known writer
  [PASS] ...including the two literal patterns criterion 4 names
  [PASS] sha256 DISCRIMINATES: two different files do not hash alike
```

**The criterion names `open(...,'w')` and `json.dump`. The house's dominant idiom is
`Path.write_text`** -- 148 sites in `scripts/qa/*.py`, and *both* scripts that write a file
named `masterplan.json` use it. A source scan built from those two literals is
under-inclusive in exactly the idiom this repo writes in. Resolving write-capable calls at
AST level satisfies the criterion strictly *more* than its wording, and the gap is stated
so the strengthening is visible rather than silent. Both required checks are kept -- the
sha256 pair AND the source scan -- and the last line exists because a `sha256` returning a
constant would make every byte-identity check above pass while proving nothing (mutant
**M10**).

### Criterion 5 -- grading is untouched

```
  [PASS] sha256 of handoff/verdict_ledger.jsonl byte-identical before and after -- cddc78f43062bdc8 -> cddc78f43062bdc8
  [PASS] no criterion text can be produced by this module (it only reads and labels)
```

No existing criterion is edited, weakened or removed; criterion 4's sha256 over the plan is
what proves it.

### Criterion 6 -- the bound is justified against the record, and its verdict is NEGATIVE

**Proposed bound:** replace *"mutation-test every new guard this step adds"* with
*"mutation-test each guard a NUMBERED criterion of this step names, plus a null control and
a real-kill control"*.

**How many it would have flagged:** 56 steps at the pin under the property detector, 44
under the brief's proxy -- each would have had its terminal criterion rewritten at filing.

**The most serious real historical finding it would have DEFERRED, and it is not
hypothetical -- it happened today.** Step 90.1's cycle-5 Q/A FAILED that step by authoring
three cells (QA1/QA1b/QA1c) that rename a **call site** inside `attempt_gate.handle_hook`,
proving the matrix's ERROR discriminator vacuous: it required a literal traceback while the
production fail-open handler at `attempt_gate.py:465` prints one line and none. QA1b
defeats **no guard at all** and still fails 7 of 25 checks, three of them belonging to
criteria 2, 3 and 4 -- a build that never runs was green-washing three criteria at once.
**No numbered criterion of 90.1 named those cells.** They were the evaluator's own probes.
Under the proposed bound they would never have been written.

**Disposition, FILED rather than dropped:** masterplan step **90.12**, with its own
immutable verification command.

**So the bound is NOT recommended as written**, and the classifier prints that conclusion
rather than a recommendation: it needs an explicit carve-out for cells the EVALUATOR
authors, because the evaluator is the one party whose probes are not scoped by the step's
own criteria. That carve-out is the difference between bounding the fuel and bounding the
fire.

### Criterion 7 -- the input surface, and a probe that first failed against itself

```
  [PASS] no classification function references a verdict history, a WIP record, a round index or a remaining attempt budget -- neither handed in nor SELF-read (AST, scoped to the classification path) -- none
  [PASS] ...and that scan is NOT vacuous: a planted qa_wip self-read IS detected
  [PASS] ...and it catches a consequence field read straight off the plan object
  [PASS] the classification path's ENTIRE I/O surface is the plan of record -- named, not asserted -- load_plan: run(); load_plan: read_text()
  [PASS] classify() is pure over its argument
  [PASS] ...and its signature admits no step id, verdict or attempt count -- ['criterion', 'variant']
```

**"Never READS" is the binding verb, not "never given".** The research correction matters
here: `qa-verdict.js` does not hand `verdict_sequence` or `attempt_number` to the judge at
all -- those names appear only in `KNOWN_ARG_KEYS`, whose job is silent-input-loss
reporting. The live consequence channel on the sibling rail is a **self-read**
(`qa_wip.py --spawned-at`). So the test has to cover a self-read path, and it does.

**A defect in my own first version of this check.** It was a text grep over the whole
source for the forbidden tokens -- and it FAILED, because `FORBIDDEN_INPUTS` *is* a list of
those tokens. **The probe matched its own definition.** A probe that cannot distinguish its
own source from its subject measures nothing. It is replaced by an AST scan scoped to the
classification functions, with two planted-positive controls so "none" cannot be confused
with a broken walk.

**Disclosed rather than claimed away:** the SELF-TEST *does* read
`handoff/verdict_ledger.jsonl` -- to hash it twice and prove it unchanged, which criterion 5
requires. A whole-module ban would have been a false claim. The scan is scoped so the
distinction between "hashing a file to prove it did not change" and "reading verdict
history as classification input" is enforced, not asserted.

## 3. The mutation matrix

13 cells: null + 10 mutants + 2 ERROR controls. Every cell is a real subprocess drive.

```
KILLED 10 | SURVIVED 0 (excl. N0) | ERROR 2 | null mutant survived: True |
real-kill control killed: True | error controls: ['ERROR', 'ERROR']
real tree untouched: True
```

**Containment is enforced, not promised.** The sandbox gets its own **copies** of the
masterplan and the verdict ledger (never symlinks), a containment guard refuses to run at
all if any sandbox path resolves inside the repository, and the real files' md5s are
compared before and after. This is the direct lesson of 90.1 cycle 4, where **one deleted
redirect line** stood between a self-test and the project's verdict history and the test
reported PASSED while truncating it.

**The ERROR discriminator is typed, and it is proven to fire without a traceback:**

```
  a SWALLOWED one-liner with no traceback  -> NameError
  a real traceback ending in AssertionError -> None
  ok: the discriminator reads the TYPE, not the shape
```

Three lessons are built in, each paid for in a real cycle: "fails to run" is not "fails to
parse", so cells are driven and the drive includes an import (QXI is a module-scope import
that does not exist); an over-eager probe silently deletes legitimate cells, so a domain
exception is never scored ERROR; and the type must be read from the **message**, because a
name failure inside a fail-open handler emits no traceback at all -- the finding that
failed step 90.1 hours before this was written.

**Two survivors on the first run, and they are the same lesson twice.** M5 (numeric-bound
escape removed) and M6 (corpus-precedence check removed) both **SURVIVED**. Both branches
looked covered -- I had fixtures for `"every attempt row"` and `"all 155 steps"` -- but
neither sentence contains a step-produced artifact noun, so the detector declines them one
step *earlier* and neither branch was ever reached. **A fixture only tests a branch if that
branch is the ONLY thing standing between the input and the outcome.** Replaced with
`"every attempt row is covered by a guard"` and `"mutation-test all 3 guards this step
adds"`, each pairing an artifact noun with the bounding feature under test, plus an
assertion that both would be flagged without it.

**And the new fixture immediately found a real bug.** `"every attempt row is covered by a
guard"` was flagged UNBOUNDED. The corpus-precedence test compared `cm.start() < m.start()`
-- but both patterns anchor on the *same* universal quantifier, so their match starts are
identical and `0 < 0` is never true. **The branch had never discriminated anything.** Fixed
to compare the NOUN positions (`cm.start(2) < m.start(2)`); the pinned unbounded count moved
62 -> 56 as a result, which is the correction, not a tuning.

## 4. Verification, verbatim

```
$ bash -c 'python3 scripts/qa/criteria_shape_90_9.py --verify && python3 scripts/qa/mutation_matrix_90_9.py --verify'
  failed: 0
KILLED 10 | SURVIVED 0 (excl. N0) | ERROR 2 | null mutant survived: True
IMMUTABLE COMMAND EXIT: 0
```

Full output in `handoff/current/live_check_90.9.md`.
