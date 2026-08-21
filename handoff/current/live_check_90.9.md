# live_check -- step 90.9

**Step:** 90.9 -- the criteria are the loop's fuel; classify criterion SHAPE at filing time.
**Date:** 2026-08-21. **Contract:** `handoff/current/contract_90.9.md`.

The live_check the masterplan asks for: *"the derived threshold with corpus size and date,
beside the historical claim"* -- here, the classification rule printed beside both
censuses, the exit-code sweep, and the mutation matrix.

---

## 1. The immutable command, unpiped

```
$ bash -c 'python3 scripts/qa/criteria_shape_90_9.py --verify && python3 scripts/qa/mutation_matrix_90_9.py --verify'
IMMUTABLE COMMAND EXIT: 0
```

Red-first baseline, captured BEFORE either script existed (2026-08-21):

```
$ bash -c 'python3 scripts/qa/criteria_shape_90_9.py --verify && python3 scripts/qa/mutation_matrix_90_9.py --verify'
python3: can't open file '.../scripts/qa/criteria_shape_90_9.py': [Errno 2] No such file or directory
EXIT 2   (captured UNPIPED -- a piped capture returns the pipe's status, not the command's)
```

## 2. The rule, both censuses, the sensitivity table and the bound (criteria 1, 6)

`python3 scripts/qa/criteria_shape_90_9.py --census`:

```
==============================================================================
THE CLASSIFICATION RULE (printed beside its output -- criterion 1)
==============================================================================
A criterion is EVIDENCE_APPARATUS when satisfying it requires producing or
exercising VERIFICATION MACHINERY -- a mutant, a control, a hash comparison, a
re-runnable checker, a fixture, a captured artifact, an exit code, or a test that
asserts something. It is PRODUCT_BEHAVIOUR when it constrains what the SYSTEM
DOES. Criteria are compound in this codebase, so ANY apparatus demand classifies
the whole criterion as apparatus: the question is whether the criterion adds
verification work, not whether that is all it does.

This rule is AUTHORED, not adopted: research_brief_90.9.md K6 records that the
published requirements-smell taxonomies carry no product/apparatus axis, so
citing one as authority would be false. Dispute the term list below, not a
citation.

  variant SHIPPED: MID
  step-inclusion rule: a node carrying an `id`, a dict `verification`, and a non-empty `verification.success_criteria`; step 90.9 itself EXCLUDED (its own criteria are the whole delta that made the filed figures look irreproducible)
  apparatus terms:
    \bmutant
    \bmutation
    \bKILLED\b
    \bSURVIVED\b
    control .{0,20}GREEN
    \bred-first\b
    \bvacuous
    \bsha256\b
    byte-identical
    \bre-runnable\b
    \bchecker\b
    \bself-test\b
    \bcardinality floor\b
    \bfixture
    \bcell\b
    \bregression\b
    \bdry.run\b
    \bverbatim\b
    \bprinted\b
    \bcaptured?\b
    \bexits? non-zero\b
    \bexits? 0\b
    \bexit code\b
    \btests?\b
    \basserted\b
    \bassertion\b
    \bproven\b
    \bproof\b

==============================================================================
CENSUS -- BOTH CORPORA PRINTED, NEITHER EDITED (criterion 1)
==============================================================================
  PINNED 252090a3 (filing)           steps=155   criteria=980    apparatus=438    44.7%   terminal=91/155   project=1263/4670 =27.0%   ratio=1.65x
  DRIFT  085c74e8 (+49 min)          steps=174   criteria=1045   apparatus=457    43.7%   terminal=95/174   project=1282/4735 =27.1%   ratio=1.62x
  LIVE tree (today)                  steps=159   criteria=999    apparatus=455    45.5%   terminal=95/159   project=1304/4784 =27.3%   ratio=1.67x

  FILED:  steps=155  criteria=980  apparatus=403  41.1%  terminal=78  project=1026/4670 =22.0%  ratio 1.6x-1.9x  unbounded=44

  WHAT REPRODUCES, AND WHAT DOES NOT:
    steps / criteria  -- REPRODUCE EXACTLY at 252090a3: 155 / 980
                         vs filed 155 / 980. The step-inclusion rule is recovered
                         and is printed above.
    apparatus / pct   -- DO NOT REPRODUCE: 438 / 44.7% vs filed 403 / 41.1%.
                         The filing never recorded the rule that produced 403 --
                         it is absent from the masterplan entry AND from
                         research_brief_90.9.md. It is therefore not recoverable,
                         not merely unmatched. This script's rule is printed above
                         and its figure stands beside the filed one. Criterion 1's
                         "the RULE is corrected and the new figure printed" is the
                         clause that governs. THE FILED NUMBER IS NOT EDITED.
    corpus drift      -- the corpus moved TWICE and in BOTH directions:
                         252090a3 155 -> 085c74e8 174 (+19, steps 86.127-86.145,
                         49 minutes after filing) -> live 159, because those same
                         19 ids have since left the 86..90 range entirely. A
                         "live tree" criterion cannot bind on a corpus that moves
                         faster than the step does.

==============================================================================
SENSITIVITY -- the LEVEL is the rule's doing; the RATIO is the corpus's
==============================================================================
  (all at the filing pin, same inclusion rule, only the term list varies)
    NARROW  apparatus=163   16.6%   project= 7.5%   ratio=2.21x
    HOUSE   apparatus=290   29.6%   project=16.6%   ratio=1.79x
    MID     apparatus=438   44.7%   project=27.0%   ratio=1.65x  <- SHIPPED
    BROAD   apparatus=612   62.4%   project=34.4%   ratio=1.82x

  The apparatus LEVEL spans 16.6% -> 62.4% across four defensible rules.
  The RATIO spans only 1.65x -> 2.21x, and the shipped rule's 1.65x sits inside
  the filed 1.6x-1.9x range. So: "phase-86..90 criteria are ~1.7x more
  apparatus-heavy than the project average" SURVIVES the choice of rule.
  "41.1% of them are apparatus" DOES NOT -- that number is a property of a rule
  nobody wrote down. Criterion 1 asks for the range to be collapsed "only by
  fixing the inclusion rule": fixing the inclusion rule and the corpus pin
  collapses it to ONE value PER RULE (1.65x for the shipped one). What the filed
  range was hiding is rule sensitivity, not corpus ambiguity.

==============================================================================
UNBOUNDED SCOPE -- the quantified NOUN CLASS, not a keyword
==============================================================================
  property-based detector : 56 steps at the filing pin
  brief's v4 keyword proxy: 44 steps (filed: 44)
  property-only: 22 ['86.102', '86.11', '86.115', '86.121', '86.28', '86.41', '86.44', '86.46', '86.47', '86.52', '86.53', '86.58']
  proxy-only:    10 ['86.123', '86.124', '86.126', '86.43', '87.1', '87.2', '87.3', '87.5', '88.3', '90.2']

  WHY THE PROXY IS NOT SHIPPED even though it reproduces 44 exactly: the brief
  measured that its literal self-reference variant returns 0 of 155 -- the
  self-reference is carried by the word "new" plus the surrounding sentence, never
  by an explicit "this step adds". A detector that lands on the right number from
  the wrong property would pass criterion 1 while measuring something else, which
  is the house rule "assert the property, not a proxy". The proxy's count is
  printed here so the difference is auditable, and it is never the gate.

==============================================================================
THE BOUND, JUSTIFIED AGAINST THE RECORD (criterion 6)
==============================================================================
  PROPOSED BOUND: replace "mutation-test every new guard this step adds" with
  "mutation-test each guard a NUMBERED criterion of this step names, plus a null
  control and a real-kill control".

  How many it would have flagged: 56 steps at the filing pin carry a criterion
  the property-based detector calls unbounded (44 under the brief's proxy, which
  is the filed 44). Every one of them would have had its terminal criterion
  rewritten at filing time.

  THE MOST SERIOUS REAL HISTORICAL FINDING THE BOUND WOULD HAVE DEFERRED --
  and it is not hypothetical, it happened on 2026-08-21:

    Step 90.1's cycle-5 Q/A FAILED the step by authoring three mutation cells
    (QA1/QA1b/QA1c) that rename a CALL SITE inside attempt_gate.handle_hook. They
    proved the matrix's ERROR discriminator VACUOUS: it requires a literal
    "Traceback (most recent call last)", while the production fail-open handler at
    attempt_gate.py:465 catches Exception and prints a one-line INTERNAL ERROR with
    no traceback. QA1b defeats NO guard yet fails 7 of 25 checks -- three of them
    belonging to criteria 2, 3 and 4 -- so a build that never runs was
    green-washing three criteria at once.

    NO NUMBERED CRITERION OF 90.1 NAMED THOSE CELLS. They were the evaluator's own
    probes of the discriminator. Under the proposed bound they would never have
    been written, and the vacuous guard would have shipped green.

  DISPOSITION, FILED RATHER THAN DROPPED: masterplan step 90.12, with its own
  immutable verification command. The bound is therefore NOT recommended as
  written: it must carry an explicit carve-out for cells the EVALUATOR authors,
  because the evaluator is the one party whose probes are not scoped by the
  step's own criteria. That carve-out is the difference between bounding the
  fuel and bounding the fire.

```

## 3. The self-test, verbatim (criteria 2, 3, 4, 5, 7)

`python3 scripts/qa/criteria_shape_90_9.py --verify`:

```
==============================================================================
A. CONTROLS OBSERVED FIRST (criterion 2)
==============================================================================
  [PASS] the all-product-behaviour control scores 0% evidence-class -- ['PRODUCT_BEHAVIOUR', 'PRODUCT_BEHAVIOUR', 'PRODUCT_BEHAVIOUR']
step 'CTRL.PRODUCT': 3 criteria
  rule: MID  (see --census for the full rule)
  1. PRODUCT_BEHAVIOUR
  2. PRODUCT_BEHAVIOUR
  3. PRODUCT_BEHAVIOUR

EXIT 0 -- no unbounded criterion. Shape is reported, never graded.
  [PASS] ...and the gate exits 0 on it
  [PASS] the control whose terminal criterion says 'mutation-test every new guard this step adds' is flagged unbounded -- a universal quantifier (every) governs 'guard', an artifact class the step itself PRODUCES
  [PASS] ...and its FIRST criterion is NOT flagged, so the detector discriminates within one step
step 'CTRL.UNBOUNDED': 2 criteria
  rule: MID  (see --census for the full rule)
  1. PRODUCT_BEHAVIOUR
  2. EVIDENCE_APPARATUS  <-- UNBOUNDED SCOPE
       a universal quantifier (every) governs 'guard', an artifact class the step itself PRODUCES, with no numeric bound in the clause -- so satisfying the criterion grows with the work

EXIT 2 -- 1 criterion(a) carry unbounded scope. This is the ONLY condition on which this tool fails a step; classification alone never fails.
  [PASS] ...and the gate exits 2 on it
  [PASS] classify DISCRIMINATES: a criterion demanding a mutant + a GREEN control is EVIDENCE_APPARATUS
  [PASS] ...while a pure behaviour statement is PRODUCT_BEHAVIOUR

==============================================================================
B. THE DETECTOR DISCRIMINATES BOUNDED FROM UNBOUNDED
==============================================================================
  [PASS] a corpus population governs, even with an artifact noun later in the clause ('every attempt row is covered by a guard') -- only the corpus-precedence branch excludes this
  [PASS] an enumerated artifact population is BOUNDED ('all 3 guards this step adds') -- only the numeral escape excludes this
  [PASS] ...and both fixtures WOULD be flagged without their bounding feature, so neither passes vacuously
  [PASS] a step-produced artifact class is UNBOUNDED ('every guard it adds')
  [PASS] ...and so is 'each new probe this step introduces'

==============================================================================
B2. THE HALF OF THE FILED FIGURES THAT DOES REPRODUCE (criterion 1)
==============================================================================
  [PASS] the step-inclusion rule reproduces the filed step count at 252090a3 -- 155 vs filed 155
  [PASS] ...and the filed criterion count -- 980 vs filed 980
  [PASS] the filed APPARATUS figure does NOT reproduce, and that is REPORTED rather than fitted -- 438 (44.7%) vs filed 403 (41.1%) -- the filing never recorded its rule
  [PASS] the shipped rule's ratio falls inside the filed 1.6x-1.9x range -- 1.65x
  [PASS] the corpus genuinely MOVED, so a live-tree criterion could not have bound -- 155 -> 174 -> 159

==============================================================================
C. EXIT-CODE SWEEP OVER EVERY STEP AT THE PIN (criterion 3)
==============================================================================
  swept 155 steps: exit 0 on 99, exit 2 on 56
  [PASS] every step was actually executed and yielded a captured exit code -- 155 of 155
  [PASS] every non-zero exit is attributable to an unbounded criterion, and nothing else -- 56 non-zero
  [PASS] every step carrying NO unbounded criterion exited 0 -- 99 zero-exit steps
  [PASS] the sweep is not vacuous: BOTH outcomes occur -- 99 / 56

==============================================================================
D. IT CANNOT MUTATE THE PLAN (criterion 4 -- BOTH checks, AST-level)
==============================================================================
  [PASS] sha256 of .claude/masterplan.json byte-identical across a FULL classification run over every step in the file -- 1a02aa22d7c5a6e5 -> 1a02aa22d7c5a6e5
  [PASS] the source contains NO write-capable call, resolved at AST level -- none
  [PASS] ...and the AST scan is not vacuous: it FINDS writes in a known writer
  [PASS] ...including the two literal patterns criterion 4 names
  [PASS] sha256 DISCRIMINATES: two different files do not hash alike

==============================================================================
E. GRADING IS UNTOUCHED (criterion 5)
==============================================================================
  [PASS] sha256 of handoff/verdict_ledger.jsonl byte-identical before and after -- cddc78f43062bdc8 -> cddc78f43062bdc8
  [PASS] no criterion text can be produced by this module (it only reads and labels)

==============================================================================
F. THE INPUT SURFACE (criterion 7 -- 'never READS' is the binding verb)
==============================================================================
  [PASS] no classification function references a verdict history, a WIP record, a round index or a remaining attempt budget -- neither handed in nor SELF-read (AST, scoped to the classification path) -- none
  [PASS] ...and that scan is NOT vacuous: a planted qa_wip self-read IS detected -- classify: 'qa_wip' in 'scripts/qa/qa_wip.py' (line 3)
  [PASS] ...and it catches a consequence field read straight off the plan object -- census: 'retry_count' in 'retry_count' (line 2)
  [PASS] the classification path's ENTIRE I/O surface is the plan of record -- named, not asserted -- load_plan: run() at line 243; load_plan: read_text() at line 242
      classification-path I/O calls, in full: load_plan: run() at line 243; load_plan: read_text() at line 242
      DISCLOSED: the SELF-TEST reads handoff/verdict_ledger.jsonl to hash it twice,
      which criterion 5 requires. That read is outside the classification path and
      is what the scope above enforces.
  [PASS] classify() is pure over its argument: same text in, same label out, with no step identity available to it
  [PASS] ...and its signature admits no step id, verdict or attempt count -- ['criterion', 'variant']

==============================================================================
SUMMARY
==============================================================================
  failed: 0

```

## 4. The mutation matrix, verbatim (criterion 2)

`python3 scripts/qa/mutation_matrix_90_9.py --verify`:

```
==============================================================================
CONTROL (the real, unmutated classifier)
==============================================================================
  exit=0   failed-check lines: 0
  CONTROL GREEN

==============================================================================
THE ERROR DISCRIMINATOR IS TYPED, AND IT FIRES WITHOUT A TRACEBACK
==============================================================================
  a SWALLOWED one-liner with no traceback  -> NameError
  a real traceback ending in AssertionError -> None
  ok: the discriminator reads the TYPE, not the shape

==============================================================================
CELLS
==============================================================================
  ok   N0   SURVIVED  expected SURVIVED  
         NULL MUTANT (comment only). If this scores KILLED the harness is broken and every other kill this run is meaningless.
  ok   M1   KILLED    expected KILLED    
         criterion 2, NAMED: classify() labels EVERY criterion evidence-class. The all-product-behaviour control must kill this.
  ok   M2   KILLED    expected KILLED    
         classify() labels every criterion product-behaviour -- the opposite constant. The all-product control alone CANNOT catch this, which is why a discriminating check exists.
  ok   M3   KILLED    expected KILLED    
         the unbounded detector never fires, so the one condition this tool exits non-zero on becomes unreachable.
  ok   M4   KILLED    expected KILLED    
         the unbounded detector always fires, so every step is flagged and the gate stops discriminating.
  ok   M5   KILLED    expected KILLED    
         the numeric-bound escape is removed, so an enumerated population ('all 155 steps') is misread as unbounded.
  ok   M6   KILLED    expected KILLED    
         the corpus-population check is removed, so quantifying over a fixed corpus ('every attempt row') is misread as growing with the work.
  ok   M7   KILLED    expected KILLED    
         criterion 1: the step-inclusion rule stops excluding step 90.9 itself, so the one figure that DOES reproduce stops reproducing.
  ok   M8   KILLED    expected KILLED    
         criterion 4: the AST write-scan returns nothing, so 'no write path' becomes vacuously true.
  ok   M9   KILLED    expected KILLED    
         criterion 7: the consequence scan returns nothing, so 'never reads a verdict history' becomes vacuously true.
  ok   M10  KILLED    expected KILLED    
         criteria 4 and 5: sha256 returns a constant, so every byte-identity check passes while proving nothing.
  ok   QX   ERROR     expected ERROR     NameError
         ERROR CONTROL 1: a CALL SITE is renamed. The code parses, imports, and then cannot RESOLVE A NAME at run time. It must score ERROR, never a kill.
  ok   QXI  ERROR     expected ERROR     ModuleNotFoundError
         ERROR CONTROL 2: a MODULE-SCOPE import that does not exist. It parses cleanly and dies on IMPORT -- 'fails to run' is not 'fails to parse'.

==============================================================================
CONTAINMENT
==============================================================================
  .claude/masterplan.json md5      817360f2dce2921d4c1d7c975b41e9a9 -> 817360f2dce2921d4c1d7c975b41e9a9
  handoff/verdict_ledger.jsonl md5 2ed4acb78ea61ff4f81c7bdcac4a7b87 -> 2ed4acb78ea61ff4f81c7bdcac4a7b87
  real tree untouched: True

==============================================================================
KILLED 10 | SURVIVED 0 (excl. N0) | ERROR 2 | null mutant survived: True | real-kill control killed: True | error controls: ['ERROR', 'ERROR']
==============================================================================

```
