STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 90.2
WRITTEN: 2026-08-21T07:14:22Z

# Q/A write-first record -- step 90.2 (cycle 1 per Main's disclosure)

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command exit code; git scope; lint; replay
C. Mutation matrix (independent, fixture+harness side)
D. Criterion-by-criterion MET/NOT MET

## Findings log (append-only)

### Attempt/sequence evidence
- qa_wip.py 90.2 --spawned-at 2026-08-21T07:14:22Z -> source_present=true,
  attempt_number=1 (status ok, is_lower_bound=false), prior_attempts=0,
  records_retained=1 (own record), prior_records=[].
- verdict_history_86_21.py --step 90.2 --evidence-only -> status=no_rows_for_step,
  verdicts=(none). prior_attempts(0) is NOT > ledger count(0): no staleness signal.
- Sequence for 90.2: NO PRIOR VERDICTS RECORDED. Consistent with Main's cycle-1
  disclosure.

### A. Harness compliance (5 items)
1. research gate: handoff/current/research_brief_90.2.md exists (38,446 B). [gate
   envelope check pending]
2. order (local mtimes): brief 2026-08-20 21:39 < contract 21:52 < code
   2026-08-21 09:08-09:13 < experiment_results 09:13. ORDER OK.
3. experiment_results_90.2.md present (12,903 B); live_check_90.2.md (14,465 B).
4. log-last: `grep -cF "phase=90.2" handoff/harness_log.md` = 0; masterplan 90.2
   status = "pending". NOT yet logged, NOT yet flipped. OK.
5. no-verdict-shopping: cycle 1, no prior verdict. N/A.

### B. Deterministic
- IMMUTABLE COMMAND: `bash -c 'node --check .claude/workflows/qa-verdict.js && node
  scripts/qa/verify_severity_routing_90_2.mjs --self-test'` -> EXIT 0.
  61 checks (floor 50), 0 failed. Mutation matrix K: N0 SURVIVED, M1-M9 KILLED,
  QX ERROR, 0 unexpected, no NO-OP.
- git status --short: only handoff/audit/*.jsonl (append-only audit streams) +
  my own WIP file. NO unintended production change.
- commit c09bd96b touches 7 files: qa-verdict.js (+178/-1), 3 new scripts/fixtures,
  2 handoff artifacts, 1 audit jsonl. Scope matches the contract.

### FINDING 1 (BLOCKING) -- criterion 4's "247" DOES reproduce; the artifact says it does not
The shipped replay filters the corpus with `if (wn !== 'qa-verdict') continue`
(verify_severity_routing_90_2.mjs, loadCorpus). The FILING's population -- named
verbatim in masterplan 90.2 audit_basis as "441 `qa-verdict` Workflow run records" --
is the `startsWith('qa-verdict')` set, which the replay itself prints as
`startsWith=441 exact=436`.

Independently re-derived by me over the startsWith population at the same pin
(1787056437731):
    with_verdict=397  PASS=109  COND=221  FAIL=67  nonPASS=288
which is BYTE-FOR-BYTE the filing's census (397 / 109 / 221 / 67 / 288).
Then driving the REAL shipped enforceSeverityRouting over that population:
    non-PASS=288  queue_residual=41  remediate=247
=> the criterion-4 pair 41/247 REPRODUCES EXACTLY.

The 5 dropped records are workflowName "qa-verdict-writefirst-82-5" (x3) and
"qa-verdict-writefirst-82-7" (x2); verdicts PASS, CONDITIONAL, FAIL, CONDITIONAL,
PASS -> 2 PASS + 3 non-PASS, all 3 routing to remediate. 247 - 3 = 244.

experiment_results/live_check and the replay footer assert: `"247" DOES NOT
reproduce ... The number is NOT edited to match.` That assertion is FALSE under the
population the criterion is derived from. The author narrowed the scope with a
filter the filing did not use and then reported the criterion's own number as
unreproducible. Classification: Contradiction / scope narrowed by the author
(qa.md 4b "Scopes must be DERIVED, not typed").

### Cross-check that PARTIALLY EXONERATES the "32" half
Same census, both populations, six strict definitions:
  startsWith: token-anywhere 41 | bracketed-anywhere 26 | starts-with-bare 11 |
              starts-with-separator 4 | bracket-or-paren prefix 0
  exact:      identical (41/26/11/4/0)
"32" reproduces under NONE. Main's "32 does not reproduce" claim STANDS and is
population-independent. Only the "247" claim is wrong.

Also wrong: the stated CAUSE. experiment_results §3-C4 says "The gap lives in
`parseable`: 43 of the 436 pinned records carry result: null". The parseable gap is
436 -> 393; the 397 -> 392 verdict gap is a DIFFERENT gap and is exactly the 5
records dropped by `wn !== 'qa-verdict'` (2 PASS, 2 CONDITIONAL, 1 FAIL). The
replay's own header prints startsWith=441 exact=436 -- the difference was on screen.

### FINDING 2 (BLOCKING) -- criterion 1's THROW has SOLE-COVERAGE source-scan vacuity
Section I's four checks are regexes over the workflow source read from disk; the
leak guard is never EXECUTED (it lives at module top-level, which the checker
cannot import -- it says so at :19-22). Applied two neutering mutants to an
IN-MEMORY copy (no file write) and evaluated the four Section-I predicates verbatim:
  CONTROL                                        PASS PASS PASS PASS  => GREEN
  L1: `if (false && leakedS.length > 0)`         PASS PASS PASS PASS  => GREEN
  L2: if/throw deleted, message left in a comment PASS PASS PASS PASS  => GREEN
qa.md 4c shapes #3 (literal kept, behaviour stripped) and #8 (comment-token trap);
skill heuristic #17 illusory-guard, sole coverage.
No mutation cell targets it: M1-M9 + QX all mutate enforceSeverityRouting internals.

NOT a fundamental limit -- a behavioural test is feasible in ~12 lines using the
checker's OWN extraction idiom. I built one and it DISCRIMINATES:
  A sibling shape           -> NO THROW
  B flattened into verdict  -> THROWS "phase-90.2 invariant violated..."
  C judge emitted `route`   -> THROWS
So the PRODUCT guard is correct; the CHECKER cannot tell if it stops being.

### C. My independent mutation battery (fingerprint diff over the checker's OWN 47 inputs)
  Q1 verdict string mutated in place ......... KILLED (21/47 differ)  [criterion 3 holds]
  Q2 delimiter `after` window 45 -> 400 ...... SURVIVES
  Q3 non-string entry filter removed ......... SURVIVES
  Q4 BLOCK admitted to the residual set ...... KILLED (2/47)
  Q5 route hardcoded 'remediate' ............. KILLED (8/47)
  Q6 derived_severities index alignment lost .. KILLED (21/47)
BEHAVIOURAL DIFFERENTIAL on the two survivors (feedback_survivor_needs_behavioural_differential):
  Q2: queue_residual 41 -> 41, symmetric difference NONE; 0 of 906 real entries change class.
  Q3: queue_residual 41 -> 41, symmetric difference NONE.
=> BOTH ARE EQUIVALENT MUTANTS on the observed population. NOT findings. The 45-char
   window is an unmeasured magic number but makes no observable difference. Retired.

### D. Supporting-claim reproduction
  "0 of 978 violation_details rows carry a severity key" -> I measure 969 rows /
     0 with severity on the startsWith population. Substance HOLDS (0), count differs
     because Main computed on the narrowed exact population.
  "0 of 66 FAILs at the pin are all-WARN/NOTE" -> 67 FAILs on startsWith, 0 all-WARN.
     Substance HOLDS. Again 66 vs 67 = the population narrowing.
  Tag-form table (initial 41 / bracket 88 / paren 29 / colon 20 / dash 7, bare 12):
     my independent tally gives 41/91/37/1/5/2 (total 177 vs ~197). "initial 41" matches
     exactly; the rest depends on an unpublished precedence rule. No reproducing command
     is given. NOTE-level (supporting rationale, not a criterion).
  24 fixture returns: VERBATIM MATCH 24/24 against the real run records, field by field.
     Buckets 6 PASS / 6 FAIL / 6 CONDITIONAL_allwn / 6 CONDITIONAL_mixed.

### E. NOTE-level observations
  N1 residual_close_gate.mjs has NO consumer outside its own checker (grep across
     .claude/, scripts/, docs/, CLAUDE.md). Criterion 5 only asks for "a checker", so
     MET -- but the artifact calls it "the consumer half ... refuses a parent step's
     close" without disclosing that nothing in a close path invokes it.
  N2 the routing scores only violated_criteria; violation_details content is never
     classified. 3 of the 41 queue_residual runs carry unmatched detail rows -- all
     three are explicitly "SEVERITY NOTE", so no live counterexample, but the bound
     is undisclosed.
  N3 wf_555a4380-3e8 routes to queue_residual with a KILL-SWITCH behaviour finding
     inside it -- judge-tagged [WARN] on all three entries. Faithful to 86.98's
     "severity comes from the judge", but worth an operator's eye before this
     routing obliges anything.

### F. Gates N/A, stated rather than skipped
  1a python lint: `git diff --name-only c09bd96b~1 c09bd96b -- '*.py'` = 0 files. N/A.
  1b frontend: diff touches no frontend/** and not .claude/agents/qa.md. N/A.
  1c live UI: the step makes no UI claim. N/A.
  1d backend smoke: diff touches no backend/**. N/A.
  Derived scope incl. untracked (`git ls-files --others --exclude-standard`): only my
  own WIP file. Uncommitted: append-only handoff/audit/*.jsonl. No unintended change.

## VERDICT DIRECTION (still not a verdict)
FAIL. Criterion 4 not met (the required 41/247 replay proof is absent, and the
artifact's claim that 247 is unreproducible is falsified by measurement) +
criterion 1's THROW clause has sole-coverage source-scan vacuity.
Criteria 2, 3, 5, 6 MET. Harness compliance clean. Product code is CORRECT.

## FINAL RE-DERIVATION (single clean pass, HEAD=4c449680, tree clean but for audit jsonl)
  FILING population  wn.startsWith("qa-verdict")
     records=441 verdicts=397 PASS=109 COND=221 FAIL=67 nonPASS=288 => queue_residual=41 remediate=247
  SHIPPED population wn === "qa-verdict"
     records=436 verdicts=392 PASS=107 COND=219 FAIL=66 nonPASS=285 => queue_residual=41 remediate=244
Every filing figure (441/397/109/221/67/288 and 41/247) reproduces on the population
the masterplan audit_basis names. Confidence: very high.

NAMED FIXES (not mine to apply):
 F1. Replay over the filing's population (startsWith) so criterion 4's 41/247 table is
     printed, or print BOTH with the criterion's population named; and retract the
     "247 DOES NOT reproduce" claim plus its parseable/result:null attribution.
 F2. Add a behavioural cell that EXECUTES the leak guard (extract the `leakedS` span the
     way severityTags is extracted; assert NO-THROW on the sibling shape and THROW on the
     flattened shape and on a judge-emitted colliding key) and a matrix mutant that
     neuters the throw while leaving the scanned literals intact.
 F3. Disclose that residual_close_gate.mjs has no caller in any close path, and that the
     routing scores violated_criteria only (violation_details is never classified).

COMPLETED: 2026-08-21T07:38:11Z
