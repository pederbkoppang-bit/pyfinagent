STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.78
WRITTEN: 2026-08-17T13:59:15Z

# Q/A cycle-4 re-evaluation of step 86.78 (write-first crash-survival record)

Read qa.md in full at 13:59:15Z. HEAD at spawn = 92d5253a (auto-changelog for
54eace07, the cycle-4 work commit, 2026-08-17 15:58:49+0200). RE-CHECKED at
14:10:21Z: HEAD moved to d3fa720c (a PEER session closed 86.75 and committed
paper-trading work); 54eace07 is still an ancestor; `git diff --name-only
92d5253a..HEAD` touches ZERO 86.78 files; subject sha256[:16] still
62e682b9f6aad93d. My grade stands on the state I measured.

## Attempt / sequence EVIDENCE (gathered, not applied)

qa_wip.py 86.78 --spawned-at 2026-08-17T13:59:15Z:
  source_present=true, attempt_number=5 (status ok, is_lower_bound=true),
  prior_attempts=4, records_retained=5 (gauge), records_pruned_known=null.
verdict_history_86_21.py --step 86.78 --evidence-only:
  status=ok, "4 verdict(s)", CONDITIONAL -> CONDITIONAL -> NO_VERDICT -> CONDITIONAL.
CROSS-CHECK: prior_attempts(4) == ledger rows(4) -> IN SYNC; the cycle-3
staleness (4 vs 0) is closed by the backfill, and the 4 backfilled rows are
each labelled BACKFILL/reconstruction with sources named. harness_log carries
only 3 rows for 86.78 (cycles 1/2/2) -- secondary source, disagrees, ledger
governs. Observation on the qa.md rule's literal wording: it compares
attempt_number (INCLUSIVE of the in-flight spawn) to the ledger count, which
would read STALE on every healthy step; prior_attempts is the correct operand.

## A. Deterministic

- IMMUTABLE COMMAND -> stdout "parses", EXIT 0.
- verify_escalation_86_78.mjs -> EXIT 0, 51 checks (floor 49), 0 failed.
  (cycle-3 measured exit 1 / 1 failed -- the blocker IS repaired.)
- mutation_matrix_86_78.mjs -> EXIT 0, CONTROL GREEN FIRST (51 checks), then
  13 cells / 13 KILLED / 0 survived, tracked sha unchanged before==after.
- live_check :6-7 recipe cardinalities (51 / 13) REPRODUCE.
- node --check green on qa-verdict.js + both .mjs.
- ruff F821,F401,F811 on the DERIVED scope (`git diff --name-only HEAD --
  '*.py'` = 5 files, all a PEER session's paper-trading work; this step's own
  commit has ZERO .py) -> "All checks passed!", exit 0.
- Gates 1b/1c/1d N/A: the step's commit is 7 files, zero frontend/**, zero
  backend/**, no UI claims.

## B. My own adversarial battery on the REPAIRED assertion (6 cells, same
   PYFIN_QA_VERDICT_OVERRIDE seam, control GREEN first, tracked sha unchanged)

  QX1-DROP-ESCALATION-FROM-MERGE    RED, killed by 'escalation is NESTED in the
                                    return, not spread into it'
  QX2-COMMENT-TOKEN-TRAP            SURVIVED  (FINDING E)
  QX3-OBJECT-ASSIGN-FLATTEN         SURVIVED  (NOT a finding -- runtime guard)
  QX4-NEUTER-RUNTIME-LEAK-GUARD     RED, killed by '...ALSO throws at runtime'
  QX5-NEST-EMPTY-OBJECT             SURVIVED  (NOTE)
  QX6-INLINE-COMMENT-INSIDE-BRACES  SURVIVED  (FINDING E, same class)

QX1 proves the repaired property assertion is NOT vacuous. QX2/QX6 defeat it
with a comment: `// was: const merged = { ...verdict, escalation, ... }` above
a line that omits escalation, or `{ ...verdict, /* escalation */ ... }`, both
pass all 51 checks while the returned object carries NO escalation and the
runtime `leaked` guard stays silent (it catches leakage IN, never omission).
QX3 is a survivor that is NOT a finding: Object.assign evades the source regex
but qa-verdict.js:839-841 computes `leaked` after it and THROWS, so the
flattened object cannot ship; QX4 proves that guard's presence is asserted.

## C. FINDING C (material) -- the rail's own prompt still names the rule's
   ordinal, unit and outcome, and the census MISATTRIBUTES it

qa-verdict.js:345 (STEP 0 of the prompt this step OWNS):
  'runtime-smoke gates, the output schema, the no-auto-PASS clause, the
   3rd-CONDITIONAL auto-FAIL rule, and the'

MEASURED (population: every agent-*.jsonl under
~/.claude/projects/-Users-ford--openclaw-workspace-pyfinagent/*/subagents/
workflows/wf_*/; qa-verdict spawn iff first non-isMeta user message contains
"IMMUTABLE SUCCESS CRITERIA"):
  qa-verdict spawns                                  : 420 (421 a minute later
                                                        = +1, MY OWN spawn)
  contain "3rd-CONDITIONAL"/"auto-FAIL"              : 420/420
  contain the RAIL's STEP-0 line VERBATIM            : 420/420
  contain qa.md BODY text (3 qa.md-unique markers:
   'Vera-Perez arXiv:1807.05030', 'SEVerA (arXiv:2603.25111',
   'pseudo-tested method')                           :   0/421

live_check:453-457 says the 413/413 hit occurs "because the prompt embeds
qa.md itself" and dismisses it as design, not exposure. FALSIFIED: qa.md is
not embedded (0/421; prompt ~11.1 KB vs qa.md 51,552 B). The hit is produced
420/420 by one line of the RAIL's own prompt -- the half F1 reports as cleaned
("the prompt no longer states any consequence (4 probes, all 0)"). It states
the value (3rd), the unit (CONDITIONAL) and the outcome (auto-FAIL) 60 lines
above the block declaring the consequence DELIBERATELY NOT STATED HERE, and
qa.md:753-756 -- written by this step -- sets the standard it fails ("the
boundary's value, unit and shape are all withheld"). The C1 literal probes and
cells M9/M12 are blind to it (M12 pins only that nothing sits BETWEEN the
criteria sentence and the withheld block; :345 is far above).

## D. FINDING D (material) -- a "queued" claim that does not reproduce

experiment_results (cycle-3 item 5) and live_check §10 both state the two
consequence-framed PASSes "are QUEUED FOR A RE-GRADE DECISION in step 86.93".
Measured: masterplan step 86.93 enumerates 85.5 / 86.25 / 86.34 (+86.86 done)
and its population is the `[object Object]` RENDER-LOSS class. In 86.93's full
blob: '86.71' 0, '86.68' 0, 'wf_20a27baa' 0, 'wf_4e01adc8' 0, 'leniency' 0,
'consequence' 0, '2604' 0. Across the WHOLE masterplan: wf_20a27baa 0,
wf_4e01adc8 0; the only steps mentioning leniency/2604.15224 are 86.78 itself
and 86.98 (a different question). So the affirmative half of criterion 2's
answer has no landing place.

## E. FINDING F (evidence-quality only) -- superseded blocks unmarked

experiment_results applies "(SUPERSEDED by §8)" to §4; live_check applies no
such marker to the same facts, so these read as current and are false today:
  :10   "`.claude/agents/qa.md` is **not modified by this step** — §5"
        (qa.md WAS edited under this step's tag: commits 9b4d5281, 77f15b4d)
  §1a   headed "The current `qa.md` text", quoting :665 / :685 / :702 -- all
        three removed by the scrub; §10 supplies the real current quote (:808,
        verified accurate)
  §5    "$ git diff --stat -- .claude/agents/qa.md  (no output — zero-line diff)"
  §6    "The `qa.md` half is NOT fixed"
Also: `scratchpad/census_8678_v2.txt`, cited in §10 as where the generator
lives, DOES NOT EXIST (the inline population rule + regexes were sufficient --
I reproduced the census from them); and §11's shell block shows three commands
and only two outputs (the `grep -c` runs BEFORE the file it reads is written;
its `1` is absent from the paste). The underlying facts all reproduce.

## F. Criterion-by-criterion (graded against the MASTERPLAN verbatim; the
   prompt's criterion 5 is a paraphrase -- "...is load-bearing" vs
   "...was grounds to delete the rubric")

1. MET. Current qa.md text quoted (:808, verified); enumeration commands
   stated inline and REPRODUCED by me: 9 positional hits over 421 spawns vs
   the author's 8/413, the 9th being wf_5038e340-6ca (the cycle-3 evaluation,
   already owned), members matching exactly. Defects inside it: FINDING C
   (falsified causal attribution) and FINDING F (stale §1a).
2. MET on the reasoning: leniency direction stated with the source, asymmetry
   argued (PASS suspect, FAIL/CONDITIONAL conservative), blanket invalidation
   declined with law-of-the-case + reliability-without-validity + PMC5557596.
   FINDING D falsifies the disposition it prescribes.
3. MET, driven and re-derived first-party from run records: across the last 34
   envelopes ~29 have attempt_number=null and compute consecutive_conditionals
   from sequence_supplied alone; arming OBSERVED (wf_aa138724-102 cc=2
   arm=true; wf_c4f9b8de-a33 cc=2 arm=true; wf_a495ce27-1af cc=2 arm=true) and
   fail-closed observed (seq absent -> cc null, arm null, status not_supplied /
   unusable, never 0/false). My own prompt supplied no attempt number and no
   sequence; qa.md now says the judge is "not asked to derive your own attempt
   number" (:805-806).
4. MET. verdict_unmodified true on 34/34 sampled envelopes and COMPUTED, not
   hardcoded (M13). Direct drive: input verdict object byte-unchanged after
   the call; arming only on CONDITIONAL (PASS and FAIL both arm=false), so no
   caller-side downgrade path. Loop TERMINATED live: wf_aa138724-102 (cc=2,
   arm=true) -> wf_8b8d1bb5-245 (cc=3) returned FAIL.
5. MET. burden_on implemented ("the party departing from the computed
   escalation"); override/override_reason slots default null and are
   mutation-tested (M8); the different-party argument (judge cannot record it
   -- additionalProperties:false; caller does) is the genuine disanalogy to the
   deleted rubric; the legal-sourcing gap is stated, not upgraded. RESIDUAL,
   re-derived by me and correctly queued to 86.107: the slot has NO WRITER --
   {override:'operator accepted'} and {override:true,override_reason:'because'}
   both yield null; the function reads only max_attempts/caller_text/
   attempt_number.
6. MET. Control GREEN observed FIRST (51 checks) then 13/13 killed with named
   killing assertions, exit 0, tracked file unchanged -- reproduced by me, plus
   my own 6-cell battery under its own green control.

## G. Harness compliance -- CLEAN

research_brief_86.78.md exists, envelope brief_status COMPLETE, gate_passed
true, 10 sources read in full (floor 5), 27 URLs (floor 10), dedicated recency
scan section. mtime chain research 08-14 09:47 < contract 08-14 10:56 <
artifacts 08-17 15:58. experiment_results present. LOG-LAST respected: 86.78
status still `pending`, harness_log has no cycle-3/4 row. NO VERDICT-SHOPPING:
evidence CHANGED between spawns (54eace07 rewrote both instruments, both
artifacts and the ledger). No unintended production change from this step.
NO WRITE WAS BLOCKED.

## H. Self-exposure disclosure

I read qa-verdict.js:345 in my own prompt AND ran the ledger tool, so the
withheld payload was reconstructible for me from two permitted sources.
ERRJ=0.000 (arXiv 2604.15224) means I cannot certify my grading was
unaffected; the measured direction is LENIENCY, which argues AGAINST the
capping findings above, not for them. Separately, my prompt's EVIDENCE section
presents the author's own conclusion ("graded all six criteria substantively
MET"); I re-derived every criterion independently rather than adopting it.
My prompt carries ZERO hits of the shipped POSITIONAL_CLAIM_RE (verified on
my own transcript wf_1b5406f4-2e1), so the cycle-4 replacement claim about
forward discipline is TRUE and first-party verified.

COMPLETED: 2026-08-17T14:10:21Z
