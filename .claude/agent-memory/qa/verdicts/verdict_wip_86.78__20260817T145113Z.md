STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.78
WRITTEN: 2026-08-17T14:51:13Z

# Q/A write-first record -- step 86.78, cycle 5 (as told by spawn prompt)

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command, git status scope, lint, node --check, checkers
C. LLM judgment vs 6 immutable criteria (read from .claude/masterplan.json first)
D. Mutation testing of the caller-side counter (criterion 6) + guard-vacuity checks

## Log
- 14:51:13Z spawned; qa.md read in full.

## A. Harness compliance (in progress)
- qa_wip 86.78 --spawned-at 2026-08-17T14:51:13Z -> source_present=true, attempt_number=6 (status ok,
  is_lower_bound true), prior_attempts=5, records_retained=6 (gauge).
- verdict_history_86_21.py --step 86.78 --evidence-only -> status=ok, 5 rows:
  CONDITIONAL -> CONDITIONAL -> NO_VERDICT -> CONDITIONAL -> FAIL
- CROSS-CHECK: prior_attempts (5) == ledger rows (5) -> ledger NOT stale by the qa.md rule.
- Work commit 651e1f78 (+ changelog 9aa2f64e). HEAD is now 470c2630; 3 later commits touched
  scripts/qa or qa.md (26d840c9 86.37, 2dbe09d4 86.79 [edits .claude/agents/qa.md], b9be4bdf 86.37).
  verify_escalation_86_78.mjs and qa-verdict.js are UNCHANGED since 651e1f78 (git diff HEAD empty).

## B. Deterministic
- IMMUTABLE: bash -c 'source .venv/bin/activate && node --check .claude/workflows/qa-verdict.js && echo parses'
  -> "parses", exit=0.
- node scripts/qa/verify_escalation_86_78.mjs -> exit 0, "checks run : 53   (cardinality floor 49)", failed 0.
- node scripts/qa/mutation_matrix_86_78.mjs -> exit 0, 13 cells / 13 killed, subject sha
  d245572c66ba0274 unchanged before/after.

### FINDING F-A (evidence quality): the check CARDINALITY does not reproduce.
experiment_results_86.78.md "Cycle 5 GENERATE" says: "verify_escalation 52 checks ALL CHECKS PASS exit 0".
MEASURED NOW: 53. live_check_86.78.md:6 re-runnable recipe still says "# 51 checks". The count is
DETERMINISTIC (46 literal check() call sites + 4 extra from the 5-element GONE loop + 3 from the
4-element verdict loop = 53; no conditional check() anywhere), and both files are byte-identical to
the graded commit, so 52 was never true at this commit. live_check section 12's "verbatim" capture
uses `tail -1`, which prints only "ALL CHECKS PASS" and hides the count -- so the discrepancy is
invisible in the capture that is supposed to evidence it. Direction is harmless (53 > 49 floor,
0 failed); this is a hand-incremented figure (51 -> +1 -> 52 while cycle 5 added TWO checks).

## A2. Harness compliance -- COMPLETE, all 5 clean
1. research-gate-before-contract: research_brief_86.78.md exists, envelope brief_status COMPLETE,
   gate_passed true, external_sources_read_in_full 10 (floor 5), urls_collected 27 (floor 10),
   recency_scan_performed true. CLEAN.
2. contract-before-generate: first-commit dates research 2026-08-14T10:11:21 < contract 11:04:12;
   mtime chain research 09:47 < contract 10:56 < experiment_results/live_check 16:21:45. CLEAN.
3. experiment_results present (428 lines, cycle-5 GENERATE at the tail). CLEAN.
4. log-last: harness_log carries cycles 1/2/2 for 86.78 (2026-08-14) and NO cycle-3/4/5 row;
   masterplan status = pending. The in-flight cycle is correctly absent. CLEAN.
5. no-verdict-shopping: evidence CHANGED since the cycle-4 verdict -- commit 651e1f78
   (2026-08-17T16:21:45+02:00) modified qa-verdict.js, verify_escalation_86_78.mjs,
   experiment_results_86.78.md, live_check_86.78.md; plus masterplan 86.93 audit_basis. Each
   change independently verified below. CLEAN.
HEAD 470c2630 unchanged across my whole evaluation; 651e1f78 is an ancestor; subject sha
d245572c66ba0274 unchanged before and after every mutant.

## B2. Lint / gates
- Step commit 651e1f78 touches ZERO .py, zero backend/**, zero frontend/**, no UI claims ->
  gates 1b/1c/1d N/A by a DERIVED empty set (git show --name-only).
- Working-tree derived .py scope = backend/api/sovereign_api.py (a concurrent peer session's
  work, outside this step): uvx ruff --select F821,F401,F811 -> "All checks passed!", exit 0.
- node --check on qa-verdict.js (the immutable command) exit 0.

## C. Criteria (each verified by EXECUTION, not reading)
C1 exposure re-derived + enumeration command stated -- MET.
  Independent census, my own population rule (1365 transcripts; qa spawn iff first user message
  contains "IMMUTABLE SUCCESS CRITERIA") -> 426 qa spawns.
    - '3rd-CONDITIONAL auto-FAIL rule, and the' (the old STEP-0 leak): 422/426
    - the POST-FIX wording 'the no-auto-PASS clause, the loop-termination rule, and the': 4/426
      (the post-fix spawns, MINE AMONG THEM -- first-party proof the fix is in force)
    - three qa.md-BODY-unique markers I chose myself ('Born inert is SQLite',
      'the 11 observed vacuity shapes', 'Figma MCP output is design-advisory'): 0/426 each.
      -> the cycle-4 falsification of the 'prompt embeds qa.md' cause REPRODUCES on markers the
      author did not pick, and live_check section 10 now states the corrected cause at the site.
  Level-2 positional census reproduced from the inline rule (SLICE + POS regex): 9 hits vs the
  recorded 8 -- the 9th (wf_5038e340-6ca, 'PASS-or-FAIL') is the one the cycle-4 GENERATE
  explicitly discloses as its own. Monotone growth, no contradiction.
  The 'quote the current qa.md text that requires the judge to state its own attempt number'
  clause: no such text exists post-scrub -- all three probes 0 hits -- and qa.md:812-813 now says
  "you are **not** asked to derive your own attempt number or to attest a run length". The
  premise is falsified BY the fix and that is stated, not papered over.
C2 leniency direction + already-collected verdicts -- MET.
  Direction stated (leniency, 58/72 cells, ERRJ 0.000), asymmetry reasoned (PASS is the suspect
  class; FAIL/CONDITIONAL conservative), no blanket re-grade, targeted re-grade queued. The
  landing place now REPRODUCES: masterplan 86.93 audit_basis contains wf_4e01adc8 AND
  wf_20a27baa with the mitigation (the cycle-4 finding that it did not is now closed).
C3 counter OUTSIDE the judge, driven with NO attempt number -- MET, verified LIVE.
  34 workflow run records carry an escalation envelope; 31 have attempt_number=null and still
  compute consecutive_conditionals + would_auto_fail from sequence_supplied alone, caller-side,
  after the verdict returned. Arming OBSERVED true on 6 live runs (wf_aa138724, wf_c4f9b8de,
  wf_a495ce27, wf_1b5406f4, wf_ded4e934, wf_07b25e6e). My own spawn supplies no attempt number.
C4 semantics UNCHANGED + 3rd-CONSECUTIVE still terminates -- MET.
  Checker C4/C4b reproduced. Live: wf_8b8d1bb5 cc=3 -> FAIL (loop terminated); wf_cb5e8948 and
  wf_eba0a6b5 are PASSes at cc=3 with would_auto_fail=false (arming cannot touch a PASS);
  no run shows a mutated verdict.
C5 two safeguards implemented or declined with a reason -- MET.
  burden_on driven ('the party departing from the computed escalation'); override/override_reason
  slots default null, driven. The no-schema-field argument is made and it DOES address the
  double standard: the rubric needed a JUDGE-side field, the override is a CALLER-side act.
  Sourcing gap disclosed, not upgraded.
C6 mutation-test with control GREEN first -- MET.
  Reproduced: control green (53 checks) then 13/13 KILLED, tracked sha unchanged. Plus my own
  13-cell independent battery below, under its own byte-identical null-mutant control (exit 0).

## D. MY INDEPENDENT MUTATION BATTERY (control first, all mutants node --check clean)
CONTROL (byte-identical copy via PYFIN_QA_VERDICT_OVERRIDE) -> exit 0, 53 checks. GREEN.
  MA  STEP-0 leak restored VERBATIM ............................ KILLED (the new 5th GONE probe)
  MB  STEP-0 leak restored REWORDED ............................ *** SURVIVED ***  [WARN-2]
  MC  QX1 flatten via ...escalation ............................ KILLED
  MD  QX2 '// was:' comment decoy + omission ................... KILLED
  ME  QX6 inline /* escalation */ .............................. KILLED
  MF  /* */ block-comment decoy + Object.assign flatten ........ SURVIVED (mitigated: runtime throw)
  MG  Object.assign flatten, no decoy .......................... KILLED
  MH  recorder regex never-matches ............................. survived HERE; KILLED by the
      family checker verify_prompt_render_86_90.mjs section [8] (exit 0, 136 green) -- coverage
      exists where the artifact says it does.
  MI  recorder reverted to hardcoded false ..................... same as MH.
  MJ  no executable 'const merged =' line at all ............... KILLED (both the main check and
      the anti-vacuity check fail TOGETHER -> the anti-vacuity check is fully SUBSUMED)
  MK  decoy present, real merge intact ......................... exit 0 (correct negative control)
  ML  runtime leaked-guard DELETED ............................. KILLED
  MN  /* */ block-comment decoy + escalation OMITTED ........... *** SURVIVED ***  [WARN-1]
  MO  omission alone, no decoy ................................. KILLED (isolates the decoy as the
      load-bearing evasion; the cycle-5 repair itself is genuine)
BEHAVIOURAL drive of the shipped runtime guard (guard text extracted from the file, not copied):
  nested -> NO THROW ; { ...verdict, ...escalation } -> THREW ; Object.assign flatten -> THREW.

### WARN-1 (MN) -- the nesting guard is defeated by a /* */ block comment, and omission is
### the shape the runtime guard CANNOT catch.
verify_escalation_86_78.mjs:173-176 filters "executable" lines by a PREFIX test (trimmed line
must not start with //, * or /*). A block comment whose interior line is UNPREFIXED passes the
filter, and execLines.find() takes the FIRST match. Mutant MN puts
  /*\nconst merged = { ...verdict, escalation, research_routing }\n*/
above a real merge reading `const merged = { ...verdict, research_routing }` -> all 53 checks
PASS while the returned object carries NO escalation at all, and the shipped `leaked` guard
cannot fire because it detects leakage IN, never omission. MO proves the repair is otherwise
genuine. This is a REPEAT of the class the cycle-4 verdict named, in the sentence that also
named the fix that would have closed it: "require the escalation token on a non-comment line
(a prefix test is insufficient for /* */ continuations), or assert Object.keys of the returned
object." The prefix test was shipped. Named fix: strip /* ... */ spans from SRC before locating
the statement, or assert Object.keys(returned) includes 'escalation' on a driven return.

### WARN-2 (MB) -- the new STEP-0 probe is a literal scan, reword-evadable.
The 5th GONE probe pins the exact string '3rd-CONDITIONAL auto-FAIL rule, and the'. MB restores
the same consequence in different words ("the rule that a third straight CONDITIONAL must be
returned as FAIL") -> exit 0, 0 failed. The content-pin remedy the cycle-1 Q/A prescribed and
the author adopted for the withheld-on-purpose block (EXPECTED_LEN + gap emptiness) was not
extended to the STEP-0 region. Partly inherent (no scan enumerates all phrasings) -> WARN.
Named fix: normalised-length/content pin on the STEP-0 enumeration line.

### FINDING F-B (Contradiction) -- "both ride the 86.107 residual queue" does not reproduce.
experiment_results cycle-4 item 4 states F4 (override has no writer) and F5 (recorder
false-negative surface) "both ride the 86.107 residual queue via the transcribed verdict", and
the cycle-4 verdict's notes state F4 is "already queued to 86.107, which exists and is pending".
MEASURED: 86.107 exists and is pending, but its whole blob (2,549 chars) contains 'override' 0,
'recorder' 0, 'false-negative' 0, 'writer' 0, 'judge_was_told' 0; its only 86.78 mention is an
unrelated R3-tail classifyArgs note. Masterplan-wide: 'override_reason' 0, 'has no writer' 0,
'structurally unsettable' 0, 'false-negative surface' 0, 'judge_was_told_consequence' 0,
'override field' 0. This is the SAME class as the cycle-4 finding about 86.93 -- fixed for the
named instance, not swept for the sibling. Re-derived independently: opts.override is never read
by enforceEscalation (override:null is hardcoded in `out`), so the slot IS structurally
unsettable; verdict_ledger.jsonl is now 91 rows, 0 carrying an override.

### NOTE-A (evidence quality) -- cardinality does not reproduce. See FINDING F-A above.
### NOTE-B (evidence quality) -- the anti-vacuity check at :187-189 cannot fail independently of
the main check (proved by MJ: they redden together; no mutant kills it alone). It adds a check
to the count without adding discriminating power. Harmless, but it is not the guard it reads as.
### NOTE-C (evidence quality) -- live_check:493 is grammatically broken by the in-place citation
replacement ("...reproduced in `the inline population rule and regexes in this section (...)`'s
generator"). The content IS sufficient -- I reproduced both census levels from it.

### LIVE INSTANCE OF F5, first-party: my own spawn's caller text scores judge_was_told_consequence
= FALSE against the shipped POSITIONAL_CLAIM_RE (verified by extracting the regex from the file
and driving it; positive controls 'attempt 5 of 5' / 'counted attempt' / 'PASS-or-FAIL' /
'rail binds' all match). Yet my EVIDENCE section reads "returned CONDITIONAL on a bound cycle,
recorded FAIL by the caller-side mechanical rule" -- which delivers the rule's unit AND outcome
in caller-controlled text. Fourth consecutive cycle in which the caller's own next spawn carried
the payload in a form the instrument misses. The artifact DOES state the lower-bound nature, so
this is disclosed, not hidden -- but it is why F5 needs a real queue entry (see F-B).

## E. Self-exposure disclosure
I read qa.md in full (boundary value/unit/shape withheld there), ran verdict_history with
--evidence-only (sequence only: CONDITIONAL -> CONDITIONAL -> NO_VERDICT -> CONDITIONAL -> FAIL,
status ok), and ran qa_wip (attempt_number 6, prior_attempts 5, source_present true). My spawn
prompt's narration additionally delivered the termination rule's outcome shape. Per arXiv
2604.15224 the effect is invisible in chain-of-thought (ERRJ 0.000) so I cannot certify my
grading was unaffected; the measured direction is LENIENCY, which argues AGAINST my capping
findings, not for them.

## VERDICT REASONING
All six immutable criteria are substantively MET by execution I re-derived independently, and
every cycle-4 finding is genuinely closed at its site (I verified each: STEP-0 wording, census
cause, 86.93 queue, QX2/QX6 kills, four SUPERSEDED marks, regenerated section 11/12). Three
findings remain that are NOT evidence-quality-only -- WARN-1, WARN-2 and F-B -- all cheap
(one predicate, one pin, one queue entry). None is a criterion miss and none can change a
verdict. Evidence-quality-only residuals for queueing rather than iteration: NOTE-A, NOTE-B,
NOTE-C. Verdict: CONDITIONAL.

COMPLETED: 2026-08-17T15:03:52Z
