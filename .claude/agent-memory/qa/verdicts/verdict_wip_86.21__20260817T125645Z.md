STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.21
WRITTEN: 2026-08-17T12:56:45Z
COMPLETED: 2026-08-17T13:10:27Z

# Q/A write-first record -- step 86.21, Cycle 7 re-evaluation

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable verification command + git scope + lint + syntax + self-test + mutation matrix
C. LLM judgment against 6 immutable criteria

## Log (append-only)
- [12:56:45Z] Read .claude/agents/qa.md in full. Created this WIP record.

### Prior-attempt evidence (gathered, NOT applied as a trigger)
qa_wip.py 86.21 --spawned-at 2026-08-17T12:56:45Z:
  source_present   : true
  attempt_number   : 3   (INCLUSIVE of me; attempt_number_is_lower_bound: true)
  prior_attempts   : 2
  records_retained : 3   (GAUGE, not a counter)
  records_pruned_known : null
  prior_records: verdict_wip_86.21__20260811T073915Z.md, __20260811T072330Z.md

verdict_history_86_21.py --step 86.21 --evidence-only:
  status   : ok
  detail   : 5 verdict(s) from the ledger
  verdicts : CONDITIONAL -> CONDITIONAL -> FAIL -> CONDITIONAL -> CONDITIONAL
  (aggregates withheld by --evidence-only, as prescribed)

CROSS-CHECK: attempt_number (3) is NOT > ledger rows (5), so the qa.md staleness
rule does not fire. The two quantities differ because WIP records exist only for
spawns that wrote them (write-first landed mid-series) -- attempt_number is
declared a LOWER BOUND by the payload itself. Sequence is KNOWN (status=ok).

## A. HARNESS-COMPLIANCE AUDIT
1. research-gate-before-contract: research_brief_86.21.md present (24,904 B,
   11 aug 09:32). Contract cites run wf_f916b683-d59, 7 sources read in full
   (>= floor 5), 26 URLs (>= floor 10), recency scan, 8 internal files. PASS.
2. contract-before-generate: contract_86.21.md 10 aug 00:01 <
   experiment_results_86.21.md 17 aug 14:48 (local). PASS.
3. experiment_results present: 39,229 B, section 13 = Cycle 7 at the tail. PASS.
4. LOG-is-last: masterplan 86.21 status = "pending", retry_count 0. harness_log
   carries 2 rows for 86.21 (Cycle 199 FAIL, Cycle 1214 CONDITIONAL "step
   PARKED") -- both from CLOSED prior cycles, none for the in-flight cycle 7.
   No premature flip. PASS.
5. no-verdict-shopping: evidence CHANGED since the last graded cycle. Commit
   d33aabe2 (2026-08-17T14:48:18+02:00) modified handoff/current/
   experiment_results_86.21.md ONLY (+49/-12). PASS.

## B. DETERMINISTIC
- IMMUTABLE COMMAND:
  $ bash -c 'grep -c "^## Cycle" handoff/harness_log.md && ls handoff/current/evaluator_critique_*.md | head -3'
  1264
  handoff/current/evaluator_critique_36.12.md
  handoff/current/evaluator_critique_36.13.md
  handoff/current/evaluator_critique_36.17.md
  EXIT=0
  (Note: this command is weak by construction and cannot go red -- disclosed by
  the author since cycle 1. It is the immutable command; I record it as run.)

- SCOPE: cycle-7 commit d33aabe2 touched exactly ONE file
  (handoff/current/experiment_results_86.21.md). ZERO production code changed in
  the cycle-7 diff. Two later commits (140f1ac3 @14:52:45, 1d9a360e @14:56:10
  local) landed before my spawn but belong to steps 86.37/86.78/86.96 and do not
  touch 86.21's scope files.

- CRITERION-1 FENCE, RE-DERIVED BY ME (bash, not zsh):
  $ git show "688ac349:handoff/harness_log.md" | grep -c 'phase=86.20 result='   -> 0
  $ git show "688ac349:.../evaluator_critique_86.20.md" | grep -c '^## Cycle'     -> 1
  $ git show "7145f566:handoff/harness_log.md" | grep -c 'phase=86.20 result='   -> 0
  $ git show "7145f566:.../evaluator_critique_86.20.md" | grep -c '^## Cycle'     -> 2
  masterplan 86.20 status at BOTH commits: pending  (re-derived independently)
  => REPRODUCES EXACTLY. The cycle-7 replacement fence is a verbatim replay of
  live_check_86.21.md section 2 (diffed by eye: identical command text + values).
  TRAP NOTE (mine, not the author's): my first attempt used "$c:handoff/..." in
  zsh; the :h history modifier ate the 'h' and yielded '.andoff/...'. git failed,
  grep read empty stdin and printed 0 -- a silent zero produced by the very shell
  I was using to audit a silent-zero defect. Re-run under bash gives the values
  above. The artifact uses LITERAL hashes, so it is not exposed to this.

- SELF-TEST: python scripts/qa/verdict_history_86_21.py --self-test
  20 cases, all listed, "SELF-TEST PASSED", exit 0. Case list matches the
  artifact's pasted block; I counted 20 case labels.
- MUTATION MATRIX: python scripts/qa/mutation_matrix_86_21.py
  control rc=0; three broken-scoring self-check cells all print "correct"
  (uncompilable->broken, behavioural->killed, behaviour-preserving->survived);
  16/16 KILLED; "[integrity] target md5 unchanged: True"; exit 0.
- md5 verdict_history_86_21.py = b8c0370a54e5fb817d4e19980dd257ed
  (matches the md5 the matrix printed).

- LINT: cycle-7 commit has ZERO .py files (derived: `git diff-tree -r d33aabe2 -- '*.py'`
  is empty), so gate 1a does not bind on the graded change. Ran it anyway over the
  three 86.21 product scripts with the non-empty-set assertion (3 files):
  `uvx ruff check --select F821,F401,F811` -> "All checks passed!" exit 0.
  ast.parse OK on all three. Gates 1b (frontend), 1c (UI), 1d (backend) NOT
  APPLICABLE -- the graded change contains no frontend/**, no backend/**, and the
  step makes no UI claim; I took no Playwright capture and none was required.
- Product scripts are CLEAN vs HEAD (git status empty), md5s unchanged after HEAD
  moved mid-run.

## MY OWN MUTATION PROBE (in-memory; repo never written; md5 verified unchanged)
Control: un-mutated self-test rc=0. 8 mutants; author's score_cell discipline
(broken != killed) reimplemented independently.

KILLED   Q-E blank verdict field silently skipped (rc=1)
KILLED   Q-F prescribed_grep_count loses the result=CONDITIONAL anchor (rc=1)
KILLED   Q-G reset scans forward not backward (rc=1)
SURVIVED Q-A --evidence-only early return deleted
SURVIVED Q-B evidence_only flag ignored in main()
SURVIVED Q-C non-dict JSON row silently skipped instead of counted bad
SURVIVED Q-D `if bad:` gated on seen_step (precedence swap)
DISCARDED Q-H LEDGER repointed -- NO DIFFERENTIAL: my harness relocates the module,
  so REPO_ROOT is wrong for baseline AND mutant alike (both ledger_missing).
  Harness artifact, NOT a finding. (Memory: run_a_null_mutant_through_every_matrix.)

BEHAVIOURAL DIFFERENTIALS MEASURED for the four real survivors:
 Q-A: evidence_only=True -- baseline prints NO "auto-FAIL armed"; mutant prints
      `consecutive     : 2` AND `auto-FAIL armed : True  (a further CONDITIONAL
      would be the 3rd)`. Exit code identical (0/0), self-test green. The
      phase-86.78 bias-control has ZERO automated coverage.
 Q-B: same class via main()'s call site.
 Q-C: ledger with a JSON ARRAY line -- baseline status=unparseable/None;
      mutant status=ok/consecutive=2. Silent under-count, fail-OPEN.
 Q-D: corrupt ledger + a step with no rows -- baseline unparseable/None/rc=1;
      mutant no_rows_for_step/0/rc=0 and PRINTS `consecutive     : 0`. That is
      literally the defect the step exists to abolish, with the suite green.
NOTE: the shipped CODE is correct in all four cases. These are GUARD-COVERAGE
gaps, not product defects. The matrix's own claim is correctly bounded ("every
guard IN THIS MATRIX can fail"), so there is no overclaim to file.

## CLAIM AUDIT (4b) -- what reproduces and what does not
REPRODUCES:
 - section 4 (criterion 3) block: BYTE-IDENTICAL to fresh `--step 36.17` stdout.
 - section 6 self-test block: member-by-member diff vs emitted = IDENTICAL, 20/20.
 - section 10 `grep -cE '^   \('` -> 20: reproduces.
 - section 7 matrix body: 16/16 KILLED, 3 self-check cells "correct", exit 0.
 - section 13 "every evaluator in the 2026-08-17 drain ran both tools": DERIVED
   scope `find handoff/current -name 'evaluator_critique_*.md' -newermt
   '2026-08-17 00:00'` -> 7 files, and 7/7 mention qa_wip AND verdict_history AND
   evidence-only. The named 86.71 cycle-5 example is verbatim present
   ("three agreeing counters ... this was attempt 5 of 5").
 - criterion 2: `verdict_history_86_21.py` writes NOTHING outside
   tempfile.TemporaryDirectory; HARNESS_LOG is READ-only (prescribed_grep_count).
   harness_log.md byte-unchanged during my run. LOG-is-last preserved.

DOES NOT REPRODUCE:
 - F1 sec.6 status TABLE contradicts the shipped code. Says "Four distinct
   outcomes"; the module defines FIVE (ledger_empty absent from the table);
   and `ledger_missing` count is stated as "0 + a caution" while the code
   returns None and _report prints "consecutive : NOT KNOWABLE (refusing to
   print 0)". Measured directly through read_ledger/_report on real files:
     ok->1/False/0 ; no_rows_for_step->0/False/0 ; ledger_missing->None/None/1 ;
     ledger_empty->None/None/1 ; unparseable->None/None/1.
   This is the CYCLE-2 table left in place after the cycle-3 change (source
   lines 92-99 document that exact change). Never flagged before:
   grep -c "Four distinct outcomes" evaluator_critique_86.21.md = 0.
   It sits in criterion 5's own section, and criterion 5 names "missing" by word.
 - F2 sec.7 `md5 : 142f6befbd7fc96689f568cb16b98820` = revision 5b7966e8 (cycle 5,
   2026-08-11). Shipped file is b8c0370a54e5fb817d4e19980dd257ed since commit
   9b4d5281 (phase-86.78, 2026-08-14) added --evidence-only. Body still reproduces.
 - F3 sec.8 immutable-command output `1189`. Live value 1264. True only at
   7897cb8c/070e6714/130a5e9b; stale since 8074e371 (2026-08-11).
 - F4 sec.11 "Every pasted figure reproduces at this tree" -- falsified by F2+F3.
 - F5 sec.8 bullet 3 "No Q/A has been asked to USE the counter. Nothing in qa.md
   points at it" and live_check sec.5 "qa.md still prescribes the grep" are FALSE
   today: qa.md:679 mandates `verdict_history_86_21.py --step <sid>
   --evidence-only`. Sec.13 corrects it but does NOT replace the stale text
   in place (accompany-instead-of-replace). Direction: UNDER-claims.

## SELF-CORRECTION (recorded rather than quietly fixed)
My first heading census used `[Cc]ycle`, which does not match "CYCLE", and I
briefly believed cycles 4 and 5 had no transcribed verdicts. FALSE: the critique
carries all five (l.3/93/187 as JSON, l.358+365 and l.507+513 as
`## VERDICT: CONDITIONAL` blocks). Transcription compliance is CLEAN. Also: my
first git-show fence used "$c:handoff/..." under zsh, where the :h modifier
yielded ".andoff/..."; git failed, grep read empty stdin and printed 0 -- a
silent zero produced inside an audit of silent zeros. Re-run under bash.

## CYCLE-5 FIX LIST -- all five verified LANDED by me
(1) third verify_broken_scoring cell asserting "survived" -> present, prints
    "correct"; the always-KILLED defect makes the matrix return rc=5.
(2) sec.2 fence replaced with the git replay -> present, and I re-derived all
    four values plus `pending` at both commits.
(3) sec.6 block 15->20 and sec.10 18->20 -> both reproduce exactly.
(4) cycle-4/5 record + seven print-layer mutants disclosed -> sec.11 lines 474-479.
(5) STALE fifth failure mode -> sec.11 + the sec.8 pointer added in cycle 7.

## HEAD MOTION DURING MY RUN
fff6d8c4 -> aa367cbf (77f15b4d, phase-86.72/86.78) landed mid-evaluation. It
touches NO 86.21 file (verified by name filter) and the 86.21 md5s + artifacts
are byte-unchanged since d33aabe2. It did add 17 lines to .claude/agents/qa.md
(optional research_needed / research_brief_spec) ~5 min after my runtime read;
those fields are not applicable here -- 86.21's residuals are closeable by
editing, not by more research.

## VERDICT REACHED: CONDITIONAL
Criteria 1, 2, 3, 4, 6 MET (each independently re-derived). Criterion 5 MET IN
SUBSTANCE (behaviour fails CLOSED on missing/empty/unparseable and is tested by
(iii)/(iii-b)/(iii-c)/(iv)/(vii) plus matrix cells M1/M5/M11/A2 -- I verified all
five statuses myself) but DEFECTIVE AS ASSERTED: the criterion's own evidence
table misstates the missing-ledger case and omits the empty-ledger case. Not a
FAIL: no criterion is materially unaddressed, the product is correct/live/
load-bearing, and every non-reproducing figure UNDER-claims.

FINAL INTEGRITY (13:10:27Z): HEAD moved twice more during my run
(fff6d8c4 -> aa367cbf -> 974de7d4); neither 77f15b4d nor 33c47416 touches any
86.21 file. Product md5s unchanged (b8c0370a / db411673 / bd11b19f), all 86.21
artifacts and scripts/qa clean vs HEAD, harness_log.md byte-unchanged, masterplan
86.21 still `pending` retry_count=0. I wrote ONLY inside
.claude/agent-memory/qa/ (this record + two feedback memories + the MEMORY.md
pointers). One write was BLOCKED by qa-write-guard.sh -- an attempt to put my
mutation harness in the session scratchpad; I re-ran the probe as an in-memory
heredoc instead, so nothing was lost and no guard was worked around.

