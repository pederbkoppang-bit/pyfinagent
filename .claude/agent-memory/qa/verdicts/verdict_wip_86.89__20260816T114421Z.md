STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.89
WRITTEN: 2026-08-16T11:44:21Z
COMPLETED: 2026-08-16T11:57:40Z

# Q/A write-first record -- step 86.89, cycle 2

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command `python scripts/qa/verify_matrix_coverage_86_85.py`;
   git scope; lint; scoped tests; re-runnables
C. LLM judgment vs 8 immutable criteria + Main's 4 judge_these questions

## Findings log (appended as established)
- (start) read .claude/agents/qa.md in full. Confirmed write-first, deterministic-first,
  4b claim-auditing, 4c guard-vacuity, prior-attempt-evidence rules.

### Prior-attempt / sequence evidence
- `qa_wip.py 86.89 --spawned-at 2026-08-16T11:44:21Z` -> attempt_number=2,
  prior_attempts=1, attempt_number_status=ok, attempt_number_is_lower_bound=false,
  source_present=true, records_retained=2 (gauge, incl. own), prior_records=[
  verdict_wip_86.89__20260816T112622Z.md].
- `verdict_history_86_21.py --step 86.89 --evidence-only` -> status=no_rows_for_step,
  verdicts=(none).
- CROSS-CHECK: attempt_number (2) > ledger verdict count (0) -> **THE LEDGER IS STALE**
  for this step. Sequence from the ledger is UNRELIABLE. Separate observation from the
  artifacts (not from the ledger): evaluator_critique_86.89.md carries one transcribed
  verdict, "Cycle 1 verdict: CONDITIONAL", run wf_940c06f4-37c.

### A. Harness compliance
1. research_brief_86.89.md 13:18 local, gate_passed evidence cited in contract Sec 1
   (run wf_abfa4db8-f13, 22 sources, 45 URLs, audit-class dry). Research < contract
   (13:22) < experiment_results (13:24) < live_check (13:25). OK.
2. contract before generate: OK.
3. experiment_results present: OK -- but see F-1, NOT UPDATED for cycle 2.
4. log-last: `grep -cF "phase=86.89" handoff/harness_log.md` -> 0; masterplan
   status="pending". OK.
5. no verdict-shopping: evidence DID change (commit 1864dba7 touches 2 scripts).
   NOT a verdict-shop. But the change is code-only; see F-1.

### B. Deterministic
- IMMUTABLE CMD exit **0**. "RESULT: OK -- every enumerated guard is touched by at
  least one cell." guards 15 covered 15 uncovered 0 cell problems 0.
- `verify_cell_vacuity_86_89.py` control: exit 0, ALL GREEN 8 passed 0 failed,
  8 assertions (floor 8), matrix md5 a9b61434... unchanged.
- `--self-test`: SELF-TEST OK, case A REJECTED, case B REJECTED, matrix restored.
- `mutation_matrix_86_85.py`: rc=0, 14/14 KILLED, 2.5 s, both halves run.

### F-1 [BLOCK-class] The cycle-2 remediation did NOT touch the two artifacts it
names. `git log -1 --format=%h -- <file>`:
  live_check_86.89.md        -> b0edad8e (cycle 1)
  experiment_results_86.89.md-> b0edad8e (cycle 1)
  evaluator_critique_86.89.md-> 1864dba7 (cycle 2)
  verify_cell_vacuity_86_89.py / mutation_matrix_86_85.py -> 1864dba7
Consequences, all reproduced by grep:
  (a) The C6 sentence cycle 1 FAILED is STILL LIVE, verbatim, in BOTH files the
      cycle-1 verdict named: live_check_86.89.md:93 "Licenses ONE claim: every cell
      in this matrix demands at least one enumerated guard." and
      experiment_results_86.89.md:132 "every cell in the matrix demands at least
      one enumerated guard." Only the SCRIPT was fixed.
  (b) live_check_86.89.md Sec 2/4 still show the OLD 7-assertion run
      ("ok [floor] 7 assertions ran (floor 7)", "ALL GREEN: 7 passed"), the OLD
      "[5] the baseline has not rotted" wording, and NO [6] -- the shipped checker
      emits 8 assertions with a renamed [5] and a new [6].
  (c) The masterplan verification.live_check requires "live_check_86.89.md carrying
      ... the per-member RED demonstration". grep "6 of 6" over handoff/current/*86.89*
      returns hits ONLY in evaluator_critique_86.89.md -- i.e. the demonstration
      exists only as the EVALUATOR's own execution transcribed into the critique.
      Criterion 4 still has no covering evidence in an author artifact.
  (d) experiment_results Sec 1 states "The existing verify_matrix_coverage_86_85.py
      and mutation_matrix_86_85.py are **unmodified** -- deliberately" -- now FALSE:
      1864dba7 modifies mutation_matrix_86_85.py.

### F-2 [judge_these B] FINGERPRINT [6] IS DEFEATED BY A DESCRIPTION-PRESERVING
REPURPOSE -- MEASURED, mutant SURVIVED.
  Mutation: M6 keeps its description ("REVERSE emit_sequence ...") and its payload
  '"\n    return out\n" -> "\n    return out[::-1]\n"' is replaced with
  '"EXIT_OK = 0" -> "EXIT_OK = 0  # repurposed-benign"'.
  Vacuity checker (SHIPPED, unmutated, exec'd in-process with __file__ pinned):
  rc=0 SURVIVED, "ALL GREEN: 8 passed, 0 failed", including "ok [6] each baselined
  cell still contains the mutation it was baselined FOR". Matrix restored
  byte-identical.
  ROOT CAUSE: payloads[cid] = the WHOLE cell tuple text (lines[span[0]:span[1]]),
  which INCLUDES the description line the fingerprint was copied from. So [6]
  asserts that the description still contains words copied out of the description --
  it does not reference the find/replace payload at all.
  The source comment at verify_cell_vacuity_86_89.py:66-67 claims "Each entry is now
  pinned to a fingerprint of its mutation payload, so repurposing a baselined cell
  fails [6]." That claim is FALSE as written.
  CREDIT MIS-ATTRIBUTION: this particular mutant IS caught -- by the matrix's own
  survivor detection (matrix rc=1, "SURVIVORS (each is a guard that cannot fail):
  M6"), a different mechanism from [6].

### F-3 [judge_these D, wiring mis-diagnosis] A VACUITY failure is reported as a
COVERAGE failure. mutation_matrix_86_85.py:258-262 does `cov_rc = cov_rc or vac_rc`
then prints, under `if cov_rc != 0`, "FATAL: the mutation matrix is INCOMPLETE over
the writer's guards". MEASURED under QA-M1b (M6 payload duplicated to M7's): the
COVERAGE gate printed "RESULT: OK", vacuity [4] fired on M7, and the run still
printed the coverage FATAL. Same class as the [5] misdiagnosis cycle 2 just fixed.
Also measured there: the licence line labels ALL vacuous cells "baselined" --
it printed "6 baselined cell(s) demand NOTHING: [... 'M7' ...]" while M7 is NOT
baselined.

### F-2b [BLOCKING] FULL COMPOSITE DEFEAT -- QA-M2 survives BOTH halves.
Mutation: M6 keeps its description; its payload is replaced with a DUPLICATE of
M5's behavioural payload ('"date": event_date or stamp.date().isoformat()' ->
'"date": stamp.isoformat()').
  vacuity checker : rc=0, ALL GREEN 8 passed 0 failed, incl. "ok [6]".
                    VACUOUS still 5 ['M5','M6','M9','M11','M12'].
  mutation matrix : rc=0, M5 KILLED and M6 KILLED, coverage "RESULT: OK",
                    "ALL GREEN: 8 passed, 0 failed".
  matrix + checker restored byte-identical.
DIFFERENTIAL: after this repurpose NO cell anywhere mutates emit_sequence
ordering -- the exact 86.85 QA-M1 / palindromic-fixture defect that opened this
whole series -- and the entire composite gate is fully green. This is a real
surviving mutant on the object the step exists to protect.
FIX: fingerprint the find/replace payload ONLY (cell tuple elements 3 and 4),
never the whole cell text.

### F-3 [judge_these A] --self-test covers 2 of 8 assertions AND is invoked by
NOTHING. Self-test mutation matrix, CONTROL observed GREEN first (shipped
checker --self-test rc=0):
  neuter [4] (not new_vacuous -> True)     rc=1  KILLED    (case B bites)
  neuter [5] (not fixed -> True)           rc=1  KILLED    (case A bites)
  neuter [6] (not drifted -> True)         rc=0  SURVIVED
  neuter [3] (before == after -> True)     rc=0  SURVIVED
  neuter [1] (bool(demanding) -> True)     rc=0  SURVIVED
  neuter [2] (not unscorable -> True)      rc=0  SURVIVED
  neuter floor (if emitted < FLOOR->False) rc=0  SURVIVED
  neuter the self-test's own scoring
        (failures += rc == 0 -> += 0)      rc=0  SURVIVED
  files_ok=True on every row (checker never written to disk; exec'd in-process
  with __file__ pinned; matrix restored in a finally).
INVOCATION: grep over the repo -> `--self-test` / `self_test` for this checker
appears ONLY at verify_cell_vacuity_86_89.py:271 (its own docstring), :311-312
(its own __main__ dispatch), and :258 of the matrix which calls `vac.main()`.
Nothing runs the self-test.
WIRED DEMONSTRATION (mutation_matrix main() exec'd in-process with a mutated
vacuity module injected into sys.modules; control first):
  pristine                                  rc=0
  red state (M6 -> duplicate of M7)         rc=1, "FAIL [4] ... ['M7']"
  same red state, [4] neutered to True      rc=0, ALL GREEN 8 passed 0 failed
So on the ONLY automated path a neutered [4] ships fully green, and the guard
built to catch that neutering is never invoked. NOTE [1] is the assertion
experiment_results Sec 7 credits as the criterion-5 over-crediting test.

### F-5 [WARN] The repo-write claim is still false, and now sits on the WIRED
path. verify_cell_vacuity_86_89.py:33-34 still reads "Read-only on the repo: the
matrix is mutated IN MEMORY via a temp copy" -- a cycle-1 finding, unfixed.
MEASURED by patching Path.write_text for one `mutation_matrix_86_85.main()` run:
**15 writes** to the repo file scripts/qa/mutation_matrix_86_85.py, at truncated
sizes 11694/11793/11892/11979/11985/11992... vs pristine 12367, plus the restore.
mutation_matrix_86_85.py's own docstring says "ZERO REPO WRITES ... This avoids
the restore step entirely, which is the only way to be sure a restore was not
gotten wrong" -- the cycle-2 wiring reintroduces exactly that restore step, on
the matrix's OWN source file, in a repo whose auto-commit hook runs `git add -A`.
Higher risk than at cycle 1 because it is now on the path people are told to run.

### F-6 [NOTE -- credit to the design, retiring Main's stated bound]
judge_these C. I built criterion 5's NAMED shape and the shipped mechanism CATCHES
it. `verify_matrix_coverage_86_85.py::spans_with_ancestors`, `if isinstance(cur,
ast.If):` -> `if isinstance(cur, (ast.If, ast.Try)):` (the exact historical
defect). CONTROL green first; mutant -> demanding 9->7, VACUOUS 5->7 adding
['M13','M14'], "FAIL [4] ... 2 cell(s) demand nothing and are not in the
baseline: ['M13','M14']", rc=1 RED. Gate + matrix restored byte-identical.
So the property HOLDS; what is missing is the author's demonstration, not the
capability. Same pattern as C4.

## Criteria mapping
1 MET      -- immutable cmd exit 0 re-run by me; recall reproduced with the M4
              positive control; probe-lied-twice disclosure is exemplary.
2 MET      -- 4 of 4 classified on members 1-4, 0 on member 5, labelled
              Recall_SD; I re-derived VACUOUS={M5,M6,M9,M11,M12} live.
3 NOT MET  -- the declaration IS present (KNOWN_VACUOUS + fingerprints); the
              set-level half bites, the CONTENT-level half does not. F-2/F-2b.
4 NOT MET  -- no per-member RED demonstration in ANY author artifact; the
              masterplan live_check names live_check_86.89.md and it has none.
              The only such demonstration is the EVALUATOR's, transcribed into
              the critique -- the author leaning on the judge's evidence.
5 MET      -- with a bound; F-6 shows the named shape is caught, though by my
              execution rather than the author's, and [1] is neuterable.
6 NOT MET  -- script fixed and verified live, but the contradicted sentence is
              STILL VERBATIM in live_check_86.89.md:93 and
              experiment_results_86.89.md:132, the two files cycle 1 named.
7 MET      -- no verdict surface touched; diff is 2 scripts + 1 handoff file.
8 NOT MET  -- the two NEW cycle-2 guards ([6] and self_test()) are not
              mutation-tested in any author artifact (live_check Sec 4 is the
              stale cycle-1 5-cell matrix at 7 assertions), and on execution [6]
              fails its own stated purpose while self_test() is un-invoked.

## Deterministic summary
immutable cmd exit 0 | ruff F821,F401,F811 on a git-DERIVED 3-file scope (non-empty
asserted, xargs) "All checks passed!" exit 0 | ast.parse OK x4 | pytest
backend/tests/test_phase_86_85_verdict_ledger_write.py -> 27 passed | matrix rc=0
14/14 KILLED | vacuity rc=0 8/8 | self-test rc=0.
No frontend/** and no backend/** in this step's diff -> gates 1b/1c/1d N/A. The
uncommitted backend/api/sovereign_api.py is mtime 2026-08-14 13:28, two days older;
NOT attributable to 86.89.
MID-EVAL TREE CHANGE: HEAD moved 65510727 -> a96bb28a during this evaluation
(ce2785d7 "docs(session): day report" + its changelog hook). Files: CHANGELOG.md,
day_report_2026-08-16.md, goal_next_2026-08-17.md. NOTHING in 86.89's scope; the
day report honestly records 86.89 as "in evaluation at session end".
FINAL INTEGRITY 2026-08-16T11:56:24Z: mutation_matrix a9b61434..., vacuity
1327efb5..., coverage 4e77ba1c... all byte-identical to their pre-mutation md5s;
`git status --short scripts/qa/` empty. No repo file was edited by me.
BLOCKED WRITE: qa-write-guard denied a scratchpad harness file; I ran every
mutation through `python3 -` heredocs instead. No workaround of the guard.
