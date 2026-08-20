STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 90.1
WRITTEN: 2026-08-20T19:50:39Z

# Q/A cycle-2 write-first record for step 90.1

Spawn: Workflow rail. Cycle 2. HEAD = 1fc7b2e6 (cycle-1 was 3bf0b0fe).

## A. HARNESS COMPLIANCE -- CLEAN on all 5

1. Research gate before contract: research_brief_90.1.md brief_status COMPLETE,
   gate_passed true, external_sources_read_in_full 10 (>=5), urls_collected 25 (>=10),
   recency_scan_performed true; enforced run wf_db313c3d-b75 cited in the contract.
2. Contract before generate (mtime, local CEST): research_brief 21:12:47 <
   contract 21:15:34 < attempt_outcomes.py 21:45:58 < attempt_gate.py 21:47:23 <
   mutation_matrix_90_1.py 21:47:53 < experiment_results 21:49:19 < live_check 21:49:42.
3. experiment_results_90.1.md present, 436 lines, with a `# CYCLE 2` remediation section.
4. LOG last: `grep -Fc 'phase=90.1' handoff/harness_log.md` = 0, with the grep proven
   LIVE by `phase=86.116` returning 2. masterplan 90.1 status=pending (not flipped).
5. No verdict-shopping: evidence CHANGED. Commit 1fc7b2e6 touched attempt_gate.py +19,
   attempt_outcomes.py +147, mutation_matrix_90_1.py +191, mutation_matrix_86_71.py +12,
   experiment_results +160, evaluator_critique +99, live_check +99.
   All 6 immutable criteria appear VERBATIM in contract_90.1.md (checked by string
   containment against masterplan.json, not by eye).

## B. DETERMINISTIC

IMMUTABLE COMMAND `python3 scripts/harness/attempt_gate.py --self-test && python3 scripts/qa/mutation_matrix_90_1.py --verify`
  IMMUTABLE_CMD_EXIT=0 (run twice, both 0).
  self-test: 34 `ok` lines + SELF-TEST PASSED -- matches the claimed "34 checks".
  (CORRECTION TO MYSELF: an earlier line in this record said 36; I counted by eye.
  `grep -c '^  ok '` = 34.)
  matrix: CONTROL GREEN then 15 cells / 25 CHECKS ->
  KILLED 14 | SURVIVED 0 (excl. N0) | ERROR 0 | null mutant survived True | tree md5 same.
  Matches "grew from 11 cells to 15 and from 21 checks to 25".
LINT: derived scope via git (`1c2f25b3..HEAD` + working tree + untracked), 9 files, piped
  through xargs, non-empty asserted -> `uvx ruff check --select F821,F401,F811` exit 0.
FRONTEND (1b) N/A -- no frontend/** in the diff. UI GATE (1c) N/A -- no UI claims.
RUNTIME: every changed module executed for real (hook drives, CLI drives, self-test,
  matrix, my own probes). Not backend/** so 1d's import smoke is satisfied by execution.
LEDGER SHAs unchanged across ALL my runs:
  attempt_budget_audit 1f2a39cf... | verdict_ledger fcfe56ad...2e3eb2
SCOPE: cycle-2 commit touches only the 3 step files + the consumer matrix + handoff +
  agent memory + audit streams. No unintended production code. (Unrelated backend/* files
  were swept into the CYCLE-1 commit by the hook's `git add -A`; pre-existing behaviour.)
EVIDENCE FROZEN under me: `git status --short` on all five 90.1 artifacts and all four
  scripts = clean vs HEAD at the end of the evaluation; live_check C2.5's md5s
  (85de2e74.../81ebe68b...) reproduce byte-for-byte.
  NOTE: Main launched a 90.9 research gate at 19:50:56Z DURING my evaluation (visible in
  the audit stream and as `?? contract_90.2.md`). No 90.1 evidence file moved.

## C. CRITERIA

C1 MET. --backfill --dry-run on the live ledger exit 0, ledger sha unchanged. On a COPY,
  3 consecutive REAL runs -> rc 0/0/0, file byte-identical after run 1 (97e0120e...) so
  IDEMPOTENT, counts identical across all three. UNKNOWN=7 with reason no_run_record=7
  -> "UNKNOWN only where no run record exists", count stated; ambiguous_match=0.
  At the committed evidence point (HEAD) there are EXACTLY 92 attempt rows, all carrying
  outcome + total_tokens (1c2f25b3=83/79, 3bf0b0fe=93/89, HEAD=96/92, tree now 98/94).
  The drift is the live gate; my own spawn is row 19:50:34Z attempt_number_inclusive=3.
  FREEZE PROBE (mine): a settled row with a DELIBERATELY WRONG outcome (FAIL/wf_WRONG)
  next to a matching PASS record stayed byte-identical; an UNKNOWN row next to the same
  record WAS re-resolved to PASS/123456. UNKNOWN-writable cannot overwrite a real outcome.

C2 MET, driven independently (not read). Non-exhaustion denial (claim 86.118.1) with a
  seeded escalation_attempt_budget_86.85.md: rc=2, wrote
  escalation_unknown_step_id_86.118.1.md, seeded file sha256 75f727f7... == 75f727f7...
  BYTE-IDENTICAL, zero ledger rows appended. Exhaustion denial for a real id still writes
  escalation_attempt_budget_<sid>.md, so the four files on disk keep their names.

C3 MET, decided by running. 1,200,001 tokens on ONE attempt -> rc 2 with 4 of 5 attempts
  unused; 1,199,999 -> rc 0. Discriminates in both directions.

C4 MET. Cells on the REAL module: 86.118 ADMITTED; 86.118.1 / 86.1180 / 999.99 DENIED --
  in the object-args, JSON-string-args and malformed-salvage forms alike.
  RECALL re-derived: real plan members=1427 missing=0, ids=1614 (Main's numbers exactly).
  Shallow-walk mutant vs the REAL plan: missing_total=123 -- the "123 missed" reproduces.
  BLAST RADIUS re-derived with the FIXED walk over 621 real run records: 81 no step_id,
  540 with, 535 ADMITTED, 5 DENIED -- and all 5 were already shape-refused pre-90.1.
  ZERO new denials across the entire launch history.
  MY OWN EXTRA WALK MUTANTS (4): no-list-recursion, id-only-if-'name', no-phase-strip,
  id-only-if-'verification' -- ALL KILLED.
  RETIRED FINDING, recorded because I nearly filed it: I suspected the self-test's recall
  check was fed a FLAT synthetic plan that could not represent the subphases bug. I tried
  to evade it first: the shallow-walk mutant FAILS recall on that flat plan too
  (members 6, missing 1), because the phase object's own `phase-9`->`9` id is a dotted
  member the shallow walk never reaches. Not vacuous. Claim withdrawn.
  Disclosed-open `--operator-extend`: verified INERT by execution -- an extension row for
  999.99 is created (rc 0) yet a subsequent launch claiming 999.99 is still DENIED rc=2,
  because build_state is only reached after extract_step_id returns non-None. Audit noise,
  not a bypass. Disclosure adequate; criterion 4 names extract_step_id and is not falsified.

C5 PARTIAL -- preamble MET (control observed GREEN before any cell; N0 survived),
  clause 1 MET (M1 KILLED), clause 2 MET (M2 KILLED), and I verified all 15 shipped cells
  IMPORT CLEANLY so those kills are genuine. CLAUSE 3 **FALSIFIED BY EXECUTION** -- F1.

C6 MET. verdict_ledger.jsonl sha256 identical before/after the whole run. AST sweep of the
  three changed files: exactly ONE write-capable call touching VERDICT_LEDGER, at
  attempt_gate.py:577, inside _self_test (which rebinds it to a temp path). Zero in
  attempt_outcomes.py and mutation_matrix_90_1.py.

## FINDINGS

F1 [WARN, Threshold_Not_Met] criterion 5 clause 3 is falsified: a mutant that CANNOT RUN
   still scores KILLED. The cycle-2 fix added `ast.parse(mutated)`, which closes only the
   SyntaxError subset; "parses" is not "runs". Anchored on `_STEP_ID_RE = re.compile(...)`
   (count==1) I built three mutants that parse cleanly and cannot be imported:
     MXE3 RuntimeError at module scope -> KILLED (should be ERROR)
     MXE4 NameError at module scope    -> KILLED (should be ERROR)
     MXE5 ImportError at module scope  -> KILLED (should be ERROR)
     MXE6 SyntaxError control          -> ERROR  (the shipped guard works)
   Mechanism: observations() drives via subprocess.run, which does not raise on non-zero
   exit; an unimportable mutant fails every CHECK and run_cell credits a KILL. Same class
   the cycle-1 Q/A raised; narrowed, not closed. live_check C2.4's heading ("A mutant that
   cannot BUILD scores ERROR") overclaims what the guard does.
   BOUNDED: I applied all 15 shipped cells to a temp copy and imported each -- all 15
   import, so no reported kill is false today. Fix: smoke-import the mutant
   (`subprocess.run([py,'-c','import <mod>'])`) and score a non-zero import as ERROR.

F2 [WARN, Circular_Reasoning] CRITERIA EROSION: the cycle-1 critique carries SIX
   violation_details; the cycle-2 disposition table covers five. The sixth -- the
   containment check `all(p.parent == ESCALATION_DIR for p in ESCALATION_DIR.iterdir())`
   being a tautology -- appears in NO disposition row, in NO cycle-2 remediation section,
   and in NO "still open" list. The code is unchanged (attempt_gate.py:747-752), and I
   proved the clause by execution: True by construction with a file written OUTSIDE the
   dir, and vacuously True on an empty dir. Worse, the CYCLE-1 prose at
   experiment_results:244 ("the containment is now itself a check") stands uncorrected
   beside the verdict that refuted it. The underlying containment is real -- nothing leaked
   into handoff/current/ during my runs -- but a WARN was dropped rather than answered.

F3 [WARN, Overgeneralization] the M11 upper-bound cell is calibrated to the DECOY, not to
   the measured ambiguity threshold. drive_join plants the decoy 7,200,000 ms out. My
   sweep through the author's own harness:
     tol 0 KILLED | 1 SURV | 60 SURV | 300 SURV | 900 SURV | 1800 SURV | 3600 SURV
     | 7199 SURV | 7200 KILLED | 86400 KILLED
   The module's docstring says ambiguity first appears at 900s. I reproduced the damage on
   the REAL ledger: tol 30 -> 20,365,361 tokens / 0 ambiguous; 900 -> 18,998,336 / 6;
   3600 -> 4,015,375 / 71; 86400 -> 2,033,007 / 80 (9.98% -- the "~9%" claim reproduces).
   So a tolerance of 3600 collapses token accounting to 20% and SURVIVES the cell.
   Not a criterion miss (criterion 5 names only M1/M2/ERROR) and not vacuous. Fix: put the
   decoy just past the measured threshold, or assert on the resolved token SUM.

F4 [NOTE] experiment_results sec.5 "90.7" claims "Membership deliberately accepts both `X`
   and `phase-X`". Measured: 'phase-90' is DENIED -- _STEP_ID_RE refuses it before
   membership is consulted. The SET holds both forms; the GATE accepts only the bare form.

F5 [NOTE] over-inclusive recursion: of the dotted ids admitted, 1426 are step-like and
   exactly ONE ('77', a phase container at /phases/100) is not. Within the letter of
   criterion 4, which speaks to ABSENCE from the file.

F6 [NOTE] a pre-existing non-vocabulary outcome value ("BOGUS") is treated as settled and
   counted under its own key in outcome_counts, escaping the closed five. The backfill
   never WRITES such a value.

## COUNTERS (evidence, not a trigger)

qa_wip.py 90.1 --spawned-at 2026-08-20T19:50:39Z: source_present true, status ok,
  attempt_number 2, prior_attempts 1, attempt_number_is_lower_bound false,
  records_retained 2 (gauge), prior_records = the cycle-1 record 20260820T192724Z.
verdict_history_86_21.py --step 90.1 --evidence-only: status `no_rows_for_step`,
  verdicts (none).
CROSS-CHECK: prior_attempts 1 > ledger rows 0 -> THE LEDGER IS STALE for this step.
  sequence: UNKNOWN. Separately noted: the gate's own audit row for my spawn says
  attempt_number_inclusive=3 -- a DIFFERENT population (it counts the research-gate launch
  too). Neither number is reconstructible from the other.

## OUTCOME OF THIS RECORD
verdict CONDITIONAL. Five of six criteria MET and independently re-derived by execution;
criterion 5 clause 3 falsified by three executed counterexamples; worst finding severity
is WARN (no BLOCK), and both cycle-1 BLOCKs are genuinely fixed and reproduced.

COMPLETED: 2026-08-20T20:12:44Z
