STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.71
WRITTEN: 2026-08-17T12:07:37Z

# Q/A cycle-5 evaluation of step 86.71 (write-first record)

Launched via Workflow rail. Prompt says: cycle-4 verdict was CONDITIONAL with a
single NOT-MET (criterion 8 -- loud-swallow fix had zero automated coverage).
Claimed sequence on record: [FAIL, CONDITIONAL, CONDITIONAL, CONDITIONAL].

## Plan
- A. Harness compliance audit (5 items)
- B. Deterministic: immutable command, git status, lint, tests
- C. Mutation / vacuity checks of the NEW cycle-5 guards
- D. Criterion-by-criterion MET/NOT MET

## Log

### Prior-attempt / verdict evidence (gathered, not used as a trigger)
- `qa_wip.py 86.71 --spawned-at 2026-08-17T12:07:37Z`: source_present=true,
  attempt_number=5 (status ok, is_lower_bound=true), prior_attempts=4,
  records_retained=5 (gauge), identity_checked=true.
- `verdict_history_86_21.py --step 86.71 --evidence-only`: status=ok,
  4 verdicts, sequence `FAIL -> CONDITIONAL -> CONDITIONAL -> CONDITIONAL`.
- Cross-check: attempt_number(5) - self(1) = 4 == ledger count 4. Ledger NOT
  stale for this step.
- INDEPENDENT third source: `handoff/audit/attempt_budget_audit.jsonl` last row
  `{"ts":"2026-08-17T12:07:32Z","step_id":"86.71",...,"attempt_number_inclusive":5}`
  written by the gate itself at MY launch. Three sources agree.

### B. Deterministic
- Immutable cmd `ast.parse(scripts/harness/attempt_budget.py)` -> "parses", EXIT=0.
- `attempt_gate.py --self-test` -> 17 ok-checks, SELF-TEST PASSED, rc=0.
  (matches the claimed "15 -> 17 ok-checks")
- `mutation_matrix_86_71.py --verify` -> CONTROL green (11 checks),
  relocated-unmutated SURVIVES, null-mutant SURVIVES, cells=9 killed=9
  survivors=0 errors=0, md5 cd2164da... before == after, VERIFY: PASS, rc=0.
  (matches the claimed md5 cd2164da and 9/9)
- Wiring confirmed live in `.claude/settings.json`: PreToolUse matcher
  `Workflow` -> `python3 .../scripts/harness/attempt_gate.py`.
- Ledger census: 18 rows; by step {999.2:5, 86.85:4, 86.71:5, 86.84:3, 9.4:1};
  by type {attempt:17, operator_extension:1}. Real production rows present.

### C. INDEPENDENT mutation work (mine, not the author's)
1. **V2_ISOLATED** (keep the loud print, flip ONLY `return []` ->
   `return [Outcome.PASS]`): matrix -> KILLED, and killed **by the
   fail-closed-direction check specifically** (at_vlerr rc went 2 -> 0).
   => the direction check is NOT vacuous; the compound G9 does not hide it.
2. Self-test discrimination under 3 mutants:
   - CONTROL rc=0, no FAIL lines
   - V1_silent (=G8) rc=1, FAIL "verdict-ledger read error is LOUD (V1)"
   - V2_isolated rc=1, FAIL "read error grants NO PASS exception (V2)"
   - V2_full (=G9) rc=1, FAILS BOTH
   => both self-test checks independently discriminate.
3. **Crash-guard widening PROVEN load-bearing.** Built drive-specific crashers:
   - `crash_only_at` (sys.exit(9) in write_escalation): rc below/at/at_vlerr =
     0/9/9. BELOW-only guard says broken=False -> would have scored KILLED
     (4 checks fail = mirage). ALL-3 guard says broken=True -> ERROR. Correct.
   - `crash_only_at_vlerr` (sys.exit(7) in verdict_outcomes' except): rc =
     0/2/7. BELOW-only guard False, BELOW+AT guard False, ALL-3 guard True.
     2 checks fail, so under either narrower guard this scores a MIRAGE KILL;
     only the widened guard scores ERROR.
   => cycle-5 claim #3 (stronger OR-branch) is real and demonstrated by me.
   - `crash_all` (sys.exit(11) in append_row): 11/2/2, all guards catch.
4. **Orthogonality**: G8 (silent revert) fails EXACTLY ONE matrix check (the
   loudness one). V2_ISOLATED fails EXACTLY ONE (the direction one). Neither
   check is carried by the other -> the compound G9 hides nothing.
5. **DECOY probe (residual, NOT capping)**: swallow reverted to silent PLUS an
   unconditional `print("verdict-ledger read failed (decoy)")` in handle_hook
   -> SURVIVES the matrix, 0 failing checks. The loudness check is a stderr
   SUBSTRING assertion, not a provenance assertion. NOT a criterion miss: the
   REAL revert does go red (which is literally what criterion 8 asks), and the
   paired direction check is behavioural rc and cannot be decoyed by a print.
   Hardening: assert the exception type name / the step id in the line.

### D. Criteria re-derived BY ME
- **C1 MET.** Ran live_check s7's script VERBATIM (extracted from the artifact
  by regex, exec'd): 500 runs / 335 repeats = 67.0%, qa 317/406 = 78.1%,
  researcher 18/93 = 19.4%, max-qa 86.85 -> 10 (corpus now 599). Snapshot
  replay: oldest 580 -> 481/320 = 66.5%, qa 302/390 = 77.4%, researcher
  18/90 = 20.0%, max-qa 36.8 -> 9 == section 1's figures EXACTLY. oldest 513
  -> 64.7% == the decomposition. 58.4% disagreement reported, not adopted.
- **C2 MET.** My own controls at 192ef652^: "BudgetState" -> 3 files,
  "DEFAULT_MAX_ATTEMPTS" -> 3, "attempt_budget" -> 3; NEGATIVE control
  "ZZZ_NO_SUCH_SYMBOL_86_71_QA" -> 0. Runtime callers (excluding module,
  tests, matrix_86_32, verify_counter_86_79) -> 0. New caller at
  attempt_gate.py:84.
- **C3 MET.** 12 separate OS processes on a temp ledger: 6 launch processes
  each followed by an INDEPENDENT `--status` process reading 1/5,2/5,3/5,4/5,
  5/5 then rc=2 at #6. The incremented value crosses the process boundary.
- **C4 MET.** settings.json PreToolUse matcher `Workflow` ->
  attempt_gate.py (parsed the file). Drive: at-ceiling rc=2 + escalation file
  1,248 bytes naming --operator-extend, "THIS IS NOT A PASS AND NOT A FAIL",
  no verdict key; other step 77.10 rc=0 CONTINUE/allow. LIVE on PRODUCTION
  data: --status 86.71 = 5/5 ESCALATE/deny, 86.85 = 5/5 ESCALATE/deny,
  86.84 = 3/5 CONTINUE/allow, 86.99 = 0/5 CONTINUE/allow. And the hook wrote
  a production row for MY OWN launch.
- **C5 MET.** My own exhaustive sweep: 4,368 cells (every non-PASS sequence
  len 1..6 x product_verified x evidence_complete) -> disposition set AND
  close_kind set BOTH exactly {CONTINUE, ESCALATE}; 0 CLOSED_* from a
  non-PASS history; all 972 at/over-ceiling non-PASS sequences ESCALATE;
  FAIL-only = CONTINUE under all 4 flag combos. Positive control (FAIL then
  PASS) discriminates -> CLOSED_PASS / CLOSED_COMPLETE /
  CLOSED_PRODUCT_RESIDUALS_QUEUED / ESCALATE. pytest 15 passed.
- **C6 MET.** Gate is role-agnostic at the Workflow seam; every production
  attempt row is `qa-verdict.js`, including 2 written during my evaluation
  (mine 12:07:32Z and a peer's 86.85 at 12:10:50Z).
- **C7 MET.** No .env in any 86.71 commit (192ef652/cbbd1566/2a6cd4b6);
  masterplan.json untouched, status still `pending`; only config change is
  the +11-line PreToolUse/Workflow block criterion 4 requires; ASK-1 at
  contract:88. All 8 criteria byte-verbatim in the contract.
- **C8 MET.** Control green first (11 checks + 2 discrimination controls),
  9/9 KILLED, md5 cd2164da before == after, and my independent reverts above.
  Capture fixes verified: s10 re-run block uses `cmd > f 2>&1; echo EXIT=$?`
  (unpiped) and all 3 values reproduce here; s8's stale marker-list passage
  REPLACED in place with the class-test wording.

### E. Harness compliance 5/5
1. research_brief_86.71.md: brief_status COMPLETE, 9 in full, 30 URLs,
   recency_scan true, gate_passed true; cited by the contract.
2. mtime order research 11:47:50 < contract 12:07:27 < attempt_gate 14:01:12
   < matrix/live_check 14:04:52 < experiment_results 14:06:02 (local).
3. experiment_results present with the cycle-5 section at the tail.
4. log-last: `grep -F "phase=86.71" harness_log.md` -> 0; masterplan pending.
5. no verdict-shopping: evidence CHANGED after the cycle-4 verdict
   (2a6cd4b6: attempt_gate +25, matrix +65, results +49, live_check +51).

### F. Integrity
- HEAD stable ca800b50 start -> end. Graded files clean in git.
- md5 after ALL my mutation work: attempt_gate cd2164da..., attempt_budget
  5511ac7e..., matrix b9e20475..., settings.json b8cbd61b... unchanged.
- My `--self-test` and ~40 subprocess drives added ZERO synthetic rows to the
  production ledger (git diff shows only 2 REAL launch rows since HEAD) --
  independent confirmation the cycle-3 def-time-default pollution fix holds.
- Peer-dirty files (sovereign_api, frontend, perf_results.tsv) reference
  attempt_budget/attempt_gate 0 times each.

### G. Residual NOTEs (none capping, none owned by a criterion)
(i) decoy/substring loudness assertion (above).
(ii) Main's artifacts state the crash-guard widening but report no revert
     differential for it; I executed the missing differential and it PASSES.
(iii) the 5 `session_id: "pipetest"` synthetic 999.2 rows remain in the
     git-tracked production ledger; self-labelling and s3 names 999.2 as the
     seeded step, but no line says they are still there. Append-only, so the
     fix is a callout, not a rewrite.
(iv) live_check s4/s5 still show the cycle-1 12-check self-test and 6-cell
     matrix under a "Criterion-8" header with no supersession marker; s9/s10
     carry the current state and s10 states it explicitly.

### VERDICT (to be returned): PASS
All 8 criteria MET with evidence I executed myself; harness compliance clean;
no unintended production change; the cycle-4 NOT-MET is closed and proven
load-bearing by orthogonal reverts I built.

COMPLETED: 2026-08-17T12:18:07Z

