STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 90.1
WRITTEN: 2026-08-20T20:10:11Z

# Q/A write-first record -- step 90.1, CYCLE 3 (Main says attempt 4 of 5)

Spawn context (from Main, ADVISORY only):
- Commit b5a9b9d6 (cycle-1 3bf0b0fe, cycle-2 1fc7b2e6). DO NOT grade 1c2f25b3.
- Changed since cycle 2: scripts/qa/mutation_matrix_90_1.py, scripts/harness/attempt_gate.py
- Main claims 3 cycle-2 WARNs fixed: (1) criterion_5_clause3 smoke-import ERROR scoring,
  (2) criteria-erosion / containment tautology replaced, (3) illusory-guard M11 decoy moved to 900s.

## Plan
A. harness-compliance audit (5 items)
B. deterministic: immutable command exit code + git status/diff + lint + smoke
C. LLM judgment vs 6 immutable criteria + adversarial attacks Main invited

## Findings log (appended as established)

### A. HARNESS COMPLIANCE (5 items) -- CLEAN so far
- research gate: handoff/current/research_brief_90.1.md envelope brief_status=COMPLETE,
  external_sources_read_in_full=10, urls_collected=25, recency_scan_performed=true, gate_passed=true. OK (>=5, >=10).
- contract-before-generate mtimes: research 21:12:47 < contract 21:15:34 < attempt_gate.py 22:07:24
  < mutation_matrix 22:08:14 < experiment_results 22:09:37. ORDER OK.
- experiment_results_90.1.md present (28144 b), evaluator_critique_90.1.md present (33954 b), live_check present.
- log-last: `grep -cF 'phase=90.1' handoff/harness_log.md` = 0, exit 1. masterplan status still "pending". OK.
- no-verdict-shopping: cycle-3 commit b5a9b9d6 changed scripts/harness/attempt_gate.py (+36) and
  scripts/qa/mutation_matrix_90_1.py (+61). EVIDENCE CHANGED. OK.
- IMMUTABILITY: criteria sha + command sha byte-identical at 3bf0b0fe / 1fc7b2e6 / b5a9b9d6 / HEAD
  (crit_sha=f98626019b331382, cmd_sha=e5ae167bc5ca36c5). b5a9b9d6 does NOT touch .claude/masterplan.json.

### B. DETERMINISTIC
- IMMUTABLE COMMAND: `python3 scripts/harness/attempt_gate.py --self-test` -> SELF-TEST PASSED, exit 0.
  `python3 scripts/qa/mutation_matrix_90_1.py --verify` -> exit 0.
  KILLED 15 | SURVIVED 0 (excl. N0) | ERROR 0 | null mutant survived: True.
- ledger sha256 BEFORE and AFTER the whole run: fcfe56ad9788f0bc248253aea49e086812ab951c4145ecc5eac2b92c982e3eb2
  == fcfe56ad9788f0bc248253aea49e086812ab951c4145ecc5eac2b92c982e3eb2 (c6 re-derived independently).
- real tree md5 before==after True (matrix self-reported AND re-derived by me on every custom cell).
- git status --short: only handoff/audit/*.jsonl (gate's own append-only audit) + my WIP file. No unintended prod change.

### C. ADVERSARIAL -- criterion 5 clause 3 ("a mutant that fails to run scores ERROR")
FINDING QX (REPRODUCED 3x, both subject modules): the cycle-3 smoke-import probe closes the
IMPORT-time seam only. A mutant that PARSES, IMPORTS CLEANLY, and then CANNOT RUN still scores KILLED.
  QX1  attempt_gate.py   `return handle_hook()` -> deferred `import nonexistent_module_zzz_qa` in main()  => KILLED
  QX2  attempt_gate.py   `return handle_hook()` -> `return handle_hook_v2()` (NameError at call time)     => KILLED
  QX3  attempt_outcomes.py  `def main(...)` + deferred missing import                                      => KILLED
All three: anchor count 1, ast.parse OK, smoke-import exit 0, real tree md5 unchanged.
QX2 is the REALISTIC authoring error (a repl naming a helper that does not exist).
This is the cycle-2 defect relocated ONE SEAM later: parse -> import -> RUN.

Failure-count is NOT an incidental discriminator: QX3 fails 5 of 25 checks -- EXACTLY the
same count as shipped cell M3 (a genuine kill). Shipped cells fail 1-5 of 25; QX1/QX2 fail 9.
So nothing in the current scoring separates apparatus failure from a real kill.

THE GAP IS CLOSABLE -- verified discriminator (so the criterion is achievable):
on the hook drive, QX1/QX2 exit rc=1 with an unhandled `Traceback` on stderr, while
ALL 16 shipped cells (N0..M14) exit rc=0 with EMPTY stderr. QX3 likewise: the resolver's
no-arg invocation gives rc=1 + Traceback vs rc=2 + no traceback for every shipped cell.
An "unhandled traceback in a drive => ERROR" rule scores QX1/QX2/QX3 ERROR and leaves all
15 kills untouched. (NOTE: a benign-path smoke such as `--status 9.1` does NOT work --
QX1/QX2 return rc=0 there because the mutation sits on the hook branch only. Measured.)

### C2. M11 tolerance band -- Main's ATTACK claim #3 is FALSIFIED BY EXECUTION
Main asserts: "Tolerances 1-899 survive, and that is correct rather than a residual gap:
they sit below the documented ambiguity threshold, so there is no ambiguity for the guard
to catch."
Independent sweep reproduces his matrix result exactly (0 KILLED; 1/60/300/899 SURVIVED;
900/901/1800/3600/86400 KILLED).
But on the REAL corpus AS OF HIS OWN COMMIT b5a9b9d6 (98 rows), re-resolved from scratch:
  tol=30  ambiguous_match=0   {C:46 F:11 NV:21 P:11 U:5}
  tol=300 ambiguous_match=0   (identical -- so <=300 IS an equivalent mutant; that half of
                               his claim is CORRECT)
  tol=386 ambiguous_match=1   {C:45 ... U:6}      <-- ambiguity FIRST APPEARS AT 386s
  tol=500 ambiguous_match=2
  tol=899 ambiguous_match=6   {C:43 F:10 NV:20 P:10 U:11}  5 graded outcomes LOST
On the CURRENT ledger (95 attempt rows) the same binary search gives first divergence at
tolerance 386 (last matching 385), and summed total_tokens:
  tol=30 20,806,242 | tol=386 20,528,843 (-1.3%) | tol=899 19,439,217 (-6.6%)
  | tol=3600 3,637,821 | tol=86400 1,655,453
So the band [386,899] IS real, measurable damage that M11 does not catch. The claim was
FALSE WHEN MADE, not merely drifted.
ROOT CAUSE: attempt_outcomes.py:35 docstring says "Ambiguity first appears at 900s",
measured over "the real 89 attempt rows". That number was ALREADY stale at b5a9b9d6, and
Main borrowed it to certify the survivor band as equivalent.
CLASSIFICATION: criterion 5 names only three mutants (M1, M2, the ERROR clause); M11 is
not one of them. Criterion 1's "UNKNOWN only where no run record exists" HOLDS at the
shipped default (ambiguous_match=0 at tol=30). So this is a RESIDUAL guard-quality +
scope-honesty finding, NOT a numbered-criterion miss.

### C4. Main's ATTACK #2 (containment replacement) -- NOT vacuous, one blind spot
Red-first INDEPENDENTLY reproduced. Sandbox with ATTEMPT_GATE_ESCALATION_DIR pointed at a
temp "real" dir; mutant `ESCALATION_DIR = old_e` (leak):
  preseed a DIFFERENT name -> BOTH checks FAIL, rc=1. Main's claim reproduces.
  preseed the EXACT name the self-test writes -> clause 1 says "ok" while the preseeded
    file was OVERWRITTEN (sha differs). Clause 2 still FAILs, so the pair still catches it.
  unmutated control -> both ok, preseeded file untouched.
So: the pair is sound; clause 1 alone is blind to an IN-PLACE OVERWRITE because it compares
a NAME SET. One-line strengthening: compare a name->sha256 map. NOTE-level; criterion 2's
byte-identical requirement is separately covered by real sha256 comparisons that I re-ran.
Placement verified: real_before is snapshotted at :540, redirect at :544, first
write_escalation at :671 -- no write can precede the snapshot.

### C5. RESIDUAL -- the VERDICT-LEDGER channel is the one containment did NOT cover
attempt_gate.py:581 `VERDICT_LEDGER.write_text(json.dumps({... "verdict": "PASS" ...}))`
(introduced by phase-86.71, NOT by 90.1; write_text TRUNCATES). Contained only by the
one-line redirect at :532. Structurally-equivalent sandbox (/tmp/X/scripts/{harness,qa} +
/tmp/X/handoff so REPO=parents[2] resolves inside the sandbox):
  CONTROL  rc=0 SELF-TEST PASSED, sandbox verdict ledger byte-identical
  MUTANT (redirect removed) rc=0 SELF-TEST PASSED, ZERO FAILs, sandbox verdict ledger
          TRUNCATED from 3 rows to `{"step_id": "9.1", "verdict": "PASS", ...}`
The immutable command has NO verdict-ledger guard (matrix only redirects it via env at
mutation_matrix_90_1.py:92). Criterion 6's own assertion method -- the manual sha256 pair --
WOULD go red on this mutation, so the guard is not vacuous; it is simply out-of-band.
Criterion 6 MET AS OPERATIONALIZED; the unguarded third channel is a RESIDUAL to file.
NOTE: running the self-test from a plain copy is location-bound -- `sys.path.insert(REPO/
scripts/qa)` with REPO=parents[2] means verdict_ledger_write is unimportable and the
"verdict-ledger PASS -> allow" check goes red. Not a defect; it invalidates naive copy-runs.

### C6. Criterion-by-criterion, independently re-derived
c1 MET  95 attempt rows + 4 extension rows; 0 missing `outcome`, 0 missing `total_tokens`;
        EXACTLY 92 rows carry an INTEGER total_tokens and the 3 nulls are in-flight
        launches tagged `unresolved_at_launch` (19:50:34Z 90.1, 19:50:56Z 90.9,
        20:10:07Z 90.1 = MY OWN spawn). Backfill re-runnable + additive-only (M8/M14
        killed; I ran it on copies at 8 tolerances, rc 0 each). Counts printed
        {C46 F11 NV21 P11 U6} at the shipped tol=30; UNKNOWN=6 with reason_counts
        no_run_record=6 and ambiguous_match=0 -> "UNKNOWN only where no run record
        exists" HOLDS and the count is stated.
c2 MET  matrix CONTROL on the REAL tree: "a NON-exhaustion denial leaves a pre-existing
        exhaustion escalation BYTE-IDENTICAL (c2, sha256 before == after)" ok; M4/M5 killed;
        self-test refusal writes NO file; live_check sec 2 shows the 4-file sha pair.
c3 MET  by EXECUTION: 1,200,001 (= DEFAULT_MAX_TOKENS+1) DENIED with 4 of 5 attempts
        unused; 1,199,999 ALLOWED -> discriminates. M6/M7 killed.
c4 MET  real module + real masterplan: 86.118 ADMITTED; 86.118.1 / 86.1180 / 999.99 DENIED.
        Self-test ids exempted BY CONSTRUCTION via an explicit synthetic plan (not silent).
        My own recall walk: 1350 dotted ids in the plan, 45 not admitted -- ALL of shape
        `25.<letter>` and ALL status=done; PENDING not-admitted = 0. Pre-existing shape
        refusals, zero new denials.
c5 NOT MET (clause 3 only) -- see section C above. Clauses 1 and 2 MET (M1, M2 KILLED,
        control observed GREEN first).
c6 MET as operationalized: handoff/verdict_ledger.jsonl sha256
        fcfe56ad...2e3eb2 IDENTICAL before and after my entire run.

### C7. NOTES (non-degrading)
N1 mutation_matrix_90_1.py drive_join comment says "Moved to 950s"; code plants 900_000 ms
   = 900 s. At 950s the comment's own stated property would be FALSE (tol=900 would
   survive). Leftover from the disclosed interim state.
N2 live_check_90.1.md sec 4 is STALE: labelled the verbatim immutable-command tail, it
   shows "KILLED 10 | SURVIVED 0", attempt_gate md5 21f35583..., and an M10 description
   that no longer exists. HEAD produces KILLED 15 and md5 61f257b7.... mtime 21:49:42 <
   the cycle-3 edits (22:07/22:08), so it was not regenerated. experiment_results' own
   "Cycle-3 verification" block IS current (15 kills).
N3 "All six cycle-1 findings ... this table enumerates them" -- the CYCLE-1 table was NOT
   touched in b5a9b9d6 (the critique diff is purely additive). It still has 5 rows for 6
   findings, still merges the M10-mislabel + ERROR-clause WARNs, and still omits the
   Circular_Reasoning item, which is instead carried by the cycle-2 table. All nine items
   ARE accounted for across the two tables; the "both tables enumerate" claim is loose.
N4 attempt_outcomes.py:35 "Ambiguity first appears at 900s" (measured over 89 rows) was
   already false at b5a9b9d6 (386s) -- and it is the number Main borrowed to certify the
   sub-900 survivor band.
N5 b5a9b9d6 bundles unrelated 90.2/90.9 artifacts via the auto-commit hook's `git add -A`.
   No production code outside the two 90.1 scripts. Not a finding.
N6 PROCESS: I wrote no repository file. All mutation/backfill work ran on tempfile copies;
   real-tree md5s and both ledger sha256s verified unchanged after every experiment. One
   shell redirect was used, to /tmp/qa901_matrix.txt (outside the repo), to capture an
   exit code; disclosed rather than omitted.

### C8. Sequence / attempt evidence (gathered, not applied)
verdict_history_86_21.py --step 90.1 --evidence-only -> status "no_rows_for_step",
  verdicts (none).
qa_wip.py 90.1 --spawned-at 2026-08-20T20:10:11Z -> source_present TRUE,
  attempt_number_status "ok", attempt_number 3, prior_attempts 2,
  attempt_number_is_lower_bound TRUE, records_retained 3 (GAUGE), records_pruned_known null.
CROSS-CHECK: prior_attempts 2 > the ledger's 0 rows -> THE LEDGER IS STALE for this step.
  sequence: UNKNOWN.
Separate population, not reconstructible from the other: the gate's own audit row for my
  spawn (20:10:07Z) feeds attempt_number_inclusive, which counts research-gate launches --
  that is where Main's "attempt 4 of 5" comes from.

### VERDICT (recorded here as the analysis conclusion; the RETURN is the deliverable)
CONDITIONAL. Immutable command exit 0. 5 of 6 criteria MET by my own execution.
Criterion 5 clause 3 NOT MET -- a NUMBERED criterion miss, third cycle running, each time
closed one seam short (parse -> import -> run). BOUNDED to WARN not BLOCK: all 16 shipped
cells were verified to RUN (rc=0, empty stderr, no traceback on both the hook drive and the
resolver --backfill drive), so none of the 15 reported kills is an apparatus artifact and
the matrix RESULT stands; the exposure is prospective.
The other findings are RESIDUAL, with NO numbered criterion unmet: M11's sub-900 band
(F2/C2), the unguarded verdict-ledger channel (C5), and N1-N4. Per the operator's routing
instruction these belong as their own masterplan steps, not as in-place remediation.

### C9. Evidence freeze / graded state
HEAD moved during this evaluation (2c9018da -> 8626e8a2; Main landed 90.2/90.9 contracts).
All seven 90.1 artifacts + the three scripts are sha256-IDENTICAL between b5a9b9d6 and the
working tree, so the state I graded is exactly the state Main named. Commit 1c2f25b3 was
NOT in any diff range I used. Real-tree md5 after every experiment:
attempt_gate 61f257b75abdd8b164417410f0665a83, attempt_outcomes 81ebe68b498c63cbc424bf1f01ae02d1,
attempt_budget 5511ac7e6f105b6b0716d4b80812a170 -- all matching the matrix's self-report.
Contract completeness: all 6 immutable criteria present VERBATIM in contract_90.1.md
(string containment against masterplan.json, not by eye).
Lint: git-derived scope (2 changed .py) F821/F401/F811 -> All checks passed, exit 0;
full step scope (4 .py since 3bf0b0fe) -> All checks passed, exit 0. Empty-set guard fired
non-empty before the exit code was read.
Gates N/A: no frontend/** in the diff (1b), no UI claims (1c), no backend/** (1d).

COMPLETED: 2026-08-20T20:27:28Z

### C3. Prose defect inside the changed file
scripts/qa/mutation_matrix_90_1.py drive_join comment says "Moved to 950s" while the code
plants `900_000` ms = 900 s (offset_ms units confirmed at _plant_run_record). The comment's
own stated property ("ambiguous for anything at or past the documented threshold") is only
true at 900s; at 950s tol=900 would SURVIVE. Leftover from the interim state Main discloses
in experiment_results W3. Code right, comment wrong. NOTE-level.


