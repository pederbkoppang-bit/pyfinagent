STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.85
WRITTEN: 2026-08-17T10:00:28Z

# Q/A write-first record -- step 86.85, cycle 6 (per Main's disclosure)

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable verification command; git status/diff scope; lint; scoped tests
C. LLM judgment against the 8 immutable criteria read VERBATIM from .claude/masterplan.json
D. Independent mutation testing of the NEW guards (esp. the cycle-6 event-date ordering fix)

## Findings log (appended as established)

### Prior-attempt / sequence evidence (gathered, not applied)
- `qa_wip.py 86.85 --spawned-at 2026-08-17T10:00:28Z`: source_present=true,
  attempt_number=6 (status ok, lower_bound true), prior_attempts=5, records_retained=6 (GAUGE).
- `verdict_history_86_21.py --step 86.85 --evidence-only`: status=ok,
  `FAIL -> FAIL -> FAIL -> CONDITIONAL -> CONDITIONAL` (5 rows).
- Cross-check: prior_attempts(5) == ledger verdict count(5) -> ledger is CURRENT for this
  step, NOT stale. (attempt_number 6 includes THIS spawn, which has no verdict yet.)
- masterplan `notes` still says "PARKED at [CONDITIONAL x4]" which the ledger contradicts;
  Main disclosed this in the spawn prompt as editable metadata to be corrected at close.

### B. Deterministic
- Immutable command: `parses`, EXIT=0. (ast.parse of scripts/qa/verdict_history_86_21.py)

### FINDING QA-C6-1 (executed, reproducible) -- QA-MUT-B is only PARTLY closed:
the event-date sort assumes ISO but validates nothing, and the escalation-clearing
backfill is STILL reachable through the shipped `--date` flag.

`emit_sequence` sorts by the raw string `row["date"]`. The docstring asserts the
premise "ISO `YYYY-MM-DD`, so lexicographic order IS chronological order".
`build_row` never validates it, and the REAL ledger already violates it:

    POPULATION: every non-blank line of handoff/verdict_ledger.jsonl, working tree 2026-08-17
    total rows 52 | date key present 52 | ISO YYYY-MM-DD 41 | non-ISO 11
    non-ISO distinct values: ['2026-08-09/10']  (steps 36.17 x6, 86.20 x3, 86.17 x2)

Driven probe (temp ledger, no repo write):

    $ verdict_ledger_write.py --step ZZ.1 --verdict CONDITIONAL --run-id wf_z1 --date 2026-08-11
    $ verdict_ledger_write.py --step ZZ.1 --verdict CONDITIONAL --run-id wf_z2 --date 2026-08-12
    $ verdict_ledger_write.py --emit-sequence --step ZZ.1
      ["CONDITIONAL", "CONDITIONAL"]
    $ verdict_ledger_write.py --step ZZ.1 --verdict PASS --run-id wf_z0 --date 2026-8-10
      append exit=0   (accepted silently; NOT zero-padded)
    $ verdict_ledger_write.py --emit-sequence --step ZZ.1
      ["CONDITIONAL", "CONDITIONAL", "PASS"]     emit exit=0

The PASS is the OLDEST event (Aug 10 < Aug 11 < Aug 12) and lands LAST.
Against the shipped `enforceEscalation` reverse scan (qa-verdict.js:572-581):
  true event order [PASS,C,C] -> n=2 -> would_auto_fail TRUE
  emitted order    [C,C,PASS] -> n=0 -> would_auto_fail FALSE
i.e. the exact fail-OPEN the cycle-5 QA-MUT-B finding named ("a backfill could
CLEAR a live escalation"), still reachable, exit 0, no warning.
The self-test check NAME overclaims: "backfilled older verdict lands in EVENT order
(a backfill cannot clear an escalation)" -- the parenthetical is refuted above.

### FINDING QA-C6-2 (BLOCKING) -- a SURVIVING MUTANT on the cycle-6 guard.
The `pos` file-position tiebreak introduced in cycle 6 is a NEW guard with NO
mutation cell and NO fixture that can observe it. Executed, control GREEN first,
zero repo writes, sha256 byte-identical before/after:

    sha256 before: dc70884ca21bf83fea77584727b7186df581d09beb55dd90dc94ca2419975ee2
    CONTROL      : rc=0 -> GREEN
    QA-M-POS-const  keyed.append((event_date, pos, verdict))
                 -> keyed.append((event_date, 0, verdict))
                 rc=0 -> *** SURVIVED ***   (writer --self-test, all 26 checks green)
                 rc=0 -> *** SURVIVED ***   (pytest, 29 passed; control also 29 passed)
    sha256 after : dc70884ca21bf83fea77584727b7186df581d09beb55dd90dc94ca2419975ee2  UNCHANGED

BEHAVIOURAL DIFFERENTIAL on the REAL production ledger (not equivalent):
  step 86.85 original ["FAIL","FAIL","FAIL","CONDITIONAL","CONDITIONAL"]
             mutant   ["CONDITIONAL","FAIL","FAIL","FAIL","CONDITIONAL"]
  step 86.74 original ["NO_VERDICT","NO_VERDICT","CONDITIONAL","CONDITIONAL","PASS","CONDITIONAL","CONDITIONAL","CONDITIONAL"]
             mutant   ["CONDITIONAL","CONDITIONAL","CONDITIONAL","CONDITIONAL","NO_VERDICT","NO_VERDICT","PASS","CONDITIONAL"]
Fed to the shipped enforceEscalation (extracted by brace-matched anchor, 52 lines):
  86.85 original -> n=2 auto_fail=TRUE ; mutant -> n=1 auto_fail=FALSE
i.e. it DISARMS the escalation on this step's own real data. Fail-OPEN, the same
direction and the same consumer as QA-MUT-B.
Root cause: within one event date the sort falls through to element 3 (the verdict
STRING) and orders alphabetically (CONDITIONAL<FAIL<NO_VERDICT<PASS). Every ordering
fixture uses DISTINCT dates, so no fixture can observe a within-date reorder -- and
same-date rows are the COMMON case (86.85 cycles 1-4 all 2026-08-15; 86.74 cycles
1-6 all 2026-08-14). This is the identical failure class that produced this step's
cycles 1, 2 and 3 FAILs: a NEW guard shipped with no mutation cell.

### Independent drive of the shipped enforceEscalation (criterion 4/6/7)
  CONTROL  1 prior C + cur CONDITIONAL   n=1  auto_fail=false  status=ok
  DRIVEN   2 prior C + cur CONDITIONAL   n=2  auto_fail=true   status=ok
  CONTROL  2 prior C + cur PASS          n=2  auto_fail=false  status=ok
  CONTROL  2 prior C + cur FAIL          n=2  auto_fail=false  status=ok
  [C,C,NO_VERDICT] + cur CONDITIONAL     n=2  auto_fail=true   (drop does NOT reset)
  absent sequence                        n=null auto_fail=null status=not_supplied

### Re-derived numbers (all reproduce experiment_results cycle 6)
  self-test SELF-TEST PASSED exit 0, 26 checks emitted
  mutation_matrix_86_85.py: 16 cells, 16 killed, 0 survived, 0 unscorable, exit 0
  coverage: guards 17 covered 17 uncovered 0 cell problems 0
  pytest -k '86_85 or ledger or verdict_ledger': 36 passed, 3514 deselected (36 dots)
  wc -l handoff/verdict_ledger.jsonl = 52
  immutable command: parses, EXIT=0

### FINDING QA-C6-3 -- live_check_86.85.md carries STALE post-cycle-6 numbers
live_check is the artifact the masterplan `live_check` field names. Its cycle-5
annotation at the "## 6. MUTATION MATRIX" heading states the CURRENT superseding
state as "the matrix grew to 14 cells / 14 KILLED in cycle 4" -- it is 16/16 now.
C8.6 "verbatim output" still reads `14 cells: 14 killed`, `guards: 15 covered: 15`,
`checks emitted: 23`, `34 passed`. Measured now: 16/16, 17/17, 26, 36.
The ONLY cycle-6 edit to live_check was the [WORKING TREE] 46 -> 52 line
(`git diff -- handoff/current/live_check_86.85.md`). experiment_results carries the
correct numbers. This is the third recurrence in this file of the defect cycle 2
was FAILed for (updating experiment_results and not live_check) and cycle 3 caught
(stale header).

### A. Harness compliance -- CLEAN on all 5
1. Research gate: brief_status COMPLETE, gate_passed true, 8 read in full (>=5),
   23 URLs (>=10), dedicated "Recency scan (2024-2026) -- PERFORMED" section.
2. Research-before-contract: brief ADDED in 9034ddfb 2026-08-14 21:41; contract ADDED
   in d1c4a79d 2026-08-15 15:44. The mtime inversion (brief 16:16 > contract 15:59) is
   later remediation annotation, not authoring order. SATISFIED.
3. experiment_results_86.85.md present, carries cycle-5 + cycle-6 GENERATE sections.
4. Log-last: masterplan status=pending; harness_log has 2 rows for phase=86.85
   (cycles 197, 220) and none for the in-flight cycle. CORRECT.
5. No verdict shopping: evidence CHANGED (354 insertions / 23 deletions across
   6 files uncommitted; writer +86 lines; M15/M16 added). Documented respawn.

### Scope / lint / consumer integrity
- ruff F821,F401,F811 over the DERIVED scope (6 .py files, non-empty set asserted,
  xargs -0): "All checks passed!" exit=0.
- UNTOUCHED, verified by `git status --short --`: .claude/workflows/qa-verdict.js,
  scripts/qa/verdict_history_86_21.py, .claude/agents/qa.md,
  scripts/qa/verify_matrix_coverage_86_85.py. The "no gate weakened, consumer
  untouched" claim REPRODUCES.
- Other dirty files (sovereign_api.py, rail_turn_cap.py, mutate_rail_turn_cap.py,
  the 86.84/86.90 handoff artifacts) belong to peer/concurrent steps; Main's
  working_tree_note named only the sovereign-UI files + perf_results.tsv, so the
  disclosure is INCOMPLETE but none of the omissions touch 86.85's subject. NOTE.

### Claims that REPRODUCE (independently re-derived)
- C2 [WORKING TREE]: total 52, step_ids 13, {C 29, F 8, PASS 8, NV 7},
  recorded_at 38/52, 14 predate the field, 47+4+1=52. ALL exact.
- C1 localisation: NEVER-WRITTEN with a positive control. Sound.
- C3 cross-process: independently corroborated -- I read back this step's own
  cycle-5 row (written by a different process earlier today) via
  verdict_history_86_21.py.
- C4 driven auto-FAIL: reproduced by my own brace-matched extraction.
- C6 drop-does-not-reset + absent-sequence-is-null: reproduced.
- C7: PASS stays PASS, FAIL stays FAIL under 2 prior CONDITIONALs.
- 86.74 emit-sequence still reproduces under the NEW event-date ordering.

### NOTE -- stale line citation
experiment_results C4 and live_check §3 cite "lines 319-370 of qa-verdict.js".
Measured: enforceEscalation was at line 319 at d1c4a79d / 39999944 / 9a18150f and is
at 535 at HEAD. The citation was ACCURATE when written and went stale via other
steps' edits; the quoted OUTPUT still reproduces. NOTE, not a false claim.

### Criterion roll-up
C1 MET | C2 MET | C3 MET | C4 MET | C5 MET | C6 MET | C7 MET
C8 NOT MET -- QA-C6-2: a new cycle-6 guard (`pos` tiebreak) has no cell and no
fixture that can observe it; the mutation survives both oracles and fails OPEN on
this step's own production data.
Also outstanding: QA-C6-1 (unvalidated ISO precondition, escalation-clearing backfill
still reachable through the shipped --date flag) and QA-C6-3 (stale live_check).

COMPLETED: 2026-08-17T10:12:41Z

