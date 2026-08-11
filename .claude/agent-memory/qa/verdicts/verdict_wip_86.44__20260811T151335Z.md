STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.44
WRITTEN: 2026-08-11T15:13:35Z

# Q/A write-first record -- step 86.44, cycle 1

Launch: Workflow rail (qa-verdict.js). Read qa.md in full at 15:13:35Z.
Commit under review: fe9a6dad.

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command + git diff scope + lint + syntax + runtime smoke
C. Criterion-by-criterion judgment, with independent re-derivation of every number
D. Mutation matrix independent re-run (esp. M1 seed size), TOCTOU probe scrutiny

## Findings log (appended as established)

### B1. IMMUTABLE VERIFICATION COMMAND -- exit 0
`bash -c 'test -f handoff/harness_log.md && grep -c "^## Cycle" handoff/harness_log.md'`
-> `1224`, exit=0. Matches the artifact's 1,224.

### A. HARNESS COMPLIANCE -- clean
- research_brief_86.44.md exists (24,943 B), committed ea5b1cd5 WITH the contract.
- contract ea5b1cd5 (16:56/16:59) PRECEDES generate commit fe9a6dad (17:13). Order OK.
- experiment_results_86.44.md present (11,912 B).
- LOG-LAST honored: `grep -F "phase=86.44" handoff/harness_log.md` -> 0 hits.
  masterplan 86.44 status=pending, retry_count=0. Not flipped.
- No verdict-shopping: 0 prior CONDITIONAL rows for 86.44. This is cycle 1.
- masterplan.json WAS edited in fe9a6dad (+21 lines) = the 86.55 filing ONLY.
  Verified 86.44's own criteria block is UNTOUCHED (diff shows only a new
  86.55 object appended after 86.54). No criteria amendment.

### B2. CENSUS RE-DERIVED INDEPENDENTLY (two rules of my own) -- ALL SIX FIGURES REPRODUCE
harness_log.md is byte-identical between census tree 915d2cb0 and HEAD (git diff empty),
so the tree naming is honest and current.
  R1 (author's rule `^## Cycle (.+?)\s*--`) and R2 (mine: split on first `--`) both give:
    headers 1224 | numeric 1064 | non-numeric 160 | token=='1' 481
    distinct dup ints 141 | headers in a dup group 969
  Every figure in experiment_results section 1 reproduces exactly. No overclaim.

### B3. THE 481-vs-482 DISCREPANCY IS RESOLVABLE AND I RESOLVED IT
`grep -cE '^## Cycle 1\s*--'` = 481 ; `grep -c '^## Cycle 1 '` = 482.
The single differing line is harness_log.md:25421:
  `## Cycle 1 (production-ready+money push) -- 2026-05-28 -- phase=47.1 result=PASS`
So the token is `1 (production-ready+money push)`, NOT literally `1`.
=> The author's 481 is CORRECT for "token is literally 1"; the gate's 482 came from a
prefix rule that also swallows this parenthetical header (which the author's own
36-strong "integer + parenthetical" class already accounts for -- internally consistent).
The author left it as "two defensible rules disagree by one" rather than resolving it in
one command. NOTE-level: disclosure was honest, resolution was cheap and skipped.

### B4. FINDING -- "the criterion's 111 is stale" IS A MIS-ATTRIBUTION (rule, not staleness)
Criterion 3's 111 derives from the masterplan's own stated commands
(`grep -c '^## Cycle'` minus `grep -cE '^## Cycle [0-9]+'`). I ran BOTH trees:
    tree 692d5935 (masterplan's tree): 1215 - 1103 = 112
    HEAD                              : 1224 - 1112 = 112
The criterion's rule gives the SAME answer at both trees. The figure is NOT stale --
it is a DIFFERENT EXTRACTION RULE (prefix-numeric vs full-token-numeric). The 48-header
delta (160 - 112) is exactly the headers whose token STARTS with digits but is not all
digits: 36 parenthetical + 10 step-id + 2 other = 48. Verified arithmetic.
The author's 160 is a strict SUPERSET of the criterion's population, so criterion 3 is
substantively over-satisfied -- but the stated PROVENANCE ("stale") is wrong, and this
step's whole subject is measurement discipline.
FURTHER: the word "stale" and the number 111/112 appear NOWHERE in
experiment_results_86.44.md. The reconciliation was made only in the spawn prompt to me,
not in the durable artifact. A reader of the artifact is never told the criterion said
111 and why 160 is reported.

### B5. CRITERION 2 -- I RE-RAN THE GREP. The author's answer is RIGHT, and INCOMPLETE.
Confirmed: `scripts/smoketest/steps/finalize.py:70-72` does int() + max()+1. The gate's
"write-only state" headline is indeed refuted. Also confirmed the gate's two bad paths:
  backend/services/harness_state_reader.py -> ABSENT (real: backend/agents/...:143,149)
  scripts/harness/scheduler.py             -> ABSENT (real: backend/slack_bot/...:464)
MISSED CONSUMER (answering the author's direct question): the SAME file reads the number
a SECOND time, as a content key --
  scripts/smoketest/steps/finalize.py:113
  new_block = log_path.read_text(...).split(f"## Cycle {append_info['cycle']}")[-1]
then :114-115 assert has_phase/has_result on that block. This is the site where D4's
collision acquires a CONSEQUENCE: under a duplicate number, `[-1]` returns the text after
the LAST occurrence, so a writer that raced to the same number makes finalize.py validate
ANOTHER writer's block as its own. The author's criterion-2 table omits :113.
This STRENGTHENS the author's case (and 86.55's) rather than refuting it.
Other sites my grep surfaced, all correctly non-number-reading or test fixtures:
  scripts/go_live_drills/smoke_test_4_17_1.py:59 (asserts `^## Cycle \d+` FORM on new text)
  scripts/harness/build_evaluator_critique.py:58,67 (writes `## Cycle {n}` into a
  DIFFERENT file, evaluator_critique.md -- producer, not consumer of this log)

### B6. LINT GATE -- ruff exit=1, ONE NEW F401 IN A FILE THIS STEP ADDED
Scope DERIVED (git diff ea5b1cd5^..HEAD -- '*.py' UNION git ls-files --others), 4 files,
non-empty guard passed, passed via xargs -0 (no zsh word-split).
  F401 [*] `subprocess` imported but unused
    --> scripts/qa/mutation_matrix_86_44.py:19:8
  Found 1 error.   ruff_exit=1
NOT pre-existing: mutation_matrix_86_44.py is NEW in fe9a6dad. qa.md section 1a states
"Non-zero exit = FAIL". Dead import in a QA instrument, not production code -- material
severity is low, but the gate is red and the gate is mine to honor.
AST parse OK on all 4. Runtime smoke OK: `import backend.api.backtest` clean;
run_harness.py exec_module clean, append_harness_log callable.

### B7. D2 INDEPENDENTLY REPRODUCED IN MEMORY (tree untouched)
  headers on disk 1224 | OLD regex 1064 | NEW regex 1224 | delta +160 | lossless True
  recovered names sane ('Cycle 30 (continued)', 'Cycle 4.15.3', ...), none multi-line,
  none >80 chars -- the non-greedy `[^\n]+?` does not over-capture.

### B8. D1 INDEPENDENTLY REPRODUCED WITH A DIFFERENTLY-CONSTRUCTED MUTANT
I did NOT run the author's matrix (it edits production files; I am read-only). Instead I
built a SECOND mutant of different construction -- an in-process re-creation of the
pre-fix shape, first asserting the shape really exists in ea5b1cd5:
  'HARNESS_LOG.write_text(existing + entry' present in ea5b1cd5 -> CONFIRMED
  CONTROL  (committed O_APPEND): seed 1064 cycles / 2,909,762 bytes -> +72 of 72  GREEN
  MUTANT   (read-modify-write) : same seed                        -> +8  of 72  KILLED
So the kill is NOT a construction artifact: two independently-built mutants both die.
Magnitude differs from the author's -1033 by construction (my entry body is shorter, so
less is clobbered per write); direction and conclusion identical.
DISCLOSURE-4 CHECK PASSES: the matrix seed at line 80 is `HARNESS_LOG.read_text()` --
the REAL 2.9 MB / 1064-cycle log, not a 14-byte stub. `_seeded` = 1064 confirmed.

### B9. TOCTOU PROBE -- BARRIER IS LEGITIMATE. Reproduced twice.
Ran the shipped probe (writes only to a tempdir):
  run 1: 5 distinct of 16, 11 collisions, 17 rows (all data survived)
  run 2: 3 distinct of 16, 13 collisions, 17 rows
Author reported 10 collisions; I get 11 and 13. Stable IN KIND, varying in magnitude --
the signature of a real timing-dependent race, not a fabricated one.
The barrier is at :51-52, BEFORE the call at :53; the production function is unmodified
and un-monkeypatched. It removes process-startup skew, it does not widen the window.
Decisive counter-evidence against "manufactured": a barrier that manufactured the defect
would collapse all 16 to one number. It does not (5 and 3 distinct remain).
ANSWER TO THE AUTHOR'S Q3: the barrier is legitimate. Caveat already disclosed by the
author in section 10 -- it establishes the window EXISTS and is losable, not the
production hit-rate.

### B10. FINDING -- "D4 IS THE MECHANISM BEHIND THE 141 DUPLICATE INTEGERS" IS
###       OVERGENERALIZED, AND THE ARTIFACT CONTRADICTS ITSELF ON IT
experiment_results section 5: "THIS is the mechanism behind the duplicate integers in
history"; section 6 reason 3: "141 duplicated integers are evidence of D4".
But section 1 of the SAME artifact says: "The 481 have one mechanical cause:
run_harness.py:953 passes the loop index as cycle".
I measured the split:
  dup-group headers total          : 969
  of which token '1'               : 481  = 49.6%   <- loop-index mechanism, NOT a race
  remaining                        : 488 across 140 integers
  top dup integers: 1x481, then 2x8, 3x6, 4x6, 5x6, 34x6, 35x6 ... 44x6
  (the flat runs of 6 across consecutive integers are the signature of repeated
   `--cycles N` harness restarts, i.e. the loop index again -- not a TOCTOU)
I also verified run_harness.py:1123 `for cycle in range(1, args.cycles + 1)` feeding the
call sites at :1149 and :1196 -- the loop-index claim is CORRECT. (Minor: the contract
and results cite ":953", which is the `def append_harness_log(...)` line, not the loop.)
AND I tested whether finalize.py even writes the real file: it does, but only 3 times
(harness_log.md:3230, :3234, :26723 carry its 'aggregate smoketest finalize' signature),
against 1,224 headers. masterplan.json:1055 does wire it to `--log handoff/harness_log.md`,
so the writer is live -- but its footprint here is 3 entries.
=> At least THREE distinct mechanisms produce duplicate integers (loop index; finalize.py
TOCTOU; two sessions hand-numbering independently, which the masterplan itself documents
for `Cycle 1211`). Attributing the 141 to D4 singularly is not supported by the author's
own data. The DECISION (do not renumber) survives on reasons 1 and 2, both of which I
verified independently; reason 3 as worded does not.
Note the 86.55 masterplan entry uses the softer, defensible "Corroborating evidence in
history", so the queued step is not built on the wrong premise -- but experiment_results
sections 5 and 6 are.

### B11. FINDING -- A WRONG FIGURE SHIPPED INTO A PRODUCTION DOC, AT THE STEP'S OWN TREE
docs/runbooks/per-step-protocol.md:335 (added by this commit):
  "**59 headers in `harness_log.md` literally read `Cycle N`, `Cycle N+1` ... `Cycle N+58`**
   (measured phase-86.44 at tree `915d2cb0`)"
I measure 58 at exactly that tree. Root cause pinned: the range is NOT contiguous --
  occurrences of N / N+k : 58 | distinct: 58 | bare 'N': 1 | k range 1..58
  MISSING k in 1..58     : [23]      <- N+23 does not exist
so `N+1 ... N+58` has 57 members, + 1 bare `N` = 58. The author inferred 59 from the
endpoints instead of counting. It CONTRADICTS this step's own experiment_results section 3
table, which says 58. A number labelled "measured ... at tree 915d2cb0" that does not
reproduce at that tree is the exact failure criterion 1 exists to prevent, and it is now
in a durable runbook a future reader will trust.

### B12. FINDING -- D3 IS FIXED IN ONE OF AT LEAST TWO LIVE COPY SOURCES, AND ITS
###        GUARD CANNOT SEE THE OTHER (section 4c vacuity shape 2)
CLAUDE.md:223 still carries the literal:
  | LOG | appended block in `handoff/harness_log.md` | `## Cycle N -- YYYY-MM-DD -- phase=X.Y result=PASS/CONDITIONAL/FAIL` header + summary |
CLAUDE.md is auto-loaded into EVERY session's context, so it is a MORE likely copy-paste
source than a runbook that must be opened. check_d3_runbook_placeholder() scans only
RUNBOOK, so the guard cannot fail on this file -- its population is 1 file while the trap
class spans at least 2 live files. (Other hits are correctly out of scope: PLAN.md:245/255/265
are different headers; harness_log_gate.py:22 is a docstring describing the format;
docs/audits/phase-24-.../24.0-charter-findings.md:92 is a dated audit -- do not rewrite.)
D3 is author-added scope, not an immutable criterion, so this caps rather than fails.

### B13. FINDING -- THE D2 FIX IS COMMITTED BUT NOT IN FORCE, AND THAT IS UNDISCLOSED
  running backend pid 66306 started : 2026-08-10 21:33:01  (~20h BEFORE the fix)
  fix commit fe9a6dad committed at  : 2026-08-11T17:13:05+02:00
  LIVE GET /api/backtest/harness/log: http 200, 323,150 bytes, cycles = 1064
  fixed code (verified in memory)   : 1224
  frontend/src/components/HarnessDashboard.tsx:209 -> getHarnessLog() -> that endpoint
So the Harness tab is STILL mis-attributing the 160 headers right now. experiment_results
section 4 says "FIXED ... the parser now returns 1,224 of 1,224" and section 10 ("What is
NOT claimed") omits the not-in-force state. True of the code, not of the running system.
Restarts batch to session end per operator instruction -- so the remedy is a
pending-restart entry + a disclosure, not a restart now.
NOTE on qa.md 1c: this step makes UI claims ("the Harness tab showed text under the wrong
cycle") and NO Playwright capture was taken by either party. I substituted a LIVE API
measurement of the exact endpoint the tab consumes, which is stronger evidence about the
running system than a screenshot -- and it shows the UI is still wrong. Recorded as a gap.

### B14. REMAINING DETERMINISTIC CHECKS
  scoped pytest: `-k "harness or backtest or hook_gate or digest"` -> 136 passed,
    1 skipped, 3315 deselected, 30.29s. GREEN.
  research gate: brief_status COMPLETE, 8 sources read in full (floor 5), 16 URLs
    (floor 10), recency_scan_performed true, 12 internal files. CLEAN.
  the gate's two bad paths independently confirmed: backend/services/harness_state_reader.py
    ABSENT, scripts/harness/scheduler.py ABSENT; backend/agents/harness_state_reader.py and
    backend/slack_bot/scheduler.py EXIST. The author's correction is accurate.
  live_check_86.44.md: NOT YET PRESENT. Expected pre-flip (the gate fires at status flip);
    owed before the masterplan flip, not a violation now.

## CRITERION ROLL-UP
  1 MET   -- every figure reproduced under two independent rules; 481/482 disclosed
             (and resolvable: 481 is correct; harness_log.md:25421 is the differing line)
  2 MET   -- grep run, answer stated, gate's headline correctly OVERTURNED.
             Incomplete: finalize.py:113 is a 2nd read (split key), omitted.
  3 MET   -- 160 characterised into 5 classes, all distinct formats; arithmetic verified
             (160-112=48=36+10+2). Provenance claim ("stale") is WRONG: rule difference.
  4 MET   -- decision stated with reasons; reasons 1-2 verified, reason 3 overgeneralized.
  5 MET   -- and it is an HONEST reading, not a dodge. Criterion 5 is conditional on a NEW
             numbering; none was introduced, so a staged proof would be the vacuous guard
             section 4c forbids. The author instead proved the WRITE is concurrency-safe
             (I reproduced: control 72/72, mutant loses data) and demonstrated the EXISTING
             numbering is not unique (I reproduced: 11 and 13 collisions of 16) + filed 86.55.
  6 MET   -- control-gated, 3 cells, all killed, byte-identical restores, post-restore green;
             M1 corroborated by my differently-constructed 2nd mutant, M2 in-memory.

## VERDICT DIRECTION
CONDITIONAL. No criterion unmet; the three fixes are real and two are independently
verified behaviorally. Capping items: (a) ruff gate RED on new code, (b) a wrong figure
shipped into the runbook at the step's own tree, (c) D3 incomplete + scope-narrow guard,
(d) D2 committed-not-in-force undisclosed, (e) D4 sole-cause overgeneralization,
(f) missed 2nd consumer. Cycle 1: 0 prior CONDITIONALs, so 3rd-CONDITIONAL auto-FAIL
does not apply. Harness compliance clean on all 5 items.

COMPLETED: 2026-08-11T15:26:40Z
