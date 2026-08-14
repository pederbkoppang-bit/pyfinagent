STATUS: INCOMPLETE -- not a verdict
STEP: 86.79
WRITTEN: 2026-08-14T10:10:30Z

# Q/A write-first record -- step 86.79 (cycle 4 evidence)

Spawn: Workflow rail, Opus 5 (1M). Read qa.md in full at 2026-08-14T10:10:30Z.

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command, git status/diff scope, ruff lint, verifier + mutation matrix re-run
C. LLM judgment vs 7 immutable criteria; independent mutations (not the author's matrix)

## Findings (appended as established)
### Deterministic (reproduced by me, not transcribed)
- IMMUTABLE cmd `ast.parse(qa_wip.py)` -> `qa_wip-parses`, exit=0. MET.
- `verify_counter_86_79.py` -> exit 0, **55** checks (grep-counted [PASS] lines = 55), 0 failed. Reproduces.
- `mutation_matrix_86_79.py` -> exit 0, 11 cells, 11 KILLED, control GREEN first,
  subject sha256[:16] before==after==146600b722a02481. Reproduces (needs >2min; timed out at 120s first).
- `qa_wip.py 86.79 --spawned-at 2026-08-14T10:10:30Z` -> attempt_number=3, prior_attempts=2,
  records_retained=3, source_present=true, attempt_number_status=ok, is_verdict=false,
  attempt_number_is_lower_bound=true, records_pruned_known=null.
- `verdict_history_86_21.py --step 86.79 --evidence-only` -> status=no_rows_for_step, verdicts=(none).
  => sequence: UNKNOWN. attempt_number(3) > ledger count(0) => LEDGER IS STALE for this step.
- git: working tree has NO uncommitted production changes; all step artifacts committed
  (HEAD=e14d3248 / 61e359b4). `git diff --name-only HEAD` = 5 audit/memory jsonl+md only.

### Harness compliance (mtime chain, UTC via date -u -r stat -f%m)
research_brief 07:05:37 < contract 07:14:15 < qa_wip.py 07:18:56  -> research<contract<artifact OK
qa-verdict.js 09:23:47 ; qa.md 10:00:02 ; experiment_results 10:09:54 (cycle-4 fixes, post-critique 08:54:57)

### MY OWN MUTATION BATTERY (6 cells, not the author's matrix) -- 2 SURVIVORS
Harness: mutant COPY of qa_wip.py in a temp dir, fed through PYFIN_QA_WIP_OVERRIDE
to verify_counter_86_79.py. Subject sha256[:16]=146600b722a02481 before AND after.
Anchor uniqueness asserted (count==1) for every cell.

| cell | result | evidence |
|---|---|---|
| X1 records_pruned_known = `lost if lost is not None else 0` | **SURVIVED** exit=0, 0 fails | differential: `records_pruned_known` None -> **0** |
| X2 `total = max(prior, int(n_lost))` (no accumulation) | KILLED | "...and RISES when more is destroyed -- lost=3" |
| X3 identity `w >= spawn_dt` -> `w > spawn_dt` | KILLED (4 fails) | attempt_number=None everywhere |
| X4 attempt_number EXCLUSIVE (`prior_attempts`) | KILLED (4 fails) | attempt_number=2 vs 3 |
| X5 `lost_n = 0` always | KILLED (6 fails) | add-back dead on both branches |
| X6 blank `_ATTEMPT_GUIDANCE["ok"]` | **SURVIVED** exit=0, 0 fails | differential: guidance '' on the OK path |

X1 = criterion 6's own words ("a counter that cannot be computed must FAIL CLOSED
rather than report 0") applied to the THIRD field. Guarded on attempt_number /
prior_attempts (C6 + M4); NOT guarded on records_pruned_known. grep: 0 assertions
that records_pruned_known is None.
X6 = F1's "the unit travels WITH the number" (qa_wip.py:508-512). Guarded ONLY on the
no_record_for_this_spawn guidance (checker :150-152). The OK path -- the one every
healthy spawn reads -- has none. C1's label "the NEW attempt_number is the same
number, but unit-stated" asserts only the two integers, not the unit.

### CRITERION-4 GUARD CENSUS (grep-derived)
verify_counter_86_79.py mentions qa.md at :378/:379/:381 ONLY, and asserts merely that
`qa_md_patch_86.79.md` EXISTS and contains "records_retained". Zero assertions on
.claude/agents/qa.md or .claude/workflows/qa-verdict.js content.
mutation_matrix_86_79.py SUBJECT = scripts/qa/qa_wip.py (single file).
=> members 4b (qa-verdict.js) and 4c (qa.md) can be reverted with the gate STILL GREEN.
=> the C4 assertion LABEL "the un-applied qa.md correction is written out for the
   operator" and the comment "The residual divergence (qa.md) must be LOUD" are now
   STALE: the correction WAS applied at cycle 4.

### FACTS VERIFIED INDEPENDENTLY
- 4a: DEFAULT_KEEP comment states TOTAL/INCLUSIVE; measured retention == keep. MET.
- 4b: qa-verdict.js :176-180 now say attempt_number/prior_attempts, "null is NEVER 0",
  "records_retained is NOT the attempt number ... a gauge, not a counter". MET.
- 4c: qa.md :622 region + :645 region corrected (read current file + git show 9b4d5281).
  Diff contains 0 lines matching `records_retained: 0` or `qa_wip.py 86.33` -- the two
  deliberately-untouched sites. Claim REPRODUCES.
- (B) both exclusions judged CORRECT: :754 is still literally true (code emits the same
  phrasing at qa_wip.py:539); :784's dated measurement STILL REPRODUCES TODAY --
  `qa_wip.py 86.33` -> records_retained 3, source_present true, 2 prior_records.
- ruff F821,F401,F811 over the step's derived commit-range scope (5 .py files,
  non-empty asserted, xargs -0): "All checks passed!" exit 0.
- enforceEscalation: would_auto_fail only when verdict==='CONDITIONAL'; consecutive
  null (not 0) on not_supplied/unusable/unparseable; budget_exhausted only when
  typeof attempt_number === 'number'. No PASS-producing path.
