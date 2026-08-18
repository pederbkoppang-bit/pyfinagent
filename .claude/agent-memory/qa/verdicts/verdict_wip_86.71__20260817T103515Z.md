STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.71
WRITTEN: 2026-08-17T10:35:15Z

# Q/A write-first record -- step 86.71 (cumulative attempt budget wiring)

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command, git status/diff scope, lint, tests, runtime smoke
C. LLM judgment vs the 8 immutable criteria read VERBATIM from .claude/masterplan.json
D. Independent mutation testing of the new guards (criterion 8) -- I mutate, not just read the author's matrix

## Findings log (appended as established)

### Context
- HEAD moved MID-EVALUATION: the graded work is commit `192ef652` ("phase-86.71: the
  cumulative attempt budget is WIRED AND LIVE"), and `65e38c3b` (86.96+86.90+86.72)
  landed after it. 192ef652 scope = 10 files, all 86.71: settings.json(+11),
  attempt_budget_audit.jsonl(+7), contract/experiment/live_check/research_brief,
  attempt_budget.py(14/-4 docstring only), attempt_gate.py(new 377),
  mutation_matrix_86_71.py(new 162). No unintended production change in that commit.
- qa_wip 86.71: attempt_number=1, prior_attempts=0, source_present=true.
  verdict_history_86_21 --evidence-only: status=no_rows_for_step, verdicts=(none).
  Cross-check consistent (prior_attempts 0 == ledger 0). Cycle 1.
- harness_log: `grep -c "phase=86.71"` = 0. masterplan status = pending. LOG-LAST OK.

### Deterministic
- IMMUTABLE COMMAND: `parses`, exit=0. REPRODUCED.
- ruff F821,F401,F811 over the commit's 3 changed .py (scope derived from
  `git diff --name-only 192ef652^ 192ef652 -- '*.py'`, non-empty asserted):
  "All checks passed!" exit=0.
- pytest backend/tests/test_phase_86_32_attempt_budget.py -q => 15 passed.
- `attempt_gate.py --self-test` => 12 checks ok, SELF-TEST PASSED, exit 0. REPRODUCED.
- `mutation_matrix_86_71.py --verify` => 6/6 KILLED, VERIFY: PASS, exit 0. REPRODUCED
  as text -- but see FINDING 1: the kills are artifacts.

### FINDING 1 (BLOCKING, criterion 8) -- the mutation matrix is NON-DISCRIMINATING
`mutation_matrix_86_71.py` writes each mutant to a tempdir and runs it as a
subprocess. `attempt_gate.py` line 74-79 does
`sys.path.insert(0, REPO/"scripts"/"harness"); from attempt_budget import ...`
where `REPO = Path(__file__).resolve().parents[2]`. Relocated to the tempdir REPO
becomes `/private/var/folders/n4/9khkbgzj...`, so the import FAILS.

MEASURED by me:
- UNMUTATED copy of attempt_gate.py at the matrix's temp path:
  `rc=1`, stderr last line `ModuleNotFoundError: No module named 'attempt_budget'`,
  rows_after=0, no escalation.
- NULL MUTANT (a pure comment: `def _now() -> str:  # inert null-mutant marker`)
  scored **KILLED**, failing 5 of the 6 CHECKS.
- Real cell G1's mutant stderr last line is also `ModuleNotFoundError`.
- TELL IN THE AUTHOR'S OWN OUTPUT: all six cells report the SAME kill reason
  `by: below-ceiling launch is ALLOWED`, including G1/G5/G6 which touch only the
  AT-CEILING path and cannot change below-ceiling behaviour. Those six `by:` lines,
  plus the two header lines and a blank, are ABSENT from the block pasted in
  live_check §5 -- the block is condensed, not verbatim, and the removed lines are
  exactly the ones that expose the defect.
Conclusion: `6/6 KILLED` measures relocation, not mutation. Criterion 8's
"revert it and show the check goes red" did not happen: the check went red for the
wrong reason (vacuity shape #11, mis-attributed kill mechanism). The matrix is the
SOLE mutation coverage for the new gate => sole-coverage vacuity.

### FINDING 2 (real survivor, found only under a REPAIRED harness)
I re-ran the same 6 cells with `PYTHONPATH=<repo>/scripts/harness:<repo>/scripts/qa`
so the relocated copy imports. Control SURVIVED (correct), NULL MUTANT SURVIVED
(the repaired harness discriminates), and:
  G1 KILLED  by "at-ceiling launch is DENIED with exit 2"
  G2 KILLED  by "below-ceiling launch is COUNTED"
  G3 KILLED  by "below-ceiling launch is COUNTED"
  G4 **SURVIVED** -- corrupt-row `rows.append({"step_id":"__corrupt__"})` -> `continue`
     is behaviourally identical on all 6 CHECKS, because `drive()` seeds only
     well-formed rows: the corrupt-row branch is NEVER EXERCISED by the matrix.
  G5 KILLED  by "at-ceiling launch is DENIED with exit 2"
  G6 KILLED  by "at-ceiling launch is DENIED with exit 2"
G4 is NOT an equivalent mutant: `--self-test`'s "corrupt row counts as an attempt"
check does go red under it. So G4's filed kill is credited to the matrix but is
actually made by a different artifact -- shape #11 again, at cell level.
Net: the PRODUCT looks sound (5/6 genuine kills + self-test covers the 6th); the
EVIDENCE for criterion 8 is invalid as filed.
Interpreter-independent: reproduced under /usr/bin/python3 AND .venv/bin/python.

### FINDING 3 (Invalid_Precondition) -- live_check §5 block is CONDENSED, not verbatim
The committed matrix (mtime 12:10:52, BEFORE live_check 12:34:36) prints, at
`mutation_matrix_86_71.py:107,108,142`, a header line, a `"=" * 78` rule and an
unconditional `by: {failed[0]}` under every KILLED cell. The block pasted at
live_check lines 96-108 omits the header, the rule, all six `by:` lines and the
md5 tail of the restore line. The omitted `by:` lines are exactly the ones that
show all six cells dying to the same below-ceiling check.

### FINDING 4 (criterion 1) -- the disagreement EXPLANATION does not reproduce
live_check §1 / experiment_results: "the corpus grew from 513 to 580+ records
between measurements". Measured by me, applying MAIN'S OWN rule to the OLDEST N
records by timestamp:
  ALL 589 today          with_sid=491 repeats=326 = 66.4%   (reproduces Main's 66.5%)
  OLDEST 527 records     with_sid=429 repeats=278 = 64.8%
  OLDEST 513 records     with_sid=417 repeats=270 = 64.7%
  before 2026-08-13      with_sid=391 repeats=252 = 64.5%
So corpus growth explains ~1.7 of the ~8.1-point gap to 58.4%; ~6.4 points are
attributable to the POPULATION-RULE change (audit_basis: journal.jsonl dirs +
"masterplan step <id>" transcript regex, 459/527; Main: wf_*.json + args.step_id,
481/580). Per-role reproduced: qa 307/396=77.5% (filed 77.4%), researcher
18/89=20.2% (filed 20.0%), max 9 qa runs on 36.8.
ALSO: live_check §1 carries NO command (`grep -n '^\$ '` finds commands at lines
26,29,56,66,78,97,119,121,122 -- none in §1), and the qa/researcher CLASSIFIER is
not stated at all. Criterion 1 asks for "the population rule AND the command
stated next to each number".

### FINDING 5 (scope honesty, WARN) -- "every Layer-3 run originates" is false
`attempt_gate.py:7` and `experiment_results_86.71.md:13` both say the Workflow
tool is "the seam where every Layer-3 run originates". MEASURED:
- session transcripts: 42 `"subagent_type":"qa"` and 44 `"subagent_type":"researcher"`
  Agent-tool spawns (grep -oh over ~/.claude/projects/<proj>/*.jsonl).
- pre_tool_use_audit tool histogram: **Agent 1,226 rows vs Workflow 663 rows**.
The gate's matcher is `Workflow` ONLY (settings.json PreToolUse matchers are
[None(danger), 'Workflow', 'Write|Edit']). CLAUDE.md documents `Agent(subagent_type:'qa')`
as the FALLBACK to use when the Workflow rail errors/returns empty -- i.e. the
ungated path is the documented next move after exactly the drops this step bounds.
"Honest limits" lists three bounds and omits this one.

### Criteria 2-7: independently verified MET
- C2 MET: control `grep -rln "attempt_budget" backend/tests/` -> hits
  test_phase_86_32_attempt_budget.py (non-zero); runtime surfaces (scripts/harness,
  backend, .claude/hooks, .claude/workflows minus module/gate/tests) -> zero.
  Cross-checked at `192ef652^`: only the test, mutation_matrix_86_32.py,
  verify_counter_86_79.py, masterplan and WIP records. No runtime caller.
- C3 MET (I drove it): 6 SEPARATE OS process invocations against a scratch ledger
  -> rc=0 rows 1..5, 6th rc=2 DENIED; a separate `--status 55.5` process reads
  attempts_used=5 / ESCALATE / next_launch=deny. Real ledger byte-unchanged.
  Also exercised the `is_relative_to` fix (escalation dir outside the repo).
- C4 MET: `jq -e` on settings.json returns the exact registered command; the real
  ledger carries hook-written rows for 86.85 and 86.71 with this session's id;
  `--status 999.2` = 5/5 ESCALATE deny, `--status 86.71`/`86.85` = 1/5 CONTINUE
  allow; escalation_attempt_budget_999.2.md exists, contains no verdict, states
  "THIS IS NOT A PASS AND NOT A FAIL". (Scope caveat = FINDING 5.)
- C5 MET: exhaustive probe -- 1,452 (non-PASS sequence x 4 flag combos)
  evaluations, ZERO reach CLOSED_PASS/CLOSED_COMPLETE/CLOSED_PRODUCT_RESIDUALS.
  5xFAIL -> ESCALATE under all four flag combos. Gate write sinks are only the
  append-only attempt ledger + the escalation md; `emit_sequence` does not write.
- C6 MET: both hook-written rows are `workflow: qa-verdict.js`. (Ceiling is shared
  with research-gate launches -- disclosed in contract plan step 2.)
- C7 MET: no .env in commit 192ef652; ASK-1 present in contract_86.71.md.

### NOTEs
- live_check §6's `$ python3 scripts/qa/mutation_matrix_86_32.py  # exit 0` does
  NOT reproduce as written: /usr/bin/python3 has no pytest, `run_suite()` returns
  rc!=0 with an EMPTY failure list, and the matrix ABORTS "control is RED ([])".
  Under `.venv/bin/python` it is green, 8/8 KILLED -- claim true, command not
  reproducible (vacuity shape #9, executor-environment).
- PASS exception is permanent per step. 8 step-ids carry a PASS in
  handoff/verdict_ledger.jsonl; one (86.74) is still `pending` -> un-budgeted.
- Key-based evasion: with 55.5 at its ceiling (rc=2), `55.50` and `55.5.0` are
  distinct keys and return rc=0. Consistent with "audited, not unforgeable" but
  not listed as an evasion surface.
- Contract plan step 4 names synthetic step "999.1"; artifacts + committed ledger
  use "999.2". Immaterial.
- 5 `session_id: pipetest` rows for synthetic step 999.2 are now COMMITTED into
  the production audit stream, permanently denying that id.

### Harness compliance: CLEAN
research_brief 11:47:50 < contract 12:07:27 < matrix 12:10:52 < gate 12:32:08 <
experiment_results 12:34:07 < live_check 12:34:36. Gate run wf_77c2679f-de9
status=completed, ENFORCED gate_passed=true, self_report_disagreed=false,
sources 9>=5, urls 30>=10, recency ok, brief COMPLETE (33,784 chars), 9/9 claimed
URLs present in the brief. harness_log `phase=86.71` rows = 0; masterplan status
= pending. Cycle 1 (qa_wip attempt_number=1, prior_attempts=0, source_present=true;
ledger status=no_rows_for_step) -- no verdict shopping possible.

### Scope: no unintended production change
Graded commit 192ef652 = 10 files, all 86.71 (settings.json +11, attempt_budget_
audit.jsonl +7, 4 handoff artifacts, attempt_budget.py docstring-only 14/-4,
attempt_gate.py new, mutation_matrix_86_71.py new). No .env, no frontend/**, no
backend/** source. The dirty sovereign-UI files + perf_results.tsv are a peer
session's and are NOT in this commit. NOTE: HEAD moved during the evaluation --
65e38c3b (86.96+86.90+86.72) landed after 192ef652 and touches no 86.71 file.

### VERDICT: FAIL
Criterion 8 is materially unaddressed: its sole evidence scores a NULL mutant as
KILLED, so no mutation demonstration exists for any new guard, and re-running it
correctly exposes a survivor (G4) the filed matrix reported as killed.
Compounding: the criterion-1 disagreement explanation does not reproduce, the
"every Layer-3 run" claim is an overgeneralization with the documented fallback
path ungated and undisclosed, and the §5 capture is condensed exactly where the
defect would have shown. The PRODUCT is close to sound (5/6 genuine kills,
criteria 2-7 all reproduce); the EVIDENCE is not.

COMPLETED: 2026-08-17T10:46:37Z

