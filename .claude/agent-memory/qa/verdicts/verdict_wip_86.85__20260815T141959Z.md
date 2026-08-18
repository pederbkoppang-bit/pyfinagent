STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.85
WRITTEN: 2026-08-15T14:19:59Z

# Q/A write-first record -- step 86.85 (EVALUATE)

Role: Layer-3 Q/A (merged qa-evaluator + harness-verifier), Workflow rail.
Read `.claude/agents/qa.md` in full at 14:19:59Z (runtime read, per STEP 0).

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command exit code + git scope + lint + tests
C. LLM judgment against the 8 immutable criteria

## Findings (appended as established)

### Prior-attempt evidence
- `qa_wip.py 86.85 --spawned-at 2026-08-15T14:19:59Z`: source_present=true,
  attempt_number=3 (status ok, is_lower_bound=true), prior_attempts=2,
  records_retained=3 (gauge, incl. my own).
- `verdict_history_86_21.py --step 86.85 --evidence-only`: status=ok,
  detail="2 verdict(s) from the ledger", verdicts = FAIL -> FAIL.
- Cross-check: prior_attempts (2) == ledger verdict count (2). Ledger is NOT
  stale for this step. (attempt_number 3 > 2 is expected: it includes me, and I
  have not returned a verdict yet.)

### B. Deterministic
- IMMUTABLE COMMAND: `bash -c 'source .venv/bin/activate && python -c "import
  ast; ast.parse(open(\"scripts/qa/verdict_history_86_21.py\").read());
  print(\"parses\")"'` -> `parses`, **exit 0**. GREEN.
- Scope derived: `git diff --name-only d1c4a79d~1 HEAD -- '*.py'` =
  backend/tests/test_phase_86_85_verdict_ledger_write.py,
  scripts/qa/mutation_matrix_86_85.py, scripts/qa/verdict_ledger_write.py.
  Non-empty set asserted before reading exit code.
- ruff F821,F401,F811 over that set (xargs, not unquoted var): "All checks
  passed!", exit 0. GREEN.
- pytest backend/tests/test_phase_86_85_verdict_ledger_write.py -q:
  **25 passed**, exit 0. REPRODUCES the claim.
- mutation matrix `scripts/qa/mutation_matrix_86_85.py`: CONTROL rc=0 GREEN
  first; **10 cells, 10 killed, 0 survived, 0 unscorable**; sha256
  e31eaf8e...68ca identical before/after, UNCHANGED: True. REPRODUCES.
- No uncommitted changes under handoff/current, scripts/qa, backend/tests
  (`git status --short` on those paths is empty). Graded tree == HEAD 3ae269de
  (work commit 39999944).

### FINDING 1 (does not reproduce): self-test is 18 checks, artifact says 19
- `python scripts/qa/verdict_ledger_write.py --self-test` -> SELF-TEST PASSED,
  exit 0, but the check lines count **18**, not 19:
  `--self-test 2>/dev/null | grep -cE '^  (ok  |FAIL)'` -> **18**.
- The cycle-2 baseline DOES reproduce: same command on
  `git show 5a3b0766:scripts/qa/verdict_ledger_write.py` -> **13**.
- So "13 -> 19 checks" is really 13 -> 18, and the gate block
  `self-test : 19/19 ok, exit 0` is unreproducible as written.
- Sites: experiment_results_86.85.md:389, :434 and
  evaluator_critique_86.85.md:176.

### FINDING 2 (BLOCKING): a REAL surviving mutant on a LIVE branch -- `_dedup_key` cycle fallback
Independent mutation run (temp copies, repo byte-identical, CONTROL rc=0 GREEN
first, 18 checks). Anchors unique; sha256 e31eaf8e... before AND after.

| my cell | mutation | self-test | pytest(25) | matrix(10) |
|---|---|---|---|---|
| QA-M1 | delete the `cycle` fallback branch of `_dedup_key` | SURVIVED | SURVIVED | n/a |
| QA-M2 | fallback ignores the cycle VALUE: `(step,"cycle:X")` | **SURVIVED** | **SURVIVED** | n/a |
| QA-M3 | drop main()'s `--emit-sequence requires --step` | KILLED | - | - |
| QA-M4 | drop main()'s `--step/--verdict required` | SURVIVED | SURVIVED | (equivalent, see below) |
| QA-M5 | drop read_rows' blank-line skip | SURVIVED | SURVIVED | (no live counterexample) |
| QA-M7 | drop build_row's pre-flight `_dedup_key(row)` | KILLED | - | - |

BEHAVIOURAL DIFFERENTIAL for QA-M2, driven as a literal replay of the REAL 86.74
backfill shape (4 run_id-less rows, same step):
  BASELINE : exits 0,0,0,0 -> 4 rows -> emit_sequence
             ["NO_VERDICT","NO_VERDICT","CONDITIONAL","CONDITIONAL"]
  QA-M2    : exits 0,2,2,2 -> 1 row  -> emit_sequence ["NO_VERDICT"]
Three rows LOST, and lost under EXIT_DUPLICATE(2) -- the benign "already
recorded" code a caller ignores. Two CONDITIONALs vanish from the sequence that
feeds enforceEscalation: the UNDER-COUNT / fail-OPEN direction, which is the
exact materiality the author used to justify his own M9.

LIVE, NOT HYPOTHETICAL: `handoff/verdict_ledger.jsonl` has **5 of 45 rows with no
run_id** -- 86.74 cycles `1-drop-a`, `1-drop-b`, `3`, `3b`, `4` -- i.e. 4 of the 8
rows of the very step this work exists to fix are keyed by the uncovered branch.

CONTRADICTS the artifact's own class-level claim (experiment_results §6):
"I enumerated every guard from source -- all 9 `raise LedgerError` sites plus
every distinguishing branch of `_dedup_key`". The cycle branch is a distinguishing
branch of `_dedup_key` and has no self-test check, no pytest test and no matrix
cell. Known-member recall test FAILS.

QA-M4 CLEARED as an EQUIVALENT mutant (differential run): all three arg shapes
still exit 3 (via the vocabulary / empty-step guards), only the message differs.
But that means the self-test check named "append without --verdict exits 3" is
NOT testing the guard it names -- mis-attributed kill mechanism (vacuity shape
#11). WARN, not blocking.
QA-M5 low materiality: the real ledger has 45 raw == 45 non-blank lines, so no
live counterexample. NOTE.

### PROBE DEFECT I MADE, recorded not hidden
`for c in ...; do git show "$c:path"; done` in zsh: `$c:s...` is parsed as the
zsh `:s` (substitute) HISTORY MODIFIER, so the path is eaten and git shows the
COMMIT DIFF instead of the blob (1432 bytes, not 22608). That produced a bogus
"sha mismatch". With `"${c}:path"` every sha256 claim REPRODUCES exactly:
d1c4a79d=146cf84e (matches §2 C8), 5a3b0766=2f0d1000 (matches live_check §6),
HEAD=e31eaf8e (matches my matrix run). No finding; my probe was wrong.

### Criteria that DO reproduce (verified independently)
- C1: positive control `--step 86.21` -> status=ok, 5 verdicts; 86.74 pre-step 0
  rows; max date 2026-08-11; recorded_by {main:35}; pre-state dist
  {C 18, F 5, P 7, NV 5} and 10814 bytes -- all EXACT.
- C2: 35 / 10 step_ids / 0 for 86.74 / run_id 35,35,35 at d1c4a79d~1;
  43 / 11 / 8 / {C23,F5,P8,NV7} / recorded_at 29-of-43 at d1c4a79d;
  45 / 12 / {C23,F7,P8,NV7} / 31-of-45 in the worktree. 45-31 = 14 = the stated
  "14 historical rows predate the field". ALL EXACT.
- C3: re-ran the two invocations myself -- write exit 0, separate read-back
  process returns ["CONDITIONAL"], exit 0.
- C4/C6/C7: extracted the REAL enforceEscalation (awk from `^function
  enforceEscalation` to `^}`, 52 lines, qa-verdict.js:319) and drove it. Every
  published row reproduces byte-for-byte, incl. 86.74's real priors -> n=2,
  auto_fail=true, and the [C,C,PASS] reversal -> n=0/false. C7 sweep: 3 verdicts
  x 7 sequences x 4 opts = 84 combos, ZERO verdict mutations, no `verdict`/`ok`
  key in the returned object.
- Cross-reader agreement: verdict_history_86_21.py and emit_sequence return
  IDENTICAL sequences for 86.74 (8), 86.21 (5), 86.85 (2). Symmetric difference
  empty.
- Scope: git diff d1c4a79d~1..HEAD touches 10 files, none production/trading.
  qa.md, verdict_history_86_21.py, qa-verdict.js and masterplan.json all UNTOUCHED.

### FINDING 3 (WARN): the gate artifact live_check_86.85.md was not updated in cycle 3
Commit 39999944 does not touch it (mtime 16:00 = cycle 2). So:
- §6 says "MUTATION MATRIX -- 7/7 killed ... Current state below" -- the
  delivered state is 10 cells. "Current" is now false.
- §8 "LEDGER STATE AFTER THIS STEP: total rows 43" is UNANCHORED and now 45 --
  the same self-referential drift the cycle-2 Q/A caught, fixed in
  experiment_results §2 C2 (every figure commit-anchored) but not in the sibling
  artifact the masterplan's `live_check` field actually names.

### Harness compliance (5 items)
1. research gate: research_brief_86.85.md first committed 9034ddfb 2026-08-14
   21:41, BEFORE contract (d1c4a79d 2026-08-15 15:44). Envelope:
   brief_status COMPLETE, gate_passed true, 8 read in full (>=5), 23 urls
   (>=10; I counted 23 unique http(s) URLs myself -- exact), recency scan
   performed with a dedicated section. GREEN.
2. contract-before-generate: research 08-14 21:41 < contract 08-15 15:44 <=
   artifacts in the same commit. GREEN.
3. experiment_results present: yes, 22,281 bytes. GREEN.
4. log-last: `grep -F "phase=86.85" handoff/harness_log.md` -> 0 rows;
   masterplan 86.85 status = pending. GREEN.
5. no-verdict-shopping: evidence CHANGED between spawns -- 39999944 adds
   backend/tests/test_phase_86_85_verdict_ledger_write.py (+233), writer +52,
   matrix +25. GREEN.

### FINDING 4 (WARN): a cycle-2 remediation item was only HALF discharged
The cycle-2 return's remediation item (2) named BOTH files: "Anchor the headline
count blocks -- 'as of d1c4a79d' -- in experiment_results §2 AND live_check §8".
Only §2 was anchored. Commit 39999944 does not touch live_check_86.85.md at all.
§6 C2 of experiment_results then reports the fix as "Every figure in §2 C2 now
names the commit it was taken at" -- a silent narrowing of the scope the critique
named. (Cycle-2 items 1, 4 and 5 ARE fully discharged: M8/M9 verified KILLED; the
brief's 33/35 is REPLACED at :29/:126/:182 with the population rule + command and
the :115 note kept as history; the promised pytest file exists with 25 passing
tests.)

### Uncommitted production files -- NOT attributable to this step
backend/api/sovereign_api.py + 5 frontend components, mtimes 2026-08-14
13:12-13:35, i.e. before this step's 2026-08-15 15:44 window. Same NOTE TO MAIN
as cycles 1 and 2: auto-commit-and-push.sh runs `git add -A` on the status flip
and will sweep all six under this step's name.

## CRITERION ROLL-UP
C1 MET | C2 NOT MET | C3 MET | C4 MET | C5 MET | C6 MET | C7 MET | C8 NOT MET

## VERDICT DIRECTION: FAIL
Blocking: C8 (a surviving mutant with a driven differential on a LIVE branch that
no guard covers, plus a false class-level completeness claim) and C2 (the gate
artifact's ledger-row count is unanchored and does not reproduce, and it is the
half of a named remediation that was skipped). Supporting: the self-test check
count published as 19 is 18.

## METHOD DISCLOSURES
(a) No UI claims and no frontend/** or backend/**-runtime file in this step's
    diff, so qa.md §1b/§1c/§1d do not apply; no Playwright capture taken.
(b) Lint scope: qa.md §1a names `git diff --name-only HEAD`, but this step is
    already COMMITTED, so HEAD-diff returns only the unrelated pre-existing
    sovereign_api.py. I used the commit range d1c4a79d~1..HEAD as the authority
    for a committed step, asserted the set non-empty BEFORE reading the exit
    code, and piped through xargs (never an unquoted variable). Both derivations
    disclosed rather than the convenient one chosen.
(c) I drove the REAL enforceEscalation, not a retyped copy: awk-extracted from
    `^function enforceEscalation` to `^}` (52 lines, qa-verdict.js:319) because
    the shipped file has top-level `return` and will not import as ESM.
(d) Every mutant ran on a temp copy in the OS tmpdir / scratchpad. ZERO repo
    writes: verdict_ledger_write.py sha256 e31eaf8e... identical before and
    after, `git status --short scripts/qa/ backend/tests/ handoff/current/`
    empty. The ledger was never written by this evaluation.
(e) I re-checked HEAD at the end: still 3ae269de, no commits landed mid-eval.
(f) I appended to this WIP with `cat >>` heredocs -- a redirect, which qa.md's
    Constraints discourage. Confined to the single path the guard permits;
    flagged rather than left implicit.

COMPLETED: 2026-08-15T14:31:27Z
Final re-check: HEAD 3ae269de (unmoved), scripts/qa + backend/tests +
handoff/current + verdict_ledger.jsonl all clean, writer sha256 e31eaf8e
unchanged by this evaluation.
