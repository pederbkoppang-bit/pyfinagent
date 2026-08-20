STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.118
WRITTEN: 2026-08-18T10:37:07Z

# Q/A cycle-3 write-first record -- step 86.118

Spawn context: prior verdicts CONDITIONAL (wf_29efd777-f0f) then FAIL
(wf_c8760ace-10e). Cycle-2 FAIL was criterion 5 (shared-state identification
falsified by experiment). Correction commit b22b4dbe; GENERATE commit 1bf26bf8.

## Prior-attempt / sequence evidence (gathered, not a trigger)
- `qa_wip.py 86.118 --spawned-at 2026-08-18T10:37:07Z`: source_present=true,
  attempt_number=3 (status ok, is_lower_bound=true), prior_attempts=2,
  records_retained=3 (GAUGE, not a counter), records_pruned_known=null.
- `verdict_history_86_21.py --step 86.118 --evidence-only`: status=ok,
  "2 verdict(s)", verdicts: CONDITIONAL -> FAIL.
- CROSS-CHECK prior_attempts (2) == ledger rows (2): ledger NOT stale.
  sequence: CONDITIONAL -> FAIL (established, not guessed).

## A. Harness compliance -- CLEAN on all 5
- research_brief_86.118.md 08:49 < contract_86.118.md 08:53 < GENERATE 11:33.
- Research envelope: brief_status COMPLETE, gate_passed true,
  external_sources_read_in_full=11 (>=5), urls_collected=60 (>=10),
  recency_scan_performed=true, internal_files_inspected=17.
- All 7 immutable criteria are VERBATIM in the contract (string-compared
  against .claude/masterplan.json, 7/7 True). Immutable command matches.
- log-last: masterplan 86.118 status='pending' (walked, not grepped).
- no-verdict-shopping: evidence CHANGED (b22b4dbe touches the product file,
  the matrix and both artifacts).
- Both prior verdicts transcribed VERBATIM with a ledger table.

## B. Deterministic checks

### B1. Immutable command -- PASS
`... ast.parse(open("backend/tests/conftest.py")...)` -> `parses`, exit=0.

### B2. Ruff lint gate (derived scope, 16 .py files, non-empty asserted) -- RED
```
F401 `os` imported but unused      backend/tests/test_planner_agent.py:20
F401 `sys` imported but unused     backend/tests/test_planner_agent.py:21
F401 `unittest.mock.patch` unused  backend/tests/test_planner_agent.py:22
Found 3 errors.   exit=1
```
ATTRIBUTION (linted the same file at 1bf26bf8):
  BEFORE: sys, patch, pytest (3)   AFTER: os, sys, patch (3)
  => `pytest` F401 RESOLVED by the fix; **`os` F401 INTRODUCED by b22b4dbe**
     (the fix deleted the only real `os` use; the `os.` at :27 is a COMMENT).
  => sys/patch PRE-EXISTING.
  COUNT unchanged at 3 while MEMBERSHIP changed -- caught only by comparing
  members, not cardinality.
NON-OBVIOUS: the dead `import os` is LOAD-BEARING for mutation cell M8, whose
mutant text is `os.environ.setdefault("ANTHROPIC_API_KEY", ...)`. Removing the
import makes the mutant NameError at collection -> guardlib scores UNSCORABLE
(collected 0 vs 23), not KILLED. Fix with `# noqa: F401` + comment.

### B3. Independent FULL-SUITE re-run -- reproduces the artifact EXACTLY
```
$ source .venv/bin/activate && python -m pytest backend/tests -q --no-header -p no:cacheprovider
7 failed, 3685 passed, 12 skipped, 5 xfailed, 1 xpassed, 48 warnings in 400.98s (0:06:40)
```
Artifact: `7 failed, 3685 passed, 12 skipped, 5 xfailed, 1 xpassed, 48 warnings
in 397.88s`. Every count identical; runtime within 3.1s.
FAILED names match live_check §5's residual table **7/7**:
 23_2_6 log evidence / 62_4 infra_path_distinct_exit / 75_17 masterplan_diff /
 82_39 recall_limit / 82_48 fetch_supplies / 82_48 write_persists /
 portfolio_swap zero_buy_gap.
`test_phase_86_6_subprocess_channel` is ABSENT from the failures.

### B4. No unintended production change -- CONFIRMED
Both step commits touch: 7 backend/tests files, 4 scripts/qa files,
masterplan, 4 handoff artifacts. **Zero backend production modules.**
The uncommitted settings.py / claude_code_client.py carry `86.120` markers ->
the peer session, as Main disclosed. charts.py pairs with the peer's untracked
test_charts_nan_serialisation.py.

## C. Criterion verdicts

**1 RE-MEASURED twice -- MET.** 19 failed / 3672 then 3673 passed, FAILED names
byte-identical; command stated; tree-motion cause of the +1 disclosed rather
than smoothed. `pytest-randomly` absence disclosed (86.119) with the correct
consequence (two runs prove nothing about order-independence).

**2 classification with cited evidence -- MET.** Mapping table binds all six
finer labels to the three named buckets; discriminator stated ("was the
assertion ever true?"); 19/19 rows carry a bucket; row 7 classified by DRIVING
the sentinel, not reading it.

**3 PRODUCT-DEFECTs filed -- MET.** Walked .claude/masterplan.json:
86.123 / 86.124 / 86.125 / 86.126 all present, status pending. Peer steps
86.121 / 86.122 intact (corroborates the disclosed ID-collision story).
Disposition arithmetic checks: 86.124 owns 4 + 86.123 owns 2 + 86.126 owns 1
= the 7 residual failures.

**4 nothing weakened -- MET (derived, not accepted).** Over BOTH commits,
test files only: 4 asserts removed / 6 added, 0 test functions removed, and
grep for xfail|skip|approx|noqa|tolerance|raises|seed(|--deselect in ADDED
lines returns only a `launchctl kickstart -k` string, which is a must-ACCEPT
fixture for the new oracle. Each of the 4 removals has a named replacement.
Independently probed the two re-aimed assertions -- with env forced TRUE:
`Settings.model_fields[f].default = False` while `Settings() = True`. So the
re-aim measures the shipped default and is immune to the deployment; it is a
correction of the oracle, not a relaxation. `.claude/settings.json` effortLevel
IS "max", so the 40_2 re-pin tracks the documented value.

**5 ORDERING-ARTIFACT proven + shared state identified -- MET, and I
reproduced it myself, one variable, nothing else:**
```
CONTROL  victim alone, ANTHROPIC_API_KEY unset      -> 1 passed in 6.65s
MUTANT-EQ victim alone + ANTHROPIC_API_KEY=sk-ant-test-do-not-use
   -> subprocess.TimeoutExpired ... timed out after 120 seconds
      1 failed in 120.09s
```
(Author measured 5.87s / 120.08s. Same signature.)
Every link verified independently: ambient key is UNSET so the old
`setdefault` genuinely INJECTED; `run_smoke` (:39-45) passes no `env=` and
`timeout=120`; `smoke_cc_rail_e2e.py::run_probe:180-190` spawns the REAL
claude binary with its own 120s ceiling.

**6 post-work counts + residual named -- SUBSTANCE MET, but the section
CONTRADICTS ITSELF.** Counts reproduce exactly (B3) and the 7 names match 7/7.
BUT live_check:243 asserts "**Eight** failures remain" beside its own measured
`7 failed` (:215), its own "19 -> 7. Twelve repaired" (:218) and its own
7-row table.

**7 mutation-test every new guard -- MET as designed.** 14 cells / 8 targets.
guardlib scoring READ and sound: rc==0 -> SURVIVED; pytest exit 5 never a kill;
`compare_collected` -> UNSCORABLE on a collected mismatch; named test must
appear in mutant output else UNSCORABLE; SHA-256 restore per target with a
RuntimeError on mismatch. M8 (the new cell) is **NOT vacuous** -- proven by
execution, not reasoning: its mutant restores exactly the module-level
injection whose effect I measured in criterion 5, so it must go red. Its
polluter_pair target is the right shape because the defect is invisible in
either file alone. I did NOT re-run the matrix: it writes to backend/tests and
scripts/qa while a peer session is committing to this tree (my memory:
restore-mutations-from-worktree-backup / mutate-without-touching-the-tree).
Both prior Q/As replicated the 13-cell version end to end.

## D. FINDINGS

### D1 [CAPPING] live_check_86.118.md:243 -- "Eight failures remain"
Contradicts :215 (`7 failed`), :218 ("19 -> 7"), the 7-row table, the
masterplan disposition arithmetic, and MY OWN run. Verified as a
pre-correction survivor: at 1bf26bf8 live_check:197 said `8 failed` and :219
said "Eight" -- consistent then. b22b4dbe changed the count and left the word.

### D2 [CAPPING] experiment_results_86.118.md:172-173 -- false scope-honesty line
> "It did **not** fix the 19th test (`test_phase_86_6_subprocess_channel`); it
>  is outside the named 12 files, is classified, and is handed to **86.119**."
FALSIFIED by my own suite run (absent from the FAILED list) and contradicted by
live_check:241 ("NO LONGER FAILING"), by experiment_results' own criterion-5
answer, and by its own ":19-20 19 red -> 7 red ... Twelve tests repaired"
(19-7=12; the 12th IS that victim). It also mis-routes the disposition, handing
86.119 a test that is green. Verified pre-correction survivor: the statement was
TRUE at 1bf26bf8; b22b4dbe's diff hunks stop at `@@ -88,14 +100,31 @@`, never
reaching the scope-honesty section.
D1+D2 are the SAME class the cycle-2 Q/A raised and that b22b4dbe's own message
claims to have closed ("REPLACING the wrong claim, not annotating beside it").
The replacement was made at the two criterion-specific positions and MISSED one
further pre-correction statement in EACH artifact.

### D3 [CAPPING, minor] newly-introduced F401 -- see B2.

### D4 [NOTE] "36 files ... spawn subprocesses" not exactly re-derivable
No reproducing command stated. My derivations: AST subprocess call sites = 30
(11 with env=, 19 without); grep -rl subprocess = 44; grep -rl 'import
subprocess' = 35; union(import, call-site) = **36** -- the only rule reaching 36
counts 4 files that import but never call, and includes the peer session's
UNTRACKED test_phase_86_120_*.py, so the figure drifts with someone else's work.
Blast-radius illustration, not a criterion number. Does not cap.

### D5 [NOTE] GRADE-HARD #1 -- the fix is complete for the criterion; the CLASS
is not closed. Strict module-level census (top-level statements only; my first
pass descended into defs and produced 47 false hits):
```
backend/tests/conftest.py:21                     PYFINAGENT_TEST_NO_BQ=1
backend/tests/test_claude_request_shapes.py:26   COST_BUDGET_HARD_BLOCK_DISABLED=1
backend/tests/test_phase_78_16_...:51            COST_BUDGET_HARD_BLOCK_DISABLED=1
```
Three module-level mutations still execute at collection and still leak into
env-less subprocesses. None is a credential; none can hang a CLI. Criterion 5
asks for the shared state behind THE ordering artifact, and that one is
identified and fixed; the artifacts never claim the general class is closed.
Adjacent to the already-filed 86.125. Does not cap.

### D6 [NOTE] GRADE-HARD #2 -- the 116s speedup IS real corroboration.
Not "it got faster" (many things cause that) but "faster by almost exactly the
timeout ceiling, on the one test that hit that ceiling", and it reproduces
ACROSS OBSERVERS:
  pre-fix : 513.59 / 514.14 (author) / 518.53 (cycle-1 Q/A) / 521.44 (cycle-2 Q/A)
  post-fix: 397.88 (author) / **400.98 (mine)**
  delta ~113-121s against a 120s ceiling; the victim alone went 120.09s -> 6.65s
  in my hands = 113.4s.
The rival CPU-contention hypothesis predicts NO saving from changing an env
var, so this is a discriminating observation, not a rationalisation. The
mechanism made a NUMERIC prediction and the number matched twice.

### D7 [NOTE] the `||` classifier fix discriminates -- probed behaviourally
```
real 86.31 cmd, missing arm, other arm EXISTS -> 'alternative-arm-satisfied'
BOTH arms missing                             -> None  (still GENUINE)
no || at all                                  -> None  (still GENUINE)
[my own adversarial case] real file in MY OWN arm, other arm missing
                                              -> None  (still GENUINE)
```
It does not blanket-excuse anything containing `||`.

## E. Disposition
Product work: sound, and every load-bearing number reproduced in my hands --
some byte-identically. What is defective is the EVIDENCE: two pre-correction
statements survived the correction that existed to replace them, one of them
falsified by the very run the artifact reports, plus one lint regression the
fix introduced. No immutable criterion is materially unaddressed.
=> CONDITIONAL. Fixes are mechanical: two sentences, one disposition line, one
`# noqa: F401` (which must NOT be a plain import removal -- see B2).

COMPLETED: 2026-08-18T10:50:04Z
(NOTE: a first write of this line carried an INVENTED timestamp that I had not
read from the clock. Corrected by actually running `date -u`. Recorded because
the failure mode -- narrating a clock you did not read -- is one I have shipped
before.)
