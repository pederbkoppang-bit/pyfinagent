STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.118
WRITTEN: 2026-08-18T09:57:00Z

# Q/A write-first record -- step 86.118 (cycle 2)

Spawn context: cycle-2 Q/A after cycle-1 CONDITIONAL (`wf_29efd777-f0f`).
Prompt claims evidence changed via commit 2e0728ae (artifacts only); GENERATE
commit 1bf26bf8.

Grading focus demanded by prompt:
- Criterion 2: is the bucket mapping a genuine classification or a relabelling?
- Criterion 5: is "wall-clock contention on a real external dependency" a real
  identification of shared state, or an excuse dressed as one?
- Also asked: is raising-not-fixing correct for the flags_match_tokens finding?

## Log

- [09:57:00Z] WIP created.

### Sequence / attempt evidence
- `qa_wip.py 86.118 --spawned-at 2026-08-18T09:57:00Z`: source_present=true,
  attempt_number=2, attempt_number_status="ok", is_lower_bound=false,
  prior_attempts=1, records_retained=2, prior_records=[verdict_wip_86.118__20260818T093446Z.md].
- `verdict_history_86_21.py --step 86.118 --evidence-only`: status=ok,
  verdicts = CONDITIONAL (1 row).
- CROSS-CHECK: prior_attempts (1) == ledger rows (1) -> no staleness signal.

### Harness compliance (5 items)
1. Research gate: research_brief_86.118.md present, mtime 08:49:27.
2. Order: research 08:49:27 < contract 08:53:31 < experiment_results 11:33:35. OK.
3. experiment_results_86.118.md present (8149 bytes).
4. Log-last: `grep -F 86.118 handoff/harness_log.md` -> 2 hits, BOTH are prose
   references from OTHER steps' cycles (lines 36447, 36476); NO
   `phase=86.118 result=` row. masterplan 86.118 status=pending. OK.
5. No verdict-shopping: evidence CHANGED. `git diff 1bf26bf8 2e0728ae --
   live_check_86.118.md` = +18/-1 in S3 (bucket mapping table + row 7 reclass),
   new S5a (~48 lines), S6 rewritten with shared-state identification +
   restored anomaly. Substantive, not cosmetic.

### Deterministic
- IMMUTABLE COMMAND: `bash -c 'source .venv/bin/activate && python -c "import
  ast; ast.parse(open(\"backend/tests/conftest.py\").read()); print(\"parses\")"'`
  -> stdout `parses`, exit=0. REPRODUCED.
- Commit scope 1bf26bf8 (GENERATE): masterplan.json, 6 backend/tests/*.py,
  4 scripts/qa/*.py, 2 handoff artifacts. ZERO backend production modules.
- Commit scope 2e0728ae: evaluator_critique_86.118.md, live_check_86.118.md,
  verdict_ledger.jsonl. ARTIFACTS ONLY -- claim confirmed.
- Uncommitted tree: backend/config/settings.py, backend/agents/claude_code_client.py,
  backend/api/charts.py modified -- PEER SESSION, matches disclosure.

### Criterion-5 environment facts I derived (bears on the shared-state claim)
- pytest-xdist NOT installed; no addopts in pytest.ini/setup.cfg/pyproject.toml.
  => THE SUITE RUNS SEQUENTIALLY. When the victim test executes, no other test
  is executing.
- pytest_randomly NOT installed (confirms the artifact's claim).
- The 120s ceiling is the TEST's own `subprocess.run(..., timeout=120)` at
  backend/tests/test_phase_86_6_subprocess_channel.py:44.
- The victim passes `--dry --backend-url http://localhost:8000
  --allow-live-backend` and does NOT pass `--no-probe`, so
  smoke_cc_rail_e2e.run_dry:439 calls `run_probe(binary, model)` which
  `subprocess.run`s the REAL claude CLI with its own timeout_s=120
  (smoke_cc_rail_e2e.py:179-186), and then runs a BigQuery 7d query
  (:447-455). The traceback's stdout_seq stops AFTER preflight, so the hang is
  in run_probe (real claude CLI) or the BQ query -- NOT in CPU-bound work.

### CRITERION-5 EXPERIMENTS I RAN (the grading focus)
- E1 (reproduce isolation): victim alone -> `1 passed in 6.66s` (author 6.80s).
  CPU accounting from `time`: `1,78s user 0,55s system 33% cpu 7,042 total`
  => only ~2.3s of the 7.0s is CPU; ~4.7s is EXTERNAL WAIT.
- E2 (candidate polluter): `pytest test_phase_4000_2_cc_rail_smoke.py <victim>`
  -> `23 passed in 25.45s`. The sibling file driving the SAME script 22x does
  NOT reproduce the timeout. Supports the author's "no polluter" direction for
  this candidate; one candidate is not a bisection.
- E3 (FALSIFIES THE STATED MECHANISM): 20 CPU burners on 10 cores (2x
  oversubscription, all spinning), victim alone -> `1 passed in 5.17s`,
  FASTER than idle. Deliberate CPU saturation produced NO slowdown.
- E4 (where the wall time goes): smoke script direct, WITH probe `9,646 total`
  (2.36s CPU) vs `--no-probe` `5,057 total` (1.71s CPU). The REAL claude CLI
  probe is ~4.6s WALL / ~0.65s CPU; the rest includes a live BigQuery 7d query.
  The variable component is EXTERNAL LATENCY, not CPU.
- ARITHMETIC: reaching 120s from ~2.3s of CPU needs ~50x CPU slowdown; E3
  measured 0x at 2x oversubscription. CPU contention cannot produce it.
- FINDING: "under whole-suite CPU contention" is an UNJUSTIFIED INFERENCE.
  The suite is SEQUENTIAL so nothing competes at that instant, and the two
  data points offered (alone 6.80s / full 120s) are the SAME two points that
  define the ordering artifact -- they cannot discriminate CPU contention from
  external-service latency. "Nothing another test wrote is responsible / there
  is no polluter test" is an untested negative (no bisection).
- WHAT IS GENUINELY DISCHARGED: the failure mode is a TIMEOUT on an external
  call, not an assertion over polluted in-process state -- established by the
  traceback (returncode -9, stdout stops after preflight). Naming the live
  backend + real claude CLI as shared external dependencies is a real, correct
  identification of the shared RESOURCE.

### CRITERION 2 -- row 7 reproduced by driving the sentinel MYSELF
- `sentinel.sh:159-160` verified verbatim at those exact line numbers:
  `infra = {"metered_source_unavailable","flags_reconciliation_unavailable"}`
  / `sys.exit(2 if set(report["gates_failed"]) <= infra else 1)`.
- clean run  -> exit=1, gates_failed ["flags_match_tokens"],
  warnings ["unauthorized true flags: PAPER_SYNTHESIS_INTEGRITY_ENABLED"],
  ok:false.  REPRODUCES the artifact byte-for-byte on every load-bearing field.
- SENTINEL_TEST_BQ_FAIL=1 -> exit=1, gates_failed
  ["metered_source_unavailable","flags_match_tokens"]. REPRODUCES.
- So the subset test fails for the stated reason. Classification defensible.
- NOTE: I could NOT read backend/.env (permission denied by the guard), so
  the ":88" line cite is taken on the sentinel's own output, which names the
  flag. Not a gap in the finding -- the gate output is the better evidence.

### E5 -- machine load DURING a real full-suite run
- 12:07 (~4 min into my own full-suite run): `load averages: 2,64 6,47 4,71`
  on 10 cores. The 1-min figure is 2.64 -- FAR below the 20-burner condition
  under which the victim still passed in 5.17s. (The 5-min 6.47 is residue
  from my own burners at 12:02, not the suite.)
- So the full-suite condition does NOT saturate the CPU, which is what the
  sequential-runner fact already predicted.

### CRITERION 4 verified by derivation (not by claim)
- `git show 1bf26bf8 -- backend/tests/` added lines grepped for
  xfail|skip|pytest.mark.(skip|xfail)|approx|tolerance|pytest.raises|noqa|
  assert True -> ZERO hits.
- assert lines: 4 removed / 6 added. test functions: 0 removed / 0 added.
- The 4 removals each have a replacement: "xhigh"->"max" (equality kept),
  two Settings() reads -> Settings.model_fields[...].default (the `is`
  identity kept), and the launchctl oracle replaced by a file-aware one that
  ships must-reject + must-accept fixtures.

### `||` classifier blast radius -- cycle-1 census REPRODUCED
- Re-derived by walking .claude/masterplan.json: 1153 verification commands,
  14 contain `||`. Same figures cycle 1 reported.
- 40.3.1 has path tokens in BOTH arms; both files are absent
  (docs/stress-tests/ holds only 2026-Q2-opus-4.7.md), so per the stated
  logic "BOTH arms missing -> None (still GENUINE)" -- no wrong excuse. No
  new finding beyond cycle 1's.

### FINDING (mine, new): experiment_results_86.118.md is STALE on BOTH
### criteria under repair
- mtime 11:33:35, untouched by 2e0728ae (artifacts-only commit changed only
  evaluator_critique + live_check + verdict_ledger).
- Its "Criterion 2" paragraph still enumerates ONLY the finer labels and
  still lists "1 exit-code drift" -- the exact row cycle 1 flagged as
  unclassified and which live_check has now reclassified to STALE-EVIDENCE.
- Its "Criterion 5" paragraph still reads "the ORDERING-ARTIFACT class is
  EMPTY in scope ... a measured `n=1 outside scope` rather than a failure to
  find shared state among the 18" -- verbatim the scope-substitution framing
  cycle 1 rejected, with NO mention of the shared-state identification.
- experiment_results.md is the GENERATE artifact named by the five-file
  protocol; a reader of it comes away with the pre-correction position on
  both repaired criteria. Correction accompanied the artifact instead of
  replacing it.

### Filed steps exist (criterion 3)
86.119/86.123/86.124/86.125/86.126 all present in masterplan, all
status=pending, all harness_required=true.

### CRITERION 6 -- my own independent full-suite run REPRODUCES
`8 failed, 3684 passed, 12 skipped, 5 xfailed, 1 xpassed, 48 warnings in
521.44s (0:08:41)` vs the artifact's `... in 514.34s`. All 8 FAILED names
match the live_check S5 residual table 8/8.

### THE POLLUTER IS IDENTIFIED -- CRITERION 5's ANSWER IS FALSIFIED
Decisive experiment. Victim test ALONE, IDLE machine, ONE env var set to the
exact literal `backend/tests/test_planner_agent.py:27` injects:

```
$ ANTHROPIC_API_KEY=sk-ant-test-do-not-use python -m pytest \
    backend/tests/test_phase_86_6_subprocess_channel.py::test_the_optin_IS_honoured_so_a_real_window_remains_possible -q
E   subprocess.TimeoutExpired: Command '[...smoke_cc_rail_e2e.py, --dry,
    --backend-url, http://localhost:8000, --allow-live-backend]'
    timed out after 120 seconds
1 failed in 120.08s (0:02:00)
```

Same failure signature as the full run. Causal chain, every link measured:
1. test_planner_agent.py:27 `os.environ.setdefault("ANTHROPIC_API_KEY",
   "sk-ant-test-do-not-use")` at MODULE level ("Pre-set a dummy key so
   `Anthropic()` in __init__ doesn't raise on import").
2. ANTHROPIC_API_KEY is ABSENT ambiently (measured) -> the setdefault APPLIES.
3. pytest imports every module during collection before any test runs;
   test_planner_agent.py IS collected in the full suite (5 tests). So the
   bogus key is in os.environ before the victim executes -- DETERMINISTIC.
4. `run_smoke` (test_phase_86_6_subprocess_channel.py:42-45) calls
   subprocess.run with NO `env=` kwarg -> the child INHERITS os.environ.
5. run_dry:439 calls run_probe -> the REAL claude CLI with a bogus key ->
   never returns -> outer 120s timeout -> TimeoutExpired / returncode -9.
6. In isolation test_planner_agent.py is never imported -> real credential ->
   probe returns in ~4.6s -> `1 passed in 6.66s`.

COMPONENT PINNED: bogus key + `--no-probe` -> exit 0 in seconds. So BQ and
backend HTTP are fine; the hang is in run_probe.
SECOND HYPOTHESIS TESTED AND DISCARDED: `GCP_PROJECT_ID=test-proj`
(test_regime_detector.py:150, same setdefault shape) -> `1 passed in 7.74s`.
Not a contributor.
BLAST RADIUS: 36 files in backend/tests spawn subprocesses and would inherit
the same poisoned environment.

WHAT THIS FALSIFIES, verbatim from live_check S6:
- "The shared resource is WALL-CLOCK on a real external dependency" /
  "under whole-suite CPU contention" -- FALSE. Falsified three ways:
  (i) no pytest-xdist, no addopts => sequential runner, nothing competes;
  (ii) measured load during MY full-suite run that REPRODUCED the failure:
       1.61-2.64 on 10 cores across 5 samples;
  (iii) 20 spinning burners (2x oversubscription) -> victim passed in 5.17s,
        FASTER than idle.
- "Nothing another test *wrote* is responsible" / "there is no polluter
  test" -- FALSE. A test writes process-global env state at import time.
- "This is not repairable by cleaning shared state, which is the usual
  order-dependence remedy (Luo FSE'14 F.10: 74%)" -- BACKWARDS. This IS the
  74% case: scope the env var (monkeypatch/fixture) or pass explicit `env=`.
- Consequence: the artifact misdirects 86.119 toward "isolation from the live
  backend or a bound that reflects loaded-machine latency" and EXONERATES a
  live polluter that will keep breaking subprocess-spawning tests.
HONEST BOUND ON MY OWN FINDING: I proved this env var is SUFFICIENT and that
it is deterministically present in any full-suite run. I did not prove it is
the ONLY contributor.

### CRITERION VERDICTS
1 MET | 2 MET | 3 MET | 4 MET | 5 NOT MET | 6 MET | 7 MET (as run)

### GRADING-FOCUS ANSWERS ASKED FOR IN THE SPAWN
- Bucket mapping: GENUINE classification, not relabelling. It applies a
  stated discriminator ("was the assertion ever true?"), which is what
  separates PROXY ASSERTION (never true -> PRODUCT-DEFECT) from ENV LEAKAGE
  (true when written -> STALE-EVIDENCE); it assigns two labels to
  PRODUCT-DEFECT at COST to the author (triggering criterion 3's filing duty
  and the "never close a defect by editing the test" tension, both of which
  it names); and its one contested row is backed by a drive I reproduced
  byte-for-byte. A relabelling exercise would have put everything in the
  cheapest bucket; this did not.
- "Wall-clock contention on a real external dependency": AN EXCUSE, and now
  a measurably wrong one. See above.
- Raising-not-fixing the sentinel gate breach: CORRECT, and fixing would
  have been wrong. The only two ways to green the gate are de-promoting an
  operator-gated flag (a live behaviour change) or writing an authorization
  token (forging the operator's authorization and defeating the gate's
  purpose). It is filed to 86.124 and stated in the commit body. Caveat: it
  is raised only in handoff artifacts and a commit message; an away-ops
  watchdog reporting ok:false deserves a channel the operator reads.

### OTHER FINDING (mine): experiment_results_86.118.md is STALE
See the section above -- the GENERATE artifact still carries the
pre-correction position on BOTH repaired criteria.

COMPLETED: 2026-08-18T10:18:01Z
