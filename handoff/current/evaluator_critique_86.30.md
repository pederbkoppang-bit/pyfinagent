# Evaluator critique -- step 86.30

**Cycle 2** (the FIRST verdict; cycle 1's rail dropped and produced none).
Workflow rail run `wf_95097277-95e` -- 155,817 subagent tokens, 24 tool uses, 535s. Opus / effort max.
Returned 2026-08-10 and transcribed in the same turn it landed.

> **Transcribed VERBATIM by Main from the captured structured return.**
> Main records the verdict and never authors it.

> Cycle 1 (`wf_4c91b666-cf7`) DROPPED after 174,972 tokens. Its write-first
> record was used as EVIDENCE ONLY -- no verdict was transcribed from it and
> nothing reached the verdict gate. This is 86.30's first and only verdict.

## VERDICT: CONDITIONAL   (ok=false, harness_compliance_ok=false, certified_fallback=false)

## reason

All 6 immutable criteria are MET on evidence I executed myself (immutable cmd exit=0/50 passed; frozen-table md5 d9f3650c4054c2504c1bbfaccea25629 byte-identical to 63074429~1; my own pre-fix reproduction 6/6 own global IPv6 called REMOTE by _is_this_machine AND not refused by address_is_live_backend, post-fix 0/16 over the full own set, addresses derived at runtime by psutil AND ifconfig with empty symmetric difference; lsof reproduced byte-for-byte IPv4-only pid 43839; M1-REVERT KILLED in both the healthy and the psutil-blocked environment). B2/B3/B4 are genuinely fixed: the corrected mechanism reproduces (block-only -> enumerable False, evict-only -> True), and M6/M7 both now DIE (1 failed each) in my own mutation run. Verdict is capped at CONDITIONAL, not PASS, because harness compliance is NOT clean (contract-before-generate breach, self-disclosed and un-repairable), plus three residual findings I measured: (1) four further degraded-branch spellings survive with non-empty behavioural differentials (not ip.is_multicast / not ip.is_reserved / both / an explicit literal list) -- same defect class as the original, an IP-property test standing in for ownership; (2) with psutil unimportable process-wide -- the exact environment the fix targets -- M5-REFUSE-ALL (whole predicate returns True) SURVIVES at 5 passed/4 skipped, because the anti-vacuity control experiment_results itself names as the guard against that mutant is one of the 4 skips; (3) Main's "corrected in all three places" does not reproduce for the third place -- experiment_results_86.30.md:50 still states the retracted inverted mechanism as fact with no supersession marker, and the cycle-1 criterion table row 6 (line 42) still reads "3 cells, all KILLED | MET" which cycle 2 itself disproved. Not FAIL: no criterion is materially unaddressed and harness_log has 0 prior 86.30 entries, so the 3rd-CONDITIONAL auto-FAIL rule is not triggered. Each residual has a named fix using machinery already in the file.

## violated_criteria

- `harness_compliance: contract-before-generate (B1, un-repairable)`
- `criterion_2: universal headline not pinned -- 4 surviving degraded-branch mutants with measured non-empty differentials`
- `criterion_4/B4 residual: anti-vacuity control skips in the psutil-absent target environment; M5-REFUSE-ALL survives there`
- `scope-honesty: B2 remediation incomplete -- retracted inverted mechanism and superseded criterion-6 row still stated as fact in experiment_results_86.30.md`

## violation_details

### 1. Invalid_Precondition

**action** -- PLAN phase ordering: contract_86.30.md written after the code

**state** -- measured mtimes -- research_brief 13:02:06, scripts/qa/live_backend_origin.py 13:44:17, test file 13:45:09, contract_86.30.md 13:46:16 (LAST). Single commit 63074429 at 13:47:35; nothing backdated; self-disclosed in a head banner. Research gate itself was NOT skipped (wf_8dfd196f-3fa completed 13:02, 9 sources / 42 URLs).

**constraint** -- CLAUDE.md harness protocol + feedback_contract_before_generate: order is research -> contract -> generate -> qa. Contract written last is a breach and cannot be repaired retroactively, so harness_compliance_ok=false and PASS is unavailable.

### 2. Overgeneralization

**action** -- Q/A mutation matrix run in a mirrored scratchpad tree (repo never written; anchor uniqueness asserted, count==1)

**state** -- HEALTHY env: C0-CONTROL 9 passed; M1-REVERT KILLED (3 failed); M5-REFUSE-ALL KILLED (2 failed); M6-NOT-V4-GLOBAL KILLED (1 failed); M7-V6-OR-NOTGLOBAL KILLED (1 failed) -- B3 confirmed fixed. But N1 `not ip.is_multicast`, N2 `not ip.is_reserved`, N3 `not (is_multicast or is_reserved)` and N4 (explicit 2-address literal) ALL SURVIVED 9 passed. Behavioural differential measured, degraded mode, shipped answers True everywhere: N1/N3 call 224.0.0.1, ff02::1, 239.255.255.250 REMOTE; N2/N3 call 240.0.0.1, 255.255.255.255 REMOTE. is_live_backend('http://224.0.0.1:8000/api/health') is True as shipped and would ALLOW under N1. Non-empty differential = genuine survivors, not equivalent mutants -- which also corrects the dropped cycle-1 record, whose M8-NOT-MULTICAST cell was scored 'EMPTY differential -> equivalent' because its probe set contained no multicast address.

**constraint** -- Criterion 2, headline clause: 'the degraded branch NEVER classifies any address as remote'. The criterion's MANDATED assertion scope (full own set + GENUINELY_REMOTE on the healthy path) IS satisfied, so criterion 2 is MET as written; the universal is not pinned. Severity WARN not BLOCK per qa.md 4c -- the guard is not vacuous (it kills 5 of 9 cells including the revert and refuse-all). NAMED FIX, one list: add a NON-own, NON-GENUINELY_REMOTE odd-class set {224.0.0.1, ff02::1, 240.0.0.1, 255.255.255.255, 203.0.113.7, 100.64.0.1, fe80::dead:beef} to the degraded assertion; kills N1-N4 at once.

### 3. Missing_Assumption

**action** -- Re-ran the suite and the mutation cells with psutil made unimportable PROCESS-WIDE via a sitecustomize __import__ block (precondition asserted: `import psutil` raises)

**state** -- C0-CONTROL rc=0, 5 passed / 4 skipped -- B4's claim reproduces exactly and all four TestDegradedBranchRefuses tests plus the criterion-2 full-set test DO run via the new ifconfig fallback. M1-REVERT rc=1 KILLED (3 failed). But M5-REFUSE-ALL (whole _is_this_machine returns True unconditionally) rc=0, *** SURVIVED *** 5 passed / 4 skipped. The 4 skips are the 3 TestNormalPathIsUntouched tests plus test_healthy_path_still_calls_remote_addresses_remote (line 271), all 'interfaces not enumerable here'.

**constraint** -- experiment_results_86.30.md section 'Criterion 4 -- the anti-vacuity control' names test_healthy_path_still_calls_remote_addresses_remote as the mirror 'without it, refuse everything unconditionally would satisfy every degraded assertion while destroying the guard'. That control does not run in the environment the fix targets, so the suite there cannot distinguish the fix from a destroyed guard. Main's cycle-2 wording ('the four remaining skips are healthy-path tests, correctly inapplicable in an environment that has no healthy path') is literally true but does not disclose this consequence. Criterion 4 itself is MET (M1-REVERT dies in both environments); this is a robustness/disclosure residual. NAMED FIX, machinery already in the test file: synthesise a healthy path without psutil by setting mod._own_cache = frozenset(_own_addresses_via_ifconfig()) and mod._own_enumerable = True, then run the anti-vacuity control there.

### 4. Contradiction

**action** -- grep for surviving copies of the retracted inverted mechanism claim across the 86.30 artifacts and source

**state** -- The shipped _NoPsutil docstring (test file lines 67-89) IS corrected and I verified the corrected claim is TRUE by measurement (block-only -> interfaces_enumerable False, branch REACHED; evict-only -> True, branch NOT reached; both -> False). The contract carries no surviving inverted claim. BUT handoff/current/experiment_results_86.30.md:50 still states, as fact and with no supersession marker at the point of the claim, 'Evicting `sys.modules["psutil"]` is the load-bearing half, and it is why _NoPsutil in the test does both' -- the correction sits ~100 lines later in a separate CYCLE 2 section. Same class: the cycle-1 criterion table row 6 (line 42) still reads '3 cells, all KILLED on a green control | MET', a matrix cycle 2 itself proved incomplete when M6/M7 were found.

**constraint** -- qa.md 4b claim-auditing: a claim must reproduce. Main's cycle-2 statement 'Corrected in all three places' reproduces for two of three. The artifact is the step's evidence of record and is archived, so a reader of section 'Criterion 1 -- the probe that lied to me first' gets a confident false mechanism with no forward pointer. NAMED FIX: annotate in place (do not rewrite history) -- a one-line SUPERSEDED marker at line 50 and on the criterion-6 table row pointing to the CYCLE 2 correction.

## checks_run

- harness_compliance_audit_5_items
- immutable_verification_command
- git_cleanliness_no_unintended_production_change
- md5_frozen_table_byte_identity
- python_lint_gate_ruff_F821_F401_F811_git_derived_scope
- backend_runtime_smoke_module_import_and_behaviour
- criterion_1_independent_prefix_reproduction
- dual_derivation_of_own_addresses_psutil_vs_ifconfig_symmetric_difference
- b2_mechanism_claim_remeasured
- mutation_matrix_9_cells_healthy_env
- mutation_matrix_3_cells_psutil_blocked_env
- behavioural_differential_for_surviving_mutants
- psutil_absent_process_wide_suite_run
- lsof_bind_family_criterion_5
- requirements_declaration_grep
- harness_log_3rd_conditional_counter
- stale_claim_grep_across_artifacts
- code_review_heuristics
- evaluator_critique_existence_check

## notes

MECHANISM HAZARD IN phase-86.31 (report to the operator): the WIP path is FIXED per step, so following the qa.md write-first rule OVERWROTE the predecessor's cycle-1 record at .claude/agent-memory/qa/verdicts/verdict_wip_86.30.md before I had read it. It was recoverable only because 5285699b had committed it (`git show 5285699b:.claude/agent-memory/qa/verdicts/verdict_wip_86.30.md`); on an UNCOMMITTED predecessor record the same sequence would have destroyed it. Suggest a per-cycle suffix or an append-only mode. I read the predecessor from git and disagreed with it in one place (its M8-NOT-MULTICAST 'EMPTY differential -> equivalent' cell is wrong; I measured a non-empty differential). Gates N/A and why: 1b frontend lint/typecheck -- the diff touches no frontend/**; 1c live UI capture -- the step makes no UI claim (scripts/qa + backend/tests only), so no Playwright capture was taken and none was needed; no Main-produced capture was relied on anywhere. 1d backend runtime smoke was run (module execs, enumerable=True, example.com:8000 False, 127.0.0.1:8000 True, 192.0.2.55:8000 False). All mutation work was done in a mirrored scratchpad tree (the test resolves SRC via parents[2]); the repo working tree was never written and `git diff HEAD --stat` shows no dirty production or test file. Positive verifications worth recording: Main's B5 claim 'no production code changed in cycle 2' is TRUE (git diff vs 63074429 empty, md5 2669cbe069b026e2d9590e37a5d275cd identical); Main's 'What I still cannot claim' section is epistemically correct and materially softens the surviving-mutant finding -- it explicitly states 'four cells killed licenses these four were killed, nothing global', so criterion 6 is not overclaimed. Code-review heuristics: no findings -- the diff is a QA harness script plus a test, touches no trading/risk/kill-switch path, contains no secrets, and the only subprocess use is list-form `subprocess.run(["ifconfig"], ...)` with shell=False (explicitly on the negation list). retry_count=0 < max_retries=3, so certified_fallback is false. This is the FIRST verdict for 86.30 (no handoff/current/evaluator_critique_86.30.md exists); it is not a reversal, and the evidence changed between spawns (5285699b: test file +81/-13, experiment_results +122), so the no-second-opinion-shopping rule is satisfied.
