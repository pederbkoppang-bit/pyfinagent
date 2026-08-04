# Evaluator critique -- Step 82.23

## Cycle 1 -- 2026-08-04 -- Q/A verdict **CONDITIONAL**

*Launched on the Workflow structured-output rail (`.claude/workflows/qa-verdict.js`,
task `wgn08733b`), model opus / effort max. The verdict IS the captured return value.
Main transcribes it VERBATIM and does not author it.*

**RECOVERY NOTE (written 2026-08-04):** this file should have been written the
moment the verdict returned. It was not -- I held both verdicts in conversation and
acted on them without persisting them. That is a five-file-protocol breach: the
`evaluator_critique` artifact is what makes a verdict auditable after the session
ends, and acting on a verdict is not a substitute for recording it. Recovered from
the workflow's own output file, absolute path
`/private/tmp/claude-501/-Users-ford--openclaw-workspace-pyfinagent/1e60f387-02c6-4289-b919-31589cfbe491/tasks/wgn08733b.output`
-- a session-scoped /private/tmp path that is cleaned on session end. The cycle-2
Q/A [W2] correctly could not locate it: my note cited the bare relative fragment
`tasks/wgn08733b.output`, and it searched the repo, ~/.claude and ~/.openclaw, none of
which contain it. A citation nobody else can resolve is not evidence. The raw
return is therefore now archived IN-REPO at
`handoff/current/qa_returns/wgn08733b.output.json`, byte-identical to the source. The in-transcript copy was
truncated mid-string and would NOT have parsed -- which is precisely why the
artifact is mandatory rather than optional. Parsed and pretty-printed; no wording
altered.

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "Immutable verification command reproduces green (exit=0, 21 passed, run 3x). Criteria 2, 3 and 4 are MET with genuinely behavioral guards -- I ran 6 independent mutations in-process (no tree writes) and ALL 6 were killed: M1 wrapper returns the raw false-good 0.0 -> 5 failed; M2 gate ignores pbo_n_trials -> 2 failed; M3 gate always refuses (the cheap-pass direction) -> 3 failed; M4 live ceiling raised to 0.80 above the measured incumbent 0.7486 -> 2 failed; M5 diversity number replaced by a constant -> 1 failed; M6 diversity keys dropped -> 2 failed; CONTROL 21 passed. Criterion 1 is NOT MET as worded and the suite ships a test that PINS the negation (test_generate_report_still_does_not_emit_a_pbo). I independently tested Main's defence and could not refute it: generate_report (analytics.py:741) takes one BacktestResult, all 16 repo call sites pass a single run, and compute_pbo returns 0.0 at N<2 -- a value that passes the live 0.20 ceiling (pinned green by test_raw_compute_pbo_returns_a_false_good_zero). Satisfying criterion 1 literally would ship the exact defect the step exists to prevent, so I will not grade it FAIL for negligence -- but criteria are immutable and a Q/A cannot waive one, so this CANNOT be flipped done against criterion 1 as written; it needs an operator decision plus a re-spec in a NEW step. Two further blockers: the evidence MUTATED DURING my evaluation (analytics.py 10:25:48, test file + adapter 10:26:10, experiment_results.md 10:26:31 -- my first verification run measured 18 tests/152 lines, the tree now has 21 tests/194 lines), collapsing the frozen-GENERATE precondition; and the mutation matrix in experiment_results.md still claims CONTROL -> 30 passed while the same file's regenerated block says 36 passed (15+21) -- a number that reproduces neither the old 28 nor the measured 36, and the 3 new criterion-4 tests carry no author-run mutation at all.",
  "violated_criteria": [
    "criterion_1_generate_report_emits_pbo",
    "Invalid_Precondition: evidence mutated during EVALUATE",
    "Contradiction: mutation matrix CONTROL count does not reproduce",
    "illusory-guard: adapter forwarding covered only by a source scan"
  ],
  "violation_details": [
    {
      "violation_type": "Missing_Assumption",
      "action": "Read backend/backtest/analytics.py::generate_report and backend/tests/test_phase_82_23_pbo_in_gate.py::test_generate_report_still_does_not_emit_a_pbo",
      "state": "generate_report emits NO pbo field. The suite asserts the NEGATION of the criterion: `src = inspect.getsource(analytics.generate_report); assert \"compute_pbo\" not in src`. Verified independently: generate_report is at analytics.py:741 and takes a single BacktestResult; all 16 repo call sites pass one run (strategy_backtest_adapter.py:162, quant_optimizer.py:205 and :267, api/backtest.py:1058, run_harness.py:134 and :143, run_validation.py:88, run_82_3_candidate_backtests.py:162, run_quick_test.py:59, run_optimizer.py:77, run_ablation.py:170, run_subperiod_test.py:55, run_seed_stability.py:57, run_experiment.py:118). compute_pbo returns 0.0 at N<2, and 0.0 <= PBO_CEILING_LIVE=0.20 -- pinned green by test_raw_compute_pbo_returns_a_false_good_zero. The criterion's own precondition ('K>=2 configurations are available') is therefore never satisfiable at that function.",
      "constraint": "Immutable criterion 1: 'generate_report emits a pbo field for a run whose daily NAV series and K>=2 configurations are available, asserted on a fixture'. Criteria are immutable (CLAUDE.md 'Never edit verification criteria'); a Q/A cannot waive one. The step's INTENT is served by compute_pbo_checked + the adapter + PromotionGate, but the criterion as worded is unmet and satisfying it literally would emit an unconditional PASS on every run. Disposition: operator decision + re-spec in a NEW step naming the sweep-level producer instead of generate_report. Precedent: auto-memory feedback_immutable_criteria_must_be_green_able (the 81.0 structurally-uncloseable class)."
    },
    {
      "violation_type": "Invalid_Precondition",
      "action": "stat -f mtime on the evidence + code files, and re-running the verification command at two points during EVALUATE",
      "state": "My first verification run returned '18 passed' against a 152-line test file. Two minutes later: backend/backtest/analytics.py mtime 10:25:48, backend/tests/test_phase_82_23_pbo_in_gate.py and backend/autoresearch/strategy_backtest_adapter.py mtime 10:26:10, handoff/current/experiment_results.md mtime 10:26:31 (again at 10:31:08) -- the suite is now 194 lines / 21 tests and the criterion-4 diversity number that the spawn prompt asked me to confirm ABSENT had been SHIPPED mid-evaluation. My ruff scope, my read of the diff, and my read of the source were all taken against a tree that no longer existed. experiment_results.md lines 11-16 record the same class biting the 82.22 Q/A ('the criterion-4 tests shipped briefly RED ... and the evaluator observed that failure mid-run'), so this is the second occurrence in this phase, not a one-off.",
      "constraint": "The doer/judge separation requires GENERATE to be FROZEN before EVALUATE. qa.md: 'You return a verdict and STOP ... Main owns any fix and spawns a FRESH Q/A on CHANGED evidence.' Editing production code and the test suite while the evaluator is mid-run makes the verdict a function of WHEN the evaluator looks, and means no single tree state was ever fully audited. Main's disclosure at experiment_results.md:90-107 is honest and creditable, but disclosure does not cure the breach. This verdict is scoped to the tree measured 10:26:10-10:32:07 (21 tests, FINAL_EXIT=0)."
    },
    {
      "violation_type": "Contradiction",
      "action": "grep -n 'passed|failed|CONTROL' handoff/current/experiment_results.md and re-derived the control count myself",
      "state": "The regenerated verbatim block (line 20/23/26) records '15 passed' + '21 passed' + '36 passed'. The mutation matrix 30 lines later (line 159) still records 'CONTROL -> 30 passed', with per-mutant rows '82.23 M1 -> 2 failed / M2 -> 2 failed / M3 -> 2 failed'. 30 reproduces neither the superseded 28 (10+18) nor the measured 36 (15+21). Main regenerated the test-count block after adding the three criterion-4 tests but did NOT re-run or re-label the matrix, so the block labelled as the mutation evidence is stale by the file's own arithmetic. Separately, my measured kill counts (5/2/3) differ from the claimed 2/2/2 -- that is a different operationalization and is NOT itself a finding, but it does mean the claimed numbers were never re-derived against the current suite. The three NEW criterion-4 tests carry no author-run mutation whatsoever; I supplied that coverage myself (M5, M6 -- both killed).",
      "constraint": "qa.md section 4b: 'Every numeric claim must carry, or you must be able to RE-DERIVE, the exact command that produces it ... Prefer FAIL when a number in a verbatim artifact does not reproduce', and 'a verbatim capture must be regenerated, never edited'. Fix: re-run the full matrix against the 21-test suite and restate CONTROL as 36, adding mutants for the diversity guards."
    },
    {
      "violation_type": "Circular_Reasoning",
      "action": "Guard-vacuity check per qa.md section 4c -- named the concrete mutation that would make each guard fail",
      "state": "SEVERITY WARN (not blocking: a genuine behavioral guard coexists). Two guards are pure SOURCE SCANS that observe no behaviour: (a) test_adapter_forwards_the_diversity_number_to_the_gate asserts `\"pbo_column_corr_mean\" in inspect.getsource(ad)` -- satisfiable by the token appearing in a comment or docstring, i.e. vacuity shapes #3 (literal kept, behaviour stripped) and #8 (comment-token trap). It is the ONLY coverage of the adapter forwarding hop: no test in this suite calls make_engine_backtest_fn and inspects the emitted dict (the real emission is strategy_backtest_adapter.py:273-274). (b) test_generate_report_still_does_not_emit_a_pbo has the same shape and would be defeated by getattr(analytics, 'compute_' + 'pbo'). Mitigating: the diversity NUMBER itself is genuinely behavioral -- test_diversity_number_discriminates_independent_from_duplicate_columns demands corr_mean < 0.1 for independent columns and > 0.99 for near-duplicates, and my M5 (constant) and M6 (keys dropped) mutations killed it.",
      "constraint": "code-review-trading-domain heuristic #17 illusory-guard; qa.md section 4c: 'a guard that cannot fail when its subject is broken does not count', WARN when a genuine behavioral guard coexists. Named fix: invoke the adapter's returned backtest fn against a stub engine and assert pbo_column_corr_mean / pbo_columns_diverse are present in the returned dict, so the forwarding hop is observed rather than grepped."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "verification_command_exit_code",
    "syntax",
    "python_lint_ruff_derived_scope",
    "backend_runtime_smoke_import",
    "mutation_test_6_mutants_all_killed",
    "guard_vacuity_check",
    "claim_audit_numeric_reproduction",
    "code_review_heuristics",
    "adversarial_worst_of_N_lenses",
    "third_conditional_counter",
    "verdict_shopping_check",
    "live_funnel_untouched"
  ],
  "harness_compliance_ok": false,
  "notes": "HARNESS-COMPLIANCE AUDIT (5 items): (1) Research-gate-before-contract PASS -- research_brief_82.22_82.23.md envelope reads external_sources_read_in_full=8, recency_scan_performed=true, gate_passed=true; mtime 10:11:37 precedes contract 10:13:42. (2) Contract-before-generate PASS by mtime chain (research 10:11:37 < contract 10:13:42 < first test-file write). (3) experiment_results present PASS. (4) Log-last PASS -- grep -cF 'phase=82.23' handoff/harness_log.md = 0 and masterplan 82.23 status=pending, retry_count=0, so nothing was logged or flipped ahead of this verdict. (5) No-verdict-shopping PASS on the literal test -- no prior evaluator_critique_82.23* exists, this is cycle 1, so CONDITIONAL is permitted (not the 3rd). The audit FAILS overall on a sixth, unlisted axis: the GENERATE artifact was not frozen for EVALUATE (see violation 2).\n\nCRITERION-BY-CRITERION: [1] NOT MET as worded -- see violation 1; the intent is served by a different, verified-correct mechanism. [2] MET -- test_checked_wrapper_refuses_instead_of_returning_zero (3 params) asserts pbo is None for N<2 / T<S*2 while test_raw_compute_pbo_returns_a_false_good_zero pins the 0.0 being defended against; mutation M1 (wrapper hands back the raw false-good 0.0) killed 5 tests. [3] MET -- test_gate_rejects_the_measured_incumbent_pbo carries the literal 0.7486 and asserts promoted is False with 'pbo_above_max' in the reason; my M4 (raise the live ceiling to 0.80, above 0.7486) killed 2, and the cheap-pass direction is blocked because M3 (always refuse) killed 3. Strongest criterion in the set. [4] MET as of the 10:26:10 edit -- compute_pbo_checked returns column_corr_mean / column_corr_max / columns_diverse alongside every non-refused pbo (analytics.py:248-271) and the adapter forwards pbo_column_corr_mean / pbo_columns_diverse (strategy_backtest_adapter.py:273-274); behaviorally guarded (M5, M6 killed), with the forwarding-hop WARN in violation 4.\n\nDETERMINISTIC RESULTS: immutable command `python -m pytest backend/tests/test_phase_82_23_pbo_in_gate.py -q` -> 21 passed, FINAL_EXIT=0, reproduced 3x. Runtime smoke: `import backend.autoresearch.gate, backend.backtest.analytics, backend.autoresearch.strategy_backtest_adapter, backend.backtest.quant_optimizer` -> IMPORT OK. Ruff on the git-derived scope (5 files, non-empty set asserted, xargs not an unquoted var) -> exit=1 with 3 F401 in quant_optimizer.py (os, compute_deflated_sharpe, GeminiClient). I INDEPENDENTLY REPRODUCED Main's pre-existing claim by linting `git show HEAD:backend/backtest/quant_optimizer.py` -- identical 3 errors at HEAD, so this is not a regression from this diff. New/untracked files (both test files + scripts/qa/check_optimizer_best_provenance.py) lint clean, exit 0.\n\nNO UNINTENDED PRODUCTION CHANGE: `git status --short | grep backend/(services|tools|agents)/` returns nothing -- live funnel untouched, CONFIRMED independently. The PromotionGate change is additive and I verified that behaviorally, not by reading the comment: test_gate_is_unchanged_for_producers_that_do_not_report_n passes, and M2 (strip the new branch) kills exactly the two tests that exercise it, so the new branch is load-bearing only when pbo_n_trials is present.\n\nNOTE-LEVEL, non-blocking: (a) compute_pbo_checked's docstring still documents the old 5-key return shape and omits the three diversity keys -- stale doc introduced by the mid-evaluation edit. (b) The refusal branches return only 5 keys while the success branch returns 8, so a consumer doing checked['column_corr_mean'] on a refusal gets a KeyError; the adapter is safe because it returns early on refusal, but the shape is non-uniform. (c) Residual money-path risk left unfixed and worth queueing: backend/services/promotion_gate.py:53 still reads `float(challenger.get('pbo', 0.0))` -- the fail-OPEN default against PBO_CEILING=0.5 at :37. I re-derived Main's narrow claim and it REPRODUCES (grep for evaluate_promotion finds zero callers outside the analytics.py comment), but the module is NOT dead: scripts/audit/promotion_gate_audit.py:25 and scripts/risk/promotion_gate.py:41 both import from it. I did not resolve which function encloses :53 -- flagging honestly rather than asserting.\n\nCREDIT WHERE DUE: Main self-flagged criteria 1 and 4 as unmet in the spawn prompt rather than hoping they passed, disclosed the fabricated-quote incident in the research gate, kept the provenance checker red on the live file as the honest state, and queued 82.25/82.26 as their own steps instead of folding them in. That is the scope-honesty behaviour this harness wants. The CONDITIONAL is driven by the immutable-criterion-1 mismatch, the mid-evaluation evidence mutation, and the stale matrix -- not by the engineering.\n\nRECOMMENDED PATH TO PASS: (1) operator decision on criterion 1, then a NEW step whose criterion names the sweep-level producer (compute_pbo_checked / the adapter) instead of generate_report -- 82.23's criterion cannot be edited; (2) freeze the tree, then re-run the full mutation matrix and restate CONTROL as 36 with mutants covering the three diversity tests; (3) replace the adapter source scan with a behavioral call-and-assert; (4) spawn a FRESH Q/A on the settled evidence."
}
```
