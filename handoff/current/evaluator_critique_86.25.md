# Evaluator critique -- step 86.25

**Cycle 1** -- Q/A verdict, Workflow rail run `wf_dd580823-63b`
(197,337 subagent tokens, 44 tool uses, 713s). Opus / effort max.
Returned 2026-08-10 and transcribed in the same turn it landed.

> **Transcribed VERBATIM by Main from the captured structured return.**
> Main records the verdict and never authors it.

## VERDICT: CONDITIONAL   (ok=false, harness_compliance_ok=true, certified_fallback=false)

## reason

All 6 immutable criteria are MET with evidence I independently reproduced (distribution 46/15/3/1 n=65, SELL 32/32 empty, immutable command exit=0 92 passed, ruff exit=0 over a git-derived 6-file scope, all 3 of Main's mutation cells re-killed plus 3 more of mine), and the shipped behaviour is correct and fail-safe on both seams; CONDITIONAL is issued for three non-blocking accuracy/scope defects in the artifacts and code comments, none of which requires a behaviour change: (W1) the production comments and experiment_results attribute the dead (A) branch to the unreachable analysis_id/round_trip_id anchor, but the operative cause is that `analyst_recommendation` is not a column of paper_trades (18 cols, verified) nor selected by LEDGER_FETCH_SQL, so the branch is dead by construction and fixing the queued round_trip_id defect would not make it resolve; (W2) the step's immutable -k filter collects 0 of the 16 new tests, so the S2 call-site and resolver guards sit entirely outside the command whose "92 passed, exit=0" is presented as the verification evidence; (W3) contract section 1 states 14 sources read in full and 44 URLs collected while the brief's own envelope says 7 and 29, both inflated ~2x in the table directly above the sentence claiming every load-bearing claim was re-verified (the gate still passes on the real numbers).

## violated_criteria

- `W1_mis_attributed_mechanism_for_the_dead_A_branch`
- `W2_immutable_command_excludes_the_new_suite`
- `W3_research_gate_numbers_do_not_reproduce`

## violation_details

### 1. Unjustified_Inference

**action** -- read backend/services/autonomous_loop.py S1 comment block + backend/services/recommendation_vocab.py header + experiment_results_86.25.md sections 2 and 7; then dumped the live paper_trades schema and _production_fns.LEDGER_FETCH_SQL

**state** -- All four artifacts explain 'the (A) branch runs for 0 rows' by the unreachable ANCHOR (analysis_id empty 32/32, round_trip_id one-sided 32/32 SELL vs 0/33 BUY). MEASURED: financial_reports.paper_trades has 18 columns and `analyst_recommendation` is not one of them; LEDGER_FETCH_SQL selects 10 named columns and does not include it; get_paper_trades does SELECT * over those same 18 columns. The shipped code performs no lookup at all -- it reads a dict key that no producer emits. My mutation cells M5 (S1 hardcodes the literal "UNKNOWN") and M8 (S1 reads risk_judge_decision through the resolver) both SURVIVED the full affected-area suite, and both are equivalent precisely because that key can never be present.

**constraint** -- A stated mechanism in a 'MEASURED' claim must be the operative cause (qa.md section 4b claim-auditing; feedback_structural_fix_needs_a_mechanism). The conclusion ('covers zero live rows') is true and even understated, but the reason given is wrong and tells the executor of the queued round_trip_id follow-up that this path will self-heal when the anchor lands. It will not.

### 2. Missing_Assumption

**action** -- source .venv/bin/activate && python -m pytest backend/tests/ -q -k "outcome_tracker or autonomous_loop or learn_loop" --collect-only | grep -c "test_phase_86_25_outcome_vocabulary_boundary.py::"

**state** -- 0. The immutable command's 92 tests come from the same 13 files as before; the only new-suite test inside its scope is the rewritten test_phase_86_25_empty_risk_judge_decision_becomes_UNKNOWN_not_HOLD, which lives in test_phase_35_1_learn_loop_writer.py and matches on 'learn_loop'. My mutation cells M2 (revert S2) and M3 (sentinel to HOLD) are killed ONLY by tests in the excluded file, so the step's own verification command stays green under both.

**constraint** -- The verification command presented as evidence ('92 passed, exit=0') must be stated at its true scope. experiment_results section 5 does show the standalone 16-test run beneath it, so this is disclosed by juxtaposition but never stated; a reader is left to infer the two commands overlap. Fixable with one sentence, or by naming the module so the -k filter matches.

### 3. Contradiction

**action** -- diff contract_86.25.md section 1 research-gate table against the JSON envelope at the end of handoff/current/research_brief_86.25.md, and against the brief's own URL tally line

**state** -- Contract states 'sources read in full | 14' and 'URLs collected | 44'. The brief's envelope states external_sources_read_in_full 7, snippet_only_sources 22, urls_collected 29, and its own tally line reads '7 read in full + 22 snippet-only = 29 unique URLs collected'. Both contract figures are inflated roughly 2x, in the direction that makes the gate look stronger. The brief's other cited facts DO reproduce: 38,615 chars (wc -c = 38615), STATUS: COMPLETE, gate_passed true, recency_scan_performed true.

**constraint** -- Every numeric claim must reproduce against the artifact it summarises (qa.md section 4b). This is the one table sitting directly above the contract's sentence 'Every load-bearing internal claim below was nonetheless re-verified by Main against source'. The gate itself still PASSES on the real numbers (7 >= 5 sources, 29 >= 10 URLs, recency scan performed), so this is a transcription defect, not a gate failure.

## checks_run

- qa_md_read_in_full_at_runtime
- harness_compliance_audit_5_items
- immutable_verification_command_exit_0_92_passed
- immutable_command_scope_collect_only_measured
- python_lint_gate_ruff_F821_F401_F811_git_derived_scope_exit_0
- backend_runtime_import_smoke_3_modules
- live_behaviour_probe_compute_outcomes_on_live_shaped_row
- bigquery_redemption_of_every_number_distribution_schema_awareness_empty_action
- outcome_tracking_row_provenance_cross_check_evaluated_at_0400
- dedup_key_and_row_count_invariance_check
- independent_mutation_matrix_8_cells_control_first_in_memory
- survivor_behavioural_differential_adjudication
- contract_completeness_criterion_by_criterion_mapping
- claim_audit_of_prose_and_verbatim_blocks
- guard_vacuity_check_section_4c
- code_review_heuristics
- consumer_contract_grep_outcome_tracking_recommendation
- research_brief_envelope_cross_check
- harness_log_3rd_conditional_counter
- write_first_wip_verdict_file

## notes

Cycle 1; harness_log grep -cF "86.25" = 0, so this is the FIRST CONDITIONAL for this step-id -- the 3rd-CONDITIONAL auto-FAIL rule does not fire. Harness compliance is clean on all five items: research brief on disk for this step with gate_passed true and the floors cleared on its real numbers (7 sources / 29 URLs -- see W3), contract mtime 13:01:11 strictly before the production .py mtimes 13:08:05-13:08:12 and before experiment_results 13:09:33 with the commit order agreeing (64d20023 then 8baecb49), both handoff artifacts present, step still status=pending with no harness_log entry, and no prior verdict to shop. I judge the deliberate reuse of research gate wf_a3511e6a-c28 ACCEPTABLE: the artifact exists for this step id, the contract cites it, and I re-derived the load-bearing internal claims myself rather than accepting the report -- F2 (directionally_correct never persisted: save_outcome passes 8 kwargs into a 9-column table and that is not one of them) and F4 (two seams) both reproduce.

WHAT I ATTACKED AND WHAT SURVIVED THE ATTACK. (1) The four P1 numbers all reproduce independently: analysis_results has no analysis_id column (the committed script still exits 1 with BadRequest 400 "Name analysis_id not found inside a"), SELL analysis_id 0/32 vs BUY 33/33, round_trip_id one-sided 32/32 SELL vs 0/33 BUY. The design rationale stands. (2) The (A)-branch disclosure is adequate as far as it goes but its stated cause is wrong -- that is W1, and it is the finding I consider most worth acting on. (3) Criterion 5's premise substitution is legitimate and well-evidenced; the criterion's own wording is "in whatever is persisted", directionally_correct is not persisted, and the answer lands on the recommendation column asserted at the write chokepoint. (4) The rewrite of test_phase_35_1_empty_risk_judge_decision_coerced_to_hold is NOT a weakened guard: the replacement has four assertions instead of one, still drives the real _learn_from_closed_trades, and my independent M1 cell -- reverting the S1 call site to the exact pre-fix two-line coercion -- turns it red. The old test asserted the defect (rec_arg == "HOLD"), so it had to change. (5) Both seams are fixed and each call site was reverted INDEPENDENTLY (M1 and M2), so a one-seam fix could not hide. (6) The row-count claim is verified with data, not accepted: action is REQUIRED and empty on 0 of 65 rows, so the pre-fix `risk_judge_decision or action` could never be falsy and the `if not recommendation` skip never fired before and cannot fire now; the dedup key is (ticker, analysis_date) only, so the three existing rows are still filtered out and no duplicate is inserted. Row count genuinely unchanged, no close dropped.

Criterion 3 is stronger than Main's evidence for it: rather than the source grep in live_check section 3, I read the live rows -- all three carry evaluated_at 2026-08-08T04:00:02.013552+00:00 (the 04:00 UTC cron) with price_at_recommendation and beat_benchmark NULL, which only build_outcome_row writes. That is direct provenance for S2.

Two additional NOTE-level items for Main's follow-up, neither degrading the verdict: (N1) a stale comment blessing the just-fixed defect survives one file over at backend/slack_bot/jobs/_production_fns.py:404-405 -- "The risk judge's decision is the recommendation that was acted on; fall back to the trade action, never to None" -- inside the same S2 pipeline whose sibling comment Main corrected for precisely this reason. (N2) live_check section 1 is labelled verbatim but truncates the Q3 traceback before the BadRequest line that is the actual evidence for the no-analysis_id-column claim; I reproduced the full error.

Mutation matrix run BY ME, control-first, via sys.modules injection so nothing was written to the tree and no restore was needed: CONTROL 84 passed exit 0; M1 revert-S1 KILLED; M2 revert-S2 KILLED (3 tests); M3 sentinel-to-HOLD KILLED (3); M4 widen canonical_recommendation so APPROVE_* returns BUY KILLED (2); M6 re-introduce the action leak THROUGH the resolver KILLED (2); M7 option-B skip-on-unknown KILLED (3); M5 S1 hardcodes the literal "UNKNOWN" SURVIVED; M8 S1 reads risk_judge_decision through the resolver SURVIVED. I adjudicated both survivors for a behavioural differential rather than reporting them as defects: they are equivalent on every value the source schema can produce today, and their equivalence is itself the proof of W1.

TO CLEAR THIS TO PASS, no code behaviour needs to change. Correct the (A)-branch rationale in autonomous_loop.py, recommendation_vocab.py and experiment_results to name the absent column as the cause and say what would have to be plumbed; state the immutable command's true scope (or rename the module so -k matches); and correct the contract's research-gate figures to 7 and 29. I did NOT take a Playwright capture: the step makes no UI claim and touches no frontend file, so section 1c does not bind. The backend has not been restarted, so these changes are committed but not in force in the running process -- Main disclosed this and it is consistent with the batch-restarts-to-session-end rule.
