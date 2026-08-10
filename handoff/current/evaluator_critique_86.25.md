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

---

# CYCLE 2 VERDICT -- Q/A, Workflow rail run `wf_a59e0a03-8c2`

(168,355 subagent tokens, 36 tool uses, 573s). Opus / effort max.
Returned 2026-08-10, transcribed in the same turn it landed.

> **Transcribed VERBATIM by Main from the captured structured return.**

## VERDICT: CONDITIONAL   (ok=false, harness_compliance_ok=true, certified_fallback=false)

## reason

All 6 immutable criteria are MET, harness compliance is clean 5/5, there is no unintended production change, and two of the three cycle-1 findings are properly fixed and reproduce EXACTLY (W2: the rename brings 16/16 new tests inside the unamended immutable filter, 108 passed exit=0, and I re-ran both mutants in-process -- S2 3 failed/105 passed, V1 10 failed/98 passed, byte-identical to Main's claim; W3: contract now 7/22/29 matching the brief's envelope and its tally line). CONDITIONAL is issued for one finding: the W1 remediation is INCOMPLETE and the artifact claims it is complete -- `backend/services/recommendation_vocab.py` was named explicitly in the cycle-1 TO-CLEAR-TO-PASS list and `git diff 8baecb49 HEAD -- backend/services/recommendation_vocab.py` is EMPTY, so its :164-169 header still gives the refuted anchor as the operative cause, while `experiment_results` W1 states "Corrected in autonomous_loop.py, nightly_outcome_rebuild.py and here", silently substituting a file that was not on the list; and in `autonomous_loop.py` the refuted sentence survives VERBATIM seven lines above the block that calls it "an earlier version of this comment". I judge Main's M5/M8 position CORRECT with executed evidence rather than reasoning: my new cells N1 (S1 reads trade action through the resolver) and N3 (S2 reads action) both KILLED, so every argument regression that can fabricate a direction dies -- the S1 guard is NOT too weak, and M8 survives only because the approval vocabulary cannot canonicalise, which is the boundary property the step built.

## violated_criteria

- `W1_remediation_incomplete_recommendation_vocab_untouched_and_refuted_text_survives_in_autonomous_loop`

## violation_details

### 1. Contradiction

**action** -- git diff 8baecb49 HEAD -- backend/services/recommendation_vocab.py ; git log --oneline -- backend/services/recommendation_vocab.py ; sed -n '160,175p' backend/services/recommendation_vocab.py ; sed -n '3417,3448p' backend/services/autonomous_loop.py

**state** -- The cycle-1 critique named FOUR artifacts carrying the mis-attribution and its TO-CLEAR-TO-PASS line reads verbatim 'Correct the (A)-branch rationale in autonomous_loop.py, recommendation_vocab.py and experiment_results to name the absent column as the cause'. MEASURED: the diff for recommendation_vocab.py between cycle-1 (8baecb49) and HEAD is EMPTY -- the file was never touched in cycle 2, its newest commit is still 8baecb49 -- and lines 164-169 still read 'MEASURED 2026-08-10, and it is why this resolver exists rather than a lookup: the analyst recommendation is reachable for 0 of 32 SELL rows. analysis_id is empty on 32/32 SELLs (BUYs carry it 33/33), and round_trip_id is ONE-SIDED ... so a SELL cannot reach its BUY leg either.' That is the exact anchor mechanism W1 ruled not operative, and its '0 of 32 SELL rows' framing additionally implies BUY rows differ -- they do not: my repo-wide grep --include='*.py' shows the ONLY emitter of the key analyst_recommendation is the test fixture at test_phase_86_25_outcome_tracker_vocabulary_boundary.py:156, and LEDGER_FETCH_SQL (_production_fns.py:229-231) selects ten named columns without it. SEPARATELY, in autonomous_loop.py:3417-3423 the refuted sentence survives verbatim ('Nothing is what is available -- MEASURED 2026-08-10: the anchor is reachable for 0 of 32 SELL rows ... So this resolves to UNKNOWN today') seven lines ABOVE the cycle-2 block that describes it as 'An earlier version of this comment' -- it is not earlier, it is still present, so the file states both the refuted and the corrected mechanism as MEASURED fact. Meanwhile experiment_results_86.25.md W1 asserts 'Corrected in autonomous_loop.py, nightly_outcome_rebuild.py and here', substituting a file that was never on the remediation list for the one that was.

**constraint** -- A remediation claim must reproduce against the files it names, and a correction must supersede the refuted text rather than sit beside it (qa.md 4b claim-auditing; feedback_recheck_prior_remediation_list -- re-derive the PRIOR cycle's list yourself, follow-ups have SUBSTITUTED the file set before). The live consequence is the one W1 was raised to prevent: recommendation_vocab.py is the boundary module, and a future reader of it -- specifically the executor of the queued round_trip_id step -- is still told this path self-heals when the anchor lands. It will not; it needs a producer change.

### 2. Invalid_Precondition

**action** -- git show 8baecb49:backend/slack_bot/jobs/nightly_outcome_rebuild.py | sed -n '70,92p' ; compare with backend/slack_bot/jobs/nightly_outcome_rebuild.py:88-91 at HEAD

**state** -- The cycle-2 correction block was pasted verbatim into BOTH production files. In nightly_outcome_rebuild.py it asserts 'An earlier version of this comment blamed the unreachable ANCHOR -- analysis_id empty on 32/32 SELLs, round_trip_id one-sided 32/32 SELL vs 0/33 BUY.' MEASURED FALSE for that file: its cycle-1 comment block never mentioned the anchor at all -- it ran from 'phase-86.25 (S2). WAS: t.get("risk_judge_decision") or t.get("action")' through '...only the label changes, from a fabricated direction to an honest absence' with no anchor claim in it.

**constraint** -- A production comment asserting the repo's own history must be true of the file it is in. This is a NOTE-level copy-paste artifact, not a blocker, but it is a fresh false historical statement introduced by the very commit that was fixing a false-statement finding -- worth one line to correct while the vocab.py residual is being fixed.

## checks_run

- qa_md_read_in_full_at_runtime
- write_first_wip_verdict_file
- harness_compliance_audit_5_items
- immutable_verification_command_exit_0_108_passed
- immutable_command_scope_collect_only_16_of_16_new_tests
- python_lint_gate_ruff_F821_F401_F811_git_derived_6_file_scope_exit_0
- backend_runtime_import_smoke_3_modules
- live_resolver_probe_over_full_measured_value_set
- git_status_unintended_change_check
- independent_mutation_matrix_6_cells_control_first_in_memory_no_tree_writes
- survivor_behavioural_differential_adjudication_M5_M8
- w1_mechanism_independently_rederived_ledger_sql_and_producer_grep
- w2_rename_remedy_verified_mutants_die_inside_immutable_command
- w3_research_gate_envelope_cross_check
- historical_filename_corruption_sweep_both_spellings
- prior_remediation_list_rederivation
- contract_completeness_criterion_by_criterion_mapping
- claim_audit_of_prose_and_verbatim_blocks
- guard_vacuity_check_section_4c
- code_review_heuristics
- harness_log_3rd_conditional_counter

## notes

CYCLE 2. Harness compliance clean on all five: research_brief_86.25.md on disk (envelope gate_passed true, 7 sources >= 5, 29 URLs >= 10, recency scan performed) with mtime 10:32 strictly before the contract; contract-before-generate holds on the cycle-1 chain (contract 13:01:11 < .py 13:08 < experiment_results 13:09, commit order 64d20023 then 8baecb49 agreeing); both handoff artifacts present; log-last respected (grep -cF "86.25" handoff/harness_log.md = 0, masterplan status still "pending"); and this is NOT verdict-shopping -- f71030b8 changed the evidence (two comment blocks, one rename, one contract table, artifact appends). 3rd-CONDITIONAL counter: zero logged CONDITIONALs for this step-id, so this is #2 and the auto-FAIL rule does not fire. Section 1c does not bind: no frontend file is touched and the step makes no UI claim, so I took no Playwright capture.

WHAT I RE-DERIVED RATHER THAN READ. Immutable command re-run by me: 108 passed, 3303 deselected, EXIT=0. --collect-only piped through grep -c for the new module: 16, so the rename genuinely brings all 16 tests inside the filter and the masterplan verification.command is byte-identical (no criterion amended -- I confirmed the rename is the RIGHT remedy, not a workaround, because it widens coverage rather than narrowing the claim). Lint over a git-derived 6-file scope via xargs (never an unquoted variable): "All checks passed!" exit 0. All three changed backend modules import in the venv, and the live resolver returns UNKNOWN for '', APPROVE_REDUCED, REJECT, APPROVE_HEDGED and None while returning STRONG_BUY for 'Strong Buy', with is_directional('UNKNOWN') False -- criterion 4 verified against the running code, not only by test. W1's CORRECTED mechanism is true: LEDGER_FETCH_SQL at _production_fns.py:229-231 selects exactly ten named columns without analyst_recommendation, and a repo-wide grep finds no production emitter of that key at all.

MUTATION MATRIX, MINE, CONTROL-FIRST, via sys.modules source injection so nothing was written to the tree and no restore was needed. CONTROL rc=0 108 passed. S2-revert (call site back to `risk_judge_decision or action`) KILLED 3 failed/105 passed; V1-sentinel (`else "HOLD"`) KILLED 10 failed/98 passed -- both reproduce Main's cycle-2 numbers exactly and both now die INSIDE the immutable command, which is the whole point of the W2 remedy. Then three cells of my own aimed squarely at the M5/M8 question: N1, S1 reads the trade ACTION through the resolver -- the fail-UNSAFE regression, because 'SELL' canonicalises to SELL and would persist a fabricated direction -- KILLED, 1 failed/107 passed; N3, the same regression at the S2 site -- KILLED, 3 failed/105 passed; N2, identical to the predecessor's M8 (S1 reads risk_judge_decision through the resolver) -- SURVIVED, reproducing the earlier result. ADJUDICATION: Main's position is CORRECT and I now have an executed differential for it rather than an argument. Every argument regression that can fabricate a direction dies; M8 and M5 survive only because the approval vocabulary cannot canonicalise, which is precisely the boundary property this step built. The S1 guard is NOT too weak. One nuance for the record, NOTE-level only: M8's equivalence is contingent on risk_judge_decision never overlapping the recommendation scale rather than true by construction -- if a producer ever wrote a directional token into that column the two would diverge, and no test would notice.

Item 5 (the self-inflicted blanket-sed error) checks out CLEAN. I swept both filename spellings across handoff/, backend/, scripts/, docs/ and the masterplan: the historical spelling correctly survives exactly where it is history (experiment_results:191 in the W2 narrative, and evaluator_critique_86.25.md:34 / .json:45 in the cycle-1 verbatim action), and the new spelling appears only where it is current. No other artifact carries a corrupted historical reference. I did note that the two "verbatim" pytest invocations at experiment_results:113 and live_check:75 were rewritten to the new filename by that same sed, which technically edits a block labelled verbatim -- but the command as written now reproduces (16 passed) and the rename is disclosed three sections down, so it is a NOTE, not a finding.

TWO ADDITIONAL NOTES FOR MAIN'S FOLLOW-UP, neither degrading the verdict. (N1) The cycle-1 critique's N1 item still stands: the stale comment at backend/slack_bot/jobs/_production_fns.py:404-405 blessing the just-fixed defect was not addressed in cycle 2 and was not part of my capped finding since it was raised as NOTE-level then too. (N2) commit 8baecb49 swept in three files belonging to step 86.30 (research_brief_86.30.md, the researcher's project_degraded_branch_direction_86_30.md, and a MEMORY.md line) under the 86.25 commit message -- cross-attribution of the `git add -A` class; harmless here because none of it is production code, but it is the pattern that has bitten this repo before.

TO CLEAR THIS TO PASS, no behaviour and no test needs to change. Amend the (A)-branch rationale in backend/services/recommendation_vocab.py:164-169 to name the absent column as the cause; in backend/services/autonomous_loop.py:3417-3423 delete or explicitly strike the superseded sentence rather than leaving it above its own correction; correct experiment_results W1's file list so it names the files actually corrected; and drop or fix the "an earlier version of this comment blamed the unreachable ANCHOR" clause in nightly_outcome_rebuild.py, where it was never true.

WRITE-FIRST: my crash-survival record is at /Users/ford/.openclaw/workspace/pyfinagent/.claude/agent-memory/qa/verdicts/verdict_wip_86.25.md, marked COMPLETE. It is evidence, not a verdict -- this structured return is the deliverable. No write was blocked; I wrote nothing outside that one permitted path.
