# Evaluator critique -- Step 75.14 (Q/A cycle 1: CONDITIONAL)

Q/A launch: Workflow `wf_b77f5c78-a1b` (qa-verdict.js, opus/max). Verdict
transcribed VERBATIM below -- Main records, never authors. Cycle-2 will
fix the three violations (all test/doc-only per the verdict) and spawn a
FRESH Q/A on the changed evidence per the canonical cycle-2 flow.

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "All six immutable criteria are MET and independently verified against the REAL shipped code (not just the author's tests): (1) format_skill escapes '{{'->'{ {' in values [SSTI kill; template's own placeholders still expand exactly once, verified behaviorally] + fences on market/deep-dive/bull/bear external blocks + unconditional SECURITY RULE; (2) four seams aligned to real Pydantic model_fields, schemas.py + debate.py:327-328 backfill byte-untouched (boundary held); (3) operator decision note present, covers sizing-input change AND the research-added frontend-goes-empty impact, token SCHEMA-EXTEND-75.14; (4) Files-API sends document+data-only (never full template inline; document dropped when no data_prompt) + comments corrected all 3 sites; (5) portfolio_sector_exposure tagged [INTERNAL] via _FACT_LEDGER_SOURCE_MAP, [YFIN] default; (6) risk-judge parse-fail fallback byte-identical APPROVE_REDUCED on default-False (else-branch identical to legacy dict, verified), REJECT/0/EXTREME on True, loud P1 warning OUTSIDE the conditional on both paths with judge_text[:1500], default proven False. Harness compliance clean (research gate passed 8>=5 sources + recency scan, mtime order research<contract<results, log-last OK, masterplan still pending, cycle-1 not a re-spawn). Deterministic reproduced: immutable cmd 18/18 exit 0; ruff F821/F401/F811 clean over 6 explicit files; all 5 changed backend modules import; full suite 10 failed/1446 passed with all 10 failures independently confirmed in modules 75.14 did NOT touch (lite-path _LITE_RISK_JUDGE_TEMPLATE constants + environmental backend-log evidence tests) => zero regressions. CONDITIONAL (not PASS) is driven by three test-quality/evidence-accuracy gaps on a P1 money-path step, chiefly a mutation-resistance hole: the criterion-6 behavioral test re-implements the warning instead of executing the real risk_debate branch, and the source-scan lockstep would NOT catch a True/False routing inversion on the money-path DARK flag. Not FAIL: no immutable criterion is missed, money path byte-identical on default, no unintended production change; the required fixes are test/doc-only (shipped production code is correct).",
  "violated_criteria": [
    "mutation_resistance_moneypath_routing_guard",
    "verbatim_stat_does_not_reproduce",
    "dead_tautology_assert"
  ],
  "violation_details": [
    {
      "violation_type": "Missing_Assumption",
      "action": "test_fallback_flag_routes_verdict re-implements the parse-fail warning inline via a monkeypatched SimpleNamespace(get_settings) instead of executing the real run_risk_debate 'if not judge_result:' branch; the only source-level guard (test_fallback_source_has_both_paths_and_loud_warning) asserts both dicts are PRESENT and warn_idx<if_idx but never that _parse_fail_reject True->REJECT/0 and False->APPROVE_REDUCED/3",
      "state": "criterion-6 money-path DARK flag (paper_risk_judge_parse_fail_reject) governs whether an unparseable Risk Judge approves 3% NAV or rejects; shipped risk_debate.py routing is CORRECT (verified by reading the diff: if->REJECT/0/EXTREME, else->byte-identical legacy APPROVE_REDUCED), but an if/else ROUTING INVERSION would pass the entire 18-test suite undetected",
      "constraint": "feedback_mutation_test_guards_and_fixtures + qa.md 4b: a guard that can't fail doesn't count; a money-path DARK-flag routing must be protected by a test that executes the real branch (or a source assert that pins True->REJECT / False->APPROVE), not a re-implementation that would pass even if the real routing were swapped"
    },
    {
      "violation_type": "Contradiction",
      "action": "live_check_75.14.md section 2 presents a labeled-verbatim '$ git diff --stat HEAD -- backend/ | tail -1' capture reading '8 files changed, 170 insertions(+), 69 deletions(-)' (experiment_results headline repeats '8 modified backend files (+170/-69)')",
      "state": "re-running that exact command against the shipped tree reproduces '9 files changed, 171 insertions(+), 70 deletions(-)'; the delta is exactly the disclosed debate_stance.md 2-line non-delivered-prose change that landed after the capture was taken; direction under-states and experiment_results discloses the 9th file in prose, but the labeled-verbatim block was not regenerated",
      "constraint": "qa.md 4b: a 'verbatim' capture must be regenerated, never left stale; a number in a verbatim artifact that does not reproduce is a flagged finding (fix: re-run the git diff --stat and reconcile the '8' headline to 9/171/70)"
    },
    {
      "violation_type": "Invalid_Precondition",
      "action": "backend/tests/test_phase_75_prompt_contracts.py line 199: assert \"98.5%\" not in src.split(\"CORRECTED phase-75.14\")[0].rsplit(\"\\n\", 200)[-1] or True",
      "state": "the trailing 'or True' makes the assertion unconditionally pass regardless of source content; the meaningful checks in test_phase_25_d9_comment_corrected are the two following asserts ('CORRECTED phase-75.14' in src, 'billed' in src), so no bug is let through, but the line is dead",
      "constraint": "feedback_mutation_test_guards_and_fixtures: a tautology guard is one of the 5 flagged anti-patterns; remove the '... or True' dead assert"
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_5item_audit",
    "research_gate_envelope_8sources_recency_gatepassed",
    "contract_before_generate_mtime_order",
    "log_last_masterplan_pending",
    "no_verdict_shop_cycle1",
    "immutable_command_pytest_18of18_exit0",
    "boundary_schemas_and_debate_byte_untouched",
    "ruff_F821_F401_F811_explicit_6files_exit0",
    "backend_runtime_smoke_5_modules_import",
    "full_suite_10failed_1446passed_reproduced",
    "regression_failset_membership_all10_in_untouched_paths",
    "criterion1_ssti_escape_and_fence_behavioral",
    "criterion1_standing_rule_unconditional_in_bull_prompt",
    "criterion2_four_seams_real_model_fields_derivation",
    "criterion3_operator_decision_note_sizing_plus_frontend",
    "criterion4_files_api_data_only_request_shape_real_assembly",
    "criterion4_phase25d9_comment_corrected",
    "criterion5_fact_ledger_internal_tag",
    "criterion6_darkflag_else_byteidentical_true_reject_default_false",
    "mutation_matrix_independent_review",
    "claim_audit_livecheck_gitdiffstat",
    "worktree_clean_head_baseline_attempt_env_confounded",
    "zsh_wordsplit_ruff_falsepass_caught_and_corrected"
  ],
  "harness_compliance_ok": true,
  "notes": "Excellent core engineering: five legs all correct, money path byte-identical on default (else-branch identical to the legacy APPROVE_REDUCED dict, verified line-by-line), boundary respected (schemas.py + debate.py:327-328 untouched), and genuinely behavioral tests for criteria 1/4/5 plus real Pydantic model_fields derivation for criterion 2 and a real settings-default test for criterion 6. Scope honesty is strong: experiment_results discloses the late-landing debate_stance.md (non-delivered prose, zero Python), the data_prompt path being capability-only (no live caller yet -- honestly satisfies criterion 4 because the criterion tests request SHAPE, which the test drives directly), the invalid-first-M3 mutant, the quadruple restart debt (75.8/75.10/75.11/75.14 -- prompt-layer changes load on next backend restart), and no live LLM call. The three CONDITIONAL items are test/doc-only fixes -- NO production code change is required; the shipped risk_debate.py/prompts.py/llm_client.py are correct. Recommended cycle-2 (Main fixes, then spawn a FRESH Q/A on changed evidence): (1) strengthen the criterion-6 guard so a routing inversion is caught -- either execute the real run_risk_debate parse-fail branch, or add source asserts pinning that the REJECT dict sits under 'if _parse_fail_reject:' and APPROVE_REDUCED under the else; (2) regenerate live_check section 2 git-diff-stat to 9 files/171/70 and reconcile the experiment_results '8 modified backend files' headline; (3) delete the 'or True' tautology at test line 199. Independent process caution for the record: my own first ruff invocation hit the zsh unquoted-newline-variable word-split trap that qa.md 1a documents (ruff printed 'All checks passed!' exit 0 while linting zero of the 5 files) -- I caught it via the 'Failed to lint ... No such file or directory' warning and re-ran with explicit file args, which genuinely passed. This is cycle 1 for 75.14 (no prior CONDITIONAL), so the 3rd-CONDITIONAL auto-FAIL rule does not apply; certified_fallback=false (retry_count 0 << max_retries 3). A left-over read-only git worktree at scratchpad/wt_head_* was pruned (git worktree prune); it was environmentally confounded by a missing backend/.env (pydantic Settings collection errors) so the definitive baseline-membership proof came from mapping all 10 failing test IDs to 75.14-untouched modules instead."
}
```

## Follow-up (cycle 2) -- Main's fixes on the three violations, before the fresh Q/A

1. Violation 1 (money-path routing guard): fallback EXTRACTED to
   `risk_debate._judge_parse_fail_fallback(judge_text)`; criterion-6 test
   now executes the REAL function both ways + lockstep branch assert.
   Routing-inversion mutant (M9) KILLED (1 failed / 17 passed), suite
   green post-restore.
2. Violation 2 (stale verbatim stat): regenerated -- 9 files / +177 / -70;
   experiment_results headline reconciled; live_check section 3 re-run
   for the cycle-2 tree (10 failed / 1446 passed, baseline-identical).
3. Violation 3 (tautology): the `or True` dead assert deleted.

Evidence CHANGED (code + tests + artifacts); a FRESH Q/A is being spawned
per the canonical cycle-2 flow (not verdict-shopping).


## Cycle 2 verdict -- Workflow `wf_dbfd2ad6-7ab` (fresh Q/A on changed evidence): PASS

Transcribed VERBATIM from the captured structured-output return value.
(Post-verdict, Main applied the verdict's non-blocking cosmetic note --
the no-op .replace(x,x) at test line 281 cleaned to a plain `token in
src` assert, 18/18 green after -- the verdict itself is untouched.)

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "Cycle-2 PASS: the three cycle-1 CONDITIONAL violations are all genuinely closed on CHANGED evidence, and all six immutable criteria remain MET with the P1 money-path default byte-identical. (1) Routing guard: the parse-fail fallback is EXTRACTED to risk_debate._judge_parse_fail_fallback(judge_text) (else-branch byte-identical to the removed legacy APPROVE_REDUCED/0.5/3/MODERATE dict; run_risk_debate routes through it: `judge_result = _judge_parse_fail_fallback(judge_text)`); the new test_fallback_routing_executes_real_branch_both_ways EXECUTES the real function both ways + a source lockstep assert. I independently REPRODUCED the exact cycle-1 mutant (in-memory `if _parse_fail_reject:`->`if not _parse_fail_reject:` on the real source): mutant off->REJECT (test expects APPROVE_REDUCED) and mutant on->APPROVE_REDUCED (test expects REJECT) — both asserts fail => KILLED; the REAL function routes off->APPROVE_REDUCED/3/MODERATE, on->REJECT/0/EXTREME with the loud P1 warning firing on BOTH paths preserving judge_text[:1500]. (2) Stale stat regenerated: `git diff --stat HEAD -- backend/` reproduces exactly 9 files/+177/-70; live_check + experiment_results headline reconciled. (3) Tautology deleted: grep 'or True' exit 1. Deterministic: immutable pytest 18 passed exit 0; boundary schemas.py + debate.py byte-untouched (empty diff); full suite 10 failed/1446 passed — byte-identical baseline with ALL 10 independently confirmed pre-existing/environmental (backend-log-evidence tests, operator .env flag-state paper_data_integrity/reject_binding ON, lite-path string is-identity, 150>150 latency); test_phase_37_1_risk_judge_schema + test_phase_66_2_risk_judge_shape PASS => boundary held and extraction behavior-preserving; zero regressions. Six criteria all MET (SSTI escape+fences+unconditional rule; four seams aligned to real model_fields with schemas unchanged; operator decision note covering sizing+frontend+token; Files-API data-only + phase-25.D9 comment corrected; fact-ledger [INTERNAL] tag; risk-judge DARK flag default False proven). Harness compliance clean 5/5. Not a 3rd CONDITIONAL (only one prior).",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_5item_audit",
    "research_gate_envelope_8sources_recency_gatepassed",
    "contract_before_generate_mtime_order_0325_0327_0400",
    "log_last_no_7514_result_row_masterplan_pending",
    "no_verdict_shop_cycle2_changed_evidence_one_prior_conditional",
    "immutable_command_pytest_18passed_exit0",
    "boundary_schemas_and_debate_byte_untouched_empty_diff",
    "change_surface_9files_177_70_reproduced",
    "tautology_or_true_deleted_grep_exit1",
    "fix1_extraction_else_byteidentical_to_legacy_dict",
    "fix1_run_risk_debate_routes_through_extracted_fn",
    "fix1_routing_inversion_mutant_reproduced_KILLED_inmemory",
    "fix1_real_fn_off_approve3_on_reject0_warning_both_paths",
    "fix2_gitdiffstat_9_177_70_matches_livecheck",
    "criterion1_ssti_escape_fence_unconditional_rule_tests_pass",
    "criterion2_four_seams_real_model_fields_schemas_unchanged",
    "criterion3_operator_decision_note_sizing_frontend_token",
    "criterion4_files_api_dataonly_and_phase25d9_comment",
    "criterion5_fact_ledger_internal_tag",
    "criterion6_darkflag_default_false_proven_true_reject0",
    "full_suite_10failed_1446passed_baseline_identical",
    "regression_all10_failures_confirmed_preexisting_environmental",
    "existing_risk_judge_37_1_schema_and_66_2_shape_pass"
  ],
  "harness_compliance_ok": true,
  "notes": "Independent, read-only verification (no production files mutated). The cycle-1 mutation-resistance hole is definitively closed: I reproduced the if/else routing inversion by mutating the REAL function source in-memory and running the test's assertions against it — the mutant is KILLED by test_fallback_routing_executes_real_branch_both_ways (a behavioral test that executes the extracted fallback, not a re-implementation). Scope honesty is strong: experiment_results/live_check disclose the quadruple restart debt (75.8/75.10/75.11/75.14 load on next backend restart), the data_prompt Files-API path shipping as capability-only (no live caller yet), and no live LLM call. The P1 money-path is fail-safe by construction: default OFF is byte-identical to the legacy silent APPROVE_REDUCED/3% fallback, and even the new True-path REJECT only binds when shape_fix/reject_binding is also ON — documented in the flag description. All 10 full-suite failures were independently traced to pre-existing/environmental causes (2 backend-log-evidence greps needing a live backend, operator-.env flag-state on paper_data_integrity + paper_risk_judge_reject_binding — a DIFFERENT flag from 75.14's — lite-path string is-identity brittleness on byte-identical content, and a 150>150 latency boundary); none are 75.14 regressions and the count is byte-identical to the claimed baseline. NON-BLOCKING NOTE for opportunistic cleanup (does NOT affect the verdict): backend/tests/test_phase_75_prompt_contracts.py:281 has `assert token.replace('\\\"risk_limits\\\": ', '\\\"risk_limits\\\": ') in src` where the .replace(x,x) is a no-op — cosmetic dead code, but unlike the removed `or True` the assertion `token in src` still carries real content, so it is not a tautology. Cycle history: one prior CONDITIONAL (cycle 1); this is cycle 2, so the 3rd-CONDITIONAL auto-FAIL rule is not engaged. certified_fallback=false (retry_count 0 << max_retries 3). This verdict is the deliverable for Main to transcribe VERBATIM into evaluator_critique.md + evaluator_critique.json; Main owns the harness_log append (log-last) and the masterplan status flip."
}
```
