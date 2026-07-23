# Evaluator Critique -- masterplan step 75.4

**Step**: 75.4 -- Audit75 S4, skill-prompt delivery integrity
**Cycle**: 1
**Date**: 2026-07-20
**Q/A launch**: Workflow structured-output (`.claude/workflows/qa-verdict.js`), run `wf_8d493697-c73`,
model claude-opus-4-8[1m], 30 tool calls, 154,869 tokens, 650s.

> **Provenance**: the JSON below is the Q/A agent's return value, transcribed
> **VERBATIM** by Main. Main records the verdict; Main never authors it. No editorial
> edits, no paraphrase, no reordering.

---

## Verdict: CONDITIONAL (ok=False)

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "All 6 immutable criteria independently verified MET (verification cmd 25 passed exit=0; byte-identity re-derived from git HEAD with 0 non-blank body drift; 8/3 file counts recounted; quant_strategy.md byte-identical by shasum and correctly excluded; masterplan criteria byte-identical to HEAD; zero unintended production change; ruff findings byte-identical to HEAD so 0 new; runtime smoke clean; 13 full-suite failures reproduced and all confirmed unattributable). CONDITIONAL is issued for a 6th instance of this project's repeat vacuous-guard defect plus a falsified headline claim: test_critic_degraded_flag_is_present_on_every_return_path asserts only target.count('critic_degraded') >= 5 and len(dict_returns) >= 4, binding neither to the other, so I produced 5 surviving mutants -- removing the flag from any 1, 2 or 3 of the 4 return paths all SURVIVE, and a maximally vacuous mutant with the flag on ZERO return paths but comment-only mentions also SURVIVES. Two supporting assertions in test_unparseable_critic_verdict_is_not_silently_treated_as_pass are bare substring checks satisfiable by a comment: deleting the entire 16-line retry block while leaving a '# Critic-Retry' comment SURVIVES, and reinstating fail-open semantics under a reworded log message SURVIVES. The underlying CODE is correct (I confirmed by AST that all 4 value-returns do carry the flag), so this is weak-guard-not-broken-feature; no immutable criterion is violated, which is why this is CONDITIONAL and not FAIL.",
  "violated_criteria": [
    "mutation_resistance_vacuous_guard",
    "anti_rubber_stamp_falsified_mutation_claim",
    "scope_honesty_minor_overstatements"
  ],
  "violation_details": [
    {
      "violation_type": "Unjustified_Inference",
      "action": "test_critic_degraded_flag_is_present_on_every_return_path (backend/tests/test_phase_75_skill_delivery.py:265-290) asserts target.count('critic_degraded') >= 5 AND len(dict_returns) >= 4, never binding a return path to the flag",
      "state": "Baseline mentions=11 of which 1 is a comment and 1 a string literal. Independently re-run mutants: remove ONE corrected_report attachment -> SURVIVED(9); remove BOTH -> SURVIVED(7); remove final_data attachment -> SURVIVED(9); remove error-dict attachment -> SURVIVED(9); flag on ZERO return paths with comment-only mentions -> SURVIVED(7). Only removing all three attachments at once -> KILLED(3). Ground truth via AST: all 4 value-returns currently DO carry the flag, so the code is correct and only the guard is vacuous.",
      "constraint": "Project durable rule (auto-memory feedback_mutation_test_guards_and_fixtures; contract sec.5 'a guard that cannot fail does not count'): a test whose NAME asserts a property must fail when that property is violated. Fix: assert per-return-path, e.g. every ast.Return with a dict/name value inside the function is preceded by an attachment of critic_degraded, rather than counting substring occurrences."
    },
    {
      "violation_type": "Overgeneralization",
      "action": "experiment_results.md sec.4 headline claims '12/12 mutations killed; 0 vacuous guards' and 'M5b added myself because it was the guard I was most likely to be defending'",
      "state": "The M1-M11+M5b matrix contains NO mutation for test_critic_degraded_flag_is_present_on_every_return_path, the very guard that protects Main's own sec.2(c) claim that critic_degraded is 'attached to all return paths'. I produced 5 surviving mutants against it and 2 more against the criterion-4 substring assertions -- at least 7 survivors the matrix never attempted. The global claim '0 vacuous guards' is therefore not measured, it is inferred from 12 passing mutations.",
      "constraint": "Project durable rule (auto-memory feedback_measure_dont_assert_claims): never assert a property you did not measure. A mutation matrix licenses only the claim 'these 12 mutations were killed', never the global 'this suite contains 0 vacuous guards'. Retract or scope the headline claim and add the missing mutation."
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "test_unparseable_critic_verdict_is_not_silently_treated_as_pass (backend/tests/test_phase_75_skill_delivery.py:254-262) guards the new retry/degraded behavior with assert 'Critic-Retry' in src and assert 'critic_degraded' in src",
      "state": "Both are bare substring scans over orchestrator.py source. Verified mutants: V1 deleting the ENTIRE 16-line retry block and leaving only '# Critic-Retry: (retry path deleted)' -> SURVIVED; V2 additionally replacing 'critic_degraded = True' with a comment -> SURVIVED; V3 restoring fail-open semantics under a reworded log message -> SURVIVED. Criterion 4 as written (literal 'treating as PASS with draft' absent, cap >= 6144) IS satisfied verbatim, so this does not miss the criterion; the unguarded property is the contract's own sec.4(c) behavioral commitment.",
      "constraint": "Substring presence in source does not imply behavior presence (qa.md sec.4 anti-rubber-stamp). Guard the retry/degraded path behaviorally -- e.g. patch _parse_json_with_fallback to return falsy and assert a second Critic-Retry generate call occurs and critic_degraded lands True on the returned dict."
    },
    {
      "violation_type": "Overgeneralization",
      "action": "experiment_results.md sec.2(b) states 'files with body-content drift: 0' and 'No non-heading line was added, removed, or altered in any file'; sec.2(c) states critic_degraded lets 'a consumer' avoid reading an absent key as False",
      "state": "My independent line-multiset diff vs git HEAD ignoring only heading lines reports exactly one blank line REMOVED in each of 8 files (bias_detector, critic_agent, deep_dive_agent, moderator_agent, quant_model_agent, risk_judge, scenario_agent, synthesis_agent) -- blank lines are non-heading lines, so the absolute wording is overstated. Substance of criterion 6 is unaffected: 0 non-blank body drift, heading TEXT identical, only levels/positions changed. Separately, grep across all .py/.ts/.tsx finds NO consumer that reads critic_degraded -- the flag is currently write-only, which experiment_results does not disclose.",
      "constraint": "Scope-honesty (qa.md sec.4): disclose real bounds rather than state absolutes. Reword to 'no non-blank body line changed; one blank separator line was consumed by the relocation in 8 files', and disclose that critic_degraded has no consumer yet so it carries no behavioral effect today."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_items",
    "immutable_verification_command_exit_0_25_passed",
    "masterplan_criteria_diff_vs_HEAD_byte_identical",
    "git_status_change_surface_vs_declared_file_list",
    "byte_identity_reverification_vs_git_show_HEAD_all_29_skills",
    "heading_text_vs_heading_level_drift_separation",
    "independent_file_count_8_uncertainty_3_codeexec",
    "quant_strategy_exclusion_shasum_and_load_skill_ValueError",
    "quant_optimizer_raw_read_path_confirmed",
    "independent_mutation_rerun_M1_M2",
    "independent_mutation_rerun_M3_per_file_x4",
    "independent_mutation_rerun_M4_per_file_x3",
    "novel_mutation_X1_heading_kept_body_deleted",
    "novel_mutation_X2_silent_reword_of_relocated_line",
    "novel_mutation_X3_sibling_section_body_deleted",
    "novel_mutation_critic_degraded_return_path_x5_SURVIVORS",
    "novel_mutation_retry_block_deleted_comment_kept_SURVIVOR",
    "novel_mutation_fail_open_semantics_reworded_SURVIVOR",
    "false_alarm_self_check_M_sot2_caught_by_other_half",
    "ruff_F821_F401_F811_gate_exit_1_but_byte_identical_to_HEAD",
    "backend_runtime_import_smoke",
    "startup_stem_assertion_negative_control_raises",
    "format_skill_caplog_live_emit_and_happy_path_silence",
    "full_backend_suite_13_failed_1246_passed",
    "per_failure_attribution_all_13_unrelated",
    "surface_touching_failure_test_phase_60_1_investigated",
    "modified_verifier_verify_phase_25_D9_1_rerun_5_5_pass",
    "known_pass_scripts_glob_scope_check",
    "bias_detector_orphan_confirmed_no_load_skill_caller",
    "critic_degraded_consumer_grep_write_only",
    "research_gate_envelope_gate_passed_true_5_sources_20_urls",
    "mtime_ordering_research_contract_generate_results",
    "third_conditional_rule_zero_prior_verdicts",
    "no_verdict_shopping_cycle_1_confirmed"
  ],
  "harness_compliance_ok": true,
  "notes": "HARNESS COMPLIANCE 5/5 CLEAN. (1) Research before contract: research_brief_75.4.md 14:47:20 < contract.md 14:58:08; envelope gate_passed=true, external_sources_read_in_full=5, urls_collected=20, recency_scan_performed=true, internal_files_inspected=44; audit_class=false so coverage.dry is correctly not required. (2) Contract before generate: contract 14:58:08 < orchestrator.py 15:01:19 < test file 15:04:12. (3) experiment_results.md present (15:13:58) with verbatim command output. (4) Log-last respected: masterplan status still 'pending', retry_count=0, and grep 'phase=75.4 .*result=' in harness_log.md returns 0. (5) No verdict-shopping: no evaluator_critique_75.4.md exists and the rolling evaluator_critique.md still belongs to step 75.2.1 -- cycle 1 confirmed, so CONDITIONAL is permitted and the 3rd-CONDITIONAL auto-FAIL rule does not trigger.\n\nATTACK-BY-ATTACK RESULT. (A) Vacuous guards -- FOUND, see violation_details 1 and 3; the predicted 13th guard exists and is precisely the one Main never mutated. Critically, the specific trap named in the prompt (asserting a canary against Path.read_text() instead of load_skill()) was NOT fallen into: every delivery assertion routes through the real load_skill(), and my novel mutations X1 (heading kept, body deleted), X2 (body line silently reworded) and X3 (sibling section body deleted) all KILLED, which a read_text-based canary could not have done. The two raw-file reads that do exist (test_uncertainty_permission_covers_every_file_that_has_one, and the quant_strategy H2 check) are coverage-list and untouched-file integrity checks, not delivery proofs -- legitimate use. M10-equivalent harness mutation confirmed meaningful. (B) Byte-identity re-derived independently from git show HEAD rather than trusting 'drift: 0': 0 non-blank body drift, heading TEXT multiset identical in all 9 changed files, only one blank separator line consumed per file in 8 files -- criterion 6 substance holds, wording slightly overstated (violation 4). (C) Counted myself: exactly 8 raw files carry 'Uncertainty Permission' and all 8 deliver it; 4 raw files carry 'Code Execution' but quant_strategy is the 4th and its exclusion is CORRECT -- it has zero '## Prompt Template' matches, load_skill raises ValueError on it, it is read whole at quant_optimizer.py:486, and it is byte-identical to HEAD (shasum 6dd7d199... both sides, git diff --exit-code clean). (D) 'treating as PASS with draft' confirmed absent; replacement is NOT fail-open at the data level -- I verified by AST that all 4 value-returns of run_synthesis_pipeline carry critic_degraded, and the degraded path logs a distinct warning; but the guard protecting this is vacuous (violation 1) and the flag has no consumer yet (violation 4). (E) Both no-file-id paths verified live, including the bare-attribute-absent case via getattr(self,'_skill_file_ids',None) -> returns {'max_output_tokens':1024}. (F) tests/verify_phase_25_D9_1.py change is a LEGITIMATE disclosed contract change, not criteria-weakening: claims 3/4/5 moved from 'is None' to exact-dict equality (strictly stronger), carry an explicit AMENDED docstring note, re-run 5/5 PASS, and are disclosed in experiment_results sec.2. The masterplan 75.4 verification block is byte-identical to HEAD (sha 36b1dced... both sides). Confirmed the verify_phase_23_1_* smoke test globs only verify_phase_23_1_*.py so it never executes the modified verifier. (G) Reproduced 13 failed / 1246 passed exactly. Attributed every one: the only failing file that greps the surface is test_phase_60_1_deep_pipeline.py, but the FAILING test (test_60_1_claude_code_rail_declares_latency_profile) asserts ClaudeCodeClient.recommended_step_timeout > _timeout_s and fails on 150 > 150 in claude_code_client.py -- the grep hit came from OTHER tests in the same file. No regression hidden in the red set. (H) bias_detector orphan admission ACCURATE: no load_skill('bias_detector') caller anywhere; production path is deterministic Python via 'from backend.agents.bias_detector import detect_biases' at orchestrator.py:38; Main correctly declines to claim a behavioral win.\n\nLINT NOTE (qa.md 1a): uvx ruff --select F821,F401,F811 exits 1 with 3 findings (F401 generate_reflection unused at orchestrator.py:48; F821 undefined 'Any' at :1009 and :1010). I checked these against HEAD and the finding set is IDENTICAL -- all 3 are pre-existing and none of the three lines appear in this step's diff. Zero new lint defects from 75.4, so the gate's non-zero exit is not attributable to this step. Latent-bug note for a future step: orchestrator.py has no 'from __future__ import annotations'; these survive only because Python 3.14 defers annotation evaluation (PEP 649), and they would still break typing.get_type_hints() on those two functions.\n\nTO CLEAR TO PASS (cycle 2): (1) replace the count-based critic_degraded guard with a per-return-path AST assertion and prove it by mutation; (2) make the retry/degraded assertions behavioral rather than substring; (3) retract or scope the '12/12 killed; 0 vacuous guards' headline to what was actually measured, and add the missing mutation row; (4) reword the criterion-6 blank-line absolute and disclose that critic_degraded is currently write-only. Items 1-2 are test-only changes; item 3-4 are experiment_results.md edits. No production code change is required -- the shipped behavior is correct."
}
```


---

# Cycle 2 -- FRESH Q/A on CHANGED evidence

**Q/A launch**: Workflow structured-output (`.claude/workflows/qa-verdict.js`), run
`wf_4b5c7466-adb`, model claude-opus-4-8[1m], 32 tool calls, 163,419 tokens, 606s.

**Why a fresh Q/A is legitimate here (not verdict-shopping)**: the cycle-1 blockers were
FIXED and the handoff files UPDATED before this spawn -- the test file was rewritten
(25 -> 27 tests, new AST helper + 2 behavioral tests), `experiment_results.md` gained
§7 plus the §4 retraction and §2b/§2c corrections, and `live_check_75.4.md` was
regenerated. The Q/A independently confirmed the evidence changed and that production
was byte-identical to cycle 1. Spawning a fresh Q/A on UNCHANGED evidence would have
been the forbidden pattern; this is the documented cycle-2 flow.

> **Provenance**: the JSON below is the Q/A agent's return value, transcribed
> **VERBATIM** by Main. Main records the verdict; Main never authors it.

## Verdict: PASS (ok=True)

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 6 immutable criteria independently verified MET (not read off Main's claims): C1 real load_skill('quant_model_agent')=2739 chars with {{quant_model_data}} and the Instructions line present, and the anti-stub clause proven real by my own M10-equivalent harness mutation (string-stub load_skill -> 19 failed/8 passed = KILLED); C2 8/8 uncertainty + 3/3 code-exec canaries delivered through the real loader, quant_strategy correctly excluded (load_skill raises ValueError, read whole by quant_optimizer, untouched); C3 format_skill live-emits on an orphan kwarg and is silent on the happy path, sector stem fixed, and my negative control on _assert_skill_stems_exist RAISES RuntimeError on an injected bogus stem; C4 critic=6144, dead twin=6144, 6144>=1.5*4096, literal 'treating as PASS with draft' absent (0 occurrences); C5 {'max_output_tokens':1024} on the no-file-id, unmapped-stem AND attribute-absent paths; C6 re-measured from git HEAD myself -> 0 non-blank body drift, heading TEXT multiset identical in all 9 files, exactly 8 blank separators consumed. Harness compliance 5/5 clean, masterplan verification block byte-identical to HEAD (sha 07d8aa8395832688a56f5713 both sides), zero unintended production change, and production is byte-identical to cycle 1 (orchestrator.py mtime 15:01:19 matches the cycle-1 record exactly) confirming the cycle-2 fixes were test+docs only. I did NOT trust the fix claims: I re-ran all 5 of the cycle-1 Q/A's surviving AST-guard mutants (ALL KILLED) and both substring-guard mutants against the real pipeline (QA6 delete-retry-block-keep-comment -> KILLED 'no retry'; QA7 fail-open-reworded -> KILLED 'critic_degraded=False, want True'), then hunted 6 novel shapes beyond what I was asked to check, of which 4 KILLED. Two residual items are recorded as notes, not blockers: (a) one unreachable-attachment mutant (if False: on a corrected_report return) survives the whole suite -- a reachability limit inherent to static AST checking that Main's correctly SCOPED claim ('22 specific mutations killed', explicitly not 'no vacuous guards') already accounts for, and every realistic regression shape is killed; (b) experiment_results.md sec.1/sec.3 still carry the cycle-1 counts (25 passed / '25 tests') while the suite returns 27 -- but the contract-designated live_check_75.4.md IS regenerated, correct, and explicitly reconciles 'Cycle 1 was 25 passed / 0 skipped. Cycle 2 is 27 passed / 0 skipped', and sec.7 states the correct 27 baseline, so no criterion rests on the stale figure and no reader is left with an unreconciled number.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_items",
    "immutable_verification_command_exit_0_27_passed",
    "zero_skipped_zero_xfail_confirmed_via_pytest_rs",
    "masterplan_75_4_verification_block_sha_vs_HEAD_byte_identical",
    "git_status_change_surface_vs_declared_file_list",
    "production_byte_identity_vs_cycle_1_via_mtime_and_git",
    "blocker1_rerun_QA1_QA5_all_five_cycle1_mutants_ALL_KILLED",
    "novel_6th_shape_hunt_6_mutants_4_killed_2_survived",
    "blocker2_rerun_QA6_QA7_behavioral_BOTH_KILLED",
    "behavioral_stub_fidelity_baseline_GREEN_and_3_mutants_killed_through_it",
    "generate_with_retry_signature_vs_instance_attr_patch_verified",
    "raising_false_masking_check_get_critic_prompt_get_synthesis_prompt_exist",
    "blocker3_zero_skips_and_no_swallowing_try_except",
    "blocker4_criterion6_blank_line_independent_remeasure_8_AGREE",
    "critic_degraded_consumer_grep_write_only_disclosure_ACCURATE",
    "N3_self_invalidation_reasoning_judged_CORRECT",
    "criterion1_real_load_skill_2739_chars_placeholder_and_instructions",
    "M10_equivalent_anti_stub_harness_mutation_KILLED_19_failed",
    "criterion2_8of8_uncertainty_3of3_codeexec_delivered_live",
    "quant_strategy_exclusion_ValueError_and_untouched_confirmed",
    "criterion3_format_skill_caplog_live_emit_and_happy_path_silence",
    "criterion3_startup_stem_assertion_NEGATIVE_CONTROL_raises_RuntimeError",
    "criterion4_6144_dead_twin_and_literal_string_absent",
    "criterion5_all_three_no_file_id_paths_return_1024",
    "criterion6_nonblank_drift_zero_heading_text_multiset_identical",
    "ruff_F821_F401_F811_gate_exit_1_finding_set_IDENTICAL_to_HEAD_zero_new",
    "backend_runtime_import_smoke_clean",
    "full_backend_suite_13_failed_1248_passed_delta_plus2_equals_new_tests",
    "log_last_masterplan_pending_retry_0_zero_logged_verdicts",
    "no_verdict_shopping_evidence_demonstrably_changed",
    "third_conditional_rule_not_triggered_cycle_2_of_2",
    "read_only_discipline_confirmed_no_QA_introduced_repo_change"
  ],
  "harness_compliance_ok": true,
  "notes": "HARNESS COMPLIANCE 5/5 CLEAN. (1) Research before contract: research_brief_75.4.md 14:47:20 < contract.md 14:58:08; envelope gate_passed=true, external_sources_read_in_full=5, snippet_only=15, urls_collected=20, recency_scan_performed=true; not audit-class so coverage.dry correctly not required. (2) Contract before generate: contract 14:58:08 < skills 14:58:51 < orchestrator 15:01:19 < prompts 15:01:42 < verify_phase_25_D9_1 15:02:01 < test file 15:04:12 (c1) / 15:28:09 (c2). (3) experiment_results.md present 15:32:06. (4) Log-last respected: masterplan status=pending, retry_count=0, grep 'phase=75.4 .*result=' returns 0. (5) NO verdict-shopping -- evidence demonstrably CHANGED: test file rewritten (25->27 tests, new _returns_missing_critic_degraded AST helper + 2 behavioral tests), experiment_results sec.2b/2c/4/7 updated, live_check regenerated; production mtimes UNCHANGED, confirming the 'test+docs only' claim. Cycle 2 with exactly ONE prior CONDITIONAL -- 3rd-CONDITIONAL auto-FAIL does not trigger, and I did not need it.\n\nBLOCKER-BY-BLOCKER, VERIFIED NOT TRUSTED. B1: I replicated the real test's exact logic against mutated SOURCE STRINGS in memory (repo untouched). Baseline passes; QA1 missing=[1642], QA2 [1642,1651], QA3 [1679], QA4 [1683], QA5 [1643,1652,1678,1681] -- all 5 KILLED. B2: I copied orchestrator.py to scratch, mutated, importlib-loaded, and drove run_synthesis_pipeline with the suite's own stub. QA6 KILLED (\"no retry; calls=['Synthesis','Critic']\"), QA7 KILLED (\"critic_degraded=False (want True)\"). B3: 0 skipped via -rs; the single try/except (lines 162-164) is the assertive exclusion-set builder asserting == {\"quant_strategy\"}, not a swallow; no pytest.skip/xfail. B4: my independent line-multiset diff vs git HEAD reproduces EXACTLY 8 blank lines (1 each in bias_detector, critic_agent, deep_dive_agent, moderator_agent, quant_model_agent, risk_judge, scenario_agent, synthesis_agent; enhanced_macro_agent 0) -- Main, the cycle-1 Q/A and I all agree. critic_degraded consumer grep across *.py/*.ts/*.tsx returns only orchestrator writer lines + the new test's assertions -> write-only disclosure ACCURATE.\n\nFIXTURE-FIDELITY SCRUTINY (the 75.2.1 failure mode I was asked to hunt): the behavioral stub is NOT divorced from production. Baseline is GREEN and three separate mutants die THROUGH it, which a divorced stub could not do. Signature checks out: real _generate_with_retry(self, model, prompt, agent_name, ...) patched on the INSTANCE (therefore unbound), so fake_generate(client, prompt, agent_name, **kwargs) aligns correctly -- raising=False is REQUIRED there, not a hole. get_critic_prompt/get_synthesis_prompt both exist in prompts.py so raising=False masks nothing today; if either were renamed the real function would be invoked with stub args and the test would go RED, not falsely green.\n\nN3 SELF-INVALIDATION: Main's reasoning is CORRECT, not a talked-out real finding. Replacing `missing = _returns_missing_critic_degraded(target)` with `missing = []` deletes the computation, which is equivalent to deleting the assertion -- it survives against ANY suite and carries zero information about guard strength. M10's shape (keep the assertion, break the FIXTURE) is the valid harness mutation; N3b is the right replacement.\n\nRESIDUAL FINDING (note, not a blocker). Of 6 novel shapes I invented beyond the assigned four, 4 KILLED (return inside a `with` block; dict(**x) call-expression return; attachment moved after the return; attachment retargeted to a different variable). 2 SURVIVED the AST guard, both the unreachable-attachment shape (`if False:`): N6a on final_data is KILLED by the new behavioral leg, but N6b on ONE corrected_report attachment survives the ENTIRE suite, because static AST cannot do reachability and the two behavioral tests exercise only the final_data return, not the two corrected_report returns or the error-dict return. This is a limit of the technique rather than the cycle-1 defect -- cycle 1 was categorical vacuity (plain deletion from any 1/2/3/4 paths survived); now every realistic regression shape dies. Critically, N6b falsifies nothing Main asserted: the cycle-2 claim is correctly scoped to \"22 specific mutations were killed\" with an explicit disclaimer that this is not \"no vacuous guards\" -- so my survivor actually vindicates the retraction rather than contradicting it. Optional future hardening: extend behavioral coverage to the corrected_report return paths.\n\nDOCUMENTATION NOTE (not a blocker). experiment_results.md sec.1 \"Verbatim verification command output\" still shows the cycle-1 transcript (25 passed, 1.94s) against an actual 27; sec.3 says \"NEW -- 25 tests\"; the sec.4 heading still literally reads \"12/12 killed, 0 vacuous guards\" though the body immediately and thoroughly retracts it. I weighed this against the project's measure-dont-assert rule and concluded it does not warrant CONDITIONAL: the contract-designated live-evidence artifact live_check_75.4.md is regenerated and correct (27 passed, EXIT=0) and explicitly reconciles the delta (\"Cycle 1 was 25 passed / 0 skipped. Cycle 2 is 27 passed / 0 skipped\"), sec.7 records the correct 27 baseline, and no immutable criterion depends on the count. Stale sub-section text with an accurate authoritative companion is a hygiene wart, not a false claim being relied upon. Worth refreshing on the next touch.\n\nLINT (qa.md 1a): uvx ruff --select F821,F401,F811 exits 1 with 3 findings (F401 generate_reflection at :48; F821 Any at :1009/:1010). I ran the same gate against the HEAD blobs: the finding set is IDENTICAL (same three, line numbers shifted only by insertions above). Zero new lint defects from 75.4; already queued as 75.4.6.\n\nFULL SUITE: 13 failed / 1248 passed, matching Main's claim exactly; cycle 1 was 13/1246 and the +2 delta is precisely the two new behavioral tests. 0 regressions. READ-ONLY DISCIPLINE: all mutation work ran in the session scratchpad via in-memory strings and importlib-loaded copies; final git status is identical to the pre-verification surface and the suite is still 27 passed."
}
```

---

## Main's follow-up on the two residual NOTES (neither a blocker)

**(a) N6b unreachable-attachment survivor.** The Q/A invented 6 novel mutant shapes
beyond the 4 it was asked to try; 4 KILLED. The 2 survivors are both the `if False:`
unreachable-attachment shape, and one (N6b, on a `corrected_report` return) survives the
whole suite because static AST cannot reason about reachability and the behavioral leg
exercises only the `final_data` return. **Accepted as a known limit of the technique,
not a defect to fix in this step** -- and it is precisely why the cycle-2 claim is scoped
to "22 specific mutations were killed" rather than "no vacuous guards". Extending
behavioral coverage to the two `corrected_report` returns and the error-dict return is
queued as **75.4.7**.

**(b) Stale counts in `experiment_results.md`.** The Q/A found §1 still showed the
cycle-1 transcript (25 passed) against an actual 27, §3 said "NEW -- 25 tests", and the
§4 heading still literally read "0 vacuous guards" though the body retracted it. Root
cause: one of my cycle-2 edits used a plain `str.replace()` **without an assertion**, so
it silently no-matched -- the same measure-don't-assert lapse in miniature. **Fixed
after this verdict**: all three corrected, every substitution asserted. No immutable
criterion depended on the count, and `live_check_75.4.md` was already correct and
explicitly reconciled the delta, which is why the Q/A correctly graded it a hygiene wart
rather than a false claim being relied upon.
