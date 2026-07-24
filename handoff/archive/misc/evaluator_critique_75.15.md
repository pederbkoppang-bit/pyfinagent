# Evaluator critique -- Step 75.15 (Q/A cycle 1: CONDITIONAL)

Q/A launch: Workflow `wf_3b89bdc1-43c` (qa-verdict.js, opus/max). Verdict
transcribed VERBATIM below. One narrow violation (the seed-lane durability
guard is vacuous -- mutation-proven by the Q/A); cycle-2 fixes it and
spawns a fresh Q/A on changed evidence.

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "All 6 immutable success criteria independently reproduce as MET (CI-equivalent green tail 1466 passed/16 deselected reproduced verbatim; raw 9-failure set = exactly 4 requires_live-marked + 5 .env-pollution with flags confirmed False-default; 23_2_14 audit note names locks 15-18 with per-lock re-entrancy analysis and passes; coverage_tier_check.py proven non-vacuous [exit 2 on missing/zero-module, exit 1 below bar, 7 bars parsed from the doc not hardcoded] and wired nightly; doc refreshed 2026-07-24; npm-audit weekly+lockfile shape correct with honest red disclosure; seed 'blocks the PR' removed; visual gated on baseline presence). Harness compliance clean; immutable command exit 0; zero production code changed; F821/F401/F811 lint clean over all 8 .py files. ONE fixable gap caps this at CONDITIONAL (not FAIL, since no criterion is missed and the underlying lane change is real): the NEW durability guard test_phase_75_ci_gates.py::test_seed_stability_no_longer_overclaims_pr_blocking is vacuous -- it mirrors the immutable command's OR-clause ('blocks the PR' not in s) or ('run_seed_stability' in s), and this step ADDED the token 'run_seed_stability' to the seed yaml comment (diff line 20), so the second clause is permanently true and the guard cannot fail if the 'blocks the PR' overclaim is ever re-introduced (mutation-proven: re-adding 'blocks the PR' with the comment token present -> guard STILL PASSES). The guard-file docstring overclaims it catches 'a future edit that silently reverts a lane'; the M1-M7 mutation matrix silently omitted legs d/e so this guard's can-fail was never established. This is the 6th instance of the operator-flagged vacuous-guard anti-pattern (mutation-test-guards memory). Fix is ~2 lines (assert the honest re-scoped sentence is present, e.g. 'structurally cannot enforce reproducibility' in s, which CAN fail on revert; drop the run_seed_stability OR escape-hatch; add a leg-d mutation), then re-spawn a fresh Q/A on the changed evidence per the cycle-2 flow.",
  "violated_criteria": [
    "mutation_resistance/qa-tests-05: seed-stability durability guard is vacuous -- test_phase_75_ci_gates.py::test_seed_stability_no_longer_overclaims_pr_blocking cannot fail on a re-introduced 'blocks the PR' overclaim because the step added the OR-escape-hatch token 'run_seed_stability' to the seed yaml comment (mutation-proven can't-fail); guard-file module docstring overclaims protection for this lane"
  ],
  "violation_details": [
    {
      "violation_type": "Overgeneralization",
      "action": "ship test_phase_75_ci_gates.py::test_seed_stability_no_longer_overclaims_pr_blocking asserting ('blocks the PR' not in s) or ('run_seed_stability' in s), after adding 'run_seed_stability' to seed-stability-check.yml's comment (diff line 20)",
      "state": "Mutation reproduced by Q/A: re-inserting 'blocks the PR' into seed-stability-check.yml while the comment token 'run_seed_stability' remains -> guard STILL PASSES (verified: overclaim re-added=True, run_seed_stability present=True, guard passes=True). The token appears ONLY in the header comment (line 16/20), never as a job/step name (job is 'seed-stability', step 'Run seed-stability drill'). M1-M7 matrix included no mutation for legs d/e, so this guard's can-fail was never established. Guard-file docstring claims it catches 'a future edit that silently reverts a lane back to advisory' -- false for the seed lane.",
      "constraint": "A regression guard must be able to FAIL on the exact regression it names (mutation-test-guards memory: 'a guard that can't fail doesn't count'; qa.md 4b: completeness/can-fail claims must be reproduced). The underlying immutable criterion 5 (seed lane re-scoped) IS met by direct diff inspection, so this is a fixable quality gap, not a criterion miss."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5item",
    "research_gate_envelope(gate_passed=true,6src,recency)",
    "mtime_order_research<contract<draft<final",
    "log_last(no_75.15_harness_log_entry;status=pending)",
    "no_verdict_shopping(first_spawn)",
    "immutable_command_part1_assertions_exit0",
    "immutable_command_part2_pytest_23_2_14=5passed",
    "git_scope_diff(zero_production_code)",
    "python_lint_gate_F821_F401_F811_clean_8files",
    "ci_equivalent_green_tail_reproduced_verbatim(1466pass/16desel/5xfail/1xpass)",
    "raw_suite_9failure_identity_reproduced(4catA+5catB)",
    "requires_live_per_test_marker_diffs(23_2_10/15/6/9)",
    "flag_defaults_false_settings_py(3flags)",
    "lock_count_18_audit_note_locks_15-18",
    "coverage_runner_nonvacuous(exit2_missing/exit2_zero-module/exit1_below-bar/bars-parsed-from-doc)",
    "coverage_workflow_nightly_wiring",
    "coverage_doc_refresh_2026-07-24_changelog_row",
    "npm_audit_yml_weekly+lockfile+auditlevel_high+never_fix+honest_red_disclosure",
    "seed_wording_diff(blocks_the_PR_removed)",
    "visual_baseline_gate_diff(can-fail,not-vacuous)",
    "guard_file_16_tests_run_all_pass",
    "guard_mutation_seed_vacuity_PROVEN",
    "legC_testside_fixes_verified(test_60_1_class->instance;pause_resume_mock_target_75.9_singleton)",
    "F401_removal_safe(23_2_6_collects_green)",
    "vitest_serverless_boundary(run-test.mjs->vitest_run;no_next_dev;:3000_untouched)",
    "collection_math_1474+16=1490"
  ],
  "harness_compliance_ok": true,
  "notes": "Quality of this step is otherwise high and the deliverable is sound RIGHT NOW -- the CONDITIONAL is narrow and cheaply fixable. POSITIVES independently confirmed: (1) Main's review-correction is legitimate and load-bearing -- the executor's '23_2_15 fixed, 0-fail tail' did NOT reproduce; Main reinstated the research gate's category-A requires_live classification with a measured docstring rationale, kept the executor's legitimate mock fix (75.9 get_bq_client lru_cache singleton silently no-op'd a class patch -- real cross-step drift), and updated the collection pin to 16 deselected. (2) All leg-C 'fixes' are genuinely TEST-side with production correct and clearly documented (test_60_1 read the stale CLASS attr vs the phase-61.2 per-instance override; pause_resume patched the wrong target post-75.9). (3) npm-audit red is honestly DISCLOSED (42 vulns incl 3 critical), explicitly NOT hidden by level-raising -- this satisfies the criterion honestly. (4) The coverage runner is a model non-vacuous guard (refuses PASS on zero-module input; can-fail proven both by my in-memory test and by the shipped M5 which mutates the bar to 99% AND asserts real measurements pass). (5) The two comment-token-trap mutation survivors (M2/M6) were self-caught and correctly re-anchored to run:/non-comment lines -- exactly the discipline the mutation-test-guards memory asks for; the seed guard is the one that slipped the same net. (6) Boundary honored: vitest is serverless, no server started, operator :3000 untouched; scope is CI-config+test+doc only. MINOR (non-blocking, not part of the verdict): broad-ruleset ruff on the 2 new files surfaces PLW1510 (subprocess.run without check= -- intentional, returncode is inspected), EXE001 (shebang w/o exec bit -- cosmetic, invoked via python), ISC004 (implicit multi-line string concat in the coverage runner's error messages -- intentional, verified renders correctly); none are in the project's required F-class gate. CYCLE-2 PATH: harden test_seed_stability_no_longer_overclaims_pr_blocking to a positive assertion of the honest re-scoped sentence (can fail on revert), optionally add a leg-d/'blocks the PR' mutation to the matrix, update experiment_results, then spawn a FRESH Q/A on the changed evidence. This is the first verdict for 75.15 (no prior CONDITIONAL; 3rd-CONDITIONAL auto-FAIL not in play)."
}
```

## Follow-up (cycle 2) -- Main's fix on the single violation, before the fresh Q/A

Seed guard hardened to two positive can-fail assertions (overclaim absent
AND honest sentence present); docstring overclaim reworded; leg-d
mutations M8 (the exact Q/A-proven vacuity) + M9 both KILLED, 16/16
post-restore; immutable command exit 0; CI-equivalent tail unchanged
(1466/0/16). Evidence CHANGED -> fresh Q/A per the cycle-2 flow.


## Cycle 2 verdict -- Workflow `wf_0c39d000-916` (fresh Q/A on changed evidence): PASS

Transcribed VERBATIM from the captured structured-output return value.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "Cycle-2 fix independently verified: the sole cycle-1 blocker (vacuous seed-stability durability guard) is genuinely resolved. test_seed_stability_no_longer_overclaims_pr_blocking is now TWO independent can-fail assertions ('blocks the PR' ABSENT + 'structurally cannot enforce' PRESENT), with the OR-escape-hatch token dropped. I reproduced the exact cycle-1 vacuity mutation M8 in-memory (re-add 'blocks the PR' while the run_seed_stability comment token remains): the hardened guard now FAILS (KILLS) where the OLD OR-form still PASSES on the same mutation -- the vacuity is dead. M9 (strip the honest sentence) also KILLS. Immutable command exit 0 (test_phase_23_2_14 5 passed); guard suite 16/16 (16 dots, internally consistent); F821/F401/F811 lint clean on the changed file; AST OK; zero production (non-test) source changed. All 6 immutable criteria re-verified MET (e2e-smoke enforcing + -m 'not requires_live' via M1/M2/M3 guards; lock-count guard green; collection count independently re-derived at 1474/1490/16-deselected by the shipped collection guard; coverage_tier_check exit-2-on-missing/exit-1-below-bar both proven; seed lane honestly re-scoped and now enforced can-fail; npm-audit shape+triggers with honest red disclosure; :3000 untouched). Harness compliance clean on all 5 items.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5item",
    "research_gate_envelope(gate_passed=true,6src>=5,recency,tier=moderate)",
    "mtime_order(research860313<contract860432<guard863964<results864123)",
    "log_last(zero_75.15_result_lines_in_harness_log;masterplan_status=pending)",
    "no_verdict_shopping(evidence_changed:guard_test_hardened;independently_reproduced)",
    "immutable_command_exit0(part2_test_23_2_14=5passed)",
    "guard_suite_16of16(16dots_internally_consistent)",
    "M8_mutation_reproduced_IN_MEMORY(re-add_blocks_the_PR+token_present->NEW_guard_KILLS;OLD_OR-form_still_PASSES)",
    "M9_mutation_reproduced(strip_honest_sentence->KILLS)",
    "python_lint_F821_F401_F811_clean_changed_file",
    "ast_parse_ok",
    "no_production_nontest_py_changed(scope=CI-config+tests+coverage_runner+doc)",
    "seed_yaml_delta_characterized(honest_rescoping_predates_cycle2_per_cycle1_run_seed_stability_reference;cosmetic_retouch_nonmaterial_inboundary)",
    "collection_count_re-derived_by_shipped_guard(1474/1490/16deselected)",
    "all_6_immutable_criteria_mapped_to_covering_evidence",
    "3rd_CONDITIONAL_check(0_prior_CONDITIONALs_logged;cycle2_of_verdict_history)"
  ],
  "harness_compliance_ok": true,
  "notes": "The fix is exactly the single cycle-1 violation and nothing material else -- I independently proved the can-fail rather than trusting Main's M8/M9 claims: the precise mutation the cycle-1 Q/A used to prove vacuity (re-insert 'blocks the PR' with the run_seed_stability comment token still present) now reddens the guard, and I confirmed the OLD OR-form would still have passed it, so the vacuity is genuinely eliminated -- the model behaviour the mutation-test-guards memory asks for. SEED-YAML mtime note (measured, non-blocking): seed-stability-check.yml is the single newest code file (mtime 1784863995, 31s after the guard file), which superficially sits oddly with the 'guard file ONLY changed' disclosure. Resolved decisively: the honest re-scoping ('structurally cannot enforce' + 'run_seed_stability') lives in one contiguous phase-75.15 comment block that the CYCLE-1 critique already analysed (its vacuity finding depended on 'run_seed_stability' being in that comment), so the honest sentence and criterion-5 satisfaction PREDATE cycle 2; any cycle-2 re-touch of that comment is cosmetic, in-boundary (CI-config), and non-regressing (immutable exit 0). This does not affect any criterion. Module docstring (lines 1-14) is accurate, not overclaiming: its three enumerated protections (advisory-revert->M1, requires_live-drop->M2, gate-removal->M6/coverage) are each backed by a can-fail guard; the seed-wording guard is additional coverage it does not even claim. Scope honesty otherwise high: experiment_results discloses the npm-audit red (42 vulns/3 critical, not hidden by level-raising), the Main-review correction of the executor's 23_2_15 misclassification, and the leg-C test-side fixes with production correct. VERIFICATION SCOPE DISCLOSURE: I did not re-run the full multi-minute -m 'not requires_live' suite for the 1466-passed tail; I relied on (a) the shipped collection-count guard independently re-deriving 1474/1490/16-deselected, (b) immutable exit 0, (c) unchanged scope (only the already-passing guard-test file changed since cycle 1), and (d) cycle-1's verbatim full-tail reproduction -- sufficient for confidence. Not the 3rd CONDITIONAL (0 prior CONDITIONAL rows in harness_log; verdict history = 1 prior CONDITIONAL, this is cycle 2). PASS."
}
```
