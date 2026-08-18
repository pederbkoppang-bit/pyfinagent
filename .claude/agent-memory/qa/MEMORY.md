# Q/A Agent Memory Index

## Guard vacuity — the assertion is narrower than its label
- [driven-guard-asserts-the-key-not-the-value](feedback_driven_guard_asserts_the_key_not_the_value.md) — `typeof x === 'object'` passes on `{}`; hollow + decoy-keyed containers survived all 57 checks (86.78 c7)
- [enum-membership-guard-passes-every-wrong-value](feedback_enum_membership_guard_passes_every_wrong_value.md) — `x in {A,B,C}` can only catch ABSENCE; author mutated to `""` and scored a KILL. Inverted-attribution mutant SURVIVED 20/20 (86.108)
- [static-form-guard-rejects-one-syntax-not-the-class](feedback_static_form_guard_rejects_one_syntax_not_the_class.md) — an AST guard rejecting `ast.Constant` at a kwarg is defeated by a `Call`/`BoolOp`; sole coverage for 3 sites survived 37/37 (86.108 c3)
- [debrittling-an-or-clause-makes-a-guard-vacuous](feedback_debrittling_an_or_clause_makes_a_guard_vacuous.md) — the `or` added to stop false positives was already True on the control, so it detected a RENAME not the vol term (86.116 c2)
- [anti-vacuity-check-that-is-itself-a-tautology](feedback_anti_vacuity_check_that_is_itself_a_tautology.md) — the fix for "your guard was vacuous" shipped `"99.40".startswith("99.4")`: True in an empty namespace (86.85 c12)
- [derived-statistic-with-a-hardcoded-conclusion](feedback_derived_statistic_with_a_hardcoded_conclusion.md) — the table was computed, the verdict sentence was `print()`: "IS surprising -- P=0.7928 ... 13 is MORE than enough" vs 168 (86.47 c2)
- [byte-presence-pin-is-satisfied-by-a-comment](feedback_byte_presence_pin_is_satisfied_by_a_comment.md) — "literal X still in file Y" passes with X alive only in a comment; prefix-strip survives a TRAILING `//`, span-strip `/* */` is sound
- [palindromic-fixture-cannot-test-order](feedback_palindromic_fixture_cannot_test_order.md) — "sequence is oldest->newest" asserted on 3 identical elements survived `out[::-1]` (86.85)
- [baseline-captured-after-the-action](feedback_baseline_captured_after_the_action.md) — `before_rows` read AFTER the call makes `len(x)==len(x)`; the killing mutant KEEPS the return code (86.71 c3)
- [test-that-cleans-its-own-fixture-is-inert](feedback_test_cleans_own_fixture.md) — a test doing the module's own dropna asserts on its own arithmetic; passed under the restored defect (80.31)
- [try-to-evade-before-calling-a-scan-vacuous](feedback_try_to_evade_before_calling_a_scan_vacuous.md) — build the dodging mutant first; a co-located behavioural assert retired my plausible-but-wrong vacuity finding (85.3 c2)
- [subprocess-drive-that-redeclares-the-entrypoint](feedback_subprocess_drive_that_redeclares_the_entrypoint.md) — `import m` never runs `__main__`; harness re-declared argparse, and `"execute" in ast.dump` can't see a `Not()` (75.11.4)
- [survivor-needs-behavioural-differential](feedback_survivor_needs_behavioural_differential.md) — a survivor isn't automatically a finding; 2 of 3 were equivalent (80.27 c2)
- [killed-mutant-needs-differential-too](feedback_killed_mutant_needs_differential_too.md) — a mutant that reddens the guard still needs a differential; two false mechanisms reached the source comments (80.3 c2)

## Mutation-matrix construction
- [a-mutant-that-cannot-build-scores-as-a-kill](feedback_a_mutant_that_cannot_build_scores_as_a_kill.md) — `catch { survived=false }` turns a SyntaxError into a KILL; record THREW vs RETURNED, run the control first (86.90)
- [run-a-null-mutant-through-every-matrix](feedback_run_a_null_mutant_through_every_matrix.md) — a tempdir-relocated mutant broke `__file__` imports: 6/6 KILLED measured relocation, not the subject (86.71)
- [isolate-each-property-of-a-compound-mutant](feedback_isolate_each_property_of_a_compound_mutant.md) — a 2-property cell dies to whichever check fires first, leaving the other unproven (86.71 c5)
- [mutate-each-half-of-an-ANDed-guard](feedback_mutate_each_half_of_an_ANDed_guard.md) — a 2nd ANDed predicate leaves the OLD half untested; shape-half mutant survived 31+31 green (86.85 c9)
- [mutate-each-duplicated-site-individually](feedback_mutate_each_duplicated_site_individually.md) — the all-N-sites cell KILLS and hides an unguarded twin (86.88 c3)
- [guard-the-reference-row-of-a-delta-table](feedback_guard_the_reference_row_of_a_delta_table.md) — the fix guarded the subtrahend's source; poisoning the baseline row flipped ASK-1's +2.1pp to -2.1pp, 6/6 guards green (86.59 c3)
- [mutate-into-the-shape-the-criterion-forbids](feedback_mutate_into_the_shape_the_criterion_forbids.md) — criterion forbade special-casing 86.86; the fixtures used only 86.86, so that mutant survived
- [enumerate-entry-points-not-the-main-path](feedback_enumerate_entry_points_not_the_main_path.md) — matrix drove the hook, self-test never called the CLI: `--reason` survived BOTH (86.71 c2)
- [enumerate-every-position-at-a-recidivist-call-site](feedback_enumerate_every_position_at_a_recidivist_call_site.md) — 5 survivors at one call site; enumerate positions, don't re-judge
- [two-mutant-forms-separate-artifact-from-kill](feedback_two_mutant_forms_separate_artifact_from_kill.md) — a `*args` wrapper faked 2 of 3 kills; kill counts are SCOPE-dependent (82.27 c3)
- [matrix-oracle-inherits-selftest-blindspots](feedback_matrix_oracle_inherits_selftest_blindspots.md) — 22/22 KILLED means nothing the self-test can't see (86.85 c11)
- [matrix-row-sums-pin-the-tree](feedback_matrix_row_sums_pin_the_tree.md) — rows summed to 69 while the shipped suite was 72: the matrix predated the new tests
- [decoy-first-defeats-first-match-guards](feedback_decoy_first_defeats_first_match_guards.md) — a `re.search` guard reads the FIRST hit; correct decoy early + wrong line later (83.1 c3)
- [control-injected-outside-the-region-it-controls](feedback_control_injected_outside_the_region_it_controls.md) — the poison landed 51 bytes BEFORE the slice anchor, so the claim held with the stripper neutered (86.92)
- [mutate-without-touching-the-tree](feedback_mutate_without_touching_the_tree.md) — sys.modules injection (py) + vite alias to a gitignored copy (fe); always run a CONTROL first (80.40 c3)
- [restore-mutations-from-worktree-backup](feedback_restore_mutations_from_worktree_backup.md) — `git checkout`/`git show HEAD` on an uncommitted step reverts the FIX; cp to scratchpad first
- [mutate-via-pytest-main-plugins](feedback_mutate_via_pytest_main_plugins.md) — write-guard blocks plugin files; use `pytest.main(argv, plugins=[...])`, and install a SINK before neutering a live-url guard
- [mutate-the-library-for-upstream-pins](feedback_mutate_the_library_for_upstream_pins.md) — a library-fact test isn't automatically vacuous; mutate site-packages to decide (80.1 c2)
- [mutate-the-flag-read-not-just-the-guard](feedback_mutate_the_flag_read_not_just_the_guard.md) — dark-launch tests patch `_flag_enabled()`, so the production flag-read runs in ZERO tests (80.27)
- [neutralize-import-time-singleton](feedback_neutralize_import_time_singleton.md) — a module-level singleton loads at exec, BEFORE fixtures redirect; rewrite the path constant first (36.7 c5)
- [neutralize-the-write-chokepoint-probe](feedback_neutralize_the_write_chokepoint_probe.md) — stub the ONE writer chokepoint and run the real pipeline (83.0 c2)
- [oracle-with-silent-fallback-survives-absent-subject](feedback_oracle_with_silent_fallback_survives_absent_subject.md) — "live if reachable else snapshot" passes when the subject is ABSENT; run all three mutants (83.0 c3)
- [unreachable-except-branch-survives-everything](feedback_unreachable_except_branch_survives_everything.md) — the collaborator returned `[]`, so the except never ran; point the path at a DIRECTORY (86.71 c4)

## Guards that stop one seam short
- [class-guard-bound-to-the-helper-not-the-call-site](feedback_class_guard_bound_to_the_helper_not_the_call_site.md) — a "pins the CLASS" test driving the HELPER misses the call site swapping it out (86.88 c4)
- [a-fix-can-relocate-the-defect-one-seam-upstream](feedback_a_fix_can_relocate_the_defect_one_seam_upstream.md) — new guards cluster where the bug WAS; hardcoding the model at the call site reinstated it, 29/29 green (86.108 c3)
- [a-provenance-fix-that-only-logs](feedback_a_provenance_fix_that_only_logs.md) — effect was a logger.warning; the additive key reached NO persisted artifact (86.88 c1+c2)
- [boundary-on-elements-not-the-container](feedback_boundary_on_elements_not_the_container.md) — a boundary on a collection's ELEMENTS leaves `Array.isArray(x) ? x : []` upstream (86.90 c3)
- [same-source-recount-cannot-see-upstream](feedback_same_source_recount_cannot_see_upstream.md) — `stored != recount_from(source)` kills hardcodes ONLY; mutate the COLLECTOR (86.84 c11)
- [slice-and-exec-with-the-collaborator-stubbed](feedback_slice_and_exec_with_the_collaborator_stubbed.md) — tell a real DRIVE from a re-implementation: stub the collaborator to CAPTURE argv
- [object-keys-walk-is-not-a-losslessness-proof](feedback_object_keys_walk_is_not_a_losslessness_proof.md) — attack lossless-or-throw with non-enumerable props and a non-enumerable `toJSON`; grade by REACHABILITY (86.90)
- [child-process-escapes-conftest-guards](feedback_child_process_escapes_conftest_guards.md) — a conftest patch covers only the pytest process; read the shelled script's argparse `default=` (86.3)
- [tightened-guard-opens-false-negative](feedback_tightened_guard_opens_false_negative.md) — narrowing a matcher to kill a false positive opens a false negative (80.3 c3)
- [probe-self-contamination-shared-module](feedback_probe_self_contamination.md) — patching a shared module poisons the probe's OWN later fetches; 8 "live tickers" were all AAPL (80.31 c3)
- [the-instrument-that-closes-a-channel-opens-one](feedback_the_instrument_that_closes_a_channel_opens_one.md) — the sweep keyed role on prompt LITERALS from two other files; drift one word and it reads GREEN
- [a-later-step-bolts-a-mode-on-with-no-guard](feedback_a_later_step_bolts_a_mode_on_with_no_guard.md) — 86.78's `--evidence-only` is never driven by 86.21's self-test; an md5 line is a free staleness detector
- [structural-fix-needs-a-mechanism](feedback_structural_fix_needs_a_mechanism.md) — "it is GENERATED so it cannot lag" had NO generator in the tree (82.0 c6)

## Claims about claims — reproduce the prose
- [credited-mechanism-is-a-documented-dead-key](feedback_credited_mechanism_is_a_documented_dead_key.md) — grep the param's READER not its writer; 86.116 credited a key the repo lists in `_DEAD_KEYS`
- [check-the-attribution-not-just-the-count](feedback_check_the_attribution_not_just_the_count.md) — a claim reproduced its COUNT and falsified its CAUSE: 0/421 vs 420/420 (86.78 c4)
- [a-bound-on-a-universal-claim-is-a-new-census](feedback_a_bound_on_a_universal_claim_is_a_new_census.md) — "falsified by exactly those two" is itself a completeness claim; my census found a third (86.21 c8)
- [census-the-declared-label-space](feedback_census_the_declared_label_space.md) — "15 issues R1-R15" → grep every label; three had ZERO mentions (36.7 c2)
- [queued-is-a-claim-that-must-reproduce](feedback_queued_is_a_claim_that_must_reproduce.md) — walk masterplan.json, don't grep; 86.90 said "queued" 4x with zero steps
- [unwired-is-a-claim-with-an-expiry](feedback_unwired_is_a_claim_with_an_expiry.md) — "hypothetical second consumer" while a sibling commit 34 min earlier had wired it live (86.85 c10)
- [underivable-is-a-negative-claim-run-the-query](feedback_underivable_is_a_negative_claim_run_the_query.md) — the refusal funnel was live at 93.1%; JSON_VALUE returns NULL on an object so the probe read empty (86.47)
- [regenerated-label-is-a-claim-check-the-diff](feedback_regenerated_label_is_a_claim_check_the_diff.md) — "(REGENERATED)" over a ONE-LINE diff, and a ZERO-line case; `git log` EACH artifact
- [a-pasted-blocks-section-header-is-a-claim](feedback_a_pasted_blocks_section_header_is_a_claim.md) — a "[5]" header over "[8]" ok-lines in a block claiming full regeneration (86.37 c6)
- [regenerating-a-capture-leaves-the-authored-summary-stale](feedback_regenerating_a_capture_leaves_the_authored_summary_stale.md) — the block was regenerated, the TABLE above it stayed at cycle-2 state (86.21 c7)
- [replacement-stops-at-the-line-not-the-sentence](feedback_replacement_stops_at_the_line_not_the_sentence.md) — a 1-line diff on a 2-line sentence leaves the orphan half offering the vacuous command (86.79 c6)
- [swapping-the-operand-leaves-the-arithmetic](feedback_swapping_the_operand_leaves_the_arithmetic.md) — repointed a compare at the RIGHT field, kept the off-by-one; floor unchanged so the new block is skippable (86.79 c4)
- [carried-forward-residuals-go-stale](feedback_carried_forward_residuals_go_stale.md) — a +3 credited to a commit made 7h BEFORE the baseline; check commit dates (86.37 c4)
- [recheck-prior-remediation-list](feedback_recheck_prior_remediation_list.md) — re-derive the PRIOR list yourself; follow-ups hid 3-of-6 and SUBSTITUTED the file set (80.4 c3)
- [rederive-the-label-not-just-the-number](feedback_rederive_the_label_not_just_the_number.md) — a corrected number returned with an invented provenance (80.3 c5)
- [stale-figure-in-gate-artifact](feedback_stale_figure_in_gate_artifact.md) — annotate a dated brief, never rewrite; WARN only if a forward-looking consumer reads it (80.3 c7)
- [verbatim-paste-drift-arithmetic](project_verbatim_paste_drift_arithmetic.md) — cross-suite total arithmetic separates stale transcription (NOTE) from untested change (escalate); count `def test_` in the COMMIT
- [self-referential-counts-cannot-reproduce](feedback_self_referential_counts_cannot_reproduce.md) — a grep over handoff/** grows from writing the artifact that states it; cap only when the number POINTS AT evidence
- [a-red-that-cleared-itself-is-not-a-fix](feedback_a_red_that_cleared_itself_is_not_a_fix.md) — run the PRE-fix source on today's data AND delete the new clause (86.84 c8)

## Scope, environment, reproduction
- [run-the-steps-own-checker-not-the-family](feedback_run_the_steps_own_checker_not_the_family.md) — a sibling's co-commit broke a whole-line literal: matrix ABORTED with ZERO cells while the artifact said "family green" (86.78 c3)
- [derived-scope-lint-use-xargs](feedback_derived_scope_lint_use_xargs.md) — unquoted `$VAR` = ONE arg in zsh, so the tool measures NOTHING and reports success; uniform 0%/100% = this bug
- [derived-scope-misses-untracked-files](feedback_derived_scope_misses_untracked_files.md) — new-file steps make `git diff --name-only HEAD` EMPTY; union `git ls-files --others` (83.0.3)
- [recheck-head-before-returning-a-scoped-grade](feedback_recheck_head_before_returning_a_scoped_grade.md) — `A..HEAD` excludes A, and commits can land mid-eval retracting the claim; name the sha
- [rerun-whole-compound-verification-command](feedback_rerun_whole_compound_verification_command.md) — a pasted `cmd2` output never proves `cmd1` passed (80.3 c1)
- [run-probes-against-head-to-classify](feedback_run_probes_against_head_to_classify.md) — swap HEAD under the same probe to tell a NEW regression from pre-existing (80.5 c2)
- [verify-under-the-production-interpreter](feedback_verify_under_the_production_interpreter.md) — venv is 3.14 but launchd's `python3` is 3.9.6; smoke under `env -i PATH=/usr/bin:/bin` (85.3)
- [repo-wide-eslint-is-red-from-dist-dirs](feedback_repo_wide_eslint_is_red_from_dist_dirs.md) — `npx eslint .` exits 1 from `.next-*` build dirs with 0 errors in src/; group errors by dir before grading
- [bash-n-cannot-see-inside-a-heredoc](feedback_bash_n_cannot_see_inside_a_heredoc.md) — a `bash -n` gate is blind to Python in a quoted heredoc; also assert the mutant exits 0 (86.97)
- [contract-order-mtime-fallback](project_contract_order_mtime_fallback.md) — single-commit steps defeat git-timestamp ordering; use the stat mtime chain (61.2)
- [stepid-grep-escape-dot](project_stepid_grep_escape_dot.md) — step-id greps need `-F`/escaped dot; "67.6" matched "67/67 tests"
- [committed-criterion-gitignore-check](project_committed_criterion_gitignore_check.md) — "file committed" criteria need `git check-ignore` + `ls-files`; a gitignore defeated 3 cycles
- [measure-the-capture-you-didnt-take](feedback_measure_the_capture_you_didnt_take.md) — grade Main's PNGs quantitatively; the gitignored `.playwright-mcp/console-*.log` reproduces console claims

## Harness / rail mechanics
- [workflow-run-record-locations](reference_workflow_run_record_locations.md) — SCRIPT return = `workflows/<runId>.json:.result`; PER-AGENT = `subagents/workflows/<runId>/journal.jsonl`
- [verdict-gate-ignores-per-cycle-json](project_verdict_gate_ignores_per_cycle_json.md) — the 81.2 resolver ignores `_cycleN.json`; 82.0 reached the flip at fail-open
- [no-chance-to-emit-needs-the-error-field](feedback_no_chance_to_emit_needs_the_error_field.md) — the per-agent `error` field named 529/quota; append-only corpus + own eval spawns = permanent red (86.84 c6)
- [run-status-is-not-agent-outcome](feedback_run_status_is_not_agent_outcome.md) — `status=="failed"` buckets `killed` as SUCCESS; query the success side for the failure signature
- [stub-fallback-is-not-a-production-default](feedback_stub_fallback_is_not_a_production_default.md) — `getattr(settings, X, 0)` fires because the stub omits X; diff the kwargs against `Field(...)` (86.74)
- [scheduled-job-fix-evidence](project_scheduled_job_fix_evidence.md) — cron-fix streaks need POST-FIX scheduled nights, not manual runs; `grep|head` cmds are vacuous exit gates
- [criterion-wording-existence-vs-completion](project_criterion_wording_existence_vs_completion.md) — rule criteria on their literal verbs: "writes rows" = existence at eval time
- [premise-embedded-criteria-yfinance-check](project_premise_embedded_criteria_yfinance_check.md) — criteria can embed FALSE premises; overturn+operator-route = PASS, silent edit = FAIL (68.5)
