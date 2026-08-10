# phase-86.12 -- EVALUATE (cycle 1)

**Verdict: CONDITIONAL**  (`ok: False`, `harness_compliance_ok: True`)

Q/A via the Workflow rail, run `wf_db520d8e-a82`. Transcribed VERBATIM.

## reason

5 of 6 immutable criteria MET and independently reproduced; C3 NOT MET. C1 provenance verified (evaluate_breach at kill_switch.py:805 takes current_nav as a parameter and never marks; 4 stored-NAV call sites in 3 functions, 2 caller-supplied -- confirmed by my own grep). C2 re-derived from the raw journal by me: 10 sod_snapshot rows (step's "8" is stale, Main's correction governs), equality 7/10 = SOMETIMES. C4 answered correctly and verified DEEPER than Main proved it: check_and_enforce_kill_switch has exactly ONE production caller (autonomous_loop.py:1400), the only code between mark (:1368) and enforce (:1400) is a try/except scale-out that logs and falls through (no early return, no re-raise), and the seam Main asserted but never demonstrated -- mark_to_market persists via bq.upsert_paper_portfolio({... "total_nav": round(nav,2) ...}) at paper_trader.py:778 while get_or_create_portfolio (:155-159) reads BQ uncached -- does hold, so enforcement reads the NAV Step 5 wrote. Guards are NOT vacuous: 4/4 mutations killed (M1 daily-leg-never-fires 5 failed; M2 armed-always-true 1 failed; M3 FIXTURE anchor forced stale 7 failed; M4 FIXTURE _sod_nav blanked 8 failed), control 12 passed. C5/C6 clean: no threshold changed, the float-boundary finding deliberately NOT fixed, journal sha256 ea78508bee73887c82df2346da408c7281e7e9229334a6131d7fa06c09977065 byte-identical before and after my mutation runs, commit 9c3e0f1a touches zero production files. C3 fails because the $0.06 explanation rests on a premise that does not cover the cockpit: the three sources Main compared are all backend readers of the same stored total_nav (hence 0.000000 spread by construction), while the cockpit NAV tile renders lp.liveNav from LivePortfolioProvider -- a live-priced 60s-polled derivation, a structurally different quantity. The "race across a mark_to_market write" inference is therefore unsupported, and is further undercut by Main's own measurement of a 10.90-hour-old stored NAV (no cycle in flight). The step's option (c) "different endpoint / different asof" is the live explanation and was ruled out prematurely. Fixable with a short trace Main already has the pieces for; the safety conclusion is unaffected.

## violated_criteria

- `C3: the $0.06 delta between the kill-switch current_nav and the cockpit-rendered NAV is EXPLAINED`

## violation_details

### 1. Unjustified_Inference

**action** -- experiment_results_86.12.md Criterion 3 / live_check_86.12.md section 5 conclude: 'All three read the same stored paper_portfolio.total_nav, so they agree exactly by construction. They can only disagree if two of them are served on opposite sides of a mark_to_market write -- a race against the cycle, not rounding and not FX.'

**state** -- The three compared sources (kill-switch current_nav, /performance nav, /portfolio portfolio.total_nav) are ALL backend readers of the same stored paper_portfolio.total_nav, so their 0.000000 spread is true by construction and carries no information about the cockpit. The cockpit NAV tile does NOT read that stored value: frontend/src/app/page.tsx:266-271 takes `const lp = useLivePortfolio(); const liveNav = lp.liveNav;` from LivePortfolioProvider (frontend/src/lib/live-portfolio-context.tsx), a live-priced value polled on a 60s interval; frontend/src/app/paper-trading/layout.tsx:137-146 consumes the same provider and documents the precedent verbatim -- 'The operator-flagged ~$18 NAV gap between Home and Paper Trading (2026-05-26) came from the two pages running INDEPENDENT polling loops; each owned its own setInterval(60_000) and polled at different millisecond offsets.' Main's own live_check section 3 records the stored NAV as 10.90 hours old, i.e. no cycle was in flight, which makes the proposed race mechanism LESS likely rather than more. The delta therefore does not require a race: a stored mark vs a live browser-side repricing sampled at a different instant produces it in ordinary steady state.

**constraint** -- Criterion 3 requires the delta be EXPLAINED and explicitly names 'different endpoint / different asof' as a candidate to rule in or out; the criterion states an unexplained delta between two numbers describing the same quantity is itself the finding. The actual finding is that the two numbers do NOT describe the same quantity -- one is a stored mark, the other a live repricing -- and that candidate was ruled out on a premise (same stored source) that is true of the three backend endpoints measured but false of the cockpit, the one surface the criterion is about. Remedy: trace liveNav's derivation in live-portfolio-context.tsx to file:line, state the delta as a stored-mark-vs-live-reprice asof difference, and (since the criterion concerns a RENDERED value) capture the cockpit NAV live beside a same-instant kill-switch payload. No threshold or guard change is implied.


## certified_fallback

False

## checks_run

- harness_compliance_audit_5_item
- research_gate_before_contract_mtime_chain
- contract_before_generate_mtime_chain
- log_last_masterplan_status_pending_and_zero_harness_log_entries
- no_verdict_shopping_cycle_1_zero_prior_conditionals
- immutable_verification_command_exit_0
- commit_scope_audit_9c3e0f1a_zero_production_files
- git_diff_clean_backend_frontend_scripts
- python_lint_gate_ruff_F821_F401_F811_derived_scope
- syntax_ast
- scoped_pytest_12_tests_baseline_green
- mutation_test_M1_daily_leg_never_fires_KILLED
- mutation_test_M2_armed_always_true_KILLED
- mutation_test_M3_FIXTURE_anchor_forced_stale_KILLED
- mutation_test_M4_FIXTURE_sod_nav_blanked_KILLED
- mutation_control_rerun_green
- guard_vacuity_check_4c
- independent_rederivation_sod_snapshot_row_count
- independent_rederivation_equality_table_7_of_10
- measurement_script_reproduction
- enforcement_caller_enumeration_grep
- mark_to_market_persistence_seam_verification
- portfolio_read_path_cache_check
- frontend_nav_provenance_trace
- journal_sha256_before_and_after
- claim_auditing_4b
- adversarial_worst_of_n_lenses_4a
- code_review_heuristics

## harness_compliance_ok

True

## notes

DISCLOSURES. (1) No Playwright capture was taken by me; the cockpit-NAV finding rests on frontend SOURCE reading (page.tsx:266-271, paper-trading/layout.tsx:137-146, lib/live-portfolio-context.tsx), which is sufficient to establish the SOURCE of the rendered number but not its rendered value. Because C3 concerns a rendered figure, the natural close is a live capture of the cockpit NAV beside a same-instant kill-switch payload; Main's live_check contains no UI capture. I did not raise a separate qa.md §1c cap since the verdict is already CONDITIONAL on C3 and the artifacts make no standalone UI-correctness claim. (2) All mutations were performed IN MEMORY via `python -c` + sys.modules patching + pytest.main -- no file was written, edited or restored, and the live journal sha256 is unchanged after all four runs. (3) NOTE, non-blocking: the label "7/10 comparable days" counts 10 ROWS spanning 9 distinct dates (2026-08-09 appears twice), so it is 7 of 10 row-comparisons, not 7 of 10 days. The qualitative answer (SOMETIMES) is unaffected. (4) NOTE, non-blocking: NAV age is 10.87h in experiment_results and 10.90h in live_check -- consistent with elapsed time between two runs of the same script, not a contradiction. (5) POSITIVE, worth preserving: the artifacts are unusually scope-honest -- Main self-corrected a wrong provenance list transcribed from the research brief's prose (naming a function, _roll_sod_anchor_if_needed, that does not exist) after an AST-derived test failed, stated plainly "I could not observe the delta, so this is a mechanism explanation consistent with the evidence, not a reproduction", and declined to fix the float-boundary display-vs-decision disagreement because doing so would be a threshold change forbidden by C5. The C3 miss is an unsound inference, not concealment. (6) Adversarial worst-of-N lenses: correctness = CONDITIONAL (C3 mechanism unsupported; C4 correct and verified beyond what Main proved); does-it-reproduce = PASS (every number reproduced -- 10 rows, 7/10, 0.000000 spread, sha256, exit codes, 12 tests); scope-honesty = CONDITIONAL (the honesty caveat is attached to a mechanism Main could have falsified by reading the frontend). verdict = min = CONDITIONAL. (7) No code-review heuristic fired: zero production files changed, no secret, no guard weakened, no threshold touched, no consumer contract altered.
