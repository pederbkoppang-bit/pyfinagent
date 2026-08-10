# phase-86.24 -- EVALUATE (cycle 1)

Launched on the Workflow structured-output rail (`.claude/workflows/qa-verdict.js`,
run `wf_c3bf9d5f-4ee`). **Transcribed VERBATIM from the captured return value.**
Main records the verdict; Main never authors it.

## ok

`false`

## verdict

`"CONDITIONAL"`

## certified_fallback

`false`

## harness_compliance_ok

`true`

## reason

All 6 immutable criteria are MET on substance and every deterministic check reproduced independently (immutable cmd exit 0 / 24 passed; ruff exit 0 over the git-derived 4-file scope; mutation matrix 5/5 killed with digests matching; recall table re-derived on pre-fix sources as Oslo 1/3, Kiritimati 1/3, Midway 3/3; differential failure sets identical by symmetric difference; no production source touched, kill_switch.py byte-unchanged; live kill_switch_audit.jsonl byte-identical throughout). CONDITIONAL for two WARN-level rationale/disclosure defects, not behaviour defects: (1) the criterion-3 adjudication reaches the RIGHT conclusion but headlines a support I measured to be FALSE in a band -- with a stale anchor, sod=100/peak=100/nav=95 at limits 4%/10% yields armed=False daily=False trailing=False any_breached=False, so "the trailing leg still fires, the overnight window is not naked" (live_check_86.24.md D; test_phase_86_24_clock_dependence.py:101-102,:115-117) does not hold between the two limits, and the guard only exercises nav=80.0; the decisive mechanism is the roll-before-evaluate ordering (paper_trader.py:1411 roll precedes evaluate_breach at :1460, and :1468 does not read armed). (2) an undisclosed member of the step's own declared blind class sits in the repaired file: test_phase_86_2_replay_poison_row.py:61 computes _UTC_TODAY ONCE at import while kill_switch.py:986 recomputes at call time -- the masterplan's own case-(a) wording -- and the author's M2 mutant is precisely that state and is scored KILLED. Both fixes are small and local; no re-work of the step is required.

## violated_criteria

- `criterion_3_adjudication_rationale_overgeneralized`
- `criterion_1_blind_class_member_undisclosed_in_repaired_file`

## violation_details

### 1. Overgeneralization

**action** -- Adjudicate the kill_switch sod_date staleness as correct-by-design and justify it with 'the date-independent trailing leg keeps firing, so the overnight window is not naked' (experiment_results_86.24.md section 1 row 1; live_check_86.24.md section D; test_phase_86_24_clock_dependence.py:101-102 and the assertion message at :115-117). SEVERITY: WARN.

**state** -- Measured by the evaluator in an isolated process (ks._AUDIT_PATH redirected to tmp; live journal md5 685bf1a5fd7beaa4f15da2babf133ca2 unchanged): STALE anchor sod_nav=100 peak_nav=100 nav=95 daily_limit=4 trailing_limit=10 -> armed=False daily_baseline_stale=True daily_loss_breached=False trailing_dd_breached=False any_breached=False. The same numbers with a FRESH anchor -> daily_loss_breached=True any_breached=True. The trailing leg therefore does NOT cover losses in the band (daily_limit, trailing_limit); the sole guard for the claim, test_a_YESTERDAY_anchor_DISARMS_the_daily_leg_but_the_trailing_leg_still_fires, only exercises nav=80.0 (a 20% drop, above the trailing limit) and so cannot detect the gap it claims to close. The conclusion 'no live defect' is nonetheless CORRECT, verified independently: the only enforcement path (autonomous_loop.py:1400 -> paper_trader.check_and_enforce_kill_switch) re-anchors at paper_trader.py:1411 via sod_anchor_needs_reroll BEFORE evaluate_breach at :1460, the flatten branch at :1468 keys on any_breached and never on armed, check_auto_resume refuses to resume when armed is False (kill_switch.py:1078-1080), and the order gate reads baselines_present (paper_trader.py:216, :1372). Research brief supports 4 and 5 do reach the roll ordering (:1414, :1448) but the graded artifacts headline the weaker claim.

**constraint** -- Criterion 3: 'The kill_switch sod_date anchor is specifically adjudicated: either demonstrate the staleness cannot occur in the running backend, or file it as a live defect.' A demonstration on a money-path safety module must rest on a support that holds across the parameter range it is stated over; a rationale embedded in a test docstring and assertion message becomes the authoritative note a future maintainer relies on. Remedy: restate the harmlessness as the roll-before-evaluate ordering (paper_trader.py:1411 before :1460), and either add a guard at a nav inside the (4%,10%) band asserting any_breached is False with an explicit note that this is safe only because enforcement never sees a stale anchor, or drop the 'not naked' wording. NO assertion is to be weakened -- verified that none was.

### 2. Missing_Assumption

**action** -- Repair test_phase_86_2_replay_poison_row.py by introducing module-level _UTC_TODAY = _dt.datetime.now(_dt.timezone.utc).date() at :61 with helpers _day()/_ts() reading that snapshot, then claim in experiment_results_86.24.md section 5 that 'the population is now empty' and disclose the blind spot in section 7 only as an abstract axis requiring time-machine. SEVERITY: WARN.

**state** -- _UTC_TODAY is computed ONCE at module import, while the value it is judged against is recomputed at call time (kill_switch.py:986, _sod_date_is_stale -> datetime.now(timezone.utc).date()). If UTC midnight falls between this module's import at collection and the execution of test_c1_c2_a_poison_row_first_no_longer_strands_the_replay, the fixture writes yesterday's anchor, the daily leg correctly disarms, and the test goes red -- the exact failure the step exists to remove. This is not hypothetical: the author's own M2 mutant (_UTC_TODAY = _dt.date(2026, 8, 9)) is precisely this state and is scored KILLED, i.e. the module is proven to go red whenever the import-time snapshot is a day behind evaluation time. The sibling module written in the same commit does NOT have this shape: test_phase_86_24_clock_dependence.py:49-51 recomputes inside _day(). The exposure window shrinks from 24h to (collection -> execution), which on the measured 375.86s full-suite run is minutes, but the defect CLASS is not eliminated and this specific member -- introduced by this step, in the file this step repaired -- is not named in any artifact.

**constraint** -- Criterion 1 requires the date-dependence population to be derived and reported, and the masterplan text defines case (a) verbatim as 'a fixture that hard-codes or once-computes a date while the assertion recomputes it'. Criterion 2 requires every member classified, an unclassified member being an open finding rather than a pass. Remedy (one line): make _day()/_ts() recompute datetime.now(timezone.utc).date() per call, matching test_phase_86_24_clock_dependence.py:49-51, or disclose this member explicitly in section 7 as a known residual with its window.

## checks_run

- harness_compliance_audit_5_item
- research_gate_envelope
- artifact_mtime_ordering
- verification_command_immutable
- python_lint_gate_ruff_F821_F401_F811_derived_scope
- backend_runtime_smoke_kill_switch_import_and_evaluate_breach
- git_diff_scope_no_production_source_touched
- assertion_not_weakened_diff_vs_d5180e27_parent
- independent_adjudication_kill_switch_enforcement_path_trace
- isolated_breach_probe_stale_vs_fresh_anchor
- evaluate_breach_caller_enumeration
- criterion_1_recall_table_reproduced_on_prefix_sources
- criterion_4_dual_clock_reproduced
- criterion_5_conftest_sweep_mutated_and_control
- criterion_6_mutation_matrix_rerun
- extra_production_side_mutants_with_identity_wrapper_control
- differential_failure_set_symmetric_difference
- full_suite_arithmetic_reconciliation
- live_kill_switch_journal_byte_identity
- claim_auditing_4b
- guard_vacuity_check_4c
- code_review_heuristics
- third_conditional_counter

## notes

HARNESS COMPLIANCE (clean, 5/5). (1) Research-gate-before-contract: handoff/current/research_brief_86.24.md present, gate_passed:true, external_sources_read_in_full=14 (floor 5), urls_collected=44 (floor 10), recency_scan_performed=true. (2) Contract-before-generate by mtime: captures 09:48/09:54 -> brief 09:59 -> contract 10:02:10 -> macro test 10:02:48 -> poison test 10:04:24 -> new module 10:06:05 -> matrix 10:06:46 -> commit d5180e27 10:07:35. Ordered, though contract-to-first-edit is only 38s. (3) experiment_results_86.24.md (10:21) and live_check_86.24.md (10:22) present and non-empty. (4) Log-last respected: zero "86.24" matches in handoff/harness_log.md and masterplan status still "pending". (5) No verdict-shopping: first Q/A cycle for this step-id, no prior evaluator_critique_86.24, so the 3rd-CONDITIONAL auto-FAIL rule does not apply (prior CONDITIONAL count = 0). retry_count 0 < max_retries 3, so certified_fallback=false.

DETERMINISTIC. Immutable command `pytest backend/tests/test_phase_82_0_macro_ingestion.py backend/tests/test_phase_86_2_replay_poison_row.py -q` -> exit 0, "24 passed in 1.11s", reproduced twice. Lint gate over a git-DERIVED scope (git diff --name-only d5180e27^ d5180e27 -- '*.py' unioned with untracked, non-empty asserted, passed via xargs -0 so no zsh word-split): 4 files, `uvx ruff check --select F821,F401,F811` -> "All checks passed!", exit 0. Frontend gates N/A (no frontend/** in diff). Live-UI gate N/A (no UI claims). Runtime smoke: the diff touches NO production module (commit contains only backend/tests/ x3, scripts/qa/ x1, handoff/ x2); I nevertheless exercised backend.services.kill_switch live in-process (import + KillSwitchState construction + evaluate_breach) as the strongest available smoke. kill_switch.py last changed at 481be943 (phase-86.2) -- byte-unchanged by this step, as claimed.

BOOK PROTECTION. handoff/kill_switch_audit.jsonl md5 685bf1a5fd7beaa4f15da2babf133ca2, sha256 ea78508bee73887c82df2346da408c7281e7e9229334a6131d7fa06c09977065, 64 lines -- verified identical before and after every action I took (matrix run, 4 mutant runs, pre-fix copy runs, all module runs). The artifact's stated sha256 prefix matches mine exactly. No mutating HTTP to :8000, no backend restart, no dev-server lifecycle touched, no repo file written; all mutants were temp-named copies removed in a finally, and I re-verified zero strays and unchanged tracked digests after my own runs.

CRITERION-BY-CRITERION. C1 MET -- I re-derived the recall table myself on PRE-FIX sources extracted from d5180e27^ into temp-named copies: TZ unset 1/3, Europe/Oslo 1/3, Pacific/Kiritimati 1/3, Pacific/Midway 3 of 3 with the three known positives named individually (not merely counted); pre-fix totals "1 failed, 23 passed" and "3 failed, 21 passed" reproduce live_check section A verbatim. Method A (own-clock AST scan) was REJECTED not adjusted, and I confirmed the structural reason from the diff: the pre-fix poison-row module has no datetime import at all, so it contains zero clock calls. C2 MET -- 4 members, each classified with evidence, none left open. C3 MET on substance with the WARN above; critically, NO assertion was weakened: diff against d5180e27^ shows `assert r["daily_loss_breached"] is True` unchanged and the ONLY edits in that file are date literals replaced by _day()/_ts(). C4 MET -- I reproduced both runs: system clock (mid-day, 08:2x UTC) 24 passed, TZ=Pacific/Midway (local one day behind UTC) 24 passed; M4 proves the in-suite differential's positive control is load-bearing. C5 MET and proven non-vacuous by execution -- I mutated the sweep: it visits 35 real conftest files (non-vacuous control) with 0 offenders, detects a planted freezegun/freeze_time conftest, and all 7 suspect tokens are individually detected. C6 MET and stronger than claimed -- matrix re-run gives 5/5 KILLED, digests 566a607e91365c67 / 5cf5073d39707e6d / 03ad07ced183b80d matching live_check exactly, zero strays, exit 0; the harness's anchor-uniqueness check (n != 1 scored as SURVIVED) correctly closes the silent-no-op-replace hole.

EVALUATOR-ADDED MUTANTS (beyond the author's matrix, which mutates only tests). Against PRODUCTION, via in-memory patching with no repo write: MX1 `_sod_date_is_stale -> always False` (reverting phase-36.9) -> 5 failed, KILLED; MX2 `any_breached` recomputed to ignore the trailing leg -> 5 failed, KILLED. An identity-wrapper control ran green (8 passed), so per the two-mutant-forms discipline these are real kills and not wrapper artifacts. This establishes that the new guards pin PRODUCTION behaviour, not just fixture shape -- the fixture/harness-side vacuity shapes (3/5/6) are not present here.

CLAIM AUDIT (4b). Every quantified claim reproduced. "16 -> 15 failures, 3351 -> 3360 passes" reconciles at MEMBER level, not merely cardinality: PRE-FIX minus POST-FIX = exactly {test_c1_c2_a_poison_row_first_no_longer_strands_the_replay}, POST-FIX minus PRE-FIX = EMPTY, and the new module contributes exactly 8 tests (I measured "8 passed"), so 3351+1+8 = 3360. The post-fix differential's BASE and SHIFTED failure sets have EMPTY symmetric difference, so "DELTA EMPTY" is a member-level result, not two equal counts. I did NOT re-run the 375.86s + 368.58s full-suite pair (~12.5 min, above the scoped-test tier); it is accepted on the capture plus exact member-level reconciliation against the independently-captured pre-fix set. The captures show no splice tells.

SCOPE HONESTY. Unusually good and partly against interest: section 7 discloses that the population is empty only along the covered axis, that time-machine was NOT added unilaterally (correct -- that is an operator ask, not a Q/A blocker), that .json/.jsonl/.csv/.sql fixture dates were unswept, that kill_switch.py is byte-unchanged, that the 15 remaining failures are pre-existing (I verified they are identical under both clocks), and voluntarily that the phase-86.27 fix ebeb03da was surfaced by this step's differential and has NOT been graded by a Q/A. ebeb03da is outside this step's commit and is NOT part of what I graded; it should be queued for its own gate. One NOTE-level inaccuracy, same root as the C3 WARN: research brief support 5 quotes paper_trader.py:1263-1265 ("a stale anchor DISARMS the daily leg (fail-safe: it refuses to trade)") -- in the enforcement path a merely-stale anchor does NOT refuse to trade, because baselines_present_in is presence-only (sod>0 and peak>0) and a stale-but-present anchor passes the order gate; that quote is local to roll_daily_anchor and does not generalize.

CODE-REVIEW HEURISTICS. No BLOCK fired. Diff is test-only, no secrets, no subprocess/eval on non-literal input, no kill-switch reachability or stop-loss change, no perf_metrics bypass, no dependency pin removed (and time-machine correctly NOT added unilaterally). illusory-guard checked and cleared: guards execute production evaluate_breach rather than scanning source; the one source-scan guard (conftest sweep) is criterion-mandated and I proved it fires; a positive control (test_a_TODAY_anchor_arms_the_daily_leg) is present against vacuity shape 5; and the module records rather than hides an earlier unsound version of its differential test (docstring :161-167). Two WARNs -> CONDITIONAL per the severity dispatch table.

PATH TO PASS (both edits are small, local, and testable; then a fresh Q/A on changed evidence): (i) restate the criterion-3 harmlessness in live_check_86.24.md section D, experiment_results section 1, and the test docstring/assert message as the roll-before-evaluate ordering (paper_trader.py:1411 precedes :1460; :1468 does not read armed), and either add a guard inside the (4%,10%) band or drop the "not naked" wording; (ii) make _day()/_ts() in test_phase_86_2_replay_poison_row.py recompute per call as the sibling module does, or disclose the import-time-snapshot residual explicitly in section 7. Do NOT weaken any kill-switch assertion in either edit.


---

# phase-86.24 -- EVALUATE (cycle 2)

Launched on the Workflow structured-output rail (`.claude/workflows/qa-verdict.js`,
run `wf_bbd40df2-100`) on CHANGED evidence -- the cycle-1 verdict's own two
remedies were executed in `7eb85983` / `14b8d32b`. **Transcribed VERBATIM.**
Main records the verdict; Main never authors it.

## ok

`false`

## verdict

`"CONDITIONAL"`

## certified_fallback

`false`

## harness_compliance_ok

`true`

## reason

All 6 immutable criteria are MET on substance and every deterministic check reproduced independently on the cycle-2 tree (immutable cmd exit 0 / 24 passed, run twice; ruff F821/F401/F811 exit 0 over a git-DERIVED 4-file scope with the non-empty guard asserted; author matrix re-run by me = 7/7 KILLED with all three digests matching live_check F exactly and zero strays; criterion-4 dual clock reproduced by me at 34 passed under TZ=Pacific/Midway (local 2026-08-09 != UTC 2026-08-10) and 34 passed at 11:02 CEST mid-day; criterion-5 sweep re-derived independently -- 34 distinct conftests, zero freeze tokens, no time-freezing library in requirements or site-packages; differential symmetric difference EMPTY at MEMBER level, 15/15 identical failure names both arms; no production source touched, kill_switch.py byte-unchanged; live kill_switch_audit.jsonl md5 685bf1a5fd7beaa4f15da2babf133ca2 / 64 lines byte-identical before and after every action I took). BOTH cycle-1 findings are genuinely fixed, not softened: I reproduced the band table myself and it matches live_check D exactly (STALE anchor sod=100/peak=100 at 4%/10%: nav 99/95/92/90.1 all any_breached=False; 89/80 fire via trailing; the same navs against a FRESH anchor breach), and I independently verified the corrected rationale by enumerating ALL SIX evaluate_breach callers -- paper_trader.py:1413 re-anchors before :1460 and :1468 keys on any_breached; the pre-roll call at :1357 feeds only baselines_present (:1372); paper_trading.py:581 and kill_switch.py:1065 both additionally refuse on armed=False (fail-safe); paper_trading.py:518 and risk_server.py:78 are read-only -- so the band is unreachable by any enforcement caller in the unsafe direction and there is NO live defect. The recompute fix is real and its guard is correctly attributed (my hand-built M6 mutant fails on the 'SNAPSHOTTED, not recomputed' assert, not an import error; a bogus seam path raises FileNotFoundError, so the seam cannot silently no-op). CONDITIONAL for ONE WARN-level violation: the cycle-1 support Main itself withdrew as FALSE-IN-A-BAND survives verbatim and unannotated in LIVE SOURCE at backend/tests/test_phase_86_2_replay_poison_row.py:55-58, in a comment block cycle 2 edited two lines below, contradicting live_check section D's claim that the claim 'is replaced rather than softened'. The fix is one comment block.

## violated_criteria

- `criterion_3_withdrawn_rationale_survives_in_live_source_and_gate_artifacts`

## violation_details

### 1. Overgeneralization

**action** -- Cycle 2 (7eb85983 / 14b8d32b) withdraws the criterion-3 support 'the rule is per-LEG, so the trailing leg keeps firing' and states the withdrawal is complete -- live_check_86.24.md:60-62 'the claim is replaced rather than softened', experiment_results_86.24.md:29 '~~the rule is per-LEG, so the trailing leg keeps firing~~ WITHDRAWN IN CYCLE 2 -- THIS WAS FALSE IN A BAND' -- while editing the very comment block in backend/tests/test_phase_86_2_replay_poison_row.py that carries the same proposition. SEVERITY: WARN.

**state** -- The withdrawn proposition survives verbatim and unannotated in three places, one of them live source. (1) backend/tests/test_phase_86_2_replay_poison_row.py:55-58: 'ADJUDICATED in phase-86.24: the staleness rule is CORRECT. It is per-LEG (the date-independent trailing leg still fires), the order gate reads `baselines_present` rather than `armed`...'. Commit 7eb85983 rewrote the comment starting at :61 (the RECOMPUTED-PER-CALL block) and left :55-58 standing directly above it, so the miss is inside the edited region, not out of reach. That comment states the withdrawn support as THE standing adjudication, names none of the roll-before-evaluate ordering that is now the decisive support (paper_trader.py:1413 before :1460; :1468 keys on any_breached), and its pointer 'the stale case gets its own test below' refers to a test that lives in a different module (test_phase_86_24_clock_dependence.py). A maintainer of the poison-row module -- the one module whose assertion requires an ARMED daily leg -- reads only the wrong reason. (2) handoff/current/contract_86.24.md:32-35 carries the strongest form, 'the date-independent trailing leg still fires. The book is not uncovered.', unannotated; grep for band/roll-before/1413 in that file returns 1 incidental hit. (3) handoff/current/research_brief_86.24.md:24, :101, :356, :441 unannotated; grep for WITHDRAWN|struck|FALSE IN A BAND in the brief returns ZERO hits. Measured by me in an isolated process (ks._AUDIT_PATH redirected to tmp, live journal md5 685bf1a5fd7beaa4f15da2babf133ca2 unchanged): STALE anchor sod=100 peak=100 at limits 4%/10% -> nav 99.0/95.0/92.0/90.1 all give armed=False daily_loss_breached=False trailing_dd_breached=False any_breached=False, i.e. the trailing leg does NOT fire anywhere in (4%,10%); the same navs against a FRESH anchor give daily_loss_breached=True any_breached=True. So the surviving sentence is false in exactly the band Main's own cycle-2 test now pins. The conclusion (no live defect) is CORRECT and I verified it independently -- this is a rationale-of-record defect, not a behaviour defect, and no assertion was weakened (assert r['daily_loss_breached'] is True is byte-unchanged across d5180e27^..HEAD; the only edits in that file are date literals -> _day()/_ts() plus comments).

**constraint** -- Criterion 3: 'The kill_switch sod_date anchor is specifically adjudicated: either demonstrate the staleness cannot occur in the running backend, or file it as a live defect.' The cycle-1 verdict's constraint added: 'a rationale embedded in a test docstring and assertion message becomes the authoritative note a future maintainer relies on', and qa.md section 4b requires a completeness claim to hold over its own population -- 'replaced rather than softened' is such a claim and it does not survive a recall test over the tree. Remedy (small, local, testable): rewrite test_phase_86_2_replay_poison_row.py:55-58 to state the roll-before-evaluate ordering (paper_trader.py:1413 before :1460; :1468 keys on any_breached, never armed) and cross-reference test_a_stale_anchor_leaves_the_band_between_the_two_limits_UNCOVERED in test_phase_86_24_clock_dependence.py; add a dated one-line annotation to contract_86.24.md section 2 and to research_brief_86.24.md supports 1/4/5 rather than rewriting those dated artifacts; and narrow live_check section D to say WHERE the claim was replaced. Do NOT weaken any kill-switch assertion in the edit.

## checks_run

- harness_compliance_audit_5_item
- research_gate_envelope
- artifact_mtime_ordering
- verification_command_immutable_run_twice
- python_lint_gate_ruff_F821_F401_F811_git_derived_scope
- backend_runtime_smoke_kill_switch_and_paper_trader_import
- ast_parse_all_touched_files
- git_diff_scope_no_production_source_touched
- commit_attribution_of_every_file_in_range
- assertion_not_weakened_cumulative_diff_vs_d5180e27_parent
- independent_band_measurement_stale_vs_fresh_anchor
- evaluate_breach_caller_enumeration_all_six
- roll_before_evaluate_ordering_verified_at_source
- criterion_4_dual_clock_reproduced_by_evaluator
- criterion_5_conftest_sweep_independently_rederived
- criterion_6_author_mutation_matrix_rerun_7_of_7
- evaluator_added_mutation_cells_E1_E2_E3
- m6_kill_mechanism_attribution_check
- env_test_seam_silent_noop_probe
- production_side_mutation_sod_date_is_stale
- differential_failure_set_symmetric_difference_member_level
- full_suite_arithmetic_reconciliation
- ast_scan_for_module_level_clock_reads_in_touched_files
- live_kill_switch_journal_byte_identity
- claim_auditing_4b
- guard_vacuity_check_4c
- code_review_heuristics
- third_conditional_counter
- no_verdict_shopping_evidence_changed

## notes

HARNESS COMPLIANCE (clean, 5/5). (1) Research-gate-before-contract: handoff/current/research_brief_86.24.md present, gate_passed:true, external_sources_read_in_full=14 (floor 5), urls_collected=44 (floor 10), recency_scan_performed=true, audit-class coverage.dry=true after 8 rounds. (2) mtime order: brief 09:59:39 -> contract 10:02:10 -> first code edit 10:02:48; cycle-2 edits 10:36:43/10:38:56/10:39:34 all AFTER the cycle-1 critique 10:36:23. (3) experiment_results_86.24.md (10:54:34) and live_check_86.24.md (10:55:05) present, non-empty, both updated in cycle 2. (4) Log-last respected: `grep -E "phase=86\.24" handoff/harness_log.md` = 0 matches (escaped dot per the 67.6 precedent); masterplan status still "pending". (5) NOT verdict-shopping: the evidence CHANGED -- code 7eb85983 and artifacts 14b8d32b vs the cycle-1 tree 70e646b7, and both changes execute the cycle-1 verdict's own two remedies. Prior CONDITIONAL count for this step-id = 1 (on-disk critique; zero in harness_log by log-last design), so this is the 2nd and the 3rd-CONDITIONAL auto-FAIL rule is NOT armed. retry_count 0 < max_retries 3 -> certified_fallback=false.

SCOPE. 86.24's own four commits (d5180e27, 7a829c09, 7eb85983, 14b8d32b) touch exactly: 3 test modules + scripts/qa/mutation_matrix_86_24.py + handoff docs. NO production source. scripts/qa/verify_research_gate_workflow.mjs appears in the d5180e27^..HEAD range but belongs to phase-86.28 (d2e987f1, a6c3c3f3) -- attributed, not this step's. kill_switch.py byte-unchanged, as claimed.

BOOK PROTECTION. handoff/kill_switch_audit.jsonl md5 685bf1a5fd7beaa4f15da2babf133ca2, sha256 ea78508bee73887c82df2346..., 64 lines -- verified identical before and after EVERY action I took (two matrix runs including my three evaluator cells, four hand-built mutant runs, the production-mutant runs, the band probe). No mutating HTTP to :8000, no backend restart, no dev-server touched, no repo file written by me; all my mutants went to the scratchpad or through the author's self-cleaning harness; `git status --short backend/ scripts/` empty after my work and zero test_zz_mutant_* strays.

CRITERION-BY-CRITERION. C1 MET -- the method is a differential run at a shifted clock whose recall is OBSERVABLE, validated before use (Midway = the only TZ where local != UTC, reproduced by me: local 2026-08-09 vs UTC 2026-08-10). Method A rejected STRUCTURALLY not by tuning, and I confirmed the reason from the diff: the pre-fix poison-row module had no datetime import at all, so it contained zero clock calls. C2 MET -- 4 members each classified with evidence, none left open; the three other date.today() uses at :215/:241/:275 are classified by measurement (inert under the shift), not by reading margins. C3 MET on substance with the WARN above -- see reason for my independent six-caller enumeration; the two resume gates fail SAFE on a stale anchor and the two read-only callers cannot flatten. C4 MET, reproduced by me: 34 passed under TZ=Pacific/Midway and 34 passed at 11:02 CEST, 34 progress dots each, internally consistent. C5 MET -- my own find gives 34 distinct conftests (the in-suite sweep reports 35 because the root conftest is matched by both globs, a benign double-visit), zero freeze tokens, and freezegun/time-machine/libfaketime are absent from requirements and site-packages. C6 MET -- 7/7 KILLED on my re-run with digests 566a607e91365c67 / 5c1ce1116769d118 / 36f469402a7e8333 matching live_check F exactly.

EVALUATOR-ADDED MUTANTS (beyond the author's matrix). E3 (give the band test's FRESH-anchor CONTROL a stale anchor) -> KILLED, so the control half is live. PRODUCTION-side, via in-memory plugin with no repo write: _sod_date_is_stale -> always False kills 6 of the new module's 10 tests including the band test (fails at `assert r["armed"] is False`), so these guards pin PRODUCTION behaviour, not fixture shape. TWO SURVIVORS, both adjudicated rather than reported raw: E1 (snapshot _utc_today at IMPORT in the macro module -- the same once-computes shape, in the OTHER repaired module) SURVIVED; it is a REAL residual but it is a hypothetical future regression, not a revert of a fix that was made, and the axis is already disclosed in section 7, so it is a NOTE not a violation -- worth recording that Main's own cycle-2 technique (an injected clock that advances a day) closes this WITHOUT time-machine and was applied to only one of the two repaired modules. E2 (re-pin _ts while _day stays relative) SURVIVED but is an EQUIVALENT mutant: kill_switch consumes row `ts` only for ordering (the keyed.sort at kill_switch.py:283) and never compares it to now, so there is no behavioural differential -- NOT a finding.

CLAIM AUDIT (4b). Every quantified cycle-2 claim reproduced. "24 passed" exit 0 (twice), "10 passed" new module (collect-only confirms exactly 10 tests), "34 passed" under Midway, ruff exit 0 over 4 files (count asserted, non-empty guard enforced, passed via xargs -0 so no zsh word-split), matrix 7/7 with matching digests, journal 64 lines / ea78508b prefix. The full-suite differential capture reconciles at MEMBER level, not merely cardinality: I parsed both arms -- 15 FAILED names each, symmetric difference EMPTY, both summaries "15 failed, 3362 passed", and 3360 + 2 = 3362 matches the two cycle-2 additions. No splice tells. I did NOT re-run the 376.94s + 373.19s pair (12.5 min, above the scoped-test tier); it is accepted on the capture plus the member-level parse.

SCOPE HONESTY. Strong and partly against interest: section 7 discloses the TZ-does-not-move-UTC blind spot and that time-machine was NOT installed unilaterally (correct -- an operator ask, not a Q/A blocker), that .json/.jsonl/.csv/.sql fixture dates were unswept, that kill_switch.py is byte-unchanged, that the 15 remaining failures are pre-existing (I verified they are identical under both clocks), the new PYFINAGENT_86_24_PROW_PATH test seam, and voluntarily that the phase-86.27 fix ebeb03da has NOT been graded by a Q/A. Two NOTES on that last item, neither degrading: 86.27 is already status=done in the masterplan while its test was repaired post-hoc without a gate, so it belongs in its own queued step per the standing queue-discovered-defects rule; and ebeb03da is outside 86.24's commit range and is NOT part of what I graded.

CODE-REVIEW HEURISTICS. No BLOCK fired. Diff is test-only; no secrets, no subprocess/eval on non-literal input, no kill-switch reachability change, no stop-loss or perf_metrics change, no dependency pin added or removed. illusory-guard checked by EXECUTION and cleared: the guards run production evaluate_breach rather than scanning source; the one source-scan guard (conftest sweep) is criterion-mandated and I re-derived its population independently; positive controls exist against vacuity shape 5 (test_a_TODAY_anchor_arms_the_daily_leg, the fresh-anchor control, the TZ-shift positive control at :261-264); the seam cannot silently no-op; and the M6 kill is attributed to the correct assertion, closing vacuity shape 11. One WARN -> CONDITIONAL per the severity dispatch table.

PATH TO PASS (one comment block plus two annotations, then a fresh Q/A on changed evidence): rewrite test_phase_86_2_replay_poison_row.py:55-58 to the roll-before-evaluate ordering with a cross-reference to the band test; annotate (do not rewrite) contract_86.24.md:32-35 and research_brief_86.24.md supports 1/4/5 as withdrawn-in-cycle-2; narrow live_check section D to name where the claim was replaced. Weaken no assertion. Optional, NOTE-level: extend the injected-clock recompute property test to the macro module's _utc_today so E1 dies too.
