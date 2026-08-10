# phase-86.22 -- EVALUATE (cycle 1)

**Verdict: FAIL**  (`ok: False`)

Q/A launched via the Workflow structured-output rail, run `wf_185a8e36-06e`.
Transcribed VERBATIM from the returned object. Main records the verdict; Main never authors it.

> The FIRST spawn (`wf_eb41dff8-828`) DROPPED -- 178K tokens, 41 tool uses, no
> StructuredOutput call, nothing recoverable past its opening line. That is the
> documented long-prompt failure mode. An empty return is NO VERDICT, never a
> PASS; this verdict comes from the re-run with a leaner prompt.

## reason

C8 is missed on its own terms. It requires "reverting the normalisation at each fixed site individually ... a guard whose mutant survives does not count". The 11-cell matrix mutates only recommendation_vocab.py (V1-V7) and the detector script (D1-D4) -- it never reverts a fixed SITE. I ran that missing axis independently (sys.modules injection of each consumer's a87add72^ source; zero tree writes, md5s 71a82b63.../4bf4c85c... verified unchanged after): reverting backend/services/outcome_tracker.py, backend/agents/memory.py, backend/api/portfolio.py and backend/slack_bot/formatters.py to pre-fix each leaves the suite fully GREEN (46 passed); only bias_detector (2 failed) and conflict_detector (1 failed) are killed. 4 of 6 migrations are unguarded, including BOTH learning-path consumers -- the ones the step calls the reason it is P1. C1 fails with it: it names outcome_tracker.evaluate_recommendation as the function to be DRIVEN, and no test drives it (grep across backend/tests: the only hits are MagicMock in test_phase_35_1_learn_loop_writer.py:88/96). test_outcome_tracker_labels_a_correct_buy_call_CORRECT, whose docstring calls it "the load-bearing behavioural assertion", imports OutcomeTracker, builds it with __new__, recomputes directionally_correct IN THE TEST BODY from is_buy_intent/is_sell_intent, and asserts only `assert t is not None and Settings is not None` about the module -- qa.md 4c shape #7 (re-implemented) plus shape #4 (tautology). The reproduce half is likewise a local copy (_legacy_title_case). The product code looks correct and the numbers are real -- I re-derived the distribution against financial_reports.analysis_results with my own canonicaliser and got HOLD 275 / Hold 115 / BUY 91 / Buy 39 / Sell 16 / Strong Buy 5 / N/A 2, TOTAL 543 and 91/543 = 16.8%, exact; and the recorded 23-site/17-offender population reproduces exactly when the rev is pinned (--against-git-rev a87add72^). This is a guard defect, not a fix defect: extend the matrix to the six sites and add consumer-level assertions for the four that survive.

## violated_criteria

- `C8_mutation_test_every_new_guard_including_reverting_each_fixed_site`
- `C1_reproduce_then_fix_must_drive_outcome_tracker.evaluate_recommendation`
- `C5_per_consumer_non_directional_assertions_partial`
- `illusory-guard`
- `tautological-assertion`

## violation_details

### 1. Threshold_Not_Met

**action** -- Independent site-level mutation the matrix omits: for each of the 6 migrated consumers, inject its a87add72^ source into sys.modules (no tree writes) and run backend/tests/test_phase_86_22_outcome_tracker_bias_detector_vocabulary.py

**state** -- CONTROL rc=0 46 passed. outcome_tracker reverted -> rc=0 46 passed (SURVIVED). memory reverted -> rc=0 46 passed (SURVIVED). api/portfolio reverted -> rc=0 46 passed (SURVIVED). slack_bot/formatters reverted -> rc=0 46 passed (SURVIVED). bias_detector reverted -> rc=1 2 failed (killed). conflict_detector reverted -> rc=1 1 failed (killed). 4 of 6 survive. Tree integrity after the run: md5 recommendation_vocab.py=71a82b632375ff0e7f983104dddb55b5, derive_recommendation_consumers_86_22.py=4bf4c85c2115e3e301333ec050a24480 (identical to the matrix's own recorded values); git status --short -- backend/ scripts/ empty.

**constraint** -- [BLOCK] C8 verbatim: 'MUTATION-TEST every new guard, including reverting the normalisation at each fixed site individually, and confirm each mutant is killed by the assertion that names it -- a guard whose mutant survives does not count.' The 11-cell matrix mutates recommendation_vocab.py and the detector only; the per-site axis the criterion names was never run.

### 2. Circular_Reasoning

**action** -- Read backend/tests/test_phase_86_22_outcome_tracker_bias_detector_vocabulary.py:136-147 and grep -rn 'evaluate_recommendation' backend/tests/

**state** -- test_outcome_tracker_labels_a_correct_buy_call_CORRECT imports OutcomeTracker, instantiates via __new__, then computes `directionally_correct = (is_buy and 12.0 > 0) or (is_sell and 12.0 < 0)` INSIDE the test from is_buy_intent/is_sell_intent, and its only module-level assertion is `assert t is not None and Settings is not None  # imports resolve`. evaluate_recommendation is never called by any test in the repo -- the sole references are MagicMock at test_phase_35_1_learn_loop_writer.py:88 (`instance.evaluate_recommendation.return_value`) and :96 (`assert instance.evaluate_recommendation.called`). The REPRODUCE half likewise executes `_legacy_title_case(...)`, a local copy of the pre-fix expression, not the consumer.

**constraint** -- [BLOCK] C1 verbatim requires 'a title-case consumer (outcome_tracker.evaluate_recommendation) is DRIVEN with the literal BUY and shown to yield directionally_correct=False even when the return is positive'. qa.md 4c shape #7 (re-implemented test executing a COPY of the logic) + shape #4 (tautology true by construction); SKILL.md Dim-4 tautological-assertion [BLOCK] and illusory-guard [BLOCK when sole coverage].

### 3. Missing_Assumption

**action** -- Map C5's 'assert PER CONSUMER that HOLD/Hold/Sell/N-A/empty/None still produce the same non-buy outcome' onto the guard set

**state** -- Consumer-level negatives exist only for bias_detector (test_bias_detector_does_NOT_fire_on_a_hold_or_an_unparseable_value, driving the real detect_biases entry point) and conflict_detector (test_conflict_detector_does_not_grade_a_SELL_as_a_BUY, driving the real _check_recommendation_alignment). For outcome_tracker, memory, api/portfolio and slack_bot/formatters there are no consumer-level assertions at all -- proven by their surviving mutants -- so C5 holds at vocabulary level only for four of six consumers.

**constraint** -- [WARN] C5: 'Assert PER CONSUMER that HOLD, Hold, Sell, N/A, empty and None still produce the same non-buy outcome they produce today.'

### 4. Unjustified_Inference

**action** -- Test the stated reason for excluding skill_optimizer: read backend/agents/skill_optimizer.py:189-199 and :243, then measure the column with an independent BQ query

**state** -- The exclusion reason in experiment_results section 4 and in the detector ALLOWED map is 'reads the schema-enforced Literal at agents/schemas.py; the producer cannot emit another spelling'. In fact :243 reads `report.get("debate_consensus")` from the SELECT at :189-199 over `self.bq.reports_table` = financial_reports.analysis_results -- the SAME persisted table whose sibling `recommendation` column carries three dialects. A Literal/enum is demonstrably not evidence of the persisted spelling here: backend/api/models.py:22 defines STRONG_BUY = "Strong Buy" (UPPER_SNAKE name, TITLE-CASE value). MEASURED distribution of debate_consensus: '' 487, NULL 51, 'HOLD' 4, 'BUY' 1 -- so the exclusion is correct in EFFECT today and hides no live defect, but the argument that supports it is wrong and is the same inference this step exists to disprove. The step's own contract demanded 'MEASURED PROVENANCE, confirmed at source rather than assumed'.

**constraint** -- [WARN] qa.md 4b: a set-membership/scope claim must be reproducible from a stated command; the allow-list is a scope the author chose, so each entry's reason must survive checking.

### 5. Invalid_Precondition

**action** -- Re-run the commands live_check block A prints, as printed

**state** -- The header says 'Every block below is captured stdout ... Regenerate with the commands shown.' `derive_recommendation_consumers_86_22.py --against-git-rev HEAD` was captured at 01:47 when HEAD was pre-fix; HEAD is now 330a9b45, so the same command today prints 'population at git rev HEAD: 6 in-scope site(s)', not the recorded 23/17. The recorded output is genuine, not fabricated -- pinning the rev reproduces it exactly ('population at git rev a87add72^: 23 in-scope site(s) / NOT on the allow-list: 17'). Fix is to pin the SHA in the artifact.

**constraint** -- [WARN] Prompt criterion 1: each of the four named artefacts must be 'present and REPRODUCIBLE'; qa.md 4b: a verbatim capture must be regenerable by the command it prints.

### 6. Overgeneralization

**action** -- Independent repo-wide search for surviving consumers of the same column outside the Python AST detector's reach

**state** -- Four TS/TSX display consumers of the same recommendation string, in the same three dialects, are undisclosed: frontend/src/components/reports-columns.tsx:16 and ReportCompareDrawer.tsx:20 (`norm === "STRONG BUY" || norm === "BUY"` -- title/space dialect), RecentReportsTable.tsx:34 (`r.includes("STRONG_BUY") || r.includes("STRONG BUY")`), DebateView.tsx:79 (`c.includes("STRONG_BUY") || c.includes("BUY")` -- substring chain). This is the same DISPLAY risk class as slack_bot/formatters._rec_color, which WAS migrated on display grounds, so the two are treated inconsistently. Not introduced by this step and outside a Python AST scan by construction -- but the artefacts never mention a frontend population exists, and criterion 6 asks for per-site risk profiles.

**constraint** -- [WARN] Prompt criterion 6 (per-site risk profiles stated: learning vs analysis vs reporting vs display) + the standing queue-discovered-defects rule (every out-of-scope defect gets its own step, never a prose omission).


## certified_fallback

False

## checks_run

- qa.md_read_at_runtime
- harness_compliance_audit_5_items
- research_gate_envelope
- contract_before_generate_mtime_chain
- log_last_masterplan_status
- immutable_verification_command
- python_lint_gate_ruff_F821_F401_F811_derived_scope
- pre_fix_lint_baseline_delta_classification
- backend_runtime_import_smoke
- independent_bigquery_re_derivation_of_distribution
- independent_canonicaliser_recomputation_of_91_543_16.8pct
- independent_site_level_mutation_all_6_consumers
- tree_integrity_md5_after_mutation
- detector_reproduction_at_pinned_sha
- detector_allow_list_audit
- skill_optimizer_exclusion_provenance_measurement
- independent_seventh_consumer_search_py_and_ts
- test_phase_82_12_line_registry_check
- code_review_heuristics
- claim_audit_4b
- guard_vacuity_4c
- adversarial_worst_of_lenses

## harness_compliance_ok

True

## notes

HARNESS COMPLIANCE -- clean, all 5 items. (1) Research gate: handoff/current/research_brief_86.22.md, envelope gate_passed=true, external_sources_read_in_full=13, recency_scan_performed=true, audit_class coverage dry=true after 2 dry rounds. (2) Contract-before-generate: contract_86.22.md 01:13:02 precedes every production edit (recommendation_vocab.py 01:14:26, outcome_tracker.py 01:15:00, conflict_detector.py 01:15:50, tests 01:25:45, detector 01:30:35). (3) experiment_results_86.22.md + live_check_86.22.md present. (4) Log-last: no `result=` entry for 86.22 in harness_log.md (the single grep hit is 86.20's "Filed, not fixed" reference); masterplan status=pending, retry_count=0. (5) No verdict-shopping: cycle 1, zero prior verdicts, so the 3rd-CONDITIONAL rule is not engaged.

DETERMINISTIC. Immutable command reproduced: 188 passed, 3097 deselected, exit=0 (10.21s here vs the recorded 6.92s -- timing only). Ruff F821/F401/F811 over a git-derived scope (12 files: commit a87add72^..a87add72 union working tree union untracked; non-empty asserted, xargs -0 so zsh cannot word-split it) exits 1 with 7 F401. ALL 7 ARE PRE-EXISTING -- I ran the same rule against `git show a87add72^:<file>` piped to `ruff --stdin-filename` and got the identical set (bias_detector json; conflict_detector json + typing.Optional; memory json; outcome_tracker timedelta + typing.Optional + compute_benchmark_return). Delta introduced by this step = 0, so this is a NOTE, not the finding. All 8 changed/related backend modules import clean in the venv. test_phase_82_12_string_column_guards.py: 30 passed; the re-derived registry entry (line 109) sits inside the +/-6 tolerance -- I read the file and the actual `_ad = report["analysis_date"]` read is at :108 with the branches at :110/:112, which is within tolerance either way; I did not independently pin what `schema_oracle._row_key_reads` reports, so I make no claim of an error there.

WHAT IS GENUINELY STRONG, stated so the fix is not over-scoped. The numbers are real and survive independent re-derivation -- I queried financial_reports.analysis_results directly and applied MY OWN canonicaliser, not the author's: HOLD 275 / Hold 115 / BUY 91 / Buy 39 / Sell 16 / Strong Buy 5 / N/A 2, TOTAL 543, and rows classified differently by the title-case dialect = 91/543 = 16.8%. Exact match. The detector is a real derivation with both-direction validation (recall 9/9, precision 10/10) and reproduces at a pinned SHA (23 sites, 17 offenders, with outcome_tracker.py:57 and bias_detector.py:119 both flagged as C2 demands). The conflict_detector threshold claim you asked me to attack is SOUND and is NOT prose-only: test_conflict_detector_grades_a_strong_buy_at_the_STRICTER_threshold imports and executes the production _check_recommendation_alignment, and against a87add72^ ("Strong Buy" -> upper "STRONG BUY" fails the "STRONG_BUY" substring, falls to `elif "BUY"` with 6.0 < 5.5 false -> no conflict) it goes red -- my independent revert of that module killed it. So conflict_detector and bias_detector are properly guarded; the 0.0% intent figure with its threshold caveat is honest. C3, C6, C7 met; no set widened (V3/V4/V7 kill widening, and Accumulate/N-A/Hold negatives assert non-directional); no lost-trade or lost-P&L claim anywhere.

WHY THIS IS FAIL RATHER THAN CONDITIONAL. C8 does not merely imply the per-site mutation, it names it, and it names the consequence: "a guard whose mutant survives does not count." Four survive, and they are the two learning-path consumers plus reporting plus display -- outcome_tracker being precisely the site the step's own P1 argument rests on. C1 fails in the same place for the same reason. Both are cheap to close: revert each of the six sites in mutation_matrix_86_22.py as cells S1-S6, and add consumer-level assertions that drive the real entry points (outcome_tracker.evaluate_recommendation with a stubbed price fetch; memory.generate_reflection; the api/portfolio accuracy path; formatters._rec_color) so those cells die. Nothing about the product code needs to change.

ON THE FOUR ATTACKS YOU NAMED. (1) conflict_detector: sound, test-covered, not prose -- confirmed above. (2) The seventh consumer: excluding skill_optimizer:244 is honest in EFFECT (debate_consensus measures '' 487 / NULL 51 / 'HOLD' 4 / 'BUY' 1, no dialect defect) but the stated reason is provenance-wrong -- it reads a BQ column from the same table, not the Literal -- so the right correction is to re-word the ALLOWED entry to cite the measurement, not the type. The real undisclosed population is the four TS/TSX display sites. (3) 91/543 and 16.8% re-derived independently, exact. (4) The 82.12 registry matches the file and the suite is green.

DISCLOSURE. Read-only throughout: no Edit/Write, no redirects, no rm/mv/sed/commit/push. The mutation testing was done entirely in memory via sys.modules injection with the pre-fix source read from `git show`; I verified afterwards that both md5s the author's matrix records are unchanged and that `git status --short -- backend/ scripts/` is empty. No UI capture was taken and none is owed: the diff touches no frontend file and the step makes no UI claim, so qa.md 1c does not bind (the frontend finding above is a source-level grep, not a rendering claim). One minor inconsistency not worth its own violation: experiment_results section 7 says "After the change: 17" while live_check section G says "after 86.22: 16" -- reconcilable as pre- and post-self-fix, but the two artefacts use the same word "after" for different numbers.


---

# phase-86.22 -- EVALUATE (cycle 2)

**Verdict: PASS**  (`ok: True`, `harness_compliance_ok: True`, `certified_fallback: False`)

Fresh Q/A on CHANGED evidence (a87add72 -> 95398fae) via the Workflow rail,
run `wf_5fbffa92-924`. Transcribed VERBATIM. Main records the verdict, never authors it.

## reason

All 8 immutable criteria MET with evidence I reproduced myself, and both cycle-1 blockers are independently confirmed closed. C8: I re-ran the per-site revert axis with a DIFFERENT mechanism than the author's (sys.modules injection of `git show 4b7dab7b:<path>` sources, zero tree writes, vs their tree-write+restore) — CONTROL 58 passed/RC=0, then all SEVEN sites die: outcome_tracker 4 failed, memory 2, bias_detector 2, conflict_detector 1, api/portfolio 1, slack/formatters 1, skill_optimizer 1. Two independently-built mutant forms agreeing rules out a construction artifact. C1: `test_outcome_tracker_evaluate_recommendation_IS_DRIVEN_with_literal_BUY` calls the real `OutcomeTracker.evaluate_recommendation(recommendation="BUY")` on a +12% fixture and asserts `outcome["directionally_correct"] is True` plus that `save_outcome` fired; it is provably NOT a re-implementation because reverting only the module kills it — a test executing a copy would be unaffected. The cycle-1 tautology (`assert t is not None and Settings is not None`) is deleted, not repaired. Immutable verification command reproduced: 200 passed, 3097 deselected, 8.57s, no failures (author claimed 200/8.63s — matches). Criterion 2's NAMED false-negative check reproduced by me: the derivation against pre-fix 4b7dab7b flags `backend/services/outcome_tracker.py:57` and `backend/agents/bias_detector.py:119` explicitly (21 offenders pre-fix, 0 off-allow-list post-fix; recall 9/9, precision 10/10). Criterion 3 verified by direct execution: Accumulate/Overweight/BUYING/'NOT A BUY'/N-A (plus N/A, '', None, 'Strong Buy!', 'BUYOUT', int, dict) all canonicalise to None, buy=False, directional=False. Criterion 5 measured both ways: agent_memories rows=0 (no wrong reflection persisted) and outcome_tracking has no directionally_correct column, so nothing to backfill. Criterion 6 delta (AMD/PANW/MU SELL rows, False->True, 3/3) reproduces from the code. Adversarial item 3: skill_optimizer scoring is NOT broken — old-vs-new differential over the measured population ('' 487, NULL 51, HOLD 4, BUY 1) changes 0 rows; and the derivation itself flags skill_optimizer:244/247/252/255 pre-fix, so removing it from the allow-list was the method's verdict, not a judgement call. Adversarial item 4: the portfolio fixture genuinely discriminates — the losing 'Strong Buy' makes pre-fix read 1/1=100% and post-fix 1/2=50%, and the assertion rejects 100.0 (excluded row) AND 33.3 (HOLD wrongly admitted); my S4 revert kills it. Adversarial item 5: md5 of all 10 touched files is identical to `git show 95398fae:<path>` — the matrix left zero residue. Harness compliance clean (research 01:10:53 < contract 01:13:02 < code 01:14:26/01:15:00; gate_passed=true, 13 sources, recency+audit-class dry coverage; 0 `phase=86.22` result rows in harness_log and masterplan still `pending` = log-last intact; evidence CHANGED between a87add72 and 95398fae so this is the documented cycle-2 flow, not verdict-shopping; 0 prior CONDITIONALs so the 3rd-CONDITIONAL rule does not fire). No unintended production change: git status shows only untracked researcher-memory files and append-only audit/heartbeat JSONL. Code-review heuristics run across all 5 dimensions: no BLOCK or WARN — no secrets, no broad-except added, no perf_metrics bypass, kill_switch/stop-loss/trade-gate untouched (the money path is 86.20's still-dark flag), and the illusory-guard hunt found every remaining guard has a named, executed, killing mutation.

## violated_criteria

none

## checks_run

- `harness_compliance_audit_5_item`
- `research_gate_envelope`
- `contract_before_generate_mtime_chain`
- `log_last_and_masterplan_status`
- `no_verdict_shopping_evidence_delta`
- `third_conditional_counter_grep`
- `immutable_verification_command`
- `python_lint_ruff_F821_F401_F811_derived_scope`
- `ruff_delta_vs_prefix_rev_per_file`
- `backend_import_smoke_8_modules`
- `independent_per_site_revert_mutation_7_sites`
- `independent_vocabulary_axis_mutation_4_forms`
- `mutation_survivor_triage_wide_scope`
- `tree_integrity_md5_vs_commit`
- `derivation_recall_precision_validate`
- `derivation_against_prefix_rev_named_lines`
- `second_normaliser_repo_scan_independent`
- `allow_list_reason_audit`
- `non_widening_direct_execution`
- `skill_optimizer_behavioural_differential`
- `driven_vs_reimplemented_test_inspection`
- `portfolio_fixture_discrimination_check`
- `claim_audit_experiment_results_live_check`
- `code_review_heuristics`
- `evaluator_critique`

## notes (four NOTE-level items, PASS-with-flag)

Four NOTE-level items (PASS-with-flag; none degrade the verdict, all disclosed so an operator can overrule). N1 RUFF: over the DERIVED scope (`git diff --name-only 4b7dab7b 95398fae -- '*.py'` = 13 files, non-empty guard asserted, array-quoted to avoid the zsh word-split trap) `uvx ruff check --select F821,F401,F811` exits 1 with 7 F401s. I measured the per-file DELTA against the pre-fix rev via `git show <rev>:<f> | ruff --stdin-filename`: pre==post on EVERY file (bias_detector 1/1, conflict_detector 2/2, memory 1/1, outcome_tracker 3/3, skill_optimizer 0/0, portfolio 0/0, formatters 0/0, recommendation_vocab 0/0) — ZERO introduced by 86.22. qa.md 1a reads non-zero as FAIL; I am not applying that here because the gate exists to catch what the DIFF introduces and the introduced count is zero. Stating it plainly rather than burying it: the dead imports are `conflict_detector.py:11 Optional`, `outcome_tracker.py:11 timedelta / :12 Optional / :17 compute_benchmark_return`, and `json` in bias_detector/conflict_detector/memory — all in files this step touched, so a sweep would have been natural. Recommend a queued hygiene step. N2 SCOPE HONESTY — the one claim I could NOT reproduce: the "14 failed / 3265 passed, all pre-existing or midnight-rollover, none mine" full-suite classification rests on the author's capture alone; my own background full-suite run returned 0 bytes within the window, so I have no independent membership list. This does not block — the actual gate (the immutable -k command) is green 200/200 under my hand, all 8 changed backend modules import clean, and no failing-test membership diff was needed to close any criterion — but the 14-failure classification is unverified by me and should be read as the author's. N3 live_check section E carries the CYCLE-1 capture ("188 passed, 6.92s") while section K carries cycle 2 ("200 passed, 8.63s"). Append-only and labelled, so this is not a spliced-verbatim defect, but a reader landing on E gets a retired number — a one-line "superseded by K" pointer would close it. N4 The allow-list is now down to one substantive entry, `formatters.py::_signal_emoji`. I audited its reason rather than accepting it (this is the exact shape the cycle-1 Q/A disproved for skill_optimizer): it is corroborated independently by `backend/agents/mcp_servers/signals_server.py:159`, which documents the SIGNAL scale as "BUY"|"SELL"|"HOLD" — a separate closed scale that never carries a spaced or STRONG_ spelling — and the consumer is display-only (an emoji colour). I did not read the literal `action = signal.get(...)` assignment line; worst-case consequence is a wrong emoji, so non-blocking. ONE MUTANT OF MINE SURVIVED AND I AM REPORTING IT WITH ITS TRIAGE: replacing `canonical_recommendation` with a strip-less variant left the 86.22 file fully green (58 passed). Widening scope resolved it — the same mutant dies (4 failed) under the immutable -k selection, killed by phase-86.20's padded-input parity oracle at test_phase_86_20_portfolio_manager_recommendation_vocabulary.py:440,488. `.strip()` is 86.20 code, not a new 86.22 guard, so it is correctly out of criterion 7's scope and this is NOT a finding — but a survivor is only dismissible after the differential is run, so the run is recorded. Two behaviour changes outside the criterion-3 named set, both benign and inherited from 86.20's canonicaliser: whitespace-padded 'buy '/' BUY' now register as buy intent (they did not pre-fix), which is correct intent recovery, and 'BUYOUT' now falls to the neutral Slack colour instead of rendering as a buy (moves the safe way). ON THE FAIL->PASS REVERSAL: this is not sycophancy under rebuttal. The code materially changed between a87add72 and 95398fae (skill_optimizer migrated rather than defended, +248 test lines, the tautological "load-bearing behavioural assertion" deleted), and I did not accept the author's 18/18 matrix — I re-derived the seven per-site kills and four vocabulary mutants myself, under a different construction, before reversing.
