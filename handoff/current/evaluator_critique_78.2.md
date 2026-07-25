# Evaluator critique — phase-78.2

**EVALUATE phase.** Verdicts produced by the Layer-3 **Q/A** agent via the
first-class `.claude/workflows/qa-verdict.js` Workflow (structured output — the
verdict IS the captured return value). Main records verdicts; Main does not
author them.

**Two cycles are recorded below, in order.** Cycle 1 returned FAIL; Main fixed
the blockers, updated the handoff files, and spawned a FRESH Q/A on the changed
evidence, which is the documented cycle-2 flow (CLAUDE.md), not verdict-shopping
— cycle 2 independently confirmed the evidence had physically changed before
grading.

**Transcription gap, disclosed:** this file should have existed after cycle 1.
It did not — cycle 2 caught that `experiment_results_78.2.md` §6/§7 asserted a
verbatim transcription that was absent from disk (its finding 5). Both verdicts
are transcribed here now, unedited.

---

## Cycle 1 — FAIL
Run id: `wf_82381b7e-58c` · verdict **FAIL** · `ok: false` · `certified_fallback: false` · `harness_compliance_ok: true`
### violated_criteria (verbatim)
- `Every rail call logs the resolved model so post-hoc audit is possible`
- `illusory-guard`
- `consumer-contract-break`
- `money-path behaviour change not flag-gated dark`
- `hand-typed scope in test_every_direct_rail_call_site_passes_a_model`

### reason (verbatim)

Criteria 1, 2 and 4 are MET and independently reproduced; criterion 3 ("Every rail call logs the resolved model so post-hoc audit is possible") is NOT MET on two independent grounds: (3a) `_resolved_model`'s max(outputTokens) heuristic returns `claude-haiku-4-5` — the 12-token CLI helper — when fed the exact envelope the author measured in live_check §0 / contract §2 (opus-5[1m] in=2 out=4, haiku in=521 out=12), i.e. it names the wrong model in precisely the multi-model case the step exists to detect, so the requested/resolved MISMATCH warning would never fire and the log would still say haiku while opus-5[1m] did the work; its sole covering guard `test_resolved_model_prefers_the_worker_not_the_cli_helper` is labelled "the real one measured 2026-07-25" but alters that entry's outputTokens from 4 to 4000 (the one number the assertion turns on), and with the real value the production code fails while the M5 mutant passes — so the live_check §2-C2 "returns claude-opus-5" claim does not reproduce and the M5 matrix kill is an artifact of the fixture; and (3c) only `ClaudeCodeClient._log_cc_call` was changed, which is unreachable from the three sites the criterion names — B1/B2 log through the UNTOUCHED `autonomous_loop._log_claude_code_call:2287` which reads the envelope's top-level `model` key and never `modelUsage`, and E1 (`ticket_queue_processor.py:206`) writes no llm_call_log row at all (grep for log_llm_call in that file returns nothing), so "for every rail call" in experiment_results §3 is false for 3 of the 9 callers. Harness compliance is clean (5/5), the immutable command reproduces at 68 passed exit=0, and no unintended production file changed.

### violation_details (verbatim)

**1. Contradiction**

- *action*: Executed ClaudeCodeClient._resolved_model(envelope, 'claude-haiku-4-5') on the exact two-key envelope recorded in live_check_78.2.md section 0 and contract_78.2.md section 2 (claude-opus-5[1m]: in=2 out=4 costUSD=0.061417 ; claude-haiku-4-5-20251001: in=521 out=12 costUSD=0.000581)
- *state*: Returns 'claude-haiku-4-5' (the 12-token CLI helper), NOT 'claude-opus-5'. Order-independent: same result with reversed dict insertion order. Both artifacts identify claude-opus-5[1m] as the model that did the work, and its $0.061417 for in=2/out=4 is ~100x the haiku entry and unexplainable at list Opus rates without large cached context, corroborating that it is the worker. live_check_78.2.md section 2 C2, in a block presented as verbatim capture output, asserts the opposite: 'Fed the exact two-key envelope measured in section 0, _resolved_model returns claude-opus-5'. Consequence: requested=claude-haiku-4-5 and resolved=claude-haiku-4-5 are EQUAL, so the 'rail model MISMATCH' warning at claude_code_client.py:574-579 -- the author's stated compensating control for the llm_call_log.model semantic change -- does NOT fire, and llm_call_log records haiku while opus-5[1m] ran. That is the original defect, relabelled.
- *constraint*: Immutable criterion 3: 'Every rail call logs the resolved model so post-hoc audit is possible' + qa.md section 4b: a number in a verbatim-labelled artifact that does not reproduce is a Contradiction; prefer FAIL.

**2. Invalid_Precondition**

- *action*: Compared test_phase_75_llm_rail.py::test_resolved_model_prefers_the_worker_not_the_cli_helper's fixture against the measurement its docstring cites
- *state*: Docstring: 'Shape below is the real one measured 2026-07-25 from a live `claude -p`.' The fixture reproduces the measured inputTokens verbatim (521 and 2) but sets the claude-opus-5[1m] entry's outputTokens to 4000 where the measurement records 4 -- a 1000x alteration of precisely the number max(outputTokens) turns on. With the measured value the assertion fails. This is the SOLE guard for multi-entry discrimination (the other two _resolved_model tests use degenerate or single-entry envelopes), so it is sole coverage for a behavioural criterion. It also invalidates matrix row M5: executed on the real numbers, production max(outputTokens) yields claude-haiku-4-5 (wrong) while the M5 mutant min(outputTokens) yields claude-opus-5 (the author's own stated correct answer) -- the M5 'kill' is an artifact of the fabricated fixture, not evidence.
- *constraint*: code-review heuristic #17 illusory-guard [BLOCK when sole coverage for a behavioural criterion]; qa.md section 4c vacuity shape 5 (fixture that cannot represent the failure) and shape 11 (mis-attributed kill mechanism); qa.md section 4b (a verbatim capture must be regenerated, never edited).

**3. Overgeneralization**

- *action*: grep -n 'log_llm_call|_log_claude_code_call|log_cc_call' backend/services/ticket_queue_processor.py (exit 1, no match) and grep -n '_log_claude_code_call(' backend/services/autonomous_loop.py (hits :2467 :2474 :2547 :2554, definition at :2287, NOT in the diff)
- *state*: Two CC-rail loggers exist; the diff touched one. ClaudeCodeClient._log_cc_call (claude_code_client.py:558) was modified and logs the resolved model, but is reachable only via ClaudeCodeClient.generate_content, i.e. the 6 C-block overlays. B1 (lite trader) and B2 (lite risk judge) log via autonomous_loop._log_claude_code_call:2287, which this diff does not touch and which logs str((envelope or {}).get('model') or 'claude-code-cli') -- the envelope's TOP-LEVEL model key, never modelUsage, never canonicalModel, and the literal 'claude-code-cli' when that key is absent. E1 (ticket_queue_processor.py:206) writes NO llm_call_log row anywhere in the file. experiment_results_78.2.md section 3 criterion-3 row states '_log_cc_call derives the model from envelope["modelUsage"] for every rail call'; that is false for 3 of the 9 callers, including all three the criterion names by letter (B1, B2, E1). live_check_78.2.md section 2 PART C evidences only the one path that was fixed.
- *constraint*: Immutable criterion 3 ('EVERY rail call') + qa.md section 4 Contract completeness: every immutable criterion must map to covering evidence; a materially unaddressed criterion is a FAIL.

**4. Missing_Assumption**

- *action*: Enumerated consumers of llm_call_log.model after the requested->resolved semantic change and checked MODEL_PRICING membership in the venv
- *state*: WARN, dashboard-only, money path safe. spend.py::fetch_llm_spend:213-232 (the $25/day breaker) excludes CC-rail rows by provider != 'claude-code' AND agent NOT LIKE 'cc_rail:%' -- by AGENT, not by model -- so it is unaffected. sovereign_api.py::_fetch_llm_cost_by_provider:256-284 does NOT exclude cc_rail rows and prices MODEL_PRICING.get(model, _DEFAULT_PRICING); verified 'claude-opus-5', 'claude-opus-5[1m]' and dated ids like 'claude-haiku-4-5-20251001' are all ABSENT from MODEL_PRICING and fall to _DEFAULT_PRICING=(0.1,0.4), ~50x under Opus rates -- and the resolved-model path can now write exactly those strings (canonicalModel for a new model, or the raw map key on the canonicalModel-missing fallback). performance_api.py:73-79 and cost_budget_api.py:82-88 do not read model; no frontend consumer exists; no test pins CC-rail model to the requested value (test_phase_56_2_ops_fixes.py:154/:166 exercise the untouched autonomous_loop logger). The diff changed a telemetry column's semantics without enumerating consumers in the same diff.
- *constraint*: code-review heuristic #16/Dim-3 consumer-contract-break [WARN] -- behavioural-break subset: changed return-value spec / changed default behaviour shipped without every consumer grep-verified.

**5. Threshold_Not_Met**

- *action*: Judged attack (1): shipping a live model-tier downgrade (opus-5[1m] -> configured tier) inside a step scoped as 'thread a flag'; read masterplan 79.55 priority and live_check section 4
- *state*: WARN, not a blocker. The tier change is INHERENT to criterion 1 -- an explicit model cannot be forwarded without changing which model runs -- so it is not executor scope creep, and it is disclosed in four places (contract section 3, experiment_results section 4, live_check section 4, masterplan 79.55) with measured before/after. That is honest. Two gaps remain: (i) the project's standing norm is 'every behavior change fail-safe or DARK' (phase-69; 75.5.1 metric shipped DARK; 75.1 AUTH_ENFORCE_ALLOWLIST DARK), and a default-OFF rail_explicit_model_enabled would have satisfied all four criteria with zero live change; (ii) 79.55 is filed P2 while live_check section 4 states the change only takes effect on a backend restart -- so a restart for any unrelated owed action silently ships the downgrade on the lite trader, the lite risk judge and six signal overlays before the operator has answered. Also, the prose implies a uniform move to haiku, but B1/B2 take settings.gemini_model or 'claude-sonnet-4-6' (autonomous_loop.py:2392), so the TRADE DECISION itself re-tiers to whatever gemini_model holds; the :2396 claude-* guard runs before the rail branch, so no Gemini id can reach --model. Remedy: raise 79.55 to P0/P1 and mark it a restart blocker, or gate the change dark.
- *constraint*: Project norm 'every behavior change on a money path is fail-safe or DARK' (phase-69 / 75.5.1 / 75.1 precedent); a P2 operator decision does not gate a restart.

**6. Missing_Assumption**

- *action*: Judged attack (6): re-derived rail call sites repo-wide with my own AST walk over backend/**/*.py + scripts/**/*.py and compared against the test's hardcoded SITES list
- *state*: WARN, no live escape today. My derivation finds exactly 4 call sites, all with model_kwarg=True: claude_code_client.py:673, autonomous_loop.py:2453, autonomous_loop.py:2534, ticket_queue_processor.py:206. The guard's SITES list hardcodes two files, so it defends only against removal or reshaping WITHIN them; a new rail call site in a new module escapes entirely, and it already misses claude_code_client.py:673 (covered by a separate behavioural test). The 'total >= 3' threshold sits exactly at the current count, so it would still pass if one site were deleted and another added. One-line fix: replace SITES with (REPO/'backend').rglob('*.py') excluding tests.
- *constraint*: qa.md section 4b: 'Scopes must be DERIVED, not typed' -- a tool that reports success over a scope the author chose is not evidence.

### checks_run (verbatim)

- `harness_compliance_audit_5_item`
- `research_gate_envelope`
- `mtime_ordering_research_lt_contract_lt_code_lt_results`
- `log_last_check`
- `no_verdict_shopping_check`
- `immutable_verification_command`
- `syntax_ast_parse`
- `python_lint_gate_ruff_F821_F401_F811_scope_derived_from_git`
- `lint_pre_existing_vs_introduced_comparison_against_HEAD`
- `backend_runtime_smoke_imports`
- `scoped_regression_pytest_199_tests`
- `unintended_production_change_check`
- `independent_repo_wide_ast_derivation_of_call_sites`
- `criterion_1_behavioural_verification`
- `criterion_2_live_execution_of_spawn_real_agent`
- `criterion_2_guard_recall_test_assert_seen`
- `criterion_3_resolved_model_executed_on_real_measured_envelope`
- `criterion_3_logger_seam_coverage_audit`
- `criterion_4_in_memory_mutation_proof`
- `fixture_fidelity_audit_vs_cited_measurement`
- `mutation_matrix_m5_re_derivation`
- `claim_auditing_numeric_reproduction`
- `llm_call_log_consumer_enumeration`
- `model_pricing_membership_check`
- `guard_vacuity_check_section_4c`
- `code_review_heuristics`
- `adversarial_worst_of_n_lenses`

### notes (verbatim)

CYCLE 1 for 78.2 -- no prior evaluator_critique_78.2.md and zero prior result entries for this step-id in handoff/harness_log.md, so the 3rd-CONDITIONAL rule does not bind; FAIL is reached on the merits. retry_count=0 < max_retries=3, so certified_fallback=false.

HARNESS COMPLIANCE 5/5 CLEAN. (1) research_brief_78.2.md 13:15:59 < contract_78.2.md 13:27:49; envelope gate_passed=true, external_sources_read_in_full=6 (>=5), recency_scan_performed=true, urls_collected=25, internal_files_inspected=13. (2) contract 13:27:49 < earliest code edit 13:49:24. (3) experiment_results_78.2.md 13:52:48 and live_check_78.2.md 13:52:08 both present. (4) log-last honoured: no 'phase=78.2 result=' entry exists (only in-flight mentions inside the Cycle 164 block) and masterplan 78.2 status=pending. (5) no verdict shopping: first spawn on this step.

DETERMINISTIC RESULTS. Immutable command reproduced: 68 passed, 1 warning in 4.78s, exit=0 -- matches the claim. The '+8 tests' and '60 passed' baseline claims both reproduce (37 def test_ now in test_phase_75_llm_rail.py; 68-8=60). Scoped regression pytest backend/tests/ -k "rail or claude_code or cc_rail or ticket_queue or spend or observability or cost_budget" -> 199 passed, exit 0. All three changed backend modules import clean in the venv. git diff --name-only HEAD is exactly the 5 declared files plus two append-only hook audit JSONLs -- no unintended production change.

LINT GATE. ruff --select F821,F401,F811 over the git-derived 4-file scope exits 1 with 6 x F401, all in backend/services/ticket_queue_processor.py (subprocess, json, typing.List, pathlib.Path, TicketClassification, TicketsDB). Verified PRE-EXISTING: the identical 6 reproduce from `git show HEAD:backend/services/ticket_queue_processor.py | ruff --stdin-filename`. Zero new lint findings introduced by this diff, so the gate does not fail the step. Per feedback_queue_discovered_defects_in_masterplan these 6 warrant their own trivial masterplan step rather than a prose mention. SELF-DISCLOSURE: my first run of this gate hit qa.md vacuity shape 9 -- zsh does not word-split unquoted variables, so `uvx ruff check $FILES` linted one nonexistent newline-joined path and printed "All checks passed!" exit 0. Re-run through `git diff --name-only HEAD -- '*.py' | xargs` with a count guard, which is what produced the finding above.

ATTACKS ANSWERED IN THE AUTHOR'S FAVOUR. Attack (2), criterion 2: the justification for honouring rather than deleting agent_model_map holds -- it is a live per-agent policy at ticket_queue_processor.py:171-177 whose only consumer was the direct-SDK branch the away-ops rail flag bypasses. I drove the real _spawn_real_agent myself and observed model='claude-opus-4-8' reach the invocation for agent_id='main'. I recall-tested the 'assert seen' guard: with paper_use_claude_code_route=False the rail branch is never reached and seen=={} -- the guard FIRES. It is genuine, not a silent pass. Attack (4), fallback chain: CONFIRMED SAFE -- executed None, {}, {"modelUsage": None}, {"modelUsage": []}, {"modelUsage": "not-a-dict"}, all return the requested label; the bare except at :551-554 is correct inside a logging path. One residual untested NOTE: when no entry carries outputTokens every key ties at 0 and max returns the first entry in dict insertion order (reproduced: {'aaa','zzz'} -> 'aaa') -- silent and order-dependent. Attack (7), appending to test_phase_75_llm_rail.py: SOUND, no finding -- the immutable command names two files by path, so a new file would never be selected by the gate, i.e. it would be a guard that cannot fail at the point of enforcement; the phase-78.2 banner delimits the block cleanly. Only cosmetic: the module docstring still describes phase-75 scope. Criterion 4 verified by in-memory mutation without touching production files: dropping model=model_name at the B1 site and re-running the guard's own AST logic yields missing=['autonomous_loop.py:2453'] -> RED, with total unchanged at 2 in that file so the denominator is not what fires.

WHAT WOULD CLOSE THIS. (a) Replace max(outputTokens) with a discriminator that is correct on the measured envelope -- max(inputTokens), max(costUSD), or cache-token weight all select claude-opus-5[1m] there; whichever is chosen, pin it with a fixture carrying the UNALTERED measured numbers and regenerate live_check section 2 C2 from a fresh run. (b) Extend the resolved-model derivation to autonomous_loop._log_claude_code_call:2287 (B1/B2) and add an llm_call_log write to the ticket_queue_processor.py:206 rail branch (E1), or narrow the criterion-3 claim in experiment_results to the ClaudeCodeClient seam and disclose B1/B2/E1 as uncovered -- note the masterplan already tracks E1's missing instrumentation under the 78.3 spend-meter-blindness step, so cross-referencing it there is a legitimate close. (c) Re-run the mutation matrix after (a); M5 as recorded is invalid. (d) Raise 79.55 to P0/P1 and mark it a restart blocker, or gate the tier change behind a default-OFF flag. (e) Optional one-liners: derive SITES from rglob, and add the missing MODEL_PRICING rows or normalise [1m]/dated suffixes before logging.

Worst-of-N-LENSES: correctness lens FAIL (3a heuristic wrong on real data), does-it-reproduce lens FAIL (live_check section 2 C2 does not reproduce), scope-honesty lens CONDITIONAL (the tier downgrade is disclosed well and the 'what this does NOT prove' section is unusually candid, but the criterion-3 'every rail call' row overclaims coverage). verdict = min = FAIL. No UI claims in this step, so the section 1c live-capture gate does not apply and no browser tools were used; no frontend/** or qa.md paths in the diff, so the section 1b gate does not apply.

---

## Cycle 2 — CONDITIONAL
Run id: `wf_f7e8d7ea-251` · verdict **CONDITIONAL** · `ok: false` · `certified_fallback: false` · `harness_compliance_ok: false`
### violated_criteria (verbatim)
- `criterion_3_substitution_blind_spot_for_6_of_9_rail_callers`
- `criterion_3_every_rail_call_E1_failure_path_writes_no_row`
- `illusory-guard: test_all_three_rail_loggers_resolve_the_model is a structural AST scan and is sole coverage for the new E1 write`
- `money-path consumer claim cites the wrong exclusion mechanism for 2 of 3 seams`
- `Missing_Assumption: handoff/current/evaluator_critique_78.2.md claimed in artifacts but absent`

### reason (verbatim)

Cycle 2 is genuinely changed evidence (not verdict-shopping) and all three cycle-1 blockers were independently confirmed remediated: B1's resolver is rewritten and behaviourally correct on the measured envelope, B2's fixture is real (I re-derived BOTH costUSD values exactly from published rates: haiku 523*1e-6+12*5e-6=0.000583 and opus 2*5e-6+4*25e-6+17140*0.5e-6=0.00868, matching the constant to the last digit), and all three rail loggers now resolve+write (I drove each seam end-to-end with a spy and saw the row). Criteria 1, 2 and 4 are MET; the verification command is exit=0 / 71 passed and adjacent rail suites are 102 passed. Criterion 3 is MET-WITH-GAPS, so PASS is withheld: (a) I REPRODUCED a substitution blind spot at the production seam -- resolve_rail_model branch (1) returns claude-haiku-4-5 while opus-5 did 0.00868 of the work and the mismatch WARNING never fires, because the CLI's internal helper IS a haiku-4-5 snapshot; that is the pre-78.2 defect class ("log says haiku, opus ran") surviving for the six haiku-tier callers, i.e. 6 of the 9, and Main's stated defence ("post-78.2 that cannot happen because we always pass --model") is invalid since passing --model does not make the worker that model -- substitution is the event being detected; (b) E1 logs only its success path, so a failed ticket rail call writes no row while the other two seams write ok=False; (c) criterion 3's wiring is guarded ONLY by a structural AST scan that I demonstrated passes when the calls sit in dead code, and E1's brand-new production BQ write has no behavioural guard at all; (d) the money-path safety claim now enshrined in an inline code comment ("spend.py excludes rail rows by AGENT") is the wrong mechanism for 2 of the 3 seams. Also: handoff/current/evaluator_critique_78.2.md is asserted in experiment_results sections 6 and 7 but does not exist on disk. All findings are named and fixable; 0 prior CONDITIONALs for this step-id, so the 3rd-CONDITIONAL escalation does not apply.

### violation_details (verbatim)

**1. Unjustified_Inference**

- *action*: resolve_rail_model(envelope, requested) branch 1 -- `if requested and requested in named: return requested` (backend/agents/claude_code_client.py:266-268), reached from ClaudeCodeClient._log_cc_call:618
- *state*: SEVERITY=WARN (blocks PASS, does not force FAIL). REPRODUCED end-to-end through the production logger, not reasoned: envelope modelUsage = {claude-haiku-4-5-20251001 (canonicalModel claude-haiku-4-5, costUSD 0.000583), claude-sonnet-5 (costUSD 0.00868)}, requested='claude-haiku-4-5' -> row model = 'claude-haiku-4-5', mismatch warning fires = False. Branch 2 on the same envelope correctly names claude-sonnet-5. The CLI's internal helper is MEASURED TWICE IN THIS STEP'S OWN ARTIFACTS as a claude-haiku-4-5 snapshot (live_check sections 0 and 2b), and 6 of the 9 rail callers request exactly 'claude-haiku-4-5' (ClaudeCodeClient model_name for the six 78.1 C-block services), so for those callers `requested in named` is ALWAYS true and a substitution can never be reported.
- *constraint*: Criterion 3: 'Every rail call logs the resolved model so post-hoc audit is possible' -- and the design rationale the executor states for it (experiment_results 1(c): 'once --model is correct, a mismatch means the CLI substituted a model on us -- including Anthropic's own automatic safety-classifier fallback -- which is exactly the event worth seeing'). The premise 'we asked for it and it appears in the map, therefore it ran' conflates co-occurrence with authorship; presence of the requested id as the helper is not evidence it was the worker. NAMED FIX: compute the dominant-by-cost entry first, accept the exact match only when the requested id IS the dominant entry, else report the dominant and warn.

**2. Threshold_Not_Met**

- *action*: backend/services/ticket_queue_processor.py:206 claude_code_invoke(...) raising ClaudeCodeError -> propagates to the method-level `except Exception as e:` at :332, which logs and re-raises
- *state*: SEVERITY=WARN. The new E1 logger (:230-254) sits INSIDE the success path, after `response_text = extract_result_text(envelope)`. A failed ticket rail call therefore writes NO llm_call_log row. The other two seams both meter failures: autonomous_loop._log_claude_code_call(None, ..., ok=False) at :2491/:2573, and ClaudeCodeClient._log_cc_call(None, ..., ok=False) at :754.
- *constraint*: Criterion 3 says EVERY rail call. A rail call that failed is precisely the call an audit needs to see, and the parity the other two seams already implement is the in-repo standard.

**3. Circular_Reasoning**

- *action*: test_all_three_rail_loggers_resolve_the_model (backend/tests/test_phase_75_llm_rail.py:810-833) -- ast.parse each module, collect the set of called names, assert 'resolve_rail_model' in called and 'log_llm_call' in called
- *state*: SEVERITY=WARN. EXECUTED the guard's own logic against a synthetic module whose real logger neither resolves nor logs, with both calls parked in a `_never_called()` function: guard PASSES. It is not comment-satisfiable (that part of the design is sound), but it cannot observe wiring. It is the SOLE guard for the existence of E1's brand-new production BQ write (no other test in backend/tests touches that path -- derived by grep), and the sole guard tying resolution to autonomous_loop._log_claude_code_call (the pre-existing behavioural tests at test_phase_56_2_ops_fixes.py:142-176 use envelopes with NO modelUsage, so they pass with or without resolution). No test anywhere asserts that a modelUsage-bearing envelope produces a row carrying the RESOLVED id -- the resolver is tested in isolation, the wiring only structurally. Consequently the M7 and M8 matrix kills in live_check section 3 (each '1 failed') are killed by this scan, and the matrix does not name which assertion killed.
- *constraint*: qa.md 4c vacuity shapes #1/#2 (source-scan asserting runtime behaviour it cannot observe / defeated by moving the scanned text) and #11 (mis-attributed kill mechanism -- name WHICH assertion killed). NAMED FIX: one spy test per seam asserting log_llm_call receives the resolved id when modelUsage names a different model than requested -- exactly the three checks I had to run by hand to verify the code works.

**4. Contradiction**

- *action*: The W4 remediation -- the consumer enumeration moved inline to backend/agents/claude_code_client.py:608-618 -- and the same claim in experiment_results 2b: 'spend.py::fetch_llm_spend (the $25/day breaker) excludes rail rows by AGENT, so it is unaffected'
- *state*: SEVERITY=WARN. Reproduced the SQL at backend/services/observability/spend.py:225-231. It carries TWO independent exclusions: `AND provider != 'claude-code'` (:228) and the agent clause (:229-230). The agent clause protects only the seams that write agent='cc_rail...' -- ClaudeCodeClient (provider=anthropic, agent=cc_rail:<role>) and the new E1 row (agent=cc_rail:ticket_<id>). It does NOT cover the B1/B2 seam THIS DIFF ALSO CHANGED: autonomous_loop._log_claude_code_call writes provider='claude-code', agent='lite_trader'/'lite_risk_judge', which the agent clause does not match; those rows are excluded solely by the provider clause, which the artifact and the new comment never mention. The safety CONCLUSION (breaker unaffected) is correct today; the cited MECHANISM is wrong for 2 of the 3 seams.
- *constraint*: qa.md 4b -- a money-path claim must be reproduced, and its stated mechanism must be the one that actually holds. A comment placed at the change site specifically to inform future readers must not enshrine a reason that does not apply to the seam the same diff modified: a later normalisation of provider 'claude-code' -> 'anthropic' would silently admit B1/B2 rows carrying resolved ids absent from MODEL_PRICING into the $25/day breaker at _DEFAULT_PRICING (0.1, 0.4).

**5. Missing_Assumption**

- *action*: experiment_results_78.2.md section 6 lists 'handoff/current/evaluator_critique_78.2.md (Q/A verdict, transcribed verbatim)' and section 7 states 'The verdict is transcribed verbatim in evaluator_critique_78.2.md'
- *state*: SEVERITY=WARN. `ls handoff/current/evaluator_critique_78.2.md` -> No such file or directory. The directory holds evaluator_critique_{75.5.12,76.9.2,76.9.3,78.0,78.16}.md and a stale rolling evaluator_critique.md from 2026-07-24 14:11; a grep for the cycle-1 verdict's own phrase 'the original defect, relabelled' across handoff/ matches only experiment_results_78.2.md. The cycle-1 FAIL exists on disk only as Main's paraphrase of it.
- *constraint*: CLAUDE.md five-file protocol (evaluator_critique.md is a NON-SKIPPABLE artifact) plus the verbatim-transcription rule that keeps the no-self-eval guarantee auditable. A past-tense claim in a handoff artifact that does not reproduce is itself a 4b finding. Remediable by Main transcribing the cycle-1 verdict and this one before the status flip.

### checks_run (verbatim)

- `harness_compliance_audit_5_items`
- `mtime_ordering_research_lt_contract_lt_code_lt_results`
- `evidence_changed_since_cycle_1_verdict_shopping_check`
- `verification_command_exit_code`
- `python_lint_gate_ruff_F821_F401_F811_derived_scope`
- `ruff_preexistence_check_vs_git_show_HEAD`
- `backend_runtime_smoke_imports`
- `adjacent_rail_test_suites`
- `derived_call_site_census_grep`
- `fourth_rail_seam_hunt`
- `adversarial_attack_on_resolve_rail_model_executed`
- `end_to_end_behavioral_verification_all_three_log_seams`
- `fixture_arithmetic_re_derivation_from_published_rates`
- `ast_guard_vacuity_demonstration_executed`
- `mutation_M1_kill_mechanism_verification`
- `money_path_consumer_sql_reproduction`
- `research_gate_envelope`
- `masterplan_immutability_check`
- `third_conditional_counter`
- `code_review_heuristics`
- `contract_completeness_criterion_mapping`

### notes (verbatim)

HARNESS AUDIT (5/5 items pass): research_brief_78.2.md 13:15:59 < contract_78.2.md 13:27:49 < code 14:09-14:10 < experiment_results 14:13:44; gate_passed=true, external_sources_read_in_full=6 (floor 5), urls_collected=25, recency scan present as section B; contract section 2 cites researcher findings R1-R8. Log-last holds: grep -E "phase=78\.2 result=" on harness_log.md returns 0 lines and masterplan 78.2 status is still "pending". NOT verdict-shopping: all six cycle-1 remediations are physically present in the working tree (files rewritten 14:09-14:13 today). harness_compliance_ok is false ONLY because the cycle-1 verdict was never written to evaluator_critique_78.2.md while the artifact asserts it was.

WHAT I VERIFIED INDEPENDENTLY, NOT FROM MAIN'S ACCOUNT.
B1 (resolver): rewritten as module-level resolve_rail_model at claude_code_client.py:215. Behaviourally correct on the measured envelope and order-independent. Branch (1) is NOT sound -- see violation 1, reproduced end-to-end, and Main's stated defence is falsified. Two further probes: canonicalModel collision collapses two snapshots of one family into a single dict key; and under branch 2 a long-prompt helper can outcost a tiny worker turn and be named (that false-positive risk is presumably why branch 1 exists, but the tradeoff is undisclosed and the artifact calls branch 1 "exact, not heuristic" -- it is exact about map-membership, not about authorship).
B2 (fabricated fixture): genuinely fixed. REAL_TWO_MODEL_ENVELOPE now carries outputTokens=4 for the opus entry. I did not take "measured" on trust -- I re-derived both cost figures from published rates and they match exactly (haiku 523*1e-6+12*5e-6=0.000583; opus 2*5e-6+4*25e-6+17140*0.5e-6=0.00868), and live_check section 0's independent capture also reproduces (521*1e-6+12*5e-6=0.000581). I swept the WHOLE test diff for the same pattern: the only other fixtures are the generic success stub in _run_capture, fake_invoke, and env/env2 in the fallback test which are explicitly labelled synthetic ("No costUSD anywhere"). No other fixture is adjusted to make an assertion pass. TRACEABILITY NOTE: live_check section 2b claims the constant is "verbatim from a live claude -p --model opus" but never prints that capture's raw numbers, so it is not cross-checkable from the artifact itself -- the arithmetic above is my compensating control, and printing the raw envelope would close it.
B3 (three loggers): confirmed all three changed, and I hunted the fourth seam independently rather than accepting the count. Derived census: exactly 4 production claude_code_invoke call sites in backend/ (claude_code_client.py:716, ticket_queue_processor.py:206, autonomous_loop.py:2469 and :2552); the other module hits are phase-78.1 comments. No other spawn of the claude binary exists in backend/ (grep for --print / --output-format / _resolve_claude_binary outside claude_code_client returns nothing); claude_code_health_probe runs `claude auth status` only -- no model, no tokens, correctly excluded; openclaw_client is an HTTP gateway, not the CC rail. So THREE loggers is right. test_all_three_rail_loggers_resolve_the_model is NOT satisfiable by a comment (it is AST-based, comments are not Call nodes) -- but it IS satisfiable by dead code, which I demonstrated by execution; see violation 3.
W4: consumer list is inline at claude_code_client.py:608-618 -- but only at that seam, and its stated mechanism is wrong for the other two (violation 4).
W5: verified in the masterplan diff -- 79.55 is "priority": "P0" and its name opens with [OPERATOR ACTION][RESTART BLOCKER -- answer BEFORE the next backend restart]. Its embedded technical claim also checks out: the guard `if not model_name.startswith("claude-")` genuinely runs BEFORE the rail branch, so no Gemini id can reach --model. The cited line numbers :2392/:2396 are pre-edit coordinates and resolve correctly to post-edit :2408/:2412 under the diff's +16 offset -- accurate, not drifted.
W6: test_every_direct_rail_call_site_passes_a_model now walks (REPO/"backend").rglob("*.py") -- derived, not hand-typed -- with a >=4 denominator floor. My independent grep census agrees at 4. I executed the guard's logic against an M1-shaped mutant (model= dropped from the asyncio.to_thread call) and against a comment-only mutant: it goes RED on both, so criterion 4's kill mechanism is genuinely behavioural. Residual: the walk covers backend/ only, so a rail call site added under scripts/ would escape (scripts/mas_harness/run_cycle.sh already passes --model, so no live gap today).

E1 SCOPE QUESTION (answered): the new llm_call_log row is WITHIN this step's boundary, not scope creep. The boundary is "claude_code_client.py callers + tests"; ticket_queue_processor.py is a caller, and criterion 3 ("EVERY rail call logs the resolved model") is unsatisfiable at E1 by construction without adding a row -- the executor's own framing is correct. What is NOT adequately handled is that this is a NEW production BQ write with zero behavioural coverage and zero live observation (violations 2 and 3), and that is the part to fix rather than the scope.

LINT GATE, HONESTLY REPORTED: my first invocation hit qa.md vacuity shape #9 -- an unquoted newline-joined variable made ruff lint ZERO files and print "All checks passed!" at exit 0. I caught it and re-ran with explicit NUL-delimited splitting over `git diff --name-only HEAD -- '*.py'` (4 files). Real result: exit=1, 6 F401 findings, ALL in backend/services/ticket_queue_processor.py (subprocess, json, typing.List, pathlib.Path, TicketClassification, TicketsDB). I tested pre-existence rather than assuming it: `git show HEAD:backend/services/ticket_queue_processor.py | uvx ruff check --stdin-filename ... -` reproduces the identical 6. So this diff introduces zero new lint findings; the gate's non-zero exit is not attributable to 78.2. Worth its own masterplan step per the queue-discovered-defects rule.

TESTS: immutable verification command exit=0, "71 passed, 1 warning in 6.08s" (baseline claim of 60 is consistent with the +11 tests). Adjacent rail suites (test_phase_60_4_observability, test_phase_66_1_rail_guard, test_phase_59_1_fable_adoption + the two above) 102 passed, exit=0. Runtime smoke: all three changed backend modules import cleanly in the venv. No unintended production change: only the 4 files the contract scopes plus a 16-line pure ADDITION to masterplan.json creating 79.55 -- 0 deletions, and 78.2's own immutable criteria are untouched (the diff contains no occurrence of "78.2" as a key).

WORST-OF-N-LENSES (P1 money-adjacent step): correctness lens = CONDITIONAL (violations 1, 2); does-it-reproduce lens = PASS (every deterministic check reproduced, including the numbers behind the fixture); scope-honesty lens = CONDITIONAL (violation 4's mis-attributed mechanism and violation 5's non-existent artifact, against genuinely strong disclosure elsewhere -- the tier downgrade, the un-restarted backend, the unobserved E1 row and the MODEL_PRICING gap are all stated plainly and the cycle-1 FAIL is recorded rather than buried). Worst-of = CONDITIONAL.

TO REACH PASS: (1) make branch 1 conditional on the requested entry being the dominant one, or drop it and warn on the false-positive case, plus a test built from the ATTACK-A envelope; (2) meter the E1 failure path with an ok=False row like the other two seams; (3) add one spy test per seam asserting the RESOLVED id reaches log_llm_call on a modelUsage-bearing envelope, and re-run M7/M8 naming which assertion kills; (4) correct the inline comment and section 2b to cite `provider != 'claude-code'` as the exclusion that actually protects B1/B2; (5) transcribe the cycle-1 verdict and this one to handoff/current/evaluator_critique_78.2.md before the status flip.

---

## Cycle 3 — CONDITIONAL
Run id: `wf_7b0c01dd-755` · verdict **CONDITIONAL** · `ok: false` · `certified_fallback: false` · `harness_compliance_ok: true`

**All four immutable criteria MET**, verified by the Q/A through direct execution of the
production loggers. PASS withheld on two NON-criterion items only.

### violated_criteria (verbatim)
- `qa.md §4b verbatim-capture reproducibility — experiment_results §2 'Verbatim verification output' says 71 passed / '+11 tests'; reality is 72 passed / 12 added`
- `qa.md §1a python lint gate — uvx ruff --select F821,F401,F811 over the derived scope exits 1 (6 pre-existing F401 in backend/services/ticket_queue_processor.py, zero introduced by this diff, no lint run reported by the author)`

### reason (verbatim)

All FOUR immutable criteria are MET and were reproduced by me independently of experiment_results §8, and every cycle-2 remediation holds up under direct execution: (1) the `requested in map` short-circuit is genuinely gone — I drove the cycle-2 reproduction through the three PRODUCTION loggers (not the tests) and `resolve_rail_model(REAL_ENV,'claude-haiku-4-5')` now returns `claude-opus-5`, with seam1 `[ClaudeCodeClient] rail model MISMATCH: requested=claude-haiku-4-5 resolved=claude-opus-5`, seam2 `rail model MISMATCH (lite path): ...`, seam3 `rail model MISMATCH (ticket queue): requested=claude-sonnet-4-6 resolved=claude-opus-5` all firing and all three rows carrying `model=claude-opus-5`; (2) the E1 failure path writes `ok=False model=claude-opus-4-8 agent=cc_rail:ticket_main`, and `_meter_rail` cannot swallow the original — with `log_llm_call` forced to raise ValueError the propagated exception still carries the original text `rail down -- ORIGINAL` (the outer `Agent main failed:` re-wrap is pre-existing at `_spawn_real_agent`, not introduced here); (3) the guards are behavioural, not illusory — I confirmed the monkeypatch target is the one production resolves at call time by using the identical mechanism myself (all three seams do a function-local `from backend.services.observability.api_call_log import log_llm_call`, so patching the module attribute IS the live path) and captured real rows from each; (4) the money-path mechanism is correct against the SQL I read at spend.py:225-231 — `provider != 'claude-code'` is what protects B1/B2 (provider='claude-code', agent='lite_trader'), the agent clause `(agent != 'cc_rail' AND agent NOT LIKE 'cc_rail:%')` protects seam 1 and correctly excludes the NEW E1 row (`'cc_rail:ticket_main' LIKE 'cc_rail:%'` → TRUE → excluded); (5) I independently DERIVED the call-site scope by walking every backend/**/*.py AST and found exactly 4 production `claude_code_invoke` sites (claude_code_client.py:738, autonomous_loop.py:2469, autonomous_loop.py:2552, ticket_queue_processor.py:244), ALL passing `model=`, matching the guard's `>=4` floor exactly. Verification command reproduces `72 passed` exit=0; runtime smoke imports all three changed modules clean. I withhold PASS on TWO fixable, reproduced, non-criterion items: (A) experiment_results §2, a block titled "Verbatim verification output", records `71 passed, 1 warning in 4.92s` — it does NOT reproduce (actual `72 passed`, `--collect-only` = 72 collected), and §1's table says "+11 tests" where `git diff | grep -cE '^\+def test_'` = 12; the author's own live_check §3 carries the correct `72 passed`, so the two artifacts disagree and the load-bearing one is stale — the same non-reproducing-claim family for a third consecutive cycle (qa.md §4b); (B) the REQUIRED qa.md §1a lint gate over the DERIVED scope (`git diff --name-only HEAD -- '*.py'`, 4 files) exits 1 with 6 F401s, all in backend/services/ticket_queue_processor.py — the author reported only `ast.parse` and ran no lint at all. I proved zero regression: HEAD's copy of that file yields the identical 6 and the other three files are clean at HEAD and now, so 78.2 introduced no lint finding. No immutable criterion is missed, no unintended backend change exists (4 .py files, matching the boundary), and this is the 2nd CONDITIONAL for 78.2 (harness_log has 0 `phase=78.2` entries), so the 3rd-CONDITIONAL auto-FAIL rule does not fire.

### violation_details (verbatim)

**1. Invalid_Precondition**

- *action*: Compared experiment_results_78.2.md §2 (block titled 'Verbatim verification output') against a fresh run of the immutable command, plus `pytest --collect-only -q` and `git diff backend/tests/test_phase_75_llm_rail.py | grep -cE '^\+def test_'`
- *state*: Artifact §2 states `71 passed, 1 warning in 4.92s`; reproduced output is `72 passed, 1 warning in 5.80s` (re-run: 6.14s) and `--collect-only` reports `72 tests collected`. §1 file table states `+11 tests`; the derived count of added test functions is 12. The author's own live_check_78.2.md §3 baseline and §3 closing line both say `72 passed`, and were written at 14:36:52 — BEFORE experiment_results at 14:37:24 — so the stale block is in the later file. Direction is benign (understates by one) and exit code is 0.
- *constraint*: qa.md §4b: 'A verbatim capture must be regenerated, never edited... Prefer FAIL when a number in a verbatim artifact does not reproduce.' §2 is the single block the harness relies on for the immutable verification command; it must be regenerated from the final tree, and '+11 tests' corrected to 12.

**2. Threshold_Not_Met**

- *action*: FILES=$(git diff --name-only HEAD -- '*.py'); non-empty guard passed (4 files); uvx ruff check --select F821,F401,F811 $FILES
- *state*: exit=1, `Found 6 errors.` — all in backend/services/ticket_queue_processor.py: F401 `subprocess` (:15:8), `json` (:16:8), `typing.List` (:17:41), `pathlib.Path` (:19:21), `backend.db.tickets_db.TicketClassification` (:22:35), `backend.db.tickets_db.TicketsDB` (:22:57). PRE-EXISTENCE PROVEN: `git show HEAD:backend/services/ticket_queue_processor.py | uvx ruff check --select F821,F401,F811 -` also yields `Found 6 errors.`; the other three changed files return `All checks passed!` both at HEAD and in the worktree. So the diff introduced ZERO lint findings. Zero F821 and zero F811 anywhere. The author reported only `ast.parse` clean (3/3) and no lint run at all.
- *constraint*: qa.md §1a: the lint gate is REQUIRED when the diff touches any *.py and 'Non-zero exit = FAIL'. Vacuity shape 10 exists precisely because pre-existing F401s in touched files were hidden three separate times. Remedy: either delete the 6 dead imports in this cycle (a one-line-each change inside a file the diff already touches) or, per feedback_queue_discovered_defects_in_masterplan, queue them as their OWN masterplan step rather than a prose note — then re-run the gate to exit 0.

### checks_run (verbatim)

- `harness_compliance_audit_5_items`
- `immutable_verification_command (72 passed, exit=0)`
- `pytest_collect_only (72 collected)`
- `python_lint_gate_derived_scope (uvx ruff F821,F401,F811, exit=1, pre-existence proven against HEAD)`
- `backend_runtime_smoke (import claude_code_client + autonomous_loop + ticket_queue_processor, signature check model default=None)`
- `frontend_typecheck (npx tsc --noEmit, exit=0)`
- `cycle2_blind_spot_reproduction_through_production_loggers (all 3 seams + warning capture)`
- `e1_failure_path_execution (ok=False row + exception-transparency under a raising log_llm_call)`
- `independent_AST_derivation_of_all_claude_code_invoke_sites`
- `money_path_SQL_read_verbatim (spend.py:225-231, 3 seams traced)`
- `guard_vacuity_check_4c (monkeypatch-target liveness, spy behavioural not scan)`
- `claim_audit_4b (71-vs-72, +11-vs-12, cost arithmetic re-derived from the measured envelope)`
- `false_positive_reachability_quantification`
- `immutable_criteria_unchanged_vs_HEAD (jq)`
- `git_status_scope_check (4 .py files, no unintended backend change)`
- `evaluator_critique_transcription_completeness`
- `3rd_conditional_counter (0 phase=78.2 entries in harness_log)`
- `code_review_heuristics`

### notes (verbatim)

HARNESS COMPLIANCE 5/5 CLEAN. (1) Research-gate-before-contract: research_brief_78.2.md 13:15:59 < contract_78.2.md 13:27:49; envelope gate_passed=true, 6 sources read in full (>=5 floor), 25 URLs (>=10), recency scan performed, 13 internal files; contract §2 cites it per-finding R1-R8, not as a bare reference. (2) Contract-before-generate: contract 13:27:49 < first code edit 14:34:09. (3) experiment_results present with a criterion-by-criterion table. (4) Log-last: `grep -E 'phase=78\.2' handoff/harness_log.md` returns 0 matches and masterplan 78.2 is still `status: pending` — correct at EVALUATE time. (5) No verdict-shopping: the evidence PHYSICALLY changed between cycle 2 and cycle 3 — I read the current source and the membership short-circuit is gone (and I ran it: haiku -> claude-opus-5), `_meter_rail` is new, the AST-scan test is replaced by behavioural spies, and evaluator_critique_78.2.md now exists where cycle 2 found it absent. Immutable criteria byte-identical to HEAD (`git show HEAD:.claude/masterplan.json | jq` on id 78.2); the masterplan diff only ADDS step 79.55.

THE FALSE-POSITIVE YOU ASKED ME TO ATTACK — QUANTIFIED, AND THE DISCLOSURE IS ADEQUATE. First, the envelope is internally consistent and not fabricated: the helper's costUSD 0.000583 reproduces EXACTLY from its own tokens at haiku $1/$5 (523e-6 + 12*5e-6), and the opus entry's 0.00868 reproduces EXACTLY at $5/$25 with cache-read $0.50/Mtok (2*5e-6 + 4*25e-6 + 17140*0.5e-6) — so cost-as-dominance is measuring what it claims to. Break-even for the helper to out-cost the worker: worker=claude-sonnet-4-6 (B1/B2) loses only if its input < 194 tok AND output < 39 tok; worker=claude-opus-4-8 (E1) only if input < 117 tok AND output < 23 tok. Both seams' system prompts alone exceed those thresholds, and B1 runs max_tokens=200 against a JSON-format instruction — so the false positive is NOT reachable with real numbers on B1/B2/E1. For the six haiku callers it is structurally unreachable: worker and helper share `canonicalModel: claude-haiku-4-5` and collapse into a single map entry (I ran it: returns `claude-haiku-4-5`, no warning). Verdict on the disclosure: adequate, no guard needed.

BUT A SHARPER, UNDISCLOSED CONSEQUENCE OF THE SAME UNDOCUMENTED FIELD (NOTE, not blocking). The collapse that protects the six is entirely load-bearing on `canonicalModel` — the field they correctly flag as undocumented. I ran the same envelope with `canonicalModel` removed: `resolve_rail_model` returns the DATED SNAPSHOT `claude-haiku-4-5-20251001`, so if Anthropic drops that field, all six haiku callers get (a) a spurious MISMATCH WARNING on every rail call and (b) a dated id written into llm_call_log.model that is absent from MODEL_PRICING, compounding the gap already queued as 78.7. `test_resolved_model_survives_a_missing_canonical_model_field` only proves the fallback does not raise; it does not cover the same-family case. Cheap fix: treat a resolved key that has the requested id as a prefix as a match. Blast radius is observability-only on a flat-fee rail that spend.py excludes — hence NOTE, and worth queueing rather than fixing under this step.

SCOPE JUDGEMENT YOU ASKED FOR — the E1 llm_call_log row is IN BOUNDARY, not creep. ticket_queue_processor.py is a claude_code_client.py caller (the stated boundary), and criterion 3 says EVERY rail call logs the resolved model; E1 wrote no row at all, so the criterion cannot be satisfied without adding a writer there. Metering the FAILURE path likewise follows from "every rail call" and matches what the other two seams already did. Adding step 79.55 rather than deciding the tier is the correct disposition per feedback_queue_discovered_defects_in_masterplan.

OTHER NOTES (none blocking). (a) The three-seam test asserts sequentially inside ONE function, so a seam-1 regression masks seams 2-3; per-seam attribution still works for single-seam mutations (M7/M8/M10/M11 each produced 1-2 failures) and each assert names its seam, so this is diagnosis ergonomics, not vacuity — parametrizing over the seams would remove the masking. (b) E1's `try` wraps only `claude_code_invoke`; `extract_result_text` runs outside it, so a CLI call that succeeds but fails extraction is unmetered — narrow (invoke already raises on non-success subtypes) and it mirrors seam 2's existing shape. (c) The inline comment at claude_code_client.py renders the agent exclusion as `agent NOT LIKE 'cc_rail%'`, dropping the colon and the `!= 'cc_rail'` half; semantically equivalent for these agent values but not the verbatim SQL — the cited line range spend.py:228-230 is exact. (d) I could NOT re-run the file-edit mutation matrix: the `.claude/hooks/qa-write-guard.sh` PreToolUse hook correctly blocks the evaluator from writing anywhere outside .claude/agent-memory/qa/, so I substituted runtime-equivalent evidence — my own AST walk demonstrably discriminates present/absent `model=` (4 sites True, 9 test sites False, so the detector is live), and my independent drive of all three production seams reproduces exactly the assertions M7/M8/M10/M11 target. Final tree verified un-mutated: 72 passed, `git diff --stat -- backend/` = 4 files. (e) evaluator_critique_78.2.md now contains BOTH prior verdicts with run ids wf_82381b7e-58c / wf_f7e8d7ea-251 and every schema field rendered under a `(verbatim)` heading (violated_criteria, reason, violation_details as action/state/constraint triples, checks_run, notes) plus an explicit disclosure of the cycle-1 transcription gap. It is a faithful field-level transcription reformatted from JSON to markdown; I cannot byte-compare it against the original workflow returns (I do not hold them), so I certify completeness and absence of paraphrase-drift, not byte identity. (f) NOT ATTRIBUTABLE TO 78.2 BUT IT WILL RIDE THE COMMIT: the working tree also carries modified `frontend/tsconfig.json` and `frontend/next-env.d.ts` (mtime 14:20:11 — from the :3100 UI-audit run, 14 minutes BEFORE the 78.2 edits) repointing type roots at the UNTRACKED `frontend/.next-audit-3100/`, plus 17 untracked `audit-*.png` at repo root. `npx tsc --noEmit` exits 0 locally only because that untracked dir exists. The auto-commit hook does `git add -A` on the status flip, so these would be swept into 78.2's commit and would break a fresh checkout's typecheck. Main should restore those two files and deal with the PNGs before flipping status. (g) No live UI claim in this step, so §1c does not bind; §1d runtime smoke ran and is clean; §1b frontend checks ran anyway because the tree touches frontend/** and tsc exits 0.

WORST-OF-N LENSES (P1 money path): correctness = PASS (all four criteria reproduce; resolver re-derived from first principles; money-path SQL traced line-by-line for all three seams). does-it-reproduce = CONDITIONAL (§2 verbatim block does not reproduce; required lint gate exits 1). scope-honesty = PASS (the tier downgrade is stated up-front and loudly, queued as P0 RESTART BLOCKER 79.55 with the non-uniformity corrected, "What this does NOT prove" is accurate — the backend at pid 70791 does predate the edits — and cycles 1-2 are recorded rather than buried). min() = CONDITIONAL.

TO CLEAR THIS: regenerate experiment_results §2 from a fresh run of the immutable command and fix "+11 tests" to 12; then either delete the 6 dead imports in ticket_queue_processor.py or queue them as their own masterplan step, and re-run `uvx ruff check --select F821,F401,F811 $(git diff --name-only HEAD -- '*.py')` to exit 0. No product-code change is required — criteria 1-4 are met.

---

## Cycle 4 — FAIL
Run id: `wf_fc4deaa7-2ce` · verdict **FAIL** · `ok: false` · `harness_compliance_ok: true`

**All four immutable criteria MET and verified by execution.** The Q/A states the FAIL is
"narrowly and mechanically, NOT on the engineering", forced by the 3rd-CONDITIONAL rule
(two consecutive CONDITIONALs already on record, so a third was barred).

### violated_criteria (verbatim)
- `no_unintended_production_change (frontend/next-env.d.ts rides the commit, out of boundary)`
- `qa.md §1b frontend lint gate (npx eslint . exit=1)`
- `4b claim-auditing: live_check §5 describes a resolver rule the shipped code does not implement`

### reason (verbatim)

CONDITIONAL COUNT (stated explicitly, as instructed): `grep -E 'phase=78\.2' handoff/harness_log.md` returns 0 lines, so I counted from evaluator_critique_78.2.md — Cycle 1 = FAIL, Cycle 2 = CONDITIONAL, Cycle 3 = CONDITIONAL. That is TWO consecutive CONDITIONALs with no intervening PASS/FAIL, so a third CONDITIONAL from me would be the third and is barred by qa.md's 3rd-CONDITIONAL rule. I am therefore forced to PASS or FAIL, and the honest call is FAIL — narrowly and mechanically, NOT on the engineering. ALL FOUR IMMUTABLE CRITERIA ARE MET and I verified each by EXECUTION rather than by reading Main's account: (1) `claude_code_invoke` carries `model: Optional[str] = None` (runtime `inspect.signature` confirms) and emits `["--model", model]`; the derived AST walker over backend/**/*.py finds >=4 sites all passing `model=`, and my in-memory mutation stripping the emission drove it RED. (2) `agent_model_map` is honored — the test drives the real `_spawn_real_agent` per agent id and asserts the model REACHED the invocation; my mutation hardcoding the model drove it RED ("reached the CLI as 'claude-haiku-4-5', want 'claude-opus-4-8'"). (3) all three seams resolve+log — I drove the three-seam behavioural test GREEN and my mutation making seam 2 log the REQUESTED model drove it RED. (4) the mutation criterion reproduces directly from (1) and (2). BOTH TASKS YOU ASKED ME TO SWEEP ARE CLEAN. (A) Every number in the regenerated §2/§2b reproduces EXACTLY in my environment: `72 passed` (exit 0), `72 tests collected`, `git diff | grep -cE '^\+def test_'` = 12, and each of those 12 defs contributes exactly 1 collected node-id so the `60 passed` baseline is arithmetically sound; §2b's MODEL_PRICING block reproduces byte-for-byte including `default: (0.1, 0.4)` (note: my first attempt to reproduce it FAILED with an ImportError — that was MY wrong module path, MODEL_PRICING lives in backend/agents/cost_tracker.py and spend.py imports it at :181; Main's claim is correct). (B) The lint gate is genuinely green over a genuinely non-empty scope: `--show-files` proves 4 files linted, not zero; I proved the gate is LIVE by piping HEAD's ticket_queue_processor.py through it (exit=1, exactly the 6 F401s Main named), the WT version exits 0, all 6 removed names have 0 residual occurrences, none sat in a try/except probe, and they are NOT 75.5.6's scope (that step covers backend/autonomous_loop.py — a DIFFERENT file, untouched by this diff, both probe imports intact). WHAT BLOCKS PASS is entirely outside the four criteria and outside the step's declared boundary: `frontend/next-env.d.ts` is still tracked-and-modified, repointing its TypeScript reference at `frontend/.next-audit-3100/types/routes.d.ts` — a directory THIS SAME DIFF adds to .gitignore (`.next-*/`) and which git confirms is untracked. The auto-commit hook `git add -A`s on the status flip, so 78.2's commit would ship a tracked file referencing a path guaranteed absent from the repo. Cycle 3 named this verbatim and told Main to restore it; Main restored frontend/tsconfig.json and the PNGs but left this one. Compounding it, qa.md §1b binds unconditionally on a frontend/**-touching diff and `npx eslint .` exits 1 (26 errors) — all 26 in the untracked generated dirs `.next-audit-3100/` and `.next-functional/`, zero in tracked source, i.e. the same uncleaned UI-audit rig, which .gitignore fixes for git but not for eslint. My PASS bar requires "no unintended production change" and §1b says non-zero exit = FAIL; two of my own gates say FAIL and I will not carve exceptions into them to reach PASS. Main should NOT redo any 78.2 engineering — it is correct and independently verified. Remediation is mechanical: restore frontend/next-env.d.ts to HEAD, get `.next-*/` into eslint's ignores so §1b goes green, fix the one stale sentence in live_check §5, then re-spawn on changed evidence.

### violation_details (verbatim)

**1. Invalid_Precondition**

- *action*: git status --short / git diff HEAD -- frontend/next-env.d.ts ; git check-ignore -v frontend/.next-audit-3100 ; git ls-files --error-unmatch frontend/.next-audit-3100
- *state*: SEVERITY=BLOCK. `frontend/next-env.d.ts` is TRACKED and MODIFIED: line 3 changed from `/// <reference path="./.next/types/routes.d.ts" />` to `/// <reference path="./.next-audit-3100/types/routes.d.ts" />`. `git check-ignore -v` returns `.gitignore:25:.next-*/  frontend/.next-audit-3100` — i.e. THIS diff's own new ignore rule — and `git ls-files --error-unmatch` confirms the directory is not tracked. So the committed tree would reference a path that cannot exist in a fresh checkout. `npx tsc --noEmit` exits 0 for me ONLY because the untracked dir is present on this machine. The step's declared BOUNDARY is `claude_code_client.py callers + tests`; frontend/** is outside it, and the contamination came from a separate 14:20 UI-audit run, 14 minutes BEFORE the 78.2 edits. The cycle-3 Q/A named this file explicitly (its note (f): "Main should restore those two files"); frontend/tsconfig.json and the root PNGs were restored, this one was not. The auto-commit-and-push hook runs `git add -A`, so it ships under 78.2's name.
- *constraint*: Spawn-prompt PASS bar: "verdict=PASS only if EVERY immutable criterion is met AND harness compliance is clean AND no unintended production change" + the masterplan step's own `[BOUNDARY: claude_code_client.py callers + tests]`. Remediation: restore frontend/next-env.d.ts to its HEAD content before the status flip.

**2. Threshold_Not_Met**

- *action*: cd frontend && npx eslint . > /dev/null 2>&1; echo exit=$?   (run BARE — my first attempt read ${PIPESTATUS[0]}, which is a bashism that silently yields an EMPTY string in this zsh 5.9 shell: vacuity shape #9, self-caught)
- *state*: SEVERITY=BLOCK-by-rule / WARN-by-attribution. eslint-exit=1, `✖ 84 problems (26 errors, 58 warnings)`. Machine-readable attribution via `npx eslint . -f json`: 6 files carry errors, ALL of them generated Next.js build output — frontend/.next-audit-3100/{server/edge-runtime-webpack.js:6, server/webpack-runtime.js:1, static/chunks/webpack.js:6} and frontend/.next-functional/{server/edge-runtime-webpack.js:6, server/webpack-runtime.js:1, static/chunks/webpack.js:6}. ZERO errors in tracked source. `npx tsc --noEmit` exits 0. The .gitignore `.next-*/` addition in this diff hides these dirs from git but NOT from eslint, which has no matching ignore — so the project's own §1b gate is red on this tree.
- *constraint*: qa.md §1b: "For ANY phase whose diff touches frontend/** ... Q/A MUST run BOTH ... Either non-zero exit = FAIL." The diff touches frontend/** solely via the unintended next-env.d.ts change above, so this finding is downstream of the same uncleaned rig. Remediation: add `.next-*/` (and `.next-functional/`) to frontend/eslint.config.mjs ignores, or remove the stray build dirs, so the gate exits 0.

**3. Contradiction**

- *action*: Read handoff/current/live_check_78.2.md:190-194 and compare against the shipped resolver, which I executed directly: resolve_rail_model(REAL_TWO_MODEL_ENVELOPE, 'claude-haiku-4-5') -> 'claude-opus-5'
- *state*: SEVERITY=WARN. live_check §5 item 1 says the cycle-1 bug was "Replaced with the documented rule in §2b (exact match first, dominant-by-cost second)." There is no exact-match-first branch in the shipped code — that WAS the cycle-2 rule, and cycle 2 killed it precisely because map membership is not authorship. The production docstring says so explicitly ("Deliberately NOT short-circuited on `requested in named`"), §2b's own capture block shows `requested=claude-haiku-4-5 ... -> claude-opus-5`, and §6 item 1 thirty lines later correctly describes the removal. So §5's parenthetical is a stale description of code that no longer exists, contradicting two other sections of the same artifact and the source. Non-blocking on its own (the artifact self-corrects, and I verified the CODE is right), but it is one more instance of the exact defect class this step has been failing on for three cycles: a sentence carried forward past the change it describes.
- *constraint*: qa.md §4b: "A 'verbatim' capture must be regenerated, never edited" / every claim in a handoff artifact must reproduce against the shipped code. Remediation: one-sentence edit to live_check §5 item 1 to describe the dominant-entry rule with no exact-match short-circuit.

### checks_run (verbatim)

- `harness_compliance_audit_5_item`
- `research_gate_envelope_check`
- `contract_before_generate_mtime_chain`
- `log_last_check`
- `no_verdict_shopping_evidence_changed_check`
- `immutable_criteria_byte_identical_vs_HEAD_sha256`
- `verification_command_exit_code`
- `pytest_collect_only`
- `added_test_count_derivation`
- `baseline_60_arithmetic_reproduction`
- `python_lint_gate_1a_derived_scope`
- `lint_gate_show_files_nonempty_scope_proof`
- `lint_gate_mutation_test_HEAD_vs_WT`
- `dead_import_residual_usage_scan`
- `75_5_6_scope_separation_check`
- `frontend_eslint_1b`
- `frontend_tsc_noemit_1b`
- `backend_runtime_smoke_1d`
- `independent_resolver_drive_on_real_envelope`
- `guard_vacuity_check_4c`
- `independent_in_memory_mutation_matrix_with_applied_check`
- `fixture_stub_mutation`
- `claim_auditing_4b_number_sweep`
- `money_path_spend_py_exclusion_reproduction`
- `MODEL_PRICING_membership_reproduction`
- `code_review_heuristics`
- `evaluator_critique_history_review`
- `conditional_count_escalation_check`

### notes (verbatim)

HARNESS COMPLIANCE 5/5 CLEAN. (1) Research-gate-before-contract: research_brief_78.2.md 13:15:59 < contract_78.2.md 13:27:49; envelope gate_passed=true, 6 sources read in full (>=5 floor), 25 URLs, recency scan performed, 13 internal files; contract §2 cites findings R1-R8 per-claim. (2) Contract-before-generate: contract 13:27:49 < experiment_results 14:50:38. (3) experiment_results_78.2.md present with a criterion-by-criterion table; live_check_78.2.md present and satisfies the immutable live_check shape (verbatim argv per site + the new log row). (4) Log-last honoured: `grep -E 'phase=78\.2' handoff/harness_log.md` returns 0 lines and masterplan 78.2 is still `status: pending`, `retry_count: 0` — correct at EVALUATE time. (5) NOT verdict-shopping: the evidence physically changed since cycle 3 — experiment_results regenerated at 14:50:38, the 6 dead imports removed from ticket_queue_processor.py, `.next-*/` added to .gitignore, frontend/tsconfig.json restored. Immutable criteria byte-identical to HEAD: sha256 of the 78.2 `verification` object matches between `git show HEAD:.claude/masterplan.json` and the working tree (8d2374db0986f58b...); the masterplan diff ONLY adds new step 79.55.

CRITERION-BY-CRITERION (all four MET, each verified by execution).
C1 MET — runtime `inspect.signature(claude_code_invoke)` shows `model: 'Optional[str]' = None`; diff shows `if model: args.extend(["--model", model])` at the argv builder; live_check §1 A1-A5 captures argv per site; the derived AST walker test (walks backend/**/*.py, excludes tests, floor >=4 sites) runs GREEN for me, and my mutation replacing the emission with `pass` drove `test_model_argv_flag_is_actually_emitted` RED.
C2 MET — honored, not deleted, and justified in §1(b). `test_ticket_queue_agent_model_map_reaches_the_rail_invocation` drives the REAL `TicketQueueProcessor._spawn_real_agent` per agent id and asserts both that the invocation was reached and which model reached it. My mutation replacing `model=model_name` with a hardcoded literal drove it RED with the seam-naming message.
C3 MET — all three seams now resolve and log: ClaudeCodeClient._log_cc_call, autonomous_loop._log_claude_code_call (new `requested_model` kwarg threaded from all four call sites), and the new `_meter_rail` in ticket_queue_processor covering BOTH the success and the `except` path. I ran the three-seam behavioural spy test GREEN; my mutation making seam 2 log `requested_model` instead of `resolved` drove it RED.
C4 MET — Main reports 12 mutations RED with the SHA-applied check. I could not re-run Main's file-edit matrix (the `.claude/hooks/qa-write-guard.sh` PreToolUse hook correctly blocks the evaluator from writing outside .claude/agent-memory/qa/), so I built my OWN write-free in-memory mutation harness: read source, apply the mutation, assert the target was FOUND and the sha CHANGED (else report INVALID — the same discipline Main added to run_case), exec into a fresh module, re-import the test module, run the target test. Results: baseline all GREEN; QA-MUT-1 (invoke stops emitting --model) RED; QA-MUT-2 (E1 hardcodes the model) RED; QA-MUT-3 (seam 2 logs requested) RED.

GUARD-VACUITY (§4c) — I mutated the FIXTURE, not only the code, since history shows the author's matrix catches the code-side shapes while the fixture shapes are caught only by the independent Q/A. F1 (re-inject the cycle-1 fabrication: opus outputTokens 4 -> 4000) drives `test_resolved_model_max_output_tokens_would_name_the_helper` RED with "the real envelope no longer has the property that made max(outputTokens) wrong" — the trap-detector genuinely detects its own trap being re-set. F2 (blank the discriminator: opus costUSD 0.00868 -> 0.0) drives BOTH `..._names_the_worker_on_the_real_envelope` and `..._reports_a_substitution_even_when_the_helper_matches` RED. `SUBSTITUTION_ENVELOPE = REAL_TWO_MODEL_ENVELOPE`, so the three-seam spies inherit the same non-vacuous fixture. F1 leaves `..._names_the_worker` GREEN, which is CORRECT rather than vacuous: at 4000 output tokens the opus entry still dominates on cost, so the worker is still named. I found no vacuous guard among the twelve.

INDEPENDENT RESOLVER DRIVE — I re-derived every line of live_check §2b myself from the fixture numbers: requested=claude-opus-4-8 -> claude-opus-5; requested=claude-opus-5 -> claude-opus-5; requested=claude-haiku-4-5 -> claude-opus-5 (the substitution IS detectable, so cycle 2's blind spot is genuinely closed); reversed key order -> claude-opus-5 (order-independent); envelope=None -> the requested label; empty modelUsage -> requested; missing canonicalModel -> falls back to the map key; costUSD absent -> token tiebreak still names the worker; a non-dict envelope -> requested (the bare `except` cannot break the rail). `max(outputTokens)` independently reproduces the cycle-1 bug (names the haiku helper).

NUMBER SWEEP (task A) — CLEAN, first time in four cycles. 72 passed / exit 0; 72 collected; 12 added `def test_`; the 12 new defs contribute exactly 12 collected node-ids so `60 passed` baseline is sound; MODEL_PRICING block reproduces byte-for-byte. Two near-misses I checked and cleared rather than flagged: (i) live_check §0 shows haiku in=521 out=12 cost=0.000581 while the fixture says 523/12/0.000583 — these are two DIFFERENT captures (§0 is the no-flag probe, the fixture is a `--model opus` probe) and both are labelled as such, so it is not a contradiction; (ii) §2b renders the spend.py agent exclusion as `AND agent NOT LIKE 'cc_rail%'` while the real SQL is `AND (agent IS NULL OR (agent != 'cc_rail' AND agent NOT LIKE 'cc_rail:%'))` — a paraphrase in backticks, but the cited range spend.py:228-230 is EXACT and it is semantically equivalent for the actual agent values ('cc_rail', 'cc_rail:<role>', 'cc_rail:ticket_<id>'). Cycle 3 already ruled that non-blocking; I agree and did not escalate it.

LINT GATE (task B) — VERIFIED, with a self-caught trap worth recording. Running qa.md §1a's literal `uvx ruff check ... $FILES` in MY zsh 5.9 produced the FALSE GREEN — zsh does not word-split unquoted parameter expansions, so ruff got one newline-joined argument, warned "No such file or directory", printed "All checks passed!" and exited 0. That is vacuity shape #9 / the 75.5.14 defect, hit live by the evaluator hunting it. I re-ran with `git diff --name-only HEAD -- '*.py' | tr '\n' '\0' | xargs -0 uvx ruff check --select F821,F401,F811` and confirmed with `--show-files` that it lints exactly 4 files (claude_code_client.py, services/autonomous_loop.py, ticket_queue_processor.py, test_phase_75_llm_rail.py) — not zero — exit 0. Gate proven LIVE: HEAD's ticket_queue_processor.py piped through the same command exits 1 with exactly the 6 F401s Main named (subprocess, json, typing.List, pathlib.Path, TicketClassification, TicketsDB), so the "pre-existing" claim reproduces; the WT version exits 0. Removal is safe: all 6 names now have 0 occurrences anywhere in the file, and the diff shows they were plain top-level imports, not try/except probes. NOT 75.5.6's findings: that step's two imports (BacktestEngine ~:409, EvaluationVerdict ~:436) are in `backend/autonomous_loop.py`, a DIFFERENT file from `backend/services/autonomous_loop.py`; it is absent from this diff and both probes are intact. 75.5.6's scope is unchanged.

E1 BQ WRITE / BOUNDARY JUDGEMENT — in scope, in my judgement. Criterion 3 says "EVERY rail call logs the resolved model"; E1 wrote no row at all, so adding one is the minimum that satisfies the criterion, not scope creep. It is fail-open (bare try/except), writes agent='cc_rail:ticket_<id>' which the $25/day breaker excludes via its agent clause, and covers both success and failure paths. §2b's blast-radius enumeration is correct on the mechanism (B1/B2 protected by the PROVIDER clause, the other two seams by the AGENT clause) and the "do not normalise that provider string" warning is the right guard to leave behind.

RUNTIME SMOKE (§1d) — all three changed backend modules import clean in the venv; resolve_rail_model is module-level and callable. §1c does NOT bind: no UI claim in this step, so I took no browser capture. §1b DOES bind (the diff touches frontend/**) and is the second blocker above.

CODE-REVIEW HEURISTICS — no BLOCK from any of the five dimensions. No secrets, no command injection, no kill-switch/stop-loss/perf-metrics path touched, no trading-domain invariant altered, no LLM output reaching execution. The one consumer-contract change (llm_call_log.model semantics: requested -> resolved) IS grep-verified across all four consumers in §2b and documented at the change site, which is what that heuristic asks for.

WHAT MAIN SHOULD DO — do NOT redo any 78.2 engineering; it is correct and independently verified, and this is by a distance the strongest evidence set of the four cycles. Three mechanical fixes: (1) `git checkout HEAD -- frontend/next-env.d.ts`; (2) add `.next-*/` and `.next-functional/` to frontend/eslint.config.mjs ignores so §1b exits 0; (3) correct the one stale sentence at live_check §5 item 1. Then update the handoff and re-spawn a fresh Q/A on changed evidence.
