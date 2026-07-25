# Evaluator critique — phase-78.16

**EVALUATE phase.** Verdict produced by the Layer-3 **Q/A** agent, launched via
the first-class `.claude/workflows/qa-verdict.js` Workflow (structured output —
the verdict IS the captured return value). Main records the verdict; Main does
not author it.

- Run id: `wf_5f99d6eb-276` · agent `a32c64354b8413006` · agentType `qa`
- Model: `claude-opus-5[1m]` · tokens 185,715 · tool calls 35 · duration 1002s
- Cycle: 1 (no prior 78.16 critique exists; the 3rd-CONDITIONAL rule is not engaged)

---

## VERDICT: PASS   (`ok: true`, `certified_fallback: false`, `harness_compliance_ok: true`)

`violated_criteria`: []
`violation_details`: []

---

## Reason (verbatim)

All 3 immutable criteria MET, harness compliance clean (5/5), no unintended production behaviour change. DETERMINISTIC: immutable cmd `.venv/bin/python -m pytest backend/tests/ -q -k 'llm_client or make_client or prompt_caching'` exit=0, `19 passed, 2016 deselected` (reproduces the claim verbatim); regression sweep `62 passed, 1973 deselected` (verbatim); ruff F821/F401/F811 on a DERIVED 8-file set (git diff + git ls-files --others -- the qa.md recipe alone misses the untracked new test) "All checks passed!" exit=0; runtime-smoke imports 7/7 changed backend modules OK; no frontend/** files so 1b/1c not triggered. C1 MET in the STRONG form (exact restoration, not the escape hatch): llm_client.py:1475-1484 is the sole determinant of the `system` shape and I reproduced both shapes myself -- flag-OFF yields a plain 19,075-char str with no cache_control, every other kwarg identical, closed against the actual pre-78.1 construction (MATCH: True). C2 MET: 6 parametrised revert-shape guards + 3 supporting, all 9 selected by the immutable command (10 pre-existing + 9 new = 19, arithmetic consistent); completeness of "the six" independently RECALL-TESTED against git history (`git log -S'enable_prompt_caching=False' -- backend/services` -> 6 add-commits; 78d794c5 removed exactly 1 line from each of the same 6) -- symmetric difference vs the test's C_BLOCK is EMPTY. C3 MET and NON-VACUOUS by my own battery run in-memory with production files untouched: M1 service-drops-kwarg RED x6 at :211; M1b behavioral leg in isolation RED x6 at :175 ("`system` is a list, not a plain str"); M2 make_client-drops-forwarding RED x6 at :175; M3 default None->False RED at :186; plus four Q/A-original mutations Main did not run -- M5 ClaudeClient class-default flip RED, M6 service flips to True RED, M7 make_client inverts value RED both directions, M8 caching branch deleted RED. Restore integrity independently verified: Main's bk78_16 backups are 3/3 byte-IDENTICAL to the current working tree, and meta_scorer.py sha256=ea4a2f87b7a697c266e03f65a2028a2101fc8cca56bc87907c67a6337db92b2b exactly matches the value quoted in live_check. Worst-of-N-lenses (P1 money path): correctness=PASS, does-it-reproduce=PASS, scope-honesty=PASS -> min=PASS. Six NOTE-level findings recorded in `notes`; none blocking.

---

## Checks run (verbatim)

- `harness_compliance_audit_5_item`
- `research_gate_envelope_verification`
- `mtime_ordering_research_lt_contract_lt_artifact`
- `log_last_and_status_flip_check`
- `no_verdict_shopping_check`
- `3rd_conditional_counter_check`
- `immutable_verification_command`
- `regression_sweep_reproduction`
- `python_lint_gate_ruff_F821_F401_F811_derived_scope`
- `backend_runtime_smoke_imports`
- `full_backend_test_suite_preexisting_failure_attribution`
- `syntax_ast_parse`
- `git_diff_scope_review`
- `masterplan_structural_json_diff_immutability_proof`
- `independent_mutation_battery_10_mutations`
- `mutation_restore_integrity_sha_verification`
- `guard_vacuity_check_4c`
- `claim_auditing_4b`
- `known_member_recall_test_the_six`
- `symmetric_difference_derivation`
- `blast_radius_caller_enumeration`
- `code_review_heuristics`
- `secret_in_diff_scan`
- `test_isolation_env_var_check`
- `adversarial_worst_of_n_lenses`

---

## Notes (verbatim)

ANSWERS TO THE 7 ATTACK POINTS.

(1) VACUITY -- NOT vacuous, proven by execution in both directions. Cannot pass when a service drops the kwarg: M1 RED x6 at :211 (`assert intent is False`), and M1b proves the BEHAVIORAL half is independently RED x6 at :175 ("`system` is a list"). Cannot pass when make_client stops forwarding: M2 RED x6 at :175. I also verified the AST extractor genuinely returns None after an in-memory deletion of the kwarg from each of the six CALL lines (6/6 OK). Caveat -> finding N1 below.

(2) RENAME -- LEGITIMATE, not gaming. I loaded both masterplan versions and compared structurally: 971 -> 972 steps, ADDED=['78.17'], REMOVED=[], and ZERO differences on all 971 shared steps (including 78.16's success_criteria, verification.command and live_check), zero phase-meta differences, only top-level `updated_at` changed. So the criteria and command are provably untouched. Under the old filename only 2 of 9 tests matched the immutable -k (by name: the two containing `make_client`); the rename brought all 9 under the gate -- MORE coverage, not less. Disclosed prominently in experiment_results section 2.

(3) SCOPE -- no smuggled unmeasured claim found. Every reproducible number reproduces: 19/2016, 62/1973, 9 tests collected, 10 baseline (derived: 19-9, and no pre-existing test file was modified), _HOUSE_INSTRUCTIONS=19,026 chars (19,075 = 19,026+49 suffix, arithmetic checks), 7 non-C-block callers, 13 total make_client callers, exactly ONE production `ClaudeClient(**)` construction site, sha ea4a2f87... The token-count figures (3,877/4,551/4,769) are explicitly labelled straddling heuristics with the authoritative measurement named as unavailable -- that is a disclosed NON-measurement, which is the honest form, not a smuggled assertion. The refusal of option (c) is well-founded.

(4) BLAST RADIUS -- verified. Independently enumerated the 7 non-C-block callers: orchestrator.py:652/653/654/659, quant_optimizer.py:478, autonomous_loop.py:2696/2932. All pass 3 args and no caching kwarg; the new param is KEYWORD-ONLY with default None, and `if enable_prompt_caching is not None` short-circuits, so their construction is byte-identical. `test_make_client_default_leaves_class_default_untouched` is a REAL guard, not a tautology: M3 (default None->False) kills it at :186 and my M5 (ClaudeClient CLASS default True->False) kills it too.

(5) M1 DISCLOSURE HONESTY -- TRUE, verified. `, enable_prompt_caching=False)` first occurs in meta_scorer.py at line 231, which IS a comment line, with exactly 2 occurrences in the file -- so `.replace(..., 1)` provably hits the comment, not the call at :242. Nothing was weakened: the working tree is byte-identical to Main's pre-matrix backups (3/3) and meta_scorer.py's sha256 matches the quoted value exactly.

(6) 78.17 -- EXISTS, status `pending`, harness_required true, P2, with executor tag, explicit BOUNDARY, the single deciding measurement named (cache_creation_input_tokens on a real haiku-4-5 response), an explicit BLOCKED-TODAY precondition, "Do NOT proceed on an estimate", and a research-gate tier. Written for an executor with no memory of this session.

(7) BOUNDARY -- NOT a breach; state this explicitly. `make_client`'s new parameter is inert unless a caller passes it, and criterion 1 is written about "the six", so the criterion is broader than the boundary sentence. The service edits are exactly one kwarg plus a comment block each -- I read all six diffs line-by-line: no logic, threshold, schema or control-flow change. Disclosed in experiment_results section 4. Necessary and minimal.

NON-BLOCKING FINDINGS (NOTE level; none degrade the verdict).

N1 [live_check completeness] -- live_check section 2's M4 row is labelled "RED / 2 failed, 17 passed" but does not state the most informative result in the whole matrix: ALL SIX `test_revert_path_restores_pre_78_1_request_shape` cases stay GREEN under the str-forcing stub (I reproduced this exactly). Taken alone the six cannot distinguish "the code correctly emits a str" from "the fixture lies and emits a str"; what closes that hole is the pair `test_make_client_forwards_caching_true` + `test_make_client_default_leaves_class_default_untouched`, in the same file under the same immutable command. The suite is non-vacuous, but the artifact should say this in one sentence rather than leave the row reading as an unqualified RED.

N2 [vacuity shape #11, mis-attributed kill mechanism] -- the test module docstring (lines 22-26) says dropping the kwarg makes "the class default True apply, the wire shape become the cached list, and the test go RED". Measured: it goes RED at :211 BEFORE `_wire_kwargs` is ever called, so the described mechanism is unreachable in that scenario. Outcome correct, causal narrative wrong. Not a guard defect (M1b shows the behavioral leg is independently red).

N3 [reproducibility] -- `scratchpad/mutate_78_16.sh` as saved still contains the comment-hitting `replace(", enable_prompt_caching=False)", ")", 1)`; re-running it verbatim prints "*** GREEN -- VACUOUS GUARD ***" for M1, i.e. it does not reproduce its own table row. The prose discloses this fully, but the "Script:" pointer under the table is stale. Low impact (not checked in).

N4 [undisclosed side effect, net-positive] -- .claude/masterplan.json was re-serialized with ensure_ascii=False, rewriting ~300 unrelated lines across phases 4.5/12/16/17/23 (— -> em dash, § -> section sign). I proved ZERO semantic drift and jq still parses it. Notably this RESTORES the encoding used by the phase-75.17 baseline commit: the masterplan diff vs that baseline SHRINKS from 117 removed lines (106 non-comma-artifact) at HEAD to 22 removed (11 non-artifact) in the working tree. So it is an improvement, not a corruption -- but experiment_results does not mention it at all and it will make this commit's diff noisy. Worth one line of disclosure.

N5 [pre-existing, queue-worthy] -- the full `backend/tests/` suite is 13 failed / 2004 passed. I measured all 13 as PRE-EXISTING and NOT caused by this step: the collection-count pin (test_phase_75_ci_gates) is pinned at 1563/1579 vs an actual 2019/2035 -- a +456 drift of which this step contributes 9; test_phase_75_17_verification_paths::test_sweep_shape_census pins dict=720 vs 808 already at HEAD (this step makes it 809); test_masterplan_diff_touches_only_the_ten_sibling_insertions already had 106 non-artifact removals at HEAD (this step reduces it to 11); the remaining 8 test files contain ZERO references to make_client/enable_prompt_caching/the six services, and the one apparent hit (test_phase_75_prompt_contracts) is its own local `_claude_kwargs` helper failing on a missing `handoff/current/operator_decision_75.14_schema_extension.md`. Main never claimed full-suite green, so this is not an overclaim -- but the three brittle count/diff pins fail on ANY test or masterplan addition and are queue-worthy per feedback_queue_discovered_defects_in_masterplan. Main's call.

N6 [un-re-derived claim] -- the BigQuery llm_call_log 60-day "zero provider='anthropic' rows for claude-haiku-4-5" claim was not independently reproduced here (it needs an execute-query approval prompt). It is a self-adverse disclosure that weakens the step's own urgency, so motivated-error risk is low, but it stands un-verified in this verdict.

HARNESS COMPLIANCE DETAIL (5/5 clean). (a) Research gate: research_brief_78.16.md, gate_passed=true, external_sources_read_in_full=7 (>=5 floor), urls_collected=21 (I independently recounted unique URLs: exactly 21), recency scan section D present, 3-variant query discipline section E present, internal_files_inspected=16; contract cites the brief. (b) Contract-before-generate: mtimes research 13:15:45 < contract 13:15:54 < llm_client.py 13:20:14 < test file 13:20:21 < meta_scorer 13:21:09 < experiment_results 13:23:38; Main's wire probe (probe_caching_wire.py, 13:09:25) predates the contract, so the contract's "measurements I made before writing this contract" is true. (c) experiment_results present. (d) Log-last: zero `phase=78.16` entries in harness_log.md and masterplan 78.16 status is still `pending` -- nothing flipped early. (e) No-verdict-shopping: this is cycle 1, no prior 78.16 critique exists, 0 prior CONDITIONALs, so the 3rd-CONDITIONAL rule is not engaged. Code-review heuristics: no BLOCK or WARN fired; the only secret-shaped literal is `sk-ant-api-test-not-real` in a test fixture (negation-list exempt); the module-level `os.environ.setdefault("COST_BUDGET_HARD_BLOCK_DISABLED","1")` follows the established precedent at test_claude_request_shapes.py:26 and test_phase_75_5_1_spend_metric.py:312 explicitly delenv's it, so there is no test-isolation regression. UI capture gate 1c NOT triggered: zero frontend/** files in the diff and the step makes no UI claims. Live-API exercise of the metered Claude path was not possible (direct-API credits dead, owed operator action 79.3); the SDK-boundary wire capture is the strongest available evidence and live_check section 3 discloses that bound honestly rather than papering over it.

---

## Raw return value (verbatim JSON)

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 3 immutable criteria MET, harness compliance clean (5/5), no unintended production behaviour change. DETERMINISTIC: immutable cmd `.venv/bin/python -m pytest backend/tests/ -q -k 'llm_client or make_client or prompt_caching'` exit=0, `19 passed, 2016 deselected` (reproduces the claim verbatim); regression sweep `62 passed, 1973 deselected` (verbatim); ruff F821/F401/F811 on a DERIVED 8-file set (git diff + git ls-files --others -- the qa.md recipe alone misses the untracked new test) \"All checks passed!\" exit=0; runtime-smoke imports 7/7 changed backend modules OK; no frontend/** files so 1b/1c not triggered. C1 MET in the STRONG form (exact restoration, not the escape hatch): llm_client.py:1475-1484 is the sole determinant of the `system` shape and I reproduced both shapes myself -- flag-OFF yields a plain 19,075-char str with no cache_control, every other kwarg identical, closed against the actual pre-78.1 construction (MATCH: True). C2 MET: 6 parametrised revert-shape guards + 3 supporting, all 9 selected by the immutable command (10 pre-existing + 9 new = 19, arithmetic consistent); completeness of \"the six\" independently RECALL-TESTED against git history (`git log -S'enable_prompt_caching=False' -- backend/services` -> 6 add-commits; 78d794c5 removed exactly 1 line from each of the same 6) -- symmetric difference vs the test's C_BLOCK is EMPTY. C3 MET and NON-VACUOUS by my own battery run in-memory with production files untouched: M1 service-drops-kwarg RED x6 at :211; M1b behavioral leg in isolation RED x6 at :175 (\"`system` is a list, not a plain str\"); M2 make_client-drops-forwarding RED x6 at :175; M3 default None->False RED at :186; plus four Q/A-original mutations Main did not run -- M5 ClaudeClient class-default flip RED, M6 service flips to True RED, M7 make_client inverts value RED both directions, M8 caching branch deleted RED. Restore integrity independently verified: Main's bk78_16 backups are 3/3 byte-IDENTICAL to the current working tree, and meta_scorer.py sha256=ea4a2f87b7a697c266e03f65a2028a2101fc8cca56bc87907c67a6337db92b2b exactly matches the value quoted in live_check. Worst-of-N-lenses (P1 money path): correctness=PASS, does-it-reproduce=PASS, scope-honesty=PASS -> min=PASS. Six NOTE-level findings recorded in `notes`; none blocking.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "harness_compliance_ok": true,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "research_gate_envelope_verification",
    "mtime_ordering_research_lt_contract_lt_artifact",
    "log_last_and_status_flip_check",
    "no_verdict_shopping_check",
    "3rd_conditional_counter_check",
    "immutable_verification_command",
    "regression_sweep_reproduction",
    "python_lint_gate_ruff_F821_F401_F811_derived_scope",
    "backend_runtime_smoke_imports",
    "full_backend_test_suite_preexisting_failure_attribution",
    "syntax_ast_parse",
    "git_diff_scope_review",
    "masterplan_structural_json_diff_immutability_proof",
    "independent_mutation_battery_10_mutations",
    "mutation_restore_integrity_sha_verification",
    "guard_vacuity_check_4c",
    "claim_auditing_4b",
    "known_member_recall_test_the_six",
    "symmetric_difference_derivation",
    "blast_radius_caller_enumeration",
    "code_review_heuristics",
    "secret_in_diff_scan",
    "test_isolation_env_var_check",
    "adversarial_worst_of_n_lenses"
  ],
  "notes": "ANSWERS TO THE 7 ATTACK POINTS.\n\n(1) VACUITY -- NOT vacuous, proven by execution in both directions. Cannot pass when a service drops the kwarg: M1 RED x6 at :211 (`assert intent is False`), and M1b proves the BEHAVIORAL half is independently RED x6 at :175 (\"`system` is a list\"). Cannot pass when make_client stops forwarding: M2 RED x6 at :175. I also verified the AST extractor genuinely returns None after an in-memory deletion of the kwarg from each of the six CALL lines (6/6 OK). Caveat -> finding N1 below.\n\n(2) RENAME -- LEGITIMATE, not gaming. I loaded both masterplan versions and compared structurally: 971 -> 972 steps, ADDED=['78.17'], REMOVED=[], and ZERO differences on all 971 shared steps (including 78.16's success_criteria, verification.command and live_check), zero phase-meta differences, only top-level `updated_at` changed. So the criteria and command are provably untouched. Under the old filename only 2 of 9 tests matched the immutable -k (by name: the two containing `make_client`); the rename brought all 9 under the gate -- MORE coverage, not less. Disclosed prominently in experiment_results section 2.\n\n(3) SCOPE -- no smuggled unmeasured claim found. Every reproducible number reproduces: 19/2016, 62/1973, 9 tests collected, 10 baseline (derived: 19-9, and no pre-existing test file was modified), _HOUSE_INSTRUCTIONS=19,026 chars (19,075 = 19,026+49 suffix, arithmetic checks), 7 non-C-block callers, 13 total make_client callers, exactly ONE production `ClaudeClient(**)` construction site, sha ea4a2f87... The token-count figures (3,877/4,551/4,769) are explicitly labelled straddling heuristics with the authoritative measurement named as unavailable -- that is a disclosed NON-measurement, which is the honest form, not a smuggled assertion. The refusal of option (c) is well-founded.\n\n(4) BLAST RADIUS -- verified. Independently enumerated the 7 non-C-block callers: orchestrator.py:652/653/654/659, quant_optimizer.py:478, autonomous_loop.py:2696/2932. All pass 3 args and no caching kwarg; the new param is KEYWORD-ONLY with default None, and `if enable_prompt_caching is not None` short-circuits, so their construction is byte-identical. `test_make_client_default_leaves_class_default_untouched` is a REAL guard, not a tautology: M3 (default None->False) kills it at :186 and my M5 (ClaudeClient CLASS default True->False) kills it too.\n\n(5) M1 DISCLOSURE HONESTY -- TRUE, verified. `, enable_prompt_caching=False)` first occurs in meta_scorer.py at line 231, which IS a comment line, with exactly 2 occurrences in the file -- so `.replace(..., 1)` provably hits the comment, not the call at :242. Nothing was weakened: the working tree is byte-identical to Main's pre-matrix backups (3/3) and meta_scorer.py's sha256 matches the quoted value exactly.\n\n(6) 78.17 -- EXISTS, status `pending`, harness_required true, P2, with executor tag, explicit BOUNDARY, the single deciding measurement named (cache_creation_input_tokens on a real haiku-4-5 response), an explicit BLOCKED-TODAY precondition, \"Do NOT proceed on an estimate\", and a research-gate tier. Written for an executor with no memory of this session.\n\n(7) BOUNDARY -- NOT a breach; state this explicitly. `make_client`'s new parameter is inert unless a caller passes it, and criterion 1 is written about \"the six\", so the criterion is broader than the boundary sentence. The service edits are exactly one kwarg plus a comment block each -- I read all six diffs line-by-line: no logic, threshold, schema or control-flow change. Disclosed in experiment_results section 4. Necessary and minimal.\n\nNON-BLOCKING FINDINGS (NOTE level; none degrade the verdict).\n\nN1 [live_check completeness] -- live_check section 2's M4 row is labelled \"RED / 2 failed, 17 passed\" but does not state the most informative result in the whole matrix: ALL SIX `test_revert_path_restores_pre_78_1_request_shape` cases stay GREEN under the str-forcing stub (I reproduced this exactly). Taken alone the six cannot distinguish \"the code correctly emits a str\" from \"the fixture lies and emits a str\"; what closes that hole is the pair `test_make_client_forwards_caching_true` + `test_make_client_default_leaves_class_default_untouched`, in the same file under the same immutable command. The suite is non-vacuous, but the artifact should say this in one sentence rather than leave the row reading as an unqualified RED.\n\nN2 [vacuity shape #11, mis-attributed kill mechanism] -- the test module docstring (lines 22-26) says dropping the kwarg makes \"the class default True apply, the wire shape become the cached list, and the test go RED\". Measured: it goes RED at :211 BEFORE `_wire_kwargs` is ever called, so the described mechanism is unreachable in that scenario. Outcome correct, causal narrative wrong. Not a guard defect (M1b shows the behavioral leg is independently red).\n\nN3 [reproducibility] -- `scratchpad/mutate_78_16.sh` as saved still contains the comment-hitting `replace(\", enable_prompt_caching=False)\", \")\", 1)`; re-running it verbatim prints \"*** GREEN -- VACUOUS GUARD ***\" for M1, i.e. it does not reproduce its own table row. The prose discloses this fully, but the \"Script:\" pointer under the table is stale. Low impact (not checked in).\n\nN4 [undisclosed side effect, net-positive] -- .claude/masterplan.json was re-serialized with ensure_ascii=False, rewriting ~300 unrelated lines across phases 4.5/12/16/17/23 (— -> em dash, § -> section sign). I proved ZERO semantic drift and jq still parses it. Notably this RESTORES the encoding used by the phase-75.17 baseline commit: the masterplan diff vs that baseline SHRINKS from 117 removed lines (106 non-comma-artifact) at HEAD to 22 removed (11 non-artifact) in the working tree. So it is an improvement, not a corruption -- but experiment_results does not mention it at all and it will make this commit's diff noisy. Worth one line of disclosure.\n\nN5 [pre-existing, queue-worthy] -- the full `backend/tests/` suite is 13 failed / 2004 passed. I measured all 13 as PRE-EXISTING and NOT caused by this step: the collection-count pin (test_phase_75_ci_gates) is pinned at 1563/1579 vs an actual 2019/2035 -- a +456 drift of which this step contributes 9; test_phase_75_17_verification_paths::test_sweep_shape_census pins dict=720 vs 808 already at HEAD (this step makes it 809); test_masterplan_diff_touches_only_the_ten_sibling_insertions already had 106 non-artifact removals at HEAD (this step reduces it to 11); the remaining 8 test files contain ZERO references to make_client/enable_prompt_caching/the six services, and the one apparent hit (test_phase_75_prompt_contracts) is its own local `_claude_kwargs` helper failing on a missing `handoff/current/operator_decision_75.14_schema_extension.md`. Main never claimed full-suite green, so this is not an overclaim -- but the three brittle count/diff pins fail on ANY test or masterplan addition and are queue-worthy per feedback_queue_discovered_defects_in_masterplan. Main's call.\n\nN6 [un-re-derived claim] -- the BigQuery llm_call_log 60-day \"zero provider='anthropic' rows for claude-haiku-4-5\" claim was not independently reproduced here (it needs an execute-query approval prompt). It is a self-adverse disclosure that weakens the step's own urgency, so motivated-error risk is low, but it stands un-verified in this verdict.\n\nHARNESS COMPLIANCE DETAIL (5/5 clean). (a) Research gate: research_brief_78.16.md, gate_passed=true, external_sources_read_in_full=7 (>=5 floor), urls_collected=21 (I independently recounted unique URLs: exactly 21), recency scan section D present, 3-variant query discipline section E present, internal_files_inspected=16; contract cites the brief. (b) Contract-before-generate: mtimes research 13:15:45 < contract 13:15:54 < llm_client.py 13:20:14 < test file 13:20:21 < meta_scorer 13:21:09 < experiment_results 13:23:38; Main's wire probe (probe_caching_wire.py, 13:09:25) predates the contract, so the contract's \"measurements I made before writing this contract\" is true. (c) experiment_results present. (d) Log-last: zero `phase=78.16` entries in harness_log.md and masterplan 78.16 status is still `pending` -- nothing flipped early. (e) No-verdict-shopping: this is cycle 1, no prior 78.16 critique exists, 0 prior CONDITIONALs, so the 3rd-CONDITIONAL rule is not engaged. Code-review heuristics: no BLOCK or WARN fired; the only secret-shaped literal is `sk-ant-api-test-not-real` in a test fixture (negation-list exempt); the module-level `os.environ.setdefault(\"COST_BUDGET_HARD_BLOCK_DISABLED\",\"1\")` follows the established precedent at test_claude_request_shapes.py:26 and test_phase_75_5_1_spend_metric.py:312 explicitly delenv's it, so there is no test-isolation regression. UI capture gate 1c NOT triggered: zero frontend/** files in the diff and the step makes no UI claims. Live-API exercise of the metered Claude path was not possible (direct-API credits dead, owed operator action 79.3); the SDK-boundary wire capture is the strongest available evidence and live_check section 3 discloses that bound honestly rather than papering over it."
}
```
