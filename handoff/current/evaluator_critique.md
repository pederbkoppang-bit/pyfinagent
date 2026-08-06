# Evaluator Critique -- phase-82.51

**Step:** 82.51 (P1) -- publication-lag look-ahead on every fundamentals read.
**Date:** 2026-08-06.
**Evaluator:** Layer-3 Q/A via the Workflow structured-output rail
(`.claude/workflows/qa-verdict.js`), model `claude-opus-5[1m]`, effort max.
**Launch:** 18 tool calls, 162,540 tokens, 618s.

Main records the verdict; Main never authors it. Transcribed verbatim from the
rail's output file.

---

## Cycle 1 -- verdict: CONDITIONAL

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "All 5 immutable criteria are MET and I proved criteria 1/2/3 by EXECUTING my own mutation matrix (7 mutants, fresh subprocess each, CONTROL green): the branch-1-only mutant you asked me to attack DIES (killed by test_both_read_paths_agree_on_the_same_fixture + test_the_sql_path_binds_the_embargoed_cutoff_as_its_parameter), so _embargoed_cutoff is a real shared seam and criterion 3 is genuinely covered; your branch-3 stub filters on the PRODUCTION-bound @cutoff value, not a hardcoded one, so it does not test itself. Verification command exit=0 (12 passed), lint clean on a 6-file derived scope, imports OK, scoped regression reproduces 1 failed/274 passed EXACTLY, and the paper_trader failure is independently confirmed pre-existing (identical `assert None is not None` with HEAD copies of all three changed modules injected). Harness compliance clean, no unintended production change. Verdict is capped at CONDITIONAL by three WARN findings OUTSIDE the criteria, all cited and all small: (1) quant_optimizer.py:176 still calls window_is_covered(window_start) with no start=, so the optimizer now says COVERED for windows the engine REFUSES with ValueError -- an inconsistency this diff introduced and the artifact does not disclose; (2) test_the_refusal_site_judges_against_the_effective_start checks only that the `start=` kwarg NAME exists, so the semantically identical mutant start=FUNDAMENTALS_COVERAGE_START SURVIVES it (I ran the guard's own AST predicate against that text: GREEN) -- your M_C \"DIED\" is true only for the kwarg-deletion construction; (3) cached_fundamentals has a LIVE non-backtest consumer at backend/agents/mcp_servers/data_server.py:149 (cutoff = date.today()), so the live MCP fundamentals tool now hides the most recent quarter for 60 days -- a real behaviour change under an artifact section that says \"No live positions touched\".",
  "violated_criteria": [
    "consumer-contract-break [WARN]: quant_optimizer.py:176 window_is_covered called without start=",
    "illusory-guard [WARN]: refusal-site guard asserts kwarg presence, not the value",
    "consumer-contract-break [WARN]: undisclosed live-path change at data_server.py:149"
  ],
  "violation_details": [
    {
      "violation_type": "Contradiction",
      "action": "grep -rn 'window_is_covered' --include='*.py' backend scripts",
      "state": "SEVERITY=WARN. backend/backtest/backtest_engine.py:489 now calls window_is_covered(window_start, start=effective_start) [2024-08-29], but backend/backtest/quant_optimizer.py:176 still calls window_is_covered(window_start) with no start=, so it judges against the RAW start [2024-06-30]. For any window starting 2024-06-30..2024-08-28 the optimizer keeps 'qarp' in the selectable pool as COVERED while BacktestEngine raises ValueError: backtest REFUSED. Pre-82.51 the two modules agreed; this diff made them disagree. The step's own guard scans only backtest_engine.py so it cannot see this.",
      "constraint": "consumer-contract-break [WARN] -- a change to a shared predicate's effective semantics requires every consumer grep-verified in the SAME diff (code-review skill Dim 3 #16). Fix: pass start=effective_coverage_start() at quant_optimizer.py:176 or record why the raw start is correct there, and widen the AST sweep to both modules."
    },
    {
      "violation_type": "Overgeneralization",
      "action": "Re-ran the guard's own AST predicate over backtest_engine.py source with window_is_covered(window_start, start=effective_start) replaced by window_is_covered(window_start, start=FUNDAMENTALS_COVERAGE_START)",
      "state": "SEVERITY=WARN. Guard result on real source = GREEN; guard result on the value-substituted mutant = GREEN. test_the_refusal_site_judges_against_the_effective_start (backend/tests/test_phase_82_51_fundamentals_embargo.py:277-298) asserts only `'start' in kwarg_names`, never that the bound value is the effective start -- so the exact 82.21 false-pass the test claims to prevent can be re-created while the test stays green. experiment_results.md sec.7 records 'M_C refusal judged against the RAW start DIED', which holds only for the kwarg-DELETION construction, not for the semantically identical value substitution.",
      "constraint": "illusory-guard shape #2 (source-scan defeated by rewording) + shape #11 (mis-attributed kill mechanism), qa.md sec.4c. WARN not BLOCK: a genuine behavioural guard for the derivation coexists (test_effective_coverage_start_is_derived_not_a_second_literal) and no immutable criterion depends solely on this guard. Fix: assert the call's start= value resolves to effective_coverage_start(), or drive the refusal behaviourally with window_start=2024-07-01 and assert ValueError."
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "grep -rn 'cached_fundamentals' --include='*.py' backend scripts; sed -n '140,158p' backend/agents/mcp_servers/data_server.py",
      "state": "SEVERITY=WARN. cached_fundamentals has two consumers outside the backtest feature builder: backend/backtest/historical_data.py:61 (in scope) and backend/agents/mcp_servers/data_server.py:149, which calls cache.cached_fundamentals(ticker, date.today().isoformat()) for the LIVE MCP data server. After this diff that live tool returns fundamentals as of today-60d, so the most recently reported quarter becomes invisible to the live agent pipeline for 60 days after period end. experiment_results.md sec.10 states 'No live positions touched. These are historical backtests' and never enumerates this consumer.",
      "constraint": "Scope honesty (qa.md sec.4: did experiment_results disclose scope bounds rather than overclaim) + consumer-contract-break [WARN]. Fix: disclose the data_server.py:149 consumer and state explicitly whether a 60-day embargo is intended on a live as-of-today query, or scope the embargo to the backtest read path."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "research_gate_envelope",
    "mtime_ordering_research_lt_contract_lt_artifact",
    "log_last_masterplan_status_pending",
    "third_conditional_counter",
    "verification_command",
    "syntax_import_smoke",
    "python_lint_gate_derived_scope",
    "git_status_unintended_change_scan",
    "mutation_matrix_7_mutants_subprocess_isolated",
    "control_run_harness_faithfulness",
    "stub_self_test_audit",
    "guard_defeatability_simulation",
    "consumer_grep_window_is_covered",
    "consumer_grep_cached_fundamentals",
    "line_number_rederivation_82_12",
    "pre_existing_failure_head_module_injection",
    "scoped_regression_suite",
    "claim_reproduction_audit",
    "code_review_heuristics",
    "evaluator_critique"
  ],
  "harness_compliance_ok": true,
  "notes": "CRITERION-BY-CRITERION (all MET).\nC1 MET -- test_a_row_whose_period_ended_but_was_not_yet_filed_is_excluded is behavioural, not a scan. Proven by execution: M_STRIP (seam call deleted = literally the pre-82.51 `report_date <= cutoff` filter) and M_A (embargo arithmetic -> days=0) each turn it RED (5 failed / 7 passed).\nC2 MET -- I ran the mutant you did not: M_EXCLUDE_ALL (embargo=100000d) is RED, killed by test_a_row_that_was_demonstrably_public_is_still_included itself plus test_the_excluded_row_becomes_visible_once_the_embargo_has_elapsed. The fix cannot pass by excluding everything.\nC3 MET -- the one I attacked hardest. M_B (embargo moved INSIDE `if ticker in _fundamentals_full:`) is RED, killed by test_both_read_paths_agree_on_the_same_fixture and test_the_sql_path_binds_the_embargoed_cutoff_as_its_parameter. Your branch-3 stub is honest: fake_query filters with captured[\"params\"][\"cutoff\"], i.e. the value production actually bound, so under M_B it returned BOTH rows and the comparison broke -- a pre-filtering stub would have stayed green. One residual I measured: M_SQL (WHERE report_date <= @cutoff deleted from the SQL text) leaves test_both_read_paths_agree GREEN -- the stub cannot see a missing predicate -- and is caught ONLY by the separate assert on \"report_date <= @cutoff\" in the captured query. The pairing is sound; naming the actual killer per shape #11. (One mutant, M_B3 = SQL-path-only, was INCONCLUSIVE: my anchor string was not unique in cache.py so the child aborted before pytest. Its semantic content is covered from the other side by M_B and M_SQL; I am not claiming it as a kill.)\nC4 MET as specified (deltas + reproducing commands recorded). I did NOT re-run the two ~2min backtests -- re-running writes mda_cache.json and I am read-only -- so 4.4201 -> 3.7449 is UNREPRODUCED BY ME and rests on your capture. What I did verify: the script's window/strategy/params match the prose exactly (2025-06-30..2026-02-28, qarp, train 6 / test 2), and the env A/B lever genuinely works (FUNDAMENTALS_EMBARGO_DAYS=0 -> effective start 2024-06-30, _embargoed_cutoff('2025-07-05') == '2025-07-05'), so the \"before\" arm really is embargo-free through the same code. You flagged the one-window / 40-sample limit and the flat n_trades honestly -- checked, and the artifact does not present the flat trade count as confirmation.\nC5 MET -- sec.5 records the fixed-60-day choice, why option (b) is unavailable (filing_date lag identically 0; producer at data_ingestion.py:278), why 60 not 45 (10-K deadline on the 744-row/443-ticker FY-end cohort), the measured cost table, and the approximation caveat. The decoy test is NOT vacuous: both arms can fail (the else arm pytest.fail's if the producer stops copying) and its regex discriminates a real filing_date filter in cache.py -- its only limit is that it scans cache.py alone, so a filing_date filter added elsewhere would evade it (NOTE).\n\nYOUR THREE SPECIFIC CHALLENGES.\n(a) test_phase_82_12 NOT weakened -- I re-derived both lines from source myself: `str(r.get(\"report_date\", \"\")) <= cutoff_date` is at 647 and `str(entry[\"date\"]) > cutoff_date` is at 707, exactly the values you wrote. Verdicts stayed \"CORRECT\", only line numbers + explanatory why-text changed, and the file's 30 tests pass. This is the table's designed maintenance path.\n(b) paper_trader failure CONFIRMED pre-existing by a method independent of your worktree: I injected `git show HEAD:` copies of cache.py, fundamentals_coverage.py AND backtest_engine.py into sys.modules (asserting each HEAD copy is 82.51-free) and ran the single test -- identical `E assert None is not None`. Note your stated reason (\"none of my changed modules import paper_trader\") is weaker than your evidence: both backend.backtest.cache and backend.backtest.backtest_engine ARE in sys.modules during that test. The worktree / HEAD-injection result is what carries the claim, not the import argument.\n(c) Applying the embargo at the refusal site rather than inside window_is_covered DOES keep 82.21's semantics intact for the engine (its boundary test is green; full scoped suite green apart from the pre-existing failure) -- but it is precisely WHY finding #1 exists: moving the rule to one call site leaves the other call site (quant_optimizer.py:176) on the old semantics. The design choice is right; it is one call site short.\n\nCLAIM-REPRODUCTION AUDIT (NOTE level, nothing hidden).\n- sec.8 says lint ran on \"a derived 5-file scope\". My derivation -- `git diff --name-only HEAD -- '*.py'` union `git ls-files --others --exclude-standard -- '*.py'` -- yields 6 .py files (backtest_engine, cache, fundamentals_coverage, test_82_12, test_82_51, run_82_51_embargo_ab). I ran `uvx ruff check --select F821,F401,F811,F541,E9` over all 6 via xargs with a non-empty-set guard: \"All checks passed!\", exit 0. The number 5 does not reproduce; the outcome does.\n- sec.7 says M_B \"dies on three separate tests\". Mine dies on 2. Kill counts are construction-dependent (I stripped the top call and moved it inside branch 1), so this is not a defect -- but cite the named killers, not a count.\n- FUNDAMENTALS_EMBARGO_DAYS is read from os.getenv at import, so a stray env var would silently change a money-path filter in production. backend/.env does not set it and the shell did not have it set; test_the_embargo_is_sixty_days_and_the_reason_is_recorded would turn the suite red rather than let it drift. NOTE only.\n\nHARNESS COMPLIANCE (clean, 5/5). Research gate before contract: research_brief_82.51.md gate_passed true, external_sources_read_in_full 8 (>=5), urls_collected 37, recency_scan_performed true with a dedicated section 4, audit-class coverage.dry true, 12 internal files. Order by mtime: brief 17:46:01 < contract 17:47:27 < test 17:51:47 < cache.py 17:57:11 < experiment_results 18:00:50. experiment_results present. Log-last respected: no result= entry for 82.51 in harness_log.md and masterplan status is still \"pending\". Not a re-spawn (cycle 1, zero prior verdicts for this step-id) so the 3rd-CONDITIONAL rule does not bind -- this is the FIRST CONDITIONAL for 82.51. retry_count 0 < max_retries 3, so certified_fallback false. git status shows only the contracted files plus mda_cache.json (disclosed as a backtest side effect) and hook-written handoff audit JSONLs -- no unintended production change.\n\nTO CLEAR TO PASS (all three are small; none require re-running the backtests): pass start=effective_coverage_start() at quant_optimizer.py:176 (or record why the raw start is correct there) and widen the AST sweep to both modules; strengthen the refusal guard to assert the VALUE, or drive the refusal behaviourally; and disclose the data_server.py:149 live consumer in the artifact with an explicit decision on whether the 60-day embargo is intended on a live as-of-today query."
}
```

---

## Follow-up (Main) -- what changed before the fresh Q/A

All five criteria were already MET; the three findings were WARN-level and sit
outside the criteria. Every one is a real defect I introduced, and all three are
fixed. Full write-up in `experiment_results.md` §11.

1. **`quant_optimizer.py:176` left on the raw coverage start** -- I changed a
   shared predicate's effective meaning at one call site and never grepped the
   others, so the optimizer reported COVERED for windows the engine REFUSES.
   Fixed, plus a new consumer sweep that derives its own file set, asserts it
   non-empty, and checks the bound VALUE at every site.
2. **My refusal guard asserted the kwarg NAME, not its VALUE** -- the Q/A
   defeated it with `start=FUNDAMENTALS_COVERAGE_START` and showed the guard
   stayed green. Replaced with a behavioural guard that drives a window inside
   the embargo gap and asserts `ValueError: REFUSED`. The Q/A's own mutant
   (M_C2) now dies.
3. **`data_server.py:149` is a LIVE consumer** -- my change would have hidden the
   most recent published quarter from the live agent pipeline for 60 days, under
   an artifact section claiming no live change. **Decision recorded:** the
   embargo is wrong on an as-of-today query and is scoped out of it via
   `apply_embargo: bool = True`, with the live site passing `False`. The default
   stays True so a forgetful caller gets the safe path.

**Also fixed, from the Q/A's NOTE list:** the "5-file" lint scope (derived: 6,
now 8) and M_B's kill count (killers now named, not counted).

**Evidence changed since cycle 1:** `cache.py` (`apply_embargo`),
`quant_optimizer.py`, `data_server.py`, the test module (12 -> 17 tests),
`experiment_results.md` (§1 table, §2, §7 matrix, new §11). Verification command
re-run: **17 passed**. Mutation matrix re-run: **9 died, 0 survived**, now
including the Q/A's own M_C2 plus M_G and M_H for the two consumer findings.
Criterion 4's measurement was **re-run**, not assumed, because the signature
changed: identical at sharpe 3.7449 / 40 trades.

Per the cycle-2 flow, a FRESH Q/A grades this changed evidence.


---

## Cycle 2 -- verdict: CONDITIONAL

Fresh Q/A on changed evidence. 18 tool calls, 148,381 tokens, 460s.
Transcribed verbatim from the rail's output file.

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "All 5 immutable criteria MET, verified by EXECUTION not reading (verification command exit=0, 17 passed in 1.81s; ruff F821/F401/F811 clean on a git-derived 8-file scope; every changed backend module imports). The three cycle-2 WARN fixes are genuinely cured -- I built an isolated mutant tree in scratchpad (control GREEN, repo untouched) and ran 8 mutants. WARN-2 is proven: E1 `effective_start = FUNDAMENTALS_COVERAGE_START` -- the exact substitution that defeated the old name-only guard -- now DIES on 3 tests led by the behavioural `test_a_window_inside_the_embargo_gap_is_REFUSED_not_silently_uncovered`. WARN-1 is cured for its target shapes: Q1 (kwarg deleted) and Q2 (start=RAW VALUE at the optimizer) both DIE via the consumer sweep, whose file set I re-derived independently by grep and which agrees by MEMBERSHIP not just cardinality ({backtest_engine.py, quant_optimizer.py}). WARN-3's reasoning is RIGHT, not a silencing: at date.today() every row in the table is already published, so embargoing the live path suppresses real data rather than preventing look-ahead; the asymmetric default is correctly placed (D1 live-path-embargoes DIES; C1 default-flipped-to-False DIES on 6 tests). Criterion 4's re-measurement claim also checks out mechanically: historical_data.py:61 does call with the default, so an identical 3.7449/40 after the signature change is exactly what the change predicts. BUT the diff leaves TWO RED TESTS in the tree, neither disclosed in the artifact, both caused by the cycle-2 edits and both introduced AFTER the artifact's own measurements. (1) `test_phase_75_mcp_truth.py::test_prices_and_fundamentals_use_today_derived_cutoffs` fails: `_FakeCache.cached_fundamentals() got an unexpected keyword argument 'apply_embargo'` -- the WARN-3 fix broke a live consumer of the call shape, the SAME unswept-consumer class as WARN-1 recurring inside the fix for WARN-3, and data_server.py's broad except swallowed the TypeError into a logged error dict, so the live MCP fundamentals path degrades SILENTLY. (2) `test_phase_82_12_string_column_guards.py::test_classified_line_numbers_still_point_at_a_row_read` fails: \"cache.py:647 no longer points at a read of 'report_date'; actual lines [658]\" -- both entries Main edited are stale by exactly 11 lines (647 vs true 658; 707 vs true 718), the count added by the cycle-2 apply_embargo docstring, so section 9's \"re-derived all four from source\" was derived at cycle-1 state and never re-derived after the final edit. Consequently section 8's regression claim (\"1 failed, 274 passed ... the failure is [paper_trader]\") no longer reproduces. Both are mechanically fixable and neither touches an immutable criterion, so CONDITIONAL, not FAIL. This is the FIRST logged CONDITIONAL for 82.51 (harness_log has 0 `phase=82.51` result entries), so the 3rd-CONDITIONAL auto-FAIL rule does not fire. Named fixes: (a) make `_FakeCache.cached_fundamentals` accept `apply_embargo` (or **kwargs) in test_phase_75_mcp_truth.py and re-run; (b) set the two cache.py entries to 658 and 718 by re-deriving AFTER the last edit, never by arithmetic; (c) re-run and update section 8's regression number. Harness compliance clean: research gate_passed=true / 8 sources read in full / recency scan true; mtime order research 17:46 < contract 17:47 < test 18:13 < results 18:17; masterplan still `pending` so log-last holds; evidence materially changed since cycle 1 so this is the documented fresh-respawn, not verdict-shopping.",
  "violated_criteria": [
    "consumer-contract-break: cached_fundamentals kwarg breaks test_phase_75_mcp_truth live-path guard",
    "stale-figure-in-shipped-guard: test_phase_82_12 cache.py line entries 647/707 vs true 658/718",
    "Contradiction: experiment_results section 8 regression claim no longer reproduces after the cycle-2 edits",
    "WARN: consumer sweep is a source scan blindable by variable naming and by aliased import; sole coverage for quant_optimizer.py"
  ],
  "violation_details": [
    {
      "violation_type": "Invalid_Precondition",
      "action": "Added `apply_embargo=False` at backend/agents/mcp_servers/data_server.py:155 (the WARN-3 fix) without sweeping consumers of the cached_fundamentals CALL SHAPE",
      "state": "backend/tests/test_phase_75_mcp_truth.py::test_prices_and_fundamentals_use_today_derived_cutoffs is RED. Verbatim: `ERROR backend.agents.mcp_servers.data_server:data_server.py:164 Error fetching fundamentals for AAPL: test_prices_and_fundamentals_use_today_derived_cutoffs.<locals>._FakeCache.cached_fundamentals() got an unexpected keyword argument 'apply_embargo'` then `KeyError: 'fund_cutoff'` at test_phase_75_mcp_truth.py:323. Causation is certain: `apply_embargo` exists only in this diff. Aggravator: data_server.get_fundamentals' broad except turns the TypeError into a logged error dict, so a call-shape mismatch on the LIVE MCP path fails silently rather than loudly.",
      "constraint": "SEVERITY BLOCK -- consumer-contract-break [WARN escalating to BLOCK when a live unverified consumer is found]: a call-shape change must have every consumer grep-verified in the SAME diff. This is the identical defect class the cycle-1 Q/A raised as WARN-1, recurring inside the fix for WARN-3. Fix: accept `apply_embargo` (or **kwargs) in the _FakeCache double, then re-run."
    },
    {
      "violation_type": "Contradiction",
      "action": "experiment_results.md section 9 claims all four classified line numbers were re-derived from source: `OK backend/backtest/cache.py: 647 report_date actual=[647]` and `STALE ... 672 date actual=[707]`",
      "state": "The claim does not reproduce. test_phase_82_12_string_column_guards.py::test_classified_line_numbers_still_point_at_a_row_read FAILS: `AssertionError: backend/backtest/cache.py:647 no longer points at a read of 'report_date'; actual lines [658]`. I re-derived independently: the report_date read is at 658 (`filtered = [r for r in all_rows if str(r.get(\"report_date\", \"\")) <= cutoff_date]`) and the cached_macro date read is at 718 (`if str(entry[\"date\"]) > cutoff_date:`). Both table entries are off by exactly 11 -- the lines the cycle-2 apply_embargo docstring inserted above them. Line 647 is in fact a docstring line. Knock-on: section 8's `1 failed, 274 passed` regression capture is also stale.",
      "constraint": "SEVERITY WARN -- vacuity shape #10 (hand-derived-scope staleness: figures derived before the last edit) plus qa.md section 4b: a number in an artifact must reproduce under the command that produced it. Re-derive AFTER the final edit; never carry a figure forward by arithmetic. Fix: entries become 658 and 718, and re-run section 8's regression command to refresh its number."
    },
    {
      "violation_type": "Overgeneralization",
      "action": "Executed 2 blinding mutants against the new consumer sweep test_every_window_is_covered_consumer_judges_against_the_effective_start in an isolated tree",
      "state": "Q3 (introduce `effective_start = FUNDAMENTALS_COVERAGE_START` in quant_optimizer.py and pass `start=effective_start`) SURVIVED 17/17 -- the sweep's check is `if \"effective\" not in ast.unparse(start_kw.value)`, so a raw-valued variable merely NAMED effective passes. Q4 (alias the import to `window_is_covered as _wic`) SURVIVED 17/17 -- the file stops containing the literal `window_is_covered(`, drops out of the rglob-derived set, and the surviving backtest_engine.py alone keeps the non-empty guard satisfied. For backtest_engine.py these are harmless (the behavioural REFUSED test kills them), but for quant_optimizer.py the sweep is SOLE coverage: no test drives selectable_strategies_for_window behaviourally.",
      "constraint": "SEVERITY WARN, non-blocking -- qa.md 4c vacuity shapes #2 (source scan defeated by rewording) and #8. Non-blocking because the sweep is defence-in-depth for a WARN fix, not coverage for an immutable criterion, and it demonstrably kills both shapes it was built for. Named fix: give selectable_strategies_for_window the same behavioural treatment the engine got -- drive it at window_start 2024-07-01 and assert qarp is dropped from the pool."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "verification_command_exit_0_17_passed",
    "python_lint_gate_ruff_F821_F401_F811_derived_scope_8_files",
    "backend_runtime_smoke_imports_all_changed_modules",
    "independent_consumer_census_cached_fundamentals_and_window_is_covered",
    "sweep_file_set_re_derivation_membership_not_cardinality",
    "mutation_matrix_8_mutants_isolated_scratchpad_tree_with_control",
    "regression_scoped_pytest_237_tests",
    "line_number_re_derivation_from_source",
    "claim_audit_experiment_results_sections_3_8_9_11",
    "code_review_heuristics",
    "evaluator_critique_cycle1_read",
    "harness_log_conditional_count_escaped_dot",
    "research_gate_envelope_check"
  ],
  "harness_compliance_ok": true,
  "notes": "MUTATION MATRIX (isolated copy at scratchpad/mut; repo tree NEVER written; control GREEN 17 passed before every mutant; all restores from captured pristine bytes): Q1 optimizer kwarg deleted DIED (killer: consumer sweep) | Q2 optimizer start=RAW VALUE DIED (killer: consumer sweep) | Q3 var named effective holding RAW SURVIVED | Q4 aliased import SURVIVED | E1 engine start=RAW VALUE DIED, 3 killers led by test_a_window_inside_the_embargo_gap_is_REFUSED_not_silently_uncovered | D1 live path embargoes after all DIED (killer: test_the_live_data_server_does_NOT_embargo) | C1 default flipped to False DIED, 6 killers | C2 embargo neutered to the pre-82.51 report_date<=cutoff filter DIED, 5 killers. C2 is the criterion-1 proof: it reverts the fix to the exact filter the criterion names and criteria 1/2/3 tests all go red. C2 also killed test_the_sql_path_binds_the_embargoed_cutoff_as_its_parameter, which proves the BQ-fallback guard is genuinely driven by the production cutoff value and not by the stub -- criterion 3 is not testing its own fake.\n\nON WARN-3, ANSWERING THE DIRECT QUESTION: you did not silence my finding, you scoped it correctly. The live path asks \"what is true now\" at date.today(); the ingester only ever writes already-reported figures, so an embargo there hides the most recent published quarter for 60 days -- suppression, not leak prevention. Default True is the right asymmetry (the forgetful caller gets the safe path; opting out is deliberate) and C1 proves that default is behaviourally pinned, not merely signature-inspected. The census confirms only two non-test callers exist: data_server.py:155 (opted out, correct) and historical_data.py:61 (default, correct -- and this is also why the identical 3.7449/40 re-measurement is not just plausible but required). No third caller needs the opt-out. My objection is only to the IMPLEMENTATION seam: adding the kwarg broke the phase-75 guard's double.\n\nSKIPPED, DISCLOSED: (1) a HEAD-tree differential run -- my `git archive` + `rm -rf` command was denied by the sandbox, correctly, since I am read-only; causation for both failures is instead established deterministically from the error text, each naming a symbol that exists only in this diff (`apply_embargo`; line 647), plus the arithmetic 612+46=658 and 647+11=658. (2) I did not re-run the A/B backtest (~2 min/arm); I verified its mechanism instead. (3) Regression was the -k scoped 237-test subset, not the full backend/tests tree. (4) No UI claims in this step, so gate 1c does not apply.\n\nNOTE (non-blocking): backend/backtest/experiments/mda_cache.json shows 30 insertions / 30 deletions -- backtest-run churn, not source. It rides along with the auto-commit; worth a line in the artifact so a later reader does not mistake it for a code change.\n\nI did not re-litigate cycle 1's confirmed findings. The three WARNs are cured; the CONDITIONAL rests entirely on two reproducible red tests that the cycle-2 edits introduced and the artifact does not mention."
}
```

---

## Follow-up (Main) -- cycle 3 changes

Both blockers were red tests my cycle-2 edits introduced. Full write-up in
`experiment_results.md` §12.

1. **`test_phase_75_mcp_truth.py` `_FakeCache` took the old call shape.** Adding
   `apply_embargo=` broke it -- **the same unswept-consumer defect the cycle-1
   Q/A raised, recurring inside the fix for it.** I swept consumers of
   `window_is_covered` because I was told to, and did not sweep consumers of the
   function whose signature I was changing in the same edit. Double now accepts
   the real shape and records the flag. The Q/A's aggravating observation stands
   and is recorded: `data_server`'s broad `except` swallowed the `TypeError`, so
   a live call-shape mismatch degrades silently.
2. **The 82.12 line entries were stale again** (647/707 -> 658/718): §9's
   "re-derived from source" was true at cycle-1 state, and my cycle-2 docstring
   moved them 11 lines. Re-derived **programmatically as the last action**, with
   the rewrite asserting it is not a no-op.
3. **Whole-suite run instead of a `-k` subset** -- which is what let both through.
   32 failures, diffed against a HEAD worktree: 29 pre-existing. Of the other 3,
   exactly **one** was a real consequence (`test_phase_82_46`'s boundary
   assertion, updated to `effective_coverage_start()` with the raw start pinned
   on the other side). The remaining 2 are environment: they read gitignored
   `handoff/logs` and passed vacuously in a log-less worktree. Final: **31 = 29
   pre-existing + 2 environment, zero code regressions.**
4. **The consumer sweep, blinded twice by the Q/A, is hardened** (file set from
   imports under any alias; bound value resolved through assignments) and is no
   longer sole coverage -- `selectable_strategies_for_window` is now driven
   behaviourally, the Q/A's named fix.

**Evidence changed:** `test_phase_75_mcp_truth.py`, `test_phase_82_12_...py`,
`test_phase_82_46_...py`, the 82.51 module (17 -> 18 tests),
`experiment_results.md` (new §12). Verification command: **18 passed**. Targeted
suite across all five affected files: **112 passed**. Lint over a 10-file derived
scope: `All checks passed!`.

Per the cycle-2 flow, a FRESH Q/A grades this changed evidence. **Note for that
Q/A: this is the third cycle. Two consecutive CONDITIONALs are logged, so per the
3rd-CONDITIONAL rule a further CONDITIONAL must be returned as FAIL.**


---

## Cycle 3 -- verdict: PASS

Fresh Q/A on changed evidence. 26 tool calls, 133,096 tokens, 608s.
Transcribed verbatim from the rail's output file.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 5 immutable criteria MET and independently reproduced; harness compliance clean; no unintended production change. Deterministic: immutable command `pytest backend/tests/test_phase_82_51_fundamentals_embargo.py -q` = 18 passed, exit 0 (18 `def test_` in the module). The two cycle-2 RED files are GREEN: test_phase_75_mcp_truth.py + test_phase_82_12_string_column_guards.py + test_phase_82_46_trial_pool_composition.py = 72 passed. I re-derived the cache.py classified lines MYSELF rather than trusting the claim: line 658 = `filtered = [r for r in all_rows if str(r.get(\"report_date\", \"\")) <= cutoff_date]`, line 718 = `if str(entry[\"date\"]) > cutoff_date:` in cached_macro -- both match the pinned 658/718 exactly. Ruff F821,F401,F811,E9 over a git-derived, non-empty, xargs-NUL-quoted 10-file scope: \"All checks passed!\" exit 0. All 5 changed backend modules import; effective_coverage_start()=2024-08-29 (= raw 2024-06-30 + 60d); signature is `cached_fundamentals(ticker, cutoff_date, apply_embargo: bool = True)` (default protected). THE PARTITION CLAIM HOLDS AND I ATTACKED IT: my own full run reproduces exactly `31 failed, 2780 passed, 12 skipped, 5 xfailed, 1 xpassed in 314.39s`, and -- stronger than a count match -- of the 21 distinct failing test FILES, ZERO reference any of the 5 changed production modules (grep for backtest.cache|backtest.quant_optimizer|backtest.backtest_engine|fundamentals_coverage|mcp_servers.data_server returned no intersection), so no failure can be a regression from this diff. The two \"environment\" failures are confirmed environment, not yours: watchdog fails with `AssertionError: watchdog log stale: latest entry 2026-08-04T18:11:23+00:00 is 46.7h old (max 24h)` and sector_cap with `no 'Skipping BUY' line in backend.log OR its newest archive` -- both files self-document as asserting LIVE machine state (phase-75.15 qa-tests-01), handoff/logs holds 46 files here vs 0 in a gitignored worktree exactly as you described, and neither is reachable from a fundamentals-embargo change. GUARD-VACUITY: the 82.46 boundary edit is a STRENGTHENING, not a weakening -- it preserves the original intent (full pool at a DERIVED boundary, moved to effective_coverage_start()) and ADDS a second assertion pinning the raw start on the excluded side. MUTATION (mine, in-memory sys.modules injection, zero writes -- `git diff --stat` on quant_optimizer.py unchanged after): CONTROL 32 passed RC=0 (harness sound); MUTANT reverting `window_is_covered(window_start, start=effective_coverage_start())` to `window_is_covered(window_start)` = 2 failed -- `test_the_optimizer_drops_a_dependent_strategy_inside_the_embargo_gap` KILLED, and the 82.46 raw-start assertion also killed it with `assert set() == {'qarp'}`. The new behavioural test is genuinely behavioural and cannot be reworded around. Four NOTE-level findings recorded in notes (PASS-with-flag), none degrading the verdict; N1 (stale §1/§2 figures) should be regenerated before the harness_log append.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit",
    "verification_command",
    "regression_partition_audit_full_suite",
    "import_intersection_scope_derivation",
    "independent_line_rederivation_cache_py",
    "independent_mutation_test_control_and_mutant",
    "guard_vacuity_check_82_46_boundary",
    "ruff_lint_gate_derived_scope",
    "backend_runtime_smoke_imports",
    "contract_completeness_mapping",
    "claim_auditing_4b",
    "code_review_heuristics",
    "evaluator_critique"
  ],
  "harness_compliance_ok": true,
  "notes": "CRITERION MAP (each MET, with the evidence I verified myself): (1) test_a_row_whose_period_ended_but_was_not_yet_filed_is_excluded + test_the_excluded_row_becomes_visible_once_the_embargo_has_elapsed; the \"fails against the current report_date <= cutoff filter\" half is carried by mutant M_A (embargo no-op) dying. (2) test_a_row_that_was_demonstrably_public_is_still_included; the cannot-pass-by-excluding-everything half was proven by the cycle-1 Q/A's own embargo=100000d mutant. (3) test_both_read_paths_agree_on_the_same_fixture + test_the_sql_path_binds_the_embargoed_cutoff_as_its_parameter + test_mutating_the_shared_seam_moves_BOTH_paths (M_B, the branch-1-only mutant, is the one that matters and dies). (4) experiment_results §3 records verbatim commands and Sharpe 4.4201 -> 3.7449 (-0.6752), n_trades 40 -> 40 (0), with honest limits stated (one walk-forward window, 40 samples, n_trades bound by max_positions); §11.4 re-measured after the signature change; cycle-3 edits were test-only so the measurement is not stale. (5) §5 records the fixed 60-day embargo AND its reason -- option (b) is refuted (filing_date is a verbatim copy of report_date per data_ingestion.py:278, measured lag identically 0 over 4798 rows), and 60 is justified by the 10-K large-accelerated deadline over the 744-row/443-ticker FY-end cohort; pinned by test_the_embargo_is_sixty_days_and_the_reason_is_recorded (M_D 60->45 dies).\n\nHARNESS COMPLIANCE (5/5): research_brief_82.51.md gate_passed=true, 8 sources read in full (>=5), recency scan performed, audit-class with coverage.dry=true after 6 rounds / 2 dry. mtime chain is correct: brief 17:46:01 < contract_82.51.md 17:47:27 < production code 18:14 < test module 18:27. experiment_results.md present with a §12 cycle-3 section. LOG-LAST intact: masterplan 82.51 status is still \"pending\" and harness_log.md has no `result=` entry for 82.51 (only a \"Queued\" mention at line 31032). NO VERDICT-SHOPPING: evidence materially changed since cycle 2 (3 test files edited, an 18th behavioural test added, new §12) -- I reproduced the greens myself rather than reading the claim, so this reversal is the documented cycle-3 flow, not sycophancy-under-rebuttal. Contract §4 quotes all five criteria word-for-word (an automated substring test returned False purely from markdown line-wrapping; verified by reading lines 98-112).\n\nNOTE-1 (stale \"verbatim\" figures -- fix before the harness_log append). §2 is titled \"Verbatim output of the immutable verification command\" but shows `17 passed in 1.41s`; the command now emits `18 passed in 1.79s`. §1's table likewise says \"new -- 17 tests | 429 (new)\" while the file is 485 lines with 18 tests. §12.5 \"Corrected figures\" refreshed the file count and superseded §8 but missed these two -- the third stale-figure instance inside this one step, and the same class §12.2 says recurred \"a second time, for the same reason\". Held at NOTE rather than blocking because the direction UNDERSTATES the work, no criterion depends on the count, and I reproduced the underlying command green at 18/18 -- but §1/§2 must be REGENERATED, not hand-edited (a hand-edit in a block labelled verbatim is the exact Invalid_Precondition shape qa.md §4b names).\n\nNOTE-2: backend/backtest/experiments/mda_cache.json (+30/-30) is a tracked file changed as a side effect of the required backtests. Disclosed in §1, so honest, but it will ride the auto-commit under this step's name -- worth one line in the harness_log entry.\n\nNOTE-3 (queue it, do not just disclose it): the live-path defect you found in §12.1 -- data_server.get_fundamentals wraps the call in a broad `except`, so a call-shape TypeError on the live MCP path degrades silently into a logged error dict -- is prose-only. I walked .claude/masterplan.json: no step covers it (82.9 is a DIFFERENT data_server defect, the `as_of: today` macro stamping). The standing operator rule is that every out-of-scope defect gets its OWN research-gated step. The tree is correctly frozen during EVALUATE, so this belongs in the next cycle, not this one -- 82.45's \"queued residuals\" shape is the precedent.\n\nNOTE-4: two inline comments in test_phase_82_12_string_column_guards.py rotted relative to the values they annotate -- \"moved this read from 612 to 647\" sits above a pinned 658, and \"+35 positional shift\" sits above a pinned 718 (672 -> 718 is +46). The VALUES are correct (I re-derived both from source independently); only the prose rationale is stale. Cosmetic, same class as NOTE-1.\n\nCODE-REVIEW HEURISTICS (5 dimensions evaluated, no BLOCK/WARN): no secret-in-diff; kill-switch/stop-loss/perf-metrics paths untouched; financial-logic-without-behavioral-test satisfied (backtest_engine.py changed WITH 18 behavioural tests plus a 9-mutant matrix); consumer-contract-break NOT fired -- the cached_fundamentals change is additive (optional kwarg WITH default), the one double taking the old shape was updated and is green, and window_is_covered's consumers are swept by a test that derives its own file set from IMPORTS under any alias and resolves the bound VALUE through local assignments; illusory-guard NOT fired -- I mutated the guard the author defended and it died. The refusal message now names the embargo and the effective start, and the availability record ADDS keys (fundamentals_embargo_days, fundamentals_effective_coverage_start) rather than replacing 82.21's record, which is what the step asked for.\n\nMETHOD DISCLOSURE: mutation was executed via in-memory sys.modules injection (module compiled from a string, never written to disk), with the anchor asserted to match exactly once before mutating and a CONTROL run first to prove the harness itself does not cause the failure; git diff --stat on quant_optimizer.py was re-checked after and is unchanged. I did NOT re-run a HEAD-baseline worktree for the \"29 pre-existing\" half -- that requires a checkout I am barred from -- so I verified the partition structurally (zero import-intersection between the 21 failing test files and the 5 changed modules) plus verbatim live-state assertion text on the 2 environment failures, which is a stronger argument than a count match and reaches the same conclusion: zero code regressions. No UI claims in this step, so §1c did not apply. Per instruction, cycles 1-2 were not re-litigated and nothing was sent outward."
}
```

---

## Follow-up (Main) -- the four PASS-with-flag NOTEs, all actioned

- **NOTE-1 (stale §1/§2 figures -- the THIRD instance in this step).** §2 was
  labelled "verbatim" while showing `17 passed`; §1 said 17 tests / 429 lines
  against 18 / 485. **Regenerated by capturing the live command's output and
  `grep -c "^def test_"` / `wc -l`, not hand-edited** -- a hand-edit inside a
  block labelled verbatim is the exact shape the Q/A named.
- **NOTE-2 (`mda_cache.json` rides the commit).** Disclosed in §1 as a backtest
  side effect and now also in the harness_log entry.
- **NOTE-3 (queue, do not merely disclose).** The `data_server` broad-except
  masking is filed as **82.62** (P2, research-gated, 4 criteria), written for an
  executor with no memory of the discovery and cross-referencing 82.9 so the two
  data_server defects are not conflated.
- **NOTE-4 (rotted inline comments).** The two `test_phase_82_12` comments said
  "612 to 647" and "+35 shift" above pinned values of 658 and 718. Values were
  correct; the prose was stale. Corrected to describe both shifts and to state
  the lines were re-derived by script after the final edit.

Verification command after these edits: **18 passed**. `test_phase_82_12`: 30
passed.
