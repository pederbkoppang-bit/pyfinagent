# Evaluator Critique -- Step 75.3 (Q/A verdict, cycle 1)

- **Launch:** `.claude/workflows/qa-verdict.js`, run `wf_fcf4f363-339` (Opus 4.8 / effort max), 2026-07-20.
- **Transcription rule:** Main records the verdict, never authors it. VERBATIM below.

## Verdict (verbatim captured return value)

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "All 6 immutable criteria are met in SUBSTANCE and independently re-verified by me end-to-end; production logic is sound, the BOUNDARY held byte-exact (get_risk_constraints identical to HEAD, no diff line touches any threshold constant or comparison operator), no consumer contract broke, and harness compliance is clean. Capped at CONDITIONAL because the step's deliverable IS a regression suite and two criteria are pinned only by proxy assertions that a realistic regression would evade: criterion 6's 'SecretStr reaches WebClient unwrapped' is guarded by `\"unwrap_secret\" in SIGNALS_SRC`, but that string occurs on BOTH the import line (:495) and the call site (:497), so reverting only the call site leaves the test green while security-05 fully regresses; and criterion 3's two behavioral halves ('re-fired after eviction reports published=false', 'a freed-up rejection can be retried') are never exercised through publish_signal, resting instead on an exact-string source scan that a reworded regression evades. I verified both behaviors myself with independent probes (WebClient captured 'xoxb-real-token'; re-fire after eviction -> published=False; funded book -> published=True/trade_executed=True), so nothing is broken -- but the guards are illusory in a step whose own thesis is that the pre-existing tests 'assert envelope shape rather than outcome, which is why all of this shipped'. Fix is test-only and additive; no production change required.",
  "violated_criteria": [
    "criterion_6_secretstr_reaches_webclient_unwrapped (guard illusory, behavior correct)",
    "criterion_3_evicted_refire_and_retry (behavioral halves unasserted, behavior correct)"
  ],
  "violation_details": [
    {
      "violation_type": "Missing_Assumption",
      "action": "test_secretstr_slack_token_reaches_webclient_unwrapped asserts unwrap_secret() in isolation plus `\"unwrap_secret\" in SIGNALS_SRC`",
      "state": "'unwrap_secret' occurs twice in signals_server.py -- import line :495 and call site :497. Reverting ONLY :497 to `slack_token = getattr(self.settings, 'slack_bot_token', '')` leaves the import (and comment) in place, so the source-scan assertion stays True and the test stays GREEN while security-05 is fully regressed. No assertion in the suite observes what WebClient actually receives; I had to write a fake-WebClient probe to establish the criterion's claim (captured token == 'xoxb-real-token', so the behavior IS correct today).",
      "constraint": "Criterion 6: 'a SecretStr slack token reaches WebClient as its unwrapped plain-string value' requires an assertion that fails when the unwrap at the SDK boundary is removed. FIX: monkeypatch slack_sdk.WebClient, publish a signal with settings.slack_bot_token = SecretStr('xoxb-...'), assert the captured token == 'xoxb-...' (and != '**********')."
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "criterion 3 covered by test_evicted_rejection_never_replays_as_published (asserts only that _recent_responses.get('sig-1') is None after eviction), test_remembered_rejection_replays_the_true_outcome (asserts the dict entry), and test_no_synthesized_published_true_in_source (exact-string scan for `resp[\"published\"] = True`)",
      "state": "No test re-fires a signal through publish_signal after eviction, and none exercises the retry half. The source scan is an exact-literal match that these regressions evade: `resp[\"published\"]=True` (no spaces), `resp.update({\"published\": True})`, or the same assignment under a different variable name. I verified both halves end-to-end independently: rejected BUY -> forced eviction -> re-fire returns published=False ('risk_rejected:insufficient_cash'), and the same signal on a funded book returns published=True/trade_executed=True. Behavior is correct; only the guard is weak.",
      "constraint": "Criterion 3: 'Test asserts a previously-rejected signal_id re-fired after cache eviction reports published=false (never a synthesized published=true), and a freed-up rejection can be retried'. FIX: add a behavioral test that calls publish_signal twice around a forced eviction and asserts published is False on the re-fire, plus one that publishes successfully after the blocking condition clears."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "research_gate_envelope (gate_passed=true, 7 sources read in full vs floor 5, 27 URLs vs floor 10, recency scan performed)",
    "mtime_ordering (research 11:48:03 < contract 11:49:54 < production 11:53:19-11:54:52 < test 12:00:54 < experiment_results 12:01:38 < live_check 12:02:02)",
    "log_last (no 75.3 result entry in harness_log; masterplan status=pending)",
    "no_verdict_shopping (cycle 1, retry_count=0, no prior 75.3 verdict)",
    "immutable_verification_command (24 passed, bare PYTEST_EXIT=0)",
    "ruff_lint_gate_F821_F401_F811 (exit 1; sole finding F401 pathlib.Path proven byte-identical PRE-EXISTING at HEAD)",
    "backend_runtime_smoke (all 3 changed modules import in venv, exit 0 each)",
    "pre_existing_suites (tests/test_mcp_servers.py + test_mcp_integration.py: 22 passed, 1 failed on missing pytest-asyncio plugin -- environmental, diff-independent)",
    "go_live_drill_slack_signals_e2e (16/16 PASS, S9-S14 AST checks survive the publish_signal restructure)",
    "go_live_drill_first_week_monitoring (15/15 PASS, incl. S7/S14 get_risk_constraints unchanged + _peak_equity seam preserved)",
    "go_live_drill_kill_switch (4/4 PASS, threshold -15.0, inclusive boundary, SELL always permitted)",
    "go_live_drill_position_limits (BLOCKED by pre-existing site-packages shadowing at line 20, byte-identical to HEAD, upstream of every 75.3 hunk)",
    "boundary_threshold_proof (get_risk_constraints byte-identical to HEAD via diff; zero diff lines touch any threshold constant or comparison operator)",
    "consumer_contract_live_execution (emit_candidates: candidates_per_call=5, dsr_annotated=True, all stub:true -- exact mcp_ab_test.py:484-490 assertions executed, not trusted)",
    "consumer_removal_scan (dsr/run_id/num_trades/timeout_seconds have zero consumers repo-wide)",
    "mutation_resistance_source_scans (all 4 scanned strings PRESENT at HEAD and absent now -> every source-scan test proven discriminating)",
    "independent_probe_fail_closed_routes (traced zero-equity BUY -> size_position equity<=0 guard -> cash gate rejects; get_or_create_portfolio always returns a dict)",
    "independent_probe_dedup_end_to_end (re-fire after eviction -> published=False; freed-up rejection retryable)",
    "independent_probe_secretstr_to_webclient (fake WebClient captured 'xoxb-real-token')",
    "independent_probe_drawdown_conflict_reason (conflicts == ['drawdown_circuit_breaker'])",
    "sell_exemption_analysis (all 4 notional gates BUY-guarded; SELL bounded by shares-based insufficient_position)",
    "scope_check (change surface exactly matches contract; 2 other modified files are generated/hook artifacts, not production logic)",
    "spec_correction_legitimacy_review",
    "test_vacuity_review (all 24 tests individually examined)"
  ],
  "harness_compliance_ok": true,
  "notes": "CALLER'S PROBES ANSWERED. (a) Fail-closed HOLDS -- no route to a trade found. Both degraded paths return stub:true AND zeroed; publish_signal refuses at step 5 before sizing. Defense-in-depth confirmed: even bypassing the marker, size_position's equity<=0 guard returns 0.0 and the cash gate rejects a real-priced BUY. get_or_create_portfolio always returns a dict, so the `or {}` unmarked-zero path is unreachable in practice. (b) gap4-06 CANNOT report published=true for a rejection -- verified structurally (_seen_signal_ids has 0 residual refs; no literal published=True assignment anywhere in the file; replay is a verbatim dict copy; cache miss falls through to re-execution) AND behaviorally end-to-end. (c) The SELL exemption is CORRECT, not a hole -- all four notional gates (per-ticker, total-exposure, cash, drawdown) are `action == \"BUY\"`-guarded, so proposed_notional is dead code for SELLs, and the SELL path is bounded by the shares-based insufficient_position check. Rejecting SELLs on a missing mark would TRAP EXITS; kill_switch drill S4 independently pins \"SELL de-risking always permitted\". Your reasoning was right. (d) emit_candidates is genuinely ADDITIVE -- verified by LIVE EXECUTION of the exact consumer assertions rather than trusting the claim: 5 candidates, all carrying dsr, all stub:true. The consumer uses len() and `all(\"dsr\" in c)` with no key-set equality, and emit_candidates has exactly one consumer that never feeds publish_signal, so the stub_provenance refusal has zero blast radius. (e) Tests are real, not vacuous -- all four source-scan strings were PRESENT at HEAD and are absent now, so each would have failed pre-fix. The two proxy tests above are the exception.\n\nBOTH SPEC CORRECTIONS ARE LEGITIMATE -- you did not move a goalpost. The path correction is directory-only (scripts/mcp_servers/ holds only smoke-test scripts; .mcp.json launches the real servers from backend/agents/mcp_servers/) with line numbers intact. On BacktestResult: dropping the dsr key rather than emitting a fabricated 0.0 is not merely acceptable, it is the ONLY choice consistent with the step's thesis -- emitting 0.0 would recreate the exact fault class being removed. No immutable criterion names those fields; criterion 6 requires extraction \"without AttributeError\", which is satisfied and tested. Adding test_backtest_result_lacks_the_fields_the_spec_named so the correction cannot be silently undone is good practice. I also confirmed the removed keys (dsr, run_id, num_trades) and timeout_seconds have ZERO consumers repo-wide, so the response-shape change breaks nothing.\n\nTWO PRE-EXISTING FAILURES, NEITHER ATTRIBUTABLE TO 75.3, both proven rather than assumed: (1) ruff F401 pathlib.Path at signals_server.py:29 reproduces byte-identically on `git show HEAD:` content -- your disclosure was accurate. (2) go_live_drills/position_limits_test.py cannot load: a stale `backend` package in .venv/lib/python3.14/site-packages/backend/ (dated 17 apr) shadows the repo's backend.utils under the drill's importlib spec-loader, failing at signals_server.py:20 -- a line byte-identical to HEAD and upstream of every 75.3 hunk, so module execution never reaches your code. Worth a separate housekeeping ticket; it means the drill that pins the strict-`>` boundary is currently unrunnable. I substituted three independent threshold proofs (byte-diff of get_risk_constraints, first_week S7/S14, kill_switch 4/4). (3) tests/test_mcp_integration.py::test_all_servers_startup fails with \"async def functions are not natively supported ... need pytest-asyncio\" -- a missing plugin, diff-independent; I separately proved create_signals_server() works by executing emit_candidates through it. Note you ran only test_mcp_servers.py, not test_mcp_integration.py.\n\nADDITIONAL OBSERVATIONS (not criterion-covered, not blocking): (i) The new track_drawdown try/except is fail-OPEN -- if it raises, current_drawdown_pct is never set and the drawdown breaker silently reverts to the inert state gap4-05 exists to fix. The comment (\"never block publishing on telemetry\") makes it deliberate, but it is a residual fail-open seam in a fail-closed step; consider logging at ERROR or refusing BUYs when the drawdown read fails. (ii) test_drawdown_breaker_blocks_buys asserts only `allowed is False` without pinning the conflict -- it would stay green if a different gate did the rejecting (I confirmed conflicts == ['drawdown_circuit_breaker'], and kill_switch covers it independently). (iii) test_seen_and_outcome_evict_together's second clause `not hasattr(...) or not srv._seen_signal_ids` short-circuits to True and is tautological; the len()==3 assertion carries that test. (iv) The _server() test helper uses SignalsServer.__new__, bypassing __init__, so no test verifies that __init__ actually builds an OrderedDict -- a regression re-introducing the paired set/dict in __init__ would be caught only by the brittle source scan. (v) Two non-scoped files are modified but are NOT production logic: .claude/.archive-baseline.json (archive-hook bookkeeping adding \"75.2\") and backend/backtest/experiments/mda_cache.json (regenerated at 12:00:41 by the un-mocked pre-existing suite running a real backtest -- itself evidence for your mocking-discipline argument).\n\nPATH TO PASS: fix the two proxy tests (F1, F2) -- both are additive, test-only, no production change -- then update experiment_results.md/live_check_75.3.md and spawn a FRESH Q/A on the changed evidence per the canonical cycle-2 flow. This is cycle 1 (retry_count=0, zero prior 75.3 entries in harness_log), so the 3rd-CONDITIONAL auto-FAIL rule is not engaged."
}
```

---

## Follow-up (Main, cycle 2 -- 2026-07-20)

Both findings accepted. The Q/A confirmed the production behavior is correct on both
counts (it wrote its own probes and observed WebClient receiving `xoxb-real-token`, and
drove an evicted re-fire end-to-end), so **no production code changed in this cycle** --
the defect was that my *guards* were illusory. That distinction matters, and the critique
lands hard precisely because this step's own thesis is that the pre-existing suite
"asserts envelope shape rather than outcome, which is why all of this shipped". I shipped
the same mistake one layer up, in the regression suite that *is* this step's deliverable.

### Fix (test-only, additive)

1. **criterion 6** -- new `test_secretstr_token_reaches_webclient_unwrapped_end_to_end`
   monkeypatches `slack_sdk.WebClient`, publishes a real signal with
   `settings.slack_bot_token = SecretStr("xoxb-real-token")`, and asserts on the token the
   client actually received (`== "xoxb-real-token"`, `!= "**********"`, not a SecretStr).
2. **criterion 3** -- two new end-to-end tests driven through `publish_signal` rather than
   inspecting the cache dict: a rejected BUY, forced eviction, then a re-fire that must
   report `published=False` with `execute_buy` never called; and a rejection that becomes
   publishable once the blocking condition clears (`published=True`, `execute_buy` called
   exactly once).

### Mutation-tested, not just written

A guard that cannot fail is worth nothing, so I verified each new test detects the exact
regression the Q/A described:

```
MUTATION 1 -- revert ONLY the unwrap call site, leave the import in place
  ("unwrap_secret" still appears in source, so the OLD source-scan guard stays green)
  => FAILED test_secretstr_token_reaches_webclient_unwrapped_end_to_end   [caught]

MUTATION 2 -- reinstate the synthesized published=True on cache miss, REWORDED as
  resp.update({"published": True, ...}) so the exact-string scan cannot see it
  grep 'resp["published"] = True' -> 0 matches  (old source-scan guard stays GREEN)
  => FAILED test_evicted_refire_of_a_rejection_reports_published_false_end_to_end
     FAILED test_a_freed_up_rejection_can_be_retried_end_to_end            [caught]

Both mutations reverted; suite restored to 27 passed.
```

Mutation 2 is the important one: it confirms the Q/A's specific claim that a reworded
regression slips past the source scan, and that only the behavioral tests catch it.

The original source-scan tests are **kept alongside** the new behavioral ones. They are
cheap, and they pin the *absence* of the old code shape even though they cannot pin
behavior; the behavioral tests are what actually guard the criteria.

Suite: 24 -> 27 tests. No production diff in this cycle. A FRESH Q/A is spawned on the
changed evidence per the canonical cycle-2 flow.

---

## Cycle-2 Q/A verdict (verbatim captured return value)

- **Launch:** run `wf_a66a87f0-756` (Opus 4.8 / effort max), 2026-07-20.

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "Both cycle-1 blockers are genuinely fixed and I re-proved it myself rather than trusting the transcript: I ran four independent mutations through a read-only module-injection harness (mutated copies written to scratchpad; the repo was never touched — git status is byte-identical before and after). Reverting ONLY the unwrap call site while leaving the import fails the new SecretStr test and nothing else (1 failed / 26 passed), with the old proxy guard staying green and the captured token printing literally as SecretStr('**********'); a reworded fabricated success on the risk-rejection path fails both new dedup tests (2 failed / 25 passed) with the exact-literal source scan finding 0 matches; and a faithful gap4-06 regression (un-evicted seen-set + resp.update({\"published\": True}) on seen-but-not-cached) reproduces Main's reported 2 failed / 25 passed on exactly the two named tests, again invisible to the grep. Criteria 3 and 6 are now behaviorally guarded. Capped at CONDITIONAL because probe (d) turned up a THIRD instance of the same defect class that cycle 1 did not reach: criterion 4's first clause, \"every emit_candidates candidate carries stub:true\", is still pinned only by a source-literal scan. I mutated emit_candidates to keep the `\"stub\": True` literal in the dict while popping the key off every candidate before return — all 27 tests stayed GREEN, and I confirmed the payload really regresses by driving the tool through create_signals_server()+fastmcp Client (production: n=5, all_stub_true=True; mutated: all_stub_true=False, key absent). Behavior is correct today on every criterion and no production code needs to change; the gap is the guard, and the fix is ~10 offline lines using the same async client path mcp_ab_test.py already uses. Deterministic checks all clean: verification command bare exit 0 (27 passed), production diff 222 insertions / 115 deletions across the 3 servers exactly as claimed, get_risk_constraints byte-identical to HEAD (independently recomputed), all 3 modules import in the venv, ruff's sole finding proven byte-identically pre-existing at HEAD, harness compliance clean, and this is the 2nd CONDITIONAL for 75.3 so the 3rd-CONDITIONAL auto-FAIL rule is not engaged.",
  "violated_criteria": [
    "criterion_4_emit_candidates_carries_stub_true (guard illusory, behavior correct)"
  ],
  "violation_details": [
    {
      "violation_type": "Missing_Assumption",
      "action": "test_emit_candidates_payload_is_stub_marked_and_keeps_consumer_contract splits SIGNALS_SRC on 'def emit_candidates' and asserts the substrings '\"stub\": True', 'PENDING_IMPLEMENTATION', '\"dsr\": dsr' and the n=max(5,...) regex — a source-literal scan, not an assertion over the emitted candidates",
      "state": "MUTATION (run by me, read-only): leave the '\"stub\": True' literal in the candidate dict untouched and add `for _c in candidates: _c.pop(\"stub\", None)` immediately before the return. Full suite result: 27 passed — the guard does not discriminate. I then executed the tool itself through create_signals_server() + fastmcp Client on both trees: production -> n=5, all_dsr=True, all_stub_true=True, candidate keys include 'stub'; mutated -> n=5, all_dsr=True, all_stub_true=False, 'stub' absent from every candidate. So criterion 4's first clause fully regresses with the suite 100% green. The behaviour is CORRECT in the shipped code (5/5 candidates carry stub:true) and publish_signal's stub_provenance refusal is separately covered by a real behavioural test, so the blast radius is bounded — emit_candidates has one consumer (scripts/harness/mcp_ab_test.py) that never feeds publish_signal. This is the third instance of the exact defect class this cycle set out to remove, and the one criterion whose wording attaches the '(source scan)' qualifier only to the OTHER clause (the compute_dsr_real branch), which reads as a deliberate distinction by the drafter.",
      "constraint": "Criterion 4: 'Test asserts every emit_candidates candidate carries stub:true'. Requires an assertion that fails when an emitted candidate lacks stub:true. FIX (test-only, additive, offline, ~1s — I verified this exact shape runs): async def _pull(): mcp = create_signals_server(); async with Client(mcp) as c: r = await c.call_tool('emit_candidates', {'ticker':'AAPL','n':5}); d = r.data if hasattr(r,'data') else r; return d.get('candidates') or []  ->  cands = asyncio.run(_pull()); assert len(cands) >= 5; assert all('dsr' in x for x in cands); assert all(x.get('stub') is True for x in cands). Keep the existing source-scan test alongside it, as was done for criteria 3 and 6."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "research_gate_envelope (gate_passed=true, 7 sources read in full vs floor 5, 27 URLs vs floor 10, recency scan present at section 4, audit_class=false so coverage.dry not required)",
    "mtime_ordering (research 11:48:03 < contract 11:49:54 < production 11:53:19+; cycle-2 touch of signals_server at 12:20:28 accounted for as mutation-and-revert, not new authoring)",
    "log_last (grep 'phase=75.3 result=' handoff/harness_log.md -> 0 entries; masterplan status=pending, retry_count=0)",
    "no_verdict_shopping (evidence CHANGED: suite 24 -> 27, three new tests, experiment_results cycle-2 addendum, live_check mutation section, cycle-1 verdict preserved verbatim + Follow-up)",
    "third_conditional_rule (0 prior result= entries for 75.3; this is the 2nd CONDITIONAL -> auto-FAIL not engaged)",
    "immutable_verification_command (bare exit 0, 27 passed)",
    "production_diff_fingerprint (numstat 222 insertions / 115 deletions across the 3 servers = cycle-1 figure exactly; unwrap_secret at :495 import / :497 call site = cycle-1's quoted line numbers exactly)",
    "boundary_threshold_proof (get_risk_constraints extracted from git show HEAD: and from the worktree — byte-identical, len 1216 == 1216)",
    "mutation_residue_scan (grep 'resp.update(|_seen_signal_ids|published\": True|if False and' on the 3 servers -> 0 matches)",
    "ruff_lint_gate_F821_F401_F811 (exit 1; sole finding F401 pathlib.Path:29 independently reproduced on `git show HEAD:` content, HEAD_RUFF_EXIT=1, and the diff touches no hunk near line 29 -> pre-existing, diff-independent)",
    "backend_runtime_smoke (all 3 changed modules import in the venv, exit 0 each)",
    "MUTATION_1_independent (revert unwrap call site only, import kept -> 1 failed / 26 passed; ONLY test_secretstr_token_reaches_webclient_unwrapped_end_to_end fails; captured token == SecretStr('**********'); old proxy guard stays GREEN)",
    "MUTATION_2_independent (reworded fabricated success on risk-rejection path -> 2 failed / 25 passed = both new dedup tests; grep 'resp[\"published\"] = True' on mutated source -> 0 matches, old guard GREEN)",
    "MUTATION_3_independent (reworded synthesized success on dedup cache miss -> 5 failed / 22 passed, source scan blind)",
    "MUTATION_4_independent (faithful gap4-06: un-evicted seen-set + resp.update synthesis on seen-but-not-cached -> EXACTLY 2 failed / 25 passed on the two named tests, corroborating Main's transcript count; source scan blind)",
    "MUTATION_5_independent (emit_candidates keeps the literal but pops 'stub' from every candidate -> 27 PASSED, guard illusory — the finding)",
    "MUTATION_6_independent (drawdown breaker neutered via `if False and ...` -> 1 failed / 26 passed = test_drawdown_breaker_blocks_buys; cycle-1 observation (ii) cleared, the guard IS discriminating)",
    "live_tool_execution_emit_candidates (create_signals_server + fastmcp Client, production vs mutated: all_stub_true True vs False, key present vs absent)",
    "vacuity_review_of_the_3_new_tests (WebClient test cannot pass without a WebClient being constructed — proven by MUT3 where publish short-circuits, captured=={} and the test FAILS)",
    "spec_correction_1_reverified (scripts/mcp_servers/ holds only smoke-test/reconcile scripts; the three servers are at backend/agents/mcp_servers/; meta_coordinator.py at backend/agents/, absent from backend/autoresearch/)",
    "spec_correction_2_reverified (BacktestResult dataclass at backtest_engine.py:110-123 — no dsr / return_pct / max_drawdown_pct / num_trades; dropping the key beats fabricating 0.0, and test_backtest_result_lacks_the_fields_the_spec_named pins it)",
    "criterion_5_source_facts (0 occurrences of '2025-12-31' in data_server.py; date.today() at :98, :148, :181, :281)",
    "proxy_guard_classification (all 27 tests classified behavioural vs source-scan; the criterion-mandated scans — paper_trader.get_portfolio(, compute_dsr_real, 2025-12-31 — are legitimate; exactly one residual illusory guard found)",
    "scope_check (only 2 non-scoped files modified: .claude/.archive-baseline.json adding \"75.2\" = archive-hook bookkeeping, and backend/backtest/experiments/mda_cache.json = regenerated cache; neither is production logic)",
    "read_only_integrity (git status byte-identical before and after my harness; all mutated copies written to scratchpad only)"
  ],
  "harness_compliance_ok": true,
  "notes": "CALLER'S PROBES, ANSWERED WITH MY OWN RUNS. (a) YES — the three new tests genuinely fail when the thing they guard is broken; I re-ran the mutations rather than trusting the transcript. M1 (revert the unwrap call site, keep the import): 1 failed / 26 passed, only the new SecretStr e2e test fails, and the assertion output shows the WebClient literally received SecretStr('**********') — the old source-scan guard stays green exactly as cycle 1 predicted. M2 (reworded fabricated success on the risk-rejection path): 2 failed / 25 passed, both new dedup tests, grep count 0. M4 (the faithful gap4-06 shape — un-evicted seen-set plus resp.update({\"published\": True}) on seen-but-not-cached): EXACTLY 2 failed / 25 passed on precisely the two named tests, which corroborates your transcript's count, including the detail that the retry test fails on execute_buy.assert_called_once() rather than on the published flag. Your MUTATION-2 narration says \"on cache miss\", which taken literally (my M3) fails 5 tests, not 2 — the seen-set variant is what reproduces your number, so the description is slightly loose but the claim and the count are both sound. (b) The WebClient test is NOT vacuous, and I have positive proof rather than an argument: under M3 publish_signal short-circuits before Step 7, `captured` stays {}, and the test FAILS with `assert None == 'xoxb-real-token'`. It cannot pass without a WebClient actually being constructed inside publish_signal. The lazy `from slack_sdk import WebClient` at :503 is what makes the module-level monkeypatch bite at call time. (c) I cannot prove byte-identity absolutely (no cycle-1 snapshot exists on disk), but four independent fingerprints all agree: numstat is 222/115 exactly as recorded; unwrap_secret sits at :495 and :497, the exact line numbers cycle 1 quoted, so nothing above line 495 shifted; get_risk_constraints is byte-identical to HEAD by direct extraction; and a residue scan for resp.update( / _seen_signal_ids / published\": True returns 0. The 12:20:28 mtime on signals_server.py is fully explained by mutate-then-revert. I am satisfied the production diff is unchanged. (d) YES, one — and it is the finding above. test_emit_candidates_payload_is_stub_marked_and_keeps_consumer_contract is a source-literal scan, and I proved by mutation that stripping 'stub' from every emitted candidate leaves all 27 tests green. Everything else is clean: the other four source scans are criterion-mandated verbatim (\"source no longer contains 'paper_trader.get_portfolio('\", \"compute_dsr_real ... (source scan)\", \"no '2025-12-31' literal remains\"), so keeping them is correct, not false confidence. Your decision to keep the old scans ALONGSIDE the new behavioural tests is right — they are cheap and they pin the absence of the old code shape; the mistake would be relying on them alone, which is now true in exactly one place. I also cleared cycle-1's observation (ii): I neutered the drawdown breaker and test_drawdown_breaker_blocks_buys failed, so that guard does discriminate despite not pinning the conflict string. Observation (iv) still stands as a residual (the _server() helper bypasses __init__, so a regression re-introducing a paired set/dict in __init__ would be caught only by the source scan) — worth a line in a follow-up, not a blocker. (e) BOTH spec corrections re-confirmed independently and both are legitimate. scripts/mcp_servers/ contains only smoke-test and reconcile scripts — none of the three servers; meta_coordinator.py is at backend/agents/ and does not exist under backend/autoresearch/. The BacktestResult dataclass has no dsr, return_pct, max_drawdown_pct or num_trades; dropping the dsr key rather than emitting a fabricated 0.0 remains the only choice consistent with the step's own thesis, no immutable criterion names those fields, and test_backtest_result_lacks_the_fields_the_spec_named prevents the correction from being silently undone.\n\nSCOPE HONESTY: the cycle-2 write-up is accurate and the self-criticism in it is warranted and correctly aimed. Two pre-existing conditions are disclosed accurately and I verified both — the ruff F401 reproduces byte-identically at HEAD and sits upstream of every hunk, and the go_live_drills/position_limits blockage is a stale `backend` package in site-packages, not a 75.3 defect (that one deserves its own housekeeping ticket, since the drill that pins the strict-`>` boundary is currently unrunnable).\n\nPROPORTIONALITY, STATED PLAINLY: this is a smaller gap than either cycle-1 blocker. emit_candidates emits avowedly synthetic data, has a single consumer that never reaches publish_signal, and publish_signal's stub_provenance refusal is separately covered by a real behavioural test — so nothing on the money path depends on this guard. A lenient reading of criterion 4 that lets \"(source scan)\" govern the whole sentence would score it MET. I am not taking that reading, for three reasons: the semicolon places the qualifier on the other clause, the step's entire thesis is that shape-asserting tests are why these bugs shipped, and cycle 1 applied exactly this standard to criteria 3 and 6 on identical evidence. Waving it through after I had personally watched 27 tests stay green on a full behavioural regression would be the rubber-stamp this role exists to prevent.\n\nPATH TO PASS: add the one behavioural test in the constraint field (test-only, additive, offline, no production change — I ran that exact client path in about a second), keep the existing source-scan test beside it, mutation-test it the way you did the other two, update experiment_results_75.3.md and live_check_75.3.md, then spawn a FRESH Q/A on the changed evidence. That would be cycle 3; a third CONDITIONAL on this step-id would auto-FAIL under the qa.md rule, so the next pass needs to land it."
}
```

---

## Follow-up (Main, cycle 3 -- 2026-07-20)

Accepted. This is the **third** instance of the same defect class in this step, and the
Q/A found it by mutating in a way I had not: keep the `"stub": True` literal in the
candidate dict, then `pop("stub")` off every candidate before returning. All 27 tests
stayed green while criterion 4's first clause fully regressed. Production behaviour was
correct, so again **no production code changed** -- production diff remains byte-identical
at 222 insertions / 115 deletions across the 3 servers.

### Fix (test-only, additive)

`test_emit_candidates_really_emits_stub_marked_candidates` drives the real tool through
`create_signals_server()` + an in-process `fastmcp` Client and asserts over the **emitted
payload**: `len(candidates) >= 5`, every candidate has a `dsr` key (consumer contract for
`scripts/harness/mcp_ab_test.py`), every candidate has `stub is True`, and every
`reason == "PENDING_IMPLEMENTATION"`. Offline -- no BQ, no network.

**Mutation-verified with the Q/A's exact mutation:**
```
keep '"stub": True' in the dict, add `for _c in candidates: _c.pop("stub", None)` before return
  $ grep -c '"stub": True' backend/agents/mcp_servers/signals_server.py
  3                                          <-- literal present, old source-scan guard GREEN
  $ pytest backend/tests/test_phase_75_mcp_truth.py -q
  FAILED test_emit_candidates_really_emits_stub_marked_candidates
  1 failed, 27 passed                        <-- CAUGHT
Mutation reverted; 28 passed.
```

### Proactive audit of EVERY remaining guard (probe (d), answered exhaustively)

Rather than wait for a fourth finding, I classified all 28 tests. Six are source-scan-only;
each is now either **literally what the criterion demands** or **paired with a behavioural
test**:

| Source-scan test | Status |
|---|---|
| `test_source_no_longer_calls_nonexistent_paper_trader_get_portfolio` | Criterion 1 demands this exact scan verbatim; paired with `test_get_portfolio_returns_real_nav_and_positions` |
| `test_no_synthesized_published_true_in_source` | Paired with both new end-to-end dedup tests |
| `test_emit_candidates_source_is_stub_marked_and_keeps_consumer_contract` | Paired with the new payload test (this cycle) |
| `test_compute_dsr_real_dead_branch_is_gone` | Criterion 4 attaches "(source scan)" to this clause explicitly. The branch is *statically unreachable*, so there is no behaviour to observe -- a scan is the only possible guard |
| `test_data_server_has_no_hardcoded_cutoff_literal` | Criterion 5 demands this exact scan verbatim; paired with `test_prices_and_fundamentals_use_today_derived_cutoffs` |
| `test_secretstr_unwrap_helper_semantics_and_call_site_present` | Paired with the WebClient-observing test (cycle 2) |

Two tests were **renamed** because their names overclaimed: the SecretStr and
emit_candidates source scans now say "source"/"helper_semantics" so no future reader
mistakes them for behavioural coverage. That mislabelling is part of how the illusion
formed in the first place.

Suite: 27 -> 28. No production diff. A FRESH Q/A is spawned on the changed evidence.

---

## Cycle-3 Q/A verdict (verbatim captured return value) -- PASS

- **Launch:** run `wf_96ce4962-bff` (Opus 4.8 / effort max), 2026-07-20.
- **Transcription rule:** Main records the verdict, never authors it.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 6 immutable criteria MET, and I re-proved the contested ones myself with six independent mutations run through a read-only injection harness (mutated copies in scratchpad; repo sha256 + git-status fingerprint byte-identical before and after). (a) The caller's mutation re-run: keeping the '\\\"stub\\\": True' literal (grep count 3, old source-scan guard GREEN) while popping the key off every candidate before return yields 1 failed / 27 passed, failing exactly test_emit_candidates_really_emits_stub_marked_candidates — criterion 4's guard is now real. (b) That test is NOT vacuous: I forced emit_candidates to return zero candidates and it fails on `assert 0 >= 5`, and popping the `dsr` key fails the consumer-contract assertion — the len>=5 guard bites before the stub assertion, so an empty payload cannot slip through. (c) Main's classification claim VERIFIED independently: I enumerated all 28 tests and found exactly the same six source-scan-only guards, each either criterion-mandated verbatim (criterion 1's \\\"source no longer contains 'paper_trader.get_portfolio('\\\", criterion 4's explicit \\\"(source scan)\\\" on the dead branch, criterion 5's \\\"no '2025-12-31' literal remains\\\") or paired with a behavioral test I mutation-proved. I did not inherit cycles 1-2's results: reverting only the unwrap call site (import kept) fails ONLY the WebClient test with the captured token printing literally as SecretStr('**********'); a faithful gap4-06 regression (never-evicting seen-set + reworded resp.update({\\\"published\\\": True})) fails EXACTLY the two behavioral dedup tests while `grep 'resp[\\\"published\\\"] = True'` returns 0. Bonus finding in Main's favor: re-introducing the gap4-01 bug fails FIVE tests, not one — create_autospec raises \\\"Mock object has no attribute 'get_portfolio'\\\", so criterion 1 has real defense-in-depth beyond the mandated scan. (d) Production diff STILL unchanged at 222 insertions / 115 deletions across the 3 servers (independently recomputed via numstat); get_risk_constraints byte-identical to HEAD (len 1216 == 1216) and no diff line touches a risk threshold — the sole `max_drawdown_pct` hit is the BacktestResult dataclass extraction, in scope. (e) Both spec corrections re-confirmed legitimate a third time. Deterministic tier clean: verification command bare exit 0 (28 passed), all 3 modules import in the venv, pre-existing suite collects 10 exit 0, and ruff's sole F401 (pathlib.Path:29) independently reproduced on `git show HEAD:` content with line 29 in no diff hunk. Harness compliance clean on all 5 items. No criterion rests on an illusory guard; residuals are documentation nits and pre-existing conditions, named below.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "research_gate_envelope (gate_passed=true, 7 sources read in full vs floor 5, 27 URLs vs floor 10, recency scan at section 4, audit_class=false so coverage.dry not required)",
    "mtime_ordering (research 11:48:03 < contract 11:49:54 < production 11:53+ < results/live_check 12:32:31)",
    "contract_criteria_verbatim (all 6 criteria matched byte-exact against masterplan.json)",
    "log_last (grep 'phase=75.3 result=' harness_log -> 0 entries; masterplan status=pending, retry_count=0)",
    "no_verdict_shopping (evidence CHANGED: suite 27->28, new fastmcp payload test, 2 overclaiming test names renamed, cycle-3 sections added to critique + live_check; both prior verdicts preserved verbatim)",
    "immutable_verification_command (bare exit 0, 28 passed)",
    "ruff_lint_gate_F821_F401_F811 (exit 1; sole F401 pathlib.Path:29 reproduced on `git show HEAD:` content, HEAD_RUFF_EXIT=1; diff hunks nearest are line 24 and 79, so line 29 untouched -> pre-existing, diff-independent)",
    "backend_runtime_smoke (all 3 changed modules import in venv, exit 0 each)",
    "pre_existing_suite_collection (tests/test_mcp_servers.py collects 10, exit 0)",
    "production_diff_fingerprint (numstat 222 insertions / 115 deletions across the 3 servers = cycle-1 figure exactly; cycles 2 and 3 test-only)",
    "boundary_threshold_proof (get_risk_constraints extracted from HEAD and worktree, byte-identical, len 1216 == 1216; diff grep for threshold constants/operators returns only the in-scope BacktestResult extraction)",
    "MUTATION_A_pop_stub_independent (keep literal grep=3, pop key from every candidate -> 1 failed / 27 passed, exactly the new test; source-scan guard stays GREEN -- caller's probe (a) reproduced)",
    "MUTATION_B_empty_candidates_vacuity (emit zero candidates -> fails `assert 0 >= 5`; the new test is non-vacuous on the empty path -- caller's probe (b) answered with positive proof)",
    "MUTATION_C_pop_dsr_vacuity (strip consumer-contract dsr key -> new test fails on the dsr assertion)",
    "MUTATION_D_revert_unwrap_independent (revert only the call site, keep import -> 1 failed / 27 passed, ONLY the WebClient e2e test; captured token == SecretStr('**********'); criterion-6 source-scan guard stays GREEN)",
    "MUTATION_E_gap4_06_faithful_independent (never-evicting seen-set + reworded resp.update synthesis -> EXACTLY 2 failed / 26 passed on the two behavioral dedup tests; grep 'resp[\"published\"] = True' -> 0 matches, old guard blind)",
    "MUTATION_F_reintroduce_get_portfolio (gap4-01 regression -> 5 failed / 23 passed; create_autospec raises AttributeError -> criterion 1 has behavioral defense-in-depth beyond the mandated source scan)",
    "control_run_of_injection_harness (unmutated copy -> 28 passed, harness proven non-distorting)",
    "test_inventory_classification_independent (all 28 tests enumerated; exactly 6 source-scan-only found, matching Main's audit -- claim (c) VERIFIED, not taken on trust)",
    "criterion_5_source_facts (0 occurrences of '2025-12-31' in data_server.py; date.today() at :98, :148, :181, :281; behavioral test compares captured cutoffs to runtime-computed today, so ANY hardcoded literal fails it)",
    "init_seam_check (__init__ at :85 genuinely builds the OrderedDict; the _server() __new__ helper is a test-harness seam, not a production defect)",
    "scope_check (only 2 non-scoped modified files: .claude/.archive-baseline.json archive-hook bookkeeping and backend/backtest/experiments/mda_cache.json regenerated cache; neither is production logic)",
    "read_only_integrity (sha256 of all 3 servers + test file, and git status fingerprint, byte-identical before and after all six mutations; all mutated copies written to scratchpad only)",
    "third_conditional_rule_evaluated (0 result= entries in harness_log; prior 2 CONDITIONALs documented in the critique -- rule considered and moot on a PASS)"
  ],
  "harness_compliance_ok": true,
  "notes": "CALLER'S PROBES, ANSWERED WITH MY OWN RUNS. (a) Reproduced exactly: 1 failed / 27 passed, literal present at grep count 3, only test_emit_candidates_really_emits_stub_marked_candidates fails while test_emit_candidates_source_is_stub_marked_and_keeps_consumer_contract stays green. Your measurement was accurate. (b) The new fastmcp test is NOT vacuous, and I have positive proof rather than an argument: forcing zero candidates fails at `assert 0 >= 5`, and stripping `dsr` fails the consumer-contract line. Your len>=5 assertion does bite first, exactly as you hoped -- I challenged it and it held. (c) Your all-28 classification is COMPLETE and ACCURATE. I enumerated the suite independently and found precisely the same six source-scan-only tests, no seventh. Each is defensible: three are criterion-mandated verbatim (criterion 1's 'source no longer contains', criterion 4's explicit '(source scan)' on the dead branch, criterion 5's 'no 2025-12-31 literal'), and all six are paired with a behavioral test I either mutation-proved or verified is structurally discriminating. Criterion 5's pairing is stronger than you claimed: test_prices_and_fundamentals_use_today_derived_cutoffs compares against a runtime-computed date.today(), so it catches ANY hardcoded cutoff, not just the one literal the scan names. Criterion 1's pairing is also stronger: re-introducing paper_trader.get_portfolio() fails FIVE tests because create_autospec refuses to auto-vivify the missing method -- your mocking-discipline choice is doing real load-bearing work, not just documentation. The two renames are the right call; the mislabelling genuinely was part of how the illusion formed. (d) Production diff STILL byte-identical: 222/115 recomputed by me, get_risk_constraints byte-identical to HEAD, and the only threshold-shaped grep hit in the diff is the in-scope BacktestResult max_drawdown_pct extraction. (e) Both spec corrections remain legitimate on a third independent look; dropping the dsr key rather than fabricating 0.0 is still the only choice consistent with the step's own thesis, and test_backtest_result_lacks_the_fields_the_spec_named prevents a silent undo.\n\nWHY THIS IS A PASS AND NOT A THIRD CONDITIONAL. I went in prepared to FAIL a hollow guard, and I looked for a fourth instance of the defect class specifically -- that is what the six mutations were for. There isn't one. Every immutable criterion now has at least one assertion that I personally watched fail when the thing it guards was broken. The three historically-illusory guards (criteria 3, 6, 4) are all discriminating under my own mutations, not merely under your transcript's. Holding the line for a third cycle would require naming a real defect, and manufacturing one to avoid the appearance of leniency would be the mirror-image failure of rubber-stamping.\n\nNON-BLOCKING RESIDUALS (none is a criterion miss; carry into the harness_log or a follow-up ticket): (i) experiment_results_75.3.md's trailing addendum still reads 'Suite 24 -> 27' while the same file's header and Verification block correctly say 28 tests / 28 passed -- the cycle-3 narrative lives only in the critique Follow-up and live_check. Stale prose, not a coverage gap; worth a one-line touch-up when you append the log. (ii) live_check line 7 pastes 27 progress dots above '28 passed' -- cosmetic paste artifact. (iii) The track_drawdown try/except at :389-394 is still fail-OPEN (logs a warning and continues), so a telemetry failure silently returns the drawdown breaker to the inert state gap4-05 exists to fix. This was cycle-1 observation (i), is deliberate per the comment, and no criterion covers it -- but it is a residual fail-open seam inside a fail-closed step and deserves its own ticket. (iv) test_seen_and_outcome_evict_together's second clause short-circuits to True and remains tautological; the len()==3 assertion carries that test, and the single-source-of-truth property is behaviorally proven anyway -- my gap4-06 mutation had to introduce its own separate set precisely because no _seen_signal_ids survives. (v) go_live_drills/position_limits remains unrunnable due to a stale `backend` package in .venv site-packages shadowing the repo module -- pre-existing, diff-independent, and it means the drill pinning the strict-`>` boundary is currently dark. That one is worth a housekeeping ticket on its own merits.\n\nSCOPE HONESTY: the cycle-3 write-up neither overclaims nor hides. It states plainly that this is the third instance of the same defect class, that production was correct all three times, and that the fix was test-only -- all of which I verified rather than accepted. The proactive all-28 audit was the right response to a repeated finding, and it checked out under independent enumeration."
}
```
