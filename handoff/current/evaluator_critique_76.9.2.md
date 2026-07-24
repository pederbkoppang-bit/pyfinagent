# Evaluator critique — Step 76.9.2

Cycle 156/161 | 2026-07-25 | Q/A launch: `.claude/workflows/qa-verdict.js` (Workflow
structured-output, model opus / effort max) | **Cycle-1 verdict: FAIL**

Main records this verdict; Main did NOT author it. Transcribed VERBATIM below.

## Verdict (verbatim)

```json
{
  "ok": false,
  "verdict": "FAIL",
  "reason": "Criterion 1 is UNMET (Main's self-assessment is correct, not under-claimed) \u2014 and I additionally MEASURED the defect that makes it unmet, which no artifact identifies. Criteria 2/3/4/5 are independently verified MET. Harness compliance 5/5 clean; no unintended production change; immutable cmd exit=0; 11 passed reproduced; ruff F821/F401/F811 exit=0 on a git-derived 5-file superset scope. C1: the only rc=0 run (2026-07-24 16:12, 15,809-char memo) went through the scratchpad bridge, which the criterion excludes on its face \u2014 and the exclusion is substantive, not pedantic: scratchpad `anthropic_bridge.py` has ZERO `protocol_version` lines (HTTP/1.0 default \u2192 server closes \u2192 SSE body correctly close-delimited), while the shipped `scripts/ops/anthropic_max_bridge.py:103` sets `protocol_version = \"HTTP/1.1\"` and its SSE-passthrough branch (:138-147) sends NEITHER Content-Length NOR Transfer-Encoding and never forces close_connection. LIVE RAW-SOCKET PROBE of the running repo bridge (PID 85602) with an HTTP/1.1 keep-alive `\"stream\": true` request returned verbatim: `HTTP/1.1 200 OK / Content-Type: text/event-stream / Cache-Control: no-cache`, `has Content-Length: False`, `has Transfer-Encoding: False`, `has Connection: close: False`; the full SSE arrived (1027 bytes incl. message_stop) and the server then held the connection open with no terminator (\"RECV TIMEOUT after 18.0s -- server never closed\"). Per RFC 7230 \u00a73.3.3 such a response is delimited ONLY by connection close, so an httpx keep-alive client (what the anthropic SDK uses) blocks forever. `gpt_researcher/actions/report_generation.py` passes `stream=True` at 6 call sites \u2014 the report-generation phase, exactly where attempts 3 and 5 wedged at 0% CPU with ONE ESTABLISHED idle client\u2192bridge connection. The hardening INTRODUCED this regression, which explains both the 16:12 success and every repo-bridge hang since. The E2E streaming test cannot catch it: `urllib.request.AbstractHTTPHandler.do_open` sets `headers[\"Connection\"] = \"close\"` (verified in the installed stdlib), so the test client forces the very close the production client does not \u2014 vacuity shape #5, sole coverage for that branch. Consequence: the durable routing does not work for its production consumer, so C1 is not merely undemonstrated but currently unachievable as shipped \u2014 a criterion miss, hence FAIL rather than CONDITIONAL. Main's disclosure discipline was exemplary and its refusal to exonerate the bridge was sound; the verdict follows the evidence, not the framing.",
  "violated_criteria": [
    "criterion_1_real_run_rc0_through_durable_routing",
    "illusory-guard: test_e2e_streaming_client_gets_sse_passthrough",
    "76.9.5 hypothesis space mis-scoped (H2 named as sse_aggregate; wedge is on the passthrough branch that never calls it)",
    "WARN illusory-guard: test_nightly_default_documented_off OR-escape-hatch satisfiable by a comment"
  ],
  "violation_details": [
    {
      "violation_type": "Threshold_Not_Met",
      "action": "Five run_memo attempts 2026-07-24/25; the only rc=0 (16:12, 15,809-char memo, 4m40s) ran through the scratchpad-lineage bridge",
      "state": "criterion 1 has ZERO covering evidence; scratchpad anthropic_bridge.py protocol_version count=0 (HTTP/1.0, close-delimited) vs shipped anthropic_max_bridge.py:103 protocol_version=\"HTTP/1.1\" -- the two are not interchangeable, so the 16:12 run cannot stand in",
      "constraint": "Immutable criterion 1: 'A real run_memo run completes rc=0 through the DURABLE routing (not the session scratchpad bridge) with a dummy metered key proving $0 leakage, evidence verbatim'"
    },
    {
      "violation_type": "Invalid_Precondition",
      "action": "Live raw-socket HTTP/1.1 keep-alive probe of the running repo bridge with {\"stream\": true}",
      "state": "Response headers: HTTP/1.1 200 OK, Content-Type: text/event-stream, NO Content-Length, NO Transfer-Encoding, NO Connection: close; 1027 bytes incl. message_stop received, then 'RECV TIMEOUT after 18.0s -- server never closed'. gpt_researcher/actions/report_generation.py sets stream=True at 6 sites, so the production report phase takes this exact branch",
      "constraint": "RFC 7230 section 3.3.3: an HTTP/1.1 response with neither Content-Length nor Transfer-Encoding is delimited only by connection close. anthropic_max_bridge.py:138-147 must frame the passthrough body (chunked) or close the connection; as shipped, any keep-alive client hangs indefinitely -- the exact wedge signature recorded in live_check sections 7 and 8"
    },
    {
      "violation_type": "Circular_Reasoning",
      "action": "test_e2e_streaming_client_gets_sse_passthrough (backend/tests/test_phase_76_9_2_max_bridge.py:181-189) asserted green as coverage for the SSE passthrough branch",
      "state": "The test client is urllib.request, whose AbstractHTTPHandler.do_open sets headers[\"Connection\"] = \"close\" (verified in the installed stdlib) -- it forces the connection close that the production httpx client never sends, so the guard stays green for every possible state of the framing defect. It is the SOLE coverage for that branch",
      "constraint": "qa.md section 4c: a guard that cannot fail when its subject is broken does not count; vacuity shape #5 (fixture/harness that cannot represent the failure). Sole-coverage vacuity on the money-path transport is BLOCKING. Fix: re-test with a raw-socket or http.client keep-alive client and prove the guard goes RED when protocol_version=\"HTTP/1.1\" is restored without framing"
    },
    {
      "violation_type": "Unjustified_Inference",
      "action": "Queued step 76.9.5 states H2 as 'the bridge's SSE->JSON aggregation returned a 200 whose BODY was subtly malformed or truncated' and lists H1 (client-side gpt_researcher asyncio defect) first with a causal story about the arxiv-429 storm",
      "state": "The wedge occurs on the stream=True passthrough branch, which NEVER calls sse_aggregate. An executor with no memory of this session would instrument the client (H1) and audit sse_aggregate (H2) and miss the defect in both directions. The 'no in-flight request / bridge logged 200' observation that Main read as pointing client-side is exactly what the framing defect produces -- the bridge finishes writing and logs 200 while the client waits forever",
      "constraint": "Masterplan 76.9.5 criterion 2 requires H1 vs H2 settled with the losing hypothesis explicitly ruled out; the hypothesis space as written cannot reach the measured mechanism. H2 must be restated as a RESPONSE-FRAMING hypothesis covering the passthrough branch, with the scratchpad-vs-repo protocol_version delta recorded as the discriminating evidence"
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "test_nightly_default_documented_off (backend/tests/test_phase_76_9_2_max_bridge.py:302-305): assert \"NEVER silently fall\" in ... or \"NOT falling back to metered\" in text",
      "state": "The first alternate is satisfied by the COMMENT at scripts/autoresearch/run_nightly.sh:75-76, so a change that strips the behaviour while leaving the comment keeps the guard green. Vacuity shape #8 (OR-escape-hatch / comment-token trap)",
      "constraint": "qa.md section 4c: never accept an alternate clause a comment can satisfy. WARN-level only -- criterion 2's behaviour is genuinely covered by test_nightly_flag_off_is_inert and test_nightly_flag_on_healthy_bridge_exports_routing, which run the REAL script and assert observed env"
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5items",
    "research_gate_envelope_gate_passed_6_read_in_full_18_urls_recency",
    "mtime_ordering_research_1654_lt_contract_1656_lt_results_0048",
    "contract_criteria_verbatim_match_vs_masterplan_5of5",
    "log_last_no_result_entry_for_76.9.2_status_still_pending",
    "no_prior_76.9.2_critique_cycle1_no_verdict_shopping",
    "verification_command_immutable_exit0",
    "python_lint_gate_ruff_F821_F401_F811_git_derived_5file_superset_exit0",
    "pytest_backend_tests_test_phase_76_9_2_max_bridge_11_passed_reproduced",
    "git_status_clean_no_unintended_production_change_run_memo_zero_hunks",
    "criterion2_flag_block_read_run_nightly_sh_78_92_default_off",
    "criterion3_independent_client_enumeration_openclaw_json_zero_18796_refs",
    "criterion3_proxy_js_diff_vs_bak_additive_modelmap_5_keys",
    "criterion3_repo_reference_copy_byte_identical_to_deployed",
    "criterion4_modelmap_coverage_resolved_three_roles_haiku45_sonnet46_opus48_all_mapped",
    "criterion4_plist_CLAUDE_PATH_present_launchctl_print_shows_not_yet_loaded",
    "criterion4_symlink_durability_local_bin_claude_rewritten_2.1.219_at_1936",
    "criterion5_behavioral_fixture_rc78_run_memo_never_executes_exit_precedes_invocation",
    "code_review_heuristics",
    "guard_vacuity_check_4c_all_5_criteria",
    "claim_audit_4b_reproduce_counts_and_scopes",
    "live_runtime_smoke_bridge_health_relay_200",
    "live_raw_socket_http11_keepalive_framing_probe_of_repo_bridge",
    "scratchpad_vs_repo_bridge_protocol_version_delta_diff",
    "gpt_researcher_report_generation_stream_true_6_call_sites",
    "urllib_do_open_forces_connection_close_stdlib_verified",
    "mutation_matrix_M1_M6_arithmetic_consistency_audit"
  ],
  "harness_compliance_ok": true,
  "notes": "ANSWERS TO THE FOUR DIRECTED QUESTIONS.\n\n(a) IS C1 REALLY UNMET? YES \u2014 Main did NOT under-claim. The criterion's parenthetical \"(not the session scratchpad bridge)\" excludes the 16:12 run on its face, and I proved the exclusion is load-bearing rather than pedantic: the scratchpad bridge at /private/tmp/claude-501/.../525d6185-.../scratchpad/anthropic_bridge.py has `grep -c protocol_version` = 0 (HTTP/1.0 default, close-delimited responses) while the shipped repo bridge sets protocol_version = \"HTTP/1.1\" at :103. That single line is the difference between a run that completes and a run that wedges. So the 16:12 success is not transferable evidence \u2014 it is evidence for a DIFFERENT transport configuration.\n\n(b) C2-C5 ALL INDEPENDENTLY VERIFIED MET (I did not take the artifact's word on any of them):\n- C2 MET. run_nightly.sh:78 `if [ \"${AUTORESEARCH_USE_MAX_RAIL:-0}\" = \"1\" ]` \u2014 default OFF; revert = one .env line; documented at :71-77. Behaviour covered by two REAL-script fixture tests, incl. the dummy key provably overriding a sourced real key.\n- C3 MET. Reproduced the enumeration myself: openclaw.json has ZERO `18796` references; only the stale 8-apr openclaw.json.bak.1 ever did; the only live ~/.openclaw file referencing it is the proxy itself. `diff` of proxy.js vs its .bak-76.9.2 is additive for MODEL_MAP (5 keys added, 0 removed/changed); the resolveModel change affects only unmapped claude-* ids (silent sonnet downgrade \u2192 verbatim passthrough), a documented trap removal. Repo reference copy is byte-identical to the deployed file (diff exit 0).\n- C4 MET on both halves. I resolved the three roles from model_tiers.py myself: autoresearch_fast \u2192 claude-haiku-4-5 (MODEL_MAP:22), smart \u2192 claude-sonnet-4-6 (:18), strategic \u2192 claude-opus-4-8 (:17). All three mapped. CLAUDE_PATH=/Users/ford/.local/bin/claude IS in the deployed plist. Honest caveat I confirmed independently and which Main also disclosed: `launchctl print gui/501/com.pyfinagent.claude-code-proxy` shows the RUNNING job's environment as {OSLogRateLimit, PATH, HOME, XPC_SERVICE_NAME} \u2014 no CLAUDE_PATH, because kickstart -k restarts the job without reloading plist EnvironmentVariables; it lands on the operator's rebind. The criterion's OR-branch is independently satisfied and genuinely durable: /opt/homebrew/bin/claude \u2192 ~/.local/bin/claude \u2192 versions/2.1.219, and ~/.local/bin/claude was auto-rewritten at 19:36 (2.1.218 \u2192 2.1.219) while the homebrew symlink still resolved.\n- C5 MET, behaviourally not by source scan. `exit 78` at :90 structurally precedes the run_memo invocation at :94, so no path leads from a failed preflight to the metered API. The fixture test runs the REAL script against a dead port and asserts rc==78, observed_env.json ABSENT (run_memo never ran), the \"NOT falling back to metered\" log line, and the 75.11 fail-state increment. I did NOT re-run M1 live: it appends to handoff/autoresearch.log, increments the away-ops fail-state and fires the Slack paging seam \u2014 out of bounds for a read-only evaluator. The green fixture is the equivalent behavioural reproduction.\n\n(c) THE M2/M3 RECORDS ARE CREDIBLE. I could not re-run them without editing production files, so I audited them for arithmetic and semantic consistency and both hold exactly. M2 (`= \"1\"` \u2192 `!= \"1\"`): flag-OFF now ENTERS the block and flag-ON now SKIPS it, so all three real-script fixture tests break \u2014 precisely the recorded \"3 failed, 8 passed\", with test_nightly_flag_off_is_inert named. M3 (preflight \u2192 `true`): only the bridge-down test can change (flag-OFF never enters the block; the healthy-stub test is unaffected), giving exactly the recorded \"1 failed, 10 passed\" with test_nightly_flag_on_bridge_down_fails_loud named. Pre/post SHA-256 identical confirms a clean revert. Note for the record: under M2 the flag-OFF kill is environment-dependent (it depends on whether :18797 is live) but goes RED on BOTH branches, so the kill is robust.\n\n(d) THE ATTEMPT-5 DIAGNOSIS \u2014 this is where I diverge from Main. The REASONING is sound: refusing to exonerate the bridge on the grounds that \"a 200 does not prove a well-formed body\" is exactly right, and it is the discipline that kept the door open to the real answer. But the EVIDENCE now points somewhere the artifact does not look. \"27/27 HTTP 200, last completing 00:22:52, then the client at 0% CPU with no in-flight request and ONE ESTABLISHED idle keep-alive connection\" is not weak evidence for H1 \u2014 it is the *signature* of the response-framing defect I measured: the bridge finishes writing, logs its 200, and waits; the client has no way to know the body ended and blocks forever. Main read \"bridge looks done\" as pointing client-side; that inference is unsound because a bridge with unterminated framing looks exactly \"done\" from the outside. The evidence does support Main's stated conclusion (both hypotheses live), but not the H1-leaning weighting.\n\n(e) IS 76.9.5 ACTIONABLE BY A FRESH EXECUTOR? Procedurally yes \u2014 it is unusually well-written: capture-not-infer is mandated, both hypotheses are kept live, the losing one must be explicitly ruled out, a bounded-timeout mitigation is pre-authorised, and the mutation direction (a stub upstream that never responds) is specified. Substantively NO: its hypothesis space cannot reach the defect. H2 is scoped to \"the bridge's SSE->JSON aggregation\" \u2014 and the streaming path never calls sse_aggregate at all. An executor would instrument the client for an asyncio ghost and audit the wrong function. 76.9.5 must be amended before it is executed.\n\nWHAT CLOSES THIS STEP (concrete, in order):\n1. Fix scripts/ops/anthropic_max_bridge.py:138-147 \u2014 either delete `protocol_version = \"HTTP/1.1\"` at :103 (restores the close-delimited HTTP/1.0 behaviour of the configuration that actually completed a run; minimal risk) or keep 1.1 and add real framing (Transfer-Encoding: chunked, or set self.close_connection = True plus a Connection: close header). Chunked is the correct HTTP/1.1 answer.\n2. Replace test_e2e_streaming_client_gets_sse_passthrough's client with one that does NOT force Connection: close (raw socket or http.client keep-alive) and assert the response terminates. Prove it is not vacuous by re-introducing the defect and showing the test goes RED \u2014 the current urllib-based test cannot go red for any state of this bug.\n3. Re-run the nightly end-to-end through the repo bridge; a clean rc=0 closes criterion 1.\n4. Amend 76.9.5: restate H2 as a RESPONSE-FRAMING hypothesis covering the passthrough branch, and record the scratchpad-vs-repo protocol_version delta as the discriminating evidence between the 16:12 success and every subsequent hang.\n\nMINOR NOTES (do not change the verdict): (i) experiment_results.md:83 records the lint command as the literal placeholder `$ {changed-py robust form} | xargs uvx ruff check ...` inside a block labelled verbatim \u2014 a reader cannot reproduce a placeholder. I re-derived the scope from git and re-ran it myself; it does pass. Flagging it as exactly the shape qa.md section 4b warns about. (ii) masterplan status for 76.9.2 is \"pending\" rather than \"in_progress\" while work is in flight \u2014 cosmetic, not a protocol breach. (iii) backend/tests/test_phase_76_9_2_max_bridge.py:215 places `import urllib.error` mid-file after its use at :207; it works (module-level, executes before collection) and ruff is clean, style only.\n\nEVIDENCE PROVENANCE: every live capture in this critique was taken BY ME (the evaluator) this cycle \u2014 the /health relay check, the raw-socket framing probe of PID 85602, the launchctl print, the proxy/plist/symlink inspections, the model-role resolution, the pytest run and the git-derived lint. No capture in this verdict was supplied by the author. No files were written or modified."
}
```
## Main's response (Main's own words)

**This verdict is correct, and it is the most valuable result of the session.** I asked
the evaluator to test my self-assessment in both directions. It confirmed criterion 1
was genuinely unmet — and then went further than I had, and **measured the defect that
made it unmet**, which no artifact of mine identified.

### What it found that I did not

I had concluded from "27/27 HTTP 200, then 0% CPU with no in-flight request" that the
wedge was client-side. I explicitly refused to exonerate the bridge — but I stopped at
"a 200 does not prove a well-formed body" and queued the question. The Q/A instead ran a
**live raw-socket probe** and found the mechanism:

- `scripts/ops/anthropic_max_bridge.py:103` sets `protocol_version = "HTTP/1.1"`.
- Its SSE **passthrough** branch sent neither `Content-Length`, nor
  `Transfer-Encoding`, nor `Connection: close`.
- Per **RFC 7230 §3.3.3** such a response is delimited *only* by connection close, so a
  keep-alive client blocks forever after the last SSE byte.
- The production client is httpx (anthropic SDK), reached by gpt_researcher via
  `stream=True` at **6 sites** in `actions/report_generation.py` — the report-generation
  phase, which is exactly where attempts 3 and 5 wedged.

**My hardening introduced the regression.** The pre-hardening scratchpad bridge set no
`protocol_version`, defaulted to HTTP/1.0, and closed the socket — the only reason the
16:12 run ever completed. So the criterion's "not the session scratchpad bridge"
exclusion was not pedantic: the two bridges were not interchangeable, and the difference
was the bug.

Note also that the observation I read as pointing *client-side* is exactly what this
defect produces: the bridge finishes writing and logs 200 while the client waits. My
inference was reasonable and wrong, and the queued 76.9.5 would have sent its executor
looking at `sse_aggregate` — a function this branch **never calls**.

### The vacuous guard

The sole coverage for that branch, `test_e2e_streaming_client_gets_sse_passthrough`,
uses `urllib`, whose `AbstractHTTPHandler.do_open` sets `headers["Connection"] = "close"`
— it forces the very close the production client never sends, so it stayed green for
every possible state of the bug. I verified that in the installed stdlib myself.

### What I changed in response

1. **The framing fix** — the passthrough branch now sends `Connection: close` and sets
   `self.close_connection = True`.
2. **A guard that can fail** — `test_e2e_streaming_body_is_framed_for_a_KEEP_ALIVE_client`
   talks raw HTTP/1.1 keep-alive over a socket and requires BOTH that the SSE body
   arrives AND that the server terminates the stream. **Mutation M7** (revert the fix)
   turns it RED while the old urllib guard stays GREEN — empirically proving the old
   guard's vacuity rather than asserting it.
3. **The OR-escape-hatch** in `test_nightly_default_documented_off` — its first alternate
   was satisfiable by a comment. It now matches only executable lines; **mutation M8**
   (strip the behaviour, keep the comment) turns it RED, where it previously stayed
   green.
4. **76.9.5 reframed** from an investigation into a verification step, carrying the
   measured root cause so its executor cannot inherit the wrong hypothesis space.
5. **Live re-probe of the running bridge**: `Connection: close` present, server closes
   the stream (was the hang).

Attempt 6 of the end-to-end run is executing against the fixed bridge. Criterion 1
remains unclaimed until a run actually completes rc=0.
