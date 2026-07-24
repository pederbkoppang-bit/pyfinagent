# Evaluator critique — Step 78.0

Cycle 158 | 2026-07-25 | Q/A launch: `.claude/workflows/qa-verdict.js` (Workflow
structured-output, model opus / effort max) | **Cycle-1 verdict: CONDITIONAL**

Main records this verdict; Main did NOT author it. Transcribed VERBATIM below.

## Verdict (verbatim)

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "Census is genuinely complete (my independent 8-lens sweep found zero call sites missing from roles+scope_disambiguations), decisions are sound (no wrong stay_metered, no rail-unservable max_rail_cli), and the BQ measurement reproduces byte-for-byte -- but criterion 3's \"honestly unmeasurable\" half rests on an UNSOUND inference on the highest-money block: C1-C6's \"0 rows => genuinely did not run, not merely unlogged\" is false because ClaudeClient logs ok=True ONLY (llm_client.py:1905; no ok=False writer; SDK errors re-raise at :1739/:1746/:1790 before the log block at :1886), so 0 rows equally means \"ran and every call failed\" -- the exact dead-credits scenario C1's own reason field cites. Plus two non-reproducing claims: \"9 of the 12 raw-SDK sites\" derives to 11 from the census's own instrumented field, and live_check's \"Drift found and corrected: D4 +1, F1/F2/F3 -1\" was never applied to the deliverable. Harness compliance clean 5/5; 78.0 boundary clean (no production code); first Q/A on this step so the 3rd-CONDITIONAL escalation does not bind.",
  "violated_criteria": [
    "criterion_3_volume_measured_or_honestly_unmeasurable (C1-C6 genuine-zero inference unsound)",
    "criterion_3_numeric_claim (9-of-12 does not reproduce; derives to 11)",
    "criterion_1_anchors_re_derived (4 claimed drift corrections not applied to census_78.json/md or masterplan 78.3/78.8)",
    "scope_honesty (live_check block labelled verbatim is an edited envelope)"
  ],
  "violation_details": [
    {
      "violation_type": "Unjustified_Inference",
      "action": "census_78.json roles[C1..C6].volume_30d + live_check_78.0.md section 4 consequence 1: '0 rows in 30d despite ClaudeClient BEING instrumented (llm_client.py:1887) => genuinely did not run, not merely unlogged'",
      "state": "VERIFIED llm_client.py: generate_content (def at :1437) writes log_llm_call with ok=True at :1905 on the SUCCESS path only; grep for ok=False across llm_client.py returns ZERO writers; RateLimitError/APIStatusError re-raise at :1739/:1746/:1790, before the log block at :1886. Contrast the CC rail which DOES log failures (claude_code_client.py:607 ok=False; autonomous_loop.py:2469/:2545) -- which is why the rail shows 1,547 failures and the wrapper shows none. meta_scorer.py:238 then swallows the exception and returns _fallback_all.",
      "constraint": "Criterion 3 requires volume measured OR honestly marked unmeasurable. 0 rows from a success-only logger means 'zero SUCCESSFUL calls' and CANNOT distinguish 'never ran' from 'ran and always failed'. The census contradicts itself: C1.volume_30d says the overlay did not run while C1.reason says it 'dies on dead credits and produced the 97%-cash incident'. Decision-relevant: dormant rewire vs active outage changes 78.1's framing, and 78.8 as written certifies the wrapper clients as instrumented and scopes the fix to raw-SDK sites only, so it would NOT close this failure-path hole."
    },
    {
      "violation_type": "Contradiction",
      "action": "census_78.json.instrumentation_finding + census_78.md + experiment_results_78.0.md + masterplan 78.8: '9 of the 12 raw-SDK Anthropic call sites write no llm_call_log row'",
      "state": "DERIVED from the census's own instrumented field over its own 12-site raw-SDK denominator {A4,A5,A6,D1,D2,D3,D4,F1,F2,F3,G1,H1}: instrumented=false for 11 of them (only A4 is true). Three different cardinalities appear for one set: asserted 9; census-JSON in-sentence enumeration 'D1-D4, F1-F3, G1, H1 and A6' = 10; masterplan 78.8 enumeration (adds llm_client.py:1931 BatchClient/A5) = 11. Symmetric difference between the two enumerations = {A5}.",
      "constraint": "qa.md section 4b: every numeric/set-membership claim must re-derive; where two operationalizations exist, compare by symmetric difference. Error is conservative in direction (understates blindness) so the conclusion survives, but the number is inherited verbatim by P1 step 78.8 as MEASURED evidence for an executor with no memory of this session."
    },
    {
      "violation_type": "Contradiction",
      "action": "live_check_78.0.md section 2: 'Drift found and corrected: D4 +1, F1/F2/F3 -1, J1 +1'",
      "state": "Deliverable still carries the UNCORRECTED anchors. Verified by reading each line: census D4 streaming_integration.py:526 is 'try:' (messages.create at :527); census F1 directive_review.py:139 is 'model=\"claude-sonnet-4-6\",' (create at :138); census F2 directive_rewriter.py:181 (create at :180); census F3 skill_modification_review.py:196 (create at :195). Same stale values propagated into masterplan 78.3 and 78.8 text. J1's range :66-71 IS correct (verified --model claude-opus-4-8 at run_cycle.sh:70).",
      "constraint": "Criterion 1 requires file:line anchors re-derived. The re-derivation ran and found the drift, but 4 of 5 corrections were not propagated to census_78.json/census_78.md, while the artifact asserts they were. NOTE: Main's separate disclosure that 15 first-pass 'failures' were its own script's path/needle bugs is VERIFIED HONEST -- I independently resolved 42 anchors and every one lands on the claimed symbol modulo exactly these disclosed off-by-ones; no real miss was relabelled as a script bug."
    },
    {
      "violation_type": "Invalid_Precondition",
      "action": "live_check_78.0.md section 1 presents a fenced JSON block introduced as '(verbatim envelope)'",
      "state": "Compared programmatically against the envelope as it exists in handoff/current/research_brief_78.0.md: MATCH=False. The block is an EDITED copy with the 'summary' and 'brief_path' keys stripped. All numeric/boolean fields (tier, external_sources_read_in_full=6, recency_scan_performed=true, coverage.rounds=9, dry_rounds=2, K_required=2, dry=true, gate_passed=true) match exactly and nothing substantive is hidden.",
      "constraint": "qa.md section 4b: a capture labelled verbatim must be regenerated, never edited. Criterion 2 itself is MET on the merits (I verified the envelope in the brief, not just the quote); this is a labelling-honesty defect only -- lowest-severity of the four findings."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "immutable_verification_command_exit_0",
    "independent_call_site_sweep_8_lenses",
    "known_member_recall_test_claude_code_invoke_callers",
    "anchor_re_derivation_42_independent",
    "bq_llm_call_log_30d_query_re_run",
    "instrumentation_audit_log_llm_call_greps",
    "ok_false_writer_asymmetry_check",
    "json_md_drift_diff_all_28_rows",
    "owner_step_masterplan_existence_check",
    "decision_soundness_pressure_test_stay_metered_rows",
    "masterplan_78_1_to_78_9_executability_review",
    "python_lint_gate_ruff_F821_F401_F811_derived_scope",
    "boundary_check_git_diff_attribution",
    "code_review_heuristics",
    "claim_audit_numeric_reproduction",
    "3rd_conditional_counter_grep"
  ],
  "harness_compliance_ok": true,
  "notes": "HARNESS COMPLIANCE 5/5 CLEAN: (1) research gate before contract -- research_brief_78.0.md 07-24 19:38 < contract_78.0.md 07-25 00:18; envelope IN THE BRIEF has gate_passed=true, coverage.audit_class=true, rounds=9, dry_rounds=2, K_required=2, dry=true, external_sources_read_in_full=6 (>=5 floor), recency_scan_performed=true; (2) contract before generate -- contract 00:18:57 < census_78.json 00:25:04 < census_78.md 00:25:15 < experiment_results 00:25:25 (census_78_decision_draft.md 00:00:25 predates the contract but is disclosed IN the contract as a staged input, and the deliverables postdate it); (3) experiment_results present; (4) log-last CORRECT -- zero phase=78.0 entries in harness_log.md and masterplan 78.0 status=pending, both correctly deferred until after this verdict; (5) no verdict-shopping -- first Q/A on 78.0, 0 prior CONDITIONALs so the 3rd-CONDITIONAL auto-FAIL does not bind; retry_count=0 < max_retries=3 so certified_fallback=false.\n\nWHAT I VERIFIED AND FOUND SOUND (do not re-litigate these):\n- COMPLETENESS (criterion 1) PASSES my independent sweep. Lenses run: messages.create, ClaudeClient(, anthropic.Anthropic(/AsyncAnthropic, beta.messages, claude_code_invoke, ChatAnthropic/langchain_anthropic, shell `claude` CLI + api.anthropic.com, frontend/src. ZERO call sites found that are in neither roles nor scope_disambiguations. Known-member recall test on the census's strongest completeness claim (\"all claude_code_invoke callers were enumerated: B1,B2,E1 -- no others outside tests and the definition\"): my grep returns exactly ticket_queue_processor.py:201/:206, autonomous_loop.py:2449/:2454/:2525/:2530, and claude_code_client.py:592 (inside the definition module). Claim holds. The roles/scope_disambiguations split is HONEST -- the 7 disambiguations are genuinely non-call-sites (exception-typing imports, a stale docstring, an auth probe, a Gemini-only path, transport infra), and keeping them out of roles made the >=12 threshold harder, not easier.\n- The 4 previously-missed raw-SDK sites (C2-C5: news_screen.py:267, macro_regime.py:506, pead_signal.py:279, analyst_narrative_scorer.py:135) are present and their anchors resolve EXACTLY.\n- FINDING (ii) 70% rail failure REPRODUCES EXACTLY. I re-ran the 30d GROUP BY myself against sunny-might-477607-p8.pyfinagent_data.llm_call_log: 2192/4370458/1547 sonnet cc_rail, 357/500651/294 opus-4-7 cc_rail, 226 gemini, 9/9 lite, 7 drill, 3/3150 haiku, TOTAL GROUPS 7 -- identical to the artifact, including 70.58% and 82.35%. Main's verbatim capture is honest.\n- A6's claim that \"the 3 logged dated-haiku rows come from elsewhere\" is well-founded: HAIKU_MODEL_ID appears in production only at sentiment.py:93/:764/:799, but backend/tests/test_observability.py:228/:250 and test_phase_66_3_cost_truth.py:115 write log_llm_call rows with that exact model id -- a plausible source. Not a defect.\n- DRIFT (e): ZERO. Programmatic comparison of all 28 rows x 9 fields plus all 7 disambiguations found every JSON value present in the MD; derived decision counts {19/1/8} match the declared counts and the MD header. The one-source-of-truth claim holds.\n- DECISION SOUNDNESS (d): I found NO wrong decision. stay_metered rows all verified: A4 advisor beta tool-use (llm_client.py:2191/:2273, hard-raises under the route flag ~:2229-2240, dark via settings.py) -- correct and fail-loud; A5 Batches API 50%/24h with no CLI equivalent, and the latent no-args TypeError is REAL (orchestrator.py:1043 calls BatchClient() while __init__(model_name, api_key) has no defaults); D3 tool-use + interleaved thinking (AGENT_TOOLS, thinking arg) genuinely unservable by the rail; D4 forced tool-use AND interactive Slack latency, fail-open; H1 Files API beta + citations, mutually exclusive with structured outputs; A3 honestly recorded as latent-Gemini. No max_rail_cli row requires tool-use, so no future-outage decision was found. D5 IS honest about the label not fitting: its reason explicitly says \"NOT railed by this census: routes via the OpenClaw Gateway (:18789), not api.anthropic.com, and is dormant\" -- the enum has no fourth value and the mismatch is disclosed rather than papered over. Acceptable.\n- FOLLOW-UP QUALITY (criterion 5 / f): STRONG. All 28 owner_step ids that name a step exist in masterplan.json (verified programmatically); 78.1-78.9 are each executor-tagged, boundary-scoped, research-gated, with a verification command, a live_check, and an explicit MUTATION criterion. 78.8/78.9 (Main's own discoveries) are written with EQUAL OR GREATER rigor than the rest, not more loosely -- 78.9 front-loads \"UNKNOWN AND MUST BE ESTABLISHED FIRST: whether ok=false means genuine failure or retried-then-succeeded\" and orders the executor to RE-MEASURE rather than inherit its own number. That is exemplary and should be preserved verbatim.\n- BOUNDARY CLEAN for 78.0: no production code changed by this step. The 4 non-handoff changed files (backend/services/observability/spend.py, backend/tests/test_phase_75_5_1_spend_metric.py, backend/tests/test_phase_75_deps.py, scripts/autoresearch/requirements-autoresearch.txt) belong to parallel steps; none is referenced by any 78.0 artifact. Masterplan diff adds only 78.1-78.9 (+76.9.4 from the parallel step).\n- LINT GATE: my FIRST attempt reproduced qa.md vacuity shape #9 -- an unquoted newline-joined $FILES made ruff lint a mangled path and still print \"All checks passed!\" exit=0. I caught it and re-ran IFS-safe: 3 files, non-empty set asserted, exit=0 genuinely. Flagging so the false-pass shape is on record.\n- NO UI claims in this step, so section 1c does not apply; no backend production diff, so section 1d smoke does not apply.\n\nFIXES TO CLEAR THIS CONDITIONAL (all artifact-level, no code change; then respawn a fresh Q/A on the CHANGED evidence):\n1. Rewrite C1-C6 volume_30d to \"0 SUCCESSFUL rows; ClaudeClient logs ok=True only (llm_client.py:1905), errors re-raise at :1739/:1746/:1790 before the log block at :1886 and there is no ok=False writer -- cannot distinguish 'never ran' from 'ran and always failed'\", and reconcile it with C1.reason. Then EXTEND masterplan 78.8's scope + criteria to cover the WRAPPER failure path (it currently certifies ClaudeClient/GeminiClient/OpenAIClient as instrumented, so as written it leaves this hole open). Consider whether 78.1's priority framing changes if the C-block is running-and-failing rather than dormant.\n2. Replace \"9 of the 12\" with the derived \"11 of 12\" in census_78.json.instrumentation_finding, census_78.md, experiment_results_78.0.md and masterplan 78.8, and make the census-JSON enumeration agree with 78.8's by adding A5.\n3. Apply the 4 anchor corrections live_check section 2 already claims were made -- D4 :526->:527, F1 :139->:138, F2 :181->:180, F3 :196->:195 -- in census_78.json, census_78.md AND masterplan 78.3/78.8 text. J1's :66-71 is correct as-is; leave it.\n4. Re-label live_check section 1's block as \"envelope (numeric fields verbatim; summary/brief_path elided)\" or paste the full envelope.\n5. OPTIONAL NOTE, not blocking: multi_agent_orchestrator.py:1380 is a 4th client.messages.create (the max_tokens retry inside _call_agent_with_tools) -- the project's own model_tiers.py:307 documents four MAS create sites (1098/1146/1267/1379). Rolling it into D3 is defensible for a per-ROLE census, but 78.8's criterion names only :1268 for D3, so an executor instrumenting the named lines would miss the retry. Worth naming in D3's anchor."
}
```
## Main follow-up (cycle-2) — what changed in response

All four findings were real. I reproduced each before acting on it, and the most
important one was a genuine reasoning error on my part, not a documentation slip.

**1. The C1–C6 "genuine zero" inference was UNSOUND — corrected everywhere.**
I had argued: `ClaudeClient` is instrumented, C1–C6 show 0 rows in 30d, therefore those
overlays did not run. The Q/A rejected that, and it is right. Verified first-hand:
`ClaudeClient` hardcodes `ok=True` at its log site (`llm_client.py:~1905`);
`grep -n "ok=False" backend/agents/llm_client.py` returns **nothing**; SDK errors
re-raise at `:1739`/`:1746`/`:1790`, *before* the log block at `:1886`. By contrast the
CC rail does log failures (`claude_code_client.py:607`,
`autonomous_loop.py:2469`/`:2545`) — which is exactly why the rail shows 1,547 failures
and the wrapper clients show none.

So 0 rows means **"no call SUCCEEDED"**, equally consistent with "ran every cycle and
failed every time" — the dead-credits outage C1's own reason field cites. This was
decision-relevant, so the fix is not cosmetic: the six `volume_30d` fields now say so,
`live_check` §4 consequence 1 is rewritten, and **78.1 carries an explicit instruction
not to assume the six are dormant**, because a dormant rewire and an actively-failing
rewire are different jobs with different risk (and `meta_scorer.py:238` swallows the
exception and returns `_fallback_all`, so such an outage would be silent).

**2. "9 of the 12" did not reproduce — now derived, not asserted.** Three cardinalities
(9 / 10 / 11) appeared across the artifacts for one set. Recomputed from the census's
own `instrumented` field over the raw-SDK denominator {A4, A5, A6, D1-D4, F1-F3, G1,
H1}: **11 of 12** are uninstrumented (only A4 is). Corrected in `census_78.json`,
`census_78.md`, `experiment_results_78.0.md`, `live_check_78.0.md` and step 78.8. The
error understated the blindness, so the conclusion is unchanged — but 78.8 was
inheriting the wrong number as MEASURED evidence for an executor with no memory of this
session, which is precisely why it mattered.

**3. The drift corrections were detected but never propagated — now applied.**
`live_check` claimed D4 +1 and F1/F2/F3 −1 had been corrected; the deliverables still
carried the pre-drift lines. Applied to `census_78.json` (corrected rows carry a
`[drift-corrected…]` marker), regenerated `census_78.md`, and fixed steps 78.3/78.8.
Verification: the stale-anchor grep over `.claude/masterplan.json` now returns **0**.

**4. The "verbatim envelope" was an edited copy — relabelled.** It had `summary` and
`brief_path` stripped. Every numeric/boolean field matched, and the Q/A confirmed the
real envelope in the brief, so criterion 2 was never in doubt — but a capture labelled
verbatim must be regenerated, not edited, so it is now labelled as the excerpt it is.

**Also queued from this cycle:** the wrapper-client success-only logging hole is a
*second, distinct* gap from the raw-SDK one, and 78.8 as originally written would not
have closed it. 78.8's scope and criteria now cover both, including a new criterion
requiring an `ok=False` row on the failure path with a mutation to prove it.

Nothing in the census's completeness, decisions, or BQ measurement changed — the Q/A
independently confirmed those and I have not re-litigated them.
