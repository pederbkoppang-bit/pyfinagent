# Q/A Critique -- phase-26.2 Adopt Advisor Tool (Sonnet executor + Opus advisor) on synthesis chain

**Q/A spawn:** single Q/A agent (merged qa-evaluator + harness-verifier), RETRY after prior interim spawn
**Date:** 2026-05-16
**Verdict authority:** Q/A (Main does NOT self-evaluate)

---

## Phase 1 -- 5-item harness-compliance audit

1. **Researcher spawn** -- PASS. `handoff/current/research_brief.md` exists, tier=complex (MAX gate), gate_passed=true, 7 sources read in full (3 Tier-1, 4 Tier-2), 3-variant search discipline applied, recency scan 2024-04 -> 2026-05 present, internal grep at file:line covers all 6 required modules. Cited as researcher_aabf565c172de860f in contract.
2. **Contract pre-commit** -- PASS. `contract.md` immutable success_criteria are copied verbatim from masterplan step 26.2 (verification command + 4 sub-criteria). Plan committed pre-Generate; experiment_results.md references the plan steps in order.
3. **Results recorded** -- PASS. Both `experiment_results.md` and `live_check_26.2.md` exist; live_check contains verbatim API output + BQ row dump + A/B comparison.
4. **Log-last** -- PASS. No phase=26.2 entry in `handoff/harness_log.md` yet -- correct state before LOG phase (LOG comes after Q/A PASS).
5. **No verdict-shopping** -- PASS. This is the first completed 26.2 Q/A; prior spawn (a33fe706f0bb37253) returned interim status without writing a critique. Not a re-spawn-on-unchanged-evidence.

---

## Phase 2 -- 5 deterministic checks

**D1. Verification command** -- PASS.
```
$ source .venv/bin/activate && python -c 'from backend.agents.llm_client import advisor_call; print(advisor_call.__module__)'
backend.agents.llm_client
```

**D2. Syntax-check 4 files** -- PASS.
```
$ python -c "import ast; [ast.parse(open(f).read()) for f in ['backend/agents/llm_client.py','backend/config/settings.py','backend/agents/cost_tracker.py','backend/agents/orchestrator.py']]; print('all 4 files parse OK')"
all 4 files parse OK
```

**D3. AgentCostEntry advisor fields** -- PASS.
```
backend/agents/cost_tracker.py:115:    is_advisor: bool = False
backend/agents/cost_tracker.py:116:    advisor_model: Optional[str] = None
backend/agents/cost_tracker.py:117:    advisor_input_tokens: int = 0
backend/agents/cost_tracker.py:118:    advisor_output_tokens: int = 0
backend/agents/cost_tracker.py:207: def record_advisor_call(...)
```
All four fields present + `record_advisor_call()` method (line 207) computes blended cost via separate executor and advisor pricing lookups (lines 228-246).

**D4. BQ rows for request_id `msg_01NfNK5aMRuLB95tj9JbnRCF`** -- PASS.
```
2026-05-16 14:59:52.217784+00:00 anthropic claude-opus-4-7    Synthesis_advisor_tool  in_tok=2891 out_tok=1831
2026-05-16 14:59:52.217774+00:00 anthropic claude-sonnet-4-6  Synthesis               in_tok=2871 out_tok=256
```
Exactly 2 rows. Advisor row uses Opus 4.7 rates; executor row uses Sonnet 4.6 rates. Encoding `agent LIKE '%_advisor_tool'` queryable.

**D5. Orchestrator wiring** -- PASS.
```
backend/agents/orchestrator.py:1061:    _enable_advisor = bool(getattr(self.settings, "enable_advisor_tool", False))
backend/agents/orchestrator.py:1064:    try:
backend/agents/orchestrator.py:1065:        from backend.agents.llm_client import advisor_call as _advisor_call
backend/agents/orchestrator.py:1066:        _adv = _advisor_call(...)
backend/agents/orchestrator.py:1077:        _ct.record_advisor_call(...)
backend/agents/orchestrator.py:1092:    except Exception as _adv_exc:
backend/agents/orchestrator.py:1094:        logger.warning("[phase-26.2] advisor_call failed (%r); falling back to Opus-solo path", ...)
```
Flag gate, try/except fallback to Opus-solo, and `record_advisor_call` invocation all present and correctly wired. Settings flag at `backend/config/settings.py:171` defaults to False.

**Deterministic verdict:** all 5 checks PASS.

---

## Phase 3 -- LLM judgment

**J1. Contract alignment** -- PASS. experiment_results.md executes Plan steps 1-5 verbatim with the documented out-of-scope deferrals (revision loop, multi_agent_orchestrator, planner_agent) explicit and consistent with the contract's "Scope honesty / out-of-scope" section. No silent divergence.

**J2. A/B cost-finding disclosure** -- PASS. The +919% advisor-cost finding is documented prominently in `live_check_26.2.md` Evidence D ("Cost delta (A vs B): +919.4% (advisor more expensive)") with verbatim cost numbers ($0.072683 vs $0.007130), explicit root-cause analysis (1831 output tokens at Opus rates dominates), and explicit refutation of the brief's 30-50% hypothesis. `experiment_results.md` line 102 and the sub-criteria self-summary (line 109) reiterate the finding. Not hand-waved.

**J3. Operator protection** -- PASS. `enable_advisor_tool: bool = Field(False, ...)` defaults to False, so production behavior is unchanged. Both `experiment_results.md` line 130 and `live_check_26.2.md` line 110 contain the explicit recommendation: "keep `enable_advisor_tool=False`; do NOT flip the flag for the synthesis chain". The operator is protected by the flag default + clear documentation. A code-level guard refusing the flag for synthesis would be over-engineering at this stage (the flag is fresh, default-off, single workload).

**J4. Sample size critique** -- ACCEPTABLE-FOR-PASS. N=1 is at the lower bound of the contract's pre-committed "1-3 synthesis prompts" scope. The "no signal quality regression" sub-criterion is satisfied as worded (recommendation match HOLD/HOLD, conviction diff 1 within threshold). N=1 is admittedly thin for statistical confidence on signal quality, but: (a) it lies within the pre-committed range, (b) the +919% cost finding is so dramatic that the operator recommendation is robust regardless of N, (c) the contract explicitly defers full multi-cycle A/B to operator-driven rollout once the flag flips True. This should be FLAGGED as a known limitation rather than blocking PASS.

---

## Phase 4 -- Final verdict envelope

Outcome chosen: **PASS** (option a). Rationale: the four immutable success sub-criteria are all satisfied AS WORDED; implementation is correct, gated, observable, and reversible; the cost finding is an operational signal (an honest disclosure with operator guidance) rather than an implementation defect; flag-default-False + prominent documentation provides adequate mis-flip protection for a fresh flag. Per `feedback_harness_rigor.md`: this is an honest PASS, not a rigged one -- the deliverable IS the helper + wiring + finding.

```json
{
  "ok": true,
  "verdict": "PASS",
  "violated_criteria": [],
  "violation_details": "None. All 4 immutable sub-criteria satisfied as worded. Implementation is correct (4-file diff, syntax OK, BQ rows confirmed, flag gated). The brief's profit hypothesis (25-45% cost reduction on synthesis) is refuted by the N=1 A/B at +919% advisor cost, but this is documented honestly in live_check_26.2.md Evidence D + experiment_results.md line 102/109/130 with explicit operator recommendation to NOT flip the flag for synthesis. The cost-reduction hypothesis was Main's stated expectation, NOT an immutable success criterion -- the literal criterion is signal-quality non-regression, which is met (HOLD/HOLD, conviction diff 1). N=1 noted as a known limitation but within pre-committed 1-3 range.",
  "certified_fallback": false,
  "checks_run": 10,
  "checks_run_detail": ["harness_audit_5", "verification_command", "syntax_4_files", "cost_tracker_fields", "bq_rows_advisor_executor", "orchestrator_wiring", "contract_alignment", "cost_finding_disclosure", "operator_protection", "sample_size_critique"],
  "cost_finding_assessment": "Brief's 25-45% cost-reduction hypothesis is REFUTED for single-call synthesis prompts (+919% advisor cost on N=1). Refutation is documented prominently and honestly in live_check_26.2.md Evidence D and experiment_results.md, with clear root-cause (1831 advisor output tokens at Opus rates dominates Opus-solo's 229 tokens) and operator-actionable conclusion. This is a finding, not a defect.",
  "operator_recommendation": "DO NOT flip enable_advisor_tool=True for synthesis -- the A/B shows +919% cost with marginal quality delta; defer to phase-27 affordance for planner/debate workloads where the advisor pattern's economics actually apply.",
  "ab_sample_size_critique": "N=1 is at the lower bound of contract's pre-committed 1-3 range; signal-quality claim is thin but cost finding is so dramatic (9.2x) that the operator recommendation is robust regardless of N. Flagged as known limitation, not blocking.",
  "post_pass_followup_for_main": [
    "Append harness_log.md cycle entry with result=PASS BEFORE flipping masterplan status (log-last discipline).",
    "Confirm masterplan.json step 26.2 status flip happens AFTER the log append.",
    "Live-check gate: live_check_26.2.md exists; auto-push should proceed."
  ]
}
```

---

## Summary

phase-26.2 PASSES. The advisor_call helper, settings flag, cost-tracker integration, and synthesis-pipeline wiring are all correctly implemented and observable in BQ. The A/B test refutes the brief's profit hypothesis honestly and recommends keeping the flag off for synthesis -- this is the system working as designed (real data updating the operator's mental model). The deliverable is the capability + finding, not the cost reduction. Main may proceed to LOG then status-flip.
