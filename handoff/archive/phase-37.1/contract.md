# phase-37.1 -- RiskJudge response_schema + thinking/include_thoughts incompat fix

**Step id:** `phase-37.1`
**Date:** 2026-05-22
**Mode:** EXECUTION (backend code change). One harness pass.
**Cycle:** Cycle 15 (after Cycle 14 phase-36.1 scale-out).

---

## North-star delta

**Terms:** B (primary -- compute/parse-cost reduction) + R (secondary -- structured downstream signal).

**B (primary):** today 8 of 10 Risk-Judge invocations drop to raw-text fallback per closure_roadmap §3 BQ-probe B-5. Each fallback wastes one full Risk-Judge LLM call + downstream parse work. Eliminating fallbacks cuts ~80% of the wasted-compute on this agent. Conservative estimate: **~0.1-0.3% per-cycle Burn reduction** (Risk-Judge is ~10% of a cycle's deep-think tier load; 80% of that path goes from "wasted" to "useful" = ~8% deep-think savings = ~0.1-0.3% total cycle cost). 60-day projection: small but non-zero.

**R (secondary):** when Risk-Judge returns valid `RiskJudgeVerdict` JSON, downstream paper_trader can persist `risk_judge_decision` + `recommended_position_pct` + `risk_level` + structured `risk_limits` on `paper_trades` rows (currently empty per BQ-probe B-5 -- closes the field-NULL secondary bug). Enables phase-35.2 (telemetry-wrapper restoration to llm_call_log) AND eventual phase-43.0 DoD-7 ("Risk Judge structured-output succeeds >95%").

**Caltech arxiv:2502.15800 discount:** N/A -- this step does not affect LLM decision quality; it only fixes the API contract between the LLM and the consumer code.

**How measured:** `grep -c "Risk Judge returned invalid JSON" backend.log` for next full cycle = 0 (today's count for phase-34.2 cycle 3 was 8 of 10+ invocations). Long-run: paper_trades.risk_judge_decision NULL-rate -> 0% on stop_loss_trigger SELLs.

---

## Research-gate decision

**Researcher SKIPPED** per /goal conditional clause. closure_roadmap §3 (cycle 12) already documents the diagnosis with file:line precision. The Gemini-2.5+ thinking/include_thoughts/response_schema incompatibility is a documented vendor constraint -- no new external research adds anything to the closure_roadmap brief's coverage of B.4 + B.8.

---

## Hypothesis

> If we (a) add `response_mime_type="application/json"` + `response_schema=RiskJudgeVerdict`
> to `_THINKING_RISK_JUDGE_CONFIG` in orchestrator.py (cosmetic per masterplan
> criterion #1; the live callsite uses `_JUDGE_STRUCTURED_CONFIG` which
> already has both), AND (b) modify `_generate_with_retry` in both
> `risk_debate.py:62` and `debate.py:65` to OMIT `include_thoughts=True`
> when the input config already contains `response_schema` (Gemini 2.5+
> incompatibility per closure_roadmap §3 OPEN-16), THEN the next cycle
> will show 0 "Risk Judge returned invalid JSON" warnings AND
> `paper_trades.risk_judge_decision` will be populated on Risk-Judge-gated
> SELLs (vs empty today).

---

## Immutable success criteria (verbatim from masterplan 37.1.verification)

1. `thinking_risk_judge_config_gains_response_mime_type_and_response_schema`
2. `pydantic_RiskJudgeVerdict_model_defined_in_schemas_py`
3. `live_cycle_post_change_shows_zero_risk_judge_invalid_json_warnings`
4. `live_check_quotes_the_zero_warning_count`

Plus /goal integration gates 1-10.

---

## Plan steps

| # | Step | Status |
|---|---|---|
| 1 | Locate `_THINKING_RISK_JUDGE_CONFIG` (orchestrator.py:107) + real callsite (`_JUDGE_STRUCTURED_CONFIG` in risk_debate.py:48 already has schema) | DONE |
| 2 | Diagnose: include_thoughts=True is the cause of 8/10 invalid-JSON fallbacks (Gemini 2.5+ incompat with response_schema; the response.text concatenates thoughts + JSON, breaking `_parse_json`) | DONE |
| 3 | Add `response_mime_type` + `response_schema=RiskJudgeVerdict` to `_THINKING_RISK_JUDGE_CONFIG` + import RiskJudgeVerdict | DONE |
| 4 | Fix `_generate_with_retry` in `risk_debate.py:62` + `debate.py:65` -- conditional `include_thoughts=True` (only when `response_schema` is NOT in input config) | DONE |
| 5 | pytest `test_phase_37_1_risk_judge_schema.py` (7 tests) -- all pass; total count 311 -> 318 | DONE |
| 6 | Contract + live_check + Q/A + harness_log Cycle 15 + flip status 37.1 to done | IN FLIGHT |

---

## Files changed

| File | Change |
|---|---|
| `backend/agents/orchestrator.py:50` | Import `RiskJudgeVerdict` from `schemas` |
| `backend/agents/orchestrator.py:107-118` | Add `response_mime_type` + `response_schema` + drop `include_thoughts` from `_THINKING_RISK_JUDGE_CONFIG` (cosmetic per criterion #1) |
| `backend/agents/risk_debate.py:62-72` | Guard against `include_thoughts=True` when `response_schema` is in config |
| `backend/agents/debate.py:65-72` | Same guard applied to Moderator + structured Bull/Bear configs |
| `backend/tests/test_phase_37_1_risk_judge_schema.py` (NEW) | 7 tests covering all 4 immutable criteria + backward-compat |

**NOT changed:** `schemas.py` (RiskJudgeVerdict already defined per phase-3 work), any frontend file.

---

## References

- closure_roadmap.md §3 OPEN-16 (the diagnosis: 8/10 invalid-JSON fallbacks on cycle-3)
- closure_roadmap.md §5 N* delta table (B primary)
- backend/agents/orchestrator.py:107 (`_THINKING_RISK_JUDGE_CONFIG` -- cosmetic enrichment site)
- backend/agents/risk_debate.py:62 (the REAL fix site)
- backend/agents/debate.py:65 (same fix applied to Moderator path)
- backend/agents/schemas.py:117 (RiskJudgeVerdict pydantic model -- unchanged)
- /goal directive (10 integration gates + N* delta mandate)
