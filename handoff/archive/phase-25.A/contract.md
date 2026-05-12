# Sprint Contract -- phase-25.A -- Decouple RiskJudge with independent LLM call in lite path

**Cycle:** phase-25 cycle 13 (P1 sprint)
**Date:** 2026-05-12
**Step ID:** 25.A
**Priority:** P1
**Audit basis:** bucket 24.4 F-1 -- `backend/services/autonomous_loop.py:765` aliases `risk_assessment.reason = analysis["reason"]`; one LLM call does Trader + RiskJudge

## Research-gate

Researcher spawned this cycle (agent ae40259a4aed28444). Brief at
`handoff/current/research_brief.md`. Gate envelope: tier=moderate,
external_sources_read_in_full=6, urls_collected=16, recency_scan_performed=true,
internal_files_inspected=7, gate_passed=true.

Key research conclusions:
- The lite-path Trader call lives at `backend/services/autonomous_loop.py::_run_claude_analysis` (lines 700-786). Today's single `client.messages.create` call returns `{action, confidence, score, reason}`. Line 765 aliases that `reason` into `risk_assessment`. The structural fix is a SECOND Anthropic call with a risk-specific system message + prompt + JSON return, parsed independently.
- The lite Risk Judge prompt must enforce independence via three risk axes (volatility, concentration, valuation) -- NOT validation of the trader's recommendation (ATLAS arXiv 2510.15949 + EvidentlyAI rubric pattern).
- Cost impact: ~$0.003/ticker for the extra Sonnet call (well under existing $0.01/ticker accounting at line 768; no budget-field change needed). Prompt caching not viable at these prompt sizes (~350 tokens).
- Consumer compatibility: `signal_attribution.py:139-142` (`is_lite_dup`) auto-resolves to False once `recommended_position_pct > 0`. `portfolio_manager.py:272` uses `.get()`. `bq.save_report` reads `risk_assessment.get("reason", "")` at line 818 -- preserve the `reason` key as an alias of `reasoning` for backward compat.
- Use `re.search(r'\{.*\}', risk_text, re.DOTALL)` to parse the nested risk_limits object (existing `r'\{[^}]+\}'` cannot capture nested JSON).

## Hypothesis

Adding a second, independent LLM call inside `_run_claude_analysis` --
with a risk-specific system message that asks the model to evaluate
volatility/concentration/valuation axes and derive position size FROM
THOSE AXES (not from the trader's recommendation) -- closes the
aliasing bug at line 765 without changing any downstream consumer.
The cosmetic patch at `signal_attribution.py:131-154` becomes inert
naturally (because `risk_weight > 0` after this fix) and will be
removed by step 25.B.

## Success criteria (verbatim from masterplan)

1. `risk_assessment_reasoning_distinct_from_analysis_reason`
2. `risk_weight_greater_than_zero_for_lite_path`
3. `second_llm_call_with_risk_specific_prompt_invoked`

Verification command (immutable):
`source .venv/bin/activate && python3 tests/verify_phase_25_A.py`

Live check (per masterplan):
`BQ paper_trades signals column shows distinct trader_rationale vs risk_rationale text on next cycle`

## Plan

1. **Backend** -- `backend/services/autonomous_loop.py::_run_claude_analysis`:
   - Insert risk-specific system message + prompt as module-level constants `_LITE_RISK_JUDGE_SYSTEM` and `_LITE_RISK_JUDGE_TEMPLATE` (string `.format()` template; no f-string sharing with trader prompt -- the two prompts must be visibly distinct).
   - After the existing trader-call parse (line ~753), make a SECOND `await asyncio.to_thread(client.messages.create, ...)` with the risk system message + formatted risk prompt. Use the same `model_name` (Claude) so cost accounting stays unchanged.
   - Parse the risk response with `re.search(r'\{.*\}', risk_text, re.DOTALL)` (handles nested `risk_limits` object). Fall back to a documented default `{decision: "APPROVE_REDUCED", recommended_position_pct: 3.0, risk_level: "MODERATE", reasoning: "<parse-fail diagnostic>"}` so the downstream pipeline never sees a None or empty dict.
   - Replace line 765's `"risk_assessment": {"reason": analysis["reason"]}` with the 6-field dict: `{decision, reasoning, reason (alias of reasoning), recommended_position_pct, risk_level, risk_limits}`.
   - Log the risk decision + position pct via `logger.info` (mirrors trader log at line 755).
2. **Verifier** -- `tests/verify_phase_25_A.py` -- minimum 8 claims:
   - Claim 1: `_LITE_RISK_JUDGE_SYSTEM` string constant exists and mentions all three risk axes (volatility, concentration, valuation).
   - Claim 2: `_LITE_RISK_JUDGE_TEMPLATE` string constant exists and the prompt is structurally distinct from the trader prompt (different question shape).
   - Claim 3: `_run_claude_analysis` makes >=2 `client.messages.create` calls (grep-level mutation-resistance).
   - Claim 4: `re.search(r'\{.*\}', risk_text, re.DOTALL)` (or `re.S`) pattern used to parse the risk JSON.
   - Claim 5: The return dict's `risk_assessment` contains `decision`, `reasoning`, `recommended_position_pct`, `risk_level`, `risk_limits` keys (grep over the source file).
   - Claim 6: **Behavioral round-trip**: monkey-patch `anthropic.Anthropic` with a fake that returns DIFFERENT text for trader-vs-risk calls. Call `_run_claude_analysis(ticker=...)`; assert that the returned `risk_assessment.reasoning` is distinct from `analysis.reason` AND that `recommended_position_pct > 0`.
   - Claim 7: **Behavioral fallback**: monkey-patch the second call's response to return malformed JSON; assert the function still returns a `risk_assessment` dict with `decision == "APPROVE_REDUCED"` and `recommended_position_pct > 0` (default).
   - Claim 8: `signal_attribution.extract_signal_stack` over the new return dict produces a `RiskJudge` row with `weight > 0` and rationale != trader_rationale (this validates the consumer-side bridge).
   - Claim 9: `_LITE_RISK_JUDGE_SYSTEM` includes the verbatim independence directive "NOT to validate the trader's recommendation".

## Non-goals

- No removal of the cosmetic patch at `signal_attribution.py:131-154` -- that's 25.B's job.
- No new BQ schema migration; existing `paper_trades.signals` JSON column already absorbs the new rationale field shape via signal_attribution's serializer.
- No change to `_run_single_analysis` non-Claude fallback (Gemini path runs through the orchestrator which already has a real Risk Judge).
- No prompt caching infrastructure (research determined it's not viable at these token sizes).

## References

- `handoff/current/research_brief.md` -- full brief this cycle
- `backend/services/autonomous_loop.py:700-786` (`_run_claude_analysis` edit site, line 765 the aliasing bug)
- `backend/services/autonomous_loop.py:818` (`bq.save_report` reads `risk_assessment.get("reason", "")` -- alias preserved)
- `backend/services/signal_attribution.py:117-155` -- downstream consumer, `is_lite_dup` auto-resolves
- `backend/services/portfolio_manager.py:272` -- reads `recommended_position_pct`
- `backend/agents/risk_debate.py:253-310` -- full-path Risk Judge prompt + return shape (reference for the lite single-call analogue)
- CLAUDE.md `Critical Rules` -- LLM API costs require explicit approval; this change uses an existing API key + an extra ~$0.003/ticker call accounted within the existing $0.01 cost field; no new approval needed
