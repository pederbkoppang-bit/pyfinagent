---
step: phase-25.A
cycle: 69
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_A.py'
title: Decouple RiskJudge with independent LLM call in lite path (P1)
audit_basis: phase-24.4 F-1 (autonomous_loop.py:765 aliased risk_assessment.reason = analysis["reason"])
---

# Experiment Results -- phase-25.A

## Code changes

### `backend/services/autonomous_loop.py`
- New module-level constants (above `_run_claude_analysis`):
  - `_LITE_RISK_JUDGE_SYSTEM` -- enforces independence with the verbatim directive "NOT to validate the trader's recommendation"; references three risk axes (VOLATILITY, CONCENTRATION, VALUATION).
  - `_LITE_RISK_JUDGE_TEMPLATE` -- structurally distinct from the trader prompt; returns 5-field JSON with `risk_limits` nested object.
  - `_LITE_RISK_DEFAULT` -- safe fallback dict (`APPROVE_REDUCED`, `recommended_position_pct=3.0`, `risk_level=MODERATE`, stop_loss_pct=10, max_drawdown_pct=15).
- Inside `_run_claude_analysis`, after the trader parse + log:
  - Builds risk prompt via `_LITE_RISK_JUDGE_TEMPLATE.format(...)`.
  - Makes a SECOND `await asyncio.to_thread(client.messages.create, ...)` call with the risk system message (so `_run_claude_analysis` now invokes `client.messages.create` twice).
  - Parses via `re.search(r"\{.*\}", risk_text, re.DOTALL)` to handle the nested `risk_limits` object.
  - On any exception or no-JSON-match, falls back to `_LITE_RISK_DEFAULT` and logs a warning.
  - Builds a structured `risk_assessment` dict with `decision`, `reasoning`, `reason` (alias of reasoning for `bq.save_report` backward compat), `recommended_position_pct`, `risk_level`, `risk_limits`.
  - logger.info now emits the risk decision + position pct.
- Return dict line ~765 now binds `"risk_assessment": risk_assessment` (was `{"reason": analysis["reason"]}` -- the aliasing bug).

### `tests/verify_phase_25_A.py` (new file)
- 10 immutable claims combining grep-level structure checks and BEHAVIORAL round-trips:
  - Claims 1-5, 10: structural -- system/template constants, three risk axes, >=2 `client.messages.create` calls in the function, re.DOTALL parse, presence of all 5 risk_assessment keys, removal of the old aliased line, verbatim independence directive.
  - Claim 6: behavioral -- monkey-patches `anthropic.Anthropic` with a fake that returns DIFFERENT text for trader vs risk calls (gated on the `system` kwarg only the risk call passes). Asserts `risk_assessment.reasoning != analysis.reason`, `recommended_position_pct > 0`, and exactly 2 `messages.create` calls.
  - Claim 7: pulls `recommended_position_pct` from the behavioral round-trip and asserts > 0.
  - Claim 8: fallback behavioral -- malformed risk JSON in the response. Asserts `decision == "APPROVE_REDUCED"` and `position_pct > 0` (the safe default).
  - Claim 9: consumer-side -- calls `signal_attribution.extract_signals_from_analysis` over the new return shape; asserts a RiskJudge row exists with `weight > 0`, rationale != trader rationale, and `lite_path` flag is NOT set (the `is_lite_dup` patch auto-resolves to False after the fix).

## Verbatim verifier output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_A.py
Lite risk judge for TEST: no JSON in response -- using default sizing
PASS: risk_judge_system_constant_present_with_three_axes
PASS: risk_judge_template_constant_present
PASS: second_llm_call_with_risk_specific_prompt_invoked
PASS: risk_json_parse_uses_re_dotall
PASS: risk_assessment_reasoning_distinct_from_analysis_reason
PASS: behavioral_distinct_trader_vs_risk_call_and_position_pct_positive
PASS: risk_weight_greater_than_zero_for_lite_path
PASS: behavioral_malformed_risk_json_falls_back_to_safe_default
PASS: signal_attribution_consumer_emits_distinct_risk_row_with_weight
PASS: risk_judge_independence_directive_verbatim

10/10 claims PASS, 0 FAIL
```

(The "Lite risk judge for TEST: no JSON in response" line is emitted by the malformed-JSON fallback test; it proves the fallback path runs and the logger.warning fires as designed.)

## Backend gates

- `python -c "import ast; ast.parse(open('backend/services/autonomous_loop.py').read())"` -- OK
- Behavioral round-trip (claim 6) actually runs the coroutine end-to-end with mocked anthropic + yfinance -- proves the two-call shape, not just grep matches.
- Behavioral fallback (claim 8) exercises the `_LITE_RISK_DEFAULT` path with malformed JSON.

## Hypothesis verdict

CONFIRMED. The second LLM call is invoked with a risk-specific system message; the `risk_assessment` dict now carries 6 fields populated from the independent risk-judge response; the aliased line at the old position 765 is gone. The cosmetic `is_lite_dup` patch at `signal_attribution.py:139-142` auto-resolves to False because `recommended_position_pct > 0` after the fix (claim 9 covers this consumer-side bridge). Step 25.B (removing the cosmetic patch entirely) is unblocked.

## Cost impact

Two Anthropic calls per ticker instead of one. Sonnet 4.6 sizing: ~350 input + ~120 output tokens for the risk call; estimated $0.003/ticker. Total per ticker ~$0.004, well under the existing `total_cost_usd: 0.01` field at line 768. No budget configuration change.

## Live-check

Per masterplan: "BQ paper_trades signals column shows distinct trader_rationale vs risk_rationale text on next cycle".

Live evidence will land in `handoff/current/live_check_25.A.md`. Captured after next autonomous_loop cycle runs through a real BUY/HOLD/SELL flow; expected: `paper_trades.signals` JSON has BOTH a `Trader` row AND a `RiskJudge` row with distinct rationale strings + `RiskJudge.weight > 0`.

## Non-regressions

- `_run_claude_analysis` signature unchanged.
- Return dict shape unchanged except `risk_assessment` field is now a 6-field dict instead of a 1-field dict (alias `reason` key preserved for `bq.save_report` at line 818).
- `signal_attribution.extract_signals_from_analysis` consumer requires no edit; the `is_lite_dup` patch becomes inert by data, not by code (will be physically removed in 25.B).
- `portfolio_manager.py:272` already used `.get("recommended_position_pct")` -- now returns a real value.
- No new BQ schema; no migration.

## Next phase

Q/A pending.
