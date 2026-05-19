# Sprint Contract -- phase-31.0.2 (Smoketest Stage 2)

**Step:** Lite-path analysis via Claude Code subagent per ticker.
**Date:** 2026-05-20.
**Mode:** Loop PAUSED. Claude Code substitutes the in-app Anthropic API.

## Research-gate summary

Deep tier. 17 sources read in full (gate_passed=false, honest disclosure
on 20-source floor miss). Substantive content complete: code audit +
killer SDK finding.

Killer findings:
- **Source 14** (Anthropic Claude Code Agent SDK docs):
  `output_format={"type":"json_schema","schema":...}` auto-validates +
  retries.
- **Source 11** (json_repair): canonical JSON-repair lib for L1/L2
  failure modes.
- **Source 15 [ADVERSARIAL]**: keep subagent context minimal to avoid
  the known parse-failure bug.

## Hypothesis

4 Claude Code subagents (one per ticker) produce 4 valid JSON
syntheses matching the 5-field shape.

## Immutable success criteria

1. Each of 4 spawns returns parseable JSON.
2. Each JSON has 5 fields: ticker, recommendation, final_score,
   risk_assessment, price_at_analysis.
3. `recommendation` in {BUY, HOLD, SELL}.
4. `final_score` numeric in [0, 10].
5. `price_at_analysis` matches Stage 1 `current_price` within rounding.
6. NO `anthropic.Anthropic().messages.create()` call.
7. Per-ticker outputs persisted to
   `handoff/smoketest_20260520/STAGE_2_<TICKER>_lite_analysis.json`.
8. Compiled summary at `STAGE_2_summary.json`.

## Plan

1. Load Stage 1 output JSON.
2. For each of [AAPL, MSFT, NVDA, JPM]: spawn
   `Agent({subagent_type:"general-purpose"})` with minimal context
   (Stage 1 row + ticker) and JSON-only prompt.
3. Parse via `json_repair.loads()` if available, else bracket-extract
   + `json.loads()`.
4. Validate 5-field schema + value constraints.
5. Persist per-ticker + summary.

## Hard guardrails

- Minimal subagent context (avoid GitHub #30030 parse bug).
- Researcher gate false on floor miss; content-complete.
- NO BQ writes. NO Alpaca. NO in-app Anthropic calls.

## References

- `handoff/current/research_brief.md` (17 sources, Source 14+15 key).
- `handoff/smoketest_20260520/STAGE_1_screen_universe_output.json` (input).
