# Experiment Results -- phase-31.0.2 (Smoketest Stage 2)

**Step:** Stage 2 -- Lite-path analysis via Claude Code subagent per ticker.
**Date:** 2026-05-20.
**Verdict:** **PASS.**

## Summary

4 Claude Code subagents (`Agent({subagent_type:"general-purpose"})`)
spawned in parallel, one per ticker [AAPL, MSFT, NVDA, JPM]. Each
returned valid JSON matching the 5-field shape. All 4 schema checks
passed. Recommendation distribution: 1 BUY (NVDA), 3 HOLD.

**Substitution rule honored**: NO `anthropic.Anthropic().messages.create()`
call. Each subagent received MINIMAL context per source 15
[ADVERSARIAL] mitigation.

## Files touched / created

| Path | Lines | Role |
|------|-------|------|
| `handoff/smoketest_20260520/STAGE_2_AAPL_lite_analysis.json` | 1 | AAPL synthesis |
| `handoff/smoketest_20260520/STAGE_2_MSFT_lite_analysis.json` | 1 | MSFT synthesis |
| `handoff/smoketest_20260520/STAGE_2_NVDA_lite_analysis.json` | 1 | NVDA synthesis |
| `handoff/smoketest_20260520/STAGE_2_JPM_lite_analysis.json` | 8 | JPM synthesis |
| `handoff/smoketest_20260520/STAGE_2_summary.json` | ~50 | Compiled summary + validation |
| `handoff/smoketest_20260520/STAGE_2_results.md` | ~90 | Human-readable results |

**No backend code modified.**

## Result data (table)

| Ticker | Recommendation | final_score | price_at_analysis | Reasoning headline |
|--------|----------------|-------------|-------------------|--------------------|
| AAPL | HOLD | 7.2 | 298.97 | RSI 84.1 overbought |
| MSFT | HOLD | 5.2 | 417.42 | Composite -1.557 neutral |
| NVDA | BUY | 8.7 | 220.61 | Top composite + momentum |
| JPM | HOLD | 3.8 | 295.70 | Composite -3.986 negative |

## Hard guardrail attestation

- NO `anthropic.Anthropic()` call.
- NO production BQ writes.
- NO Alpaca calls.
- 4 subagents spawned in parallel, each with minimal context.

## Success criteria check (Stage 2)

| Criterion | Status | Evidence |
|-----------|--------|----------|
| 4 spawns return parseable JSON | PASS | All 4 `json.loads()` succeeded |
| 5 required fields present | PASS | Validation script confirmed |
| recommendation in {BUY,HOLD,SELL} | PASS | 1 BUY + 3 HOLD |
| final_score numeric in [0,10] | PASS | range 3.8 to 8.7 |
| price_at_analysis matches Stage 1 | PASS | within $0.01 tolerance |
| NO anthropic.Anthropic call | PASS | Subagents only |
| Per-ticker JSON files persisted | PASS | 4 files in smoketest dir |
| Compiled summary persisted | PASS | STAGE_2_summary.json |

## Claude vs Gemini (placeholder)

Stage 3 will compare the Claude Code-subagent lite-path output for NVDA
against the Gemini 28-agent full-path output. Per-ticker delta table
deferred to `STAGE_2_VS_3_delta.md` after Stage 3.
