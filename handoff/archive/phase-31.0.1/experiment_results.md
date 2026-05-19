# Experiment Results -- phase-31.0.1 (Smoketest Stage 1)

**Step:** Smoketest Stage 1 -- screen + rank + sector enrichment.
**Date:** 2026-05-20.
**Verdict:** **PASS.**

## Summary

Executed Test Design #3 (full production chain) per researcher
recommendation. 4 tickers returned with non-empty `sector` + numeric
`composite_score`. Output persisted to
`handoff/smoketest_20260520/STAGE_1_screen_universe_output.json` AND
`handoff/smoketest_20260520/STAGE_1_results.md`.

## Files touched / created

| Path | Lines | Role |
|------|-------|------|
| `handoff/smoketest_20260520/STAGE_1_screen_universe_output.json` | 52 | Stage 1 machine-readable output for downstream stages |
| `handoff/smoketest_20260520/STAGE_1_results.md` | 70 | Stage 1 human-readable results + assertions table |

**No backend code modified.** Stage 1 is pure code execution + yfinance
read.

## Result data

| Ticker | Sector | composite_score | current_price | momentum_3m | rsi_14 |
|--------|--------|-----------------|---------------|-------------|--------|
| NVDA | Technology | 15.283 | 220.61 | 17.36 | 58.5 |
| AAPL | Technology | 8.107 | 298.97 | 13.20 | 84.1 |
| MSFT | Technology | -1.557 | 417.42 | 4.70 | 45.4 |
| JPM | Financial Services | -3.986 | 295.70 | -3.75 | 36.9 |

## Assertions (all PASS)

1. `len(ranked) == 4` -> PASS
2. Tickers match request -> PASS
3. `current_price` numeric -> PASS
4. `composite_score` numeric -> PASS
5. `sector` non-empty string -> PASS
6. Schema invariants {ticker, current_price, composite_score, sector} -> PASS

## Hard guardrail attestation

- NO LLM calls (Stage 1 has no LLM step).
- NO production BQ writes.
- NO Alpaca calls.
- Loop STAYS PAUSED.

## Researcher gate disclosure

Researcher reported `gate_passed: false` honestly because 18 of 20
sources were fetched in full (paywalls/404 on 4 sources). The
substantive content (code audit + Test Design #3) is complete. Q/A
judges whether the floor miss counts as a substantive blocker or as
honest disclosure on a P3-equivalent smoke verification step.

## Success criteria check (Stage 1, from morning goal)

| Criterion | Status | Evidence |
|-----------|--------|----------|
| `screen_universe(4 tickers) -> dicts with sector + score` | PASS | 4 dicts returned; sector populated via yfinance enrichment (Design #3); composite_score from rank_candidates |
| NO production BQ writes | PASS | Only local JSON file + handoff markdown |
| NO LLM calls | PASS | No Anthropic / Gemini call in this stage |
| Loop PAUSED | PASS | kill_switch.paused==True throughout |
