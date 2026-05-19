# Sprint Contract -- phase-31.0.1 (Smoketest Stage 1)

**Step:** phase-31.0.1 -- Smoketest Stage 1: screen + rank + sector-enrich for 4-ticker basket.
**Date:** 2026-05-20.
**Mode:** Smoketest. Loop STAYS PAUSED. NO production BQ writes.

## Research-gate summary

Researcher deep tier delivered to `handoff/current/research_brief.md`.
Envelope: **18 sources read in full** (floor 20; honest gate_passed=false
disclosure due to paywalled 403/404 responses on 4 sources). 26 URLs,
[ADVERSARIAL] tag present (Wright Research 2025 momentum-decay
contradicts naive long-momentum signal). Content-complete despite
floor miss: code audit verified verbatim + Test Design #3 fully
specified.

Code-audit verbatim (research_brief.md lines 23-82):
- `screen_universe` returns dicts WITHOUT `sector` and WITHOUT
  `composite_score` by default.
- `composite_score` is added by `rank_candidates`
  (`screener.py:370`).
- `sector` is enriched POST-rank by `_fetch_ticker_meta`
  (`autonomous_loop.py:579-596`).
- Production caller (`autonomous_loop.py:305-310`) does NOT pass
  `sector_lookup` to `screen_universe`.

**Test design: #3 (full production chain).** screen -> rank -> sector
enrichment. Mirrors production exactly per researcher recommendation.

## Hypothesis

`screen_universe(["AAPL","MSFT","NVDA","JPM"], period="6mo")` ->
`rank_candidates(..., top_n=4, strategy="momentum")` -> sector
enrichment via yfinance.Ticker.info produces 4 candidate dicts with
non-empty `sector` AND numeric `composite_score`.

## Immutable success criteria (per morning goal Stage 1)

```
1. screen_universe(["AAPL","MSFT","NVDA","JPM"]) -> 4 dicts.
2. Each dict has non-empty `sector` (post-enrichment).
3. Each dict has a numeric `composite_score` (post-rank).
4. Output persisted to handoff/smoketest_20260520/STAGE_1_screen_universe_output.json.
5. NO production BQ writes; NO LLM calls; NO Alpaca calls.
```

## Plan

1. Run Test Design #3 inline as a Python script (NOT pytest -- this
   uses live yfinance; treat as integration smoketest, not unit
   test). Persist the post-enrichment list to
   `handoff/smoketest_20260520/STAGE_1_screen_universe_output.json`.
2. Write per-stage results to
   `handoff/smoketest_20260520/STAGE_1_results.md` with PASS/PARTIAL/FAIL.
3. Spawn `qa` ONCE for verdict; circuit-breaker max 2 fresh retries.

## Hard guardrails

- NO production BQ writes. NO LLM calls. NO Alpaca calls.
- Live yfinance is permitted (read-only data fetch, no orders).
- Output is local file only.
- Stage 1 verifies shape, NOT strategy quality (per researcher
  citation of Sealos ML smoke-test discipline).
- Researcher gate is `false` on the 20-source floor; content
  completeness compensates. Q/A may judge this as PASS-with-NOTE,
  CONDITIONAL, or FAIL depending on strictness.

## References

- `handoff/current/research_brief.md` -- deep brief, 18 sources, gate honest=false.
- `backend/tools/screener.py:64-72, 179-201, 370` -- function signatures.
- `backend/services/autonomous_loop.py:305-310, 541-596` -- production
  caller chain.
- Morning goal Stage 1 spec.
