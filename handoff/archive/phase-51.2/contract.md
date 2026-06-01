# Contract -- phase-51.2: sector diversification (research-recommended money lever) + candidate-lever measurement

**Step id:** 51.2 | **Priority:** P1 (money) | **depends_on:** 50.5
**Date:** 2026-06-01 | **harness_required:** true | **$0 LLM** | no pip | **measure-first, no live flag flip**

## Research-gate summary (PASSED -- TWO briefs)
- `handoff/current/research_rotation_element2_verdict.md` (rotation gate: 8 sources, gate_passed). VERDICT: REDIRECT away from winner-take-all rotation (architecturally disconnected from live money via RC-B; alt strategies LOSE money -6.13/-1.21/-0.59). The cited-evidence money path is breadth INSIDE the working momentum engine.
- `handoff/current/research_51_2_sector_div.md` (51.2 gate: 9 sources, gate_passed). DECISIVE:
  - **The fix is MINIMAL + the lever already exists.** `autonomous_loop.py:629` already passes `sector_neutral=settings.sector_neutral_momentum_enabled` into `rank_candidates`; the scoring (`screener.py:415-445`) is correct + gated (default OFF). The ONLY defect: `screen_universe(...)` at `autonomous_loop.py:369-374` is called WITHOUT `sector_lookup=`, so at rank time every candidate has `sector=None` -> all fall into the `_UNKNOWN_` global pool (`screener.py:425`) -> single percentile pool -> monotone transform -> **byte-identical to OFF (the no-op)**.
  - **CRITICAL CAVEAT (the single most important finding):** for a LONG-ONLY book (which pyfinagent is), Harvey et al. (Duke) find keeping across-sector exposure beats full sector-neutralization in **78% of trials**. So HARD within-sector-percentile replacement (`screener.py:438-440` overwrites the composite entirely) may HURT Sharpe. -> measure the SIGN on our own universe BEFORE any live enable; prefer a SOFT tilt if pursued.
  - **Higher-EV adjacent lever:** vol-scaling the momentum book (CFA/Barroso-Santa-Clara: ~doubles Sharpe, halves drawdown) -- larger + better-evidenced than sector-neutral. Measure it alongside.
  - **The ML backtest engine CANNOT measure this** (it ranks via `candidate_selector._rank_candidates`, a DIFFERENT formula, no sector field). Need a NEW screener-level replay reusing the PRODUCTION `rank_candidates` formula over historical dates.

## Hypothesis
Wiring `sector_lookup` into the first-pass screen makes the (already-built) sector-neutral lever functional at rank time; a $0 screener-level replay over historical dates -- comparing BASELINE vs SECTOR-NEUTRAL vs VOL-SCALED top-N baskets on our own S&P 500 universe -- measures the SIGN (Sharpe / sector-spread / turnover tradeoff) so we enable a diversification lever ONLY if the evidence (not folklore) supports it. With the flag default-OFF, the wiring is byte-identical live (the sector field is metadata not used in the OFF ranking path).

## Success criteria (IMMUTABLE -- verbatim from masterplan step 51.2)
1. candidates carry a sector label AT rank time (sector_lookup passed into the first-pass screen before rank_candidates), so the within-sector/sector-neutral ranking path is no longer a silent no-op
2. a backtest compares diversified/sector-neutral ranking ON vs OFF on the US universe and reports the Sharpe / return / sector-spread tradeoff (evidence-based, not assumed)
3. the change is config-gated and does NOT regress the working US momentum core (default behavior preserved unless the diversification flag is explicitly enabled)
4. live_check records the backtest ON-vs-OFF comparison + the resulting candidate sector distribution

**Verification command:** `pytest backend/tests/test_phase_51_2_sector_div.py` + `ast.parse(screener.py, autonomous_loop.py)` + `test -f live_check_51.2.md`.
**live_check:** REQUIRED -- the $0 replay ON-vs-OFF comparison (Sharpe/sector-spread/turnover) + the sector distribution; NO live flag flip (the flip is a later, evidence-gated, operator-confirmed step).

## Plan steps (GENERATE)
1. **Wiring (criterion #1):** build a `{ticker: sector}` map for the full screened universe and pass it as `sector_lookup=` into `screen_universe` at `autonomous_loop.py:369`. Source from a CHEAP static GICS map (avoid per-ticker yfinance latency on ~400-500 tickers per live cycle) -- prefer BQ `analysis_results.sector` / a static S&P 500 GICS dict; yfinance only as cold-cache fallback. Confirm: with `sector_neutral_momentum_enabled=False` (default), the ranked output is BYTE-IDENTICAL to today (the sector field is not used in the OFF path).
2. **Measurement (criterion #2) -- the core:** `scripts/ablation/sector_neutral_replay.py` (NEW). Reuse the PRODUCTION `rank_candidates` formula. For N historical monthly dates (S&P 500, BQ historical_prices for speed, $0): build `screen_data` rows (momentum_1m/3m/6m + RSI + vol + sector) as-of each date, then rank under 3 configs -- BASELINE (raw momentum), SECTOR_NEUTRAL (within-sector percentile), VOL_SCALED (vol-targeted basket) -- and score each top-N basket's realized forward-1mo return -> Sharpe, plus sector spread (# distinct GICS) + turnover. Emit a TSV/verdict (mirror the ablation runner shape). Gate framing: "keep sector-neutral" iff breadth rises (+>=2 sectors) AND Sharpe delta >= -0.05.
3. **Honest reporting:** if the replay shows sector-neutral HURTS (the long-only caveat), report it, keep the flag OFF (criterion #3), and recommend vol-scaling as the next lever -- criterion #2 is satisfied by the MEASUREMENT + tradeoff report, NOT by sector-neutral winning.
4. **Tests:** `backend/tests/test_phase_51_2_sector_div.py` -- assert (a) with sector_lookup + flag OFF the ranked order is byte-identical to no-sector-lookup (byte-identity), (b) with flag ON + a constructed multi-sector screen_data the top-N spreads across >1 sector (the no-op is fixed), (c) the replay scorer's basket-Sharpe/turnover math on a tiny fixture.
5. **Verify:** pytest; run the replay (capture Sharpe/sector-spread/turnover per config) into `live_check_51.2.md`; confirm US live ranking byte-identical with flag OFF.
6. **EVALUATE:** fresh qa. Then harness_log.md (LAST), then flip masterplan 51.2 -> done.

## Safety / scope notes
- **No live flag flip in this step.** The flag stays default-OFF; enabling it live is a later, evidence-gated, operator-confirmed action. This step makes the lever FUNCTIONAL + MEASURES it. Honors "measure before fixing."
- **Byte-identity:** with `sector_neutral_momentum_enabled=False`, passing `sector_lookup` adds a metadata field unused by the OFF ranking path -> identical live behavior (test-proven). The only live-path cost is the sector-map build -> use a cheap static map, not per-ticker yfinance.
- **If sector-neutral measures NEGATIVE on our long-only universe (Harvey 78% case):** do NOT enable; report + pivot to vol-scaling (a separate future step). Prefer a SOFT tilt over hard replacement if sector-div is pursued live.
- $0 LLM; no pip; no spend; no DROP/DELETE; no live trading change.

## References
- handoff/current/research_51_2_sector_div.md + research_rotation_element2_verdict.md
- backend/services/autonomous_loop.py:369 (wiring site), :629 (rank_candidates call), :659-676 (today's post-rank enrichment)
- backend/tools/screener.py:69 (screen_universe sector_lookup arg), :206-213 (attaches sector), :268-282 (momentum), :415-445 (sector_neutral percentile)
- backend/api/paper_trading.py:1058 (_fetch_ticker_meta sector source); scripts/ablation/run_ablation.py (TSV/verdict shape to mirror)
- Harvey et al. "Is Sector Neutrality a Mistake?" (long-only 78%); Moskowitz-Grinblatt industry momentum; CFA Institute Dec 2025 + Barroso-Santa-Clara (vol-scaling)
