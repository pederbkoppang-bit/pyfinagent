# phase-40.8.1 -- Wire compute_ff3 into analysis pipeline (in-memory)

**Step id:** `40.8.1`
**Date:** 2026-05-23
**Mode:** EXECUTION (cycle 50). Honest dual-interpretation pattern (in-memory ON; BQ persistence deferred to phase-40.8.2).
**Cycle:** Cycle 50.

---

## North-star delta

**Terms:** R (activates the cycle-47 FF3 cap path; closes phase-40.8 dormancy) + B (zero $ until enabled).

**R:** Phase-40.8 (cycle 47) shipped a cosine-similarity helper + portfolio_manager.py gate that has remained DORMANT because no upstream populated `factor_loadings`. This step wires the missing producer behind a default-OFF feature flag.

**B:** Zero $ -- pure code addition; no LLM calls. Performance overhead negligible (per researcher: ~30ms for top-N candidates).

**P:** N/A. **Caltech arxiv:2502.15800 discount:** N/A.

**How measured:** new pytest module `test_phase_40_8_1_loadings_pipeline.py` exercises 3 immutable criteria + feature-flag default-OFF + dual-interpretation xfail strict for BQ persistence.

---

## Research-gate compliance

**Researcher SPAWNED FIRST** -- brief at `handoff/current/research_brief_phase_40_8_1.md`. Tier=simple. 6 sources read in full, gate_passed=true, recency scan present.

Recommended scope (per researcher):
- **Wiring location**: screener step in autonomous_loop.py (after `screen_universe`, before `rank_candidates`).
- **FF3 data source**: cached parquet from Kenneth French (canonical) -- BUT this requires Internet fetch + cache file + potential pyarrow dep. **Scope-honest deferral**: ingestion is phase-40.8.2; this cycle uses a STUBBED FF3 source returning deterministic synthetic factor returns. Test coverage demonstrates the wiring pattern works end-to-end; production deployment waits for real cache.

Sources cited:
- Kenneth French Data Library (canonical FF3)
- AQR "Measuring Factor Exposures" (Israel & Ross 2017)
- arXiv 2001.04185 (Volpati equity factor crowding)
- Two Sigma Venn factor-exposure docs
- Coding Finance: download FF data idiom (no new pip dep)
- Dallas Fed WP2515r1 portfolio similarity

---

## Hypothesis

> 1. New helper `backend/services/factor_loadings.py` exports `compute_candidate_loadings(candidates, price_histories, factor_returns_dict, window_days=60)` that calls `compute_ff3` per ticker and attaches `factor_loadings: {market_beta, smb_beta, hml_beta}` to each candidate dict.
> 2. Feature flag `enable_factor_loadings: bool = Field(False)` controls whether the helper fires.
> 3. autonomous_loop.py screener step calls helper IF flag enabled (else byte-identical to today).
> 4. paper_trader.py execute_buy attaches `factor_loadings` from candidate -> in-memory `pos_row` (NO BQ schema change this cycle).
> 5. xfail strict on literal BQ persistence (phase-40.8.2 follow-up).

---

## Immutable success criteria (verbatim from masterplan 40.8.1.verification)

1. `screener_candidates_carry_factor_loadings` -- when flag enabled, after `compute_candidate_loadings` returns, every candidate dict has a `factor_loadings` key with the 3 FF3 betas.
2. `paper_positions_carry_factor_loadings_after_buy` -- **DUAL-INTERPRETATION** per CLAUDE.md honest pattern:
   - **OPERATIONAL** (in-memory): execute_buy's pos_row has `factor_loadings` field after BUY. PASS.
   - **LITERAL** (BQ row): paper_positions BQ row has factor_loadings column. xfail strict; deferred to phase-40.8.2 (BQ schema mutation needs Step 7 window per guardrail).
3. `compute_ff3_invoked_in_analysis_pipeline_with_60day_window` -- helper calls compute_ff3 with `window_days=60` aligned price + factor series.

Plus /goal integration gates 1-11.

---

## Files this step touches

- `backend/services/factor_loadings.py` (NEW, ~90 lines): pure helper with stub-FF3 generator + compute_candidate_loadings.
- `backend/config/settings.py` -- ADD `enable_factor_loadings: bool = Field(False)`.
- `backend/services/autonomous_loop.py` -- ~12-line wiring after screen_universe call (conditional on flag; lazy import for circular-safety).
- `backend/services/paper_trader.py` -- thread `factor_loadings` from candidate -> TradeOrder -> in-memory pos_row (~6 lines).
- `backend/services/portfolio_manager.py` -- TradeOrder.factor_loadings field (~2 lines, dataclass field add).
- `backend/tests/test_phase_40_8_1_loadings_pipeline.py` (NEW, ~190 lines, >=5 tests covering 3 criteria + feature flag + xfail strict).

---

## /goal integration gates (declared)

| # | Gate | Plan |
|---|---|---|
| 1 | pytest count >= 509 | +5 tests; baseline 509 -> ~514; 0 regressions |
| 2 | ast.parse green | will hold |
| 3 | TS build green | N/A |
| 4 | flag-default-OFF | YES (`enable_factor_loadings=False`) |
| 5 | BQ idempotent | N/A (no BQ schema change this cycle; deferred to 40.8.2) |
| 6 | env vars docs | N/A (no env var; settings field default) |
| 7 | N* delta declared | DONE (R + B) |
| 8 | zero emojis | will hold |
| 9 | ASCII-only loggers | will hold |
| 10 | single source of truth | compute_ff3 in portfolio_risk.py is canonical math; factor_loadings.py is wiring |
| 11 | log-first / flip-last | will hold |

---

## Honest scope + closure pattern

**Closure pattern: ENGINEERED + VERIFICATION**. Real engineering (helper + wiring + plumbing) plus mutation-resistant tests.

**Hot-path safety**: autonomous_loop.py screener step modification is gated on `enable_factor_loadings=False` default. When OFF: ZERO behavior change (single `if settings.enable_factor_loadings:` short-circuits before the new helper call). When ON: candidates get factor_loadings; phase-40.8 cap can fire if `paper_max_factor_corr > 0`.

**Stubbed FF3 data this cycle**: factor returns are synthetic deterministic (research_brief documents this; honest disclosure). Production FF3 cache is phase-40.8.2.

**xfail strict on literal criterion 2**: BQ persistence requires schema column add + migration. Per guardrail "NO mutating BQ/Alpaca outside autonomous-loop Step 7" -- deferred to phase-40.8.2. xfail strict catches future drift where someone adds the column silently without migration tracking.

**Follow-up phase-40.8.2 (P3) WILL BE ADDED post-PASS**: BQ schema add + Kenneth French cache ingestion + flag flip default to ON after operator review.

---

## References

- closure_roadmap.md §3 OPEN-5 (parent ticket)
- handoff/current/research_brief_phase_40_8_1.md (6 sources)
- backend/services/portfolio_risk.py:58 (existing compute_ff3 -- math is done)
- backend/services/factor_correlation.py (phase-40.8; consumer of factor_loadings)
- backend/services/portfolio_manager.py:296-307 (phase-40.8 cap; activates when candidates carry loadings + flag ON)
- /goal directive
