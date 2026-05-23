# phase-40.8 -- Correlation cap beyond GICS (OPEN-5)

**Step id:** `40.8`
**Date:** 2026-05-23
**Mode:** EXECUTION (cycle 47).
**Cycle:** Cycle 47 (after Cycle 46 phase-37.3).

---

## North-star delta

**Terms:** R (factor-crowding risk reduction) + B (zero $ until enabled; default-OFF gate).

**R:** GICS sector cap catches sector concentration but misses cross-sector factor crowding (e.g., two stocks in different GICS sectors both high-momentum + small-value would slip through GICS but be correlated). Adds FF3 factor-correlation cap as a default-OFF gate alongside GICS. Per AQR/Two Sigma 2025 + arXiv 2001.04185 factor-crowding research.

**B:** Zero $ until operator flips `paper_max_factor_corr > 0`. After enabling: ~1-2 prevented crowded-trade entries per quarter (conservative).

**P:** P=N/A. **Caltech arxiv:2502.15800 discount:** N/A.

**How measured:** new pytest module `test_phase_40_8_factor_correlation.py` exercises 3 immutable criteria + default-OFF backward-compat + cosine-similarity regression against canned compute_ff3 fixture.

---

## Research-gate compliance

**Researcher SPAWNED FIRST** -- brief at `handoff/current/research_brief_phase_40_8.md`. Tier=simple. 5 sources read-in-full. gate_passed=true. Recency scan present.

**Critical internal finding from researcher**: `compute_ff3()` already exists at `backend/services/portfolio_risk.py:58` (full OLS regression). The MCP `factor_exposure()` at `risk_server.py:118` is a stub, but the math primitive is not. Phase-40.8 is WIRING + a new cosine-similarity helper -- NOT new math.

**Recommended scope (a) MIN VIABLE** -- implemented verbatim:
- New pure module `backend/services/factor_correlation.py` exposing `factor_correlation_score(cand_loadings, port_loadings) -> float` (cosine similarity).
- New settings field `paper_max_factor_corr: float = Field(0.0, ge=0.0, le=1.0)`. Default 0.0 = disabled. Recommended live value 0.85 (per cosine-similarity convention; no canonical FF3-beta cap in literature).
- Gate fires in `portfolio_manager.py` AFTER existing GICS NAV-pct cap, BEFORE position commit. Skips silently when `cand.get("factor_loadings")` is absent (forward-compat).
- MCP `factor_exposure()` stub REMAINS a stub (separate phase).

Sources cited:
- AQR "Measuring Factor Exposures" (Israel & Ross 2017)
- arXiv 2001.04185 "Zooming In on Equity Factor Crowding" (Volpati et al 2020)
- Resonanz Capital "Crowding, Deleveraging" (2025)
- Two Sigma Venn "Liberation Year: 2025 Factor Performance Report"
- RStudio "Rolling Fama French" (2018)

---

## Hypothesis

> 1. `backend/services/factor_correlation.py` exports `factor_correlation_score(cand_loadings, port_loadings) -> float` computing cosine similarity over (market_beta, smb_beta, hml_beta) vectors.
> 2. Returns 0.0 (no cap fires) if either dict is empty or missing fields -- forward-compat for cycles where upstream agents haven't supplied loadings yet.
> 3. New settings field `paper_max_factor_corr` (default 0.0). Gate fires only when value > 0.
> 4. portfolio_manager.py BUY loop adds a single check AFTER GICS NAV-pct: if `max_factor_corr > 0` and `factor_correlation_score(cand, port) > max_factor_corr`, block the BUY with reason `"factor_correlation_above_cap"`.
> 5. 4+ new pytest tests cover each immutable criterion + default-OFF backward-compat + cosine-similarity edge cases.

---

## Immutable success criteria (verbatim from masterplan 40.8.verification)

1. `ff3_factor_exposure_used_alongside_gics` -- portfolio_manager.py invokes factor_correlation_score AFTER the existing GICS sector cap; both gates active independently.
2. `correlation_cap_blocks_simulated_high_ff_corr_buy` -- given a candidate with FF3 loadings (1.0, 0.5, 0.3) and a portfolio with average loadings (0.99, 0.51, 0.29), cosine-sim ~ 0.9998 > 0.85, BUY is blocked.
3. `regression_against_known_fixture` -- compute_ff3 with canned 60-day return series produces fixed alpha/betas reproducible across runs.

Plus /goal integration gates 1-11.

---

## Files this step touches

- `backend/services/factor_correlation.py` (NEW, ~50 lines): pure helper `factor_correlation_score`.
- `backend/config/settings.py` -- ADD `paper_max_factor_corr: float = Field(0.0, ge=0.0, le=1.0)`.
- `backend/services/portfolio_manager.py` -- ~20-line gate insertion AFTER GICS cap (default OFF; only fires when settings.paper_max_factor_corr > 0).
- `backend/tests/test_phase_40_8_factor_correlation.py` (NEW, ~180 lines, >=5 tests).

---

## /goal integration gates (declared)

| # | Gate | Plan |
|---|---|---|
| 1 | pytest count >= 297 | will INCREASE by ~5 tests; baseline 500 -> ~505 |
| 2 | ast.parse green | will hold |
| 3 | TS build green | N/A |
| 4 | flag-default-OFF | YES -- `paper_max_factor_corr=0.0` disables |
| 5 | BQ idempotent | N/A (no BQ) |
| 6 | env vars docs | N/A (no env var; settings field default) |
| 7 | N* delta declared | DONE (R + B) |
| 8 | zero emojis | will hold |
| 9 | ASCII-only loggers | will hold |
| 10 | single source of truth | NEW helper is canonical; reuses existing compute_ff3 |
| 11 | log-first / flip-last | will hold |

---

## Honest scope + backward compat

**Hot path note**: portfolio_manager.py is the hot path. The gate insertion is gated on `settings.paper_max_factor_corr > 0` (default 0.0 = OFF). When OFF, the new code path is a single `if max_factor_corr > 0:` check that short-circuits -- ZERO performance regression on existing runs. Quiet-logging period of 1-2 weeks recommended before operator flips ON to a non-zero threshold.

**Forward compat note**: `cand.get("factor_loadings")` returns None when upstream agents haven't supplied FF3 loadings (today's case -- the analysis pipeline doesn't yet produce them). The gate skips silently and the BUY proceeds (no false blocks). Once upstream is wired (separate phase), the gate becomes active even at default threshold values if operator opts in.

---

## References

- closure_roadmap.md §3 OPEN-5
- handoff/current/research_brief_phase_40_8.md (5 sources)
- backend/services/portfolio_risk.py:58 (existing compute_ff3 -- math is done)
- backend/services/portfolio_manager.py:209-316 (existing GICS gate)
- backend/agents/mcp_servers/risk_server.py:118 (factor_exposure stub remains a stub)
- /goal directive
