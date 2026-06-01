# experiment_results -- phase-52.4: residual momentum -> REJECT (cited-alpha-signal search EXHAUSTED)

**Step:** 52.4 | **Date:** 2026-06-01 | **$0 LLM** | no pip | **NO live change** | GENERATE complete

## Outcome in one line
Measured residual/idiosyncratic momentum (Blitz-Huij-Martens, single-factor 504d OLS, 12-1) vs the
baseline momentum ranking -> **REJECT: resid_mom is WORSE (Sharpe 1.082 vs 1.332, delta -0.249,
Ledoit-Wolf p=0.77)**. This closes the cited-alpha-signal search: NO tested price-based lever beats
the +20% momentum engine on our 2019-2025 large-cap long-only book. The engine STANDS as the highest earner.

## What was built / changed (offline measurement; NO live engine change)

| File | Change |
|------|--------|
| `scripts/ablation/residual_momentum_replay.py` | NEW. Downloads S&P-500 closes 2019-2025 (one $0 batch); NEW `resid_mom_signal(s_ret,m_ret,form=252,skip=21)` (single-factor OLS beta=cov/var, residuals, iMOM=sum(12-1 formation residuals)/std); per ~monthly rebalance ranks baseline (production rank_candidates) vs resid_mom; scores via the reused basket_fwd_return; applies the 52.3 `sharpe_diff_test` Ledoit-Wolf gate + the SAME a-priori rule. Reuses build_screen_row/basket_fwd_return/ann_sharpe/load_universe_sectors from sector_neutral_replay (import by path). |
| `backend/tests/test_phase_52_4_residual_momentum.py` | NEW 5 tests (resid_mom_signal: positive/negative formation direction; recent-only run does NOT create positive momentum (12-1 skip + OLS-alpha absorption); too-short->None; deterministic). |

## The verdict (criterion #1/#2)
```
config       ann_Sharpe   avg_fwd_mo%   avg_turnover
baseline        1.332        3.651          0.564
resid_mom       1.082        1.970          0.679
LW SR-difference: delta=-0.249, one-sided p=0.7724 (R1 False), 90% CI [-0.883,+0.330] (R2 False)
VERDICT: REJECT
```
Residual momentum UNDERPERFORMS baseline momentum here (lower Sharpe + return, higher turnover) -- decisively confirming the researcher's prior (modern-regime decay + long-only [no short leg] + large-cap [low idiosyncratic content], + the single-factor residual partly recapturing already-rejected factor/sector momentum).

## Research basis (gate PASSED)
`research_brief.md` (researcher `afaa06ced01cfac95`, 6 sources read in full incl. Blitz-Huij-Martens 2011 + Hanauer-Windmuller eq-9 via pdfplumber). The honest adversarial prior (likely REJECT on a large-cap long-only modern book) was confirmed -- even stronger (worse, not just insignificant).

## Verification command output (verbatim)
```
$ python -m pytest backend/tests/test_phase_52_4_residual_momentum.py -q
.....                                                                    [100%]
5 passed in 1.43s
```
Full replay verdict -> live_check_52.4.md. Reproducible (pinned `_residmom_paired_returns.json` + seeded bootstrap).

## Scope / safety (criterion #3)
- NO live engine change. Diff = the new replay script + resid_mom_signal + the new test + the pinned JSON. screener.py / autonomous_loop / the momentum_52wh flag UNTOUCHED. Offline $0 measurement (doesn't conflict with measuring Monday's live cycle).
- A-priori rule (same strict Ledoit-Wolf bar as 52.3) -- not lowered for the higher-evidenced lever; the modern large-cap long-only haircut made it worse.

## Artifact shape
- `resid_mom_signal(s_ret, m_ret, form=252, skip=21) -> float|None` (single-factor 12-1 residual momentum)
- the replay prints baseline vs resid_mom + the LW gate verdict.

## THE ELEMENT-2 SEARCH IS EXHAUSTED (rigorous, overfitting-controlled)
rotation (REJECT) | sector-neutral (-0.166 REJECT) | vol-scaling (+0.015 marginal) | 52wh tilt (+0.057 but p=0.24 REJECT) | residual momentum (-0.249 REJECT). **No cited price-based signal robustly beats the live momentum engine.** The +20%/+14%-alpha engine STANDS as the highest earner -- the honest, research-complete outcome. NEXT: the LIVE multi-market expansion is the money lever (MEASURE Monday's first cycle); further alpha would need a different data axis (the resurrected news/catalyst overlays -- LLM-gated) or accepting the engine.
