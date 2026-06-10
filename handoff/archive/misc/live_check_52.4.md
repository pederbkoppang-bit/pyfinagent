# live_check -- phase-52.4: residual momentum (Blitz-Huij-Martens) -> REJECT (search exhausted)

**Step:** 52.4 | **Date:** 2026-06-01 | **Result shape:** measured residual/idiosyncratic momentum
(the higher-evidenced, structurally-different lever) vs the baseline momentum ranking over ~58
monthly rebalances (2021-2025, 2019-start data for the 504d beta window), then the SAME Ledoit-Wolf
SR-difference gate as 52.3. **Outcome: REJECT -- residual momentum is WORSE (delta -0.249, p=0.77).
The cited-alpha-signal search is EXHAUSTED; the +20% momentum engine stands.** NO live change.

**Command:**
```
source .venv/bin/activate && PYTHONPATH=. python scripts/ablation/residual_momentum_replay.py
```
Single-factor (equal-weight market) residual momentum: 504d rolling OLS per name -> residuals ->
iMOM = sum(12-1 formation residuals)/std. $0: free yfinance + numpy OLS. Reuses the 52.3
`sharpe_diff_test` (Ledoit-Wolf) + the SAME a-priori rule. NO live engine change.

## Verbatim result
```
config          ann_Sharpe   avg_fwd_mo%  avg_turnover
baseline             1.332         3.651         0.564
resid_mom            1.082         1.970         0.679

--- ROBUSTNESS GATE (52.3 Ledoit-Wolf SR-difference; a-priori rule) ---
SR_resid_mom=1.082  SR_base=1.332  delta=-0.249  (n=58, n_boot=5000)
one-sided p (H0 SR_resid<=SR_base) = 0.7724  -> R1 (p<0.05): False
90% CI for delta = [-0.883, +0.330]  (se=0.364)  -> R2 (delta>=+0.05 AND CI_low>0): False
VERDICT: REJECT
```

## Interpretation (criterion #1/#2, honestly reported)
- **Residual momentum is WORSE, not better:** Sharpe 1.082 vs baseline 1.332 (**delta -0.249**), lower forward return (1.97% vs 3.65%/mo), HIGHER turnover (0.679 vs 0.564). The one-sided p=0.7724 (>0.05) and the 90% CI [-0.883,+0.330] -> R1 + R2 both FAIL -> REJECT.
- This DECISIVELY confirms the researcher's adversarial prior: the documented ~2x iMOM edge is a FULL-SAMPLE LONG-SHORT result; on a 2019-2025 LARGE-CAP LONG-ONLY S&P-500 book the three haircuts (modern-regime decay, no short leg, low idiosyncratic content in large-caps) don't just attenuate it -- they make residual momentum UNDERPERFORM total-return momentum. The single-factor residual also partly recaptures sector/size factor momentum (already-rejected levers) + adds turnover.

## Criterion-by-criterion

| # | Criterion | Evidence | Verdict |
|---|-----------|----------|---------|
| 1 | residual momentum (Blitz-Huij-Martens, price-only OLS) measured ON-vs-OFF vs baseline, reporting Sharpe/return/turnover | the table above (58 rebalances, 2019-start, W=504d, 12-1) | PASS |
| 2 | the improvement subjected to the SAME Ledoit-Wolf SR-difference gate (a-priori rule); honest 'not robust' REJECT is valid | gate applied (p=0.77, CI straddles 0); REJECT reported | PASS |
| 3 | NO live engine change; US momentum core untouched | diff = new replay script + resid_mom_signal + new test + pinned JSON; screener.py/autonomous_loop/the flag UNTOUCHED | PASS |
| 4 | live_check records ON-vs-OFF + SR-difference stats + cited basis + promote/reject rec | this file (REJECT; do NOT promote) | PASS |

## pytest
```
$ python -m pytest backend/tests/test_phase_52_4_residual_momentum.py -q
.....                                                                    [100%]
5 passed in 1.43s
```
(resid_mom_signal: positive formation residual -> iMOM>0; negative -> <0; recent-only run does NOT create positive momentum (12-1 skip + OLS-alpha absorption); too-short -> None; deterministic.)

## THE ELEMENT-2 ALPHA-SIGNAL SEARCH IS NOW EXHAUSTED (the rigorous conclusion)
Every cited price-based lever tested on our universe, with overfitting controls:
| Lever | Result |
|-------|--------|
| winner-take-all rotation | architecturally disconnected from live money + alt strategies LOSE money -> REJECT |
| sector-neutral breadth | -0.166 Sharpe (Harvey long-only caveat) -> REJECT |
| vol-scaling | +0.015, marginal |
| 52-week-high tilt | +0.057 point but Ledoit-Wolf p=0.242 (not robust) -> REJECT (52.3) |
| **residual momentum** | **-0.249 (WORSE), p=0.77 -> REJECT (52.4)** |
**No cited price-based signal robustly beats the live momentum engine on a 2019-2025 large-cap
long-only S&P-500 book.** The +20% / +14%-alpha momentum engine STANDS as the highest earner. This is
the honest, overfitting-controlled, research-complete outcome -- the discipline prevented shipping
any of these as alpha. The remaining money lever is the LIVE multi-market expansion (now running).

## Recommendation / next
- **Do NOT promote residual momentum** (it's worse). Keep the momentum engine as the highest earner.
- **The cited cheap-lever alpha search is closed.** Further alpha would need a DIFFERENT data axis (e.g. the now-resurrected news/catalyst overlays -- LLM-gated; or non-price signals) or accepting the engine + riding the multi-market expansion.
- **MEASURE Monday's first multi-market cycle** -- the real, live money test (the actual shipped lever).
- Reproducibility: pinned `_residmom_paired_returns.json` + seeded bootstrap -> deterministic verdict.
