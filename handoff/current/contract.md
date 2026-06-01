# Contract -- phase-52.4: residual momentum (Blitz-Huij-Martens) -- MEASURE + robustness-gate

**Step id:** 52.4 | **Priority:** P1 | **depends_on:** 52.3 | **Date:** 2026-06-01
**harness_required:** true | **$0 LLM** | no pip | **NO live change** (offline measurement) | a-priori rule -> promote/REJECT

## Research-gate summary (PASSED)
`handoff/current/research_brief.md` (researcher `afaa06ced01cfac95`: gate_passed=true, tier complex, 6 sources read IN FULL incl. Blitz-Huij-Martens 2011 founding paper + Hanauer-Windmuller eq-9 via pdfplumber). Decisive:
- **Signal (canonical, price-only, single-factor):** market proxy `m = closes.pct_change().mean(axis=1)` (equal-weight). Per stock at rebalance t: OLS `r_i = alpha + beta*m + eps` over a W=504d window (beta=cov/var); residual `eps = r - alpha - beta*m`; **iMOM = sum(eps over formation) / std(eps over formation)**, formation = the 12-1 window (residuals from t-252d to t-21d, SKIP the most recent ~21d). Rank by iMOM desc -> top_n. (The 52.1 "do NOT skip" note was for the 52wh/total-return signal; canonical residual momentum DOES skip-1 -- confirmed across 4 sources.)
- **Feasibility:** the 36mo Blitz window is NOT load-bearing (504d is literature-sanctioned, window-robust per FRL2025/Lin2020/Chaves). Extend the replay START to **2019-01-01** (one bigger $0 batch download) so W=504d is satisfied from the first 2021 rebalance -> ~48 paired rebalances matching 52.3's power; recompute the baseline on the SAME window. Reuse `sharpe_diff_test` (analytics.py, 52.3) + the SAME a-priori rule. OLS is ~30-40 LOC numpy; compute trivial.
- **HONEST PRIOR -- likely REJECT:** the ~2x iMOM edge is a FULL-SAMPLE LONG-SHORT result; modern-regime decay (post-2000 attenuation) + long-only (loses the short-leg crash-protection) + large-cap (low idiosyncratic content) all haircut it -> the modern large-cap long-only edge is likely SMALL, possibly insignificant. The strict Ledoit-Wolf gate is designed to adjudicate this; an honest REJECT exhausts the cited-alpha-signal search.

## Hypothesis
Residual/idiosyncratic momentum (strip market beta via 504d OLS, rank by the 12-1 std-normalized residual sum) is a structurally-DIFFERENT, higher-evidenced momentum signal -- but on a 2019-2025 LARGE-CAP LONG-ONLY S&P-500 book its improvement over the baseline momentum ranking is MOST LIKELY within noise (Ledoit-Wolf SR-difference p >= 0.05 and/or CI lower bound <= 0) -> the a-priori rule REJECTS -> no live promotion, and the cited-alpha-signal search is exhausted (the +20% engine stands). If it surprisingly SURVIVES the gate, it is the NEW promotable highest earner (element 2).

## Success criteria (IMMUTABLE -- verbatim from masterplan step 52.4)
1. residual/idiosyncratic momentum (cited: Blitz-Huij-Martens 2011, price-only -- regress stock returns on the market, rank by trailing residual momentum) is measured ON-vs-OFF against the baseline momentum ranking on the S&P 500 universe via the $0 replay, reporting Sharpe / return / turnover
2. the +improvement (if any) is subjected to the SAME Ledoit-Wolf SR-difference robustness gate as 52.3 (paired stationary-bootstrap, the a-priori rule R1 p<0.05 AND R2 delta>=+0.05 & CI_low>0); the result is honestly reported (a 'not robust' REJECT is a VALID outcome)
3. NO live engine change in this step (measure-first; any live wiring/enable is a separate operator-gated step); the working US momentum core is untouched
4. live_check_52.4.md records the ON-vs-OFF comparison + the SR-difference test stats + the cited basis + a keep/reject (promote/don't) recommendation

**Verification command:** `pytest backend/tests/test_phase_52_4_residual_momentum.py` + `test -f live_check_52.4.md`.
**live_check:** REQUIRED -- ON-vs-OFF (Sharpe/return/turnover) + Ledoit-Wolf SR-difference stats + a-priori-rule verdict + promote/reject rec; NO flag flip.

## Plan steps (GENERATE)
1. **scripts/ablation/residual_momentum_replay.py (NEW):** download S&P-500 daily closes 2019-01-01..2025-12-31 (one batch); reuse `build_screen_row`/`basket_fwd_return`/`ann_sharpe`/`load_universe_sectors` from sector_neutral_replay.py (import by path). NEW `resid_mom_signal(stock_ret_window, mkt_ret_window, form=252, skip=21)`: OLS beta=cov/var, residuals, iMOM = sum(form residuals)/std(form residuals). Per monthly rebalance (>=504d lookback): baseline = rank_candidates(rows, strategy="momentum") top-N; resid_mom = rank by iMOM top-N; score both via basket_fwd_return. Dump the paired monthly arrays -> handoff/current/_residmom_paired_returns.json (reproducibility pin).
2. **Robustness gate:** `sharpe_diff_test(resid_mom_monthly, baseline_monthly)` (the 52.3 Ledoit-Wolf, seeded) + the SAME a-priori rule (R1 p<0.05 one-sided + R2 delta>=+0.05 & CI_low>0) -> ENABLE(promote)/REJECT.
3. **test** `backend/tests/test_phase_52_4_residual_momentum.py`: `resid_mom_signal` correctness (a stock that is pure-market -> ~0 residual momentum; a stock with a strong positive idiosyncratic run over the formation -> high positive iMOM; a recent-only spike inside the skip window -> excluded; deterministic). $0, synthetic.
4. **Verify:** pytest; run the replay -> capture baseline vs resid_mom Sharpe/return/turnover + the LW SR-difference stats + the a-priori verdict into live_check_52.4.md (honest promote/REJECT). Report ALL stats.
5. **EVALUATE:** fresh qa. Then harness_log.md (LAST), then flip masterplan 52.4 -> done.

## Safety / scope notes
- **NO live change.** Diff = the new replay script + resid_mom_signal + the new test + the pinned JSON. screener.py / autonomous_loop / the momentum_52wh flag are UNTOUCHED. The live engine is unaffected; this is an offline $0 measurement (doesn't conflict with measuring Monday's live cycle).
- **Same strict a-priori gate as 52.3** -- the literature's ~2x is NOT a reason to lower the bar (modern large-cap long-only haircut likely brings the realized edge near baseline). Honest REJECT is the expected + valid outcome (criterion #2).
- Run BOTH 12-1 (skip, the gate) + optionally 12-0 (no-skip, reporting) -- pre-registered, no skip p-hacking.
- Reproducibility: pinned JSON + seeded bootstrap -> deterministic verdict for Q/A.
- $0 LLM; no pip; no flag flip.

## References
- handoff/current/research_brief.md (52.4 gate) + the researcher memory
- Blitz-Huij-Martens 2011 (J.Emp.Fin., founding); Hanauer-Windmuller 2023 (eq 8/9); Chaves 2016 (single-factor); FRL 2025 / Lin 2020 (window robustness); the modern-regime/large-cap/long-only decay caveats
- backend/backtest/analytics.py `sharpe_diff_test` (52.3) + `ann_sharpe`/`compute_sharpe`; scripts/ablation/sector_neutral_replay.py (build_screen_row/basket_fwd_return/ann_sharpe/load_universe_sectors to reuse; the market proxy closes.pct_change().mean(axis=1)); backend/tools/screener.py rank_candidates (baseline)
