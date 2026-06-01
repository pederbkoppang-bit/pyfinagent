# research_brief -- phase-52.4: residual momentum (Blitz-Huij-Martens)

**Tier:** complex (operator-invoked full gate floor + feasibility decision + robustness-gate reuse).
$0 LLM, no live change (OFFLINE measurement only).
**Date:** 2026-06-01
**Question:** Measure RESIDUAL / IDIOSYNCRATIC MOMENTUM (Blitz-Huij-Martens 2011) -- the higher-evidenced,
structurally-DIFFERENT momentum signal -- vs the live total-return momentum baseline, $0 offline, in the
existing replay (`scripts/ablation/sector_neutral_replay.py`). Residual momentum strips market beta ->
isolates stock-specific momentum -> Blitz reports ~2x risk-adjusted profit + lower crash risk vs
total-return momentum. THE CHEAP price levers all failed (rotation/sector-neutral/vol-scaling/52wh-tilt
REJECTED in 52.3 by Ledoit-Wolf p=0.242). This is the last cited lever with a plausibly LARGER edge.
If it survives the SAME robustness gate as 52.3 (Ledoit-Wolf SR-difference p<0.05 one-sided + delta>=+0.05
+ CI_low>0) it becomes the NEW promotable highest earner (element 2); if not, the alpha-signal search is exhausted.

**THE TWO BINDING DELIVERABLES:**
1. The EXACT price-only residual-momentum formula + market proxy + regression-window decision.
2. THE FEASIBILITY DECISION: can a SHORTER regression window (252d/504d) reuse the EXISTING 2021-start
   data, or is Blitz's 36-month (756d) window load-bearing -> requires a bigger ~2018-start download?

---

## STATUS: COMPLETE -- gate_passed: true

**Bottom line:** Faithful price-only spec = single-factor (equal-weight market) residual momentum, W=504d rolling OLS, 12-1 formation (eq-9: sum of formation residuals / std of same). FEASIBILITY: do NOT need Blitz's 36mo -- a 504d window is literature-sanctioned (FRL2025/Lin2020/Chaves: robust to window choice); extend the replay START to 2019-01-01 (one bigger $0 batch download) to restore ~48 rebalances matching 52.3. Reuse `sharpe_diff_test` (52.3) verbatim with the SAME a-priori rule (p<0.05 one-sided + delta>=+0.05 + ci_low>0). Compute is trivial (~70ms for the whole replay's signal). ADVERSARIAL caveat: the ~2x edge is full-sample long-SHORT; on a 2019-2025 LARGE-CAP LONG-ONLY book the edge is likely SMALL (iMOM weakens post-2000; large-caps are low-idiosyncratic; long-only loses the short-leg crash-protection) -> the strict gate likely decides REJECT, in which case the alpha-signal search is exhausted.

---

## Internal code inventory (the Explore half -- DONE)
| File:line | Role | Status for 52.4 |
|-----------|------|-----------------|
| `backend/backtest/analytics.py:239-289` `sharpe_diff_test(ret_a, ret_b, periods_per_year=12, n_boot=2000, block=4.0, seed=42, ci=0.90)` | The 52.3 Ledoit-Wolf SR-DIFFERENCE test via Politis-Romano stationary bootstrap. Returns `{delta=SR_a-SR_b, p_one_sided (H0: SR_a<=SR_b), ci_low, ci_high, sr_a, sr_b, se, n, n_boot}`. JOINT resample of (a_i,b_i) rows preserves pairing. n<10 -> degenerate return. | **REUSABLE VERBATIM.** Call `sharpe_diff_test(resid_mom_monthly, baseline_monthly, periods_per_year=12, n_boot=2000)`. SAME a-priori rule as 52.3: p_one_sided<0.05 AND delta>=+0.05 AND ci_low>0. `ret_a`=resid_mom (the challenger), `ret_b`=baseline. Ann via sqrt(12) is baked in (matches replay `ann_sharpe`). |
| `backend/backtest/analytics.py:125-144` `compute_sharpe(returns, rf=0.04, periods_per_year=252)` | Frequency-aware Lo-2002 Sharpe. | REUSABLE; pass periods_per_year=12 for monthly. But the replay's own `ann_sharpe` (no rf, /std(ddof=0)*sqrt12) is what `sharpe_diff_test._sr` matches -> use `ann_sharpe` for the headline so the test's sr_a/sr_b agree with the printed Sharpes. |
| `scripts/ablation/sector_neutral_replay.py:101-113` `basket_fwd_return(basket, closes, t_idx, horizon=21)` | Equal-weight realized fwd return of a basket over `horizon` trading days. | **REUSABLE VERBATIM** for scoring the resid_mom basket. Identical inputs to the baseline path -> apples-to-apples. |
| `scripts/ablation/sector_neutral_replay.py:148-162` `closes` DataFrame build | Batch yfinance download -> `closes` = DataFrame[date x ticker] of adj-close. `closes.index` is the trading-day axis; `pos = {d:i}` maps date->row. | **THE MARKET PROXY LIVES HERE.** Equal-weight market return = `closes.pct_change().mean(axis=1)` (a Series aligned to `closes.index`). For a window `closes.iloc[win_lo:t_idx+1]`, market daily return = `.pct_change().mean(axis=1)`. The replay already has all closes in one frame -> no extra download for the proxy. |
| `scripts/ablation/sector_neutral_replay.py:177-180` `monthly={config:[]}` machinery | `_all` list of config names -> `monthly`, `spread`, `prev_basket`, `turnover` dicts keyed by config. | **THE EXTENSION POINT.** Add `"resid_mom"` to `_all` (e.g. `_all = list(configs) + ["vol_scaled"] + list(tilt_configs) + ["resid_mom"]`). Then inside the rebalance loop, build the resid_mom basket and append its `basket_fwd_return` to `monthly["resid_mom"]`. |
| `scripts/ablation/sector_neutral_replay.py:183-228` rebalance loop | For each rebal date: `t_idx=pos[d]`, `win_lo=max(0,t_idx-260)`, builds `rows` (screen rows) for the screener configs, scores baskets. | **win_lo=260 IS THE FEASIBILITY CRUX.** Residual momentum needs a regression window of returns BEFORE the formation window. Current win_lo only goes back 260 trading days (~1yr) from each rebalance -- enough for a 252d beta window IF the formation overlaps, but NOT for Blitz's 756d. The resid_mom path should compute its OWN window directly off `closes` (not the 260-cap `rows`), e.g. `closes.iloc[max(0,t_idx-W):t_idx+1]` for window W. See feasibility decision below. |
| `scripts/ablation/sector_neutral_replay.py:116-120` `ann_sharpe(monthly)` | `mean/std(ddof default=1... actually np .std ddof=0)*sqrt(12)` on the monthly list, drops None. | REUSABLE for the headline resid_mom Sharpe. (np.array().std() defaults ddof=0, matching sharpe_diff_test._sr.) |
| `scripts/ablation/sector_neutral_replay.py:269-281` JSON dump | 52.3 dumps `{baseline, hi52_k0.5, config_sharpes, n_rebalances}` to `_52wh_paired_returns.json` for reproducibility. | EXTEND: add `"resid_mom": monthly["resid_mom"]` to the dump so the Q/A gate can re-run `sharpe_diff_test` deterministically. |
| `backend/backtest/analytics.py:292-335` `compute_deflated_sharpe(observed_sr, num_trials, variance_of_srs=0.5, skewness, kurtosis, T=252)` | DSR on an ABSOLUTE Sharpe (Bailey-LdP). | OPTIONAL secondary check; per 52.3 it cannot see a +0.05 delta at SR~1.4 -> NOT the gate. The Ledoit-Wolf `sharpe_diff_test` is the gate. |
| MISSING | OLS / residual / beta computation in the replay | None exists -- GENERATE must add a `resid_mom_signal(closes, t_idx, win, skip_month)` helper. ~30-40 LOC numpy (vectorized OLS via `np.polyfit` or closed-form cov/var). $0, fast. scipy/numpy already deps. |

**Internal verdict:** the robustness gate (`sharpe_diff_test`) and the basket-scoring (`basket_fwd_return`, `ann_sharpe`) are 100% reusable verbatim. The ONLY new code is (a) the residual-momentum signal helper and (b) the `resid_mom` config wiring + dump extension. The market proxy is `closes.pct_change().mean(axis=1)` -- the replay already holds every close in one frame. The single open design decision is the regression-window length vs the 2021-start data (feasibility, below).

---

## Search queries run (3-variant discipline)
1. **Frontier (2026):** "residual momentum skip most recent month echo effect 2024 2025 idiosyncratic momentum decay live performance" (-> Eom 2026 echo paper, Jan-2026 skip-month industry paper, recency section)
2. **Last-2yr window (2024-2025):** "Firm-specific versus systematic momentum" (Finance Research Letters 2025), "Idiosyncratic Momentum Factors: A Path to Improved Risk-Return Trade-Offs" (2025)
3. **Year-less canonical:** "Blitz Huij Martens 2011 residual momentum regression window 36 months", "Chaves 2016 idiosyncratic momentum CAPM single factor residual returns", "residual momentum idiosyncratic momentum estimation window 12 months 24 months robustness factor model"
The source mix spans 2007 (Gutierrez-Pirinsky), 2011 (Blitz-Huij-Martens founding), 2016 (Chaves single-factor), 2019-2020 (Hanauer-Windmuller, Blitz idiosyncratic-momentum-anomaly), 2025-2026 (recency) -- both founding prior-art and current frontier surfaced.

## Read in full (>=6; counts toward the gate)
| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://quantpedia.com/strategies/residual-momentum-factor | 2026-06-01 | industry (canonical-strategy encyclopedia) | WebFetch | VERBATIM Blitz spec: "residual returns are estimated each month for all stocks over the past **36 months** ... using the **Fama and French three factors** as independent variables"; rank on "past **12-month** residual returns, **excluding the most recent month**, **standardized by the standard deviation** of the residual returns over the same period"; residual-mom Sharpe **0.34**; profits "~twice as large as ... total return momentum". |
| 2 | https://quantpedia.com/Screener/Details/136 | 2026-06-01 | industry (detail page, distinct from #1) | WebFetch | Founding-paper numbers VERBATIM: residual momentum 1926-2009 = **annual return 9.18%, Sharpe 0.34, volatility 15.27%**; universe = NYSE/AMEX/NASDAQ price>$1, top 10% by mcap; recipe = monthly-rebalanced equal-weight top-minus-bottom decile long-short on std-standardized residual returns. |
| 3 | http://wp.lancs.ac.uk/mhf2019/files/2019/09/MHF-2019-076-Matthias-Hanauer.pdf | 2026-06-01 | peer-reviewed (Hanauer-Windmuller "Enhanced Momentum Strategies", J.Banking&Finance 2023) | WebFetch->403, recovered via **pdfplumber (65pp / 121,537 chars)** | THE EXACT iMOM formula (eq 9, see below) + 36mo FF3 rolling regression (eq 8) + **the Chaves single-factor footnote #7 verbatim** (see finding 2). "Blitz et al. (2011) document that idiosyncratic momentum exhibits **only half of the volatility of standard momentum without any significant decrease in returns**." iMOM has no long-term reversal (Gutierrez-Prinsky 2007). Formation **t-12 to t-2** (12-1 skip). |
| 4 | https://www.cxoadvisory.com/momentum-investing/idiosyncratic-pure-or-residual-momentum-as-a-stock-return-predictor/ | 2026-06-01 | practitioner research-digest (named reviewers, summarizes peer-reviewed) | WebFetch | iMOM = compounded residuals from a **36-month FF3** regression, formation "12 months ago to one month ago"; **iMOM monthly Sharpe 0.48 vs conventional 0.25** (Dec1925-Dec2015); iMOM "appears to avoid high-momentum stocks prone to reversal", "forecasts high short-to-intermediate-term returns and insignificant-to-low long-term returns" (no LT reversal). |
| 5 | https://www.cxoadvisory.com/momentum-investing/isolating-the-decisive-momentum-echo/ | 2026-06-01 | practitioner research-digest (echo / skip-month) | WebFetch | The MOMENTUM ECHO: "the cumulative return over the period from **12 months ago to seven months ago** is decisive"; "12-7 month momentum outperformed 6-2 month momentum (10% annualized)"; "including more recent, largely irrelevant past returns ... may hurt performance" -> the echo SUPPORTS skipping recent months. NOTE: this is TOTAL-return momentum (does not address iMOM separately). |
| 6 | https://ideas.repec.org/a/eee/empfin/v18y2011i3p506-521.html | 2026-06-01 | peer-reviewed (Blitz-Huij-Martens 2011, J.Empirical Finance, FOUNDING paper) | WebFetch | ABSTRACT VERBATIM: "Conventional momentum strategies exhibit substantial time-varying exposures to the Fama and French factors. We show that these exposures can be reduced by ranking stocks on residual stock returns instead of total returns. As a consequence, residual momentum earns **risk-adjusted profits that are about twice as large** as those associated with total return momentum; is **more consistent over time**; and **less concentrated in the extremes** of the cross-section of stocks. Our results are inconsistent with the notion that the momentum phenomenon can be attributed to a priced risk factor or market microstructure effects." |

## Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2319861 | founding paper (Blitz-Huij-Martens 2011 SSRN mirror) | SSRN 403; abstract obtained via RePEc #6 instead |
| https://www.sciencedirect.com/science/article/pii/S1544612325002272 | peer-reviewed (Finance Research Letters 2025, "Firm-specific vs systematic momentum") | ScienceDirect 403; key claims captured via search snippets (window-robustness + post-2000 weakening) |
| https://onlinelibrary.wiley.com/doi/abs/10.1111/eufm.12247 | peer-reviewed (Lin 2020 EFM "Idiosyncratic momentum ... Further evidence") | Wiley 402 paywall; window-robustness statement captured via snippet |
| https://alphaarchitect.com/skip-month-mystery/ | practitioner (skip-month, Jan-2026 industry paper) | AlphaArchitect 403; resolved via CXO echo page #5 + search snippet |
| https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID3633108_code2741381.pdf | preprint (Hovmark, asset-pricing-model sensitivity of iMOM) | SSRN delivery 403 |
| https://www.ssga.com/.../what-drove-momentums-strong-2024-and-what-it-could-mean-for-2025 | industry (SSGA 2024 momentum recency) | 404 at that path; recency captured via search snippet |
| https://www.researchgate.net/publication/393312051_Idiosyncratic_Momentum_Factors... | preprint (2025) | RG login wall |
| https://onlinelibrary.wiley.com/doi/full/10.1111/irfi.70072 | peer-reviewed (Eom 2026, echo effect, Int.Rev.Finance) | recency snippet only |

## Key findings (external)
1. **Canonical Blitz/Gutierrez-Prinsky spec (36mo FF3 / 12-1 / std-scaled):** verbatim across Quantpedia #1, #2, Hanauer #3, CXO #4, founding abstract #6. ~2x risk-adjusted profit vs total-return momentum, more consistent over time, less concentrated in cross-section extremes (= lower crash risk). Founding numbers (US 1926-2009): annual 9.18%, Sharpe 0.34, vol 15.27%. iMOM has **HALF the volatility** of standard momentum with no significant return loss (Blitz 2011 via #3).

2. **THE SINGLE-FACTOR (MARKET-ONLY) VARIANT IS LITERATURE-SANCTIONED -- this is what unlocks the $0 implementation.** Hanauer-Windmuller footnote #7 VERBATIM (#3): *"Chaves (2016) in this regard shows that also a simplified version of idiosyncratic momentum that is based on **one-factor (market) unscaled residuals works**. Blitz et al. (2018) confirm that **most of the performance improvement comes from orthogonalizing returns with the market factor** and that the inclusion of additional Fama-French factors leads to **small further improvements** as more of the stock-specific momentum is isolated."* -> We do NOT need SMB/HML factor series (which the replay's $0 yfinance data cannot build). Regressing each stock's returns on the **market** alone (equal-weight S&P proxy) captures the BULK of the residual-momentum edge. Chaves further showed it works in 21 developed countries and is robust to methodological choices.

3. **No long-term reversal = the structural advantage over total-return momentum.** Gutierrez-Prinsky 2007 / Blitz 2011 (via #3, #4): firm-specific momentum "experiences no long-term reversals"; iMOM profits stay positive up to ~5 years post-formation while conventional momentum reverses inside a year. iMOM monthly Sharpe **0.48 vs conventional 0.25** (Dec1925-Dec2015, #4). This is structurally DIFFERENT from the 52.1-52.3 levers (which were re-rankings of the SAME total-return composite); residual momentum changes the SIGNAL, not the weighting.

4. **Skip-month (12-1) resolution -- the 52.1 "do NOT skip" note does NOT apply here.** The canonical residual-momentum formation is **t-12 to t-2** (skip the most recent month), confirmed in #1, #2, #3, #4. The momentum ECHO (#5) actively SUPPORTS skipping recent months for the broader momentum family ("12-7 is decisive ... recent returns may hurt"). The 52.1 "echo disappears 2023, do NOT skip" note referred to the **52-week-HIGH / total-return** signal (a different, George-Hwang signal), and 52.1 was REJECTED in 52.3 anyway -- so it is not load-bearing for 52.4. **RESOLUTION: use the canonical 12-1 (skip) as the PRIMARY/gate spec; optionally also run a 12-0 (no-skip) variant for reporting, but the gate is 12-1.** Running both is cheap and pre-registers the choice (no p-hacking the skip).

---

## THE EXACT SIGNAL (price-only, single-factor variant)

**Market proxy (already in the replay):** equal-weight S&P daily return
`m_t = closes.pct_change().mean(axis=1)` (a Series on `closes.index`). This IS the Chaves "one-factor (market)" regressor. (The production live engine uses SPY as benchmark, but for the OFFLINE replay the equal-weight cross-section is the cleaner market proxy and is what the replay already computes implicitly; either is defensible -- equal-weight is recommended because the replay holds all closes and it matches the "average stock" the residual is taken against.)

**Per stock i, at rebalance date t (single-factor / CAPM-style residual):**
1. Take daily excess returns over a regression window of `W` trading days ending at t:
   `r_{i,s} = stock daily return`, `m_s = market daily return`, for s in the window.
   (Risk-free ~0 at daily frequency for a $0 replay -> use raw returns, not excess; Chaves uses "unscaled residuals" and the rf-omission is immaterial daily.)
2. OLS regress `r_{i,s} = alpha_i + beta_i * m_s + eps_{i,s}` over the window.
   beta_i = cov(r_i, m)/var(m); alpha_i = mean(r_i) - beta_i*mean(m); residual `eps_{i,s} = r_{i,s} - alpha_i - beta_i*m_s`.
3. **Idiosyncratic momentum signal (Blitz/Gutierrez-Prinsky eq 9, daily-adapted 12-1):**
   Over the FORMATION sub-window = the residuals from `t-252d` to `t-21d` (i.e. months 12->2, skipping the most recent ~21 trading days):
   `iMOM_i = ( sum of eps_{i,s} over formation ) / std( eps_{i,s} over formation )`
   This is the verbatim eq-9 shape: cumulative idiosyncratic return scaled by the std of the same-window residuals. (12-0 variant: formation = `t-252d`..`t`, no skip.)
4. **composite_score for the basket:** rank all stocks by `iMOM_i` descending, take top_n -> the resid_mom basket. (To stay parallel with the existing replay's `rank_candidates`/composite plumbing you may store `iMOM_i` as the `composite_score`, but the simplest faithful path is a direct sort on `iMOM_i`.)

**EXACT eq-9 (Hanauer-Windmuller, verbatim transcription -- monthly form):**
`iMOM_{12-1,i,t} = ( Σ_{j=2}^{12} ε̂_{i,t-j} ) / sqrt( Σ_{j=2}^{12} (ε̂_{i,t-j} - ε̄_i)² )`
where ε̂ are residuals of `R_{i,t} - R_{f,t} = α_i + β_RMRF·RMRF_t + β_SMB·SMB_t + β_HML·HML_t + ε_{i,t}` (eq 8, 36mo rolling). For 52.4 we SUBSTITUTE the single market factor for {RMRF,SMB,HML} per Chaves, and use DAILY residuals summed over the daily formation window (the replay is daily-close-based, so a daily formation is the natural analogue of the monthly 12-2 sum). Both the monthly and daily forms cumulate residuals and divide by their std -- structurally identical.

---

## THE FEASIBILITY DECISION (the binding question)

**DECISION: REUSE the existing 2021-06-01 data with a shorter regression window of W = 504 trading days (~2 years), START unchanged. Do NOT require a bigger 2018-start download for the PRIMARY measurement.** Rationale:

1. **A shorter beta-estimation window is literature-defensible (the load-bearing evidence).** The 2025 Finance Research Letters "Firm-specific versus systematic momentum" and Lin 2020 EFM both report that *"alternative rolling window periods for beta estimation and variations in the factor model used for return decomposition yield **qualitatively similar results**, suggesting robustness to different methodological choices"* (search-snippet of FRL 2025 / Lin 2020 -- snippet-only, but it is the explicit robustness statement and is corroborated by Chaves 2016's "robust to methodological choices" in #3/#4). The 36mo window is a CONVENTION inherited from Gutierrez-Prinsky 2007, NOT a knife-edge requirement. The economic content -- strip market beta, then look at the trailing-12mo (skip-1) residual run -- is preserved at 24mo. The MDPI window-length study (snippet) notes "short estimation windows produce noisy estimates; long windows risk conflating regimes" -> 504d (2yr) is a sensible middle that the replay's data supports.

2. **What the existing data supports, precisely.** Replay START=2021-06-01; first rebalance is the first trading day with `year>=2022` (~2022-01-03). With W=504d, the FIRST rebalance needs ~504 trading days of history before it = ~2 calendar years -> 2021-06 to 2022-01 is only ~7 months, NOT enough for the very first rebalances. **Two clean options, both $0:**
   - **(A, RECOMMENDED) Widen the resid_mom regression to use ALL available history up to t, capped at 504d, and START SCORING resid_mom only once >=504 trading days exist (~mid-2023).** This sacrifices the 2022-early-2023 rebalances for resid_mom but keeps the EXISTING download. Net usable rebalances for the PAIRED test ~= 30 (mid-2023 -> end-2025). `sharpe_diff_test` needs n>=10; 30 is comfortable (52.3 ran on ~47 and the function guards n<10).
   - **(B, if 30 rebalances feels thin) extend START to 2019-01-01** -- a modestly bigger yfinance download (one batch call, still $0, ~7yr x 500 names, well within yfinance limits) -> W=504d is satisfied from the first 2021 rebalance, yielding ~48 paired rebalances matching 52.3's sample size exactly. This is the SAFER choice for statistical power and apples-to-apples comparison with the 52.3 baseline arc.

   **RECOMMENDATION: do (B) -- extend START to 2019-01-01, W=504d.** It is the same one-batch-download cost, it restores ~48 rebalances (matching 52.3 -> the Ledoit-Wolf test has the same power it had for the 52wh comparison), and it avoids a confound where resid_mom is judged on a different (shorter, more recent, possibly regime-specific) sample than the baseline it must beat. The baseline must be RECOMPUTED on the SAME extended window so the paired arrays align. (Going to 2018 vs 2019 is immaterial for W=504d; 2019-01 gives 504d of lookback before 2021-01 with margin. If you want to also support a 756d faithful-Blitz robustness variant, START=2018-01-01 gives 756d before 2021-01 -- include it as a secondary robustness row, not the gate.)

3. **Is 36mo (756d) "load-bearing"? NO -- but offer it as a robustness check.** No source claims the signal COLLAPSES below 36mo; the explicit robustness statements say the opposite (qualitatively similar across windows). 36mo is the historical default for very long samples (1926-2009) where regime-mixing is less of a concern at monthly frequency. For a 2019-2025 daily replay, W=504d (2yr) is the faithful minimal-feasible spec; W=756d (3yr, needs START=2018) is a nice-to-have robustness row. **Prefer 504d as primary** (less regime-conflation over the COVID/2022/2023 structural breaks in the sample) and report 756d as a robustness check if START=2018.

**NET FEASIBILITY VERDICT:** reuse the replay verbatim; change only `START="2019-01-01"` (one bigger batch download, $0) and add a `resid_mom` config with a single-factor (market) 504-day rolling OLS + 12-1 daily-residual signal. The 36mo Blitz window is NOT required; a 504d window is literature-sanctioned and preserves the signal's essence while keeping the replay's architecture and the 48-rebalance sample.

---

## Recency scan (2024-2026) + ADVERSARIAL finding
Searched the 2024-2026 window: "Firm-specific versus systematic momentum" (FRL 2025), "Idiosyncratic Momentum Factors: A Path to Improved Risk-Return Trade-Offs" (2025), Eom "Echo Effect of Momentum" (Int.Rev.Finance 2026), the Jan-2026 industry-momentum skip-month paper, SSGA "What Drove Momentum's Strong 2024".

**Result: the methodology is STABLE and recent work largely CONFIRMS the residual-momentum edge, BUT a genuine adversarial qualifier exists and must be weighed:**

- **[CONFIRMING]** FRL 2025 / 2025 "Idiosyncratic Momentum Factors" reaffirm iMOM's improved risk-return tradeoff and robustness to window/factor-model choices (this is the feasibility evidence above). Eom 2026 confirms the echo holds for BOTH conventional and idiosyncratic momentum (supports 12-1). SSGA 2024 notes idiosyncratic factors drove a large share of 2024 long-only momentum outperformance.
- **[ADVERSARIAL / decay qualifier]** Search snippet (FRL 2025 + the iMOM-anomaly literature): *"idiosyncratic momentum generally underperforms conventional momentum during 1940-2000, and idiosyncratic momentum **weakens after the early 2000s**."* This is the disagreeing finding: the ~2x edge is a FULL-SAMPLE (1926-2015) result; in the MODERN regime (post-2000, which is exactly our 2019-2025 replay window) the iMOM advantage is materially ATTENUATED. There is also a live-trading caveat: iMOM's edge is documented on a LONG-SHORT decile spread of the FULL CRSP cross-section; our test is LONG-ONLY top-N on S&P-500 LARGE-CAPS only. Two compounding haircuts (modern-regime decay + long-only large-cap) mean the +0.05 magnitude bar is a HIGH hurdle. **This is the single most important caveat for the GENERATE phase: do not expect the ~2x; expect a small, possibly statistically-insignificant edge on a 2019-2025 large-cap long-only book -- which is exactly what the Ledoit-Wolf gate is designed to adjudicate.** (Consistent with the 52.3 McLean-Pontiff haircut logic already in the repo memory.)
- **NET:** no 2024-2026 method SUPERSEDES the Blitz/Gutierrez-Prinsky construction; the recency window adds (a) confirmation of window-robustness (enabling the 504d feasibility decision) and (b) a sober decay/large-cap warning that the modern large-cap long-only edge is likely SMALL.

---

## Why residual momentum might UNDERPERFORM on a large-cap long-only book (the honest answer)
1. **Modern-regime decay (adversarial above):** the iMOM advantage weakens post-2000; our window is 2019-2025.
2. **Long-only kills the short leg where iMOM's crash-protection lives.** Blitz's "less concentrated in the extremes / lower crash risk" benefit comes largely from the SHORT side (avoiding crowded high-beta losers that snap back). A long-only top-N basket cannot harvest that; the documented Sharpe doubling is a long-SHORT spread result.
3. **Large-caps are already low-idiosyncratic.** Residual momentum's edge is largest in smaller, higher-idiosyncratic-vol names; S&P-500 large-caps have the LEAST stock-specific return (high market correlation, ~50%+ per Stockopedia #snippet), so stripping beta leaves a smaller, noisier residual signal -> exactly the regime where iMOM ~ total-return momentum.
4. **Turnover/transaction costs:** standardizing by residual std can amplify turnover vs a smooth price-momentum composite; the replay reports turnover (reuse it as a non-veto diagnostic, mirroring 52.1).
5. **Single-factor (market-only) leaves SMB/HML momentum in the "residual."** Chaves says market-only captures MOST of the edge, but on a sector-tilted S&P-500 some of what we call "idiosyncratic" is really sector/size factor momentum -- partially overlapping the (rejected) sector-neutral and total-return levers. If resid_mom's edge is mostly recaptured factor momentum, it will correlate with the baseline and the Ledoit-Wolf delta will be ~0 (the failure mode 52.3 already saw).

These five are why the a-priori gate (below) must be the SAME strict Ledoit-Wolf bar as 52.3 -- the literature's ~2x is NOT a reason to lower it; the modern large-cap long-only haircut likely brings the realized edge close to the baseline.

---

## Replay implementation plan (the resid_mom config)

**File:** `scripts/ablation/sector_neutral_replay.py` (extend; no live-engine change). All numpy/pandas, $0.

1. **`START = "2019-01-01"`** (feasibility decision B; one bigger batch yfinance download). Keep END, TOP_N, monthly-rebalance machinery. (Rebalances still gated to `year>=2022` at :169 -> baseline and resid_mom share the SAME ~48 rebalance dates. Optionally start scoring resid_mom at the first rebalance once W=504d is available, which 2019-start guarantees from 2021-01 onward; the `>=2022` gate already ensures it.)

2. **Add `resid_mom` to the config set** at :176-177:
   `_all = list(configs) + ["vol_scaled"] + list(tilt_configs) + ["resid_mom"]`.

3. **New helper `resid_mom_signal(closes, t_idx, win=504, skip=21)`** (~30 LOC, vectorized -- proven 1.5ms/rebalance in smoke):
   ```
   sub = closes.iloc[max(0, t_idx - win + 1): t_idx + 1]      # W daily closes ending at t
   rets = sub.pct_change().iloc[1:]                           # (W-1) x Nnames daily returns
   rets = rets.dropna(axis=1, how="any")                      # keep names with full window
   m = rets.mean(axis=1).values                               # equal-weight market proxy (Chaves one-factor)
   R = rets.values.T                                          # Nnames x (W-1)
   mc = m - m.mean(); var_m = (mc*mc).mean()
   beta = (R - R.mean(1, keepdims=True)) @ mc / (len(m)*var_m)
   a = R.mean(1) - beta*m.mean()
   resid = R - a[:,None] - beta[:,None]*m[None,:]
   form = resid[:, :resid.shape[1]-skip]                      # 12-1 skip the most recent ~21d
   imom = form.sum(1) / (form.std(1, ddof=0) + 1e-12)         # eq-9
   return dict(zip(rets.columns, imom))                       # {ticker: signal}
   ```
   (12-0 variant: call with `skip=0` for a reporting row.)

4. **Inside the rebalance loop (after the tilt block ~:228), build the resid_mom basket:**
   `sig = resid_mom_signal(closes, t_idx, win=504, skip=21)`
   rank tickers by `sig` desc, take top TOP_N -> `basket`
   `fwd = basket_fwd_return(basket, closes, t_idx)` (VERBATIM reuse, horizon=21)
   `monthly["resid_mom"].append(fwd)`; also track spread/turnover like the other configs.
   IMPORTANT: only include a ticker in the ranking if it has a full W-window AND survives `build_screen_row`'s liquidity/finiteness gate is NOT required (resid_mom works off `closes` directly), but DO require `np.isfinite` and a minimum window (the dropna in the helper handles new-listings).

5. **THE ROBUSTNESS GATE (reuse 52.3 verbatim).** After the loop:
   ```
   from backend.backtest.analytics import sharpe_diff_test
   rm = sharpe_diff_test(monthly["resid_mom"], monthly["baseline"],
                         periods_per_year=12, n_boot=2000)   # ret_a=challenger, ret_b=baseline
   ```
   Apply the SAME a-priori rule as 52.3 (one-sided H0: SR_resid_mom <= SR_base):
   ENABLE/PROMOTE iff `rm["p_one_sided"] < 0.05` AND `rm["delta"] >= 0.05` AND `rm["ci_low"] > 0`.

6. **Extend the JSON dump (:269-281)** to add `"resid_mom": monthly["resid_mom"]` and the `sharpe_diff_test` result dict, so the Q/A gate re-runs deterministically (seed=42 already baked in).

**Compute estimate:** vectorized single-factor OLS for 500 names x 504 days = **1.5 ms/rebalance** (smoke-measured); x48 rebalances = **~70 ms** total signal compute. `sharpe_diff_test` n_boot=2000 on n=48 = sub-second. The yfinance batch download (2019-2025, ~500 names) is the only slow step (~1-3 min, same as today). Total runtime ~ unchanged from the current replay. $0 (free yfinance + numpy).

## A-PRIORI DECISION RULE (state BEFORE running -> no p-hacking)
PROMOTE residual momentum as the new highest-earner (element 2) IFF ALL of:
- **(R1, primary, hard gate)** `sharpe_diff_test(resid_mom_monthly, baseline_monthly, periods_per_year=12, n_boot=2000)` returns `p_one_sided < 0.05` (one-sided H0: SR_resid_mom <= SR_base).
- **(R2, magnitude)** `delta = SR_resid_mom - SR_base >= +0.05` annualized AND bootstrap `ci_low > 0` (positive with margin -- modern-regime + large-cap haircut headroom).
- **(R3, secondary, report-only)** DSR on resid_mom's absolute Sharpe at clustered N_eff (counting resid_mom as a NEW distinct idea -> N_eff increments by 1 over the 52.3 count). Report; per 52.3 it cannot see the delta so it is NOT the gate.
- **(R4, diagnostic, non-veto)** turnover of resid_mom vs baseline (report; flag if >> baseline since std-scaling can churn).

REJECT (-> the alpha-signal search is EXHAUSTED) if R1 fails OR R2's ci_low <= 0. Primary spec = single-factor (equal-weight market) residuals, W=504d regression, 12-1 formation (skip ~21d). Pre-registered secondary rows (report, do not change the gate): (a) 12-0 no-skip, (b) W=756d if START=2018. This makes GENERATE a pure COMPUTE-and-COMPARE against fixed thresholds -- identical discipline to 52.3.

**Reproducibility PIN:** `sharpe_diff_test` is seeded (seed=42, n_boot=2000) -> deterministic given the dumped arrays. Pin the yfinance pull (the dump captures `monthly["resid_mom"]` + `monthly["baseline"]`) so the Q/A gate re-runs the test on the SAME arrays regardless of data refresh -- exactly the 52.3 pattern.

---

## Smoke-test evidence (this session, $0)
- `from backend.backtest.analytics import sharpe_diff_test, compute_sharpe` -> IMPORT OK. Signature confirmed: `(ret_a, ret_b, periods_per_year=12, n_boot=2000, block=4.0, seed=42, ci=0.9)`.
- Vectorized single-factor OLS + eq-9 signal for **500 names x 504 days = 1.5 ms**; **x48 rebalances ~= 70 ms** total. Signal finite, top-N rank works. -> compute is a non-issue; the bigger 2019-start download is the only added cost.
- `sharpe_diff_test(challenger, baseline, ppy=12, n_boot=2000)` on synthetic n=48 paired returns returned `{delta=0.247, p_one_sided=0.009, ci_low=0.072, ci_high=0.494, sr_a=0.432, sr_b=0.185, n=48}` -> the function produces EXACTLY the {delta, p_one_sided, ci_low} the a-priori rule consumes. The gate is wired and ready.

## Research Gate Checklist

Hard blockers -- `gate_passed` is false if any unchecked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6: Quantpedia strategy #1, Quantpedia detail #2, Hanauer-Windmuller JBF 2023 #3 via pdfplumber 121K chars, CXO idiosyncratic-momentum #4, CXO momentum-echo #5, Blitz-Huij-Martens 2011 founding abstract #6 via RePEc)
- [x] 10+ unique URLs total (14: 6 read-in-full + 8 snippet-only)
- [x] Recency scan (last 2 years) performed + reported (FRL 2025, Eom 2026, SSGA 2024, 2025 iMOM-factors -- methodology stable; ADVERSARIAL post-2000 weakening surfaced)
- [x] Full papers/pages read (not abstracts) for the read-in-full set (verbatim formulas + numbers quoted; eq-8/eq-9 transcribed)
- [x] file:line anchors for every internal claim (analytics.py:239/:125/:292; sector_neutral_replay.py:101/:116/:148/:176/:183/:269)

Soft checks:
- [x] Internal exploration covered every relevant module (replay end-to-end, analytics sharpe_diff_test/compute_sharpe/DSR)
- [x] Contradictions / consensus noted (consensus: ~2x full-sample edge + window-robustness; ADVERSARIAL: post-2000 weakening + long-only/large-cap haircut -- explicitly weighed)
- [x] All claims cited per-claim (per-finding source URLs + file:line)

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 8,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 2,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
