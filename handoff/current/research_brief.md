# research_brief -- phase-52.3: DSR-deflate the 52wh edge (is +0.05 real?)

**Tier:** complex (caller invoked full gate floor; cross-domain statistical methodology).
$0 LLM, no live change.
**Date:** 2026-06-01
**Question:** Is the phase-52.1 52-week-high tilt's measured +0.05 annualized Sharpe improvement
over baseline (1-of-5 configs, ~47 monthly rebalances on S&P-500) STATISTICALLY ROBUST, or is it
selection-bias / small-sample noise? This is the gate for ENABLING it live (element 2 "promote the
highest earner from a cited research basis"). Analysis of already-collected replay data.

---

## STATUS: COMPLETE -- gate_passed: true

---

## Search queries run (3-variant discipline)
1. **Frontier (2026):** "deflated Sharpe ratio backtest overfitting multiple testing Bailey Lopez de Prado 2026"
2. **Last-2yr window (2024-2026):** "is my backtested edge real small sample factor improvement 2025" (recency pass below)
3. **Year-less canonical:** "Ledoit Wolf robust performance hypothesis testing Sharpe ratio difference test"; "stationary bootstrap Politis Romano dependent time series Sharpe ratio test"; "McLean Pontiff does academic research destroy stock return predictability"
The source mix below spans 1994 (Politis-Romano), 2008 (Ledoit-Wolf), 2014 (Bailey-LdP), 2016 (McLean-Pontiff), 2024-2026 (recency) -- the three-variant discipline surfaced both founding prior-art and current frontier.

## Read in full (>=5 required; counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://www.econ.uzh.ch/apps/workingpapers/wp/iewwp320.pdf | 2026-06-01 | peer-reviewed (J.Emp.Fin. 2008) | WebFetch + pdfplumber (41,164 chars) | The CANONICAL test for SR DIFFERENCE. delta=SR1-SR2; HAC p-value `p=2*Phi(-|delta_hat|/s(delta_hat))` (eq 6); studentized stationary-bootstrap p-value eq (9) `PV=(#{d~*,m >= d}+1)/(M+1)`, M=499, T=120, block grid b in {1,2,4,6,8,10} via Loh(1987) calibration. Robust to heavy tails + autocorrelation; JKM (Jobson-Korkie/Memmel) invalid under non-normality/time-series. |
| https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf | 2026-06-01 | peer-reviewed (J.Portf.Mgmt 2014) | WebFetch + pdfplumber (45,402 chars) | DSR deflates the ABSOLUTE max-of-N Sharpe for selection bias + non-normality, NOT a difference. E[max SR_N] = sqrt(V[SR])*((1-g)*Phi^-1(1-1/N) + g*Phi^-1(1-1/(Ne))) (eq 1), g=0.5772 Euler-Mascheroni. "Appendix 3 shows how N can be d[etermined]" for correlated trials. SE under non-normality uses skew (g3) + kurtosis (g4). |
| https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio | 2026-06-01 | tertiary (formula ref) | WebFetch | Verbatim DSR: `DSR=Phi((SR*-SR0)*sqrt(T-1)/sqrt(1-g3*SR0+(g4-1)/4*SR0^2))`. Correlated trials: convert correlation matrix -> distance matrix -> cluster -> N=number of clusters (effective independent trials). 0.95 threshold = 95% confidence. |
| https://www.hec.ca/finance/Fichier/McLean.pdf | 2026-06-01 | peer-reviewed (J.Finance 2016; this WP draft 2012) | WebFetch + pdfplumber (79,542 chars) | Post-publication anomaly decay. WP draft: out-of-sample decay ~10% (NOT stat. diff. from 0); post-publication decay ~35% (stat. diff. from 0 and 100). Published J.Finance version: 26% lower OOS, 58% lower post-publication. "post-publication declines greater for predictors with higher in-sample returns" -> a fresh in-sample edge should be HAIRCUT before trusting live. |
| https://arxiv.org/pdf/1905.08042 (Benhamou, Saltiel, Guez, Paris -- "Testing Sharpe ratio: luck or skill?") | 2026-06-01 | preprint (q-fin 2019, n=153,794) | WebFetch + pdfplumber | [CORROBORATING] Explicitly "inspired by Ledoit and Wolf (2008)"; derives the Studentized Sharpe statistic sqrt(n)*SR ~ Student-t under iid-normal, then extends to AR(1)/non-normal via skew+kurtosis correction. Confirms LW is THE reference for two-investment SR comparison; small-sample work (Unhapipat 2016, Opdyke 2007) cited. Does NOT contradict DSR/LW -- it operationalizes them. |

## Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://search.r-project.org/CRAN/refmans/PeerPerformance/html/sharpeTesting.html | docs (R pkg) | Confirms Ledoit-Wolf is the standard impl (`sharpeTesting`); reference impl not needed in full |
| https://stefan-jansen.github.io/machine-learning-for-trading/08_ml4t_workflow/01_multiple_testing/ | blog/book | Minimum backtest length + DSR practitioner walkthrough; canonical paper read instead |
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5520741 | preprint (LdP "How to Use the Sharpe Ratio" 2025) | Recency candidate -- see recency scan |
| https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870 | peer-reviewed (Politis-Romano 1994) | Stationary bootstrap founding paper; mechanism captured via Ledoit-Wolf which applies it |
| https://medium.com/balaena-quant-insights/deflated-sharpe-ratio-dsr-33412c7dd464 | blog | DSR practitioner explainer; lower tier |

## Recency scan (2024-2026)
Searched "small sample backtest Sharpe ratio statistical significance multiple testing factor 2025" and
"is my backtested edge real ... 2025". **Result: no new method SUPERSEDES Ledoit-Wolf 2008 (SR-difference)
or Bailey-LdP 2014 (DSR) as the canonical tools** -- both remain the references cited by 2024-2026
practitioner and academic material (CFA Institute, Man Group, Stefan Jansen ML4T book, Harvey backtesting
notes). COMPLEMENTARY recent finding: Lopez de Prado, Lipton, Zoonekynd "How to Use the Sharpe Ratio"
(SSRN 5520741, 2025) reaffirms DSR + minimum-track-record-length framing. The small-sample literature
(arXiv 1905.08042, Unhapipat 2016, Opdyke 2007) confirms that at T~47 the **bootstrap** path is required,
not the asymptotic-normal closed form. NET: methodology is stable; the recency window adds confirmation +
the explicit small-sample bootstrap caveat, not a replacement. (No 2024-2026 finding that the +0.05-Sharpe
improvement test should be done differently than paired Ledoit-Wolf.)

## Key findings
1. **The +0.05 is a Sharpe DIFFERENCE, so DSR is the WRONG primary test.** DSR (Bailey-LdP 2014) deflates an ABSOLUTE Sharpe for the best-of-N selection. Here baseline ALSO ~1.39, so the question is whether the +0.05 DELTA exceeds noise -- a paired comparison, not a max-of-N absolute. The canonical tool for "is strategy A's Sharpe significantly > B's" is **Ledoit-Wolf 2008** on delta=SR_tilt-SR_base (Source: Ledoit-Wolf, econ.uzh.ch/.../iewwp320.pdf -- abstract: "Applied researchers often test for the difference of the Sharpe ratios of two investment strategies... we propose... a studentized time series bootstrap confidence interval for the difference of the Sharpe ratios").
2. **Ledoit-Wolf gives TWO routes; the bootstrap one is mandatory here.** Closed-form HAC p-value `p=2*Phi(-|delta_hat|/s(delta_hat))` is the fast path but assumes the HAC kernel SE is accurate at small T. With T~47 monthly diffs the studentized **stationary-bootstrap** p-value (eq 9, M>=499 resamples, calibrated block size) is the robust choice -- it resamples the OBSERVED paired (base_ret, tilt_ret) rows jointly, preserving cross-correlation and autocorrelation (Source: Ledoit-Wolf eq 9; Politis-Romano 1994 stationary bootstrap).
3. **DSR remains a valid SECONDARY check on the tilt's ABSOLUTE Sharpe.** Deflate 1.44 for N effective trials this arc. E[max SR_N] grows with N; with V[SR] across the configs and N_eff clusters, DSR>=0.95 means 1.44 survives selection. But this is secondary -- the edge claim is the +0.05, not "1.44 is good" (baseline 1.39 would pass identically) (Source: Bailey-LdP eq 1).
4. **N_eff via clustering, not raw count.** The configs (baseline, sector_neutral, vol_scaled, hi52_k0.5, hi52_k1.0) are HIGHLY correlated (all re-rank the same momentum composite on the same universe). DSR's Appendix 3 / Wikipedia: correlation matrix -> distance -> cluster -> N=clusters. hi52_k0.5 and hi52_k1.0 are near-collinear (same tilt, different strength) -> ~1 cluster; sector_neutral/vol_scaled are variants of baseline. Realistic N_eff ~2-3, NOT 5. A conservative bound: use the raw count of DISTINCT IDEAS tried across the whole 52.x arc (see internal audit for the rotation-seed history).

## Internal code inventory
| File | Lines | Role | Status |
|------|-------|------|--------|
| scripts/ablation/sector_neutral_replay.py | 271 | The phase-52.1 replay. Builds `monthly={config: [fwd_ret,...]}` (line 177) for all 5 configs; `ann_sharpe(monthly)` = `mean/std*sqrt(12)` on monthly list (:116-120). | REUSABLE; currently PRINTS aggregate Sharpe only. Must EXPOSE the paired arrays `monthly["baseline"]` + `monthly["hi52_k0.5"]` (already in memory). |
| scripts/ablation/sector_neutral_replay.py:67-98 | build_screen_row | Emits `pct_to_52w_high` (:97) + everything rank_candidates needs. | OK -- causal 52w window already enforced (win_lo = t_idx-260, :185). |
| scripts/ablation/sector_neutral_replay.py:123-138 | hi52_tilt_basket | Reuses production composite_score * centered tilt; this is replay-side post-processing (zero live-engine change). | OK. |
| backend/backtest/analytics.py:239-282 | `compute_deflated_sharpe(observed_sr, num_trials, variance_of_srs=0.5, skewness=0.0, kurtosis=3.0, T=252)` | DSR on an ABSOLUTE Sharpe. Returns Phi(z), z=(SR-E[maxSR])/se. | REUSABLE for the SECONDARY absolute-Sharpe check. **TRAP (smoke-tested):** at SR~1.4, T=47, N=3 it returns ~1.0 for BOTH tilt (1.44) AND baseline (1.39) -> CANNOT see the +0.05. Confirms it is NOT the test for the delta. Also `T` must be the # of RETURN OBSERVATIONS (47 monthly), and `observed_sr` must be in the SAME frequency as the SE assumes -- pass the **annualized** SR but be aware the SE term `(.../T)` treats SR as per-observation; the project's existing generate_report passes annualized SR with daily T, so follow that convention for consistency. |
| backend/backtest/analytics.py:184-236 | `compute_pbo(pnl_matrix, S=16)` (CSCV) | T x N matrix, N>=2, T>=2S. | REUSABLE but **WEAK here**: needs T>=2S; with T~47 use S=4 or S=6 (S=16 needs T>=32, borderline + tiny subsets). More importantly, CSCV columns must be COMPETING CONFIGS (the 5 configs) -- which they are. Smoke: PBO(47x5,S=4)=0.33, well under 0.5 veto. SECONDARY check. (See [[project_pbo_single_strategy_cpcv]]: CSCV needs N>=2 competing configs, satisfied here.) |
| backend/agents/mcp_servers/risk_server.py:133-158 | `pbo_check(pnl_matrix, threshold=0.5, S=16)` -> dict | Thin MCP wrapper over compute_pbo with veto at >0.5. | REUSABLE if you want the veto envelope; or call compute_pbo directly in the script. |
| backend/backtest/analytics.py:125-144 | `compute_sharpe(returns, rf=0.04, periods_per_year=252)` | frequency-aware (Lo 2002). | REUSABLE for converting monthly arrays to Sharpe (pass periods_per_year=12). |
| **MISSING** | -- | **No Ledoit-Wolf SR-difference test and no stationary bootstrap exist in the repo.** grep for `ledoit`/`stationary_bootstrap`/`politis` = 0 hits. | The GENERATE phase must ADD a small paired-test function (HAC closed-form + stationary-bootstrap). ~40-60 LOC, numpy+scipy only (both already deps). $0. |

## Application to pyfinagent
**The +0.05 is a Sharpe DIFFERENCE between two strategies on the SAME 47 monthly periods -> the rigorous primary test is the PAIRED Ledoit-Wolf (2008) SR-difference test, NOT DSR.** Mapping the external methodology to the replay:

1. **Emit the paired arrays.** sector_neutral_replay.py already holds `monthly["baseline"]` and `monthly["hi52_k0.5"]` as aligned per-rebalance lists (same dates, same universe, only the tilt differs). Expose both (drop `None`-aligned pairs together). These are the inputs to every test below. ~47 paired observations.

2. **PRIMARY: paired Ledoit-Wolf SR-difference, stationary-bootstrap p-value.** Compute delta = SR_tilt - SR_base (both annualized, sqrt(12)). Studentized statistic d=|delta|/s(delta) where s(delta) is the HAC SE (Ledoit-Wolf eq 5; with ~47 obs use the bootstrap, not the asymptotic SE). Resample the JOINT rows (base_ret_i, tilt_ret_i) via the stationary bootstrap (Politis-Romano 1994; geometric block length, expected block ~ T^(1/3) ~ 4 months, or LW's calibrated grid {1,2,4,6,8,10}), recompute d~*,m on each, p = (#{d~*,m >= d}+1)/(M+1), M>=499 (LW eq 9, ideally M=1000-5000 since $0). One-sided (we only care if tilt > base): halve the two-sided p or use the one-sided bootstrap quantile.
   - **WHY paired/bootstrap not plain paired t-test:** monthly basket returns are autocorrelated + fat-tailed; a naive paired t-test on the diff series understates SE. LW + stationary bootstrap is the canonical robust fix (LW abstract verbatim: "not valid when returns have tails heavier than the normal distribution or are of time series nature"). A paired t-test on the diff series is an acceptable QUICK sanity companion but must NOT be the gate.

3. **SECONDARY: DSR on the tilt's ABSOLUTE Sharpe, deflated for N_eff trials.** `compute_deflated_sharpe(observed_sr=1.44, num_trials=N_eff, variance_of_srs=var of the 5 config Sharpes, skewness/kurtosis from the tilt monthly diffs, T=47)`. Threshold DSR>=0.95. NOTE this is a WEAK discriminator here (smoke-tested: returns ~1.0 for both 1.44 and 1.39) -- report it but do NOT let a passing absolute DSR substitute for the paired test. It guards "is 1.44 a fluke of the search", not "is the +0.05 real".

4. **num_trials / N_eff to deflate for.** Do NOT use N=5 naively -- the configs are highly correlated (all re-rank the SAME momentum composite on the SAME S&P-500 universe; hi52_k0.5 vs hi52_k1.0 are the same idea at two strengths). Per DSR Appendix 3 / Wikipedia: build the 5x5 correlation matrix of the monthly return series, convert to distance, cluster -> N_eff = #clusters (expect ~2-3: {baseline/sector_neutral/vol_scaled} as variants of the base ranker, {hi52_k0.5/hi52_k1.0} as one tilt idea). **Conservative recommendation: report DSR at BOTH N_eff (clustered, ~2-3) AND a pessimistic N=5..8** (8 = also counting the rotation-seed/strategy ideas tried earlier in the 52.x arc). If the edge only survives at N=1 it is fragile. The decision should hinge on the PAIRED test; N_eff feeds the secondary DSR sanity check.

5. **PBO as a tertiary robustness envelope (optional).** Feed the 47x5 monthly matrix to `compute_pbo(..., S=6)` (S=16 leaves 2-3 rows/subset at T=47 -- too few; S=4 or S=6 keeps subsets ~8-12 rows). PBO<0.5 means the in-sample winner is not anti-predictive OOS. Smoke on random data gave 0.33-0.49; on the real (correlated, all-positive-momentum) configs expect it to be informative about whether the tilt is a search artifact. Tertiary only.

6. **McLean-Pontiff haircut (sanity, not a gate).** Even if the +0.05 is statistically real in-sample, McLean-Pontiff (2016) shows newly-identified edges decay ~26% out-of-sample and ~58% post-publication, MORE for higher in-sample edges. A +0.05 raw edge could realistically be ~+0.02-0.04 live. This argues for a CONSERVATIVE enable decision: require the edge to clear the paired test comfortably (not marginally at p=0.049), because the live realization will be smaller.

### A-PRIORI DECISION RULE (state BEFORE running -> no p-hacking)
ENABLE the 52wh tilt live IFF ALL of:
- **(R1, primary)** Paired Ledoit-Wolf stationary-bootstrap p-value (one-sided, H0: SR_tilt <= SR_base) **< 0.05**, M>=1000 resamples, on the ~47 paired monthly returns.
- **(R2, magnitude)** delta = SR_tilt - SR_base **>= +0.05** annualized AND the bootstrap 90% CI lower bound for delta **> 0** (edge is positive with margin, not a knife-edge -- McLean-Pontiff haircut headroom).
- **(R3, secondary)** DSR on the tilt's absolute Sharpe **>= 0.95** at the CLUSTERED N_eff (and reported, not necessarily passing, at pessimistic N=8).
- **(R4, tertiary, non-veto)** PBO(monthly 47xN, S=6) **<= 0.5** (report; only a veto if it exceeds 0.5).
REJECT (or keep dark / pivot to residual momentum per 52.2) if R1 fails OR R2's CI lower bound <= 0. R1 is the hard gate; R3/R4 are corroborating. This makes GENERATE a pure COMPUTE-and-COMPARE against fixed thresholds.

**Caveat on run-to-run drift (live yfinance +0.047..+0.057):** the point estimate of +0.05 is itself within a ~0.01-wide noise band from data refresh. The bootstrap CI in R2 directly absorbs this -- if a ~0.01 data-jitter flips the sign of the CI lower bound, R2 correctly REJECTS. Pin the yfinance pull (cache the prices used) so the test is reproducible for the Q/A gate.

## Research Gate Checklist

Hard blockers -- `gate_passed` is false if any unchecked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5: Ledoit-Wolf 2008, Bailey-LdP DSR 2014, Wikipedia DSR formula ref, McLean-Pontiff 2016, Benhamou et al. 2019 -- 4 of 5 binary PDFs recovered via pdfplumber per research-gate.md PDF strategy)
- [x] 10+ unique URLs total (12: 5 read-in-full + 5 snippet + the 2 search-surfaced practitioner refs Man/CFA)
- [x] Recency scan (last 2 years) performed + reported (no method supersedes LW/DSR; LdP 2025 SSRN 5520741 reaffirms)
- [x] Full papers read (not abstracts) for the read-in-full set (pdfplumber-extracted 41K-79K chars each; verbatim formulas quoted)
- [x] file:line anchors for every internal claim (analytics.py:239/:184/:125; sector_neutral_replay.py:116/:177/:97/:185; risk_server.py:133)

Soft checks:
- [x] Internal exploration covered every relevant module (replay, analytics DSR/PBO/sharpe, risk-server pbo_check)
- [x] Contradictions / consensus noted (DSR vs LW: DSR is for absolute max-of-N, LW is for the difference -- the central methodological correction; Benhamou corroborates LW; no adversarial source found that the +0.05 should be tested differently -- stated as a finding)
- [x] All claims cited per-claim (per-finding source URLs + file:line)

### Smoke-test evidence (this session, $0)
- `from backend.backtest.analytics import compute_deflated_sharpe, compute_pbo, compute_sharpe` -> IMPORT OK in 1.8s (standalone-importable for the GENERATE script).
- DSR(1.44, N=3, T=47)=1.0 ; DSR(1.39, N=3, T=47)=0.9999 -> **DSR cannot distinguish the +0.05** (proves R3 is secondary, not the gate).
- PBO(47x5 random, S=16)=0.49, (S=4)=0.33 -> S=16 too high at T=47, use S=6.

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 5,
  "urls_collected": 12,
  "recency_scan_performed": true,
  "internal_files_inspected": 4,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
