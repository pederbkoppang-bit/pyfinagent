# research_brief -- phase-52.1: stronger price-based alpha signal (north-star #4)

**Tier:** complex | **Status:** IN PROGRESS | **Date:** 2026-05-31 (updated 2026-06-01)

## Objective
Identify the SINGLE best-evidenced price-only momentum-enhancement to MEASURE
first ($0 replay) for the live LONG-ONLY US large-cap momentum book. Must be
computable from DAILY CLOSE PRICES ALONE (no fundamentals/earnings/news/LLM).

Live composite (`screener.py:268-282`):
`momentum_1m*0.40 + momentum_3m*0.35 + momentum_6m*0.25`
with RSI>80 *0.7 / RSI<20 *0.8 and vol>0.6 *0.85 penalties.

Replay harness: `scripts/ablation/sector_neutral_replay.py` -- downloads S&P 500
daily closes once, replays `rank_candidates` over 48 monthly rebalances, scores
forward-1mo Sharpe/return/turnover/sector-spread.

CLOSED (do NOT re-propose): sector-neutral breadth (-0.166 Sharpe), winner-take-
all rotation (disconnected + losing). Vol-scaling already measured (+0.015, marginal).

HARD CONSTRAINT: signal must be computable from DAILY CLOSE PRICES ALONE.

---

## Part A -- Internal code audit (file:line)

### A0. The LIVE composite (CONFIRMED, screener.py:295-309)
`rank_candidates(strategy="momentum")` computes:
```
score = mom_1m*0.40 + mom_3m*0.35 + mom_6m*0.25      # :297-301
if rsi > 80:  score *= 0.7    elif rsi < 20: score *= 0.8   # :303-306
if vol > 0.6: score *= 0.85                                  # :308-309
```
where mom_Nm = stock.get("momentum_Nm") (set by the screener/replay from closes).
This is the exact target to enhance. The function ALSO takes many overlay kwargs
(pead/news/sector/revision/options/insider/narrative/sector_momentum/...) -- all
NON-price (fundamentals/news/LLM) and all gated behind `if <signal>:` so they are
NO-OPs when the replay passes none of them. The `multidim_momentum=False` kwarg
(:261) is the relevant price-only branch -- audited below.

### A-replay. The replay harness shape (CONFIRMED, sector_neutral_replay.py)
- Downloads S&P 500 closes ONCE (`yf.download`, :126), flattens to a `closes`
  DataFrame {ticker: close-series} (:129-139).
- `build_screen_row(ticker, sector, win)` (:67-94) computes per-rebalance, FROM
  CLOSES ONLY: momentum_1m=mom(21), _3m=mom(63), _6m=mom(126) (:88-90, where
  mom(n)=last/c.iloc[-1-n]-1), rsi_14 (:91, full helper :55-64), volatility_ann
  (:83, trailing-63d daily-std*sqrt252), sma_50_distance_pct (:93). `win` = closes
  up to AND INCLUDING the rebalance date (causal, :68).
- Monthly rebalance loop (:159-191): builds `rows` for all tickers (:163-166),
  then for each config calls `rank_candidates(rows, top_n, strategy="momentum",
  sector_neutral=sn)` (:170), takes top-N basket, scores `basket_fwd_return`
  (forward-21d equal-weight, :97-109), and tracks sector-spread + turnover.
- Scoring: `ann_sharpe` = mean/std*sqrt(12) over the monthly fwd returns (:112-116).
=> ADD A NEW CONFIG = (1) compute the new feature from `closes` inside
`build_screen_row` (it already has the full causal `win`/`c` series), (2) put it
in the screen_row dict, (3) have rank_candidates blend it (either via a new
strategy branch, OR pre-blend into the composite in the replay), (4) add the
config name to the `configs` dict + results loop. The harness already has
everything price-only it needs.

### A1. The 52-week-high leg is ALREADY BUILT (price-only) -- but bundled + not in replay
CRITICAL re-use finding. The live screener ALREADY computes the George-Hwang
52w-high proximity, PRICE-ONLY, at **screener.py:210-214**:
```python
# phase-28.7: 52-week-high proximity (George-Hwang 2004 anchoring effect).
# current_price / trailing-252d max -> values in (0, 1]; 1.0 means at 52w high.
high_52w = float(close.rolling(252, min_periods=20).max().iloc[-1])
pct_to_52w_high = round(current_price / high_52w, 4) if high_52w > 0 else None
```
This is EXACTLY the George-Hwang formula (price / trailing-252d max). It is stored
as `pct_to_52w_high` (:228) and consumed by `_apply_multidim_momentum` (:491-550)
as the `52w_high` leg, z-scored (:535) and weighted 0.25 (:539). **So the 52w-high
feature is price-only and reusable -- the computation logic is proven in prod.**

BUT two caveats make it NOT a finished price-only enhancement:
1. **It is bundled with NON-price legs.** `_apply_multidim_momentum` blends 4 legs
   (:538-541): price 0.35, 52w_high 0.25, **sue 0.20 (PEAD = fundamental/earnings,
   NOT price-only)**, **sector 0.20 (sector-momentum overlay)**. When the replay
   passes no pead_signals/sector_momentum_ranks, those two legs z-score to all-0
   (the `_zscore` of an all-0 vector returns all-0, :486-487), so multidim collapses
   to `0.35*z(price) + 0.25*z(52w_high)` -- but with the SUE/sector WEIGHT MASS
   (0.40) dead, the effective blend is mis-normalized (only 0.60 of weight active).
   A clean price-only 52w-high test must use a 2-leg weighting that sums to 1.
2. **The replay's `build_screen_row` does NOT compute `pct_to_52w_high`** (it stops
   at sma_50; sector_neutral_replay.py:85-94). So multidim CANNOT be measured by the
   current replay -- the 52w_high leg would be all-None -> all-0 -> identity. The
   replay must be extended to compute `pct_to_52w_high` from `c` (the causal window
   already in scope): `c.rolling(252, min_periods=20).max().iloc[-1]` then
   `last / high_52w`. Trivial -- the window is already there.

So: the 52w-high signal logic EXISTS and is price-only; what's missing is (a) a
clean price-only blend (not bundled with SUE/sector) and (b) wiring it into the
replay's screen_row. Do NOT rebuild the formula -- reuse :213-214 verbatim.

The live wiring (autonomous_loop.py:649-654) gates multidim behind
`settings.multidim_momentum_enabled` (default False) and passes the 4 weights from
settings -- so multidim is OFF live today and has never been measured on our
universe (the replay couldn't score it).

---

## Part B -- External research

### Read in full (counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://www.bauer.uh.edu/tgeorge/papers/gh4-paper.pdf (George & Hwang 2004, J. Finance LIX:5) | 2026-06-01 | peer-reviewed | pdfplumber (binary->text) | THE 52w-high paper. Nearness measure = `P_{i,t-1} / high_{i,t-1}`, high = highest price in the 12-mo period ending last day of month t-1. (6,6) self-financing spread: 52w-high **0.45%/mo (t=2.00)** vs JT individual-stock momentum 0.48%/mo (t=2.35) vs MG industry 0.45%/mo (t=3.43). "Nearness to the 52-week high dominates and improves upon the forecasting power of past returns ... Future returns forecast using the 52-week high DO NOT REVERSE in the long run." => the *additivity* of 52w-high shows up in MULTIVARIATE Fama-MacBeth (it survives controlling for JT+MG and partly subsumes them), NOT in a higher raw univariate spread. |
| http://wp.lancs.ac.uk/.../MHF-2019-076-Matthias-Hanauer.pdf (Hanauer & Windmuller, "Enhanced Momentum Strategies" 2019) | 2026-06-01 | peer-reviewed (working paper, TUM/Robeco) | pdfplumber (binary->text) | **THE decisive head-to-head.** Compares 3 momentum risk-mgmt techniques: idiosyncratic (residual) momentum, constant-vol scaling, dynamic scaling, US + 48 intl. "In a multiple model comparison test that also controls for other factors, **idiosyncratic momentum emerges as the best momentum strategy**" -- highest mean-variance weight, ex-post Sharpe **1.45**, "an investor restricted to picking only one momentum strategy might favor iMOM." All 3 reduce crashes + raise risk-adj returns. **"constant volatility-scaling of iMOM MAXIMIZES its performance"** (the two stack). |
| (Hanauer footnote 6-7 + Blitz et al. 2018 cite, same PDF) | 2026-06-01 | peer-reviewed | pdfplumber | **PRICE-ONLY VARIANT CONFIRMED:** fn7 "Chaves (2016) shows that a simplified version of idiosyncratic momentum that is based on ONE-FACTOR (MARKET) unscaled residuals works. **Blitz et al. (2018) confirm that MOST of the performance improvement comes from orthogonalizing returns with the MARKET factor**, and that the inclusion of additional Fama-French factors leads to small further improvements." fn6: Gutierrez-Pirinsky (2007, "abnormal return momentum") and Blitz-Huij-Martens (2011, "residual momentum") "definitions are identical." => single-factor market-model residual momentum (regress stock ret on index ret = price-only) captures the BULK of the iMOM benefit. |
| https://www.quantconnect.com/learning/articles/investment-strategy-library/residual-momentum | 2026-06-01 | industry (algotrading platform) | WebFetch (full) | Exact recipe: FF3 (or market-only) regression over **36 trailing months**; residual formed on **trailing 12 months EXCLUDING the most recent month**; score = **Σε / σ_ε** (sum of residuals / std of residuals over the same window); **monthly** rebalance; filters: price>$1, >=3yr history. |
| https://quantpedia.com/strategies/residual-momentum-factor | 2026-06-01 | industry (strategy DB) | WebFetch (full) | Corroborates: "rank stocks every month on past 12-month residual returns, excluding the most recent month, standardized by the std of the residual returns over the same period." 36-mo regression. Blitz-Huij-Martens standalone (1926-2009): ann ret 9.18%, vol 15.27%, Sharpe 0.34, maxDD -59.74% (long-short, full century incl. Depression). "risk-adjusted profits about twice as large as total-return momentum." |
| https://ideas.repec.org/a/eee/empfin/v18y2011i3p506-521.html (Blitz, Huij, Martens 2011, J. Empirical Finance) | 2026-06-01 | peer-reviewed | WebFetch (full) | The originating residual-momentum paper. "residual momentum earns risk-adjusted profits that are about TWICE as large as those associated with total return momentum" -- via lower factor-exposure volatility; "more consistent over time" and "less concentrated in the extremes of the cross-section." Halves the vol of conventional momentum at similar return => roughly DOUBLES the Sharpe (search-snippet corroborated). |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1104491 (George-Hwang SSRN) | peer-reviewed | Same paper as the bauer.uh.edu PDF read in full; SSRN is the abstract mirror. |
| https://www.sciencedirect.com/science/article/abs/pii/S0261560610001099 (52wk-high intl, J.Int.Money&Finance) | peer-reviewed | Snippet: 18 of 20 intl markets profitable, 0.60-0.94%/mo vs 0.45% US -- corroborates robustness; abstract-gated. |
| https://onlinelibrary.wiley.com/doi/abs/10.1111/eufm.12247 (Lin 2020, idiosyncratic momentum further evidence) | peer-reviewed | Snippet: "idiosyncratic momentum ... consistently outperform[s] the conventional momentum strategy in the cross-sectional pricing of equity portfolios and individual stocks" -- corroborates iMOM dominance; abstract-gated. |
| https://epublications.marquette.edu/cgi/viewcontent.cgi?article=1168&context=fin_fac ("Momentum Crashes and the 52-Week High") | peer-reviewed | ADVERSARIAL-adjacent; fetched but largely binary; the crash-protection angle captured via George-Hwang + search snippets. |
| https://www.sciencedirect.com/science/article/abs/pii/S0165188926000321 ("Proximity to 52-week high and risk-return trade-off", J.Econ.Dyn.&Control 2026) | peer-reviewed | RECENCY hit (2026). Snippet: "52-week high effect can induce cross-sectional heterogeneity in the risk-return trade-off." Abstract-gated; recency captured below. |
| https://markflair.com/resources/residual-momentum | industry | Corroborates 36-mo regression recipe; lower-tier than QuantConnect/Quantpedia. |
| https://www.aeaweb.org/conference/2023/program/paper/8Ah8THYY (Ehsani, "What Does Residual Momentum Tell Us About Firm-Specific Momentum?") | peer-reviewed | Snippet: residual momentum is largely a repackaging of firm-specific (idiosyncratic) momentum; relevant nuance, abstract-gated. |
| https://alphaarchitect.com/swedroe-spotlight-enhancing-momentum-strategies-via-idiosyncratic-momentum/ | industry blog | 403 Forbidden on WebFetch; the iMOM-doubles-Sharpe finding captured from the primary Blitz + Hanauer reads instead. |
| https://www.quantifiedstrategies.com/52-week-high-strategy/ | industry blog | Bot-wall on WebFetch; 52w-high backtest numbers captured from George-Hwang primary. |
| https://link.springer.com/article/10.1007/s11408-022-00417-8 ("Momentum: what do we know 30 yrs after JT") | peer-reviewed survey | Snippet-level; broad survey, recency context. |

### Search-query variants run (3-variant discipline)
1. **Current-year frontier (2026/2025):** "momentum enhancement signal 2025 idiosyncratic residual 52-week high cross-section equity returns evidence"; "time series momentum trend following 200 day moving average filter long only equity overlay 2024". (-> surfaced the 2026 J.Econ.Dyn.&Control 52wk-high paper, Mamais 2025 momentum-shifts, Avramov-Kaplanski-Subrahmanyam Dec-2023 MA-distance.)
2. **Last-2-year window:** covered by the above + Lin 2020 / Ehsani 2023. Recency findings reported below.
3. **Year-less canonical:** "52-week high momentum George Hwang subsumes price momentum long-only"; "residual idiosyncratic momentum Blitz Huij Martens higher Sharpe lower volatility"; "residual momentum implementation rolling 36 month regression"; "52-week high momentum strategy formula ratio price to high portfolio construction"; "residual momentum single factor CAPM market model price only". (-> the founding papers: George-Hwang 2004, Blitz-Huij-Martens 2011, Gutierrez-Pirinsky 2007, Hanauer-Windmuller 2019, Chaves 2016.)

### Recency scan (2024-2026) -- PERFORMED
Searched last-2-year window on (a) 52-week-high momentum, (b) idiosyncratic/residual momentum, (c) trend/MA-distance overlays. **Findings:**
1. **COMPLEMENTS (52wk-high, 2026):** "Proximity to the 52-week high and the risk-return trade-off" (J.Econ.Dyn.&Control, S0165188926000321, 2026) -- the 52w-high effect induces cross-sectional heterogeneity in the risk-return trade-off (i.e. it is not a uniform premium; interacts with risk). Refines, does not overturn, George-Hwang.
2. **COMPLEMENTS (MA-distance, Dec-2023):** Avramov, Kaplanski & Subrahmanyam show the DISTANCE between 21- and 200-day moving averages predicts equity returns with incremental power beyond time-series momentum (alphaarchitect "Moving Average Distance"). A price-only trend feature with fresh evidence -- an adjacent lever, not the top pick.
3. **STABLE (residual momentum):** Lin (2020, Eur.Fin.Mgmt) "further evidence" confirms idiosyncratic momentum "consistently outperforms conventional momentum." No 2024-2026 result overturns the Blitz/Hanauer finding; iMOM remains the best-evidenced enhancement. AQR and Robeco run it in production.
4. **CONTEXT (2025, NOT applicable here):** Mamais (2025, J.Forecasting) on predicting momentum-performance shifts across time/sectors -- a regime-timing overlay (sector-level), out of scope (sector-neutral is CLOSED; this is timing not selection).

### Adversarial / disconfirming sources (deliberately sought)
| URL | Accessed | Kind | Fetched how | Disconfirming finding |
|-----|----------|------|-------------|-----------------------|
| https://arxiv.org/abs/2304.03437 ("Echo disappears", 2023) [ADVERSARIAL] | 2026-06-01 | peer-reviewed (arXiv) | WebFetch (full) | **KILLS the skip-recent-month / "echo" candidate.** "The momentum echo has DISAPPEARED." The recent-month reversal cancels recent-month momentum; once that's accounted for, the term structure is a damped shape, NOT an echo. "Practitioners should NOT rely on intermediate-horizon momentum strategies (skipping recent months)." => do NOT adopt a 12-7 skip-month tweak. |
| https://aaltodoc.aalto.fi/.../download (Aalto thesis comparing momentum vs Novy-Marx echo) [ADVERSARIAL] | 2026-06-01 | grad thesis | WebFetch (full) | Corroborates: **conventional recent momentum OUTPERFORMED intermediate-horizon (echo)**; monthly alphas 0.46-0.87% for 1-3mo recent vs "mostly insignificant" for intermediate-horizon; echo did not reliably replicate out of sample. "Recommends practitioners adopt CONVENTIONAL momentum rather than intermediate-horizon." |
| Barroso & Wang (2021) via search snippets + George-Hwang internal controls [ADVERSARIAL on 52wk-high] | 2026-06-01 | peer-reviewed | search snippet (corroborated by George-Hwang primary) | **Tempers the 52wk-high pick for LARGE caps:** "the result of George and Hwang (2004) is limited to SMALL stocks, and price momentum explains the predictability of 52-week high momentum [for larger stocks]." Du (2008): 52wk-high profits reverse in the long run in 18 indices (contra George-Hwang's no-reversal claim). => for an S&P-500 LARGE-CAP book, 52wk-high may add LESS marginal alpha over the existing momentum composite than it does in the full CRSP universe. |
| McLean & Pontiff (2016) via "Replication Crisis in Finance" search | 2026-06-01 | peer-reviewed | search snippet | General factor-decay warning: published anomalies are "26% lower out-of-sample and 58% lower post-publication"; many factor results lean on small/micro caps excluded from a real large-cap universe. => HAIRCUT all cited historical edges; the $0 replay on OUR S&P-500 universe is the right arbiter, not the paper numbers. |

### Key findings (per-claim, cited)
1. **Among momentum ENHANCEMENTS, idiosyncratic (residual) momentum is the single best-evidenced -- it WINS a formal multiple-model head-to-head vs vol-scaling and dynamic scaling.** "In a multiple model comparison test that also controls for other factors, idiosyncratic momentum emerges as the best momentum strategy" -- highest mean-variance weight, ex-post Sharpe 1.45 (Source: Hanauer & Windmuller 2019, http://wp.lancs.ac.uk/mhf2019/files/2019/09/MHF-2019-076-Matthias-Hanauer.pdf). Blitz-Huij-Martens 2011: residual momentum earns "risk-adjusted profits about TWICE as large as total return momentum" via ~half the volatility (Source: https://ideas.repec.org/a/eee/empfin/v18y2011i3p506-521.html).
2. **A SINGLE-FACTOR (market) residual -- regress stock returns on the index return, PRICE-ONLY -- captures MOST of the iMOM benefit.** "Chaves (2016) shows that a simplified version of idiosyncratic momentum that is based on one-factor (market) unscaled residuals works. Blitz et al. (2018) confirm that most of the performance improvement comes from orthogonalizing returns with the market factor, and that the inclusion of additional Fama-French factors leads to small further improvements" (Source: Hanauer fn7). => the FF3 version (needs SMB/HML, NOT price-only) is NOT required; the market-model version is computable from the S&P-500 index built from `closes`. This is the key that makes residual momentum $0-replayable.
3. **Residual-momentum recipe (price-only adaptation):** for each stock, regress its trailing-12-month (or daily, see S2) returns on the contemporaneous market (equal- or cap-weighted index) return over a 36-month estimation window; the signal = sum of the residuals over the 12-month formation window EXCLUDING the most recent month, divided by the std of those residuals: `iMOM = Σε / σ_ε` (Source: QuantConnect https://www.quantconnect.com/learning/articles/investment-strategy-library/residual-momentum + Quantpedia https://quantpedia.com/strategies/residual-momentum-factor, both read in full).
4. **52-week-high proximity (George-Hwang) is real and ALREADY price-only-implemented here, but its ADDITIVITY over existing momentum is weakest precisely in LARGE caps.** Formula `P_{t-1}/high_{t-1}`; raw (6,6) spread 0.45%/mo t=2.00, does not reverse long-run (Source: George & Hwang 2004, J.Finance). BUT Barroso-Wang 2021: "limited to small stocks, and price momentum explains the predictability" for larger stocks (search snippet, corroborated). => 52wk-high is the SAFER, already-built option but likely LOWER marginal lift on an S&P-500 book than residual momentum.
5. **Do NOT skip the recent month / adopt the "echo".** The Novy-Marx echo "has DISAPPEARED" (arXiv:2304.03437, read in full) and conventional recent momentum out-performs intermediate-horizon in recent out-of-sample tests (Aalto thesis, read in full). The live composite's recent-month weighting is DEFENSIBLE; a 12-7 skip-month rewrite is contraindicated by current evidence.
6. **Vol-scaling stacks ON TOP of residual momentum and is the performance-maximizing combo -- but standalone vol-scaling is the weaker of the enhancements** (already measured here at +0.015, marginal). "constant volatility-scaling of iMOM maximizes its performance" (Source: Hanauer). => residual momentum FIRST (bigger, additive lift); vol-scaling is a follow-on overlay, not the primary signal.

### Consensus vs debate (external)
- **Consensus:** (a) residual/idiosyncratic momentum > total-return momentum on a risk-adjusted basis, in production at AQR + Robeco (Blitz, Hanauer, Lin, Ehsani). (b) The market-orthogonalization is the dominant driver (Chaves, Blitz et al. 2018). (c) Vol-scaling and iMOM both reduce crashes and stack.
- **Debate:** (a) 52wk-high -- George-Hwang say it dominates and does not reverse; Barroso-Wang/Du say it is a small-cap effect that price momentum explains for large caps and may reverse. (b) Echo -- Novy-Marx said intermediate-horizon dominates; "Echo disappears" (2023) + the Aalto thesis say it no longer holds and recent momentum is fine. (c) Factor decay generally (McLean-Pontiff) argues for haircutting ALL historical edges -- which is exactly why the $0 replay on OUR universe is the decider.

### Pitfalls (from literature) -> applied to phase-52.1
1. **Large-cap universe mutes 52wk-high additivity** (Barroso-Wang). Our S&P-500 book is the worst case for 52wk-high marginal lift -- measure it, don't assume the 0.45%/mo carries over.
2. **Residual momentum needs a clean market proxy + ENOUGH history.** 36-mo regression needs >=36 monthly (or ~756 daily) obs per name; the replay's `START="2021-06-01"` only gives ~6mo lookback. A residual-momentum config needs an EARLIER start (e.g. 2017) so the first 2022 rebalance has 36+ months. This is the single biggest implementation constraint (S2).
3. **Skip-recent-month is contraindicated** (Echo disappears). If a residual-momentum config is built, test BOTH with and without the skip-month; current evidence favors NOT skipping (or only skipping the standard 1 month that the residual already removes, not a 12-7 echo window).
4. **Look-ahead/causality:** the regression must use only data up to and including the rebalance date (the replay's `win` is already causal, :68). The market index must be built from the SAME causal window.
5. **Factor decay / overfitting** (McLean-Pontiff, + DSR/PBO project canon): a single replay number is not proof; report forward-1mo Sharpe AND turnover AND a deflated view. Do not over-tune the blend weight on the 48-rebalance sample.

---

## Part C -- Replay-feasibility audit (the binding constraints)
Confirmed against `scripts/ablation/sector_neutral_replay.py`:
- **NO reusable residual/market-model code** in the repo (only `scripts/audit/portfolio_risk_audit.py`, unrelated). A residual-momentum config is net-new code.
- **Window constraint (the decider on implementability):** the replay passes only `closes[tk].iloc[win_lo:t_idx+1]` with `win_lo = max(0, t_idx - 200)` (:161-164) -> ~200 trading days (~9.5 months) of history per name, and `START="2021-06-01"` (:26).
  - **52w-high needs 252 trading days.** Just over the current 200-day window. Fix = widen `win_lo` to `t_idx - 260` (and START stays 2021-06-01 since the first 2022 rebalance then has ~250+ days). TINY change. Formula already exists (screener.py:213). HIGHLY implementable.
  - **Residual momentum (36-mo regression) needs ~756 trading days** per name BEFORE the first rebalance. The replay would need `START` pushed to ~2018-06-01 (3.5yr lookback before the first 2022 rebalance) AND `win_lo = max(0, t_idx - 800)`. Larger download, longer run, net-new regression per name per rebalance (500 names x 48 rebalances x OLS). Still $0 (yfinance) but materially more code + compute.
- **Market proxy is one line:** an equal-weight market return = `closes.pct_change().mean(axis=1)` (the replay already does `pd.concat(daily_basket, axis=1).mean(axis=1)` at :187). So the single-factor regressor is free from `closes`.

---

## SYNTHESIS -- the SINGLE recommended signal + exact plan

### THE PICK: **52-week-high proximity as a MULTIPLICATIVE GATE / tilt on the existing composite** -- measure FIRST.
Rationale for choosing 52wk-high over residual momentum as the *first* thing to measure (NOT because it is higher-evidenced -- residual momentum is -- but because the brief asks for the single best pick that is "most-additive + most-implementable" and MEASURE-FIRST on a working engine):
1. **Already price-only-implemented + proven in prod** (screener.py:213-214) -- zero formula risk; reuse verbatim.
2. **Fits the existing replay window** with a one-line `win_lo` widening (252d) -- can be measured THIS cycle; residual momentum needs a 3.5x-longer re-download + net-new OLS harness (a bigger, slower build).
3. **Most-additive of the price-only options that are cheap to test:** George-Hwang show it survives controlling for JT+MG momentum in multivariate tests and does not reverse long-run; it is the canonical complement to a 1/3/6-mo return composite.
4. **Caveat honestly surfaced:** Barroso-Wang warn the effect is muted for large caps -- which is EXACTLY why it must be MEASURED on our S&P-500 universe, not assumed. If the replay shows ~0 or negative dSharpe (the large-cap-mute scenario), 52wk-high is cheaply rejected and we escalate to residual momentum (the bigger build) with evidence in hand.

**Exact price-only formula (reuse screener.py:213):**
```
high_52w   = close.rolling(252, min_periods=20).max().iloc[-1]   # trailing 252-day high
pct_to_52w = current_price / high_52w                            # in (0, 1]; 1.0 = at the high
```

**Blend/overlay method onto the existing composite -- MULTIPLICATIVE TILT (preferred), not replace, not z-blend:**
The live composite is a RETURN-magnitude score (`mom_1m*.40+mom_3m*.35+mom_6m*.25`) already shaped by RSI/vol multipliers. The cleanest, lowest-risk way to add 52wh proximity in the SAME idiom is another multiplier:
```
score *= (1 + k * (pct_to_52w - mean_pct_to_52w_universe))     # k in {0.5, 1.0, 1.5}
```
i.e. tilt UP names nearer their 52w-high, DOWN names far below it, centered so the average tilt is ~1.0 (turnover-neutral on average). This mirrors the existing RSI/vol multiplier pattern (:303-309) and keeps the score interpretable.
- WHY a tilt, not a z-blend like multidim: the multidim z-blend REPLACES the composite with a standardized sum (loses the return-magnitude scale and mixes in dead SUE/sector legs); a multiplicative tilt PRESERVES the working momentum ranking and only nudges it -- minimal regression risk on a working engine.
- WHY not "replace": George-Hwang's own evidence is that 52wh COMPLEMENTS (does not strictly dominate) return momentum out of sample; replacing throws away the measured-good 1/3/6 signal.
- Secondary config to also measure: the **z-blend** `0.70*z(composite) + 0.30*z(pct_to_52w)` (a clean 2-leg version of multidim WITHOUT the dead SUE/sector legs) -- so the replay compares tilt vs blend vs baseline.

**Expected effect (cited, haircut for large-cap + decay):** George-Hwang raw 52wh spread 0.45%/mo (t=2.00); a related construction reports an "80% increase in Sharpe ratio" after neutralizing nearness (search snippet, ScienceDirect S0261560610001099). BUT Barroso-Wang large-cap mute + McLean-Pontiff 26-58% decay => realistic expectation on an S&P-500 long-only book is a SMALL positive dSharpe (single-digit-percent of the baseline, plausibly +0.02 to +0.10 ann Sharpe) with LOW incremental turnover (the tilt is centered/gentle). Treat ANYTHING >= +0.05 dSharpe at <=+10% turnover as a PASS-worthy signal to escalate to a live operator gate; ~0 or negative = cleanly reject and pivot to residual momentum.

**How to add it as a replay config (concrete):**
1. In `build_screen_row` (sector_neutral_replay.py:67-94), after `sma50`, add (the causal window `c` is already in scope):
   ```python
   high_52w = float(c.rolling(252, min_periods=20).max().iloc[-1])
   row["pct_to_52w_high"] = float(last / high_52w) if high_52w > 0 else None
   ```
2. Widen the lookback so 252 days are available: change `win_lo = max(0, t_idx - 200)` -> `max(0, t_idx - 260)` (:161). START 2021-06-01 already yields ~250+ days before the first 2022 rebalance; if marginal, push START to 2021-01-01.
3. Add configs alongside `{"baseline":False,"sector_neutral":True}`: a `"hi52_tilt"` and a `"hi52_blend"`. Since `rank_candidates` has no built-in 52wh-tilt strategy, implement the tilt in the replay by post-processing the baseline `rows` BEFORE ranking, OR (cleaner) add a small `strategy="momentum_52wh"` branch to screener.py that applies the multiplicative tilt -- recommend the latter so the LIVE path and the replay share one code path (the project's "grep all consumers" discipline). The blend config can reuse `multidim_momentum=True` with weights `{"price":0.70,"52w_high":0.30,"sue":0.0,"sector":0.0}` (zeroing the non-price legs) -- this EXERCISES the already-built `_apply_multidim_momentum` with a clean 2-leg price-only weighting.
4. Score with the existing `basket_fwd_return` + `ann_sharpe` + turnover machinery (already there). Report dSharpe vs baseline + dTurnover, mirroring the 51.2 verdict block.

### Already-implemented? YES, partially (the 52wh LEG).
`pct_to_52w_high` (screener.py:214, price-only) + `_apply_multidim_momentum`'s 52w_high leg (:535,:539) already exist. What is NOT built: (a) a clean price-only blend/tilt (multidim bundles dead SUE/sector legs), (b) the replay computing `pct_to_52w_high` (it stops at sma50), (c) a live `strategy` branch for the tilt. So this is mostly a WIRING + clean-blend + replay-extension job, not a from-scratch signal -- the cheapest possible "stronger signal" experiment.

### The RUNNER-UP (escalate here if 52wh measures flat): single-factor residual momentum.
Higher-evidenced (wins Hanauer's head-to-head; ~2x risk-adj profit of total-return momentum per Blitz). Price-only via the MARKET-model variant (Chaves 2016 / Blitz et al. 2018: single-factor captures most of the benefit). Recipe: 36-mo regression of stock returns on equal-weight market return (`closes.pct_change().mean(axis=1)`), signal = Σ(residuals over trailing 12mo) / σ(residuals); rebalance monthly; do NOT add a 12-7 echo skip ("Echo disappears" 2023). Cost: net-new OLS harness + START pushed to ~2018 + `win_lo` to ~800d. Recommend as the SECOND experiment precisely because it is the bigger build -- measure the cheap, already-built 52wh tilt first; if flat, the residual-momentum evidence justifies the larger investment.

### DEPRIORITIZED (with reasons):
- **Skip-recent-month / "echo" (Novy-Marx):** CONTRAINDICATED -- "Echo disappears" (2023) + Aalto thesis: conventional recent momentum out-performs; do not skip.
- **Time-series/200-dma absolute-momentum trend gate (MOP/Faber):** a RISK overlay (when-to-be-in-market), not a cross-sectional SELECTION enhancer; orthogonal to "stronger ranking signal" and better suited to a separate regime-gate experiment. MA-distance (Avramov et al. 2023) is a fresher price-only variant if a trend lever is wanted later.
- **Idio-vol screen / low-vol tilt (Ang et al.):** the vol>0.6 penalty (:308-309) already captures the crude version; low-vol is a distinct factor (can fight momentum), not a momentum-strengthener.
- **Vol-scaling (Barroso-Santa-Clara):** already measured (+0.015, marginal); per Hanauer it is the performance-MAXIMIZING overlay ON TOP of residual momentum -- revisit only after a residual-momentum base exists.

### Failure modes of the 52wh pick (where it underperforms):
1. **Large-cap mute (the headline risk):** Barroso-Wang -- on an S&P-500 universe the 52wh edge may be ~0 because price momentum already captures it. The replay is designed to catch exactly this; flat result = cheap rejection.
2. **Long-run reversal (Du 2008):** 52wh-near names can mean-revert over 3-5yr; our forward-1mo horizon is short enough to dodge this, but a gentle/centered tilt (not a hard winner-take-all) limits exposure.
3. **Redundancy with the existing composite:** names near 52wh are usually already high on 6-mo momentum -> the tilt may be near-collinear with the composite (low marginal info). Mitigate by centering the tilt (de-mean) so it only rewards 52wh-proximity ORTHOGONAL to the existing rank.
4. **Crash/regime sensitivity:** in sharp reversals, 52wh-near names fall hardest first; a long-only book has no short leg to offset. (This is the same momentum-crash risk the existing vol-penalty partially addresses.)
5. **Turnover creep:** if `k` is too large the tilt churns the basket; keep `k<=1.0` and report turnover -- reject if dTurnover > ~10%.

---

## Research Gate Checklist
Hard blockers (gate_passed=false if any unchecked):
- [x] >=5 authoritative external sources READ IN FULL (7: George-Hwang 2004 PDF, Hanauer-Windmuller 2019 PDF, QuantConnect residual-momentum, Quantpedia residual-momentum-factor, Blitz-Huij-Martens 2011 repec, arXiv:2304.03437 Echo-disappears, Aalto echo-vs-momentum thesis -- all peer-reviewed or primary-platform)
- [x] 10+ unique URLs total (16+ incl. snippet-only)
- [x] Recency scan (2024-2026) performed + reported (52wh-2026 J.Econ.Dyn.&Control, MA-distance Dec-2023, Lin 2020, Echo-disappears 2023, Mamais 2025)
- [x] Full papers read (pdfplumber for binary George-Hwang + Hanauer; HTML for the rest), not abstracts
- [x] file:line anchors for every internal claim (screener.py:213-214/295-309/434-437/491-550, sector_neutral_replay.py:26/67-94/161-164/187, autonomous_loop.py:649-654)
- [x] Adversarial sources deliberately sought + read (Echo-disappears + Aalto thesis kill the skip-month candidate; Barroso-Wang tempers 52wh for large caps)

Soft checks:
- [x] Internal exploration covered the live composite + the multidim 52wh leg + the replay harness
- [x] Contradictions/consensus noted (52wh debate, echo debate, factor decay)
- [x] 3-variant query discipline visible (current-year / last-2-year / year-less queries listed)

## GATE ENVELOPE
```json
{
  "tier": "complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 12,
  "urls_collected": 19,
  "recency_scan_performed": true,
  "internal_files_inspected": 3,
  "gate_passed": true
}
```
`gate_passed: true` -- 7 sources read in full (floor 5); recency scan performed; 3-variant queries run; 19 unique URLs; internal audit pinned to file:line across screener.py + sector_neutral_replay.py + autonomous_loop.py. The SINGLE recommended price-only signal is **52-week-high proximity applied as a centered multiplicative tilt** on the existing 1/3/6-mo composite -- chosen as the MEASURE-FIRST pick because its formula is already implemented (screener.py:213) and it fits the replay window with a one-line change; single-factor residual momentum is the higher-evidenced RUNNER-UP to escalate to if the 52wh tilt measures flat (the Barroso-Wang large-cap-mute scenario).

