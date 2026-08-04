# Research Brief -- phase-82.5: Exit-quality tile blowup (capture ratio / edge ratio)

**Tier:** moderate (caller-specified). **Audit-class:** false.
**Researcher:** Layer-3 researcher (Workflow rail). **Started:** 2026-08-04.
**Status:** IN PROGRESS -- written incrementally (write-first discipline).

## Question set

Internal (I1-I5): aggregation site, MFE/MAE provenance, frontend double-scale,
other consumers, existing robust-aggregation convention.
External (Q1-Q4): canonical capture-ratio / MFE-MAE efficiency definition,
degenerate-case handling, robust aggregation of heavy-tailed ratios, recency scan.

## Search queries run (3-variant discipline)

| # | Variant | Query |
|---|---|---|
| 1 | year-less canonical | `maximum favorable excursion MFE MAE trade efficiency ratio definition Sweeney` |
| 2 | year-less canonical | `ratio estimator ratio of means versus mean of ratios heavy tails near-zero denominator bias` |
| 3 | last-2-year / current-year | `winsorized mean trimmed mean robust aggregation of ratios Cauchy distribution undefined mean 2025 2026` |
| 4 | last-2-year / current-year | `MFE MAE exit efficiency capture ratio trading journal metric 2025 2026 median outlier` |

---

## Internal code inventory

### I1 -- The aggregation site (RE-DERIVED 2026-08-04)

`backend/api/paper_trading.py:974-1047` = `GET /api/paper-trading/mfe-mae-scatter`.
Caller's "996-1032" is correct as a span; the precise anchors are:

| Anchor | Code | Behaviour |
|---|---|---|
| `:990-992` | `trades = bq.get_paper_trades(2000)`; `rts = pair_round_trips(trades)` | Rows come from the FIFO pairer, NOT from the stored `capture_ratio` column |
| `:999` | `mfe = float(rt.get("mfe_pct") or 0.0)` | `or 0.0` maps None AND 0.0 to 0.0 -- indistinguishable |
| `:1000` | `mae_abs = abs(float(rt.get("mae_pct") or 0.0))` | same collapse |
| `:1001` | `capture = float(rt.get("capture_ratio") or 0.0)` | **second** `or 0.0` -- a legitimately-computed 0.0 and a missing value are identical |
| `:1002-1003` | `if mae_abs > 0: edge_ratios.append(mfe / mae_abs)` | **THE SILENT DROP.** mae==0 rows never enter `edge_ratios`; they DO stay in `points`, so `n` (32) != `len(edge_ratios)` (26) |
| `:1031` | `edge_ratio = sum(edge_ratios)/len(edge_ratios) if edge_ratios else 0.0` | **arithmetic mean over the mae>0 subset only** |
| `:1032` | `avg_capture = sum(p["capture_ratio"] for p in points)/n if n else 0.0` | **arithmetic mean over ALL 32 points**, including the 8 zero-filled mfe==0 rows |

So the two tiles use DIFFERENT denominators (26 vs 32) and neither is disclosed.

**Where `capture_ratio` is actually born (two independent sites, same formula):**

1. `backend/services/paper_round_trips.py:97` -- `capture = realized_pnl_pct / mfe if mfe > 0 else 0.0`
   (this is the one the scatter endpoint consumes, recomputed per FIFO lot at `:88-97`).
2. `backend/services/paper_trader.py:591` -- `capture_ratio = realized_pnl_pct / mfe_pct if mfe_pct > 0 else 0.0`,
   persisted onto the SELL row (`:614`) and the `paper_round_trips` row (`:635`). Comment at
   `:589-590` already states the intent: *"Undefined when MFE <= 0 (never printed a gain);
   use 0.0 for that edge."* -- i.e. the code AUTHOR knew it was undefined and chose a
   sentinel that is inside the metric's own value range.

**Root cause, stated precisely:** `mfe > 0 else 0.0` is a *fabricated-value* guard --
undefined is coerced to a number that is legal for the metric (0.0 = "captured none of the
run-up"), then that number is averaged as if it were a measurement. This is the same
absence-becomes-affirmative class recorded in memory `project_fabricated_safe_80_36`.
It does NOT cause the -42.08 blowup on its own (it drags the mean toward 0); the blowup
comes from the *unbounded* rows where `mfe` is positive but ~0 (000660.KS: mfe 0.0001).
Both defects are live simultaneously: 8 rows fabricated, ~1 row unbounded.

### I3 -- Does the FRONTEND double-scale? **NO.**

`frontend/src/components/MfeMaeScatter.tsx:112-116`:
```tsx
<StatCard label="Avg capture"
  value={`${(data.summary.avg_capture_ratio * 100).toFixed(0)}%`}
  hint="realized_pnl / MFE" />
```
`capture_ratio = realized_pnl_pct / mfe_pct` -- percent divided by percent, so the units
cancel and the quantity is **dimensionless**. Multiplying a dimensionless ratio by 100 to
render it as a percent is CORRECT. -42.08 x 100 = -4208%, exactly the observed tile.
There is no second defect in the frontend; the value it is handed is already wrong.

Corroborating anchors:
- `MfeMaeScatter.tsx:111` -- `edge_ratio.toFixed(2)`, **no** x100 (edge ratio is shown as a
  bare ratio, hint `mean(MFE / |MAE|)`). Consistent, and it matches the reported 86.92.
- `MfeMaeScatter.tsx:168` -- per-point tooltip `(p.capture_ratio * 100).toFixed(0)}%` --
  same convention, also correct.
- `MfeMaeScatter.tsx:121` -- threshold rendered `(0.4 * 100) = 40%`, confirming the
  0-1-ratio-rendered-as-percent contract is intentional and consistent throughout.

**Conclusion for I3: single defect, backend-side, in the metric definition. The frontend
formatter is correct and must NOT be changed** -- if the backend starts emitting a value
already in percent, this multiply becomes a real double-scale.

---

### I2 -- Where MFE/MAE come from; can MFE be 0 or negative?

`backend/services/paper_trader.py:715-721` (inside `mark_to_market`):
```python
prev_mfe = float(pos.get("mfe_pct") or 0.0)
prev_mae = float(pos.get("mae_pct") or 0.0)
new_mfe = max(prev_mfe, pnl_pct)
new_mae = min(prev_mae, pnl_pct)
```
`pnl_pct` is the position's *unrealized* return vs cost basis (`:713`). MFE/MAE are running
max/min of that series, persisted on the position row (`:731-732`) and read at exit
(`:587-588`).

**Both are seeded from 0.0, not from the first observation.** `prev_mfe` starts at 0.0 on a
brand-new position (the `or 0.0` at `:718`), so `new_mfe = max(0.0, pnl_pct)` is
**clamped at >= 0 by construction**. Symmetrically `mae <= 0` always. Consequences:

- **`mfe == 0` is NOT a market outcome and NOT a data defect -- it is a CLAMP ARTEFACT.**
  It means "the position never closed a mark above its entry". The true MFE (best excursion)
  for such a trade is NEGATIVE (e.g. "the best it ever got was -3%"), but the clamp records
  it as 0. So the 8 rows with `mfe == 0` are *censored*, not measured. You cannot recover
  the real MFE for them from the stored column.
- Equally, `mae == 0` (6 of 32 rows) means "never closed a mark below entry" -- a trade that
  went straight up. That IS a real and desirable outcome, and it makes `edge_ratio = MFE/|MAE|`
  divide by zero for the **best** trades in the book. The current `if mae_abs > 0` filter at
  `paper_trading.py:1002` therefore **systematically deletes the strongest trades from the
  edge-ratio mean** -- a survivorship bias with the sign pointing the wrong way.
- MFE is sampled only at `mark_to_market` cadence (one point per cycle), so it is a
  *close-to-close* excursion, not an intraday high. `mfe = 0.0001` (000660.KS) is a real
  measured near-zero, and the ratio at that denominator is numerically meaningless.

### I4 -- Other consumers of these two metrics

Every consumer of `pair_round_trips` was enumerated (`grep pair_round_trips backend/`):

| Consumer | file:line | Reads capture/edge? | Affected by a definition change? |
|---|---|---|---|
| `/mfe-mae-scatter` | `backend/api/paper_trading.py:992` | **YES** (both) | **YES -- the tile under repair** |
| `/performance` -> `summarize()` | `backend/api/paper_trading.py:344-346` | **YES** -- `avg_capture_ratio` via `paper_round_trips.py:157` | **YES -- SECOND, INDEPENDENT COPY of the same mean.** This is finding **F-10** of the phase-55.3 audit (`handoff/archive/phase-55.3/55.3-synthesis-checkpoint.md:25`: *"`avg_capture_ratio=-53.7` implausible"*) -- filed 55.1, never fixed. A fix that only patches the endpoint leaves `/performance` still emitting the blown-up number. |
| **Go-Live Gate** | `backend/services/paper_go_live_gate.py:131` | **NO** -- uses only `len(round_trips)` for `trades_ge_100` | **NO.** The promotion booleans are `trades_ge_100 / psr_ge_95_sustained_30d / dsr_ge_95 / sr_gap_le_30pct / max_dd_within_tolerance` -- capture/edge appear in NONE of them. Safe, *provided the fix does not change the PAIRING* (row count). |
| `_compute_attribution` | `backend/api/paper_trading.py:389` | NO -- `realized_pnl_usd` only | NO |
| `_compute_learnings` reconciliation | `backend/api/paper_trading.py:883` | NO -- entry/exit price only | NO |
| `sovereign_api` efficiency | `backend/api/sovereign_api.py:551` | NO -- `realized_pnl_usd` only | NO |
| optimizer / promoter / BQ views / Slack bot | -- | **NO occurrences** (grep across `backend/backtest/`, `backend/meta_evolution/`, `backend/slack_bot/`, `*.sql`) | NO |

Frontend consumers: `MfeMaeScatter.tsx:34,111,114,168` and the type
`PaperRoundTripSummary.avg_capture_ratio` at `frontend/src/lib/types.ts:766` /
`frontend/src/lib/api.ts:524`. **`round_trip_summary.avg_capture_ratio` is currently
declared in the types but rendered NOWHERE** (grep of `frontend/src/app/` +
`frontend/src/components/` returns only `MfeMaeScatter.tsx`) -- so `/performance`'s copy is
API-visible but not on screen today. Fix it anyway: it is a live API field and the 55.3 F-10
finding is still open.

**Verdict on I4: no sizing, promotion, or trading decision reads these metrics. They are
pure diagnostics.** That materially lowers the risk of the fix -- but the two-copies
problem means the fix has TWO edit sites.

### I5 -- Existing repo convention for robust aggregation

There is **no** winsorize / trimmed-mean / robust-aggregation helper anywhere in the repo
(`grep -rn "winsor\|trim_mean\|trimmed"` over `backend/` returns only unrelated
string-trimming). `backend/services/perf_metrics.py` -- the documented single source of
truth for metrics (`.claude/rules/backend-services.md`: *"Never compute Sharpe, drawdown,
or alpha outside `perf_metrics.py`"*) -- contains **no** robust-location estimator. Closest
existing idioms:

| Idiom | file:line | Note |
|---|---|---|
| Hand-rolled median | `backend/services/paper_round_trips.py:153-154` (`median_holding_days`) | **In the same module as the defect** -- a median is already an accepted aggregation here |
| `np.quantile` | `backend/services/perf_metrics.py:806-807` | Bootstrap CI bounds for Sharpe |
| `np.median` | `backend/tools/monte_carlo.py:98` | `median_return` |
| `statistics.median` | `backend/agents/mcp_servers/signals_server.py:1723` | |
| Hand-rolled P75 | `backend/api/paper_trading.py:1022-1025` | Already in the endpoint being fixed, for the leakage rule |
| Zero-denominator contract | `backend/api/sovereign_api.py:566-569` | **The house precedent for exactly this problem:** *"Zero-denominator contract: return None for the ratio, not infinity"* -- returns `None`, not a fabricated 0.0 |

`sovereign_api.py:566-569` is the strongest internal precedent and it directly contradicts
the `else 0.0` at `paper_round_trips.py:97` / `paper_trader.py:591`. There is a house rule
here; the exit-quality path predates it and violates it.

---

## External sources -- READ IN FULL (7; gate floor is 5)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|---|
| E1 | https://ar5iv.labs.arxiv.org/html/0710.2024 (Franz, *Ratios: a short guide to confidence limits and proper use*) | 2026-08-04 | preprint (peer-tier) | WebFetch via ar5iv HTML chain | *"Neither the expected value nor the variance exist for this distribution"* (ratio of two normals = Cauchy). *"if we calculate the mean of independent, identically Cauchy-distributed variables we find that the mean follows the **same** Cauchy distribution as each of the individual variables."* *"the function y/x has a singularity at x=0. Therefore, if the denominator is noisy and 'too close' to zero the estimate for the ratio goes astray."* Recommends **ratio of means** `rho_hat = ybar/xbar` over the "index method" (mean of per-item ratios); Fieller / Hwang-bootstrap for CIs. |
| E2 | https://en.wikipedia.org/wiki/Ratio_estimator | 2026-08-04 | reference | WebFetch | Ratio estimator `r = ybar/xbar = (sum y_i)/(sum x_i)`. *"Ratio estimates are biased and corrections must be made"*; *"The bias is of the order O(1/n)"*, relative bias `O(n^-1/2)` under SRS; jackknife reduces to `O(n^-2)`. *"ratio variables are skewed to the right, are leptokurtic and their nonnormality is increased"* as the CV rises. |
| E3 | https://pmc.ncbi.nlm.nih.gov/articles/PMC2430201/ (Friedrich et al., *ratio of means* simulation, BMC Med Res Methodol) | 2026-08-04 | peer-reviewed | WebFetch | RoM *"was bias-free except for some scenarios with broad distributions ... bias ranged from -4 to 2%"*. Explicit degenerate warning: a denominator *"heavily skewed towards zero"* can *"result in a high proportion of exceedingly large ratios"* and be *"biased to higher values"*. Sign constraint: *"The mean values ... must both be positive or negative, since the logarithm of a negative ratio is undefined."* |
| E4 | https://www.processexcellencenetwork.com/lean-six-sigma-business-performance/articles/averaging-ratios-and-the-perils-of-aggregatio | 2026-08-04 | industry practitioner | WebFetch | The correct combination of ratios is *"the ratios to be averaged are first multiplied by their respective weights, that is, by the **denominator of the ratio** ... the total is divided by the sum of the weights."* Worked example: unweighted mean of 6/8/10% = 8%, correct weighted answer = **9.4%**. (Algebraically `sum(r_i * x_i)/sum(x_i) == sum(y_i)/sum(x_i)` -- an independent derivation of E1/E2's ratio-of-sums.) |
| E5 | https://docs.tradingmetrics.com/en/technical-analysis/trading-metrics/trade-specific-metrics/max-favorable-excursion | 2026-08-04 | vendor docs | WebFetch | `MFE% = (Highest Price - Entry Price)/Entry Price x 100` (long). *"By comparing your actual profit to the MFE, you can calculate your 'Exit Efficiency.'"* Aggregation given as `Avg MFE% = sum(MFE%)/N` -- **arithmetic mean only; no median, no ratio-of-sums, no degenerate-case rule.** Losing-trade example still carries MFE% = 6%. |
| E6 | https://www.tradingdiarypro.com/mae-mfe-explained/ | 2026-08-04 | industry practitioner | WebFetch | **The decisive convention quote:** *"if the trade immediately turns against you and the position was never in the profit zone then the MFE is zero."* Defines MAE/MFE as price extremes; presents MFE/MAE only as scatter diagnostics -- *"doesn't calculate an aggregated ratio metric."* |
| E7 | https://arxiv.org/html/2403.12110 (*Robust estimations from distribution structures: I. Mean*, 2024) | 2026-08-04 | preprint (peer-tier) **[RECENCY]** | WebFetch (arXiv native HTML) | *"the Winsorized mean typically has smaller biases compared to the trimmed mean"*; highlights *"the superiority of the median Hodges-Lehmann mean"*; for heavy tails, median-of-means *"nears the optimum of sub-Gaussian mean estimation with regards to concentration bounds when the distribution has a heavy tail"*; H-L is *"asymptotically equivalent to MoM"*. |

## Identified but snippet-only (does NOT count toward the gate)

| URL | Kind | Why not read in full |
|---|---|---|
| https://www.quantifiedstrategies.com/maximum-adverse-excursion-and-maximum-favorable-excursion/ | industry | **Bot-blocked.** WebFetch returned "Bot Verification"; `curl` + tag-strip (the `feedback_gcloud_docs_fetch` fallback) returned 72 bytes of the same interstitial. Search snippet retained: Sweeney introduced MAE/MFE in *Campaign Trading* (1996); *"exit P&L / MFE is your profit efficiency"*; *"Above 0.50 is solid, above 0.75 excellent; a well-optimized strategy typically achieves 65-80%."* |
| https://help.tradervue.com/article/3440-mfe-and-mae-calculations | vendor docs | Fetched, but the page is a 2-line stub ("Position MFE: The maximum interim profit during the trade") with no formula, no degenerate rule. Counted honestly as snippet-only. |
| https://arxiv.org/abs/1409.4896 (Formenti, *Mean of Ratios or Ratio of Means*) | preprint | ar5iv 307-redirects to the abs page; only the abstract was retrievable. Abstract finding retained: *"the Ratio of means has a lower statistical uncertainty"*; applied to a mortgage book it moved PD by 11bp. |
| https://www.tradesviz.com/blog/mfe-mae-charts/ | industry | 2025-2026 snippet: *"The MFE Capture Ratio (exit P&L / MFE) ... ratios consistently below 40% signal noise-driven exits."* |
| https://www.tradewink.com/learn/trade-journal-mfe-mae-analysis-guide | industry | 2026 snippet: breakout/trend *">0.60 when managed correctly"*; mean-reversion *"0.40-0.50 the realistic target"*; *"A capture ratio based on fewer than 30 trades ... is statistically unreliable, so median and outlier analysis should account for sample size."* |
| https://journalplus.co/metrics/maximum-favorable-excursion/ | industry | Fetch failed (redirect loop >10). |
| https://www.tradesviz.com/blog/mfe-mae-duration/ | industry | 2025-2026 MFE/MAE duration extension; no aggregation rule. |
| https://tradersync.com/mfe-and-mae-metrics/ | industry | Duplicate coverage of E5/E6. |
| https://traderssecondbrain.com/guides/mae-mfe-analysis | industry | Duplicate coverage. |
| https://trademetria.com/blog/understanding-mae-and-mfe-metrics-a-guide-for-traders/ | industry | Duplicate coverage. |
| https://journalplus.co/learn/glossary/mae/ , /learn/guides/mae-mfe-guide/ , /learn/glossary/mfe/ | industry | Same domain as the failed fetch. |
| https://www.jmp.com/en/statistics-knowledge-portal/inferential-statistics/probability-distributions/cauchy-distribution | reference | Corroborates E1 (*"mean and variance are undefined, or infinite"*; *"results from dividing one normal random variable by another"*). |
| https://www.datacamp.com/tutorial/winsorized-mean , https://www.sfu.ca/sasdoc/sashtml/insight/chap38/sect17.htm , https://rdrr.io/cran/psych/man/winsor.html | tutorial/docs | Trimmed-vs-winsorized mechanics; superseded by E7. |
| https://www.sciencedirect.com/science/article/abs/pii/S0378375808001924 | peer-reviewed | Paywalled abstract. |
| https://www.statisticshowto.com/ratio-estimator/ , https://www.researchgate.net/publication/222776446 | tutorial | Superseded by E2/E1. |

**URLs collected: 26 unique. Read in full: 7. Snippet-only: 19.**

## Recency scan (2024-2026) -- PERFORMED

Queries 3 and 4 above were scoped to 2024-2026. **Result: 2 new findings that COMPLEMENT
(do not supersede) the canonical sources.**

1. **E7 (arXiv:2403.12110, 2024)** is the only recent *methodological* advance: it ranks
   robust location estimators and finds median-of-means / median-Hodges-Lehmann optimal
   under heavy tails, and winsorized-mean less biased than trimmed-mean. This does not
   overturn Franz 2007 (E1) -- E1's point is that no location estimator rescues a Cauchy
   sample, so the fix must be at the *definition*, not the estimator. E7 tells us which
   estimator to pick **after** the domain is fixed.
2. **2025-2026 practitioner convergence on the median.** TradesViz / TradeWink (2026)
   independently reached the same operational conclusion this step needs:
   *"median and outlier analysis should account for sample size"* and *"a capture ratio
   based on fewer than 30 trades ... is statistically unreliable."* pyfinagent has 32 rows
   -- barely at that floor.
3. **No 2024-2026 source defines the degenerate cases.** Every MFE/MAE vendor page (2024,
   2025, 2026) still prescribes the plain arithmetic mean with no MFE=0 / MAE=0 rule. The
   defect in `paper_trading.py:1031-1032` is therefore a *faithful implementation of the
   industry convention*, and the industry convention is wrong. There is no newer trading
   source to defer to; the statistics literature (E1-E4) is the governing authority.

## Key findings

1. **The mean of these ratios does not merely have outliers -- it does not exist.**
   A ratio whose denominator can approach zero is Cauchy-like: *"Neither the expected value
   nor the variance exist"* and *"the mean of independent, identically Cauchy-distributed
   variables ... follows the same Cauchy distribution as each of the individual variables"*
   (E1). Adding trades will never make `avg_capture` converge. -42.08 is not a bad estimate
   of a true value; there is no true value to estimate. (E1; corroborated by JMP.)
2. **The correct aggregation of a set of ratios weights each by its own denominator**, which
   is algebraically the ratio of sums (E4's weighted-average procedure = E2's
   `r = sum(y)/sum(x)`), and the ratio of sums has *"lower statistical uncertainty"*
   (Formenti 1409.4896) with bias only `O(1/n)` (E2). The unweighted mean of ratios is the
   documented error (E4's 8% vs 9.4% worked example).
3. **But the ratio of sums has its own failure mode that applies HERE**: a denominator
   *"heavily skewed towards zero"* produces *"a high proportion of exceedingly large ratios"*
   and is *"biased to higher values"* (E3). With 8/32 rows at `mfe == 0` the capture
   denominator is exactly that -- so ratio-of-sums must be a SECONDARY readout, not the
   headline, and only over a non-degenerate domain.
4. **`mfe == 0` is the industry-standard encoding of "never in profit", not a data bug**:
   *"if the trade immediately turns against you and the position was never in the profit zone
   then the MFE is zero"* (E6). This matches `paper_trader.py:720`'s
   `new_mfe = max(prev_mfe, pnl_pct)` seeded at 0. So the domain of "capture ratio" genuinely
   excludes those trades -- there was no favorable excursion to capture.
5. **The published interpretive scale for capture is bounded in [0,1]**: <0.40 = noise-driven
   exits, >0.50 solid, 0.60+ for trend setups, 0.65-0.80 for a well-optimized strategy,
   >0.75 excellent (QuantifiedStrategies snippet; TradesViz 2026; TradeWink 2026). A tile
   reading -4208% is not a bad score on that scale -- it is off the scale entirely.
6. **The degeneracies are ASYMMETRIC and require opposite treatments.** `mae == 0` means the
   trade never traded against us -- a genuine, *desirable*, measurable property whose edge
   ratio is `+inf`. Excluding it (as `paper_trading.py:1002` does) deletes the best trades
   and biases the mean downward via survivorship. `mfe == 0` means there was no exit decision
   to grade -- the failure was the entry -- so including it (as `:1032` does, at a fabricated
   0.0) blames the exit for an entry failure. **Exclude for capture; keep for edge.**
7. **A median is well-defined over the extended reals; a mean is not.** The `+inf` rows from
   `mae == 0` can be *ranked* rather than deleted, so a median needs no exclusion at all as
   long as fewer than half the rows are degenerate (6/32 = 19%). The `if mae_abs > 0` filter
   at `:1002` exists *only* because the code chose a mean. Under the median it is unnecessary
   and harmful. (Order-statistic property; consistent with E7's robust-estimator ranking.)

## Consensus vs debate

**Consensus:** every statistical source (E1-E4) agrees the unweighted mean of per-item ratios
is the wrong estimator; all agree the problem is the denominator, not the outlier.

**Debate:** *which* replacement. E1/E2/E4/Formenti favour ratio-of-sums; E3 shows
ratio-of-sums itself degrades when the denominator piles up near zero; E7 and the 2026
practitioner sources favour a robust location estimator (median / MoM / H-L). **Resolution
for this step:** they are not in conflict once the *domain* is fixed. Fix the domain first
(define when the ratio exists), then a median over the valid domain answers "typical exit
quality" and a ratio-of-sums over the same domain answers "aggregate fraction captured".
Report both; make the headline the median because every published threshold (0.40 / 0.60 /
0.75) is a **per-trade** number, so a per-trade estimator is the like-for-like comparison.

## Pitfalls (from the literature + this repo)

- **P1.** Don't "fix" this by clipping/winsorizing at some bound. Winsorizing a Cauchy sample
  produces a finite number with no interpretation (E1: no population mean exists to
  estimate). E7's estimator ranking only applies once the sample is from a distribution with
  a mean.
- **P2.** Don't switch to "winners only". A negative capture is meaningful and important when
  MFE is large (MFE +30%, exit -10% -> capture -0.33 = "gave back the whole run-up and a
  third again"). The pathology is denominator magnitude, not numerator sign. E3's sign
  constraint applies to *log*-ratio pooling, which is not what we are doing.
- **P3.** Don't ship ratio-of-sums alone for capture: E3's near-zero-denominator warning
  applies with 8/32 rows at exactly zero.
- **P4 (repo-specific).** Returning `None` for an undefined aggregate will hit
  `MfeMaeScatter.tsx:114` -- `(null * 100).toFixed(0)` renders **`0%`**, a fabricated
  "captured nothing". Same class as auto-memory `project_fabricated_safe_80_36`. The
  frontend type at `frontend/src/lib/types.ts:766` and `frontend/src/lib/api.ts:524` must
  become `number | null` and the tile must discriminate on **presence**, rendering an
  explicit "n/a" state.
- **P5 (repo-specific).** `backend/services/paper_go_live_gate.py:131` depends on
  `len(pair_round_trips(trades))` for `trades_ge_100`. **Do not change the pairing loop**
  (`paper_round_trips.py:60-124`) while fixing the aggregation, or you silently move a
  promotion boolean.
- **P6.** Two edit sites, not one: `paper_trading.py:1031-1032` AND
  `paper_round_trips.py:157`. Patching only the first leaves `/performance` emitting the
  same blown-up `avg_capture_ratio` (open finding 55.3 F-10).

## RECOMMENDED DEFINITION (implementable)

### Per-trade `capture_ratio` -- domain-restricted, nullable

```
capture_ratio(rt) =
    None                                  if mfe_pct < MIN_MFE_PCT      # undefined
    realized_pnl_pct / mfe_pct            otherwise
```
`MIN_MFE_PCT = 1.0` (one percentage point of favorable excursion). Justification: the
published interpretive scale is a *fraction of an economically meaningful move*; below ~1pp
the "available move" sits inside the round-trip cost + noise band
(`settings.paper_transaction_cost_pct` is of this order, applied twice at
`paper_trader.py:575`), so "what fraction did you keep" is not an exit-quality statement.
This one threshold subsumes BOTH observed pathologies: it removes the 8 `mfe == 0` rows
*and* 000660.KS at `mfe = 0.0001`. **`MIN_MFE_PCT` is the one free parameter -- surface it
in the response payload so the tile can disclose it.**

Replaces `paper_round_trips.py:97` (`... if mfe > 0 else 0.0`) and
`paper_trader.py:591` (same). Persisted column becomes NULLable -- this is the
`sovereign_api.py:566-569` house rule (*"return None for the ratio, not infinity"*) applied
consistently.

### Aggregate capture (the tile)

```
defined   = [rt for rt in rts if rt.capture_ratio is not None]
n_defined = len(defined); n_undefined = len(rts) - n_defined

capture_median  = median(rt.capture_ratio for rt in defined)   if n_defined >= 1 else None
capture_agg     = sum(rt.realized_pnl_pct for rt in defined)
                / sum(rt.mfe_pct          for rt in defined)   if that sum > 0 else None
```
**Headline tile = `capture_median`** (per-trade estimator, directly comparable to the
0.40/0.60/0.75 literature thresholds). `capture_agg` (ratio-of-sums, E1/E2/E4) ships as a
secondary field labelled "aggregate capture". Both `n_defined` and `n_undefined` go in the
payload and the tile must show `n_defined` -- the current tile shows `n_points = 32` beside a
number computed from a different set.

### Per-trade `edge_ratio` -- extended-real, never dropped

```
edge_ratio(rt) =
    +inf                       if abs(mae_pct) == 0 and mfe_pct > 0   # perfect: no adverse excursion
    None                       if abs(mae_pct) == 0 and mfe_pct == 0  # no excursion either way
    mfe_pct / abs(mae_pct)     otherwise
```

### Aggregate edge (the tile)

```
ranked    = [rt for rt in rts if rt.edge_ratio is not None]      # +inf rows INCLUDED, sorted last
edge_median = median(ranked)                                      if len(ranked) >= 1 else None
edge_agg    = sum(mfe_pct for rt in rts) / sum(abs(mae_pct) for rt in rts)  if denom > 0 else None
```
**Headline tile = `edge_median`**, computed over ALL rows including the `mae == 0` ones
ranked at `+inf`. This deletes the `if mae_abs > 0` filter at `paper_trading.py:1002` and
with it the survivorship bias. `edge_median` is finite whenever `<50%` of rows are degenerate
(19% here). If `edge_median` itself lands on an `+inf` row, return `None` and disclose --
do not render `Infinity`.

### Degenerate-case table (every case enumerated)

| Case | capture | edge | Rendered |
|---|---|---|---|
| `mfe >= 1.0`, `mae < 0` | `pnl/mfe` (may be negative) | `mfe/|mae|` | both numeric |
| `mfe >= 1.0`, `mae == 0` | `pnl/mfe` | `+inf` (ranked, not dropped) | capture numeric; edge counted in median |
| `0 < mfe < 1.0` (000660.KS) | `None` -- excluded, counted in `n_undefined` | `mfe/|mae|` if `mae != 0` | capture "n/a" |
| `mfe == 0`, `mae < 0` (8 rows) | `None` -- excluded | `0.0` (a real, meaningful zero) | capture "n/a"; edge 0.0 |
| `mfe == 0`, `mae == 0` | `None` | `None` | both "n/a" |
| `n_defined == 0` | `None` | `None` | tile shows "n/a", NOT `0%` |
| all `mfe == 0` (so `sum(mfe) == 0`) | `capture_agg = None` | -- | guarded |

### What the 32-row fixture should report

The exact numbers must be **re-measured** after `MIN_MFE_PCT` is fixed -- do not copy these
forward as asserted values (they are predictions, and the 1.0pp floor may exclude rows the
caller's 2026-07-31 measurement did not):

- `avg_capture_ratio` (median): **~0.63**, in `[0, 1]`. It is exactly the caller's measured
  0.63 **iff** the excluded set is exactly the 8 `mfe == 0` rows (n_defined = 24); the 1.0pp
  floor also removes 000660.KS and any other sub-1pp row, which shifts the median by at most
  one order statistic. Re-derive.
- `n_defined` <= 24, `n_undefined` >= 8, and `n_defined + n_undefined == 32` exactly.
- `edge_ratio` (median over all 32 with the 6 `mae == 0` rows ranked at `+inf`): **>= 0.81**
  and finite -- strictly greater than the caller's 0.81, because 0.81 was measured over the
  26-row surviving subset and re-admitting 6 rows at the top of the order shifts the median
  up by 3 positions. Re-derive; do not assert 0.81.
- Neither tile may render `-4208%` or `86.92` under any row ordering.

### Invariant tests (must FAIL on mutation, per `feedback_mutation_test_guards_and_fixtures`)

- **INV1 (the actual bug).** Fixture row `{mfe_pct: 0.0001, realized_pnl_pct: -0.13}` --
  the real 000660.KS row -- must not move `avg_capture_ratio` outside `[-3, 3]`.
  **Mutation:** revert `:1032` to the mean and this test must go red.
- **INV2 (conservation, kills the silent drop).** `n_defined + n_undefined == n_points` for
  both metrics, and the edge denominator count must equal `n_points` (no filter).
  **Mutation:** restore `if mae_abs > 0` and this must go red.
- **INV3 (no fabricated zero).** A fixture where every row has `mfe_pct == 0` must return
  `avg_capture_ratio is None`, and the tile must render "n/a", not `0%`.
  **Mutation:** restore `else 0.0` and this must go red.
- **INV4 (stability).** Deleting the single most extreme row must move each headline by
  `< 0.10`. A mean fails this by ~40 points.
- **INV5 (both sites agree).** `/performance -> round_trip_summary.avg_capture_ratio` must
  equal `/mfe-mae-scatter -> summary.avg_capture_ratio` on the same trade list.
  **Mutation:** patch only `paper_trading.py` and this must go red.

## Application to pyfinagent -- change sites

| # | file:line | Change |
|---|---|---|
| 1 | `backend/services/paper_round_trips.py:97` | `capture = ... if mfe > 0 else 0.0` -> `None` when `mfe < MIN_MFE_PCT` |
| 2 | `backend/services/paper_round_trips.py:157` | `avg_capture = sum(...)/n` -> median over defined; add `n_capture_defined` / `n_capture_undefined` to the returned dict (`:159-170`) |
| 3 | `backend/services/paper_round_trips.py:141` | zero-round-trip branch: `avg_capture_ratio: 0.0` -> `None` |
| 4 | `backend/api/paper_trading.py:1001` | drop the `or 0.0`; carry `None` through |
| 5 | `backend/api/paper_trading.py:1002-1003` | delete the `if mae_abs > 0` filter; emit extended-real edge per row |
| 6 | `backend/api/paper_trading.py:1031-1032` | mean -> median (+ `_agg` ratio-of-sums secondary) |
| 7 | `backend/api/paper_trading.py:1027` | leakage rule `p["capture_ratio"] < 0.4` currently reads the fabricated `0.0`, so **all 8 `mfe == 0` rows are silently eligible to be flagged as leakers**. Must skip rows whose capture is `None`. (Second-order defect surfaced by the same root cause.) |
| 8 | `backend/services/paper_trader.py:591` | same `else 0.0` -> nullable, for the persisted column |
| 9 | `frontend/src/lib/types.ts:766`, `frontend/src/lib/api.ts:524`, `frontend/src/components/MfeMaeScatter.tsx:111,114,168` | types -> `number \| null`; discriminate on presence; **do NOT remove the `* 100`** (see I3) |
| 10 | `backend/tests/test_paper_trading_v2.py:235-243` | existing test only asserts key presence -- cannot fail on any value. Add INV1-INV5. |

**Out of scope, queue separately** (per `feedback_queue_discovered_defects_in_masterplan`):
`paper_round_trips.py:147-152` mixes units -- `profit_factor` is built from
`realized_pnl_usd` while `avg_win_pct` / `avg_loss_pct` / `expectancy_pct` are built from
`realized_pnl_pct`. This is the *other half* of open finding 55.3 F-10 (`profit_factor=0.0229`
alongside `win_rate=0.64`). **Not measured in this session** -- filed as a lead, not a
verdict.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **7** (E1-E7:
      2 preprints, 1 peer-reviewed journal, 1 reference, 1 vendor doc, 2 industry)
- [x] 10+ unique URLs total -- **26** (7 full + 19 snippet-only)
- [x] Recency scan (2024-2026) performed + reported -- 2 complementary findings + the
      explicit negative result that no trading source defines the degenerate cases
- [x] Full papers / pages read (not abstracts) -- E1 via the ar5iv chain, E7 via arXiv
      native HTML; `arXiv:1409.4896` was abstract-only and is therefore filed as
      snippet-only, NOT counted toward the gate
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module -- all 6 `pair_round_trips`
      consumers enumerated by grep, plus both `capture_ratio` birth sites, the frontend
      formatter, the go-live gate, and the existing test
- [x] Contradictions / consensus noted -- E1/E2/E4 (ratio-of-sums) vs E3 (its
      near-zero-denominator failure) vs E7 + 2026 practitioners (robust location); resolved
      by fixing the domain first
- [x] All claims cited per-claim
- [ ] **Gap disclosed:** the canonical practitioner source (QuantifiedStrategies, the
      Sweeney-anchored page) is bot-blocked to both WebFetch and curl; the Sweeney
      *Campaign Trading* (1996) primary text is offline-only. Sweeney's definition is
      carried here at second hand via E5/E6 + snippets, not from the primary source.
- [ ] **Length:** the moderate tier targets <=700 words. This brief exceeds it. The overrun
      is in evidence tables and the enumerated degenerate-case spec the caller explicitly
      requested; the analytic prose is held tight. Disclosed rather than trimmed.

---

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 19,
  "urls_collected": 26,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Single defect, backend-side. The frontend does NOT double-scale: capture_ratio is pnl_pct/mfe_pct, dimensionless, so MfeMaeScatter.tsx:114's x100 is correct and must not change. Root cause is the metric definition: a ratio with a denominator that can reach zero is Cauchy-like, so its mean does not exist (Franz arXiv:0710.2024) -- adding trades will never make -42.08 converge. mfe==0 is the industry encoding of 'never in profit' (TradingDiaryPro), and paper_trader.py:720 clamps MFE at 0 by construction, so 8/32 rows are censored, not measured. The degeneracies are asymmetric: exclude mfe~0 rows from capture (no exit decision to grade), but KEEP mae==0 rows in edge (they are the best trades; paper_trading.py:1002 deletes them, a survivorship bias). Recommend nullable per-trade ratios with MIN_MFE_PCT=1.0, a MEDIAN headline (comparable to the published 0.40/0.60/0.75 per-trade thresholds), a ratio-of-sums secondary, and disclosed n_defined. No sizing or promotion path reads these metrics; the go-live gate uses only the round-trip COUNT. Two edit sites (paper_trading.py:1031-1032 and paper_round_trips.py:157) -- the second is open finding 55.3 F-10.",
  "brief_path": "handoff/current/research_brief_82.5.md",
  "gate_passed": true
}
```

---
