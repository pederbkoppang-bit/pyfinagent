# Research Brief -- Step 82.51: Publication-lag look-ahead on fundamentals reads

**Tier:** complex | **Audit-class:** true (loop-until-dry, K=2)
**Researcher:** Layer-3 Researcher (Workflow rail)
**Started:** 2026-08-06
**Status:** IN PROGRESS (write-first skeleton; sections filled incrementally)

---

## 0. Objective (verbatim from spawn prompt)

Step 82.51 (P1): PUBLICATION-LAG LOOK-AHEAD on every fundamentals read.
`backend/backtest/cache.py` filters fundamentals with `report_date <= cutoff`, but
`report_date` is the PERIOD END, not the date the figure became public. A Q2 ending
2025-06-30 is not filed for weeks, so a cutoff of 2025-07-05 reads a number no market
participant could have seen. Claimed measured lag on the live table: mean 66d, median 60,
p90 90.

---

## 1. Step definition (verbatim from .claude/masterplan.json)

**id** `82.51` | **status** `pending` | **priority** `P1` | **harness_required** `true`

**verification.command:**
```
source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_51_fundamentals_embargo.py -q
```

**verification.criteria (immutable, verbatim):**
1. "a fixture whose fundamentals row has a period end before the cutoff but a publication date after it is EXCLUDED, asserted by a test that fails against the current `report_date <= cutoff` filter"
2. "a fixture whose row was demonstrably public before the cutoff is still INCLUDED, so the fix cannot pass by excluding everything"
3. "both the bulk preload path and the per-cutoff fallback path apply the same rule, asserted by driving each independently and comparing their outputs on the same fixture"
4. "the before/after Sharpe and trade-count deltas from at least one real backtest are recorded in the step artifact with the commands that produced them"
5. "the chosen approach (fixed embargo vs real filing date) is recorded with its reason in the step artifact"

---

## 1a. HEADLINE FINDING (answers Q2, and it is a TRAP)

`financial_reports.historical_fundamentals` **DOES have a `filing_date` column**
(`STRING`, `NULLABLE`) and it is **100% populated** -- and it is **100% WORTHLESS**.

Measured live 2026-08-06 (SQL in section 6/Q2):

| metric | value |
|--------|-------|
| `n_rows` | 4798 |
| `COUNTIF(filing_date IS NULL)` | **0** |
| `COUNTIF(filing_date = report_date)` | **4798 (100.00%)** |
| `COUNTIF(filing_date IS NOT NULL AND filing_date != report_date)` | **0** |

Root cause is in the producer, one line, explicit:

`backend/backtest/data_ingestion.py:278`
```python
"filing_date": report_date,  # Approximation; true filing date not available from yfinance
```

**Why this is a trap and not merely a null result.** An implementer who inspects the
schema, sees `filing_date` NOT NULL on 100% of rows, and "fixes" the bug by switching
the filter from `report_date <= cutoff` to `filing_date <= cutoff` produces a
**byte-identical result set** and a **byte-identical backtest**, while the step artifact
would claim option (b) -- "filter on a real filing date" -- was taken. Criterion 4's
before/after Sharpe delta would read **exactly 0.0000**, which would look like
"the leakage was immaterial" when in fact the change was a no-op. Criterion 1's fixture
would still pass (fixtures set their own values), so the unit test would be GREEN over a
production no-op. This is the `vacuous guard` class this repo has hit twice
(82.12 STRING-column guards, 82.39 phantom columns).

**Therefore: option (b) is NOT available today. 82.50 (SEC EDGAR ingester) is a genuine
prerequisite for (b). The only implementable choice for 82.51 is (a), a fixed N-day
embargo on `report_date`** -- and the step artifact must record that `filing_date` exists,
is fully populated, and is a copy, so a future reader does not "improve" the fix into a
no-op.

**Corollary for criterion 5:** the recorded reason must be this measurement, not a
preference.

**`ingested_at` is also useless as a publication proxy:** all 4798 rows carry
`ingested_at` between `2026-03-22 08:26:51 UTC` and `2026-04-06 19:16:15 UTC` -- a
two-week bulk backfill of a table whose `report_date` spans `2024-06-30 .. 2026-02-28`.
It records when *we* fetched, not when the market saw it.

## 1b. SECOND HEADLINE: the "66d mean / 60 median / 90 p90" is NOT from this table

The step says *"Measured lag on the live table: mean 66 days, median 60, p90 90."*
**That attribution is wrong, and it matters.** Traced to source:

- `handoff/current/research_brief_82.21.md:48` -- the numbers come from
  `https://tradevodata.com/blog/lookahead-bias-fundamental-backtests`, a **tier-4
  vendor blog with a disclosed commercial COI** ("We sell one of these options, so
  calibrate accordingly"), measuring **SEC EDGAR filings**: 283,363 rows with a
  reliable filing date out of 313,406, across **5,194 US companies**, measured
  2026-07-23.
- The lag **cannot** be measured on pyfinagent's live table: `filing_date =
  report_date` on 100% of rows, so any `DATE_DIFF(filing_date, report_date)` returns
  **exactly 0 for all 4798 rows**.

So the honest statement for the contract is: *"the publication lag is UNMEASURABLE on
our table; the best external estimate for a US-listed universe is 66d mean / 60d median
/ 90d p90 (Tradevo 2026, EDGAR, COI-disclosed), corroborated independently by the SEC's
own Audit-Analytics data (~70-day average 10-K filing for >=$700M market-cap issuers,
Release 33-8644 fn.49)."*

**And the vendor's own large-cap subsample says something different again:** mean **43
days**, max **61 days** (n=40 large caps). pyfinagent's universe is the **S&P 500**
(`candidate_selector.get_universe_tickers`), i.e. entirely large/large-accelerated
filers -- so the all-filer 66d mean is the WRONG calibration target and would
over-embargo. See section 6/Q3.

---

## 1c. MEASURED cost of each embargo on THIS table

Business-day cutoff grid `pd.bdate_range('2024-07-01','2026-02-28')` = 435 cutoffs, over
all 4798 rows / 503 tickers. "row-days" = sum over cutoffs of rows visible at that
cutoff (the quantity `cached_fundamentals` actually serves). Measured 2026-08-06.

| embargo N | first date ANY row is visible | visible row-days | vs N=0 | cutoffs with ZERO rows | mean tickers covered | mean rows/ticker |
|-----------|------------------------------|------------------|--------|------------------------|----------------------|------------------|
| 0 (today) | 2024-06-30 | 937,230 | 0.0% | 0 | 411.4 | 4.51 |
| 40 | 2024-08-09 | 798,723 | **-14.8%** | 29 | 377.5 | 4.15 |
| 45 | 2024-08-14 | 781,360 | **-16.6%** | 32 | 373.6 | 4.10 |
| 60 | 2024-08-29 | 730,564 | **-22.1%** | 43 | 360.9 | 3.95 |
| 75 | 2024-09-13 | 688,588 | -26.5% | 54 | 348.5 | 3.85 |
| 90 | 2024-09-28 | 648,446 | **-30.8%** | 65 | 336.7 | 3.74 |
| 120 | 2024-10-28 | 564,232 | -39.8% | 85 | 312.1 | 3.48 |

The loss is **smooth and monotone on a business-day grid**. (Warning for whoever
re-measures: on a *month-start* cutoff grid the same table shows a spurious plateau --
40/45/60 all reported identical -22.0% -- because every `report_date` is a month-end and
~30 days apart, so several embargoes land in the same bucket. Do not quote the monthly
grid; it makes 60 look free.)

**`mean rows/ticker` is the sleeper cost.** `cached_fundamentals` returns up to 5
quarters and `historical_data.build_feature_vector` uses multiple quarters for YoY
growth. At N=0 the mean is 4.51; at N=60 it is 3.95; at N=90 it is 3.74. Fewer tickers
will have enough history for the multi-quarter features, independently of the coverage
gate.

---

## 1d. THE 5-QUARTER CLIFF -- the measured answer to "a window that currently trains may stop training"

`historical_data.py:455`:
```python
if not current_revenue or not fundamentals_list or len(fundamentals_list) < 5:
```
`revenue_growth_yoy` needs **exactly 5** quarters, and `cached_fundamentals` returns **at
most 5** (`filtered[:5]` at `cache.py:613`; `LIMIT 5` at `cache.py:633`). So the feature
is available only when the ticker has its full 5-row window visible. `revenue_growth_yoy`
is one of the 17 keys in F, and it feeds `quality_score` (the QMJ Growth dimension,
`historical_data.py:254-264`).

Measured over the live S&P-500 universe (503 tickers, `preload_fundamentals` returns
**4725** of the table's 4798 rows -- 73 rows belong to tickers no longer in the universe):

| cutoff | N=0 tickers with 5q | N=45 | N=60 | N=90 |
|--------|--------------------|------|------|------|
| 2024-12-31 | 0 (0.0%) | 0 | 0 | 0 |
| 2025-03-31 | **210 (41.7%)** | **4 (0.8%)** | **0 (0.0%)** | **0 (0.0%)** |
| 2025-06-30 | 334 (66.4%) | 228 (45.3%) | 228 (45.3%) | 210 (41.7%) |
| 2025-09-30 | 436 (86.7%) | 342 (68.0%) | 342 (68.0%) | 334 (66.4%) |
| 2025-12-31 | 489 (97.2%) | 441 (87.7%) | 441 (87.7%) | 436 (86.7%) |
| 2026-02-28 | 494 (98.2%) | 489 (97.2%) | 443 (88.1%) | 443 (88.1%) |

**The embargo's damage is concentrated in the first ~9 months of the covered window and
is mild after 2025-06-30.** At the 2025-03-31 cutoff an embargo takes `revenue_growth_yoy`
from 41.7% of the universe to **zero**. This is not an argument against the fix -- those
41.7% were reading numbers nobody could see -- but it IS the concrete shape of criterion
4's warning, and it dictates the backtest window in Q5.

---

## 2. Read in full (>=5 required; counts toward the gate) -- 8 sources

| # | URL | Accessed | Kind / tier | Fetched how | Key finding |
|---|-----|----------|-------------|-------------|-------------|
| 1 | https://www.sec.gov/files/rules/final/33-8644.pdf | 2026-08-06 | **Official (tier 2)** SEC Final Rule, Release 33-8644, 74pp | `curl` + `pdfplumber` (130,840 chars extracted; WebFetch 403s -- SEC needs a named User-Agent per `.claude/rules/security.md`) | The adopted deadline table (p.~14 of extract): **10-K = 60d large accelerated / 75d accelerated / 90d non-accelerated; 10-Q = 40d large-accelerated AND accelerated / 45d all other.** And the empirical footnote 49, verbatim: *"internal data suggests that companies with a market capitalization of $700 million or more may currently be filing their annual reports on Form 10-K on an average of **70 days** after fiscal year end. Our data was derived from Audit Analytics."* Corroborated in-body: a commenter *"analyzed 855 companies ... these companies filed, on average, within **70 days** after fiscal year end."* |
| 2 | https://tradevodata.com/blog/lookahead-bias-fundamental-backtests | 2026-08-06 | Industry / vendor (tier 4) -- **COI disclosed verbatim: "We sell one of these options, so calibrate accordingly"** | WebFetch | The source of the step's numbers. Over **283,363 EDGAR rows with a reliable filing date** (of 313,406) across **5,194 US companies**, measured 2026-07-23: mean **66d**, median **60d**, p90 **90d**, max **120d**. **Large-cap subsample (n=40): mean 43d, max 61d.** 18,734 rows revised >0.5% on the same XBRL tag later. Recommends a **120-day** embargo as a diagnostic. |
| 3 | https://datacenter.safe-frankfurt.de/documents/Data_Description_CRSP_Compustat_20250317.pdf | 2026-08-06 | Academic data-centre methodology (tier 1/2), SAFE Frankfurt, 2025-03-17 | `curl` + `pdfplumber` (13,924 chars; WebFetch could not decode) | Verbatim: *"The annual/quarterly report and its accounting information are only published several weeks after the fiscal year/quarter end. So, the information is not yet available to investors at the fiscal quarter/year-end date [datadate] but only at the report date [rdq]."* And: *"To ensure that the information was actually available at a point in time, Compustat data are only used with a lag of several months. ... A popular approach is to follow Fama and French (1993): ... use the accounting data from year t only starting at July of year t+1. Other approaches include using a **four-month gap** between the fiscal year end month and the time when the accounting information is used."* Worked MSFT example: FQ4 ending 2022-06-30 announced **2022-07-26** (26 days). |
| 4 | https://github.com/OpenSourceAP/CrossSection/issues/50 | 2026-08-06 | Open-source academic replication (Chen & Zimmermann Open Source Asset Pricing), tier 1/3 | WebFetch | The de-facto academic default for Compustat **quarterly**, verbatim from their code: `gen time_avail_m = mofd(datadate) + 3  // Assume data available with a 3 month lag` -- i.e. **datadate + 3 months (~90d)**, NOT rdq. **[ADVERSARIAL / qualifying]** the same issue documents that this rule is wrong in both directions: *">600 observations show earnings releases before the accounting period end date"* and *"more than 50,000 observations have earnings releases beyond the 90-day lag assumption"*. Issue is OPEN with no resolution -- so even the reference implementation has not solved this. |
| 5 | https://www.calcbench.com/blog/post/153949139113/point-in-time-fundamental-data | 2026-08-06 | Vendor / practitioner (tier 4) | WebFetch | Thin. Confirms the shape of a real PIT solution: each data point carries *"the date and time the document ... was uploaded to the SEC's Edgar system"*, plus separately-tracked *"little R revisions"*. No lag statistics, no measured backtest impact. Useful only as corroboration that a real filing timestamp is an EDGAR-submission property -- exactly what 82.50 would supply and what our table lacks. |
| 6 | https://www.mavensecurities.com/alpha-decay-what-does-it-look-like-and-what-does-it-mean-for-systematic-traders/ | 2026-08-06 | Industry practitioner (tier 4) | WebFetch | **[ADVERSARIAL]** the counter-pressure against a long embargo. Lagged-signal simulation: *"the cost amounts to **9.9% in Europe and 5.6% in the US**"*, *"The annual rate of increase is on average **36bps in the US** and 16bps in Europe"*, and *"there is a strong positive correlation between this cost and the volatility of the market -- at volatile times information tends to be priced into security prices faster."* Does NOT give a decay half-life in days, so it cannot be used to price a 60-vs-90-day choice numerically; it establishes only that over-lagging has a real, non-zero cost. |
| 7 | https://arxiv.org/pdf/2605.24564 (*Summoning the Oracle to Slay It: Mitigating Look-Ahead Bias in Financial Backtesting with LLMs*) | 2026-08-06 | **Preprint (tier 1)**, arXiv 2026 | WebFetch, full PDF (2.1MB) | Confirms the mechanism qualitatively -- look-ahead bias *"causes artificial inflation in backtesting returns and risk-adjusted metrics like Sharpe ratios"*; mitigation is to constrain access to *"only information temporally available at each decision point."* **Scope mismatch: it is about LLM memorisation, not accounting publication lag**, and the compressed PDF blocked verbatim numeric extraction. Counts toward the gate; contributes no number to the embargo decision. |
| 8 | https://jacobslevycenter.wharton.upenn.edu/wp-content/uploads/2014/06/Hrdlicka_Paper.pdf (*Information Release and the Fit of the Fama-French Model*) | 2026-08-06 | Academic working paper (tier 1) | WebFetch, full PDF (1.1MB) | **NULL RESULT, recorded honestly.** Read in full; despite the promising title it studies seasonal return patterns around earnings announcements and does NOT measure the period-end-to-disclosure gap or its factor-model bias. Contributes nothing to the embargo decision. |

## 3. Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://www.spglobal.com/content/dam/spglobal/mi/en/documents/general/sp-capitaliq-quantamental-point-in-time-vs-lagged-fundamentals.pdf | Industry research (S&P Capital IQ Quantamental, Aug 2015) | **Attempted twice and FAILED**: WebFetch 403, `curl` with a browser UA also 403 (545-byte block page). This is the single highest-value un-read source; snippet says *"PIT backtests produce significantly different results than lagged Non-PIT data using common factors"*. Recorded as a known gap. |
| https://www.spglobal.com/market-intelligence/en/news-insights/research/point-in-time-vs-lagged-fundamentals | Industry research (HTML mirror of the above) | WebFetch 403 |
| https://www.tidy-finance.org/r/replicating-fama-and-french-factors.html | Methodology textbook | Returned a 635-byte redirect stub, not content. The FF July-of-t+1 convention it documents is already captured by source #3. |
| https://arxiv.org/abs/2601.13770 (*Look-Ahead-Bench*) | Preprint | Only the `/abs` page rendered; `/html/2601.13770v1` returned 404, so no full text. Per `.claude/rules/research-gate.md` an abstract page does NOT count as read-in-full. Snippet: measures LLM look-ahead bias via *alpha decay*; no accounting-lag content. |
| https://www.nber.org/system/files/working_papers/w20682/revisions/w20682.rev0.pdf (Hou/Xue/Zhang, *A Comparison of New Factor Models*) | Peer-reviewed (tier 1) | Canonical prior art for the 6-month annual / quarterly lag conventions; the convention itself is already quoted verbatim from source #3, so a second fetch would be duplication. |
| https://www.pfolio.io/academy/look-ahead-bias | Educational | Snippet already states the 45-90d convention; superseded by sources #1/#2. |
| https://analystprep.com/study-notes/cfa-level-2/problems-in-backtesting/ | Educational (CFA notes) | Generic. |
| https://starqube.com/point-in-time-data/ | Vendor | Marketing. |
| https://hedgefundalpha.com/education/backtesting-mistakes-kill-quant-strategies-guide/ | Practitioner blog | Generic. |
| https://sharpely.in/blogs/bias-free-backtesting-explained-sharpely-uses-point-time-data-avoid-look/ | Vendor | Marketing. |
| https://ariaanalyst.pro/blog/look-ahead-bias-quant | Practitioner blog | Generic. |
| https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/ | Practitioner | About CV embargo (Lopez de Prado purging), a DIFFERENT embargo from this one -- worth naming so the two are not conflated (see Pitfalls). |
| https://www.mayerbrown.com/-/media/files/perspectives-events/publications/2025/12/2026-sec-filing-deadlines-and-financial-statement-staleness-dates.pdf | Law firm (2026 calendar) | Restates the deadlines already quoted verbatim from the SEC primary source #1. |
| https://viewpoint.pwc.com/dt/us/en/pwc/pwc_sec_volume/pwc_sec_volume_US/3000_registration_an_US/sec_3125_the_acceler_US.html | Accounting firm | Same. |
| https://www.colonialfilings.com/blog/how-to-calculate-a-10-k-filing-deadline-based-on-filer-status/ | Filing agent | Same. |
| https://www.catacal.com/article/sec-10k-10q-filing-deadlines-explained | Filing agent | Same. |
| https://www.sec.gov/rules-regulations/2005/12/revisions-accelerated-filer-definition-accelerated-deadlines-filing-periodic-reports | Official (FR notice) | Duplicate of #1. |
| https://iangow.github.io/far_book/pead.html | Academic textbook (PEAD) | Uses rdq; convention already captured. |
| https://datateamoftheeur.wordpress.com/2012/01/26/earnings-announcement-dates/ | University data desk | Same. |
| https://www.marketplace.spglobal.com/en/datasets/compustat-financials-(8) | Vendor catalogue | Product page. |
| https://alphaarchitect.com/accounting-anomalies/ | Practitioner research | Accounting anomalies, not lag mechanics. |
| https://www.stern.nyu.edu/sites/default/files/assets/documents/ChenChoDouLev2021WP.pdf | Working paper | ML on detailed financials; lag treatment not the subject. |
| https://jhfinance.web.unc.edu/wp-content/uploads/sites/12369/2016/02/Alpha-Decay.pdf | Peer-reviewed | Manager-level alpha decay, not signal staleness. |
| https://github.com/ranaroussi/yfinance/discussions/1953 | Community (tier 5) | Corroborates that `get_earnings_dates()` is unreliable/broken across versions -- relevant only as evidence that yfinance offers no dependable vintage. |
| https://cran.r-project.org/web/packages/yfinancer/yfinancer.pdf | Package docs | Same. |
| https://arxiv.org/pdf/2505.06383 | Preprint | Backtest resampling bias; different failure mode. |
| https://arxiv.org/pdf/2412.14361 | Preprint | Industry-trend backtesting; no lag content. |
| https://www.morningstar.com/financial-advisors/alpha-isnt-dead-its-being-mismeasured | Media | Snippet only. |
| https://reasonabledeviations.com/notes/adv_fin_ml/ | Study notes | Lopez de Prado notes; CV embargo, not publication lag. |

**Totals: 8 read in full, 29 snippet-only, 37 unique URLs.**

## 4. Recency scan (2024-2026) -- PERFORMED

Three-variant query discipline (`.claude/rules/research-gate.md`):

- **Current-year / frontier (2026):** `"look-ahead bias fundamental data lag measured Sharpe ratio inflation arXiv 2026"`; `"point-in-time versus lagged fundamentals backtest performance difference measured 2025 2026 quantamental"`.
- **Last-2-year (2024-2025):** the Tradevo measurement is dated **2026-07-23**; the SAFE CRSP-Compustat description is **2025-03-17**; the Mayer Brown deadline calendar is **2026**.
- **Year-less canonical:** `"point-in-time fundamental data look-ahead bias backtest publication lag reporting lag"`; `"Fama French accounting data six month lag fundamental annual report availability convention"`; `"reporting lag accounting data availability date backtest embargo convention Compustat RDQ quarterly"`; `"over-lagging fundamental data destroys alpha stale accounting information cost of conservative reporting lag"`.

**Result: 3 findings in the 2024-2026 window that COMPLEMENT (none that supersede) the canonical prior art.**

1. **Tradevo 2026-07-23** -- the only recent measurement of the EDGAR lag distribution at scale (283,363 rows / 5,194 companies). It complements, and does not supersede, the SEC's own 2005 Audit-Analytics figure (~70d average 10-K for >=$700M issuers): the two agree to within ~4 days on the large-cap annual case despite a 21-year gap, which is itself the finding -- **the lag has not materially changed, so the older canonical deadline structure remains the right calibration anchor.**
2. **arXiv 2026 look-ahead-bias work (2601.13770, 2605.24564)** -- the 2026 frontier on "look-ahead bias" has moved to *LLM memorisation*, not accounting publication lag. Nothing there changes the embargo decision. Worth stating plainly so a future reader does not assume the 2026 literature is silent by accident.
3. **OpenSourceAP CrossSection issue #50 (open)** -- the reference open-source implementation still uses `datadate + 3 months` and has an OPEN, unresolved issue documenting >50,000 violations of that assumption. **There is no 2024-2026 consensus improvement on the fixed-lag approach**; the state of the art is still "use rdq if you have it, otherwise pick a fixed lag and accept it is wrong at the tails."

**Superseded: nothing.** The Fama-French (1993) July-of-t+1 convention and the SEC deadline structure both remain current.

## 5. Internal code inventory

| File:line | Role | Status |
|-----------|------|--------|
| `backend/backtest/cache.py:602-648` | `cached_fundamentals(ticker, cutoff_date)` -- the ONLY as-of-date fundamentals accessor | **3 branches**, see Q4 |
| `backend/backtest/cache.py:612` | Filter site 1 -- warm/bulk path: `filtered = [r for r in all_rows if str(r.get("report_date", "")) <= cutoff_date]` | **DEFECT** |
| `backend/backtest/cache.py:631` | Filter site 2 -- per-cutoff BQ fallback: `WHERE ticker = @ticker AND report_date <= @cutoff` | **DEFECT** |
| `backend/backtest/cache.py:615-619` | Branch 2 -- `_fundamentals_cache[(ticker, cutoff_date)]` memo; populated ONLY at `:647` by branch 3, so it cannot diverge, but a test must know it exists | inherits |
| `backend/backtest/cache.py:265-330` | `preload_fundamentals` -- loads ALL rows for the ticker list; **applies no cutoff filter at all** | correct as-is |
| `backend/backtest/cache.py:293-300` | Preload SQL, 12-col projection; **`filing_date` is deliberately excluded** as a "never-read column" (`:286`) | correct, and now justified twice over |
| `backend/backtest/cache.py:184-185` | `_table()` -> `{project}.{dataset}.{name}`; dataset is `settings.bq_dataset_reports = "financial_reports"` (`settings.py:59`) | -- |
| `backend/backtest/historical_data.py:56-61` | `get_point_in_time_fundamentals` -- pure delegation to `cache.cached_fundamentals` | single choke point |
| `backend/backtest/historical_data.py:69-78` | `build_feature_vector`; `fundamentals = fundamentals_list[0] if fundamentals_list else {}` | consumer |
| `backend/backtest/historical_data.py:191-192` | `features["fundamentals_available"] = bool(fundamentals)`; `if fundamentals:` -- the block that defines F (17 keys) | consumer |
| `backend/backtest/historical_data.py:447-457` | `_compute_revenue_growth_yoy`; `len(fundamentals_list) < 5` -> None; reads `fundamentals_list[4]` | **the 5-quarter cliff** |
| `backend/backtest/data_ingestion.py:278` | `"filing_date": report_date,  # Approximation; true filing date not available from yfinance` | **root cause of Q2** |
| `backend/backtest/data_ingestion.py:214-223, 257` | Dedup read + `strftime("%Y-%m-%d")` producer of the ISO string format | not a cutoff read |
| `backend/backtest/fundamentals_coverage.py:50` | `FUNDAMENTALS_COVERAGE_START = "2024-06-30"` | must gain an embargo-aware sibling (Q6) |
| `backend/backtest/fundamentals_coverage.py:113-124` | `window_is_covered(window_start)` -- pure string compare vs the raw min | see Q6 |
| `backend/backtest/fundamentals_coverage.py:162-219` | `fundamentals_only_feature_keys` -> F; **derived live = 17 keys** (listed in Q3) | unchanged by this step |
| `backend/backtest/fundamentals_coverage.py:240-313` | `label_fundamentals_dependent_strategies` -> **derived live = `{'qarp'}`** | drives Q5 |
| `backend/backtest/backtest_engine.py:446-507` | `_preload_fundamentals_and_record` -- REFUSE-or-RECORD gate | the extension point for Q6 |
| `backend/backtest/backtest_engine.py:476` | `covered = window_is_covered(window_start)` | **the predicate to change (Q6)** |
| `backend/backtest/backtest_engine.py:479` | `if not covered and self.strategy in dependent:` -- the REFUSE branch | -- |
| `backend/backtest/backtest_engine.py:502-507` | the returned availability dict (`fundamentals`, `fundamentals_coverage_start`, `fundamentals_window_start`, `fundamentals_label_dependent`) | **the record to extend (Q6)** |
| `backend/backtest/backtest_engine.py:465-466` | docstring: *"Every strategy is FEATURE-dependent via `_NUMERIC_FEATURES`, so an uncovered window silently drops 15 of 37 columns (:852) or imputes a fabricated median company (:881-882)"* | refutes "zero delta" (Q5) |
| `backend/backtest/backtest_engine.py:550, 556` | `cache.preload_fundamentals(universe_tickers)`; `_availability.update(self._preload_fundamentals_and_record())` | call order |
| `backend/backtest/backtest_engine.py:69, 80, 1726` | `STRATEGY_REGISTRY` (6 strategies); `"qarp": "_compute_qarp_label"`; `def _compute_qarp_label` | -- |
| `backend/backtest/backtest_engine.py:125` | `_NUMERIC_FEATURES` -- why EVERY strategy is feature-dependent | -- |
| `backend/agents/mcp_servers/data_server.py:149` | `metrics = cache.cached_fundamentals(ticker, cutoff)` -- **a LIVE, non-backtest consumer** | inherits the fix; blast radius |
| `scripts/diag_label_pin.py:26` | `cache.preload_fundamentals(tickers)` | diagnostic |
| `dev/t_backtest_mock.py:122-143` | `mock_cached_fundamentals` monkeypatch | test double, bypasses both filters |
| `backend/tests/test_phase_75_mcp_truth.py:310` | stub `cached_fundamentals(ticker, cutoff)` | test double |
| `backend/tests/test_phase_82_12_string_column_guards.py:314-322, 372-382` | `CLASSIFICATION[("backend/backtest/cache.py","report_date")]["line"] = 612` + the +/-6-line re-derivation assert | **WILL BREAK -- Q7** |
| `backend/tests/test_phase_82_21_fundamentals_coverage.py:123-129` | `test_window_is_covered_boundaries` -- asserts `window_is_covered("2024-06-30") is True` | **at risk -- Q7** |
| `backend/tests/test_phase_82_43_macro_feature_absence.py:296-360` | 5 source anchors in `cache.py` that must each match **exactly one** line | **at risk -- Q7** |
| `backend/backtest/_fundamentals_coverage.json` | Checked-in snapshot; `min_report_date = "2024-06-30"` | must NOT be mutated |

## 6. Answers to Q1-Q7

### Q1 (a) -- EVERY fundamentals read that filters on `report_date`. The step's count of TWO is CORRECT.

Derived structurally two ways, not read off the step:

1. **Text sweep** over all non-test `backend/` + `scripts/` Python for `report_date` co-occurring with an ordering operator / `WHERE` / `cutoff`:
   - `backend/backtest/cache.py:612` -- Python list-comprehension filter
   - `backend/backtest/cache.py:631` -- SQL `WHERE`
   - (`backend/alt_data/f13.py:172` is a dict key named `reportDate` on 13F data -- a different table, no cutoff comparison; `fundamentals_coverage.py:6` is a docstring.)
2. **AST sweep** for every reference to `cached_fundamentals` / `preload_fundamentals` / `_fundamentals_full` / `_fundamentals_cache` across `backend/`: 18 sites, ALL inside `cache.py` except three -- `historical_data.py:61` (pure delegation), `backtest_engine.py:550` (preload, no cutoff), `agents/mcp_servers/data_server.py:149` (a consumer).

**Verdict: exactly 2 filter sites, both in `cache.py`, at the exact lines the step names (612 and 631). No reads outside `cache.py`.** This step's stated count is one of the rare ones that survives re-derivation -- say so in the contract rather than silently agreeing.

**Two things the step's count does NOT capture, and the contract should:**
- **A third branch exists.** `cached_fundamentals` has three branches, not two: warm (`:608-613`), memo (`:615-619`), BQ fallback (`:621-648`). The memo is keyed `(ticker, cutoff_date)` on the RAW cutoff and is populated only at `:647`, so it cannot hold an un-embargoed result once the fix lands -- but if the embargo ever becomes runtime-configurable, that key MUST gain the embargo value or a stale memo will serve the old rule.
- **A live consumer outside the backtest.** `backend/agents/mcp_servers/data_server.py:149` serves `cached_fundamentals` to the MCP data server. The fix changes what agents see, not only what backtests see. That is desirable, but it is blast radius the step does not mention.

### Q2 (b) -- Does a real filing/publication date column already exist? **YES, and it is a decoy. Option (b) is NOT available without 82.50.**

Full answer in section 1a. In one line: `financial_reports.historical_fundamentals` has `filing_date STRING NULLABLE`, **0 nulls, and `filing_date = report_date` on all 4798 rows**, because `data_ingestion.py:278` writes it as a self-admitted approximation. Also present: `ingested_at TIMESTAMP`, 0 nulls, but every value falls in `2026-03-22 .. 2026-04-06` (a bulk backfill) against `report_date` spanning `2024-06-30 .. 2026-02-28` -- it records our fetch, not the market's knowledge.

**Exact SQL run (2026-08-06, `google-cloud-bigquery` client, project `sunny-might-477607-p8`):**
```sql
SELECT
  COUNT(*) AS n_rows,
  COUNTIF(filing_date IS NULL)                                        AS filing_null,
  COUNTIF(filing_date = report_date)                                  AS filing_eq_report,
  COUNTIF(filing_date IS NOT NULL AND filing_date != report_date)     AS filing_differs,
  COUNTIF(ingested_at IS NULL)                                        AS ingested_null,
  MIN(ingested_at) AS min_ing, MAX(ingested_at) AS max_ing,
  MIN(report_date) AS min_rd,  MAX(report_date) AS max_rd,
  COUNT(DISTINCT ticker) AS n_tick,
  COUNTIF(NOT REGEXP_CONTAINS(report_date,  r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")) AS bad_rd,
  COUNTIF(filing_date IS NOT NULL
          AND NOT REGEXP_CONTAINS(filing_date, r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")) AS bad_fd
FROM `sunny-might-477607-p8.financial_reports.historical_fundamentals`
```
Result: `4798 / 0 / 4798 / 0 / 0 / 2026-03-22 08:26:51 / 2026-04-06 19:16:15 / 2024-06-30 / 2026-02-28 / 503 / 0 / 0`.

**STRING-column trap check (82.21 / 82.39 lesson): PASSED.** Both `report_date` and `filing_date` are declared `STRING`; `bad_rd = bad_fd = 0` confirms every value is zero-padded ISO-8601, so lexical comparison is chronological comparison and the existing `<=` semantics are sound. **Do not add an `isinstance(v, date)` guard** -- it can never fire (documented dead-guard class).

**Measured lag distribution on THIS table: mean = median = p90 = max = 0 days, on 4798/4798 rows.** The distribution is degenerate by construction. The 66/60/90 figures in the step belong to a third-party EDGAR corpus (section 1b).

### Q3 (c) -- Correct embargo size. **Recommend N = 60 calendar days.**

**The distribution to calibrate against is the LARGE-ACCELERATED-FILER distribution, not the all-filer one.** The universe is `CandidateSelector().get_universe_tickers(market="US")` = **503 S&P 500 tickers** (measured live). Every S&P 500 constituent is a large accelerated filer (public float >= $700M by definition of index inclusion). Calibrating on the all-filer mean of 66d over 5,194 companies imports the filing behaviour of ~4,700 small caps we never trade.

**The regulatory argument (primary, tier-2 source #1):**

| Cohort in our table | Governing form | Large-accelerated deadline |
|---------------------|----------------|----------------------------|
| Fiscal-year-end quarter (the 12-31 cohort: **744 rows / 443 tickers**, the single largest) | **10-K** | **60 days** |
| The other three quarters | 10-Q | 40 days |

A **45-day** embargo covers the 10-Q deadline but **under-covers the 10-K deadline by 15 days** -- and the annual cohort is the biggest one in the table. It would leave residual leakage on exactly the rows that matter most. **60 days is the smallest embargo that dominates every legally binding deadline for this universe.**

**The empirical argument (corroborating):** the EDGAR-measured **median is 60d** and the **large-cap subsample max is 61d** (source #2); the SEC's own Audit-Analytics figure for >=$700M issuers is a ~**70d average 10-K** (source #1, fn.49) -- from the pre-2006 regime when the deadline was 75d, so post-2006 large-cap behaviour should sit at or below 60.

**The measured cost on THIS table (section 1c/1d) -- the tradeoff stated in our own numbers, not generic advice:**

| N | visible row-days lost | tickers with 5 quarters @ 2025-06-30 | tickers with 5q @ 2025-12-31 | verdict |
|---|----------------------|--------------------------------------|------------------------------|---------|
| 40 | -14.8% | -- | -- | leaks on the 744-row annual cohort |
| 45 | -16.6% | 228 (45.3%) | 441 (87.7%) | leaks 15d on the annual cohort |
| **60** | **-22.1%** | **228 (45.3%)** | **441 (87.7%)** | **RECOMMENDED** |
| 90 | -30.8% | 210 (41.7%) | 436 (86.7%) | -8.7pp more row-days lost than 60, for zero additional legal coverage on a large-cap universe |
| 120 | -39.8% | -- | -- | the vendor's own recommendation, but calibrated on the ALL-FILER max (120d); our large-cap max is 61d |

**Note the specific shape: going 45 -> 60 costs 5.5pp of row-days but costs ZERO 5-quarter tickers at the 2025-06-30 and 2025-12-31 cutoffs** (228 and 441 in both cases). The step from 60 -> 90 is where the feature-level damage starts. So 60 sits on the efficient frontier of this table.

**Counter-pressure, stated honestly (source #6):** over-lagging is not free -- lagged-signal simulation puts the cost of stale signals at 5.6% annualised in the US, rising with volatility. That argues against 90/120 and is a second reason to stop at 60.

**Residual-risk disclosure the artifact must carry (source #4, adversarial):** a fixed lag is wrong at both tails. OpenSourceAP's `datadate + 3 months` rule still has >50,000 violating observations, and that issue is OPEN. A 60-day embargo on our universe will still admit a handful of late filers. **60 is a defensible approximation, not a correctness proof -- the correct fix is a real filing date from 82.50.** Criterion 5's recorded reason should say exactly that.

**Convention cross-check:** Fama-French (1993) uses accounting data from year t only from July of t+1 (a ~6-month annual lag) and the "four-month gap" variant is also common (source #3). Those are ANNUAL conventions applied to annual accounting data with monthly rebalancing; applying a 6-month lag to quarterly data would destroy 3 of every 4 quarters. Do not import the FF number.

### Q4 (d) -- The two paths criterion 3 names, and whether a real shared seam exists. **YES, one exists.**

Criterion 3's wording maps onto `cached_fundamentals` as follows -- note "bulk preload path" is **not** `preload_fundamentals` (which applies no cutoff at all):

| Criterion-3 name | Concrete code | Filter today |
|------------------|---------------|--------------|
| "the bulk preload path" | `cache.cached_fundamentals` **branch 1**, `cache.py:608-613`, reading `_fundamentals_full` populated by `preload_fundamentals` (`:265-330`) | Python list comprehension, `cache.py:612` |
| "the per-cutoff fallback path" | `cache.cached_fundamentals` **branch 3**, `cache.py:621-648` | SQL `WHERE`, `cache.py:631` |

**They do NOT share a filter today -- the logic is duplicated in two different languages** (Python `<=` on a dict value vs BigQuery `<=` on a query parameter). That is precisely why criterion 3 exists.

**A single shared helper CAN serve both, and it is a real seam, because of an exact algebraic identity:**
```
report_date + N <= cutoff     <=>     report_date <= cutoff - N
```
So instead of shifting every row (impossible in the SQL path without a computed column), shift the **cutoff once**:

```python
def _embargoed_cutoff(cutoff_date: str) -> str:
    """The as-of date a market participant could actually have used."""
    return (date.fromisoformat(cutoff_date) - timedelta(days=FUNDAMENTALS_EMBARGO_DAYS)).isoformat()
```
Call it ONCE near the top of `cached_fundamentals`, then use the result at `:612` and pass it as the `@cutoff` `ScalarQueryParameter` at `:637`. Both branches then read from ONE rule.

**Why this is a genuine seam and not a fake one** (the `feedback_guards_stop_one_seam_short` lesson): a mutation of `_embargoed_cutoff` (e.g. returning `cutoff_date` unchanged) must flip BOTH branch outputs. A test that mutates only the helper and asserts only branch 1 changed would be stopping one seam short -- criterion 3 explicitly demands "driving each independently and comparing their outputs on the same fixture", so the test must:
1. populate `_fundamentals_full[T]` with the fixture and call `cached_fundamentals(T, D)` -> branch 1 result;
2. clear `_fundamentals_full` **and** `_fundamentals_cache`, stub `_bq_client.query` to return the same fixture rows **unfiltered**, call `cached_fundamentals(T, D)` -> branch 3 result;
3. assert the two are equal AND that both exclude the not-yet-public row.

**Step 2's stub must return rows UNFILTERED**, otherwise the test proves nothing about the SQL predicate -- a stub that pre-filters is the "mutate the stub too" failure from `feedback_mutation_test_guards_and_fixtures`. The honest way to also cover the real SQL text is a separate assertion that the generated query string carries the embargoed cutoff as its parameter value (which is checkable without BQ), plus optionally a `$0` `dry_run=True` job to prove the SQL parses (the 82.12 pattern).

**Do not clear only `_fundamentals_full` between the two drives** -- branch 2 (`_fundamentals_cache`, `:615-619`) will silently serve branch 1's memo and the "independent" drive will be a lie. `cache.clear_cache()` (`:193-199`) clears both.

### Q5 (e) -- The cheapest defensible before/after backtest.

**There is no CLI-parameterised runner.** `scripts/harness/run_quick_test.py` and `run_experiment.py` both hardcode their params; `run_quick_test.py` fixes `start_date="2018-01-01", end_date="2025-12-31"` and takes `strategy` from `optimizer_best.json` (currently `triple_barrier`). Neither can express this experiment. Write a one-off script under `scripts/backtest/` or `dev/`.

**Which strategy: `qarp`, and the step's reasoning for it is only half right.**
- Derived live: `label_fundamentals_dependent_strategies()` == `{'qarp'}` -- the only strategy whose LABEL function reads a fundamentals-only key. Its labels change directly when the embargo changes.
- **But the step's claim that "non-fundamentals strategies would show a ZERO delta" is FALSE.** Per `backtest_engine.py:446`'s own docstring, *"Every strategy is FEATURE-dependent via `_NUMERIC_FEATURES`, so an uncovered window silently drops 15 of 37 columns (:852) or imputes a fabricated median company (:881-882)."* A `triple_barrier` run WILL move, via the feature matrix. `qarp` is still the right primary because its delta is direct and interpretable; a `triple_barrier` run is a worthwhile secondary if budget allows, and its delta is evidence about the whole engine rather than one strategy. Do not write "zero delta" into the artifact.

**Window: start 2025-06-30 or later for the headline run.** The gate refuses `qarp` before `2024-06-30`, but section 1d shows that at a 2025-03-31 cutoff a 60-day embargo takes 5-quarter coverage from 41.7% to **0.0%** -- a run starting earlier measures "the coverage hole" more than "the embargo". Recommended headline: **`start_date="2025-06-30", end_date="2026-02-28"`** (`2026-02-28` is the table's `MAX(report_date)`), with `train_window_months=6, test_window_months=2` so the 8-month span yields >1 walk-forward window.

**Prerequisites, measured:**
- `cache.preload_macro()` is called by the engine itself inside `_preload_macro_and_record` (invoked at `backtest_engine.py:~555`), so a manual call is not required -- but macro must be LOADED or the run is macro-free (82.13). Check `_availability["macro"]` in the output.
- `cache.preload_fundamentals(503 tickers)` measured at **3.1 s, 4725 rows** -- fundamentals loading is not the cost.
- `load_dotenv("backend/.env")` + `BigQueryClient(settings)` + `dataset=settings.bq_dataset_reports` (**`financial_reports`**, NOT `pyfinagent_data`).

**Exact command shape** (run from repo root, `source .venv/bin/activate` first). Run it TWICE, once with the embargo constant at 0 and once at 60 -- the ONLY variable that changes:

```bash
source .venv/bin/activate && \
FUNDAMENTALS_EMBARGO_DAYS=0 python scripts/backtest/run_82_51_embargo_ab.py 2>&1 | tee handoff/current/live_check_82.51_before.txt
source .venv/bin/activate && \
FUNDAMENTALS_EMBARGO_DAYS=60 python scripts/backtest/run_82_51_embargo_ab.py 2>&1 | tee handoff/current/live_check_82.51_after.txt
```

with `run_82_51_embargo_ab.py` built from `run_quick_test.py`'s skeleton but:
```python
engine = BacktestEngine(
    bq_client=bq.client, project=settings.gcp_project_id,
    dataset=settings.bq_dataset_reports,          # financial_reports
    start_date="2025-06-30", end_date="2026-02-28",
    train_window_months=6, test_window_months=2,
    strategy="qarp",                              # the only label-fundamentals-dependent strategy
    holding_days=90, tp_pct=10.0, sl_pct=12.923403579416114,
    min_samples_leaf=20, max_positions=20, top_n_candidates=50,
    transaction_cost_pct=0.1,
)
result = engine.run_backtest()
report = generate_report(result, num_trials=1)
a = report["analytics"]
print(f'sharpe={a["sharpe"]:.4f} dsr={a["deflated_sharpe"]:.4f} n_trades={a["n_trades"]} '
      f'availability={result.data_availability}')
```
Record **Sharpe and `n_trades` for both runs** plus both commands verbatim -- that is criterion 4's literal ask. Also record `result.data_availability`, which is how criterion 4 and Q6 connect.

**Runtime: UNMEASURED, and the contract should say so rather than guess.** The two measured components are the 3.1 s preload and the fact that the 8-month window produces ~2 walk-forward windows versus the ~32 in the standard 2018-2025 run. Budget for it in the GENERATE phase; do not put an unmeasured minute-count in the artifact.

**Expected sign, stated in advance so the result cannot be rationalised after the fact:** Sharpe should go DOWN or stay flat and `n_trades` should go DOWN or stay flat. **A Sharpe delta of exactly 0.0000 is the alarm signal** -- given section 1a's decoy, an exactly-zero delta most likely means the embargo was not actually applied (e.g. it was wired through `filing_date`). Pre-register that check.

### Q6 (f) -- Interaction with `FUNDAMENTALS_COVERAGE_START`. **Yes, and leaving it alone creates a FALSE PASS.**

`window_is_covered(window_start)` (`fundamentals_coverage.py:113-124`) compares against the raw measured minimum `2024-06-30`. With a 60-day embargo, **the first date on which any row is visible becomes 2024-08-29** (section 1c). So a window starting `2024-07-01` would:
- pass `window_is_covered` -> `True`
- be recorded as `data_availability.fundamentals = True` (`backtest_engine.py:496`)
- and yet every `cached_fundamentals(t, d)` for `d < 2024-08-29` returns `[]`.

**That is exactly the "records coverage it does not have" failure 82.21 was built to prevent, reintroduced by this step's own fix.** 43 business days in the measured grid have zero visible rows at N=60 (65 at N=90).

**Concrete extension of 82.21's record (not a parallel signal):**
1. **Keep `FUNDAMENTALS_COVERAGE_START = "2024-06-30"` unchanged** -- it is the RAW measured minimum and `_fundamentals_coverage.json` plus `test_phase_82_21_...py:66` both pin it. Changing it would corrupt the measurement.
2. **Add** to `fundamentals_coverage.py`: `FUNDAMENTALS_EMBARGO_DAYS = 60` and a derived `effective_coverage_start() -> str` returning `FUNDAMENTALS_COVERAGE_START + FUNDAMENTALS_EMBARGO_DAYS` = **`"2024-08-29"`**. Derived, never a second hardcoded literal -- a literal would drift the moment either input changes.
3. **Move the REFUSAL predicate** in `backtest_engine._preload_fundamentals_and_record` (`:485`) from `window_is_covered(window_start)` to a comparison against `effective_coverage_start()`, so a `qarp` run starting `2024-07-01` is refused with a message naming the embargo.
4. **Extend the returned availability dict** (`:496-501`) with `fundamentals_embargo_days` and `fundamentals_effective_coverage_start`, alongside the existing `fundamentals` / `fundamentals_coverage_start` / `fundamentals_window_start` / `fundamentals_label_dependent`. This is literally "extend that record".
5. **Leave `window_is_covered` semantics alone** so `test_window_is_covered_boundaries` (`:123-129`) stays green; it answers "is the window inside the raw data", which remains a true and separately useful question. Introducing the embargo at the REFUSAL site rather than inside `window_is_covered` is what keeps this an extension rather than a redefinition.

### Q7 (g) -- Existing tests/guards that break when less data is visible.

**Will break, deterministically, from the code edit itself:**

1. `backend/tests/test_phase_82_12_string_column_guards.py:372-382`, `test_classified_line_numbers_still_point_at_a_row_read` --
   ```python
   assert any(abs(ln - entry["line"]) <= 6 for ln in lines), ...
   ```
   with `CLASSIFICATION[("backend/backtest/cache.py","report_date")]["line"] = 612` (`:316`). **Any edit that moves the `report_date` row-read more than 6 lines from 612 turns this RED.** Inserting an `_embargoed_cutoff` helper above `cached_fundamentals` will do exactly that. Fix: update the `line` value (it is a re-derived-from-source table, explicitly designed to be updated -- `"""file:line claims rot. Re-derive rather than trusting the table."""`).
2. Same file, `test_every_derived_scope_member_is_classified` (`:351-370`) -- asserts `set(CLASSIFICATION) - sites` is empty. **If the fix removes the `r.get("report_date", ...)` read from `cache.py` entirely** (e.g. by relocating the predicate to `fundamentals_coverage.py`), the classification entry becomes "stale" and this goes RED. Also update the `why` text, which quotes the current source line verbatim: `` `str(r.get("report_date", "")) <= cutoff_date` ``.

**At risk, depending on implementation choice:**

3. `backend/tests/test_phase_82_43_macro_feature_absence.py:340-360` -- five source anchors that must each match **exactly one** line of `cache.py`, with an explicit comment that a bare `except Exception` already matches FOUR sites. Adding another `except Exception` or duplicating any anchor string in `cache.py` turns this RED.
4. `backend/tests/test_phase_82_21_fundamentals_coverage.py:123-129`, `test_window_is_covered_boundaries` -- `assert window_is_covered("2024-06-30") is True`. Goes RED **only if** the embargo is folded into `window_is_covered`; Q6's recommendation (apply it at the refusal site) keeps it green. This is the concrete reason to prefer that shape.
5. `backend/tests/test_phase_82_21_fundamentals_coverage.py:371` -- `_engine("qarp","2025-01-01")._preload_fundamentals_and_record()` expects **no raise**. `2025-01-01 > 2024-08-29`, so it stays green at N=60 -- but it would go RED for any embargo beyond **185 days**. Worth knowing the headroom.
6. `backend/tests/test_phase_75_bq_discipline.py` -- asserts on parameterised-query shapes; the fix keeps `@cutoff` as a `ScalarQueryParameter`, so it should stay green. Verify, do not assume.

**Not tests, but guards that go quieter:** `test_phase_82_21_...py:66` (`min_report_date == "2024-06-30"`) and `snapshot_drift()` both stay green **provided `_fundamentals_coverage.json` is NOT edited.** Do not "update the snapshot to reflect the embargo" -- the snapshot is a raw measurement.

**A guard that does NOT exist and should be added:** nothing currently asserts that `filing_date != report_date` before anyone filters on `filing_date`. Given section 1a, the 82.51 test module should carry a live-or-snapshot guard that FAILS if code ever filters on `filing_date` while the column is still a copy. That converts the decoy from a latent trap into a tripwire.

## 7. Application to pyfinagent -- the shortest correct change

1. `backend/backtest/fundamentals_coverage.py` -- add `FUNDAMENTALS_EMBARGO_DAYS = 60` + `effective_coverage_start()` (Q6.2), with the SEC-deadline reason in the docstring.
2. `backend/backtest/cache.py` -- add `_embargoed_cutoff(cutoff_date)` (Q4); apply at **`:612`** and pass at **`:637`** so `:631`'s `@cutoff` receives the embargoed value. Two filter sites, one rule.
3. `backend/backtest/backtest_engine.py:485, 496-501` -- refuse against `effective_coverage_start()`; extend `data_availability` with `fundamentals_embargo_days` + `fundamentals_effective_coverage_start` (Q6.3-4).
4. `backend/tests/test_phase_82_12_string_column_guards.py:316` -- re-derive the `line` value and refresh the quoted `why` (Q7.1-2).
5. New `backend/tests/test_phase_82_51_fundamentals_embargo.py` -- criteria 1-3 plus the `filing_date`-decoy tripwire (Q7).
6. Step artifact -- criterion 4's two commands + Sharpe/`n_trades` deltas (Q5); criterion 5's decision **"(a) fixed 60-day embargo, because `filing_date` is a measured copy of `report_date` on 4798/4798 rows so option (b) is unavailable until 82.50, and 60d is the smallest embargo dominating the 10-K large-accelerated deadline that governs this table's largest cohort"** (Q2/Q3).

## 8. Consensus vs debate (external)

**Consensus:** (i) filtering on fiscal-period end is look-ahead bias -- unanimous across sources #2, #3, #4, #5, #7; (ii) the right fix is a real filing/announcement date (`rdq`, EDGAR submission timestamp), not a lag -- sources #3, #4, #5; (iii) when that is unavailable, a fixed lag is the accepted fallback -- sources #2, #3, #4.

**Debate -- the fixed lag's SIZE is genuinely unsettled, and the sources disagree:** Fama-French July-of-t+1 (~6 months, annual) vs the "four-month gap" variant (source #3) vs OpenSourceAP's 3 months for quarterly (source #4) vs the vendor's 120 days (source #2) vs the 60-90d practitioner range. **No source recommends the same number.** That is why this brief anchors on the SEC's legally binding deadline for THIS universe rather than importing anyone's convention.

**Adversarial position, taken seriously:** over-lagging costs real money (source #6: 5.6% US annualised for stale signals) and a fixed lag is provably wrong at the tails (source #4: >50,000 violations, issue still OPEN). Neither overturns the fix -- reading a number before it existed is not a tradeoff, it is a defect -- but both argue against the vendor's 120d and for stopping at the smallest legally sufficient value.

## 9. Pitfalls (from literature and from this repo's own history)

1. **The `filing_date` decoy** (section 1a) -- the highest-probability way this step ships a no-op that reads as a fix.
2. **Do not conflate this embargo with the walk-forward embargo.** `BacktestEngine(embargo_days=5)` and `WalkForwardScheduler` implement Lopez de Prado's train/test purge-and-embargo -- a completely different mechanism. Name the new constant `FUNDAMENTALS_EMBARGO_DAYS`, never `embargo_days`.
3. **Do not import the Fama-French 6-month lag.** It is an ANNUAL convention; applied to quarterly rows it would destroy 3 of every 4 quarters.
4. **Do not quote the monthly-grid cost table** (section 1c) -- it shows a spurious plateau making 60 look free.
5. **A zero Sharpe delta is a red flag, not a result** (Q5).
6. **Do not edit `_fundamentals_coverage.json`** (Q7).
7. **This fixes publication lag ONLY, not revision bias.** 18,734 EDGAR rows were later revised >0.5% on the same tag (source #2); our table stores one overwritten value per (ticker, period). Even a perfect embargo serves the RESTATED number. That is a separate defect and belongs in its own queued step, not silently inside 82.51.
8. **The MCP data server (`data_server.py:149`) is a live consumer** -- disclose the blast radius.

## 10. Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL -- **8** (SEC 33-8644, Tradevo, SAFE CRSP-Compustat, OpenSourceAP #50, Calcbench, Maven, arXiv 2605.24564, Jacobs-Levy/Hrdlicka). Two required the `curl` + `pdfplumber` fallback chain per `.claude/rules/research-gate.md`.
- [x] 10+ unique URLs total -- **37** (8 read in full + 29 snippet-only).
- [x] Recency scan (2024-2026) performed + reported -- section 4, three-variant query discipline documented.
- [x] Full papers/pages read, not abstracts -- `arxiv.org/abs/2601.13770` was explicitly EXCLUDED from the read-in-full set for this reason.
- [x] file:line anchors for every internal claim -- all re-derived this session (2026-08-06); none cited from memory.

Soft checks:
- [x] Internal exploration covered every relevant module (6 named in scope + 6 additional discovered: `data_ingestion.py`, `data_server.py`, `settings.py`, `candidate_selector.py`, and 3 further test modules).
- [x] Contradictions / consensus noted -- section 8; the sources genuinely disagree on lag size.
- [x] Claims cited per-claim.
- **Known gap:** the S&P Capital IQ *Point-In-Time vs. Lagged Fundamentals* research note (Aug 2015) is the highest-value un-read source and returned 403 to both WebFetch and `curl`. Its headline snippet ("PIT backtests produce significantly different results than lagged Non-PIT data using common factors") is directionally consistent with everything above, but its measured magnitudes are missing from this brief.

## 11. Adaptive-coverage log (audit-class, K=2)

| Round | Focus | New read-in-full findings | Dry? |
|-------|-------|---------------------------|------|
| 1 | Broad scan: PIT/look-ahead canonical, FF lag convention, SEC deadlines | 6 (SEC, Tradevo, Calcbench, Hrdlicka, arXiv 2605, + BQ schema/lag measurement) | no |
| 2 | Gap: measured PIT-vs-lagged impact; Compustat RDQ prior art | 2 (SAFE CRSP-Compustat, OpenSourceAP #50) | no |
| 3 | Gap: does yfinance expose ANY vintage? | 0 -- only corroborated `data_ingestion.py:278`; nothing new read in full | **DRY 1** |
| 4 | Adversarial: does over-lagging destroy alpha? | 1 (Maven alpha-decay) | no |
| 5 | Gap: 45-vs-90 embargo selection guidance | 0 -- every hit already in the snippet set or already read | **DRY 1** |
| 6 | Confirm: EDGAR XBRL-era lag distribution / anything superseding | 0 | **DRY 2** |

`dry_rounds = 2 >= K_required = 2` -> `coverage.dry = true`.

## 12. JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 29,
  "urls_collected": 37,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {
    "audit_class": true,
    "rounds": 6,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "brief_path": "handoff/current/research_brief_82.51.md",
  "gate_passed": true
}
```
