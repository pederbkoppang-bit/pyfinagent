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

## 2. Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| _(pending)_ |

## 3. Identified but snippet-only

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| _(pending)_ |

## 4. Recency scan (2024-2026)

_(pending)_

## 5. Internal code inventory

_(pending)_

## 6. Q1-Q7 answers

_(pending)_

## 7. Research Gate Checklist

_(pending)_

## 8. JSON envelope

_(pending)_
