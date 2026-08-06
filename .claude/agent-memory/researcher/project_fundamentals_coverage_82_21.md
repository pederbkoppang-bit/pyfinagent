---
name: fundamentals-coverage-82-21
description: 82.21 fundamentals floor -- EDGAR companyfacts carries filed+accn (PIT possible, restatement measured on AAPL); the existing yfinance table has NO vintage AND leaks ~66d; MinBTL admits ~2 trials at 20 months so "Branch A" is really NOT-EVALUABLE; only qarp is label-dependent but all six are feature-dependent via silent zero-fill
metadata:
  type: project
---

Measured 2026-08-06 during the 82.21 research gate. Every number below came from a live
fetch or a re-derived file:line, not from the step description.

**The branch framing in the step is too generous.** "Accept that fundamentals-dependent
strategies are evaluable from 2024-07 on" is not what the evidence supports. Coverage is
2024-06-30..2026-02-28 ≈ 1.67 yr; Bailey/Borwein/López de Prado/Zhu Theorem 2
(`MinBTL < 2·ln[N]/E[max_N]²`, Notices of the AMS 61(5) 2014) inverts to **N < ~2.3
independent configurations** at E[max]=1. The optimizer sweeps 17 params. So the honest
Branch-A statement is **"not evaluable"**, full stop — and that is still a defensible
choice, because qarp already produces 0 samples and quality_momentum/factor_model are
already demoted (82.16). Branch A costs nothing that currently works.

**The covered window is ALSO contaminated — do not treat 2024-07+ as clean.**
`data_ingestion.py:278` writes `"filing_date": report_date` with the comment *"true
filing date not available from yfinance"*, and the PIT read filters `report_date <=
cutoff` (`cache.py:612`, `:631`). Measured publication lag over 313,406 rows / 5,194 US
companies: **mean 66d, median 60d, p90 90d, max 120d** (Tradevo 2026, vendor — COI
disclosed in-source). A 45-day embargo would still leak on ~40% of filings; **use 90**.
Also ≈6.0% of rows are later revised >0.5% on the same tag. TRAP: `cache.py:283-292`
deliberately projects `filing_date` OUT of `preload_fundamentals` — any vintage fix must
widen it or the bulk path and the `:626-634` per-ticker fallback silently disagree.

**EDGAR: measure it, don't read about it.** `curl -A "<name> <email>"` gets 200;
**WebFetch on sec.gov returns 403**. AAPL companyfacts = 3.79 MB / 1.2 s, 503 us-gaap
tags, fact keys `accn/end/filed/form/fp/fy/val` (+`start`, +`frame`).
- **PIT is possible**: Assets@end=2008-09-27 was $39.572B (10-Q filed 2009-07-22) then
  **$36.171B** (10-K/A filed 2010-01-25) — a -8.6% restatement visible in the data.
  18 of 70 distinct end-dates carry >1 vintage.
- **The cost is normalisation, not bandwidth.** 503 tickers ≈ 504 requests, ~2 GB,
  2-5 min wall-clock at the 10 req/s cap. But: `total_debt` **has no us-gaap tag** (sum
  LongTermDebtNoncurrent + LongTermDebtCurrent + CommercialPaper, and their start dates
  differ); revenue drifts 3 ways across ASC 606 on ONE company (`SalesRevenueNet`
  ..2018-06-30 → `Revenues` → `RevenueFromContractWithCustomerExcludingAssessedTax`
  2017-09-30..); duration facts are **3/6/9/12 months sharing the same `end`** (filter on
  `end` alone = ~4x magnitude error); only **70 of 146** Assets facts carry `frame`.
  The taxonomy versions ANNUALLY (EDGAR Release 26.1, 2026-03-16) so tag maintenance is
  recurring, not one-off.
- Du/Huddart/Jiang 2021 (Penn State/Waterloo) is the strongest *signal-quality* argument
  for EDGAR: *"accruals calculated from as-filed data do predict returns and accruals
  calculated from Compustat data do not"* — and it survives an unrestated-Compustat
  control, so it is standardization, not vintage. It cuts both ways: as-filed data
  *"may contain errors and use custom tags"*.

**Fundamentals-dependency: derive it, don't eyeball it.** Rule = the label fn named by
`STRATEGY_REGISTRY[S]` reads a key assigned ONLY inside `if fundamentals:`
(`historical_data.py:140-266`, 17 keys). Result: **exactly ONE selectable strategy is
label-dependent — `qarp`** (`backtest_engine.py:1589-1592`, hard-refuses at :1594-1595).
`triple_barrier`/`meta_label` never call `build_feature_vector` at all (:891→:903);
`stretch_regime` LOOKS dependent but reads only price/vol — eyeballing gets that wrong.
`quality_momentum`/`factor_model` are dependent but demoted (:54-67).

**But ALL SIX are FEATURE-dependent, and it fails silently two different ways.**
`_NUMERIC_FEATURES` (:124-136) carries 15 fundamentals keys for every strategy.
Fully-uncovered window → `:852 [c for c in _NUMERIC_FEATURES if c in df.columns]`
silently drops them (model trains on 22 of 37, no record). Straddling window →
`:881-882 X.fillna(train_medians)` then `.fillna(0)` **fabricates a median company for
four-fifths of the sample** — strictly worse than dropping. And
`quality_momentum:1283 fv.get("quality_score", 0) or 0` makes `>0.3` unreachable and
`<0.1` always true with no fundamentals: **a bearish label manufactured from absence.**

**Reuse 82.13, don't invent.** `data_availability` already exists: default at
`backtest_engine.py:209`, `_preload_macro_and_record` :368-400, labelled at construction
:450-458, double-surfaced at `analytics.py:843-845` (top level AND `["analytics"]`,
because strategy_backtest_adapter + strategy_candidate_producer read only the latter).
Its test module `test_phase_82_13_preload_refusal_handling.py:184-218,:419-425` even
AST-asserts the field is derived, not literal. That is the template for criterion 3.

**The guard for "structurally unavailable vs genuinely null" must be THREE-way.**
A sentinel is unusable (debt_equity/roe/profit_margin are legitimately negative) and
`None` vs `NaN` collapses at `:754 fv.get(f, np.nan)`. Use a `fundamentals_available`
boolean set on BOTH branches of `if fundamentals:` — and **never add it to
`_NUMERIC_FEATURES`** (it is a perfect "date >= 2024-07" regime dummy). The test must
assert (a) uncovered → False + keys absent, (b) covered → True + real numeric values,
(c) **covered-but-null → True AND `pe_ratio is None`** (a loss-making company;
`pe_ratio` is only assigned when `net_income > 0`, `historical_data.py:156-158`).
Only (c) catches a fix that merely renames None to "unavailable". See
[[fabricated-safe-80-36]] and [[guards-stop-one-seam-short]].

`report_date` is a BQ **STRING**; `MIN()` is lexicographic and is safe ONLY because
`data_ingestion.py:257` writes `strftime("%Y-%m-%d")` — assert that format with a regex,
never write an `isinstance(v, date)` guard (see [[vacuous-bq-guards-82-12]]).
`schema_oracle.py` (:84/:116/:130/:136/:453/:550) is the right snapshot+drift+`dry_run`
instrument for criterion 1.

**Brief:** `handoff/current/research_brief_82.21.md`.
