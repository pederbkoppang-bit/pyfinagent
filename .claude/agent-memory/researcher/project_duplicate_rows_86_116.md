---
name: duplicate-rows-86-116
description: historical_prices is 38% duplicate rows (LEGACY, stopped in 2026); drop_duplicates is the WRONG tool because 55.8% of dup pairs differ by float noise; positional vs distributional statistics break on DIFFERENT subsets
metadata:
  type: project
---

Step 86.116 (2026-08-18) measured duplicate-row contamination in the stored price
series. Five things a future session should not have to re-derive.

**1. The contamination is real, large, and LEGACY.**
`sunny-might-477607-p8.financial_reports.historical_prices` (location `us-central1`,
NOT `pyfinagent_data` -- a census against the wrong dataset 404s):
total_rows 1,859,482 / distinct (ticker,date,market) 1,152,607 =
**706,875 duplicate rows, 38.01% of rows, 61.3% of KEYS**, max multiplicity exactly 2,
336 of 513 tickers. Per-year share of keys duplicated: 2017 **90.5%**, 2018-2025 flat at
**62-64.5%**, **2026 = 0.1%**. So one historical backfill ran twice and the phase-75.9
write-side `SELECT DISTINCT` probe (`data_ingestion.py:108-112`) is holding now. That
makes a one-time repair *bounded and terminal*, which is the fact that decides
repair-vs-read-dedup (answer: both, read-side first).

**Why:** the risk was already named in-repo at `data_ingestion.py:99-106` ("producing
duplicate (ticker,date) bars that distort features/MTM/Sharpe downstream") but only the
fail-closed re-raise shipped -- nobody backfilled the damage or guarded the read.

**How to apply:** before proposing a dedup fix, re-run the census; do not assume the
2026-clean trend continued. `date` is a **STRING** column here (an `EXTRACT(YEAR FROM
date)` 400s) -- use `SUBSTR(date,1,4)`.

**2. `drop_duplicates()` is the wrong primitive -- with a measured reason.**
pandas docs, verbatim: *"Indexes, including time indexes are ignored."* It is
VALUE-keyed. **394,719 of the 706,875 duplicated keys (55.8%) have a differing `close`**
-- so value-keyed dedup silently leaves more than half of them. The divergence is float
noise, not a re-adjustment basis: p50 = p90 = p99 = 0.0% gap, max 0.93%. Correct
primitive is date-keyed: `df[~df.index.duplicated(keep=...)]`.
Repo-wide: **ZERO** `drop_duplicates`/`duplicated` calls exist in `backend/`; all 3 hits
are in `scripts/qa/` (86.59 tooling). `validate_ohlcv` (`price_quality.py`, 84 lines) does
NOT dedupe either.

**3. Positional lookbacks and distributional statistics break on DIFFERENT subsets.**
Measured (pandas 3.0.1, 252 synthetic bdays, seed 7):
duplicating **only the last 40 bars** corrupts `momentum_1m`/`momentum_3m`/`rsi_14` to
*exactly the same values* as duplicating the whole history (-10.598 / -16.106 / 13.00),
while annualized vol barely moves (0.2040 vs 0.1557 vs clean 0.2194).
Because `_pct_change` uses `series.iloc[-periods-1]`, an **iloc lookback is corrupted by
RECENT duplicates only**; `std()`-type statistics are corrupted by the **GLOBAL** rate.
Full duplication understates vol by ~29% (ratio 0.7093 vs closed-form
`sqrt(n/(n+m)) = 1/sqrt(2) = 0.7071`).
**How to apply:** a tail-only or head-only sample will mis-diagnose one of the two. Audit
both the local density and the global rate. Note `momentum_6m = _pct_change(close,
len(close)-1)` is endpoint-to-endpoint and therefore duplicate-INVARIANT.

**4. This is NOT a Sharpe-formula bug.** `backtest_trader.py:188` builds NAV from a
`dict[str,float]`, one entry per business day, so row-count inflation cannot dilute the
NAV series. Contamination reaches the DSR/PBO gate through **corrupted features** and
**mis-scaled triple-barrier widths** (`historical_data.py:124-132` daily_volatility),
i.e. measurement error in X. Direction of the net Sharpe bias is therefore
**undetermined** without an A/B replay -- do not assert it.

**5. The 2025 ML literature says duplicates often HELP -- and it does not transfer.**
arXiv 2511.10964 and 2504.00638v1 both find duplicate *records* raise test accuracy
(regularization). There, duplication changes sample COMPOSITION while each row's feature
values stay correct. Here, features are computed FROM the duplicated series. Keep this
distinction handy; it is the obvious counterargument to the whole step.

Related: [[reference-webfetch-gcloud-docs]] -- the curl workaround for cloud.google.com
now fails on the new `docs.cloud.google.com` host too (229 chars extracted).
</content>
