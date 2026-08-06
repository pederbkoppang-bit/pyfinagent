# Contract -- phase-82.51

**Step:** 82.51 (P1) -- publication-lag look-ahead on every fundamentals read.
**Date:** 2026-08-06. **Cycle:** 1.
**Research gate:** PASSED -- `handoff/current/research_brief_82.51.md`,
`gate_passed: true`, **audit_class** with `dry: true` after 6 rounds / 2 dry,
8 sources read in full, 37 URLs, 12 internal files.

*(The rail dropped this gate's return value -- `agents_empty_result: 1` after
207K tokens. The brief survived in full on disk because of the write-first
discipline. Nothing was re-run.)*

---

## 1. The step's central factual claim is REFUTED, and the refutation is a trap

The step says the fix may filter on "a real filing date where one exists".
**`filing_date` exists, is 100% populated, and is a verbatim copy of
`report_date`.** Measured by me, then independently by the gate:

```
n_rows                 4798
n_filing_missing       0
n_filing_unparseable   0
lag (filing - report): mean 0.0  p50 0  p90 0  p99 0  min 0  max 0
```

Root cause, found by the gate at `backend/backtest/data_ingestion.py:278`:

```python
"filing_date": report_date,  # Approximation; true filing date not available from yfinance
```

**Why this is a trap and not merely a null result.** An implementer who sees a
NOT-NULL `filing_date` and "fixes" the bug by switching the filter to it
produces a **byte-identical result set and a byte-identical backtest**, while
the artifact would claim option (b) was taken. Criterion 4's Sharpe delta would
read exactly `0.0000` -- which reads as "the leakage was immaterial" when the
change was a no-op. Criterion 1's fixture would still pass, because fixtures set
their own values. Same class as 82.12's STRING-column guards and 82.39's
phantom columns.

**So option (b) is NOT available. 82.50 is a genuine prerequisite for it.**
This step implements option (a), a fixed embargo, and ships a tripwire (§5) so
the decoy cannot be walked into later.

`ingested_at` is also unusable: all 4798 rows were bulk-backfilled over two
weeks in 2026-03/04 against `report_date`s spanning 2024-06-30..2026-02-28. It
records when *we* fetched, not when the market saw it.

## 2. The step's "measured mean 66 days" is misattributed

The step presents `mean 66 / median 60 / p90 90` as **"measured lag on the live
table"**. It cannot have been: the lag on our table is identically 0. The gate
traced the figures to `research_brief_82.21.md:48` -> a **tier-4 vendor blog
with a disclosed commercial COI**, measuring **SEC EDGAR** across 5,194 US
companies.

**And the same source's large-cap subsample says 43d mean / 61d max.** Our
universe is `get_universe_tickers(market="US")` = **503 S&P 500 tickers**, all
large accelerated filers by definition of index inclusion. The all-filer 66d is
the wrong calibration target and would over-embargo.

Honest form for the record: *the publication lag is UNMEASURABLE on our table;
the best external estimate for a large-cap universe is ~43d mean / 61d max.*

## 3. DECISION (criterion 5): a fixed **60-calendar-day** embargo on `report_date`

**Regulatory argument (primary).** The largest cohort in the table is the
fiscal-year-end quarter -- **744 rows / 443 tickers** -- which is governed by
the **10-K**, whose large-accelerated deadline is **60 days**. The other three
quarters are 10-Qs at 40 days. A 45-day embargo would cover the 10-Q deadline
but **under-cover the 10-K by 15 days**, leaving residual leakage on precisely
the biggest cohort. **60 is the smallest embargo that dominates every legally
binding deadline for this universe.**

**Cost, measured on THIS table** (business-day cutoff grid, 435 cutoffs):

| N | visible row-days | vs N=0 | tickers w/ 5 quarters @2025-06-30 | @2025-12-31 |
|---|-----------------|--------|-----------------------------------|-------------|
| 45 | 781,360 | -16.6% | 228 (45.3%) | 441 (87.7%) |
| **60** | **730,564** | **-22.1%** | **228 (45.3%)** | **441 (87.7%)** |
| 90 | 648,446 | -30.8% | 210 (41.7%) | 436 (86.7%) |

45 -> 60 costs 5.5pp of row-days but **zero** 5-quarter tickers at either
cutoff; 60 -> 90 is where feature-level damage starts. 60 sits on the efficient
frontier. Over-lagging is not free either -- stale-signal cost is ~5.6%
annualised in the US, a second reason to stop at 60.

**Residual risk, recorded rather than hidden:** a fixed lag is wrong at both
tails. OpenSourceAP's analogous `datadate + 3 months` rule still has >50,000
violating observations (open issue). **60 days is a defensible approximation,
not a correctness proof. The correct fix is a real filing date from 82.50.**

Do NOT import the Fama-French 6-month convention: it applies to annual data,
and on quarterly data it would destroy 3 of every 4 quarters.

## 4. Immutable success criteria (verbatim)

1. "a fixture whose fundamentals row has a period end before the cutoff but a
   publication date after it is EXCLUDED, asserted by a test that fails against
   the current `report_date <= cutoff` filter"
2. "a fixture whose row was demonstrably public before the cutoff is still
   INCLUDED, so the fix cannot pass by excluding everything"
3. "both the bulk preload path and the per-cutoff fallback path apply the same
   rule, asserted by driving each independently and comparing their outputs on
   the same fixture"
4. "the before/after Sharpe and trade-count deltas from at least one real
   backtest are recorded in the step artifact with the commands that produced
   them"
5. "the chosen approach (fixed embargo vs real filing date) is recorded with its
   reason in the step artifact"

**Verification command (immutable):**
`source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_51_fundamentals_embargo.py -q`

## 5. Plan

- **D1 -- one shared seam, via an algebraic identity.** The two paths cannot
  share a filter directly (Python list-comp vs BigQuery SQL), but
  `report_date + N <= cutoff` **iff** `report_date <= cutoff - N`. So shift the
  **cutoff once**: `_embargoed_cutoff(cutoff_date)` called at the top of
  `cached_fundamentals`, feeding both `cache.py:612` and the `@cutoff`
  parameter at `:637`. Mutating the helper must flip BOTH branches -- that is
  what makes it a real seam and not one-seam-short.
- **D2 -- criterion 3's test drives each branch independently.** Branch 1 via
  `_fundamentals_full`; branch 3 via `clear_cache()` (**not** just clearing
  `_fundamentals_full` -- branch 2's `_fundamentals_cache` would silently serve
  branch 1's memo and the "independent" drive would be a lie) plus a stubbed
  `_bq_client.query` returning the fixture rows **UNFILTERED**. A stub that
  pre-filters proves nothing about the SQL predicate. Plus a separate assertion
  that the generated query carries the embargoed value as its parameter.
- **D3 -- extend 82.21's record; do NOT redefine it.** Keep
  `FUNDAMENTALS_COVERAGE_START = "2024-06-30"` (it is the raw measurement, and
  the snapshot + its test pin it). ADD `FUNDAMENTALS_EMBARGO_DAYS = 60` and a
  **derived** `effective_coverage_start()` = 2024-08-29 -- derived, never a
  second literal. Move the refusal predicate in
  `_preload_fundamentals_and_record` to compare against it, and extend the
  availability dict with `fundamentals_embargo_days` +
  `fundamentals_effective_coverage_start`.
- **D4 -- the decoy tripwire.** Add a guard that FAILS if any code filters on
  `filing_date` while the column is still a copy of `report_date`. This converts
  the §1 trap from latent to loud.
- **D5 -- criterion 4's real backtest**, twice, embargo 0 vs 60 as the only
  variable. Window `2025-06-30 .. 2026-02-28`, strategy `qarp`.

### THIS FIX WOULD OTHERWISE RE-CREATE THE EXACT DEFECT 82.21 CLOSED

Without D3, a window starting 2024-07-01 would pass `window_is_covered()`, be
recorded `fundamentals: True`, and yet have **every** `cached_fundamentals` call
return `[]` -- because the first date any row is visible at N=60 becomes
2024-08-29. **43 business days in the measured grid have zero visible rows at
N=60.** That is "records coverage it does not have", reintroduced by this step's
own fix. D3 is not optional polish.

### Pre-registered expected sign (so the result cannot be rationalised after)

Sharpe should go **DOWN or flat**; `n_trades` **DOWN or flat**. **A Sharpe delta
of exactly 0.0000 is an ALARM, not a pass** -- given §1's decoy, an exactly-zero
delta most likely means the embargo was not actually applied.

## 6. Known-breaking guards (from the gate, to be fixed deliberately)

1. `test_phase_82_12_string_column_guards.py:372-382` pins
   `cache.py:report_date` at **line 612 ±6**. Inserting the helper moves it.
   The table is explicitly designed to be re-derived -- update the value and the
   quoted `why` text.
2. Same file, `test_every_derived_scope_member_is_classified` goes RED if the
   `report_date` read leaves `cache.py` entirely.
3. `test_phase_82_43_macro_feature_absence.py:340-360` requires five source
   anchors to match **exactly one** line of `cache.py`; a new bare
   `except Exception` or a duplicated anchor string turns it RED.
4. `test_phase_82_21_...py:123-129` stays green **only because** D3 applies the
   embargo at the refusal site rather than inside `window_is_covered`. That is
   the concrete reason for that shape.
5. Do **NOT** "update `_fundamentals_coverage.json` to reflect the embargo" --
   the snapshot is a raw measurement and `snapshot_drift()` depends on it.

## 7. Non-scope

No new data source (82.50 owns EDGAR). No change to `window_is_covered`
semantics. No change to `_fundamentals_coverage.json`. No change to
`data_ingestion.py:278` -- fixing the producer requires a real filing date,
which is 82.50's job; this step only makes the decoy loud. No live positions.

## 8. References

- `handoff/current/research_brief_82.51.md` (audit-class, dry after 2 rounds)
- SEC Release 33-8644 (accelerated filer deadlines; Audit-Analytics fn.49)
- Lopez de Prado -- backtest overfitting / leakage
- OpenSourceAP -- open issue on fixed-lag violations
- Internal: `backend/backtest/cache.py:265-330,600-648`,
  `backend/backtest/data_ingestion.py:278`,
  `backend/backtest/fundamentals_coverage.py`,
  `backend/backtest/historical_data.py:254-264,455`,
  `backend/backtest/backtest_engine.py:485-501`
