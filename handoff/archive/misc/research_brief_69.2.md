# Research Brief -- phase-69.2 (P0 gate correctness, OFFLINE backtest/analytics)

Tier: **moderate**. Builds on `research_brief_69.0.md` (do NOT redo settled
theory: DSR SE derivation ref=0.9004 / bug-path 0.9999999, AFML Ch.7
purge/embargo rule, sign-safe algebra already established there, 8 sources
read in full).

Scope (5 incremental topics):
1. Fracdiff at predict time (persist train weights + per-feature medians).
2. Business-day / holiday boundary price-lookup snap.
3. Walk-forward purge+embargo as CODE (overlap test, embargo length).
4. Deflated-Sharpe SE-units reconfirmation (de-annualize SR AND V).
5. Go-live / promotion gate design for the two under-spec booleans.

Status: COMPLETE (gate_passed=true; 5 sources read in full; see Provenance note).

---

## Read in full (>=5 required; counts toward the gate)
| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 1 | https://pandas-market-calendars.readthedocs.io/en/latest/usage.html | 2026-07-11 | official docs | WebFetch | `valid_days(start,end)` returns a DatetimeIndex of exchange OPEN dates; `schedule()` gives open/close times. Snap-to-prior pattern: `valid_days(...).asof(pd.Timestamp(target))` "Snaps weekend to prior Friday". Library provides sessions; use pandas `DatetimeIndex.asof()`/`searchsorted()` for nearest-prior |
| 2 | https://mlfinpy.readthedocs.io/en/latest/FractionalDifferentiated.html | 2026-07-11 | official docs (mlfin.py, AFML Ch.5 impl) | WebFetch | FFD weight computation is "a one-time exercise"; weights `omega` "determined entirely by the differencing coefficient d and threshold parameter, NOT by the data itself." Values computed for `t=T-l*+1..T` (initial l*-1 rows drop to NaN); l* = where weight-loss lambda exceeds tolerance tau |
| 3 | https://microalphas.com/fractional-differentiation/ | 2026-07-11 | authoritative blog (quant) | WebFetch | "the weights for a given d and threshold are computed once and then applied to the series as a fixed convolution" -> same weight vector reused across ALL observations; "FFD simply drops every weight whose absolute value falls below a chosen threshold" (fixed window width) |
| 4 | https://scikit-learn.org/stable/common_pitfalls.html | 2026-07-11 | official docs (scikit-learn) | WebFetch (Main; researcher stalled) | "transformations are only learnt from the training data"; "Never include test data when using fit / fit_transform"; "the average should be the average of the train subset, not the average of all the data" -> imputation medians MUST be fit on train and APPLIED (transform) at predict; a different predict-time fill is leakage/skew |
| 5 | https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html | 2026-07-11 | official docs (skfolio, AFML CPCV impl) | WebFetch (Main) | "Purging consist of removing from the training set all observations whose labels overlapped in time with those labels included in the testing set." Embargo = "observations that immediately follow an observation in the testing set"; params are integer observation counts (purged_size / embargo_size) |

## Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/ | industry | purge/embargo definitions -- already read IN FULL in research_brief_69.0.md |
| https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf | peer-reviewed | DSR SE + reference 0.9004 -- already read IN FULL in research_brief_69.0.md |
| https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/cross_validation/combinatorial.py | code | CPCV reference implementation (corroborates skfolio) |
| docs/GO_LIVE_CHECKLIST.md | internal doc | go-live boolean documented spec (internal; see inventory) |

## Recency scan (2024-2026)

Performed. No new finding supersedes the canonical sources. 2024-2026 work (skfolio CombinatorialPurgedCV docs current; mlfin.py/mlfinlab active) APPLIES AFML Ch.5 fracdiff + Ch.7 purge/embargo as the standard rather than replacing them. pandas-market-calendars is a current, maintained exchange-calendar library. sklearn common-pitfalls (current docs) is the authoritative statement of the fit-on-train-only discipline. No 2025-2026 reversal of DSR per-period-units, purge/embargo, or native-currency-ledger practice found.

## Search queries run (3-variant discipline)

- Fracdiff-at-predict: "fractional differentiation fixed weights inference apply predict 2026" / "fracdiff FFD weights data independent one-time 2025" / "fractional differentiation Lopez de Prado weights convolution".
- Boundary snap: "pandas nearest previous trading day exchange calendar snap 2026" / "pandas_market_calendars valid_days asof weekend 2025" / "business day price lookup weekend holiday backtest".
- Purge/embargo code: "purged k-fold embargo implementation python overlap test 2026" / "combinatorial purged cross validation skfolio 2025" / "purging embargo train sample label span Lopez de Prado".
- Transform discipline: "scikit-learn fit transform train only test data leakage" (year-less canonical) + sklearn common_pitfalls.
- Go-live gate: internal (paper_go_live_gate.py docstring + docs/GO_LIVE_CHECKLIST.md); external "probabilistic sharpe ratio minimum track record length" reconfirmed via 69.0.

## Key findings

### KF1 -- Fracdiff at predict (topics 1 & 4, sources 2/3/4)
FFD weights are DATA-INDEPENDENT -- "determined entirely by the differencing coefficient d and threshold, NOT by the data itself" (mlfin.py), "computed once and applied as a fixed convolution" (microalphas). BUT fracdiff is a WINDOWED time-series convolution; a single cross-sectional predict row (one ticker's feature vector at test_start) has NO series window to convolve, and the TRAIN-time code applies fracdiff over the INTERLEAVED mixed-ticker column (`X[col]` across all samples) -- statistically dubious in both regimes. sklearn (source 4) is decisive on the concrete, testable half: transforms (incl. median imputation) MUST be fit on train and merely APPLIED at predict; a different predict-time fill (the current `fillna(0)` vs train `fillna(median)`) is leakage/skew. **Fix that is verifiable + do-no-harm**: persist the train-time per-feature MEDIANS and apply them at predict (replaces `fillna(0)`); make the fracdiff application CONSISTENT between train and predict -- since it cannot be replicated per single row, the honest consistency fix is to apply the identical transform state (or neutralize the dubious interleaved fracdiff so neither regime applies it). GENERATE decides the exact fracdiff approach; the fixture asserts train and predict produce the SAME transform + SAME fill on a known feature vector.

### KF2 -- Boundary snap (topic 2, source 1)
pandas-market-calendars `valid_days(start,end)` returns a DatetimeIndex of exchange OPEN dates; snap a weekend/holiday boundary to the prior session via `DatetimeIndex.asof(target)` ("Snaps weekend to prior Friday"). Minimal-dependency alternative: widen the exact-date lookup to a small `[d-5, d]` range and take the last available close (the daily-MTM loop at backtest_engine.py:505 already tolerates per-day misses via bdate_range). **Fix**: business-day-snap (or range-widen) the entry (`:488`) and liquidation (`:514`) price lookups.

### KF3 -- Purge + embargo (topic 3, source 5 + 69.0's LdP/QuantInsti)
Purge = drop training observations whose LABEL interval overlaps the test fold's time span; embargo = drop training observations immediately AFTER the test fold (serial-correlation guard); params are integer observation counts (skfolio). **Fix in `_build_training_data`**: purge any training sample whose `[sample_date, sample_date + 1.5*holding_days]` overlaps `[test_start, test_end]` (use the TRUE `1.5*holding_days` horizon, confirmed at `_compute_triple_barrier_label` backtest_engine.py:658 -- the recorded `exit_dates` at :596 use plain `holding_days`, understating); keep an embargo (walk_forward.py:61's gap) as the post-test separation.

### KF4 -- DSR units (topic 4, 69.0's Bailey source)
Reconfirmed: de-annualize BOTH `observed_sr` (=SR_ann/sqrt(ppy)) AND `variance_of_srs` (=V_ann/ppy, since SR scales sqrt(ppy)); skew/kurt stay from per-period returns; prefer T-1. Reference pin DSR=0.9004 (bug 0.9999999) established + independently re-derived with scipy.

### KF5 -- Go-live documented spec (topic 5, internal)
Located at `paper_go_live_gate.py:7-15` docstring + `docs/GO_LIVE_CHECKLIST.md`:
- `psr_ge_95_sustained_30d` -- "PSR >= 0.95 over the most recent 30 days" (a per-day/rolling-min SUSTAINMENT), NOT the current point-in-time `psr>=0.95 AND n_obs>=30`.
- `max_dd_within_tolerance` -- "realized max DD not worse than backtest max DD + 5pp buffer" (relative), currently only a 20% ABSOLUTE cap with no `backtest_max_dd` load. **Fix**: implement the 30-day PSR sustainment (e.g. min PSR over the last 30 daily snapshots >= 0.95) and load backtest_max_dd for the +5pp relative comparison (keep the 20% absolute cap as a floor). Constants `PSR_SUSTAINED_DAYS=30`, `PSR_THRESHOLD=0.95`, `MAX_DD_ABS_TOLERANCE=20.0` byte-untouched; DSR>=0.95/PBO<=0.5 untouched.

## Internal code inventory (re-verified 2026-07-11, read-only)
| File:line | Verbatim anchor | Status vs 69.0 target |
|-----------|-----------------|-----------------------|
| `analytics.py:322-325` | `se_sr = math.sqrt((1 - skewness*observed_sr + (kurtosis-1)/4 * observed_sr**2) / T)` | CONFIRMED. Formula SHAPE correct (raw kurtosis, `(k-1)/4`, `/T`); units wrong at caller |
| `analytics.py:654-661` | `compute_deflated_sharpe(observed_sr=result.aggregate_sharpe, ..., variance_of_srs=sr_variance, ..., T=T)` where `T=len(daily_returns)` (:650) and `sr_variance=np.var(window_sharpes)` (:642) | CONFIRMED unit mix: `aggregate_sharpe` ANNUALIZED, `T` DAILY, `sr_variance` = var of ANNUALIZED window Sharpes -> must de-annualize BOTH SR and V |
| `analytics.py:317-320` | `e_max_sr = sqrt(variance_of_srs) * ((1-0.5772)*ppf(1-1/N) + 0.5772*ppf(1-1/(N*e)))` | Matches canonical Bailey-LdP form; inherits V's annualized units -> de-annualize V=V/ppy |
| `walk_forward.py:61` | `test_start = train_end + timedelta(days=self.embargo_days + 1)`; default `embargo_days=5` | CONFIRMED: 5-day fixed gap, no purge; vs true label horizon 1.5*holding_days |
| `backtest_engine.py:488` | `p = cache.cached_prices(ticker, test_start_str, test_start_str)` (trade entry) | CONFIRMED exact-date single-day lookup; empty on weekend/holiday test_start |
| `backtest_engine.py:514` | `p = cache.cached_prices(ticker, test_end_str, test_end_str)` (liquidation) | CONFIRMED exact-date; empty on weekend/holiday test_end (:505 daily MTM is bdate_range-safe) |
| `backtest_engine.py:596-597` | `exit_dates.append(pd.Timestamp(sample_date) + timedelta(days=self.holding_days))` | CONFIRMED uses `holding_days` NOT `1.5*holding_days`; understates true label horizon; NO purge filter in `_build_training_data` |
| `backtest_engine.py:628-637` | TRAIN: `diffed = HistoricalDataProvider.fractional_diff(series, d=self.frac_diff_d)` over `_NON_STATIONARY` cols; then `X.fillna(X.median())` (:637) then `.fillna(0)` (:638) | CONFIRMED. Fracdiff weights + medians computed at train, NOT persisted |
| `backtest_engine.py:793-801` | PREDICT: `row = {f: fv.get(f, 0) ...}`; `X_test = pd.DataFrame([row])[feature_names].fillna(0)`; NO fracdiff | CONFIRMED train/predict transform skew: no fracdiff applied, fillna(0) not train-median |
| `paper_go_live_gate.py:111-113` | `psr_ge_95_sustained_30d: bool(psr is not None and psr >= PSR_THRESHOLD and metrics.get("n_obs",0) >= PSR_SUSTAINED_DAYS)` | UNDER-SPEC: checks LATEST psr>=0.95 AND >=30 obs. NOT "sustained over 30 days" (each of last 30d / rolling-min) |
| `paper_go_live_gate.py:117` | `max_dd_within_tolerance: bool(realized_max_dd <= MAX_DD_ABS_TOLERANCE)` (=20.0 abs) | UNDER-SPEC: only 20% absolute cap; the docstring's "backtest max DD + 5pp buffer" is NEVER loaded/compared |
| `paper_go_live_gate.py:7-15` (docstring) | `2. psr_ge_95_sustained_30d -- PSR >= 0.95 over the most recent 30 days` / `5. max_dd_within_tolerance -- realized max drawdown is not worse than the backtest max drawdown + 5pp buffer; default tolerance = 20% absolute cap` | THE DOCUMENTED SPEC. Carried from `handoff/current/phase-4.5-contract.md`; both booleans diverge from it |

### Go-live documented-spec location (task deliverable d)
The spec the two under-spec booleans should match is IN-MODULE at
`paper_go_live_gate.py:7-15` (module docstring, "Boolean definitions
(carried from handoff/current/phase-4.5-contract.md)") + cross-doc at
`docs/GO_LIVE_CHECKLIST.md`. Constants: `PSR_SUSTAINED_DAYS=30`,
`PSR_THRESHOLD=0.95`, `MAX_DD_ABS_TOLERANCE=20.0` (:35-40). There is NO
`backtest_max_dd` load and NO `+5pp` buffer constant -> the relative-tolerance
half of boolean #5 is entirely unimplemented; boolean #2 has no rolling/min
sustained test.

## Application to pyfinagent

69.2 GENERATE implements (offline backtest/analytics; zero live surface; DSR>=0.95/PBO<=0.5 byte-untouched):
- **DSR** (`analytics.py:292-335` + caller `:654-661`): de-annualize `observed_sr` and `variance_of_srs`; fixture pins DSR(SR_ann=2.5,T=1250,N=100,skew=-3,kurt=10,ppy=250)=0.9004 and asserts pre-fix path ~0.9999999.
- **Purge+embargo** (`backtest_engine.py:566-598`, `walk_forward.py:61`): purge samples where `[sample_date, sample_date+1.5*holding_days]` overlaps `[test_start,test_end]`; stamp exit_dates with 1.5*holding_days; fixture asserts zero overlap.
- **Boundary snap** (`backtest_engine.py:488` entry, `:514` liquidation): snap to prior session (pandas `DatetimeIndex.asof`) or range-widen; fixture asserts a Sat/Sun-bounded window fills trades and liquidates at a real close.
- **Fracdiff-at-predict** (`:628-637` train, `:793-801` predict): persist train medians + apply at predict (sklearn discipline); make the fracdiff transform consistent train/predict; fixture asserts identical transform+fill on a known vector.
- **Go-live booleans** (`paper_go_live_gate.py:111,117`): 30-day PSR sustainment (rolling-min) + backtest_max_dd+5pp relative comparison to the documented spec; fixture-tested.

Tests: `backend/tests/test_gate_correctness_69.py`. Incumbent re-validation under corrected gates is DEFERRED behind the historical_macro un-freeze token (code+fixtures only this step).

## Provenance note

External research (5 sources read in full: 3 by the researcher subagent before it STALLED on the end-of-session flush at ~311s -- the 4th harness-subagent stall this session -- and 2 by Main after the kill: sklearn common-pitfalls + skfolio CPCV) plus the complete internal inventory (re-verified by the subagent) and the go-live documented-spec location. Main finalized the synthesis/envelope from the read sources -- the documented "Main updates the stalled handoff file" pattern; every claim traces to a source row or a file:line. The DSR/purge/fracdiff/boundary theory is additionally grounded in research_brief_69.0.md (8 sources read in full).

## Research Gate Checklist

- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5: pandas-market-calendars, mlfin.py, microalphas, scikit-learn, skfolio) + 8 more in the 69.0 brief this builds on
- [x] 10+ unique URLs total (incl. snippet-only + 69.0 carryover)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (11 sites)
- [x] Each of the 5 topics has >=1 authoritative source (go-live spec located internally)
- [x] DSR reference reconfirmed (0.9004 / 0.9999999)

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 4,
  "urls_collected": 13,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "gate_passed": true
}
```
