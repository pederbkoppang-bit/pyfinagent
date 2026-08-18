# live_check -- step 86.116

**Required shape** (immutable): *"the re-measured duplication census and its
query, the before/after momentum and volatility numbers from a real driven
computation on a real affected ticker, the positive-controlled proof that
nothing de-duplicates today, and the parity oracle."*

Re-runnable:

```
python scripts/qa/verify_86_116.py             # criteria 1, 2, 3, 5, 6
python scripts/qa/verify_86_116.py --offline   # the non-BQ subset
python scripts/qa/mutation_86_116.py           # criterion 7
python -m pytest backend/tests/test_phase_86_116_price_dedup.py -q
```

All output below is verbatim from one `verify_86_116.py` run and one
`mutation_86_116.py` run. Nothing is spliced across runs.

---

## 1-3, 5, 6 -- the evidence script, verbatim

**Re-captured in cycle 3.** Both blocks below come from single fresh runs at the
pinned `--base-rev` default; nothing is spliced across runs.

```
==============================================================================
phase-86.116 -- DUPLICATE PRICE ROWS: EVIDENCE
==============================================================================

-- criterion 3: did anything de-duplicate BEFORE this step? --
pre-fix revision under test: 539f16eb~1
  files containing `drop_duplicates` under backend/ : 0
  files containing `index\.duplicated` under backend/ : 0
  files containing `is_unique` under backend/ : 0
  POSITIVE CONTROL, `set_index` (must exist): 2 files
  A grep returning nothing proves nothing until the same probe
  is shown able to return something. It is, so the zeros count.

-- criterion 5: parity oracle --
12 frame shapes: 7 unique-index (fix must be INERT, same object) / 5 duplicated (fix must be EXACT)
  Both branches are exercised, so the fix is provably neither
  a no-op nor an always-on rewrite.

-- criterion 1: the census, RE-MEASURED by this step --
table: sunny-might-477607-p8.financial_reports.historical_prices
  total rows          : 1,859,482
  distinct keys       : 1,152,607   (ticker, date)
  duplicated keys     : 706,875 = 61.33% OF KEYS
  excess rows         : 706,875 = 38.01% OF ROWS
  max multiplicity    : 2
  tickers affected    : 336 of 513
  dup keys whose close differs: 394,719

  NORMALISATION RULE, stated because the two shares differ and are
  NOT interchangeable: 'share of KEYS' divides by distinct
  (ticker,date) pairs; 'share of ROWS' divides by total table rows.
  Quoting one as the other misstates the defect by ~23 points.

  by year:
    2017  keys  81,887  dup  74,095   90.5%
    2018  keys 120,098  dup  74,836   62.3%
    2019  keys 122,097  dup  76,989   63.1%
    2020  keys 123,707  dup  78,603   63.5%
    2021  keys 124,684  dup  79,638   63.9%
    2022  keys 124,747  dup  80,069   64.2%
    2023  keys 124,727  dup  80,227   64.3%
    2024  keys 126,387  dup  81,518   64.5%
    2025  keys 125,787  dup  80,832   64.3%
    2026  keys  78,486  dup      68    0.1%
  The newest bucket (2026) is at 0.1%,
  so the write side is no longer producing duplicates and a repair
  is terminal rather than a treadmill. It is also why nobody noticed.

-- criterion 2: the harm, DRIVEN on AKAM 2025-01-01..2025-12-31 --
   (loaded through the REAL preload_prices; factors from the REAL
    screener functions -- a reimplementation would measure itself)
                           as-loaded (pre-fix)     de-duplicated
  n                                        390               250
  mom_1m                                  0.83             -0.52
  mom_3m                                 -1.60             15.04
  rsi_14                                  23.7              54.5
  vol_ann                               0.3343            0.4182
  span_21_sessions                          12                22

  * SIGN FLIP on mom_1m, mom_3m -- the pre-fix value does not
    merely differ in magnitude, it points the other way.
  * a '21-period' lookback covered 12 real sessions, not 22.
  * RSI moved 23.7 -> 54.5. rank_candidates applies score*=0.8 below 20 and score*=0.7
    above 80, so an RSI error is not cosmetic.

-- criterion 6: how this reaches the DSR/PBO gates --
  NOT a Sharpe-formula bug. The engine's NAV is a per-day dict, so
  duplication does not double-count NAV points. (Research gate.)

  NOT the triple-barrier width either -- I credited that in cycle 1
  and it was wrong. `vol_barrier_multiplier` is a DEAD KEY: written
  into engine._strategy_params, read by nothing, and named in
  rotation_runner._DEAD_KEYS as 'NO engine reader (reverted in
  9fbd9cd6)'. _compute_triple_barrier_label uses fixed tp_pct/sl_pct
  with NO volatility term. (Cycle-1 Q/A.)

  THE WIRED PATH is inverse-volatility POSITION SIZING:
    historical_data  features['annualized_volatility']
      -> backtest_engine  volatility = fv.get('annualized_volatility')
      -> signal dict
      -> backtest_trader  size_position(probability, volatility, nav)
      -> backtest_trader  vol_scale = min(target_vol / stock_vol, 3.0)

  measured volatility ratio (pre/post)   : 0.7995
  => position-size inflation (1/ratio)   : 1.2508x
  cap at backtest_trader.py              : 3.0x
  closed form under FULL duplication     : 0.7071 (1/sqrt(2)), giving at
                                           most 1.4142x inflation

  DIRECTION MATTERS AND IS COUNTER-INTUITIVE: stock_vol is in the
  DENOMINATOR, so an UNDERSTATED volatility makes positions LARGER,
  not smaller. The backtest was taking bigger risk than its own
  vol-targeting believed.

  NO THRESHOLD IS ADJUSTED by this step. min_dsr=0.95 / max_pbo=0.20
  are untouched; if a re-run moves them, that is a finding to report.

OK: all 43 invariants hold
```

## 7 -- mutation matrix, verbatim

The matrix now covers **two targets**. Criterion 7 says mutation-test *every*
new guard, and cycle 2 added two tripwires in `verify_86_116.py` that had no
cells -- which is precisely why one of them shipped vacuous.

```
control -> rc=0  collected=13  GREEN

[KILLED] M1 preload_prices stops de-duplicating -> the bulk path hands out a doubled index
           `test_preload_prices_returns_a_unique_index` failed (rc=1, collected=13)
[KILLED] M2 cached_prices stops de-duplicating -> the per-ticker fallback path leaks
           `test_cached_prices_returns_a_unique_index` failed (rc=1, collected=13)
[KILLED] M3 index-keyed dedup swapped for VALUE-keyed drop_duplicates()
           `test_value_keyed_dedup_is_insufficient` failed (rc=1, collected=13)
[KILLED] M3b value-keyed swap must be caught by the READ PATHS, not only the helper
           `test_preload_prices_returns_a_unique_index` failed (rc=1, collected=13)
[KILLED] M3c ... and by the OTHER read path too
           `test_cached_prices_returns_a_unique_index` failed (rc=1, collected=13)
[KILLED] M4 _dedupe_index made a pass-through -> the fix is present but inert
           `test_dedupe_removes_duplicate_index_entries` failed (rc=1, collected=13)
[KILLED] M5 inertness guard inverted -> the fix fires only when there is nothing to do
           `test_dedupe_removes_duplicate_index_entries` failed (rc=1, collected=13)
[KILLED] M6 empty/None guard dropped -> the loader crashes on an empty result set
           `test_dedupe_handles_empty_and_none` failed (rc=1, collected=13)

tripwire control (verify_86_116.py --offline) -> rc=0 GREEN
[KILLED] T1 the dead-key tripwire is disarmed -> a re-wired barrier key goes unnoticed
           invariant `tripwire_predicates_reject_known_bad_inputs` fired (rc=1)
[KILLED] T2 the volatility-term tripwire is disarmed -> a vol-scaled barrier goes unnoticed
           invariant `tripwire_predicates_reject_known_bad_inputs` fired (rc=1)
[KILLED] T4 the saturation guard is made ALWAYS TRUE -- the precise cycle-3 vacuity, where the script reported 1.2500x while the TRUE inflation was 1.0000x
           invariant `gate_guard_rejects_saturating_inputs` fired (rc=1)
[KILLED] T5 the saturation guard is made ALWAYS FALSE -> it fires on everything and masks the real measurement
           invariant `gate_guard_accepts_unsaturated_inputs` fired (rc=1)
[KILLED] T6 the known-bad fixture is replaced by a healthy one -> the paired negative goes unfalsifiable
           invariant `gate_guard_rejects_saturating_inputs` fired (rc=1)
[KILLED] T3 the tripwire FIXTURE is emptied -> both predicates go unfalsifiable
           invariant `tripwire_predicates_reject_known_bad_inputs` fired (rc=1)

[EQUIVALENT-BY-DESIGN] E1 keep='first' -> keep='last'
           EQUIVALENT BY DESIGN: which of two same-date rows survives is immaterial
            -- measured across the 394,719 duplicated keys whose close differs, the
            gap is 0.0% at BOTH p50 and p99 with a 0.93% maximum. The code comment 
           says the choice is immaterial, so a cell asserting otherwise would contr
           adict the shipped rationale. Declared rather than omitted, and scored as
            neither a kill nor a survivor.

------------------------------------------------------------------------------
KILLED 14 / 14   SURVIVED 0   UNSCORABLE 0   EQUIVALENT-BY-DESIGN 1 (not scored)
restore verified: cache.py 9f5f1d6798833281... verify_86_116.py 9fcf8806b56755ef...
```

## The query criterion 1 requires beside the counts

```sql
WITH k AS (
  SELECT ticker, date, COUNT(*) n, COUNT(DISTINCT close) nc
  FROM `sunny-might-477607-p8.financial_reports.historical_prices`
  GROUP BY 1, 2
)
SELECT (SELECT COUNT(*) FROM `...historical_prices`) total_rows,
       COUNT(*) keys, COUNTIF(n>1) dup_keys, SUM(n-1) excess_rows,
       MAX(n) max_mult,
       COUNT(DISTINCT IF(n>1, ticker, NULL)) tk_affected,
       COUNTIF(n>1 AND nc>1) dup_keys_close_differs
FROM k
```

`date` is a **STRING** column on this table, so every predicate is a string
comparison; a timestamp predicate errors.

## Regression: the change introduced ZERO failures, and that was MEASURED

A `-k` selection is not a regression suite -- step 86.108 shipped a red guard to
main through exactly that gap. So the **full** suite was run:

```
20 failed, 3633 passed, 12 skipped, 5 xfailed, 1 xpassed in 504.54s (0:08:24)
```

Attribution was then measured rather than assumed. The same 20 node ids were
re-run against the **pre-change** `backend/backtest/cache.py` (restored via
`git show HEAD:`, then restored back with a **verified matching sha256**):

- **18 of 20 already failed before this change** -- filed as step **86.118**.
- **1 was mine**: `test_phase_82_12_string_column_guards::
  test_classified_line_numbers_still_point_at_a_row_read` pins line numbers in
  `cache.py` within +/-6, and inserting `_dedupe_index` shifted them. **Fixed by
  RE-DERIVING the pins from source** (`so._row_key_reads(ast.parse(...))` ->
  `report_date` 658->700, `date` 718->760), never by arithmetic -- that table's
  own comment records what arithmetic cost a previous phase.
- **1 was an ordering artifact**: `test_phase_86_6_subprocess_channel::
  test_the_optin_IS_honoured_so_a_real_window_remains_possible` **passes when
  run alone** and fails only in the full run.

Re-running the 20 after the pin fix: **0 failures attributable to this change.**

**An incident worth recording rather than hiding.** The first attempt at this
comparison ran past the 10-minute command ceiling and was SIGTERMed **while the
pre-fix file was swapped in**, leaving `cache.py` in its pre-fix state on disk.
It was caught by checking `grep -c _dedupe_index` immediately afterwards and
restored from a backup taken before the swap, with sha256 confirmed identical.
This is the same class of hazard the mutation matrix installs signal handlers
for; an ad-hoc shell command had no such protection.

**A second self-inflicted error, also recorded.** The first failure-set diff
reported all 20 as newly-introduced. That was wrong: I stripped the `FAILED `
prefix from one file and not the other, so `comm` compared two formats that
could never match. The corrected diff is the one above.
