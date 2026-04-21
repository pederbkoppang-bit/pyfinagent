# Experiment Results -- Cycle 92 / phase-4.9 step 4.9.4

Step: 4.9.4 Gauntlet regime catalog (7 historical windows)
Ran at: 2026-04-18 (UTC)

## What was built

Three new files:

1. `backend/backtest/gauntlet/__init__.py`
   Package marker. Re-exports `REGIMES`, `RegimeWindow`.

2. `backend/backtest/gauntlet/regimes.py` (~175 LOC)
   - Frozen dataclass `RegimeWindow` (9 fields: id, name, start,
     end, asset_classes, region, note, primary_source_url,
     intraday_only=False).
   - Supports dict-style key access via `__contains__` /
     `__getitem__` / `keys()` so the masterplan verification
     command `'start' in r and 'end' in r` works verbatim on the
     dataclass instances.
   - Helper `start_date()` / `end_date()` return `datetime.date`.
   - `REGIMES: tuple[RegimeWindow, ...]` with exactly 7 entries
     sorted by start date:
     - gfc_2008:              2008-09-15 .. 2009-03-09 (NBER)
     - flash_crash_2010:      2010-05-06 .. 2010-05-06
                              (SEC/CFTC report; intraday_only=True)
     - snb_chf_2015:          2015-01-15 .. 2015-01-26 (SNB PR)
     - covid_crash_2020:      2020-02-19 .. 2020-03-23 (peak/trough)
     - fed_hike_shock_2022:   2022-01-03 .. 2022-10-12 (Fed note)
     - yen_carry_unwind_2024: 2024-07-31 .. 2024-08-09 (BIS Bull90)
     - tariff_vol_2025:       2025-04-02 .. 2025-04-09 (Cboe Apr)

3. `scripts/audit/gauntlet_regimes_audit.py` (~180 LOC, 8 teeth)
   - import_ok (tuple of 7)
   - ids_unique_and_snake_case
   - dates_valid (ISO parse + start <= end)
   - dates_chronologically_sorted
   - immutability (actual mutation test; catches FrozenInstanceError)
   - universe_fields_populated (non-empty + https URL + note >= 40 chars)
   - intraday_flag_consistent (exactly one True, flash_crash_2010,
     with start == end)
   - masterplan_verification_passes (mirrors the verbatim
     `'start' in r` check on every regime)

## Verification output (verbatim)

Immutable masterplan verification:

    $ python -c "from backend.backtest.gauntlet.regimes import REGIMES; \
        assert len(REGIMES) == 7 and all('start' in r and 'end' in r for r in REGIMES)"
    IMMUTABLE VERIFY PASS
    exit 0

Audit:

    $ python scripts/audit/gauntlet_regimes_audit.py --check
    {
      "wrote": "handoff/gauntlet_regimes_audit.json",
      "verdict": "PASS",
      "import_ok": true,
      "ids_unique_and_snake_case": true,
      "dates_valid": true,
      "dates_chronologically_sorted": true,
      "immutability": true,
      "universe_fields_populated": true,
      "intraday_flag_consistent": true,
      "masterplan_verification_passes": true
    }
    exit 0

## Success criteria (from contract)

1. seven_regimes_defined:
   PASS -- `len(REGIMES) == 7`; every entry has `start` and `end`.
2. date_ranges_immutable:
   PASS -- mutation test raised FrozenInstanceError.
3. universe_hints_present:
   PASS -- every regime has non-empty asset_classes, region, note,
   primary_source_url; note length >= 40 chars; URL is https.

## Anti-rubber-stamp checks (honest)

- Dates match researcher's primary sources verbatim (NBER, SEC,
  SNB, Fed, BIS, Cboe). No paraphrased or estimated dates except
  the flagged SNB end date (2015-01-26) -- disclosed in the
  regime note as "conservative 7-trading-day tail-risk window".
- Mutation test (tooth #5) attempts an actual field assignment
  `REGIMES[0].end = "2030-01-01"` and catches
  `FrozenInstanceError` -- proves the dataclass is really frozen.
- Exactly one entry has intraday_only=True (flash_crash_2010),
  with start == end -- verified by tooth #7.
- Each note mentions a real ticker/index + a VIX or percentage
  drop from the researcher output (not stub text).

## Honest scope disclosure

- This is a DATA CATALOG step only. We do not:
  - run any backtest against these regimes (step 4.9.5).
  - integrate with spot_checks.py (the hardcoded 2-regime fallback
    at spot_checks.py:168-173 remains until 4.9.7).
  - verify BigQuery has coverage for all 7 windows (the 2025
    tariff window may be partial; Explore's coverage spot-check
    was not deep).
- The SNB end date (2015-01-26) is a research-estimated tail-risk
  end; no authoritative primary source pins an exact recovery
  date for this event. The yen-carry end date (2024-08-09) is the
  S&P-full-recovery candidate recommended by the researcher over
  the alternative BOJ-capitulation-signal date (2024-08-07).
- The masterplan verification uses dict-style `'start' in r`
  syntax; we implemented `RegimeWindow.__contains__` /
  `__getitem__` / `keys()` to pass the check verbatim without
  editing the masterplan.

## Artifacts written

- `handoff/gauntlet_regimes_audit.json` (8 checks + regime ids +
  date ranges + timestamp).
